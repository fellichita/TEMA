"""Scheduling regressions with real coordinator/result validation, offline I/O.

These tests exercise processing ownership and conservation, not model quality.
Only discovery and naming decisions are controlled. No E5 or provider calls.
"""
import json

import pytest

from app.pilot.completion import evaluation_progress
from app.pilot.contracts import QUEUE_LIMIT, AnalysisResult, content_hash
from app.pilot.evidence import label_candidates
from app.pilot.export import read_result_package
from app.runtime.credentials import CredentialStore, LEGACY_ENVIRONMENT
from app.runtime.jobs import TaskCancelled, TaskFailure
from tests.test_pilot_evidence import Context, candidate, query_plan
from tests.test_pilot_service_signals import runtime_for, source_documents, wait_success


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch):
    for name in LEGACY_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    def forbidden(_self):
        pytest.fail("A scheduling regression must not access Keychain")
    monkeypatch.setattr(CredentialStore, "_load_backend", forbidden)


def _runtime(tmp_path, monkeypatch, *, reject_first):
    runtime, _calls, _providers = runtime_for(tmp_path, monkeypatch, source_documents())
    attempted = []
    discovered_ids = []

    def discovered(_context, snapshot, _plan):
        pool = [candidate(snapshot, candidate_id=f"candidate-{index:03d}").model_dump(mode="json")
                for index in range(70)]
        discovered_ids.extend(item["candidate_id"] for item in pool)
        return {"candidates": pool[:30], "review_queue": pool[30:]}

    def naming(candidates, *args, **kwargs):
        first = not attempted
        attempted.append(tuple(item.candidate_id for item in candidates))
        kept = (candidates[reject_first:] if first else candidates)
        # The production local naming path still validates/freeze-keeps every
        # surviving candidate; rejected membership is the controlled boundary.
        return label_candidates(kept, *args, **kwargs)

    monkeypatch.setattr(runtime, "_discover", discovered)
    monkeypatch.setattr("app.pilot.evidence.label_candidates", naming)
    return runtime, attempted, discovered_ids


def test_queue_continues_after_an_entire_first_batch_is_rejected(tmp_path, monkeypatch):
    runtime, attempted, discovered = _runtime(tmp_path, monkeypatch, reject_first=30)
    try:
        job = runtime.start(query_plan().original_query, query_plan().english_query)
        result, _ = wait_success(runtime, job)
        flat = [identifier for batch in attempted for identifier in batch]
        assert len(flat) == 60, "Rejection must not terminate processing before the attempt cap"
        assert len(flat) == len(set(flat))
        assert {card.candidate.candidate_id for card in result.cards}.union(
            item.candidate_id for item in result.candidate_queue) == set(discovered)
        progress = evaluation_progress(result.model_dump(mode="json"))
        assert progress.queued_count == 40 and progress.rejected_count == 30
        assert "проверка всей очереди не завершена" in progress.details
        states = result.candidate_review_states
        assert len(states) == 60
        assert sum(item.outcome == "definition_rejected" for item in states) == 30
        checkpoint = runtime.coordinator.checkpoint_value(job, states[0].stage)
        assert content_hash(checkpoint) == states[0].stage_hash
        path = tmp_path / "queue.trendresult"
        runtime.export_result(job, str(path))
        with read_result_package(path) as package:
            assert package.result == result
        # Re-examining one saved rejected definition creates a new result and
        # replaces only that candidate's last attempt, including after import.
        imported = runtime.import_result(str(path))["id"]
        refinement = runtime.refine_candidate(imported, states[0].candidate_id)
        revised, _ = wait_success(runtime, refinement)
        assert len(revised.cards) == 31 and len(revised.candidate_queue) == 39
        assert len(revised.candidate_review_states) == 60
        updated = next(item for item in revised.candidate_review_states if item.candidate_id == states[0].candidate_id)
        assert updated.source_run_id == refinement and updated.outcome == "passport_created"
        assert AnalysisResult.model_validate(runtime.result(job)["result"]) == result
        assert AnalysisResult.model_validate(runtime.result(imported)["result"]) == result
    finally:
        runtime.close()


def test_history_slots_are_frozen_for_later_specific_candidates_within_total_cap(tmp_path):
    from app.pilot.archive import DocumentArchive
    from app.pilot.service import _candidate_attempt_plan, _history_subbudgets
    from tests.test_pilot_evidence import document, snapshot

    archive = DocumentArchive(tmp_path / "revisions")
    data = snapshot((document(1),), archive)
    candidates = [candidate(data, candidate_id=f"candidate-{index:03d}").model_dump(mode="json")
                  for index in range(70)]
    discovered = {"candidates": candidates[:30], "review_queue": candidates[30:]}
    context = Context()
    planned, allocations, digest = _candidate_attempt_plan(discovered, query_plan(), data, context,
                                                           settings_hash="a" * 64)
    assert len(planned) == len(allocations) == 60
    assert sum(allocations.values()) == query_plan().limits.new_historical_documents
    assert all(0 < amount <= query_plan().limits.historical_documents_per_candidate for amount in allocations.values())
    # Antecedent and recent-history page requests share each candidate's slot;
    # existing discovery studies do not mint extra collection allowances.
    assert sum(sum(_history_subbudgets(amount)) for amount in allocations.values()) <= 20000
    assert allocations[planned[-1].candidate_id] > 0
    assert _candidate_attempt_plan(discovered, query_plan(), data, context, settings_hash="a" * 64) == (
        planned, allocations, digest)
    with pytest.raises(TaskFailure, match="другим кандидатам"):
        _candidate_attempt_plan(discovered, query_plan(), data, context, settings_hash="b" * 64)


def test_resume_uses_identical_batches_and_reuses_completed_passports(tmp_path, monkeypatch):
    import app.pilot.evidence as evidence

    runtime, attempted, _ = _runtime(tmp_path, monkeypatch, reject_first=0)
    local_naming = evidence.label_candidates
    local_passport = evidence.build_passport
    created = []

    def interrupted_naming(candidates, snapshot, archive, context, **kwargs):
        named = local_naming(candidates, snapshot, archive, context, **kwargs)
        if kwargs.get("stage_offset") == 30 and context.attempt == 1:
            context.cancel_event.set()
            raise TaskCancelled()
        return named

    def passport(candidate, snapshot, archive, context, **kwargs):
        stage = "passport_" + content_hash({"id": candidate.candidate_id})[:20]
        if context.load_checkpoint(stage) is None:
            created.append(candidate.candidate_id)
        return local_passport(candidate, snapshot, archive, context, **kwargs)

    monkeypatch.setattr(evidence, "label_candidates", interrupted_naming)
    monkeypatch.setattr(evidence, "build_passport", passport)
    try:
        job = runtime.start(query_plan().original_query, query_plan().english_query)
        runtime.coordinator.wait(timeout=10)
        assert runtime.get(job)["state"] == "cancelled"
        saved_plan = runtime.coordinator.checkpoint_value(job, "candidate_attempt_plan")
        saved_labels = runtime.coordinator.checkpoint_value(job, "labels_0")
        saved_budget = json.loads(runtime.get(job)["input_json"])
        assert len(created) == 30
        assert runtime.resume(job) == job
        result, _ = wait_success(runtime, job)
        assert len(result.cards) == len(created) == len(set(created)) == 60
        assert runtime.coordinator.checkpoint_value(job, "candidate_attempt_plan") == saved_plan
        assert runtime.coordinator.checkpoint_value(job, "labels_0") == saved_labels
        assert json.loads(runtime.get(job)["input_json"]) == saved_budget
        assert runtime.get(job)["attempt"] == 2
        assert len(result.candidate_review_states) == 60
        assert len(attempted) == 4  # Replay calls reuse frozen stages; there is no new naming/payment.
        assert runtime.result(job)["budget"]["calls"] == 0
    finally:
        runtime.close()


@pytest.fixture
def result_with_state(tmp_path):
    from app.pilot.archive import DocumentArchive
    from app.pilot.evidence import build_passport
    from tests.test_pilot_evidence import NOW, document, snapshot

    archive = DocumentArchive(tmp_path / "revisions")
    data = snapshot((document(1),), archive)
    item = label_candidates((candidate(data),), data, archive, Context(), query_plan=query_plan())[0]
    card = build_passport(item, data, archive, Context(), methodology_version="3.4.0")
    state = {"candidate_id": item.candidate_id, "candidate_hash": content_hash(item),
        "discovery_snapshot_id": data.snapshot_id, "source_run_id": "run-one", "attempt_ordinal": 0,
        "stage": "candidate_batch_outcome_0", "stage_hash": "a" * 64, "input_hash": "b" * 64,
        "outcome": "passport_created", "reason_code": None}
    return AnalysisResult(result_id="result-one", run_id="run-one", query_plan=query_plan(),
        methodology_version="3.4.0", created_at=NOW, quality="partial", snapshots=(data,), cards=(card,),
        candidate_review_states=(state,), top_trend_ids=(), top_limit=15, limitations=("Incomplete",))


@pytest.mark.parametrize("change", [
    {"candidate_id": "not-in-result"}, {"candidate_hash": "c" * 64},
    {"discovery_snapshot_id": "wrong-snapshot"}, {"attempt_ordinal": 60},
    {"stage": "candidate_batch_outcome_30"}, {"reason_code": "off_scope_or_mixed"},
    {"outcome": "definition_rejected", "reason_code": None},
])
def test_attempt_state_rejects_changed_candidate_or_invalid_outcome(result_with_state, change):
    payload = result_with_state.model_dump(mode="json")
    payload["candidate_review_states"][0].update(change)
    with pytest.raises(ValueError):
        AnalysisResult.model_validate(payload)


def test_legacy_serialization_does_not_gain_attempt_fields_or_change_hash(result_with_state):
    payload = result_with_state.model_dump(mode="json")
    payload.pop("candidate_review_states")
    payload["methodology_version"] = "3.2.0"
    payload["cards"][0]["methodology_version"] = "3.2.0"
    legacy = AnalysisResult.model_validate(payload)
    assert legacy.candidate_review_states is None
    assert "candidate_review_states" not in legacy.model_dump(mode="json")
    assert content_hash(legacy) == content_hash(payload)
    assert AnalysisResult.model_validate_json(legacy.model_dump_json()).model_dump_json() == legacy.model_dump_json()


def test_new_attempt_fields_are_not_silently_added_to_historical_methodology(result_with_state):
    payload = result_with_state.model_dump(mode="json")
    payload["methodology_version"] = "3.2.0"
    with pytest.raises(ValueError, match="Naming-attempt"):
        AnalysisResult.model_validate(payload)


def test_later_batch_receives_its_own_frozen_feasible_ai_plan(tmp_path):
    from dataclasses import replace
    from app.pilot.archive import DocumentArchive
    from app.pilot.service import _passport_ai_selection, _passport_ai_plan
    from app.runtime.budget import BudgetLimits, BudgetService
    from app.sqlite_runtime import sqlite3
    from tests.test_pilot_evidence import document, snapshot
    from tests.test_pilot_history import frozen
    from tests.test_pilot_llm import CONFIG

    archive = DocumentArchive(tmp_path / "revisions")
    data = snapshot((document(1),), archive)
    candidates = tuple(candidate(data, candidate_id=f"candidate-{index:03d}") for index in range(60))
    context = Context()
    connection = sqlite3.connect(tmp_path / "budget.sqlite3", isolation_level=None)
    try:
        budget = BudgetService(connection)
        cap = BudgetLimits(24, 200000, 30000, 1000000)
        budget.create_scope("run", cap, currency="USD")
        budget.create_scope("day", cap, currency="USD")
        first = _passport_ai_selection(candidates[:30], data, archive, context, budget, ("run", "day"), CONFIG,
            settings_hash="a" * 64, checkpoint_stage="passport_ai_plan_0")
        assert not first
        later = tuple(frozen(item) for item in candidates[30:])
        selected = _passport_ai_selection(later, data, archive, context, budget, ("run", "day"), CONFIG,
            settings_hash="a" * 64, checkpoint_stage="passport_ai_plan_30")
        assert selected and selected <= {item.candidate_id for item in later}
        allowances = _passport_ai_plan(later, data, archive, context, budget, ("run", "day"), CONFIG)
        for identifier in selected:
            budget.reserve(identifier, ("run", "day"), allowances[identifier])
        used = budget.snapshot("run").used
        assert used.calls <= cap.calls and used.input_tokens <= cap.input_tokens
        assert used.output_tokens <= cap.output_tokens and used.cost_micro <= cap.cost_micro
        budget.update_limits("day", replace(cap, calls=96))
        resumed = _passport_ai_selection(later, data, archive, context, budget, ("run", "day"), CONFIG,
            settings_hash="a" * 64, checkpoint_stage="passport_ai_plan_30")
        assert resumed == selected
        assert context.checkpoints["passport_ai_plan_0"]["selected_candidate_ids"] == []
    finally:
        connection.close()


def test_refinement_keeps_the_source_plans_zero_ai_caps(tmp_path, monkeypatch, result_with_state):
    from app.pilot.archive import DocumentArchive
    from app.pilot.evidence import admission_hash
    from app.pilot.export import export_result

    source = result_with_state
    limits = source.query_plan.limits.model_copy(update={"llm_calls": 0, "input_tokens": 0, "output_tokens": 0})
    plan = source.query_plan.model_copy(update={"limits": limits})
    item = source.cards[0].candidate.model_copy(update={"plan_hash": plan.plan_hash})
    item = item.model_copy(update={"admission_rule_hash": admission_hash(item)})
    source = AnalysisResult.model_validate(source.model_dump(mode="python") | {
        "query_plan": plan, "snapshots": (source.snapshots[0].model_copy(update={"plan_hash": plan.plan_hash}),),
        "cards": (source.cards[0].model_copy(update={"candidate": item}),),
        "candidate_review_states": (source.candidate_review_states[0].model_copy(
            update={"candidate_hash": content_hash(item)}),)})
    archive = DocumentArchive(tmp_path / "revisions")
    path = tmp_path / "zero-ai.trendresult"
    export_result(path, source, archive)
    records = tuple(archive.get(ref.revision_id) for ref in source.snapshots[0].documents)
    runtime, _, _ = runtime_for(tmp_path / "runtime", monkeypatch, records)
    monkeypatch.setattr("app.pilot.service.plan_query", lambda *_a, **_k: plan)
    try:
        source_id = runtime.import_result(str(path))["id"]
        job = runtime.refine_candidate(source_id, item.candidate_id)
        revised, _ = wait_success(runtime, job)
        assert revised.query_plan.limits == limits
        scoped = next(row for row in runtime.budget_status()["scopes"] if row["scope_id"] == "run/" + job)
        assert scoped["limits"]["calls"] == scoped["limits"]["input_tokens"] == scoped["limits"]["output_tokens"] == 0
        assert AnalysisResult.model_validate(runtime.result(source_id)["result"]) == source
    finally:
        runtime.close()


def test_rejected_ids_are_attempted_once_and_count_toward_the_total_cap(tmp_path, monkeypatch):
    runtime, attempted, discovered = _runtime(tmp_path, monkeypatch, reject_first=29)
    try:
        job = runtime.start(query_plan().original_query, query_plan().english_query)
        result, _ = wait_success(runtime, job)
        flat = [identifier for batch in attempted for identifier in batch]
        assert len(flat) <= 60, "The cap bounds attempted hypotheses, not only accepted cards"
        assert len(flat) == len(set(flat)), "Rejected IDs must not be paid/reviewed again in the same run"
        assert len(result.cards) <= 60 and len(result.candidate_queue) <= QUEUE_LIMIT
        assert {card.candidate.candidate_id for card in result.cards}.union(
            item.candidate_id for item in result.candidate_queue) == set(discovered)
    finally:
        runtime.close()
