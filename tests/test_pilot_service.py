"""Desktop service integration: durable runs, settings recovery and offline imports."""

from datetime import date
import json
from threading import Event
from types import SimpleNamespace

import pytest

from app.pilot.contracts import CorpusSnapshot, Coverage, content_hash
from app.pilot.export import export_result
from app.pilot.service import PilotService, _antecedent_shortlist
from app.pilot.settings import PilotSettings, apply_collection_profile, load_settings, save_settings
from app.runtime.credentials import CredentialStore, CredentialUnavailable
from app.runtime.jobs import TaskFailure
from app.runtime.model_resources import ModelLocation
from tests.test_pilot_export import make_result


def test_explicit_cuda_fails_before_starting_analysis_when_runtime_is_unavailable(monkeypatch):
    from app.runtime.inference import PROVIDER_VARIABLE, RequestedCudaUnavailable

    monkeypatch.setenv(PROVIDER_VARIABLE, "cuda")

    def unavailable():
        raise RequestedCudaUnavailable("CUDA недоступна")

    monkeypatch.setattr("app.runtime.inference.execution_providers", unavailable)
    with pytest.raises(TaskFailure, match="CUDA недоступна"):
        PilotService._run(object.__new__(PilotService), None, {})


def empty_coverage(plan):
    return (Coverage(source="openalex", purpose="discovery", query_hash=content_hash(plan.queries[0]),
                     state="complete", requested_years=plan.completed_years,
                     completed_years=plan.completed_years, pagination_exhausted=True,
                     comparable=False),)


@pytest.fixture
def service(tmp_path, monkeypatch):
    from app.pilot.funding_sources import FundingSnapshot

    credentials = CredentialStore()
    monkeypatch.setattr(credentials, "get", lambda _: None)
    monkeypatch.setattr("app.pilot.encoder.verify_artifacts", lambda _, **kwargs: None)
    monkeypatch.setattr("app.pilot.funding_sources.fetch_nih_grants",
                        lambda topic, from_date, to_date, **kwargs: FundingSnapshot(
                            topic, from_date, to_date, (), 0, 0, 0, False, "complete"))
    # ТОП технологий веб-анализа ходит в OpenAlex и arXiv; тесты сервиса — нет.
    monkeypatch.setattr("app.radar.pipeline.build_radar",
                        lambda pool, **kwargs: {"technologies": [], "excluded": [], "failed": []})
    runtime = PilotService(tmp_path / "data", credentials)
    yield runtime
    runtime.close()


def test_damaged_preferences_are_visible_and_repairable_without_automatic_overwrite(service):
    path = service.data_dir / "settings.json"
    path.write_text('{"run_cost_micro":"bad"}')
    status = service.status()
    assert status["settings_error"]
    assert path.read_text(encoding="utf-8") == '{"run_cost_micro":"bad"}'
    with pytest.raises(TaskFailure):
        service.start("materials")
    service.configure(status["settings"])
    assert service.status()["settings_error"] is None
    assert load_settings(service.data_dir) == PilotSettings()


def test_unavailable_keychain_does_not_hide_settings_or_saved_results(service, monkeypatch):
    def unavailable(_):
        raise CredentialUnavailable("Хранилище недоступно")
    monkeypatch.setattr(service.credentials, "get", unavailable)
    status = service.status()
    assert status["keys"]["deepseek_api_key"] is None
    assert "deepseek_api_key" in status["key_errors"]
    assert service.list_runs() == []


def test_model_status_reuses_verified_identity_but_manual_check_forces_hash(service, monkeypatch):
    calls = []
    monkeypatch.setattr("app.pilot.encoder.verify_artifacts", lambda *_args, **_kwargs: calls.append(True))
    assert service.status()["model_installed"] is True
    assert service.status()["model_installed"] is True
    assert len(calls) == 1
    assert service.status(verify_model=True)["model_installed"] is True
    assert len(calls) == 2
    service.model_dir.mkdir(parents=True, exist_ok=True)
    (service.model_dir / "model.onnx").write_bytes(b"changed model identity")
    assert service.status()["model_installed"] is True
    assert len(calls) == 3


def test_offline_weights_are_reported_and_installed_on_request_with_progress(service, monkeypatch):
    assert service.status()["local_llm_installed"] is False
    seen = {}

    def install(directory, *, cancel=None, progress=None):
        seen.update(directory=directory, cancel=cancel)
        progress(5, 10)
        monkeypatch.setattr("app.pilot.local_llm.artifacts_present", lambda *_args, **_kwargs: True)
        return {"installed": True}

    monkeypatch.setattr("scripts.install_local_llm.install", install)
    reported = []
    status = service.install_local_llm(lambda done, total: reported.append((done, total)))
    assert seen["directory"] == service.local_llm_dir
    assert seen["cancel"] is service.model_cancel
    assert reported == [(5, 10)]
    assert status["local_llm_installed"] is True


def test_bundled_native_load_failure_is_separate_from_model_corruption(service, monkeypatch):
    service.model_location = ModelLocation(service.model_dir, "bundled", "multilingual-e5-small", "a" * 40)
    def fail_native(*_args, **_kwargs):
        raise RuntimeError("native diagnostic detail")
    monkeypatch.setattr("app.pilot.encoder.MultilingualEncoder", fail_native)
    status = service.status()
    assert status["model_state"] == "unavailable"
    assert status["model_installed"] is False
    assert "native diagnostic detail" not in status["model_error"]
    assert service.list_runs() == []


def test_bundled_status_distinguishes_missing_and_corrupt_files(service, monkeypatch):
    from app.pilot.encoder import EncoderError

    service.model_location = ModelLocation(service.model_dir, "bundled", "multilingual-e5-small", "a" * 40)
    def fail_integrity(*_args, **_kwargs):
        raise EncoderError("internal developer-install hint")
    monkeypatch.setattr("app.pilot.encoder.verify_artifacts", fail_integrity)
    missing = service.status()
    assert missing["model_state"] == "missing"
    assert "отсутствует" in missing["model_error"]
    service.model_dir.mkdir(parents=True, exist_ok=True)
    (service.model_dir / "model.onnx").write_bytes(b"tampered")
    corrupt = service.status()
    assert corrupt["model_state"] == "corrupt"
    assert "повреждена" in corrupt["model_error"]
    assert "developer" not in corrupt["model_error"]


@pytest.mark.parametrize(("method", "arguments"), [
    ("result", ("import-" + "0" * 64,)),
    ("import_result", ("does-not-exist.trendresult",)),
    ("import_arxiv", ("does-not-exist.atom",)),
    ("install_model", ()),
])
def test_shutdown_signal_cannot_be_cleared_by_late_operations(service, method, arguments):
    from app.runtime.jobs import TaskCancelled

    service.view_cancel.set()
    service.model_cancel.set()
    with pytest.raises(TaskCancelled):
        getattr(service, method)(*arguments)
    assert service.view_cancel.is_set() and service.model_cancel.is_set()


def test_settings_preserve_exact_currency_and_reject_invalid_caps(tmp_path):
    values = PilotSettings.for_provider("yandex")
    save_settings(tmp_path, values)
    assert load_settings(tmp_path) == values
    # A tariff is only checked for a provider that charges one, so the invalid
    # combinations below are built on an explicit paid provider.
    paid = PilotSettings.for_provider("deepseek").model_dump()
    for changes in ({"run_cost_micro": -1}, {"day_cost_micro": 0}, {"discovery_documents": True},
                    {"api_key": "never persist here"}, {"currency": "RUB"}):
        with pytest.raises(ValueError):
            PilotSettings.model_validate(paid | changes)


def test_collection_profiles_preserve_model_identity_and_apply_explicit_network_limits():
    base = PilotSettings(run_cost_micro=123_000, day_cost_micro=456_000)
    expected = {
        "fast": (1000, 8, 24, True, False),
        "deep": (10000, 15, 60, True, True),
    }
    for profile, controls in expected.items():
        configured = apply_collection_profile(base, profile)
        assert (configured.discovery_documents, configured.candidate_limit, configured.candidate_attempts,
                configured.history_enabled, configured.patents_enabled) == controls
        assert (configured.provider, configured.model_name, configured.model_version,
                configured.pricing_version, configured.run_cost_micro, configured.day_cost_micro) == (
                    base.provider, base.model_name, base.model_version, base.pricing_version,
                    base.run_cost_micro, base.day_cost_micro)


def test_candidate_attempts_default_to_the_former_pool_and_cover_the_top():
    # Настройки, сохранённые до появления поля, получают прежний пул в 60.
    legacy = PilotSettings.model_validate({key: value for key, value in PilotSettings().model_dump().items()
                                           if key != "candidate_attempts"})
    assert legacy.candidate_attempts == 60
    with pytest.raises(ValueError, match="ТОП"):
        PilotSettings(candidate_limit=12, candidate_attempts=8)


def test_attempt_plan_reviews_only_the_mode_pool(monkeypatch):
    from app.pilot.service import _candidate_attempt_plan

    monkeypatch.setattr("app.pilot.service._freeze_processing_checkpoint", lambda *_: None)
    requested = []

    def pool(_discovered, limit):
        requested.append(limit)
        return tuple(SimpleNamespace(candidate_id=f"c{number:02d}", specificity="specific_technology",
                                     model_dump=lambda number=number, **_: {"id": number})
                     for number in range(limit))

    monkeypatch.setattr("app.pilot.service._candidate_pool", pool)
    plan = SimpleNamespace(plan_hash="plan", limits=SimpleNamespace(new_historical_documents=1500,
                                                                    historical_documents_per_candidate=150))
    snapshot = SimpleNamespace(snapshot_hash="snapshot")
    for attempts in (24, 40):
        schedule, allocations, _ = _candidate_attempt_plan({}, plan, snapshot, None,
                                                           settings_hash=str(attempts), attempts=attempts)
        assert len(schedule) == attempts and len(allocations) == attempts
    _candidate_attempt_plan({}, plan, snapshot, None, settings_hash="default")
    assert requested == [24, 40, 60]


def test_antecedent_shortlist_uses_cheap_rank_and_deduplicates_concepts():
    def card(identifier, label):
        candidate = SimpleNamespace(candidate_id=identifier, label=label, synonyms=(),
                                    specificity="specific_technology", admission_rule_version="test")
        return SimpleNamespace(candidate=candidate)

    def artifact(identifier, failures, upper, lower, recent):
        assessment = SimpleNamespace(candidate_id=identifier, gate_failures=failures,
                                     priority_upper_bound=upper, priority_lower_bound=lower,
                                     recent_studies=recent)
        return SimpleNamespace(assessment=assessment)

    cards = [card("weaker", "Lithium membrane"), card("best", "Photonic engine"),
             card("duplicate", "Photonic engine")]
    artifacts = [artifact("weaker", ("missing_evidence",), 90, 60, 20),
                 artifact("best", ("earlier_search_not_complete",), 70, 50, 8),
                 artifact("duplicate", ("earlier_search_not_complete",), 60, 40, 7)]
    assert _antecedent_shortlist(cards, artifacts, {"weaker", "best", "duplicate"}, limit=2) == (
        "best", "weaker")


def test_offline_import_survives_restart_without_keys_model_or_network(service, tmp_path, monkeypatch):
    result, archive, assessments = make_result(tmp_path / "source", historical=True)
    path = tmp_path / "saved.trendresult"
    export_result(path, result, archive, assessments)
    import socket
    monkeypatch.setattr(socket, "socket", lambda *_a, **_k: pytest.fail("offline import accessed network"))
    imported = service.import_result(str(path))
    identifier = imported["id"]
    assert imported["payload"]["view_id"] == identifier
    assert service.result(identifier)["result"] == result.model_dump(mode="json")
    assert service.documents(identifier)["total"] == len(result.snapshots[0].documents)
    assert service.list_runs()[0]["imported"] is True
    assert service.import_result(str(path))["id"] == identifier
    assert len(service.list_runs()) == 1
    reopened_path = tmp_path / "reexport.trendresult"
    service.export_result(identifier, str(reopened_path))
    assert reopened_path.is_file()


def test_corrupt_import_remains_visible_without_hiding_other_history(service, tmp_path):
    result, archive, assessments = make_result(tmp_path / "source")
    path = tmp_path / "saved.trendresult"
    export_result(path, result, archive, assessments)
    identifier = service.import_result(str(path))["id"]
    (service.data_dir / "imported-results" / (identifier.removeprefix("import-") + ".json")).write_text("{}")
    rows = service.list_runs()
    assert rows[0]["state"] == "failed"
    with pytest.raises(TaskFailure, match="повреждён"):
        service.result(identifier)


def test_manual_workflow_uses_coordinator_and_changed_daily_cap_without_reset(service, monkeypatch):
    seen = []
    def collect(plan, context, archive, credentials):
        seen.append(plan)
        return CorpusSnapshot(snapshot_id="empty-snapshot", plan_hash=plan.plan_hash, purpose="discovery",
            as_of=plan.as_of, created_at="2026-09-10T12:00:00Z", documents=(), coverage=empty_coverage(plan),
            normalizer_version="test", deduplication_version="test")
    monkeypatch.setattr("app.pilot.service.collect_snapshot", collect)
    monkeypatch.setattr(service, "_discover", lambda *_: dict(candidates=[], input_records=0, unique_studies=0,
        retained_studies=0, quality="insufficient_data", limitations=["No matching documents"]))
    first = service.start("Произвольная новая технология", "synthetic mechanism")
    service.coordinator.wait()
    assert service.get(first)["state"] == "succeeded"
    assert service.result(first)["result"]["quality"] == "insufficient_data"
    assert service.result(first)["budget"]["calls"] == 0
    new_settings = PilotSettings(day_cost_micro=3_000_000)
    service.configure(new_settings.model_dump(mode="json"))
    second = service.start("Произвольная новая технология", "synthetic mechanism")
    service.coordinator.wait()
    assert service.get(second)["state"] == "succeeded"
    # The run and its budget remain separate, while the identical source plan is reused.
    assert len(seen) == 1
    assert seen[0].as_of == date.today()


def test_fast_profile_uses_reduced_source_limits_and_marks_result_preliminary(service, monkeypatch):
    plans = []

    def collect(plan, *_):
        plans.append(plan)
        return CorpusSnapshot(snapshot_id="fast-empty", plan_hash=plan.plan_hash, purpose="discovery",
            as_of=plan.as_of, created_at="2026-09-20T12:00:00Z", documents=(), coverage=empty_coverage(plan),
            normalizer_version="test", deduplication_version="test")

    monkeypatch.setattr("app.pilot.service.collect_snapshot", collect)
    monkeypatch.setattr(service, "_discover", lambda *_: dict(candidates=[], input_records=0, unique_studies=0,
        retained_studies=0, quality="insufficient_data", limitations=[]))
    run_id = service.start("Произвольная технология", "synthetic mechanism", collection_profile="fast")
    service.coordinator.wait()
    assert service.get(run_id)["state"] == "succeeded"
    assert len(plans) == 1
    assert plans[0].limits.discovery_documents == 1000
    assert plans[0].limits.historical_documents_per_candidate == 150
    assert plans[0].limits.new_historical_documents == 1500
    result = service.result(run_id)["result"]
    assert result["top_limit"] == 8
    assert any("предварительный" in item for item in result["limitations"])


def test_fast_and_deep_modes_apply_their_own_sample_history_and_top(service, monkeypatch):
    """Глубокий режим больше быстрого по выборке, истории и ТОП, а не почти равен ему."""
    plans = []

    def collect(plan, *_):
        plans.append(plan)
        return CorpusSnapshot(snapshot_id=f"ladder-{len(plans)}", plan_hash=plan.plan_hash,
            purpose="discovery", as_of=plan.as_of, created_at="2026-09-24T12:00:00Z", documents=(),
            coverage=empty_coverage(plan), normalizer_version="test", deduplication_version="test")

    monkeypatch.setattr("app.pilot.service.collect_snapshot", collect)
    monkeypatch.setattr(service, "_discover", lambda *_: dict(candidates=[], input_records=0, unique_studies=0,
        retained_studies=0, quality="insufficient_data", limitations=[]))
    tops = []
    for profile in ("fast", "deep"):
        run_id = service.start(f"Технология {profile}", f"synthetic {profile}", collection_profile=profile)
        service.coordinator.wait()
        assert service.get(run_id)["state"] == "succeeded", service.get(run_id)["error"]
        tops.append(service.result(run_id)["result"]["top_limit"])
    ladder = [(plan.limits.discovery_documents, plan.limits.new_historical_documents) for plan in plans]
    assert ladder == [(1000, 1500), (10000, 20000)]
    assert tops == [8, 15]


def test_approved_sources_collect_alongside_scientific_discovery(service, monkeypatch):
    from app.pilot.approved_sources import unavailable_snapshot

    started = Event()

    def collect_approved(query, *, as_of, cancel, **_kwargs):
        started.set()
        return unavailable_snapshot(query, as_of, "test_source_unavailable")

    def collect_science(plan, *_args):
        assert started.wait(5), "Дополнительные источники не были запущены параллельно"
        return CorpusSnapshot(snapshot_id="parallel-empty", plan_hash=plan.plan_hash,
            purpose="discovery", as_of=plan.as_of, created_at="2026-09-20T12:00:00Z",
            documents=(), coverage=empty_coverage(plan), normalizer_version="test",
            deduplication_version="test")

    monkeypatch.setattr("app.pilot.approved_sources.collect_approved_sources", collect_approved)
    monkeypatch.setattr("app.pilot.service.collect_snapshot", collect_science)
    monkeypatch.setattr(service, "_discover", lambda *_: dict(candidates=[], input_records=0,
        unique_studies=0, retained_studies=0, quality="insufficient_data", limitations=[]))
    run_id = service.start("quantum sensors", "quantum sensors")
    service.coordinator.wait()
    assert service.get(run_id)["state"] == "succeeded", service.get(run_id)["error"]
    snapshot = service.result(run_id)["approved_sources"]
    assert snapshot["query"] == "quantum sensors"
    from app.pilot.approved_sources.contracts import SOURCE_IDS

    assert len(snapshot["coverage"]) == len(SOURCE_IDS)
    assert service.coordinator.checkpoint_value(run_id, "approved_sources") == snapshot


@pytest.mark.parametrize(("query", "expected_topic"), [
    ("quantum sensors", "quantum sensors"),
    ("AI-driven sensors", "AI driven sensors"),
])
def test_web_funding_collects_in_parallel_and_stays_out_of_publications(
        service, monkeypatch, query, expected_topic):
    from decimal import Decimal

    from app.pilot.approved_sources import unavailable_snapshot
    from app.pilot.funding_sources import FundingAward, FundingSnapshot
    from app.web_api import web_result

    funding_started = Event()

    def collect_funding(topic, from_date, to_date, **_kwargs):
        funding_started.set()
        assert topic == expected_topic
        award = FundingAward(123, "R01-123", "Quantum sensor", to_date,
                             Decimal("100000"), "RP", "grant_or_cooperative",
                             "https://reporter.nih.gov/project-details/123")
        return FundingSnapshot(topic, from_date, to_date, (award,), 1, 1, 0, False, "complete")

    def collect_science(plan, *_args):
        assert funding_started.wait(5), "Денежный источник не был запущен параллельно"
        return CorpusSnapshot(snapshot_id="funding-parallel", plan_hash=plan.plan_hash,
            purpose="discovery", as_of=plan.as_of, created_at="2026-09-25T00:00:00Z",
            documents=(), coverage=empty_coverage(plan), normalizer_version="test",
            deduplication_version="test")

    monkeypatch.setattr("app.pilot.funding_sources.fetch_nih_grants", collect_funding)
    monkeypatch.setattr("app.pilot.approved_sources.collect_approved_sources",
                        lambda query, *, as_of, **_kwargs: unavailable_snapshot(query, as_of))
    monkeypatch.setattr("app.pilot.service.collect_snapshot", collect_science)
    monkeypatch.setattr(service, "_discover", lambda *_: dict(candidates=[], input_records=0,
        unique_studies=0, retained_studies=0, quality="insufficient_data", limitations=[]))
    run_id = service.start(query, query, collection_profile="fast")
    service.coordinator.wait()
    assert service.get(run_id)["state"] == "succeeded", service.get(run_id)["error"]
    saved = service.result(run_id)
    assert saved["funding_sources"]["awards"][0]["award_amount_usd"] == "100000"
    assert service.coordinator.checkpoint_value(run_id, "funding_sources") == saved["funding_sources"]
    displayed = web_result(saved, archive=service.archive)
    assert displayed["publication_total"] == 0
    assert displayed["funding_evidence"]["awards"][0]["application_id"] == 123


def test_completed_run_persists_publication_model_scores_for_web_result(service, monkeypatch):
    from app.pilot.approved_sources import SourceSnapshot, unavailable_snapshot
    from app.web_api import web_result

    service.configure(PilotSettings.for_provider("deepseek").model_dump(mode="json"))

    def collect_approved(query, *, as_of, cancel, **_kwargs):
        snapshot = unavailable_snapshot(query, as_of).model_dump(mode="json")
        snapshot["observations"] = [{
            "source_id": "arxiv", "item_id": "paper-1", "kind": "preprint",
            "title": "Quantum sensor article", "url": "https://arxiv.org/abs/2609.12345",
            "published_at": as_of.isoformat(), "observed_at": "2026-09-25T00:00:00Z",
            "summary": "A quantum sensor study.", "rights": "local_only", "license_ref": None,
        }]
        next(item for item in snapshot["coverage"] if item["source_id"] == "arxiv").update(
            state="partial", requested_limit=1, scanned=1, accepted=1,
            limit_reached=True, reason_code="source_limit")
        return SourceSnapshot.model_validate(snapshot)

    def collect_science(plan, *_args):
        return CorpusSnapshot(snapshot_id="score-empty", plan_hash=plan.plan_hash,
            purpose="discovery", as_of=plan.as_of, created_at="2026-09-25T00:00:00Z",
            documents=(), coverage=empty_coverage(plan), normalizer_version="test",
            deduplication_version="test")

    scored_ids = []

    def score(top, **_kwargs):
        assert len(top) == 1 and top[0]["source_id"] == "arxiv"
        scored_ids.append(top[0]["publication_id"])
        return {top[0]["publication_id"]: {
            "score": 91, "reason": "The article matches the requested quantum sensors.",
            "evidence_quote": "Quantum sensor article", "basis": "title_and_summary"}}

    monkeypatch.setattr("app.pilot.approved_sources.collect_approved_sources", collect_approved)
    monkeypatch.setattr("app.pilot.service.collect_snapshot", collect_science)
    monkeypatch.setattr(service, "_discover", lambda *_: dict(candidates=[], input_records=0,
        unique_studies=0, retained_studies=0, quality="insufficient_data", limitations=[]))
    monkeypatch.setattr("app.pilot.encoder.MultilingualEncoder",
                        lambda *_args, **_kwargs: SimpleNamespace(fingerprint="test-e5"))
    monkeypatch.setattr("app.pilot.publication_confidence.score_publications", score)
    run_id = service.start("quantum sensors", "quantum sensors")
    service.coordinator.wait()
    assert service.get(run_id)["state"] == "succeeded", service.get(run_id)["error"]
    saved = service.result(run_id)
    assert set(saved["publication_confidences"]) == set(scored_ids)
    publication = web_result(saved, archive=service.archive)["top_publications"][0]
    assert publication["publication_id"] == scored_ids[0]
    assert publication["model_confidence"]["score"] == 91


def test_completed_approved_source_collection_is_reused_after_scientific_failure(service, monkeypatch):
    from app.pilot.approved_sources import unavailable_snapshot
    from app.pilot.funding_sources import FundingSnapshot

    collected, funding_collected = Event(), Event()
    calls, funding_calls = [], []

    def collect_approved(query, *, as_of, cancel, **_kwargs):
        calls.append(query)
        collected.set()
        return unavailable_snapshot(query, as_of, "test_source_unavailable")

    def collect_funding(topic, from_date, to_date, **_kwargs):
        funding_calls.append(topic)
        funding_collected.set()
        return FundingSnapshot(topic, from_date, to_date, (), 0, 0, 0, False, "complete")

    def collect_science(plan, *_args):
        assert collected.wait(5) and funding_collected.wait(5)
        return CorpusSnapshot(snapshot_id="resume-empty", plan_hash=plan.plan_hash,
            purpose="discovery", as_of=plan.as_of, created_at="2026-09-20T12:00:00Z",
            documents=(), coverage=empty_coverage(plan), normalizer_version="test",
            deduplication_version="test")

    attempts = 0

    def discover(*_args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TaskFailure("Проверяем возобновление после сбоя научной стадии.")
        return dict(candidates=[], input_records=0, unique_studies=0, retained_studies=0,
                    quality="insufficient_data", limitations=[])

    monkeypatch.setattr("app.pilot.approved_sources.collect_approved_sources", collect_approved)
    monkeypatch.setattr("app.pilot.funding_sources.fetch_nih_grants", collect_funding)
    monkeypatch.setattr("app.pilot.service.collect_snapshot", collect_science)
    monkeypatch.setattr(service, "_discover", discover)
    run_id = service.start("quantum sensors", "quantum sensors", collection_profile="fast")
    service.coordinator.wait()
    assert service.get(run_id)["state"] == "failed"
    assert service.coordinator.checkpoint_value(run_id, "approved_sources") is not None
    assert service.coordinator.checkpoint_value(run_id, "funding_sources") is not None
    service.resume(run_id)
    service.coordinator.wait()
    assert service.get(run_id)["state"] == "succeeded", service.get(run_id)["error"]
    assert calls == ["quantum sensors"]
    assert funding_calls == ["quantum sensors"]


def test_finished_external_sources_survive_discovery_failure(service, monkeypatch):
    from app.pilot.approved_sources import unavailable_snapshot
    from app.pilot.funding_sources import FundingSnapshot

    approved_done, funding_done = Event(), Event()
    calls = {"approved": 0, "funding": 0, "discovery": 0}

    def collect_approved(query, *, as_of, **_kwargs):
        calls["approved"] += 1
        approved_done.set()
        return unavailable_snapshot(query, as_of)

    def collect_funding(topic, from_date, to_date, **_kwargs):
        calls["funding"] += 1
        funding_done.set()
        return FundingSnapshot(topic, from_date, to_date, (), 0, 0, 0, False, "complete")

    def collect_science(plan, *_args):
        assert approved_done.wait(5) and funding_done.wait(5)
        calls["discovery"] += 1
        if calls["discovery"] == 1:
            raise TaskFailure("Сбой научной стадии после завершения внешних источников.")
        return CorpusSnapshot(snapshot_id="resume-discovery", plan_hash=plan.plan_hash,
            purpose="discovery", as_of=plan.as_of, created_at="2026-09-20T12:00:00Z",
            documents=(), coverage=empty_coverage(plan), normalizer_version="test",
            deduplication_version="test")

    monkeypatch.setattr("app.pilot.approved_sources.collect_approved_sources", collect_approved)
    monkeypatch.setattr("app.pilot.funding_sources.fetch_nih_grants", collect_funding)
    monkeypatch.setattr("app.pilot.service.collect_snapshot", collect_science)
    monkeypatch.setattr(service, "_discover", lambda *_: dict(candidates=[], input_records=0,
        unique_studies=0, retained_studies=0, quality="insufficient_data", limitations=[]))
    run_id = service.start("quantum sensors", "quantum sensors", collection_profile="fast")
    service.coordinator.wait()
    assert service.get(run_id)["state"] == "failed"
    assert service.coordinator.checkpoint_value(run_id, "approved_sources") is not None
    assert service.coordinator.checkpoint_value(run_id, "funding_sources") is not None
    service.resume(run_id)
    service.coordinator.wait()
    assert service.get(run_id)["state"] == "succeeded", service.get(run_id)["error"]
    assert calls == {"approved": 1, "funding": 1, "discovery": 2}


def test_shared_source_cache_rejects_expiry_and_payload_tampering(service):
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 9, 20, 12, tzinfo=UTC)
    key = "a" * 64
    service._shared_cache_write("discovery", key, {"value": 1}, now=now)
    assert service._shared_cache_read("discovery", key, now=now + timedelta(hours=5)) == {"value": 1}
    assert service._shared_cache_read("discovery", key, now=now + timedelta(hours=6)) is None

    path = service.data_dir / "cache" / "source-artifacts" / "discovery" / (key + ".json")
    damaged = json.loads(path.read_text(encoding="utf-8"))
    damaged["payload"]["value"] = 2
    path.write_text(json.dumps(damaged))
    assert service._shared_cache_read("discovery", key, now=now + timedelta(hours=1)) is None


def test_source_failure_is_retried_instead_of_cached(service, monkeypatch):
    calls = []

    def collect(plan, *_):
        calls.append(plan.plan_hash)
        coverage = Coverage(source="openalex", purpose="discovery", query_hash=content_hash(plan.queries[0]),
            state="unavailable", requested_years=plan.completed_years, pagination_exhausted=False,
            comparable=False, reasons=("source_unavailable",))
        return CorpusSnapshot(snapshot_id="unavailable", plan_hash=plan.plan_hash, purpose="discovery",
            as_of=plan.as_of, created_at="2026-09-20T12:00:00Z", documents=(), coverage=(coverage,),
            normalizer_version="test", deduplication_version="test")

    monkeypatch.setattr("app.pilot.service.collect_snapshot", collect)
    monkeypatch.setattr(service, "_discover", lambda *_: dict(candidates=[], input_records=0, unique_studies=0,
        retained_studies=0, quality="insufficient_data", limitations=[]))
    for _ in range(2):
        run_id = service.start("Произвольная технология", "synthetic mechanism")
        service.coordinator.wait()
        assert service.get(run_id)["state"] == "succeeded"
    assert len(calls) == 2


def test_corrupt_disposable_plan_cache_is_rebuilt_but_not_trusted(service, monkeypatch):
    from app.pilot.query import plan_query
    from threading import Event
    from app.pilot.contracts import content_hash
    from app.pilot.service import WORKFLOW_VERSION

    settings = PilotSettings()
    # The local provider starts without an English formulation; the run derives
    # one from the Latin query itself, so the stored payload carries None.
    payload = dict(query="new materials", english_query=None, english_source="user",
                   as_of=date.today().isoformat(),
                   settings=settings.model_dump(mode="json"))
    cache_key = content_hash({"workflow": WORKFLOW_VERSION, "request": payload})
    service._cache_write(cache_key, "plan", {"tampered": True})
    expected = plan_query("new materials", None, as_of=date.today(), request_id="test", scope_ids=(),
                          cancel=Event(), english_query="new materials")
    def collect(plan, *_):
        assert plan.english_query == expected.english_query
        return CorpusSnapshot(snapshot_id="empty", plan_hash=plan.plan_hash, purpose="discovery", as_of=plan.as_of,
            created_at="2026-09-10T12:00:00Z", documents=(), coverage=empty_coverage(plan), normalizer_version="test", deduplication_version="test")
    monkeypatch.setattr("app.pilot.service.collect_snapshot", collect)
    monkeypatch.setattr(service, "_discover", lambda *_: dict(candidates=[], input_records=0, unique_studies=0,
        retained_studies=0, quality="insufficient_data", limitations=[]))
    run_id = service.start("new materials")
    service.coordinator.wait()
    assert service.get(run_id)["state"] == "succeeded"
    saved = json.loads((service.data_dir / "cache" / "analyses" / cache_key / "plan.json").read_text(encoding="utf-8"))
    assert saved["english_query"] == "new materials"


def empty_science(service, monkeypatch):
    from app.pilot.approved_sources import unavailable_snapshot

    def collect(plan, *_):
        return CorpusSnapshot(snapshot_id="radar-empty", plan_hash=plan.plan_hash, purpose="discovery",
            as_of=plan.as_of, created_at="2026-09-27T12:00:00Z", documents=(), coverage=empty_coverage(plan),
            normalizer_version="test", deduplication_version="test")

    monkeypatch.setattr("app.pilot.approved_sources.collect_approved_sources",
                        lambda query, *, as_of, **_kwargs: unavailable_snapshot(query, as_of))
    monkeypatch.setattr("app.pilot.service.collect_snapshot", collect)
    monkeypatch.setattr(service, "_discover", lambda *_: dict(candidates=[], input_records=0, unique_studies=0,
        retained_studies=0, quality="insufficient_data", limitations=[]))


def test_web_analysis_counts_the_technology_top_alongside_and_saves_it(service, monkeypatch):
    empty_science(service, monkeypatch)
    finishing, calls = Event(), []

    def radar(query, plan, discovery, approved_sources, *, cancel, progress):
        calls.append((query, discovery["purpose"], approved_sources()["query"]))
        progress(1, 3)
        # The radar ends only after the analysis reached its own last stage.
        assert finishing.wait(10)
        return {"state": "ready", "result": {"technologies": [{"title": "quantum dots"}], "excluded": []}}

    def score(*_args, **_kwargs):
        finishing.set()
        return {}

    monkeypatch.setattr(service, "_technology_radar", radar)
    monkeypatch.setattr("app.pilot.publication_confidence.score_publications", score)
    run_id = service.start("quantum sensors", "quantum sensors", collection_profile="fast")
    service.coordinator.wait()
    assert service.get(run_id)["state"] == "succeeded", service.get(run_id)["error"]
    assert calls == [("quantum sensors", "discovery", "quantum sensors")]
    saved = {"state": "ready", "result": {"technologies": [{"title": "quantum dots"}], "excluded": []}}
    assert service.result(run_id)["radar"] == saved
    assert service.coordinator.checkpoint_value(run_id, "radar") == saved


def test_desktop_analysis_has_no_technology_top_and_a_radar_failure_keeps_the_result(service, monkeypatch):
    empty_science(service, monkeypatch)
    calls = []

    def broken(*_args, **_kwargs):
        calls.append(True)
        raise RuntimeError("OpenAlex недоступен")

    monkeypatch.setattr(service, "_technology_radar", broken)
    desktop = service.start("quantum sensors", "quantum sensors")
    service.coordinator.wait()
    assert service.get(desktop)["state"] == "succeeded"
    assert service.result(desktop)["radar"] is None and not calls
    web = service.start("quantum sensors", "quantum sensors", collection_profile="fast")
    service.coordinator.wait()
    assert service.get(web)["state"] == "succeeded", service.get(web)["error"]
    assert service.result(web)["radar"] == {"state": "unavailable", "message": "Не удалось собрать ТОП технологий."}
    assert calls == [True]
