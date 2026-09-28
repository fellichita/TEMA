"""Real Coordinator/_run, library and source-page boundaries for scientific TOP."""

import pytest

from app.backend.contracts import SourcePage
from app.pilot.contracts import AnalysisResult, Candidate, TrendCard
from app.pilot.evidence import admission_hash
from app.pilot.export import export_result, read_result_package
from app.pilot.methodology import AssessmentArtifact
from app.pilot.service import PilotService
from app.runtime.credentials import CredentialStore
from tests.test_pilot_evidence import candidate, document, query_plan
from tests.test_pilot_history import frozen
from tests.test_pilot_result_signals import automatic_result
from tests.test_pilot_review import decision_for
from tests.test_pilot_signal_evidence import NOVELTY, EXPERIMENT


def source_documents(counts=(0, 0, 0, 1, 2, 4)):
    docs = tuple(document(year * 100 + index, year=year) for year, count in zip(
        range(2020, 2026), counts, strict=True) for index in range(count))
    return (*docs[:-1], docs[-1].model_copy(update={"abstract": document(1).abstract + " " + NOVELTY + " " + EXPERIMENT}))


class SourceBoundary:
    """Only source I/O is replaced; real collectors classify coverage and archive data."""
    def __init__(self, docs, calls, *, field_counts=(1000,) * 6, historical_complete=True):
        self.docs, self.calls = docs, calls
        self.field_counts = field_counts
        self.historical_complete = historical_complete
        self.closed = False

    def iter_pages(self, request, cancel):
        self.calls.append(request)
        if request.max_results == 1 and request.topic == query_plan().english_query:
            count = self.field_counts[request.from_date.year - 2020]
            docs = (document(999, year=request.from_date.year),) if count else ()
            yield SourcePage(documents=docs, scanned=len(docs), total_available=count, exhausted=not bool(count))
            return
        if request.until_date.year < 2020:
            yield SourcePage(scanned=0, total_available=0, exhausted=True)
            return
        selected = tuple(item for item in self.docs if request.from_date.year <= item.publication_year <= request.until_date.year)
        is_history = request.from_date.year == 2020
        yield SourcePage(documents=selected, scanned=len(selected), total_available=len(selected),
                         exhausted=self.historical_complete if is_history else True)

    def close(self):
        self.closed = True


def runtime_for(tmp_path, monkeypatch, docs, *, field_counts=(1000,) * 6, historical_complete=True):
    credentials = CredentialStore()
    monkeypatch.setattr(credentials, "get", lambda _: None)
    monkeypatch.setattr("app.pilot.encoder.verify_artifacts", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.pilot.service.plan_query", lambda *_args, **_kwargs: query_plan())
    calls, providers = [], []

    def factory(*_args):
        provider = SourceBoundary(docs, calls, field_counts=field_counts, historical_complete=historical_complete)
        providers.append(provider)
        return provider

    monkeypatch.setattr("app.pilot.sources.make_provider", factory)
    monkeypatch.setattr("app.pilot.antecedents.make_provider", factory)
    # Any unexpected API call fails this scenario instead of silently becoming partial coverage.
    monkeypatch.setattr("httpx.Client.send", lambda *_args, **_kwargs: pytest.fail("Unexpected real HTTP dispatch"))
    runtime = PilotService(tmp_path / "profile", credentials)

    def discovered(context, snapshot, plan):
        item = frozen(candidate(snapshot))
        return {"candidates": [item.model_dump(mode="json")], "review_queue": [],
                "input_records": len(snapshot.documents), "retained_studies": len(snapshot.documents)}

    monkeypatch.setattr(runtime, "_discover", discovered)
    return runtime, calls, providers


def wait_success(runtime, job):
    runtime.coordinator.wait(timeout=10)
    row = runtime.get(job)
    assert row["state"] == "succeeded", row.get("error")
    payload = runtime.result(job)
    return AnalysisResult.model_validate(payload["result"]), tuple(AssessmentArtifact.model_validate(item) for item in payload["assessments"])


def test_real_service_from_pages_to_top_and_portable_export_without_ai(tmp_path, monkeypatch):
    runtime, calls, providers = runtime_for(tmp_path, monkeypatch, source_documents())
    try:
        job = runtime.start(query_plan().original_query, query_plan().english_query)
        result, artifacts = wait_success(runtime, job)
        assert result.methodology_version == "3.4.0"
        assert len(result.cards) == len(artifacts) == 1
        card, artifact = result.cards[0], artifacts[0]
        assert card.candidate.specificity == "specific_technology"
        assert card.category == "weak_signal_candidate" and result.top_trend_ids == (card.candidate.candidate_id,)
        assert artifact.inputs.source_novelty and artifact.inputs.antecedents.search_complete
        assert artifact.inputs.novelty is None and artifact.assessment.relative_growth.observed_growth
        assert not artifact.assessment.growth_confirmed
        assert len([request for request in calls if request.max_results == 1]) == 6
        assert all(provider.closed for provider in providers)
        path = tmp_path / "real-run.trendresult"
        runtime.export_result(job, str(path))
        with read_result_package(path) as package:
            assert package.result == result and package.assessments == artifacts
            assert package.result.top_trend_ids == result.top_trend_ids
    finally:
        runtime.close()


@pytest.mark.parametrize("counts,field,complete,reason", [
    ((1, 1, 1, 10, 10, 10), (100, 100, 100, 1000, 1000, 1000), True, "no_observed_field_relative_growth"),
    ((1, 1, 1, 2, 100, 3), (1000,) * 6, True, "recent_decline_or_transient_burst"),
    ((0, 0, 0, 1, 2, 4), (1000,) * 6, False, "incomplete_or_incomparable_history"),
    ((0, 0, 0, 1, 2, 4), (1000, None, 1000, 1000, 1000, 1000), True, "field_exposure_unavailable"),
])
def test_real_service_retains_noise_guards_and_marks_partial_early_evidence(tmp_path, monkeypatch, counts, field, complete, reason):
    runtime, _, _ = runtime_for(tmp_path, monkeypatch, source_documents(counts), field_counts=field, historical_complete=complete)
    try:
        job = runtime.start(query_plan().original_query, query_plan().english_query)
        result, artifacts = wait_success(runtime, job)
        assert len(result.cards) == len(artifacts) == 1
        if reason in {"incomplete_or_incomparable_history", "field_exposure_unavailable"}:
            assert result.top_trend_ids == (result.cards[0].candidate.candidate_id,)
            assert artifacts[0].assessment.signal_priority == 10
            assert artifacts[0].assessment.confidence == "low"
            assert not artifacts[0].assessment.growth_confirmed
        else:
            assert result.top_trend_ids == () and artifacts[0].assessment.signal_priority is None
        assert reason in (*artifacts[0].assessment.gate_failures, *artifacts[0].assessment.limitations)
        runtime.export_result(job, str(tmp_path / "negative.trendresult"))
    finally:
        runtime.close()


def test_refining_frozen_candidate_keeps_specificity_other_cards_and_queue_without_ai(tmp_path, monkeypatch):
    original, archive, artifact = automatic_result(tmp_path / "original")
    target = original.cards[0].candidate

    def other(identifier):
        changed = Candidate.model_validate(target.model_dump(mode="python") | {"candidate_id": identifier,
            "label": identifier, "specificity": "uncertain"})
        return changed.model_copy(update={"admission_rule_hash": admission_hash(changed)})

    extra = TrendCard(candidate=other("other-card"), methodology_version="3.2.0", category="unassessed_cluster",
                      quality="partial", claims=(), evidence=(), limitations=("Needs specific mechanism assessment",))
    queued = other("other-queued-hypothesis")
    original = AnalysisResult.model_validate(original.model_dump(mode="python") | {"cards": (*original.cards, extra),
        "candidate_queue": (queued,), "quality": "partial", "limitations": ("Additional hypotheses require assessment",)})
    path = tmp_path / "original.trendresult"
    export_result(path, original, archive, (artifact,))
    docs = tuple(archive.get(item.revision_id) for item in original.snapshots[0].documents)
    runtime, _, _ = runtime_for(tmp_path / "runtime", monkeypatch, docs)
    try:
        source_id = runtime.import_result(str(path))["id"]
        refined_job = runtime.refine_candidate(source_id, target.candidate_id)
        result, artifacts = wait_success(runtime, refined_job)
        revised = next(item for item in result.cards if item.candidate.candidate_id == target.candidate_id)
        assert revised.candidate.specificity == "specific_technology"
        assert revised.category == "weak_signal_candidate" and target.candidate_id in result.top_trend_ids
        assert next(item for item in result.cards if item.candidate.candidate_id == "other-card") == extra
        assert result.candidate_queue == (queued,)
        assert AnalysisResult.model_validate(runtime.result(source_id)["result"]) == original
        assert next(item for item in artifacts if item.inputs.candidate.candidate_id == target.candidate_id).inputs.field_exposure
        runtime.export_result(refined_job, str(tmp_path / "refined.trendresult"))
    finally:
        runtime.close()


def test_actual_manual_review_service_preserves_methodology_exposure_and_author_inputs(tmp_path, monkeypatch):
    from app.pilot.antecedents import AntecedentBundle

    original, archive, artifact = automatic_result(tmp_path / "original")
    path = tmp_path / "original.trendresult"
    export_result(path, original, archive, (artifact,))
    docs = tuple(archive.get(item.revision_id) for item in original.snapshots[0].documents)
    runtime, _, _ = runtime_for(tmp_path / "runtime", monkeypatch, docs)
    try:
        source_id = runtime.import_result(str(path))["id"]
        job = runtime.begin_review(source_id, original.cards[0].candidate.candidate_id)
        runtime.coordinator.wait(timeout=10)
        row = runtime.review_progress(job)
        assert row["state"] == "succeeded", row.get("error")
        prepared = row["review_data"]
        bundle = AntecedentBundle.model_validate(prepared["bundle"])
        card = TrendCard.model_validate(prepared["card"])
        values = decision_for(bundle, card).model_dump(mode="json")
        for key in ("schema_version", "version", "reviewed_at", "candidate_id", "admission_rule_hash", "bundle_hash"):
            values.pop(key, None)
        saved = runtime.apply_review(job, values)
        payload = runtime.result(saved["id"])
        result = AnalysisResult.model_validate(payload["result"])
        reviewed = AssessmentArtifact.model_validate(payload["assessments"][0])
        assert result.methodology_version == reviewed.assessment.methodology_version == "3.2.0"
        assert result.cards[0].category == "early_signal"
        assert reviewed.inputs.field_exposure == artifact.inputs.field_exposure
        assert reviewed.inputs.source_novelty == artifact.inputs.source_novelty
        assert result.top_trend_ids == original.top_trend_ids
        assert AnalysisResult.model_validate(runtime.result(source_id)["result"]) == original
        runtime.export_result(saved["id"], str(tmp_path / "reviewed.trendresult"))
    finally:
        runtime.close()
