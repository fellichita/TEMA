"""Reviewed results survive the real local library and portable-package boundary."""

from datetime import datetime, UTC
import shutil
from threading import Event

import pytest

from app.pilot.antecedents import collect_antecedents
from app.pilot.archive import DocumentArchive
from app.pilot.contracts import AnalysisResult, CorpusSnapshot, Coverage, content_hash
from app.pilot.export import export_result, read_result_package
from app.pilot.history import assess_snapshot
from app.pilot.library import ResultLibrary, write_artifact
from app.pilot.review import apply_novelty_review, record_review
from app.pilot.service import PilotService
from app.runtime.backup import ArchiveError
from app.runtime.credentials import CredentialStore
from app.runtime.jobs import TaskCancelled, TaskFailure
from tests import test_pilot_history
from tests.test_pilot_antecedents import Provider
from tests.test_pilot_evidence import Context, document, query_plan
from tests.test_pilot_export import make_reviewed_result
from tests.test_pilot_review import decision_for

scenario = test_pilot_history.scenario


def test_reviewed_library_roundtrip_remains_verified_after_source_package_closes(tmp_path, monkeypatch):
    result, source_archive, artifacts, reviews = make_reviewed_result(tmp_path / "source")
    package = export_result(tmp_path / "reviewed.zip", result, source_archive, artifacts, reviews=reviews)
    store = CredentialStore()
    monkeypatch.setattr(store, "get", lambda _: None)
    runtime = PilotService(tmp_path / "profile", store)
    try:
        imported = runtime.import_result(str(package.path))
        run_id = imported["id"]
        reopened = runtime.result(run_id)
        assert reopened["result"]["cards"][0]["category"] == "confirmed_trend"
        assert reopened["reviews"][0]["decision"]["reviewer_name"] == "Test reviewer"
        shutil.rmtree(source_archive.directory)
        package.path.unlink()
        runtime.export_result(run_id, str(tmp_path / "again.zip"))
    finally:
        runtime.close()
    reopened_runtime = PilotService(tmp_path / "profile", store)
    try:
        assert reopened_runtime.result(run_id)["reviews"] == reopened["reviews"]
        with read_result_package(tmp_path / "again.zip") as restored:
            assert restored.reviews == reviews and restored.result == result
    finally:
        reopened_runtime.close()


def test_library_import_copies_review_only_revisions_outside_displayed_result_snapshots(scenario, tmp_path):
    archive, _, discovery, historical, candidate, context, source_card = scenario
    older = (document(81, year=2010), document(82, year=2011))
    bundle = collect_antecedents(candidate, query_plan(), archive, CredentialStore(), Context(),
        provider_factory=lambda _: Provider(older))
    decision = decision_for(bundle, source_card)
    expanded, novelty = apply_novelty_review(decision, bundle, source_card, archive)
    artifact, card, _ = assess_snapshot(candidate, query_plan(), historical, archive, context,
        passport=expanded, verified_novelty=novelty, antecedents=bundle)
    review = record_review(tmp_path / "reviews", decision, bundle, source_card=source_card, reviewed_card=card,
        artifact=artifact, historical_snapshot=historical, archive=archive)
    used = {evidence.revision_id for evidence in card.evidence}
    displayed_refs = tuple(reference for reference in bundle.snapshot.documents if reference.revision_id in used)
    extra_refs = {reference.revision_id for reference in bundle.snapshot.documents} - used
    assert len(displayed_refs) == 1 and len(extra_refs) == 1
    original = bundle.snapshot.coverage[0]
    subset_coverage = Coverage.model_validate(original.model_dump() | dict(state="partial", completed_years=(),
        accepted_records=1, rejected_records=1, reasons=("selected_evidence_subset",)))
    displayed = CorpusSnapshot.model_validate(bundle.snapshot.model_dump() | dict(
        snapshot_id=content_hash({"displayed": [ref.revision_id for ref in displayed_refs]}),
        documents=displayed_refs, coverage=(subset_coverage,)))
    result = AnalysisResult(result_id="reviewed-subset", run_id="original", query_plan=query_plan(),
        created_at=datetime.now(UTC), quality=card.quality, snapshots=(discovery, historical, displayed), cards=(card,))
    package = export_result(tmp_path / "subset.zip", result, archive, (artifact,), reviews=(review,))
    destination = DocumentArchive(tmp_path / "destination" / "revisions")
    library = ResultLibrary(tmp_path / "destination", destination)
    imported = library.import_file(package.path)
    for revision_id in extra_refs:
        assert destination.get(revision_id) == archive.get(revision_id)
    assert library.read(imported["id"])["reviews"][0]["bundle"] == bundle.model_dump(mode="json")
    package.path.unlink()
    assert library.read(imported["id"])["result"] == result.model_dump(mode="json")
    # The warm view must also watch earlier evidence that is not displayed.
    destination.path(next(iter(extra_refs))).write_text("{}")
    with pytest.raises((TaskFailure, ArchiveError)):
        library.documents(imported["id"], limit=1)


def test_library_cannot_strip_review_and_rehash_payload_to_preserve_confirmed_status(tmp_path):
    result, archive, artifacts, reviews = make_reviewed_result(tmp_path / "source")
    library = ResultLibrary(tmp_path / "source", archive)
    legitimate = library.save_result(result, artifacts, reviews)
    payload = dict(legitimate["payload"])
    payload.pop("view_id", None)
    payload.pop("reviews")
    digest = write_artifact(library.directory, payload)
    with pytest.raises(ArchiveError, match="ReviewRecord"):
        library.read("import-" + digest)


def test_real_manual_review_service_rejects_arbitrary_nonreview_job(tmp_path, monkeypatch):
    store = CredentialStore()
    monkeypatch.setattr(store, "get", lambda _: None)
    runtime = PilotService(tmp_path / "profile", store)
    try:
        monkeypatch.setattr(runtime.coordinator, "result", lambda _: {"kind": "not-antecedents"})
        with pytest.raises(Exception, match="не содержит обзор"):
            runtime.apply_review("unrelated-job", {"kind": "new_mechanism"})
        assert not (runtime.data_dir / "reviews").exists()
    finally:
        runtime.close()


def test_actual_service_review_job_manual_submission_and_preserved_original_result(tmp_path, monkeypatch):
    from app.pilot.antecedents import AntecedentBundle
    from app.pilot.contracts import TrendCard
    from tests.test_pilot_export import make_result

    result, source_archive, artifacts = make_result(tmp_path / "source", historical=True)
    package = export_result(tmp_path / "original.zip", result, source_archive, artifacts)
    store = CredentialStore()
    monkeypatch.setattr(store, "get", lambda _: None)
    monkeypatch.setattr("app.pilot.antecedents.make_provider", lambda *_: Provider((document(81, year=2010),)))
    runtime = PilotService(tmp_path / "profile", store)
    try:
        original_id = runtime.import_result(str(package.path))["id"]
        candidate_id = result.cards[0].candidate.candidate_id
        job_id = runtime.begin_review(original_id, candidate_id)
        runtime.coordinator.wait(timeout=10)
        row = runtime.review_progress(job_id)
        assert row["state"] == "succeeded"
        assert runtime.begin_review(original_id, candidate_id) == job_id
        prepared = row["review_data"]
        bundle = AntecedentBundle.model_validate(prepared["bundle"])
        card = TrendCard.model_validate(prepared["card"])
        decision = decision_for(bundle, card)
        values = decision.model_dump(mode="json")
        for key in ("schema_version", "version", "reviewed_at", "candidate_id", "admission_rule_hash", "bundle_hash"):
            values.pop(key, None)
        rejected = dict(values, earlier_analogues_checked=False)
        with pytest.raises(Exception, match="отдельной проверки"):
            runtime.apply_review(job_id, rejected)
        assert not (runtime.data_dir / "reviews").exists()
        revised = runtime.apply_review(job_id, values)
        updated = runtime.result(revised["id"])
        assert updated["result"]["cards"][0]["category"] == "confirmed_trend"
        assert updated["reviews"][0]["decision"]["reviewer_name"] == "Test reviewer"
        assert runtime.result(original_id)["result"]["cards"][0]["category"] == "insufficient_evidence"
        assert len(list((runtime.data_dir / "reviews").glob("*.json"))) == 1
    finally:
        runtime.close()


@pytest.mark.parametrize("cancel_at", ["copy", "publication"])
def test_cancelled_reviewed_library_import_never_publishes_partial_result(tmp_path, monkeypatch, cancel_at):
    from app.pilot import library as library_module

    result, source, artifacts, reviews = make_reviewed_result(tmp_path / "source")
    package = export_result(tmp_path / "reviewed.zip", result, source, artifacts, reviews=reviews)
    archive = DocumentArchive(tmp_path / "destination" / "revisions")
    library = ResultLibrary(tmp_path / "destination", archive)
    cancel = Event()
    if cancel_at == "copy":
        original_put = archive.put

        def put_then_cancel(document):
            reference = original_put(document)
            cancel.set()
            return reference

        monkeypatch.setattr(archive, "put", put_then_cancel)
    else:
        original_verify = library_module.verify_result

        def verify_then_cancel(*args, **kwargs):
            verified = original_verify(*args, **kwargs)
            cancel.set()
            return verified

        monkeypatch.setattr(library_module, "verify_result", verify_then_cancel)
    with pytest.raises(TaskCancelled):
        library.import_file(package.path, cancel=cancel)
    assert library.count() == 0 and library.list_rows() == []
    assert not list(library.directory.glob("*.json"))
