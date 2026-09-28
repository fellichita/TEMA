"""Verifier input/provenance contracts; no model downloads or fake inference."""

from copy import deepcopy
from datetime import date, datetime, UTC
import json
import os
import signal

import pytest

from app.backend.contracts import DocumentRecord
from app.pilot.contracts import QueryPlan, SearchQuery
from app.runtime.credentials import CredentialStore, LEGACY_ENVIRONMENT
from app.runtime.worker import WorkerCancelled
from tools import verify_pilot_runtime as verifier
from tests.platform_support import require_symlinks


def test_discovery_manifest_does_not_claim_to_have_executed_scoring():
    from app.pilot.contracts import SIGNAL_METHODOLOGY_VERSION
    from app.pilot.discovery import DISCOVERY_VERSION

    assert verifier.executed_versions() == {
        "discovery": DISCOVERY_VERSION, "methodology": None,
        "methodology_scope": "not_executed_discovery_only",
        "service_methodology": SIGNAL_METHODOLOGY_VERSION,
    }


def input_payload():
    query = QueryPlan(original_query="membrane separation", english_query="membrane separation", language="en",
        definition="membrane separation", subdirections=("selective membranes",),
        queries=(SearchQuery(source="openalex", text="membrane separation"),),
        completed_years=tuple(range(2020, 2026)), as_of=date(2026, 9, 15), planner_version="test")
    doc = DocumentRecord(source="openalex", source_id="W1", title="Selective membranes",
        abstract="An example for parser validation.", url="https://openalex.org/W1", publication_year=2025,
        date_precision="year", fetched_at=datetime(2026, 9, 15, tzinfo=UTC))
    return {"query_plan": query.model_dump(mode="json"), "documents": [doc.model_dump(mode="json")],
            "discovery_snapshot_id": "a" * 64}


def result(version="semantic-discovery-v1"):
    value = {"schema_version": 3, "discovery_version": version, "plan_hash": "p", "discovery_snapshot_id": "s",
        "input_records": 2, "unique_studies": 2, "retained_studies": 2, "studies": [
            {"study_id": "work-a", "representative_index": 0, "input_indices": [0]},
            {"study_id": "work-b", "representative_index": 1, "input_indices": [1]}],
        "relevance": [{"study_id": sid, "score": score, "decision": "retained", "exclusion_score": None}
                      for sid, score in (("work-a", .9), ("work-b", .85))],
        "clusters": [{"candidate_id": "cluster-a", "study_ids": ["work-a"]}],
        "candidates": [{"candidate_id": "cluster-a", "discovery_study_ids": ["work-a"]}],
        "early_signal_study_ids": ["work-b"],
        "excluded_studies": [{"study_id": "retracted-work", "reason": "explicitly_retracted"}]}
    if version != "semantic-discovery-v1":
        value.update(review_queue=[{"candidate_id": "single-b", "discovery_study_ids": ["work-b"]}],
                     review_queue_metadata=[], hierarchy=[], unassigned_study_ids=["work-b"])
    if version == verifier.STATUS_DISCOVERY_VERSION:
        value.update(publication_status_version="publication-status/2.0.0", status_evidence=[])
        for row in value["studies"]:
            row.update(identity_version="study-families-v2-publication-units", identity_keys=[row["study_id"]],
                       family_links=[])
    return value


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e10000}', b'[]', b''])
def test_strict_json_rejects_duplicates_nonfinite_and_invalid_roots(tmp_path, raw):
    path = tmp_path / "input.json"
    path.write_bytes(raw)
    with pytest.raises(verifier.VerificationError):
        verifier.read_json(path)
    assert path.read_bytes() == raw


def test_input_byte_limit_and_symlink_refusal(tmp_path):
    require_symlinks()
    path = tmp_path / "input.json"
    path.write_text('{"value":123}')
    with pytest.raises(verifier.VerificationError):
        verifier.read_json(path, maximum=2)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(verifier.VerificationError):
        verifier.read_json(link)


@pytest.mark.parametrize("mutate", [
    lambda value: value.update(unknown_field=True),
    lambda value: value.update(discovery_snapshot_id="not-a-digest"),
    lambda value: value.update(documents=[]),
    lambda value: value.update(options={"maximum_candidates": 999}),
    lambda value: value["documents"][0].pop("title"),
])
def test_input_uses_production_schema_and_bounds(mutate):
    value = input_payload()
    mutate(value)
    with pytest.raises(verifier.VerificationError):
        verifier.validate_input(value)


def test_existing_output_is_refused_before_model_inference_and_input_is_unchanged(tmp_path, monkeypatch):
    source, output, model = tmp_path / "input.json", tmp_path / "already", tmp_path / "model"
    raw = json.dumps(input_payload()).encode()
    source.write_bytes(raw)
    model.mkdir()
    output.mkdir()
    saved = output / "result.json"
    saved.write_text("previous result")
    monkeypatch.setattr(verifier, "run_in_process", lambda *_a, **_k: pytest.fail("Inference must not start"))
    with pytest.raises(verifier.VerificationError, match="already exists"):
        verifier.verify(source, model, output)
    assert source.read_bytes() == raw
    assert saved.read_text(encoding="utf-8") == "previous result"


def test_output_cannot_write_into_model_and_artifacts_cannot_be_replaced(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    with pytest.raises(verifier.VerificationError):
        verifier.claim_output_directory(model / "audit", model)
    destination = verifier.claim_output_directory(tmp_path / "fresh", model)
    path = destination / "manifest.json"
    verifier.write_new(path, {"state": "original"})
    with pytest.raises(FileExistsError):
        verifier.write_new(path, {"state": "replacement"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"state": "original"}


def test_credential_policy_discards_inherited_and_later_environment_keys_without_keychain(monkeypatch):
    for name in LEGACY_ENVIRONMENT:
        monkeypatch.setenv(name, "SYNTHETIC-KEY")
    monkeypatch.setenv("CUSTOM_SERVICE_TOKEN", "SYNTHETIC-TOKEN")
    monkeypatch.setenv("HTTP_PROXY", "http://synthetic:password@invalid.example")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "SYNTHETIC-KEY-ID")
    monkeypatch.setattr(CredentialStore, "_load_backend", lambda _self: pytest.fail("Keychain must not be loaded"))
    credentials = verifier.IsolatedCredentialStore()
    try:
        assert not any(name in os.environ for name in LEGACY_ENVIRONMENT)
        assert "CUSTOM_SERVICE_TOKEN" not in os.environ
        assert "HTTP_PROXY" not in os.environ
        assert "AWS_ACCESS_KEY_ID" not in os.environ
        assert credentials.get("openalex_api_key") is None
        monkeypatch.setenv("OPENAI_API_KEY", "LATE-SYNTHETIC-KEY")
        assert credentials.import_legacy_environment().imported == ()
        assert "OPENAI_API_KEY" not in os.environ
        assert credentials.get("openai_api_key") is None
    finally:
        credentials.close()


def test_v1_trace_does_not_claim_orphans_are_in_a_review_queue():
    trace = {row["study_id"]: row for row in verifier.study_trace(result())}
    assert trace["work-a"]["cluster_ids"] == ["cluster-a"]
    assert trace["work-b"]["queue_candidate_ids"] == []
    assert trace["work-b"]["unassigned"]
    assert trace["work-b"]["queue_format"] == "no_explicit_queue_v1"
    assert trace["retracted-work"]["representative"] is None
    assert trace["retracted-work"]["exclusion_reasons"] == ["explicitly_retracted"]


@pytest.mark.parametrize("version", ["semantic-discovery-v3", "semantic-discovery-v4-primary-units",
                                    verifier.STATUS_DISCOVERY_VERSION])
def test_known_queue_formats_trace_representatives_and_memberships(version):
    docs = input_payload()["documents"] * 2
    trace = {row["study_id"]: row for row in verifier.study_trace(result(version), docs)}
    assert trace["work-b"]["queue_candidate_ids"] == ["single-b"]
    assert trace["work-b"]["representative"]["title"] == "Selective membranes"
    assert trace["work-b"]["representative"]["input_index"] == 1


@pytest.mark.parametrize("mutate", [
    lambda value: value.update(discovery_version="semantic-discovery-v99"),
    lambda value: value.update(review_queue=[]),
    lambda value: value["clusters"][0].update(study_ids=["missing"]),
    lambda value: value["relevance"].pop(),
    lambda value: value["studies"][0].update(representative_index=-1),
    lambda value: value["relevance"][0].update(score=float("nan")),
    lambda value: value.update(retained_studies=1),
])
def test_unknown_mixed_or_broken_results_fail_closed(mutate):
    value = result()
    mutate(value)
    with pytest.raises(verifier.VerificationError):
        verifier.study_trace(value)


def test_baseline_diff_reports_actual_changes_without_claiming_identical_raw_inputs():
    before = result()
    after = deepcopy(before)
    after["relevance"][0]["score"] = .92
    diff = verifier.compare_results(before, after)
    assert diff["identity_fields_match"]
    assert diff["exact_same_input_proven"] is False
    assert [row["study_id"] for row in diff["changed_studies"]] == ["work-a"]
    assert diff["changed_studies"][0]["before"]["score"] == .9
    after["plan_hash"] = "different-plan"
    assert not verifier.compare_results(before, after)["identity_fields_match"]


def status_result(*, notice=False):
    from app.pilot.contracts import content_hash
    from app.pilot.retractions import retraction_status_evidence

    raw = input_payload()["documents"][0]
    raw["raw_metadata"] = ({"update-to": [{"DOI": "10.1234/absent-paper", "type": "retraction"}]}
                           if notice else {"is_retracted": True})
    document = DocumentRecord.model_validate(raw)
    value = result(verifier.STATUS_DISCOVERY_VERSION)
    value.update(input_records=1, unique_studies=0, retained_studies=0, studies=[], relevance=[], candidates=[],
        clusters=[], review_queue=[], unassigned_study_ids=[], early_signal_study_ids=[],
        excluded_studies=[{"study_id": document.document_key, "reason": "explicitly_retracted"}],
        status_evidence=[item | {"revision_id": content_hash(document)} for item in retraction_status_evidence(
            document, rules_version="publication-status/2.0.0")])
    return value, [document.model_dump(mode="json")]


@pytest.mark.parametrize("notice", [True, False])
def test_v5_status_evidence_is_verified_against_exact_input_revision(notice):
    value, documents = status_result(notice=notice)
    trace = verifier.study_trace(value, documents)
    assert len(trace) == 1  # A notice's absent target does not become an invented input study.
    assert trace[0]["publication_status_version"] == "publication-status/2.0.0"
    assert trace[0]["publication_status_evidence"] == value["status_evidence"]
    assert trace[0]["decision"] == "excluded_before_embedding"
    assert trace[0]["exclusion_reasons"] == ["explicitly_retracted"]


@pytest.mark.parametrize("change", [
    {"revision_id": "a" * 64}, {"target_key": "doi:10.1234/invented"},
    {"source_id": "unknown-source"}, {"kind": "unknown-status"},
])
def test_v5_rejects_unsupported_or_unmatched_publication_status(change):
    value, documents = status_result()
    value["status_evidence"][0].update(change)
    with pytest.raises(verifier.VerificationError):
        verifier.study_trace(value, documents)


@pytest.mark.parametrize("change", [
    {"publication_status_version": "publication-status/1.0.0"}, {"status_evidence": None},
])
def test_v5_requires_explicit_known_status_shape(change):
    value = result(verifier.STATUS_DISCOVERY_VERSION)
    value.update(change)
    with pytest.raises(verifier.VerificationError):
        verifier.study_trace(value)


def test_v5_identity_rule_is_frozen_and_legacy_cannot_gain_status_fields():
    value = result(verifier.STATUS_DISCOVERY_VERSION)
    value["studies"][0]["identity_version"] = "study-families-v1"
    with pytest.raises(verifier.VerificationError):
        verifier.study_trace(value)
    value = result("semantic-discovery-v3")
    value["status_evidence"] = []
    with pytest.raises(verifier.VerificationError):
        verifier.study_trace(value)


def test_v5_family_reason_keeps_its_archived_revision_references():
    value = result(verifier.STATUS_DISCOVERY_VERSION)
    value["studies"][0].update(identity_keys=["work-a", "work-a-preprint"], revisions=[
        {"revision_id": "a" * 64, "document_key": "work-a-preprint"},
        {"revision_id": "b" * 64, "document_key": "work-a"}], family_links=[{
            "reason": "exact_title_author_preprint_crosswalk", "source_key": "work-a-preprint",
            "target_key": "work-a", "source_revision_id": "a" * 64, "target_revision_id": "b" * 64,
            "matching_author_count": 2}])
    trace = {row["study_id"]: row for row in verifier.study_trace(value)}
    assert trace["work-a"]["family_links"] == value["studies"][0]["family_links"]
    value["studies"][0]["family_links"][0]["target_revision_id"] = "c" * 64
    with pytest.raises(verifier.VerificationError, match="archived revision"):
        verifier.study_trace(value)


@pytest.mark.parametrize("deadline", [0, -1, 86401, float("nan"), float("inf"), True])
def test_invalid_deadlines_refused_without_creating_output(tmp_path, deadline):
    with pytest.raises(verifier.VerificationError):
        verifier.verify(tmp_path / "input", tmp_path / "model", tmp_path / "output", deadline_seconds=deadline)
    assert not (tmp_path / "output").exists()


def test_cancel_file_and_signal_handler_prevent_start_and_restore_prior_handler(tmp_path):
    cancel_path = tmp_path / "cancel"
    cancel = verifier.Cancellation(cancel_path)
    before = signal.getsignal(signal.SIGINT)
    with verifier.cancellation_signals(cancel):
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        assert cancel.is_set()
    assert signal.getsignal(signal.SIGINT) is before
    cancel_path.touch()
    with pytest.raises(WorkerCancelled):
        verifier.verify(tmp_path / "input", tmp_path / "model", tmp_path / "output", cancel=verifier.Cancellation(cancel_path))
    assert not (tmp_path / "output").exists()


def test_rss_report_exposes_units_and_scope_without_calling_inference():
    report = verifier.resource_snapshot()
    assert report["unit"] == "bytes"
    if report["status"] == "available":
        assert report["parent_peak_rss_bytes"] >= 0
        assert report["children_peak_rss_bytes"] >= 0
        assert "Not a per-worker delta" in report["scope"]
