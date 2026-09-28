"""Small replayable status proofs, reference integrity and untouched old hashes."""
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timezone

import pytest

from app.pilot.archive import DocumentArchive
from app.pilot.contracts import content_hash
from app.pilot.evidence import EvidenceError, quote_evidence
from app.pilot.methodology import AssessmentInput, evaluate_candidate
from app.pilot.publication_status import collect_status_context
from app.runtime.jobs import TaskCancelled
from tests.test_pilot_evidence import Context, document, query_plan
from tests.test_pilot_result_signals import automatic_result


AS_OF = query_plan().as_of


def test_material_closure_preserves_implicit_evidence_and_provider_alias_bridge_without_unrelated_history(tmp_path, monkeypatch):
    archive = DocumentArchive(tmp_path)
    plain = document(1, doi=None)
    doi_copy = document(1)
    notice = document(2, raw_metadata={"update-to": [{"type": "retraction", "DOI": doi_copy.doi}]})
    unrelated = document(3)
    plain_ref, doi_ref, notice_ref, unrelated_ref = (archive.put(doc) for doc in (plain, doi_copy, notice, unrelated))
    evidence = quote_evidence(plain_ref, plain, text_field="title", quote=plain.title)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    monkeypatch.setattr(archive, "put", lambda *_args, **_kwargs: pytest.fail("Status helper must never write the archive"))
    status = collect_status_context((doi_ref, notice_ref, unrelated_ref), archive, Context(), as_of=AS_OF,
                                    evidence=(evidence,))
    assert status.withdrawn == {plain.document_key, doi_copy.document_key, notice.document_key}
    assert {ref.revision_id for ref in status.references} == {ref.revision_id for ref in (plain_ref, doi_ref, notice_ref)}
    assert list(status.references) == sorted(status.references, key=lambda ref: ref.revision_id)
    assert collect_status_context(status.references, archive, Context(), as_of=AS_OF) == status
    assert {path: path.read_bytes() for path in tmp_path.rglob("*.json")} == before
    with pytest.raises(FrozenInstanceError):
        status.withdrawn = frozenset()


def test_every_version_relation_dependency_is_retained_and_unrelated_paper_dropped(tmp_path):
    archive = DocumentArchive(tmp_path)
    first = document(1, raw_metadata={"is_retracted": True, "relation": {
        "is-preprint-of": [{"id-type": "doi", "id": "10.1234/study2"}]}})
    bridge = document(2, raw_metadata={"relation": {"is-version-of": [{"id-type": "doi", "id": "10.1234/study3"}]}})
    last = document(3)
    unrelated = document(4)
    references = tuple(archive.put(doc) for doc in (first, bridge, last, unrelated))
    status = collect_status_context(references, archive, Context(), as_of=AS_OF)
    assert status.withdrawn == {first.document_key, bridge.document_key, last.document_key}
    assert {ref.revision_id for ref in status.references} == {ref.revision_id for ref in references[:3]}
    assert collect_status_context(reversed(status.references), archive, Context(), as_of=AS_OF) == status


def test_reverse_supplement_parent_is_material_but_is_not_itself_a_supporting_asset(tmp_path):
    archive = DocumentArchive(tmp_path)
    asset = document(1)
    parent = document(2, raw_metadata={"relation": {"is-supplemented-by": [{"id-type": "doi", "id": asset.doi}]}})
    unrelated = document(3)
    refs = tuple(archive.put(doc) for doc in (asset, parent, unrelated))
    status = collect_status_context(refs, archive, Context(), as_of=AS_OF)
    assert status.supporting == {asset.document_key}
    assert status.withdrawn == frozenset()
    assert set(status.references) == set(refs[:2])
    assert collect_status_context(status.references, archive, Context(), as_of=AS_OF) == status


def test_identical_references_are_deduplicated_and_unrelated_clean_context_is_empty(tmp_path):
    archive = DocumentArchive(tmp_path)
    ref = archive.put(document(1))
    status = collect_status_context((ref, ref), archive, Context(), as_of=AS_OF)
    assert status.references == () and not status.withdrawn and not status.supporting


@pytest.mark.parametrize("change", [
    {"study_id": "foreign-study"}, {"source_id": "foreign-source"}, {"source": "crossref"},
    {"text_hash": "a" * 64}, {"publication_year": 2001}, {"publicly_available_at": date(2020, 1, 1)},
    {"observed_at": datetime(2020, 1, 1, tzinfo=timezone.utc)},
])
def test_reference_fields_cannot_be_detached_from_the_original_revision(tmp_path, change):
    archive = DocumentArchive(tmp_path)
    ref = archive.put(document(1))
    with pytest.raises(EvidenceError, match="immutable source"):
        collect_status_context((ref, ref.model_copy(update=change)), archive, Context(), as_of=AS_OF)


def test_hash_is_rechecked_even_for_an_archive_object_returning_wrong_data(tmp_path, monkeypatch):
    archive = DocumentArchive(tmp_path)
    ref = archive.put(document(1))
    monkeypatch.setattr(archive, "get", lambda _identifier: document(2))
    with pytest.raises(EvidenceError, match="revision hash"):
        collect_status_context((ref,), archive, Context(), as_of=AS_OF)


@pytest.mark.parametrize("changes", [
    {"year": 2027},
    {"year": 2026, "publication_month": 10, "date_precision": "month"},
    {"year": 2026, "publication_date": date(2026, 9, 11), "date_precision": "day"},
])
def test_future_publication_cannot_supply_status(tmp_path, changes):
    archive = DocumentArchive(tmp_path)
    ref = archive.put(document(1, **changes))
    with pytest.raises(EvidenceError, match="after as_of"):
        collect_status_context((ref,), archive, Context(), as_of=AS_OF)


def test_fetched_later_is_preserved_as_observed_status_without_inventing_a_historical_retrieval_cutoff(tmp_path):
    archive = DocumentArchive(tmp_path)
    later = document(1, fetched_at=datetime(2026, 12, 31, tzinfo=timezone.utc), raw_metadata={"is_retracted": True})
    ref = archive.put(later)
    status = collect_status_context((ref,), archive, Context(), as_of=AS_OF)
    assert status.references == (ref,) and status.withdrawn == {later.document_key}


@pytest.mark.parametrize("change", [{"source_url": "https://example.org/other"}, {"quote": "a fabricated quote"},
                                    {"text_hash": "a" * 64}, {"study_id": "other"}, {"start": -1}])
def test_implicit_evidence_must_match_real_archived_source_and_exact_quote(tmp_path, change):
    archive = DocumentArchive(tmp_path)
    record = document(1)
    ref = archive.put(record)
    evidence = quote_evidence(ref, record, text_field="title", quote=record.title)
    with pytest.raises((EvidenceError, ValueError)):
        collect_status_context((), archive, Context(), as_of=AS_OF, evidence=(evidence.model_copy(update=change),))


def test_unique_revision_limit_and_cancellation_are_checked_before_unbounded_work(tmp_path, monkeypatch):
    import app.pilot.publication_status as status_module
    archive = DocumentArchive(tmp_path)
    refs = tuple(archive.put(document(index)) for index in range(3))
    monkeypatch.setattr(status_module, "MAX_STATUS_REVISIONS", 2)
    with pytest.raises(EvidenceError, match="100000 unique revisions"):
        collect_status_context(refs, archive, Context(), as_of=AS_OF)
    cancelled = Context()
    cancelled.cancel_event.set()
    monkeypatch.setattr(archive, "get", lambda *_: pytest.fail("Cancelled context must not read the archive"))
    with pytest.raises(TaskCancelled):
        collect_status_context(refs, archive, cancelled, as_of=AS_OF)


def test_optional_status_provenance_does_not_change_legacy_input_hash_and_is_version_guarded(tmp_path):
    _, _, artifact = automatic_result(tmp_path)
    inputs = artifact.inputs
    encoded, digest = inputs.model_dump_json(), content_hash(inputs)
    assert "publication_status_revisions" not in inputs.model_dump(mode="json")
    assert AssessmentInput.model_validate_json(encoded).model_dump_json() == encoded
    assert content_hash(AssessmentInput.model_validate_json(encoded)) == digest
    reference = artifact.inputs.antecedents.snapshot.documents
    # Empty antecedent fixtures use another real archived reference for this
    # schema check; scientific status extraction is tested above separately.
    if not reference:
        from app.pilot.archive import DocumentArchive
        reference = (DocumentArchive(tmp_path / "schema-extra").put(document(99999)),)
    changed = AssessmentInput.model_validate(inputs.model_dump(mode="python") | {
        "publication_status_revisions": reference, "source_novelty": ()})
    for version in ("3.0.0", "3.1.0", "3.2.0", "3.3.0"):
        with pytest.raises(ValueError, match="requires methodology 3.4"):
            evaluate_candidate(changed, version=version)
    assert evaluate_candidate(changed, version="3.4.0").input_hash == content_hash(changed)
    with pytest.raises(ValueError, match="Duplicate publication status"):
        AssessmentInput.model_validate(inputs.model_dump(mode="python") | {"publication_status_revisions": reference * 2})
