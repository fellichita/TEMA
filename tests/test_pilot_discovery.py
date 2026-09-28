"""Structural discovery checks; fixture vectors are not semantic-quality claims."""

from concurrent.futures import CancelledError
from datetime import date, datetime, timezone
import json
from threading import Event

import numpy as np
import pytest

from app.backend.contracts import DocumentRecord
from app.pilot.contracts import Candidate, QueryLimits, QueryPlan, SearchQuery
from app.pilot.discovery import DiscoveryError, DiscoveryOptions, deduplicate, discover, task
from app.pilot.encoder import ChunkedText, EncoderError, TextUnit

NOW = datetime(2026, 9, 10, tzinfo=timezone.utc)


def plan(direction="advanced membrane separation"):
    return QueryPlan(original_query=direction, language="en", english_query=direction, definition=direction,
                     subdirections=(direction,), queries=(SearchQuery(source="openalex", text=direction),),
                     completed_years=tuple(range(2020, 2026)), as_of=date(2026, 9, 10), planner_version="test")


def document(index, **changes):
    values = dict(source="openalex", source_id=f"W{index}", title=f"Material membrane technology experiment {index}",
                  abstract="The method achieves selective separation of dissolved contaminants.",
                  url=f"https://openalex.org/W{index}", fetched_at=NOW, publication_year=2025,
                  date_precision="year", authors=("A. Researcher",))
    return DocumentRecord(**(values | changes))


class FixtureEncoder:
    fingerprint = "b" * 64

    def chunk_text(self, text, **kwargs):
        return ChunkedText((TextUnit(text, 0, len(text), 20),), False, 20)

    def encode(self, texts, *, kind="passage", cancel=None, progress=None):
        result = np.zeros((len(texts), 384), dtype=np.float32)
        if kind == "query":
            result[:, 0] = 1
        else:
            for index, text in enumerate(texts):
                result[index, 0] = 0.9
                group = 1 if "membrane" in text else 2
                result[index, group] = np.sqrt(1 - 0.9 ** 2)
                result[index, 10 + index % 50] = 0.001 * (index % 7)
            result /= np.linalg.norm(result, axis=1, keepdims=True)
        return result


def test_cross_source_doi_duplicates_retain_versions_but_count_one_study():
    records = [document(1, doi="10.1234/work"), document(2, source="crossref", doi="10.1234/work")]
    studies = deduplicate(records)
    assert len(studies) == 1
    assert studies[0]["study_id"] == "doi:10.1234/work"
    assert len(studies[0]["revisions"]) == 2


def test_query_limit_applies_to_unique_studies_not_source_variants():
    query = plan().model_copy(update={"limits": QueryLimits(discovery_documents=1)})
    records = [document(1, doi="10.1234/a"), document(2, doi="10.1234/a", source="crossref")]
    result = discover(records, query, encoder=FixtureEncoder(), snapshot_id="s")
    assert result["input_records"] == 2
    assert result["deduplicated_input_studies"] == 1
    assert len(result["studies"][0]["revisions"]) == 2
    with pytest.raises(DiscoveryError, match="лимит уникальных"):
        discover([*records, document(3, doi="10.1234/b")], query, encoder=FixtureEncoder(), snapshot_id="s")


def test_more_than_ten_thousand_variants_are_allowed_within_unique_limit():
    # Repeated versions exercise raw-volume admission without expensive inference.
    records = [document(1, doi="10.1234/a")] * 10001
    assert len(deduplicate(records)) == 1
    with pytest.raises(DiscoveryError, match="20 000"):
        deduplicate(records * 2)


def test_crossref_supplement_relation_survives_openalex_rich_abstract_variant():
    records = [document(1, doi="10.1234/parent"),
               document(2, doi="10.1234/supplement", abstract="Rich detailed abstract " * 20),
               document(3, doi="10.1234/supplement", source="crossref", abstract=None, raw_metadata={
                   "relation": {"is-supplement-to": [{"id-type": "doi", "id": "10.1234/parent"}]}})]
    result = discover(records, plan(), encoder=FixtureEncoder(), snapshot_id="s")
    assert result["unique_studies"] == 1
    assert len(result["studies"][0]["revisions"]) == 3
    assert result["studies"][0]["supplementary_study_ids"] == ["doi:10.1234/supplement"]


def test_doi_less_record_cannot_bridge_conflicting_source_identities():
    records = [document(1, doi="10.1234/a"), document(1), document(1, doi="10.1234/b")]
    with pytest.raises(DiscoveryError, match="разным DOI"):
        deduplicate(records)


def test_same_long_title_does_not_merge_different_dois_or_generic_short_titles():
    title = "A comprehensive investigation of selective membrane materials for industrial separation systems"
    records = [document(1, title=title, doi="10.1234/a"), document(2, title=title),
               document(3, title=title, doi="10.1234/b")]
    assert len(deduplicate(records)) == 3
    assert len(deduplicate([document(1, title="Editorial"), document(2, title="Editorial")])) == 2


def test_conservative_long_title_year_author_crosswalk_merges_doi_less_duplicate():
    title = "A comprehensive investigation of selective membrane materials for industrial separation systems"
    records = [document(1, title=title, doi="10.1234/a"), document(2, title=title, source="crossref")]
    assert len(deduplicate(records)) == 1


def test_any_retracted_variant_excludes_whole_study_and_preserves_reason():
    records = [document(1, doi="10.1234/a"), document(2, doi="10.1234/a", source="crossref",
                                                          raw_metadata={"is_retracted": True})]
    result = discover(records, plan(), encoder=FixtureEncoder(), snapshot_id="s")
    assert result["unique_studies"] == 0
    assert result["deduplicated_input_studies"] == 1
    assert result["excluded_studies"] == [{"study_id": "doi:10.1234/a", "reason": "explicitly_retracted"}]


def test_discovery_reports_a_counter_for_every_document_and_ends_complete():
    records = [document(index, doi=f"10.1234/w{index}") for index in range(1, 13)]
    events = []
    result = discover(records, plan(), encoder=FixtureEncoder(), snapshot_id="s",
                      progress=lambda completed, total: events.append((completed, total)))
    assert result["unique_studies"] == 12
    assert events[:12] == [(position, 12) for position in range(12)]
    assert events[-1] == (result["text_units"], result["text_units"])
    assert all(0 <= completed <= total for completed, total in events)


def test_explicit_supplement_adds_provenance_without_adding_independent_study():
    records = [document(1, doi="10.1234/parent"), document(2, doi="10.1234/supplement", raw_metadata={
        "relation": {"is-supplement-to": [{"id-type": "doi", "id": "10.1234/parent"}]}})]
    result = discover(records, plan(), encoder=FixtureEncoder(), snapshot_id="s")
    assert result["unique_studies"] == 1
    assert result["studies"][0]["study_id"] == "doi:10.1234/parent"
    assert result["studies"][0]["supplementary_study_ids"] == ["doi:10.1234/supplement"]
    assert len(result["studies"][0]["revisions"]) == 2


def test_absent_supplement_parent_and_future_publication_are_excluded():
    records = [document(1, doi="10.1234/supplement", raw_metadata={
        "relation": {"is-supplement-to": [{"id-type": "doi", "id": "10.1234/missing"}]}}),
               document(2, publication_year=2027)]
    result = discover(records, plan(), encoder=FixtureEncoder(), snapshot_id="s")
    assert result["unique_studies"] == 0
    assert {item["reason"] for item in result["excluded_studies"]} == {
        "supplement_parent_not_in_corpus", "not_publicly_available_as_of"}


def test_retracted_parent_cannot_regain_evidence_via_supplement():
    records = [document(1, doi="10.1234/parent", raw_metadata={"is_retracted": True}),
               document(2, doi="10.1234/supplement", raw_metadata={
                   "relation": {"is-supplement-to": [{"id-type": "doi", "id": "10.1234/parent"}]}})]
    result = discover(records, plan(), encoder=FixtureEncoder(), snapshot_id="s")
    assert result["unique_studies"] == 0
    assert len(result["excluded_studies"]) == 2


def test_malformed_supplement_metadata_cannot_crash_discovery():
    records = [document(1, doi="10.1234/a", raw_metadata={"relation": {"is-supplement-to": [
        {"id-type": "doi", "id": None}, {"id-type": "doi", "id": []}, {"id-type": "doi", "id": "bad"}]}})]
    assert discover(records, plan(), encoder=FixtureEncoder(), snapshot_id="s")["unique_studies"] == 1


def test_representative_and_candidate_identity_are_stable_when_input_order_changes():
    records = [document(index) for index in range(12)] + [
        document(100 + index, title=f"Electrolyte battery storage experiment {index}",
                 abstract="Solid electrolytes improve lithium transport.") for index in range(12)]
    options = DiscoveryOptions(minimum_cluster_size=4, minimum_samples=2, algorithm="hdbscan")
    first = discover(records, plan(), encoder=FixtureEncoder(), snapshot_id="s", options=options)
    second = discover(list(reversed(records)), plan(), encoder=FixtureEncoder(), snapshot_id="s", options=options)
    assert len(first["candidates"]) == 2
    assert {item["candidate_id"] for item in first["candidates"]} == {
        item["candidate_id"] for item in second["candidates"]}
    for candidate in first["candidates"]:
        Candidate.model_validate(candidate)
        assert candidate["specificity"] == "uncertain"
    assert first["quality"] == "partial"
    assert first["methodology_calibrated"] is False
    assert sum(cluster["unique_study_count"] for cluster in first["clusters"]) == 24


@pytest.mark.parametrize("direction", [
    "synthetic field alpha", "synthetic field beta", "synthetic field gamma", "synthetic field delta",
    "synthetic field epsilon", "synthetic field zeta", "synthetic field eta", "synthetic field theta",
    "тестовая область альфа", "тестовая область бета", "тестовая область гамма", "тестовая область дельта",
])
def test_unseen_direction_enters_same_discovery_path_without_profiles(direction, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy direction profiles must not be consulted")

    monkeypatch.setattr("app.ml.directions.direction_profile", forbidden)
    result = discover([document(index) for index in range(4)], plan(direction), encoder=FixtureEncoder(),
                      snapshot_id="new-field")
    assert result["plan_hash"] == plan(direction).plan_hash
    assert result["unique_studies"] == 4
    assert not result["candidates"]
    assert len(result["early_signal_study_ids"]) == 4


def test_empty_and_irrelevant_input_are_honest_insufficient_data():
    empty = discover([], plan(), encoder=FixtureEncoder(), snapshot_id="s")
    assert empty["quality"] == "insufficient_data"
    assert empty["unique_studies"] == 0
    rejected = discover([document(1)], plan(), encoder=FixtureEncoder(), snapshot_id="s",
                        options=DiscoveryOptions(minimum_relevance=0.99))
    assert rejected["retained_studies"] == 0
    assert rejected["relevance"][0]["decision"] == "below_semantic_threshold"
    assert rejected["quality"] == "insufficient_data"


def test_cancelled_discovery_stops_before_embedding():
    cancel = Event()
    cancel.set()
    with pytest.raises(CancelledError):
        discover([document(1)], plan(), encoder=FixtureEncoder(), snapshot_id="s", cancel=cancel)


def test_nonfinite_embedding_fails_instead_of_publishing_candidates():
    class Invalid(FixtureEncoder):
        def encode(self, texts, **kwargs):
            return np.full((len(texts), 384), np.nan, dtype=np.float32)

    with pytest.raises(EncoderError):
        discover([document(1)], plan(), encoder=Invalid(), snapshot_id="s")


@pytest.mark.parametrize("options", [
    {"minimum_relevance": float("nan")}, {"minimum_relevance": 2}, {"minimum_cluster_size": True},
    {"minimum_samples": 0}, {"maximum_candidates": 31}, {"algorithm": "fabricate"},
])
def test_invalid_algorithm_parameters_are_rejected(options):
    with pytest.raises(DiscoveryError):
        DiscoveryOptions(**options)


def test_worker_validates_input_and_does_not_overwrite_it(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"query_plan": {}, "documents": []}))
    original = source.read_bytes()
    with pytest.raises(DiscoveryError):
        task(source, source, Event())
    assert source.read_bytes() == original
    source.write_text("{\"documents\": NaN}")
    with pytest.raises(DiscoveryError):
        task(source, tmp_path / "output.json", Event())
    assert not (tmp_path / "output.json").exists()


def test_worker_writes_atomic_result_and_preserves_immutable_input(tmp_path, monkeypatch):
    source, target = tmp_path / "input.json", tmp_path / "output.json"
    payload = {"query_plan": plan().model_dump(mode="json"),
               "documents": [document(index).model_dump(mode="json") for index in range(3)],
               "discovery_snapshot_id": "saved-snapshot"}
    source.write_text(json.dumps(payload))
    original = source.read_bytes()
    monkeypatch.setattr("app.pilot.discovery.MultilingualEncoder", lambda *args, **kwargs: FixtureEncoder())
    task(source, target, Event())
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["schema_version"] == 3
    assert result["unique_studies"] == 3
    assert result["scope_review_required"] is True
    assert source.read_bytes() == original
    assert sorted(path.name for path in tmp_path.iterdir()) == ["input.json", "output.json"]


def test_worker_rejects_deep_and_nonfinite_metadata_before_loading_model(tmp_path):
    source, target = tmp_path / "input.json", tmp_path / "output.json"
    nested = []
    for _ in range(66):
        nested = [nested]
    payload = {"query_plan": plan().model_dump(mode="json"), "documents": [], "options": {"nested": nested}}
    source.write_text(json.dumps(payload))
    with pytest.raises(DiscoveryError, match="вложенности"):
        task(source, target, Event())
    payload["options"] = {"minimum_relevance": "replace-me"}
    source.write_text(json.dumps(payload).replace('"replace-me"', "1e309"))
    with pytest.raises(DiscoveryError, match="нечисловое"):
        task(source, target, Event())
    assert not target.exists()


def test_worker_rejects_duplicate_json_keys(tmp_path):
    source = tmp_path / "input.json"
    source.write_text('{"documents": [], "documents": []}')
    with pytest.raises(DiscoveryError, match="повторяющиеся"):
        task(source, tmp_path / "output.json", Event())


def test_explicit_preprint_relation_unifies_different_dois_with_all_provenance():
    records = [document(1, doi="10.1234/journal", raw_metadata={"type": "journal-article"}),
               document(2, doi="10.1234/preprint", publication_year=2023, raw_metadata={
                   "type": "posted-content", "relation": {"is-preprint-of": [
                       {"id-type": "doi", "id": "https://doi.org/10.1234/journal"}]}})]
    study, = deduplicate(records)
    assert study["study_id"] == "doi:10.1234/journal"
    assert study["identity_keys"] == ["doi:10.1234/journal", "doi:10.1234/preprint"]
    assert study["first_publication_year"] == 2023
    assert len(study["revisions"]) == 2
    assert study["family_links"][0]["reason"] == "explicit_version_relation"
    assert len(deduplicate(records, version="semantic-discovery-v1")) == 2


def test_family_links_stay_with_their_final_family_after_multiple_unions():
    records = [
        document(1, doi="10.1234/journal-a", raw_metadata={"type": "journal-article"}),
        document(2, doi="10.1234/journal-b", raw_metadata={"type": "journal-article"}),
        document(3, doi="10.1234/preprint-a", raw_metadata={"type": "posted-content", "relation": {
            "is-preprint-of": [{"id-type": "doi", "id": "10.1234/journal-a"}]}}),
        document(4, doi="10.1234/preprint-b", raw_metadata={"type": "posted-content", "relation": {
            "is-preprint-of": [{"id-type": "doi", "id": "10.1234/journal-b"}]}}),
        document(5, doi="10.1234/manuscript-a", raw_metadata={"type": "posted-content", "relation": {
            "is-version-of": [{"id-type": "doi", "id": "10.1234/preprint-a"}]}}),
    ]
    studies = deduplicate(records)
    assert len(studies) == 2
    by_family = {study["study_id"]: study for study in studies}
    assert len(by_family["doi:10.1234/journal-a"]["family_links"]) == 2
    assert len(by_family["doi:10.1234/journal-b"]["family_links"]) == 1
    for study in studies:
        assert all(link["source_key"] in study["identity_keys"]
                   and link["target_key"] in study["identity_keys"] for link in study["family_links"])


def test_preprint_journal_crosswalk_uses_exact_title_and_multiple_authors():
    title = "Pulsed vector atomic magnetometer using an alternating fast-rotating field"
    records = [document(1, doi="10.1234/journal", title=title, authors=("Tao Wang", "W. Lee", "Mark Limes")),
               document(2, doi="10.1234/preprint", title=title.upper(), publication_year=2023,
                        authors=("Tao Wang", "Won-Jae Lee", "Mark Limes"), raw_metadata={"type": "preprint"})]
    assert len(deduplicate(records)) == 1
    assert len(deduplicate([records[0], records[1].model_copy(update={"authors": ("Other team",)})])) == 2
    assert len(deduplicate([records[0], records[1].model_copy(update={"raw_metadata": {}})])) == 2


def test_conference_extension_and_citation_are_not_identity_relations():
    records = [document(1, doi="10.1234/journal"), document(2, doi="10.1234/conference", raw_metadata={
        "relation": {"is-derived-from": [{"id-type": "doi", "id": "10.1234/journal"}],
                     "references": [{"id-type": "doi", "id": "10.1234/journal"}]}})]
    assert len(deduplicate(records)) == 2


def test_future_journal_does_not_remove_past_preprint_or_supply_future_text():
    records = [document(1, doi="10.1234/journal", publication_year=2027,
                        abstract="Future text must not be embedded."),
               document(2, doi="10.1234/preprint", publication_year=2025, raw_metadata={
                   "relation": {"is-preprint-of": [{"id-type": "doi", "id": "10.1234/journal"}]}})]

    class AsOfEncoder(FixtureEncoder):
        def chunk_text(self, text, **kwargs):
            assert "Future text" not in text
            return super().chunk_text(text, **kwargs)

    result = discover(records, plan(), encoder=AsOfEncoder(), snapshot_id="past")
    assert result["unique_studies"] == 1
    assert result["studies"][0]["study_id"] == "doi:10.1234/preprint"
    assert result["studies"][0]["input_indices"] == [1]


def test_later_metadata_without_abstract_does_not_erase_rich_representative():
    from datetime import timedelta

    records = [document(1, doi="10.1234/work", abstract="Original full abstract."),
               document(2, doi="10.1234/work", abstract=None, fetched_at=NOW + timedelta(days=1))]
    study, = deduplicate(records)
    assert study["representative_index"] == 0
    assert len(study["revisions"]) == 2


def test_query_synonyms_are_encoded_without_replacing_scope_anchors():
    captured = []

    class CapturingEncoder(FixtureEncoder):
        def encode(self, texts, *, kind="passage", **kwargs):
            if kind == "query":
                captured.extend(texts)
            return super().encode(texts, kind=kind, **kwargs)

    query = plan().model_copy(update={"synonyms": ("membrane fractionation", "мембранное разделение")})
    discover([document(1)], query, encoder=CapturingEncoder(), snapshot_id="s")
    assert captured[:1] == [query.original_query]
    assert set(query.synonyms).issubset(captured)


def test_rare_unclustered_study_is_reviewable_without_positive_lifecycle(monkeypatch):
    def no_groups(vectors, matrix, options, cancel):
        return np.full(len(vectors), -1), "hdbscan", {"nmf": {"converged": True}}

    monkeypatch.setattr("app.pilot.discovery._labels", no_groups)
    records = [document(1, title="Quantum spin resonance in engineered proteins for multimodal sensing")]
    result = discover(records, plan(), encoder=FixtureEncoder(), snapshot_id="s")
    candidate, = result["review_queue"]
    assert candidate["discovery_study_ids"] == [records[0].document_key]
    assert candidate["specificity"] == "uncertain"
    assert "category" not in candidate
    assert result["unassigned_study_ids"] == [records[0].document_key]
    assert result["review_queue_metadata"][0]["origin"] == "unclustered_study"


def test_large_cluster_does_not_displace_smaller_group_before_review(monkeypatch):
    def clusters(vectors, matrix, options, cancel):
        return np.array([0] * 36 + [1] * 3), "hdbscan", {"nmf": {"converged": True}}

    monkeypatch.setattr("app.pilot.discovery._labels", clusters)
    # Identical geometry ensures no synthetic hierarchy split in this scheduling test.
    class SameEncoder(FixtureEncoder):
        def encode(self, texts, **kwargs):
            vectors = np.zeros((len(texts), 384), dtype=np.float32)
            vectors[:, 0] = 1
            return vectors

    result = discover([document(index) for index in range(39)], plan(), encoder=SameEncoder(), snapshot_id="s",
                      options=DiscoveryOptions(maximum_candidates=1))
    assert len(result["candidates"][0]["discovery_study_ids"]) == 3
    assert len(result["review_queue"][0]["discovery_study_ids"]) == 36


def test_semantically_separated_large_group_has_reviewable_child_branches():
    from app.pilot.hierarchy import refine_groups

    vectors = np.zeros((48, 384), dtype=np.float32)
    vectors[:24, 0], vectors[24:, 1] = 1, 1
    leaves, nodes = refine_groups([list(range(48))], vectors)
    assert len(leaves) == 2
    assert {tuple(leaf["members"]) for leaf in leaves} == {tuple(range(24)), tuple(range(24, 48))}
    assert all(leaf["parent_node_id"] == 0 for leaf in leaves)
    assert nodes[0]["split"] is True


def test_ambiguous_preprint_cannot_bridge_two_distinct_journal_dois():
    title = "A comprehensive investigation of selective membrane materials for industrial separation systems"
    records = [document(1, doi="10.1234/a", title=title), document(2, doi="10.1234/b", title=title),
               document(3, doi="10.1234/preprint", title=title, raw_metadata={"type": "preprint"})]
    assert len(deduplicate(records)) == 3


def test_malformed_version_metadata_does_not_crash_or_create_identity():
    title = "A comprehensive investigation of selective membrane materials for industrial separation systems"
    records = [document(1, doi="10.1234/a", title=title, raw_metadata={"type": {}, "relation": []}),
               document(2, doi="10.1234/b", title=title, raw_metadata={"type": [], "relation": {
                   "is-preprint-of": [{"id-type": "doi", "id": []}, {"id-type": "doi", "id": "bad"}]}})]
    assert len(deduplicate(records)) == 2


def test_rich_preprint_canonical_keeps_actual_snapshot_identity_for_evidence(tmp_path):
    from types import SimpleNamespace

    from app.pilot.archive import DocumentArchive
    from app.pilot.contracts import CorpusSnapshot, Coverage
    from app.pilot.evidence import member_documents, quote_evidence

    records = [document(1, doi="10.1234/journal", abstract=None, raw_metadata={"type": "journal-article"}),
               document(2, doi="10.1234/preprint", abstract="The membrane mechanism improves separation efficiency.",
                        raw_metadata={"type": "preprint", "relation": {"is-preprint-of": [
                            {"id-type": "doi", "id": "10.1234/journal"}]}})]
    query = plan()
    archive = DocumentArchive(tmp_path / "revisions")
    references = tuple(archive.put(record) for record in records)
    data = CorpusSnapshot(snapshot_id="s", plan_hash=query.plan_hash, purpose="discovery", created_at=NOW,
                          as_of=query.as_of, documents=references, normalizer_version="test",
                          deduplication_version="doi-source-id-v1", coverage=(Coverage(
                              source="openalex", purpose="discovery", query_hash="a" * 64, state="complete",
                              requested_years=(2025,), completed_years=(2025,), pagination_exhausted=True,
                              scanned_records=2, accepted_records=2, comparable=False),))
    result = discover(records, query, encoder=FixtureEncoder(), snapshot_id="s")
    candidate = Candidate.model_validate(result["review_queue"][0])
    assert candidate.discovery_study_ids == ("doi:10.1234/preprint",)
    selected, = member_documents(candidate, data, archive, SimpleNamespace(check_cancelled=lambda: None))
    reference, rich = selected
    assert reference.study_id == rich.document_key == "doi:10.1234/preprint"
    assert rich.abstract == records[1].abstract
    assert result["studies"][0]["identity_keys"] == ["doi:10.1234/journal", "doi:10.1234/preprint"]
    evidence = quote_evidence(reference, rich, quote=rich.abstract, text_field="abstract")
    assert evidence.study_id == rich.document_key


def test_equal_text_version_observation_order_does_not_change_family_canonical():
    from datetime import timedelta

    records = [document(1, doi="10.1234/journal", raw_metadata={"type": "journal-article"}),
               document(2, doi="10.1234/preprint", raw_metadata={"type": "preprint", "relation": {
                   "is-preprint-of": [{"id-type": "doi", "id": "10.1234/journal"}]}})]
    first, = deduplicate(records)
    later, = deduplicate([records[0], records[1].model_copy(update={"fetched_at": NOW + timedelta(days=10)})])
    assert first["study_id"] == later["study_id"] == "doi:10.1234/journal"


def test_supplement_parent_alias_resolves_after_rich_preprint_becomes_canonical():
    records = [document(1, doi="10.1234/journal", abstract=None),
               document(2, doi="10.1234/preprint", raw_metadata={"type": "preprint", "relation": {
                   "is-preprint-of": [{"id-type": "doi", "id": "10.1234/journal"}]}}),
               document(3, doi="10.1234/supplement", raw_metadata={"type": "component", "relation": {
                   "is-supplement-to": [{"id-type": "doi", "id": "10.1234/journal"}]}})]
    result = discover(records, plan(), encoder=FixtureEncoder(), snapshot_id="s")
    study, = result["studies"]
    assert result["unique_studies"] == 1
    assert study["study_id"] == "doi:10.1234/preprint"
    assert study["supplementary_study_ids"] == ["doi:10.1234/supplement"]
    assert len(study["revisions"]) == 3
    assert not result["excluded_studies"]


def test_a_full_sized_corpus_is_not_refused_by_the_structural_walk():
    """The node ceiling must follow the revision ceiling, not contradict it.

    Measured on a real standard-profile run: 9 991 archived revisions averaged
    454 JSON nodes each — about 4.5 million. Against the previous fixed ceiling
    of 1 000 000 that run failed at discovery, so only the 400-document fast
    profile ever completed and no corpus was ever large enough to cluster.
    """
    from app.pilot.discovery import MAX_JSON_NODES, MAX_REVISIONS

    observed_nodes_per_revision = 454
    largest_observed = 1920
    assert MAX_JSON_NODES >= MAX_REVISIONS * largest_observed
    assert MAX_JSON_NODES // observed_nodes_per_revision > 10_000


def test_a_full_corpus_queues_more_hypotheses_than_the_old_result_bound():
    """A result must be able to carry what a full corpus actually produces.

    Measured on a standard-profile run: 8 601 unique studies, 7 590 retained,
    7 607 queued hypotheses. Against the previous bound of 7 500 the completed
    analysis was rejected at its final validation and the whole run was lost.
    """
    from app.pilot.contracts import QUEUE_LIMIT
    from app.pilot.discovery import MAX_REVISIONS

    measured_queue_of_a_ten_thousand_document_corpus = 7_607
    assert QUEUE_LIMIT > measured_queue_of_a_ten_thousand_document_corpus
    # One individual study per retained document, plus disjoint leaves.
    assert QUEUE_LIMIT >= MAX_REVISIONS // 2 + 2_500
