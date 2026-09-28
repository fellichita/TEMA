"""Received source evidence stays auditable under current and frozen admission rules.

Vectors below are fixtures: these tests assert provenance, accounting and study
membership, not semantic accuracy. The rich texts and notices are real revisions.
"""
from datetime import date, timedelta
import json
from pathlib import Path
import random

import pytest

from app.backend.contracts import DocumentRecord, SourcePage
from app.pilot.archive import DocumentArchive
from app.pilot.discovery import DISCOVERY_VERSION, _eligible_studies, deduplicate, discover
from app.pilot.retractions import (
    LEGACY_STATUS_RULES, STATUS_RULES, is_explicitly_retracted, retracted_family_keys, retraction_status_evidence,
)
from app.pilot.sources import (collect_snapshot, exclusion_reason, primary_research_exclusion,
                               supporting_asset_keys, supporting_asset_status)
from app.runtime.credentials import CredentialStore
from tests.test_pilot_antecedents import bundle_for
from tests.test_pilot_discovery import FixtureEncoder, plan as discovery_plan
from tests.test_pilot_evidence import document, query_plan
from tests.test_pilot_history import scenario as scenario
from tests.test_pilot_sources import Context, Provider, plan as source_plan

FIXTURE = Path(__file__).parent / "fixtures/pilot/publication_status_real_records.json"


def real_records(field):
    return [DocumentRecord.model_validate(item) for item in json.loads(FIXTURE.read_text(encoding="utf-8"))[field]]


def replay(records, plan=None):
    return discover(records, plan or discovery_plan(), encoder=FixtureEncoder(), snapshot_id="status-regression")


@pytest.mark.parametrize("sources", [("openalex", "crossref"), ("crossref", "openalex")])
@pytest.mark.parametrize("reverse_relation", [False, True])
@pytest.mark.parametrize("repeat", [False, True])
def test_provider_order_preserves_supplement_metadata_through_archive_and_discovery(tmp_path, sources, reverse_relation, repeat):
    parent = document(1)
    rich = document(2, abstract="Substantial supplementary measurements. " * 20)
    if reverse_relation:
        crossref = document(1, source="crossref", abstract=None, raw_metadata={"relation": {
            "is-supplemented-by": [{"id-type": "doi", "id": rich.doi}]}})
    else:
        crossref = document(2, source="crossref", abstract=None, raw_metadata={"relation": {
            "is-supplement-to": [{"id-type": "doi", "id": parent.doi}]}})
    records = [parent, rich, crossref]
    original = [record.model_dump_json() for record in records]

    def factory(source):
        delivered = [record for record in records if record.source == source]
        if repeat:
            delivered += delivered
        return Provider([SourcePage(documents=tuple(delivered), scanned=len(delivered), exhausted=True)])

    archive = DocumentArchive(tmp_path)
    snapshot = collect_snapshot(source_plan(sources=sources), Context(), archive, CredentialStore(), provider_factory=factory)
    assert len(snapshot.documents) == 3 == len({ref.revision_id for ref in snapshot.documents})
    restored = [archive.get(ref.revision_id) for ref in snapshot.documents]
    assert supporting_asset_keys(restored) == {rich.document_key}
    assert sorted(restored, key=lambda record: (record.source, record.source_id)) == sorted(
        records, key=lambda record: (record.source, record.source_id))
    assert sum(item.accepted_records for item in snapshot.coverage) == (2 if reverse_relation else 1)
    assert sum(item.scanned_records for item in snapshot.coverage) == (6 if repeat else 3)
    assert sum(item.rejected_records for item in snapshot.coverage) == (6 if repeat else 3) - (2 if reverse_relation else 1)
    assert all(item.state == "complete" and item.unresolved_records == 0 for item in snapshot.coverage)
    assert any("archived_supporting_revisions" in item.reasons for item in snapshot.coverage)
    result = replay(restored)
    assert result["unique_studies"] == 1
    study = result["studies"][0]
    assert study["study_id"] == parent.document_key
    assert restored[study["representative_index"]] == parent
    assert study["supplementary_study_ids"] == [rich.document_key]
    assert {revision["revision_id"] for revision in study["revisions"]} == {ref.revision_id for ref in snapshot.documents}
    assert [record.model_dump_json() for record in records] == original


def test_real_publisher_correction_with_retraction_update_is_versioned_and_targeted():
    notice = real_records("status_notices")[0]
    assert notice.doi == "10.1007/978-3-031-28516-5_12"
    target_doi = notice.raw_metadata["update-to"][0]["DOI"]
    target = document(41, doi=target_doi, title="Critical Analysis of the Recent Advances, Applications and Uses on Luminescence Thermometry")
    linked = document(42, raw_metadata={"relation": {"is-preprint-of": [{"id-type": "doi", "id": target_doi}]}})
    discussion = document(43, title="Retraction notices and publication practices", raw_metadata={"reference": [
        {"DOI": target_doi, "article-title": notice.title}]})
    docs = [notice, target, linked, discussion]
    assert not is_explicitly_retracted(notice, rules_version=LEGACY_STATUS_RULES)
    assert exclusion_reason(notice, date(2020, 1, 1), date(2026, 12, 31), rules_version=LEGACY_STATUS_RULES) is None
    assert is_explicitly_retracted(notice, rules_version=STATUS_RULES)
    assert not is_explicitly_retracted(discussion, rules_version=STATUS_RULES)
    assert retracted_family_keys(docs, rules_version=STATUS_RULES) == {notice.document_key, target.document_key, linked.document_key}
    assert retracted_family_keys(docs, rules_version=LEGACY_STATUS_RULES) == set()
    families = deduplicate(docs, version=DISCOVERY_VERSION)
    assert len(families) == 3  # The notice is not a manifestation of the target.
    result = replay(docs)
    assert [study["study_id"] for study in result["studies"]] == [discussion.document_key]
    evidence = result["status_evidence"]
    assert len(evidence) == 1 and evidence[0]["metadata_path"] == "update-to[0]"
    assert evidence[0]["source_key"] == notice.document_key and evidence[0]["target_key"] == target.document_key
    assert evidence[0]["revision_id"] in {rev["revision_id"] for family in families for rev in family["revisions"]}
    assert result["publication_status_version"] == STATUS_RULES


@pytest.mark.parametrize("sources", [("openalex", "crossref"), ("crossref", "openalex")])
def test_notice_target_rejection_survives_collection_order_and_duplicate_pages(tmp_path, sources):
    notice = real_records("status_notices")[0]
    target = document(51, doi=notice.raw_metadata["update-to"][0]["DOI"])
    archive = DocumentArchive(tmp_path)
    snapshot = collect_snapshot(source_plan(sources=sources), Context(), archive, CredentialStore(),
        provider_factory=lambda source: Provider([SourcePage(documents=(notice, notice) if source == "crossref" else (target,),
                                                             scanned=2 if source == "crossref" else 1, exhausted=True)]))
    assert len(snapshot.documents) == 2
    assert sum(item.accepted_records for item in snapshot.coverage) == 0
    assert sum(item.rejected_records for item in snapshot.coverage) == 3
    assert replay([archive.get(ref.revision_id) for ref in snapshot.documents])["unique_studies"] == 0


def test_both_real_notices_and_three_real_withdrawn_versions_are_removed_before_inference():
    records = real_records("status_notices")
    withdrawn = json.loads((FIXTURE.parent / "retracted_nested_neural_networks.json").read_text(encoding="utf-8"))["documents"]
    records += [DocumentRecord.model_validate(item) for item in withdrawn]
    assert len(records) == 5 and all(is_explicitly_retracted(item) for item in records)
    result = replay(records)
    assert result["unique_studies"] == result["text_units"] == 0 and not result["candidates"]
    assert {item["reason"] for item in result["excluded_studies"]} == {"explicitly_retracted"}


def test_actual_paper_analyzing_retracted_computer_science_articles_is_not_withdrawn():
    discussion = real_records("discussion_records")[0]
    assert discussion.doi == "10.1371/journal.pone.0285383"
    assert "retract" in discussion.title.lower() and "retract" in discussion.abstract.lower()
    for version in (LEGACY_STATUS_RULES, STATUS_RULES):
        assert not is_explicitly_retracted(discussion, rules_version=version)
        assert not retraction_status_evidence(discussion, rules_version=version)
    result = replay([*real_records("status_notices"), discussion])
    assert [study["study_id"] for study in result["studies"]] == [discussion.document_key]


def test_actual_supplement_labeled_as_journal_article_is_supporting_evidence_only():
    asset = real_records("supporting_records")[0]
    assert asset.doi == "10.1037/pas0001186.supp" and asset.document_type == "journal-article"
    assert primary_research_exclusion(asset, rules_version=LEGACY_STATUS_RULES) is None
    assert supporting_asset_keys([asset]) == {asset.document_key}
    result = replay([asset])
    assert result["unique_studies"] == 0 and result["excluded_studies"] == [
        {"study_id": asset.document_key, "reason": "supporting_asset_not_research"}]


@pytest.mark.parametrize("metadata", [
    {"update-to": [{"type": "correction", "DOI": "10.1234/target"}]},
    {"update-to": [{"type": "retraction", "DOI": "not a DOI"}]},
    {"update-to": [{"type": "retraction", "DOI": None}]},
    {"reference": [{"type": "retraction", "DOI": "10.1234/target"}]},
    {"relation": {"references": [{"id-type": "doi", "id": "10.1234/target"}]}},
])
def test_non_status_citations_and_malformed_updates_do_not_withdraw_a_paper(metadata):
    record = document(1, raw_metadata=metadata)
    assert not retraction_status_evidence(record)
    assert not retracted_family_keys([record])


@pytest.mark.parametrize("version", [LEGACY_STATUS_RULES, STATUS_RULES])
def test_false_self_status_words_remain_primary_research(version):
    record = document(1, title="Quantifying tissue retraction during surgical experiments",
                      abstract="This paper has not been retracted. We measure mechanical tissue retraction.")
    assert not is_explicitly_retracted(record, rules_version=version)
    assert primary_research_exclusion(record, rules_version=version) is None


@pytest.mark.parametrize("seed", range(6))
def test_actual_twelve_lost_abstracts_are_selected_under_reordering_and_later_empty_metadata(seed):
    records = real_records("rich_revisions")
    expected = {key: max((record for record in records if record.document_key == key), key=lambda record: len(record.abstract or ""))
                for key in json.loads(FIXTURE.read_text(encoding="utf-8"))["representative_study_ids"]}
    assert len(expected) == 12 and len(records) == 24
    # An observation timestamp is not a measure of how much source text is available.
    records += [record.model_copy(update={"abstract": None, "fetched_at": record.fetched_at + timedelta(days=1)})
                for record in expected.values()]
    random.Random(seed).shuffle(records)
    families = deduplicate(records, version=DISCOVERY_VERSION)
    assert len(families) == 12
    for family in families:
        representative = records[family["representative_index"]]
        assert representative == expected[family["study_id"]]
        assert len(representative.abstract) >= 543
        assert len(family["revisions"]) == 3
        assert family["identity_keys"] == [representative.document_key]
    # Original v1 replay keeps the demonstrated metadata-only representative bug.
    assert all(not records[family["representative_index"]].abstract for family in deduplicate(records, version="semantic-discovery-v1"))


@pytest.mark.parametrize("explicit_link", [False, True])
@pytest.mark.parametrize("dataset_has_doi", [False, True])
def test_dataset_cannot_merge_into_a_paper_via_posted_content_or_identical_title(explicit_link, dataset_has_doi):
    title = "A comprehensive investigation of selective membrane materials for industrial separation systems"
    paper = document(11, title=title, authors=("A. Author", "B. Author"), raw_metadata={"type": "journal-article"})
    metadata = {"type": "posted-content", "subtype": "dataset"}
    if explicit_link:
        metadata["relation"] = {"is-version-of": [{"id-type": "doi", "id": paper.doi}]}
    asset = document(12, title=title, doi="10.1234/data" if dataset_has_doi else None,
                     authors=paper.authors, raw_metadata=metadata, abstract="Dataset measurements. " * 100)
    docs = [asset, paper]
    families = deduplicate(docs, version=DISCOVERY_VERSION)
    assert len(families) == 2
    selected, excluded = _eligible_studies(families, docs, discovery_plan(), None)
    assert [item["study_id"] for item in selected] == [paper.document_key]
    assert excluded == [{"study_id": asset.document_key, "reason": "supporting_asset_not_research"}]
    assert len(deduplicate(docs, version="semantic-discovery-v4-primary-units")) == 1
    assert primary_research_exclusion(asset, rules_version=LEGACY_STATUS_RULES) is None


def test_dataset_and_status_do_not_fabricate_antecedents_and_old_bundles_replay(scenario, monkeypatch):
    from app.pilot.antecedents import verify_antecedents

    archive, _, _, _, candidate, _, _ = scenario
    old_data = document(81, year=1901, document_type="dataset")
    real_paper = document(82, year=2010)
    current, _ = bundle_for(scenario, (old_data, real_paper))
    assert current.version == "openalex-antecedent-title-review/1.2.0"
    assert current.earliest_observed_year == 2010 and current.matched_study_ids == (real_paper.document_key,)
    for old_version in ("openalex-antecedent-title-review/1.0.0", "openalex-antecedent-title-review/1.1.0"):
        with monkeypatch.context() as old:
            old.setattr("app.pilot.antecedents.ANTECEDENT_VERSION", old_version)
            legacy, _ = bundle_for(scenario, (old_data, real_paper))
        assert legacy.earliest_observed_year == 1901 and len(legacy.matched_study_ids) == 2
        encoded, digest = legacy.model_dump_json(), legacy.bundle_hash
        verify_antecedents(legacy, candidate, query_plan(), archive)
        assert legacy.model_dump_json() == encoded and legacy.bundle_hash == digest
    notice = document(83, year=2015, raw_metadata={"update-to": [{"DOI": real_paper.doi, "type": "retraction"}]})
    withdrawn, _ = bundle_for(scenario, (real_paper, notice))
    assert not withdrawn.matched_study_ids and withdrawn.earliest_observed_year is None


def test_unknown_rule_versions_fail_closed():
    record = document(1)
    for method in (is_explicitly_retracted, primary_research_exclusion):
        with pytest.raises(ValueError, match="Unknown publication-status"):
            method(record, rules_version="publication-status/unknown")


def test_later_doi_and_type_metadata_propagate_through_exact_source_id_without_order_dependence():
    plain = document(91, doi=None)
    doi_copy = document(91)
    dataset = document(91, document_type="dataset")
    withdrawal = document(91, raw_metadata={"is_retracted": True})
    for ordered in ([plain, doi_copy, dataset], [dataset, doi_copy, plain]):
        assert supporting_asset_keys(ordered) == {plain.document_key, doi_copy.document_key}
    for ordered in ([plain, doi_copy, withdrawal], [withdrawal, doi_copy, plain]):
        assert retracted_family_keys(ordered) == {plain.document_key, doi_copy.document_key}
        assert retracted_family_keys(ordered, rules_version=LEGACY_STATUS_RULES) == {doi_copy.document_key}


def test_supporting_status_retains_local_proof_without_marking_reverse_parent_as_asset():
    plain = document(91, doi=None)
    doi_copy = document(91)
    dataset = document(91, document_type="dataset")
    asset = document(92)
    parent = document(93, raw_metadata={"relation": {
        "is-supplemented-by": [{"id-type": "doi", "id": asset.doi}]}})
    records = (plain, doi_copy, dataset, asset, parent, document(94))

    for version in (LEGACY_STATUS_RULES, STATUS_RULES):
        keys, material = supporting_asset_status(records, rules_version=version)
        assert keys == supporting_asset_keys(records, rules_version=version)
        assert material == frozenset(index for index, record in enumerate(records)
            if supporting_asset_keys((record,), rules_version=version))
        assert 2 in material and doi_copy.document_key in keys
        assert parent.document_key not in keys
        if version == STATUS_RULES:
            assert plain.document_key in keys and asset.document_key in keys and 4 in material
        else:
            assert plain.document_key not in keys and asset.document_key not in keys and 4 not in material


def test_reverse_parent_relation_cannot_make_dataset_the_earliest_antecedent(scenario):
    asset = document(81, year=1901)
    parent = document(82, year=2010, raw_metadata={"relation": {
        "is-supplemented-by": [{"id-type": "doi", "id": asset.doi}]}})
    bundle, _ = bundle_for(scenario, (asset, parent))
    assert bundle.matched_study_ids == (parent.document_key,)
    assert bundle.earliest_observed_year == 2010


def test_current_family_link_provenance_is_stable_under_input_order_and_duplicate_revision():
    title = "A comprehensive investigation of selective membrane materials for industrial separation systems"
    preprint = document(11, title=title, authors=("Same Author",), raw_metadata={"type": "posted-content"})
    journal = document(12, title=title, authors=preprint.authors, raw_metadata={"type": "journal-article"})
    before = deduplicate([preprint, journal], version=DISCOVERY_VERSION)[0]
    after = deduplicate([journal, preprint, preprint], version=DISCOVERY_VERSION)[0]
    assert before["family_links"] == after["family_links"]
    assert before["study_id"] == after["study_id"]
    link = before["family_links"][0]
    assert {link["source_revision_id"], link["target_revision_id"]} == {rev["revision_id"] for rev in before["revisions"]}
