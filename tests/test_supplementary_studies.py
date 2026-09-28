"""Supplementary records preserve provenance, not independent evidence counts."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from app.ml.contracts import AnalysisOptions
from app.ml.engine import analysis_entries, prepare
from app.ml.study_relations import author_name_key, first_author_surname, supplement_relations
from app.ml.text import clean
from app.ml.validation import exclude_known_versions
from tests.mvp_fixture import snapshot

OPTIONS = AnalysisOptions(topic="photonic neuromorphic computing")


def parent_entry():
    entry = deepcopy(snapshot()["batches"][0]["documents"][0])
    doi = "10.1021/test.123"
    entry.update(document_key="doi:" + doi, revision_id="parent-version")
    entry["document"].update(doi=doi, source_id="parent", authors=["Weiming Lin"],
                             url="https://doi.org/" + doi)
    return entry


def supplement(parent, suffix=".s001"):
    entry = deepcopy(parent)
    doi = parent["document"]["doi"] + suffix
    entry.update(document_key="doi:" + doi, revision_id="supplement" + suffix)
    entry["document"].update(doi=doi, source_id="supplement" + suffix,
                             authors=["Weiming Lin (13253381)"], url="https://doi.org/" + doi)
    entry["document"]["abstract"] += " Additional material has a longer, different passage that must not replace the paper."
    return entry


@pytest.mark.parametrize("name,expected", [
    ("Weiming Lin (13253381)", "weiming lin"), (" Weiming Lin ( 13253381 ) ", "weiming lin"),
    ("Weiming Lin", "weiming lin"), ("Alex Smith (MD)", "alex smith md"),
    ("Researcher 2020-0-0", "researcher 2020 0 0"), ("", ""),
])
def test_author_key_removes_only_a_numeric_repository_suffix(name, expected):
    assert author_name_key(name) == expected


def test_numeric_id_does_not_become_first_author_surname():
    assert first_author_surname(["Weiming Lin (13253381)"]) == ("lin",)


def test_legacy_supplement_keeps_parent_text_url_date_and_all_source_versions():
    parent = parent_entry()
    child = supplement(parent)
    entries = [parent, child]
    before = deepcopy(entries)
    studies, diagnostic = prepare(entries, OPTIONS)
    assert len(studies) == 1
    study = studies[0]
    assert (study["id"], study["doi"], study["url"], study["year"]) == (
        parent["document_key"], parent["document"]["doi"], parent["document"]["url"], 2020)
    assert study["title"] == clean(parent["document"]["title"])
    assert study["abstract"] == clean(parent["document"]["abstract"])
    assert len(study["versions"]) == 2
    assert not study["possible_versions_merged"]
    assert diagnostic["possible_versions_merged"] == 0
    assert diagnostic["supplementary_materials_attached"] == 1
    assert diagnostic["rejected"] == {"supplementary_material": 1}
    assert diagnostic["supplementary_relations"][0]["status"] == "attached"
    attached = next(version for version in study["versions"] if version.get("relation") == "supplement")
    assert attached["document_key"] == child["document_key"]
    assert attached["parent_identity"] == parent["document_key"]
    assert attached["relation_evidence"] == ["acs_doi_exact_title_author_year"]
    assert prepare(list(reversed(entries)), OPTIONS) == (studies, diagnostic)
    assert entries == before


def test_multiple_supplements_are_attached_without_adding_studies():
    parent = parent_entry()
    entries = [parent, supplement(parent, ".s001"), supplement(parent, ".s002")]
    studies, diagnostic = prepare(entries, OPTIONS)
    assert len(studies) == 1 and len(studies[0]["versions"]) == 3
    assert diagnostic["supplementary_materials_attached"] == 2


@pytest.mark.parametrize("difference", ["author", "year", "title"])
def test_unconfirmed_acs_pair_is_not_silently_merged_by_the_old_heuristic(difference):
    parent = parent_entry()
    child = supplement(parent)
    if difference == "author":
        child["document"]["authors"] = ["Another Lin (13253381)"]
    elif difference == "year":
        child["document"].update(publication_year=2021, publication_date="2021-06-01")
    else:
        child["document"]["title"] += " in a different experiment"
    assert supplement_relations([parent, child], 2025) == ({}, {})
    studies, diagnostic = prepare([parent, child], OPTIONS)
    assert len(studies) == 2
    assert diagnostic["supplementary_materials_attached"] == 0


def test_a_suffix_alone_does_not_invent_a_missing_parent():
    child = supplement(parent_entry())
    studies, diagnostic = prepare([child], OPTIONS)
    assert supplement_relations([child], 2025) == ({}, {})
    assert len(studies) == 1 and studies[0]["id"] == child["document_key"]
    assert diagnostic["supplementary_relations"] == []


def explicit_link(parent, child, reverse=False):
    claimant = parent if reverse else child
    kind = "is-supplemented-by" if reverse else "is-supplement-to"
    target = child if reverse else parent
    claimant["document"]["raw_metadata"] = {
        "relation": {kind: [{"id": target["document"]["doi"], "id-type": "doi"}]}}


@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_source_relation_supports_different_titles_and_dates(reverse):
    parent = parent_entry()
    child = supplement(parent, ".dataset")
    child["document"].update(title="Additional dataset from a separate repository", authors=["Another author"],
                              publication_year=2021, publication_date="2021-06-01")
    explicit_link(parent, child, reverse)
    studies, diagnostic = prepare([parent, child], OPTIONS)
    assert len(studies) == 1 and studies[0]["url"] == parent["document"]["url"]
    assert studies[0]["year"] == 2020
    assert diagnostic["supplementary_materials_attached"] == 1
    assert diagnostic["supplementary_relations"][0]["evidence"] == [
        "source_metadata:" + ("is-supplemented-by" if reverse else "is-supplement-to")]


@pytest.mark.parametrize("parent_state", ["absent", "unusable", "conflicting_year"])
def test_explicit_material_without_eligible_parent_is_excluded_with_a_reason(parent_state):
    parent = parent_entry()
    child = supplement(parent)
    explicit_link(parent, child)
    entries = [child]
    if parent_state != "absent":
        entries.append(parent)
        if parent_state == "unusable":
            parent["document"]["abstract"] = None
        else:
            conflicting = deepcopy(parent)
            conflicting["document"]["publication_year"] += 1
            entries.append(conflicting)
    studies, diagnostic = prepare(entries, OPTIONS)
    assert studies == []
    assert diagnostic["rejected"]["supplementary_material_without_eligible_parent"] == 1
    assert diagnostic["supplementary_relations"][0]["status"] == "parent_unavailable"


@pytest.mark.parametrize("withdrawn", ["parent", "supplement"])
def test_retraction_propagates_from_parent_to_material_only(withdrawn):
    parent = parent_entry()
    child = supplement(parent)
    (parent if withdrawn == "parent" else child)["document"]["raw_metadata"] = {"is_retracted": True}
    retained, _ = analysis_entries([parent, child], 2025)
    assert retained == ([] if withdrawn == "parent" else [parent])
    direct, _ = prepare([parent, child], OPTIONS)
    selected, _ = prepare(retained, OPTIONS)
    assert len(direct) == len(selected) == (0 if withdrawn == "parent" else 1)
    if selected:
        assert selected[0]["id"] == parent["document_key"]


def test_future_claimant_does_not_create_a_relationship_in_the_past():
    parent = parent_entry()
    child = supplement(parent, ".dataset")
    child["document"]["authors"] = ["Independent Author"]
    future_parent = deepcopy(parent)
    future_parent["document"].update(publication_year=2026, publication_date="2026-06-01")
    explicit_link(future_parent, child, reverse=True)
    assert supplement_relations([parent, child, future_parent], 2025) == ({}, {})


def test_future_material_revision_is_not_attached_to_an_earlier_result():
    parent = parent_entry()
    child = supplement(parent)
    future = deepcopy(child)
    future["revision_id"] = "future-material"
    future["document"].update(publication_year=2026, publication_date="2026-06-01")
    studies, _ = prepare([parent, child, future], OPTIONS)
    assert {v["revision_id"] for v in studies[0]["versions"]} == {
        parent["revision_id"], child["revision_id"]}


def test_disputed_parent_date_cannot_be_restored_by_its_materials():
    parent = parent_entry()
    doi = "10.1088/2515-7647/ae2e67"  # Existing, independently documented source review.
    parent["document_key"] = "doi:" + doi
    parent["document"].update(doi=doi, publication_year=2025, publication_date="2025-12-17")
    child = supplement(parent, ".dataset")
    explicit_link(parent, child)
    retained, diagnostic = analysis_entries([child, parent], 2025)
    assert retained == []
    assert diagnostic["excluded_disputed_parent_materials"] == [child["document_key"]]


def test_nested_materials_preserve_provenance_under_the_actual_publication():
    parent = parent_entry()
    child = supplement(parent, ".data")
    nested = supplement(child, ".analysis")
    explicit_link(parent, child)
    explicit_link(child, nested)
    studies, diagnostic = prepare([nested, child, parent], OPTIONS)
    assert len(studies) == 1 and len(studies[0]["versions"]) == 3
    assert studies[0]["abstract"] == parent["document"]["abstract"]
    assert diagnostic["supplementary_materials_attached"] == 2


def test_cyclic_metadata_never_turns_material_into_independent_evidence():
    parent = parent_entry()
    child = supplement(parent)
    explicit_link(parent, child)
    explicit_link(child, parent)
    studies, diagnostic = prepare([parent, child], OPTIONS)
    assert studies == []
    assert diagnostic["rejected"] == {"cyclic_supplement_relation": 2}


def test_conflicting_parent_claims_are_visible_and_do_not_choose_arbitrarily():
    parent = parent_entry()
    child = supplement(parent)
    explicit_link(parent, child)
    child["document"]["raw_metadata"]["relation"]["is-supplement-to"].append(
        {"id": "10.9999/different-parent", "id-type": "doi"})
    studies, diagnostic = prepare([parent, child], OPTIONS)
    assert len(studies) == 1 and studies[0]["id"] == parent["document_key"]
    assert diagnostic["rejected"] == {"conflicting_supplement_parents": 1}


def test_author_identifier_cannot_reintroduce_training_work_into_holdout():
    parent = parent_entry()
    future = supplement(parent, ".independent-version")
    # Different DOI; this uses the existing exact-title/author version rule.
    train, _ = prepare([parent], OPTIONS)
    later, _ = prepare([future], OPTIONS)
    retained, excluded = exclude_known_versions(train, later)
    assert retained == []
    assert excluded == [{"id": later[0]["id"], "reason": "train_long_title_first_author"}]


def test_long_authors_and_relation_chains_have_bounded_work_and_output():
    script = """
import json
from app.ml.study_relations import author_name_key, supplement_relations, supplement_descendants
assert author_name_key(' ' * 200000) == ''
entries = []
for i in range(6000):
    entries.append({'document': {'doi': f'10.9999/node-{i}', 'publication_year': 2020,
        'raw_metadata': {'relation': {'is-supplement-to': [
            {'id': f'10.9999/node-{i+1}', 'id-type': 'doi'}]}}}})
links, conflicts = supplement_relations(entries, 2025)
assert len(links) == 6000 and conflicts == {}
assert links['doi:10.9999/node-0']['parent_id'] == 'doi:10.9999/node-6000'
assert len(json.dumps(links)) < 1500000
assert len(supplement_descendants(links, {'doi:10.9999/node-6000'})) == 6000
print('ok')
"""
    completed = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True, timeout=5, check=False)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "ok\n"


@pytest.mark.parametrize("raw_type", [None, "article", "book", "dataset"])
def test_component_relation_requires_explicit_component_type(raw_type):
    parent = parent_entry()
    child = supplement(parent, ".independent")
    child["document"]["raw_metadata"] = {"type": raw_type, "relation": {
        "is-component-of": [{"id-type": "doi", "id": parent["document"]["doi"]}]}}
    assert supplement_relations([parent, child], 2025) == ({}, {})


def test_raw_typed_component_is_attached_without_a_title_heuristic():
    parent = parent_entry()
    child = supplement(parent, ".figure")
    child["document"].update(title="Supporting figure", authors=[])
    child["document"]["raw_metadata"] = {"type": "component", "relation": {
        "is-component-of": [{"id-type": "doi", "id": parent["document"]["doi"]}]}}
    studies, diagnostic = prepare([parent, child], OPTIONS)
    assert len(studies) == 1 and studies[0]["id"] == parent["document_key"]
    assert diagnostic["supplementary_relations"][0]["evidence"] == ["source_metadata:is-component-of"]


def reviewed_pair(doi):
    parent = parent_entry()
    parent_doi = doi.removesuffix(".s001")
    parent["document_key"] = "doi:" + parent_doi
    parent["document"].update(doi=parent_doi, url="https://doi.org/" + parent_doi,
                              publication_year=2025, publication_date="2025-06-01")
    child = supplement(parent)
    child["document"].update(title="Supporting component with different typography", authors=["Different spelling"])
    return parent, child


@pytest.mark.parametrize("doi", [
    "10.1021/acs.jpclett.5c02552.s001", "10.1021/acsami.5c06829.s001",
    "10.1021/acsphotonics.1c00526.s001",
])
def test_verified_catalog_links_exact_dois_and_preserves_missing_parent_diagnostic(doi):
    parent, child = reviewed_pair(doi)
    before = deepcopy([parent, child])
    studies, diagnostic = prepare([parent, child], OPTIONS)
    assert len(studies) == 1 and studies[0]["abstract"] == parent["document"]["abstract"]
    assert studies[0]["url"] == parent["document"]["url"]
    assert diagnostic["supplementary_relations"][0]["evidence"] == ["verified_source_metadata:is-component-of"]
    orphan, orphan_diagnostic = prepare([child], OPTIONS)
    assert orphan == []
    assert orphan_diagnostic["rejected"] == {"supplementary_material_without_eligible_parent": 1}
    assert [parent, child] == before


def test_review_catalog_cannot_override_conflicting_raw_source_relation():
    parent, child = reviewed_pair("10.1021/acsami.5c06829.s001")
    child["document"]["raw_metadata"] = {"type": "component", "relation": {
        "is-component-of": [{"id-type": "doi", "id": "10.9999/another-parent"}]}}
    studies, diagnostic = prepare([parent, child], OPTIONS)
    assert len(studies) == 1 and len(studies[0]["versions"]) == 1
    assert diagnostic["rejected"] == {"conflicting_supplement_parents": 1}


def test_component_registered_later_does_not_supply_a_past_relationship():
    parent, child = reviewed_pair("10.1021/acsami.5c06829.s001")
    for entry in (parent, child):
        entry["document"].update(publication_year=2020, publication_date="2020-06-01")
    assert supplement_relations([parent, child], 2024) == ({}, {})


@pytest.mark.parametrize("withdrawn", ["parent", "supplement"])
def test_reviewed_relation_obeys_one_way_retraction(withdrawn):
    parent, child = reviewed_pair("10.1021/acsami.5c06829.s001")
    (parent if withdrawn == "parent" else child)["document"]["raw_metadata"] = {"is_retracted": True}
    retained, _ = analysis_entries([child, parent], 2025)
    assert retained == ([] if withdrawn == "parent" else [parent])


def test_catalog_hash_is_visible_and_changes_result_fingerprint(tmp_path, monkeypatch):
    from app.ml import study_relations
    from app.ml.corpus import unpack_snapshot
    from app.ml.engine import analyze
    from tools.audit_ml_result import audit_result

    data = snapshot()
    for period, batch in zip(data["history"]["periods"], data["batches"], strict=True):
        period["job"].update(stored=0, scanned=0, total_available=0)
        batch.update(total=0, documents=[])
    corpus = unpack_snapshot(data)
    first = analyze(corpus, OPTIONS)
    assert first["source_relations_fingerprint"] == study_relations.source_relations_fingerprint()
    assert audit_result(corpus, first)["ok"]
    catalog = json.loads(study_relations.SOURCE_RELATIONS_PATH.read_text(encoding="utf-8"))
    catalog["review_note"] = "Fingerprint includes the exact reviewed data artifact."
    replacement = tmp_path / "reviews.json"
    replacement.write_text(json.dumps(catalog), encoding="utf-8")
    monkeypatch.setattr(study_relations, "SOURCE_RELATIONS_PATH", replacement)
    second = analyze(corpus, OPTIONS)
    assert first["fingerprint"] != second["fingerprint"]
    assert first["source_relations_fingerprint"] != second["source_relations_fingerprint"]
    assert "source_relations_fingerprint" in {error["code"] for error in audit_result(corpus, first)["errors"]}


def test_catalog_entries_have_matching_archived_primary_evidence():
    from app.ml.study_relations import SOURCE_RELATIONS_PATH

    reviews = json.loads(SOURCE_RELATIONS_PATH.read_text(encoding="utf-8"))["reviews"]
    root = Path(__file__).resolve().parents[1]
    assert len(reviews) == 3
    for doi, review in reviews.items():
        evidence = root / "tests/fixtures/ml/supplementary-relations" / (doi.split("/")[1] + ".json")
        assert json.loads(evidence.read_text(encoding="utf-8")) == review
        assert review["source_url"] == "https://api.crossref.org/works/" + doi
        assert review["metadata"]["DOI"] == doi
        assert review["metadata"]["type"] == "component"
        assert review["registered_from_year"] == review["metadata"]["created"]["date-parts"][0][0]
        assert len(review["source_metadata_sha256"]) == 64
