"""Version eligibility must preserve usable evidence and known retractions."""

from copy import deepcopy
from threading import Event
from concurrent.futures import CancelledError

import pytest

from app.ml.contracts import AnalysisOptions
from app.ml.corpus import unpack_snapshot
from app.ml.engine import analysis_entries, prepare
from app.ml.validation import exclude_known_versions
from tests.mvp_fixture import snapshot


OPTIONS = AnalysisOptions(topic="photonic neuromorphic computing")


def original_entry():
    return deepcopy(snapshot()["batches"][0]["documents"][0])


@pytest.mark.parametrize("marker", ["metadata", "title"])
@pytest.mark.parametrize("alias", ["same", "https", "http", "http_dx", "doi_prefix"])
def test_retraction_applies_to_all_versions_of_the_same_doi(marker, alias):
    original = original_entry()
    withdrawn = deepcopy(original)
    withdrawn["revision_id"] = "withdrawn"
    withdrawn["document_key"] = "openalex:another-source-id"
    doi = original["document"]["doi"]
    withdrawn["document"]["doi"] = {"same": doi, "https": "https://doi.org/" + doi.upper(),
                                   "http": "http://doi.org/" + doi.upper(), "http_dx": "http://dx.doi.org/" + doi,
                                   "doi_prefix": "doi:" + doi}[alias]
    if marker == "metadata":
        withdrawn["document"]["raw_metadata"] = {"is_retracted": True}
    else:
        withdrawn["document"]["title"] = "RETRACTED: " + withdrawn["document"]["title"]
    entries = [original, withdrawn]
    before = deepcopy(entries)
    retained, diagnostic = analysis_entries(entries, OPTIONS.end_year)
    assert retained == []
    assert diagnostic["excluded_retracted_occurrences"] == 2
    assert set(diagnostic["retracted_document_keys"]) == {e["document_key"] for e in entries}
    assert entries == before
    assert analysis_entries(list(reversed(entries)), OPTIONS.end_year) == (retained, diagnostic)


def test_same_source_identity_without_doi_also_cannot_restore_retracted_version():
    original = original_entry()
    original["document"]["doi"] = None
    withdrawn = deepcopy(original)
    withdrawn["revision_id"] = "withdrawn"
    withdrawn["document"]["raw_metadata"] = {"is_retracted": True}
    assert analysis_entries([original, withdrawn], 2025)[0] == []


@pytest.mark.parametrize("through_bridge", [False, True])
def test_later_doi_enrichment_preserves_retraction_of_the_original_source_identity(through_bridge):
    original = original_entry()
    original["document_key"] = "openalex:stable-source-key"
    original["document"]["doi"] = None
    enriched = deepcopy(original)
    enriched["revision_id"] = "doi-added"
    enriched["document"]["doi"] = "10.9999/versioned-study"
    withdrawn = deepcopy(enriched)
    withdrawn["revision_id"] = "withdrawn"
    withdrawn["document"]["raw_metadata"] = {"is_retracted": True}
    if through_bridge:
        withdrawn["document_key"] = "doi:10.9999/versioned-study"
    entries = [original, enriched, withdrawn]
    retained, diagnostic = analysis_entries(entries, 2025)
    assert retained == []
    assert diagnostic["excluded_retracted_occurrences"] == 3
    assert prepare(entries, OPTIONS)[0] == []
    assert analysis_entries(list(reversed(entries)), 2025) == (retained, diagnostic)


def test_retraction_does_not_remove_a_different_doi_with_the_same_title():
    original = original_entry()
    withdrawn = deepcopy(original)
    withdrawn["revision_id"] = "unrelated"
    withdrawn["document"]["doi"] = "10.9999/unrelated-study"
    withdrawn["document_key"] = "doi:10.9999/unrelated-study"
    withdrawn["document"]["raw_metadata"] = {"is_retracted": True}
    assert analysis_entries([original, withdrawn], 2025)[0] == [original]


def test_future_retraction_version_does_not_leak_into_an_earlier_window():
    original = original_entry()
    withdrawn = deepcopy(original)
    withdrawn["revision_id"] = "future"
    withdrawn["document"].update(publication_year=2026, publication_date="2026-01-01")
    withdrawn["document"]["raw_metadata"] = {"is_retracted": True}
    retained, diagnostic = analysis_entries([original, withdrawn], 2025)
    assert retained == [original]
    assert diagnostic["excluded_after_end_year"] == 1
    assert diagnostic["excluded_retracted_occurrences"] == 0


def test_direct_preparation_used_by_retrospective_cannot_restore_a_retracted_doi():
    original = original_entry()
    withdrawn = deepcopy(original)
    withdrawn["revision_id"] = "withdrawn"
    withdrawn["document"]["raw_metadata"] = {"is_retracted": True}
    studies, diagnostic = prepare([original, withdrawn], OPTIONS)
    assert studies == []
    assert diagnostic["rejected"]["retracted_study"] == 1


@pytest.mark.parametrize("invalid_kind", ["oversized", "unsupported_type", "service_title", "invalid_url", "off_direction"])
def test_unusable_longer_version_does_not_mask_a_usable_version(invalid_kind):
    original = original_entry()
    invalid = deepcopy(original)
    invalid["revision_id"] = "longer-version"
    invalid["document"]["abstract"] *= 2
    if invalid_kind == "oversized":
        invalid["document"]["abstract"] = " ".join(["photonic neural computation"] * 1001)
    elif invalid_kind == "unsupported_type":
        invalid["document"]["document_type"] = "dataset"
    elif invalid_kind == "service_title":
        invalid["document"]["title"] = "Contents"
    elif invalid_kind == "invalid_url":
        invalid["document"]["url"] = "javascript:alert(1)"
    else:
        invalid["document"]["title"] = "Submarine channels and ocean measurements"
        invalid["document"]["abstract"] = "Measurements of ocean temperature and salinity describe marine currents and water movement in the deep ocean. " * 15
    entries = [original, invalid]
    before = deepcopy(entries)
    studies, diagnostic = prepare(entries, OPTIONS)
    assert len(studies) == 1
    assert studies[0]["abstract"] == original["document"]["abstract"]
    assert studies[0]["title"] == original["document"]["title"]
    assert studies[0]["url"] == original["document"]["url"]
    assert {v["revision_id"] for v in studies[0]["versions"]} == {e["revision_id"] for e in entries}
    assert diagnostic["rejected"] == {}
    assert entries == before
    assert prepare(list(reversed(entries)), OPTIONS) == (studies, diagnostic)


def test_two_usable_versions_still_prefer_the_longer_one():
    original = original_entry()
    longer = deepcopy(original)
    longer["revision_id"] = "longer"
    longer["document"]["abstract"] += " Additional experiments confirmed reproducibility of the measured photonic neural device."
    studies, _ = prepare([original, longer], OPTIONS)
    assert studies[0]["abstract"] == longer["document"]["abstract"]


def test_all_unusable_versions_keep_a_specific_rejection_reason():
    original = original_entry()
    original["document"]["abstract"] = None
    oversized = deepcopy(original)
    oversized["revision_id"] = "oversized"
    oversized["document"]["abstract"] = " ".join(["photonic neural computation"] * 1001)
    studies, diagnostic = prepare([original, oversized], OPTIONS)
    assert studies == []
    assert diagnostic["rejected"] == {"oversized_abstract_requires_review": 1}


def test_version_fallback_does_not_bypass_conflicting_years():
    original = original_entry()
    other = deepcopy(original)
    other["document"].update(publication_year=2021, publication_date="2021-06-01")
    other["document"]["abstract"] = " ".join(["photonic neural computation"] * 1001)
    studies, diagnostic = prepare([original, other], OPTIONS)
    assert studies == []
    assert diagnostic["rejected"] == {"conflicting_year": 1}


def test_version_preparation_remains_cancellable():
    event = Event()
    event.set()
    with pytest.raises(CancelledError):
        prepare(unpack_snapshot(snapshot())["entries"], OPTIONS, event)


@pytest.mark.parametrize("prefix", ["http://doi.org/", "https://doi.org/", "doi:"])
def test_known_short_abstract_train_alias_cannot_reappear_as_new_holdout_study(prefix):
    early = original_entry()
    early["document_key"] = "openalex:early"
    early["document"].update(doi=prefix + "10.9999/versioned-study", authors=[],
                             publication_year=2019, publication_date="2019-06-01", abstract=None)
    late = deepcopy(early)
    late["document_key"] = "openalex:late"
    late["revision_id"] = "late"
    late["document"].update(doi="10.9999/versioned-study", publication_year=2022, publication_date="2022-06-01",
                            abstract=original_entry()["document"]["abstract"])
    train, _ = prepare([early], OPTIONS.model_copy(update={"start_year": 2014, "end_year": 2019}))
    future, _ = prepare([late], OPTIONS)
    assert train == [] and len(future) == 1
    retained, excluded = exclude_known_versions(train, future, raw_train_entries=[early])
    assert retained == []
    assert excluded == [{"id": future[0]["id"], "reason": "train_identity"}]
