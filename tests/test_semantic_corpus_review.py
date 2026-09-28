"""Uncalibrated corpus expansion also affects groups with only lexical members."""

from copy import deepcopy

import pytest

from app.ml.corpus import unpack_snapshot
from app.ml.engine import analyze, metrics
from app.ml.selection import selection_for
from app.ml.semantic_contracts import validate_semantic_result
from tests.test_semantic_integration import Encoder, generic_snapshot, options
from tools.audit_ml_result import BUCKETS, audit_result


@pytest.fixture(autouse=True)
def local_encoder(monkeypatch):
    # Imported helper fixtures are not implicitly active in this test module.
    monkeypatch.setattr("app.ml.local_encoder.LocalEncoder", Encoder)


def test_new_denominators_can_create_growth_without_new_group_documents():
    years = list(range(2020, 2026))
    lexical_members = [{"year": year} for year in years for _ in range(10)]
    baseline = metrics(lexical_members, dict.fromkeys(years, 1000), years, .8)
    # All additions belong to other topics, so this group's numerator is fixed.
    expanded = metrics(lexical_members, dict(zip(years, (2000, 2000, 2000, 1600, 1300, 1000), strict=True)), years, .8)
    assert [row["documents"] for row in expanded["years"]] == [10] * 6
    assert baseline["growth_ratio"] == 1 and not baseline["growth_pattern"]
    assert expanded["growth_ratio"] == pytest.approx(6001 / 3901)
    assert expanded["growth_pattern"]
    group = {"metrics": expanded,
             "card": {field: {"text": "Supported source quote"} for field in ("problem", "advantage", "example")}}
    assert selection_for(group, True)["bucket"] == "candidates"
    group["semantic_relevance"] = {"semantic_only_documents": 0,
                                   "corpus_requires_review": True, "requires_review": True}
    selection = selection_for(group, True)
    assert selection["bucket"] == "preliminary_signals"
    assert selection["reasons"] == ["semantic_scope_requires_review"]


def test_generic_lexical_only_corpus_keeps_original_buckets():
    data = generic_snapshot()
    for batch in data["batches"]:
        for entry in batch["documents"]:
            entry["document"]["abstract"] += " Thermal storage systems retain heat for later use."
    corpus = unpack_snapshot(data)
    lexical = analyze(corpus, options(corpus, "lexical"))
    semantic = analyze(corpus, options(corpus))
    assert semantic["semantic_relevance"]["retained_studies"] > 0
    assert semantic["semantic_relevance"]["semantic_only_studies"] == 0
    assert not semantic["semantic_relevance"]["guarded_direction"]
    for bucket in BUCKETS:
        assert [{key: value for key, value in group.items() if key != "semantic_relevance"}
                for group in semantic[bucket]] == lexical[bucket]
        assert all(not group["semantic_relevance"]["corpus_requires_review"] for group in semantic[bucket])


def test_semantic_additions_before_window_require_review_even_with_unchanged_denominators():
    data = generic_snapshot()
    earlier_batch, earlier_period = deepcopy(data["batches"][0]), deepcopy(data["history"]["periods"][0])
    # Add one technology only, leaving other lexical groups without semantic members.
    earlier_batch["documents"] = [entry for entry in earlier_batch["documents"]
                                  if "-0-" in entry["document"]["source_id"]]
    for batch in data["batches"]:
        for entry in batch["documents"]:
            entry["document"]["abstract"] += " Thermal storage systems retain heat for later use."
    for entry in earlier_batch["documents"]:
        document = entry["document"]
        for field in ("source_id", "doi", "url", "title"):
            document[field] = document[field].replace("2020", "2019")
        document.update(publication_year=2019, publication_date="2019-06-01")
        document["authors"] = [author.replace("2020", "2019") for author in document["authors"]]
        entry["document_key"] = "doi:" + document["doi"]
        entry["revision_id"] = document["source_id"]
    size = len(earlier_batch["documents"])
    earlier_batch.update(job_id="2019", total=size)
    earlier_period.update(id="2019", from_date="2019-01-01", until_date="2019-12-31")
    earlier_period["job"].update(id="2019", stored=size, scanned=size, total_available=size)
    earlier_period["job"]["request"].update(from_date="2019-01-01", until_date="2019-12-31")
    data["batches"].insert(0, earlier_batch)
    data["history"]["periods"].insert(0, earlier_period)
    corpus = unpack_snapshot(data)
    original = deepcopy(corpus)
    lexical = analyze(corpus, options(corpus, "lexical"))
    semantic = analyze(corpus, options(corpus))
    added_ids = {"doi:" + entry["document"]["doi"] for entry in earlier_batch["documents"]}
    assert {decision["study_id"] for decision in semantic["semantic_relevance"]["study_decisions"]
            if decision["semantic_only"]} == added_ids
    assert all(entry["document"]["publication_year"] < options(corpus).start_year
               for entry in earlier_batch["documents"])
    assert semantic["direction_counts"] == lexical["direction_counts"]
    groups = [group for bucket in BUCKETS for group in semantic[bucket]]
    assert groups and any(group["semantic_relevance"]["semantic_only_documents"] == 0 for group in groups)
    assert semantic["candidates"] == semantic["established"] == []
    assert all(group["semantic_relevance"]["corpus_requires_review"] for group in groups)
    assert all(group["selection"]["bucket"] == "preliminary_signals" for group in groups)
    validate_semantic_result(semantic)
    report = audit_result(corpus, semantic)
    assert report["ok"], report["errors"]
    assert corpus == original
