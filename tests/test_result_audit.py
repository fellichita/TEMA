"""Independent saved-result checks: mutations must be detected without an NMF replay."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from app.ml.contracts import AnalysisOptions
from app.ml.corpus import unpack_snapshot
from app.ml.engine import analyze
from tests.mvp_fixture import snapshot
from tools.audit_ml_result import BUCKETS, _passage_is_original, audit_result, markdown_report, write_report_pair


@pytest.fixture(scope="module")
def fixture_result():
    corpus = unpack_snapshot(snapshot())
    result = analyze(corpus, AnalysisOptions(topic=corpus["topic"]))
    assert any(result[bucket] for bucket in BUCKETS)
    return corpus, result


def first_card(result):
    return next(card for bucket in BUCKETS for card in result[bucket])


def test_valid_result_all_sections_replay_without_refitting(fixture_result, monkeypatch):
    corpus, result = fixture_result
    def forbidden(*args, **kwargs):
        pytest.fail("The result audit must not fit or transform NMF")
    monkeypatch.setattr("app.ml.model.fit_topics", forbidden)
    import sklearn.decomposition
    monkeypatch.setattr(sklearn.decomposition.NMF, "fit_transform", forbidden)
    original = deepcopy(result)
    report = audit_result(corpus, result)
    assert report["ok"], report["errors"]
    assert report["semantic_acceptance"] == "not_performed"
    assert len(report["rows"]) == sum(len(result[b]) for b in BUCKETS)
    assert result == original
    assert report["checks"]["bucket_list"] == 4
    assert "не содержательная приёмка" in markdown_report(report)


@pytest.mark.parametrize("mutation,code", [
    (lambda r: first_card(r)["metrics"]["years"][0].update(documents=999), "metrics_replay"),
    (lambda r: first_card(r)["metrics"].update(growth_ratio=-10), "metrics_replay"),
    (lambda r: first_card(r)["metrics"]["score_weights"].update(growth=99), "metrics_replay"),
    (lambda r: first_card(r)["metrics"].update(first_observed_year_in_corpus=1900), "metrics_replay"),
    (lambda r: first_card(r)["metrics"].update(first_observed_year_in_window=1900), "metrics_replay"),
    (lambda r: first_card(r).update(study_count=999), "study_count"),
    (lambda r: first_card(r)["sources"][0].update(url="https://example.org/foreign"), "source_metadata_versions"),
    (lambda r: first_card(r)["sources"][0]["versions"][0].update(revision_id="unknown"), "source_metadata_versions"),
    (lambda r: first_card(r)["card"]["example"].update(text="A fabricated source quote."), "quote_exact_source"),
    (lambda r: first_card(r)["card"]["example"].update(url="https://example.org/foreign"), "quote_url_title"),
    (lambda r: first_card(r)["document_evidence"][0].update(evidence_level="nominal"), "execution_replay"),
    (lambda r: first_card(r)["document_evidence"].pop(), "all_member_execution"),
    (lambda r: first_card(r)["direction_guard"].update(axis_check="off_direction"), "axis_replay"),
    (lambda r: first_card(r)["selection"]["reasons"].append("invented"), "selection_reasons"),
    (lambda r: first_card(r).update(stage="confirmed"), "no_unverified_confirmed_stage"),
    (lambda r: r.update(status="confirmed"), "result_status"),
    (lambda r: r.update(fingerprint="0" * 64), "result_fingerprint"),
    (lambda r: r["selection_summary"].update(shortfall=-1), "top_shortfall"),
    (lambda r: r["model"].update(random_state=43), "frozen_model_parameters"),
    (lambda r: r["model"].update(type="BERTopic"), "model_type"),
    (lambda r: r["model"]["excluded_groups"].update(off_direction=999), "excluded_model_count"),
    (lambda r: r["model"].update(topics=99), "model_topics_bound"),
    (lambda r: r["model"].update(iterations=999), "model_iterations_bound"),
    (lambda r: r["direction_counts"][0].update(documents=999), "direction_counts"),
    (lambda r: r.update(growth_data_comparable=not r["growth_data_comparable"]), "comparability_replay"),
])
def test_tampered_result_is_rejected(fixture_result, mutation, code):
    corpus, original = fixture_result
    result = deepcopy(original)
    mutation(result)
    report = audit_result(corpus, result)
    assert not report["ok"]
    assert code in {error["code"] for error in report["errors"]}, report["errors"]


@pytest.mark.parametrize("bucket", BUCKETS)
def test_duplicate_ids_are_detected_in_every_bucket(fixture_result, bucket):
    corpus, original = fixture_result
    result = deepcopy(original)
    result[bucket].append(deepcopy(first_card(result)))
    report = audit_result(corpus, result)
    assert not report["ok"]
    assert "unique_group_id" in {error["code"] for error in report["errors"]}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_anywhere_are_rejected_before_metric_replay(fixture_result, value):
    corpus, original = fixture_result
    result = deepcopy(original)
    result["diagnostics"] = {"nested": [value]}
    report = audit_result(corpus, result)
    assert not report["ok"]
    assert report["errors"] == [{"code": "finite_numbers", "location": "result.diagnostics.nested[0]"}]


def test_neighboring_evidence_requires_actual_adjacency_and_exact_original_text():
    study = {"title": "An optical device", "abstract": "We build a photonic chip. It computes neural inference. The output is recorded."}
    assert _passage_is_original("We build a photonic chip.\nIt computes neural inference.", "neighboring_excerpts", study)
    assert not _passage_is_original("We build a photonic chip.\nThe output is recorded.", "neighboring_excerpts", study)
    assert not _passage_is_original("It computes neural inference.\nWe build a photonic chip.", "neighboring_excerpts", study)
    assert not _passage_is_original("We build a photonic chip. It computes neural inference.", "neighboring_excerpts", study)


def test_empty_analysis_is_a_valid_structural_result():
    corpus = unpack_snapshot(snapshot())
    corpus["entries"] = []
    result = analyze(corpus, AnalysisOptions(topic=corpus["topic"]))
    report = audit_result(corpus, result)
    assert report["ok"], report["errors"]
    assert report["rows"] == []


@pytest.mark.parametrize("status", ["ranked", "no_groups"])
def test_empty_unfitted_result_cannot_claim_a_completed_model(status):
    corpus = unpack_snapshot(snapshot())
    corpus["entries"] = []
    result = analyze(corpus, AnalysisOptions(topic=corpus["topic"]))
    result["status"] = status
    report = audit_result(corpus, result)
    assert "status_matches_result" in {e["code"] for e in report["errors"]}


def test_hidden_below_top_cards_cannot_exist_when_the_visible_top_is_not_full():
    corpus = unpack_snapshot(snapshot())
    corpus["entries"] = []
    result = analyze(corpus, AnalysisOptions(topic=corpus["topic"]))
    result["selection_summary"].update(below_top_ids=["invented-hidden-id"], eligible_before_limit=1)
    report = audit_result(corpus, result)
    assert "eligible_fill_before_hiding" in {e["code"] for e in report["errors"]}
    assert "no_groups_without_model" in {e["code"] for e in report["errors"]}


def test_retracted_input_preserves_incomparability_even_when_no_other_quality_issue_remains():
    corpus = unpack_snapshot(snapshot())
    corpus["entries"] = [corpus["entries"][0]]
    corpus["entries"][0]["document"]["raw_metadata"]["is_retracted"] = True
    result = analyze(corpus, AnalysisOptions(topic=corpus["topic"]))
    assert not result["growth_data_comparable"]
    assert result["temporal_selection"]["excluded_retracted_occurrences"] == 1
    report = audit_result(corpus, result)
    assert report["ok"], report["errors"]


def test_reviewed_date_disagreement_preserves_incomparability_and_review_provenance():
    corpus = unpack_snapshot(snapshot())
    corpus["entries"] = [corpus["entries"][-1]]
    entry = corpus["entries"][0]
    entry["document"]["doi"] = "10.1088/2515-7647/ae2e67"
    entry["document_key"] = "doi:10.1088/2515-7647/ae2e67"
    assert entry["document"]["publication_year"] == 2025
    result = analyze(corpus, AnalysisOptions(topic=corpus["topic"]))
    assert not result["growth_data_comparable"]
    assert result["temporal_selection"]["excluded_disputed_date_occurrences"] == 1
    report = audit_result(corpus, result)
    assert report["ok"], report["errors"]


def test_quote_selection_replay_receives_the_actual_cluster_terms(fixture_result, monkeypatch):
    from app.ml.evidence import supported_card
    calls = []
    def replay(studies, proposed, topic_terms=None):
        assert topic_terms, "The audit must replay the same topic constraint as the engine"
        calls.append(topic_terms)
        return supported_card(studies, proposed, topic_terms)
    monkeypatch.setattr("tools.audit_ml_result.supported_card", replay)
    report = audit_result(*fixture_result)
    assert report["ok"], report["errors"]
    assert calls == [c["keywords"] for bucket in BUCKETS for c in fixture_result[1][bucket]]


def test_invalid_card_structure_becomes_an_audit_error(fixture_result):
    corpus, original = fixture_result
    result = deepcopy(original)
    result["candidates"] = ["not a card"]
    report = audit_result(corpus, result)
    assert not report["ok"]
    assert "valid_card_structure" in {error["code"] for error in report["errors"]}


def test_report_pair_is_valid_and_does_not_overwrite_existing_files(fixture_result, tmp_path):
    report = audit_result(*fixture_result)
    base = tmp_path / "audit"
    paths = write_report_pair(report, base)
    assert json.loads(paths[0].read_text(encoding="utf-8")) == report
    before = [p.read_bytes() for p in paths]
    with pytest.raises(FileExistsError):
        write_report_pair(report, base)
    assert [p.read_bytes() for p in paths] == before


def test_pair_publish_failure_removes_only_our_first_new_file(fixture_result, tmp_path, monkeypatch):
    report = audit_result(*fixture_result)
    base = tmp_path / "audit"
    import os
    real_link = os.link
    count = 0
    def fail_second(source, target):
        nonlocal count
        count += 1
        if count == 2:
            Path(target).write_text("another writer's report")
            raise FileExistsError()
        real_link(source, target)
    monkeypatch.setattr("tools.audit_ml_result.os.link", fail_second)
    with pytest.raises(FileExistsError):
        write_report_pair(report, base)
    assert not Path(str(base) + ".json").exists()
    assert Path(str(base) + ".md").read_text(encoding="utf-8") == "another writer's report"
    assert len(list(tmp_path.iterdir())) == 1
