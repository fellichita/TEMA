"""Frozen split integrity, human-label denominators and explicit execution."""

import argparse
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.evaluate_pilot import (
    EvaluationError, HumanLabel, ReviewTask, cases_for, digest, execute_split,
    load_labels, load_suite, quality_metrics, resolve_labels,
)


@pytest.fixture
def suite():
    return load_suite(Path(__file__).parents[1] / "tests/evaluation/pilot-v1.json")


def manifest_for(suite):
    cases = cases_for(suite, "development")
    tasks = []
    for case in cases:
        for rank in range(1, 21):
            tasks.append(ReviewTask(task_id=digest({"case": case["case_id"], "rank": rank}), **case, kind="pair",
                rank=rank, secondary_required=rank <= 5, payload={"study_id": "study-" + str(rank)}))
    return {"suite": suite, "suite_hash": digest(suite), "split": "development", "cases": cases,
        "outcomes": [{**case, "state": "succeeded", "numeric_provenance_verified": True,
                      "truncated_confirmed": 0} for case in cases],
        "tasks": [task.model_dump(mode="json") for task in tasks]}


def labels_for(manifest):
    labels = []
    for task in manifest["tasks"]:
        labels.append(HumanLabel(task_id=task["task_id"], reviewer_id="First reviewer", label="relevant"))
        if task["secondary_required"]:
            labels.append(HumanLabel(task_id=task["task_id"], reviewer_id="Second reviewer", phase="secondary", label="relevant"))
    return labels


def test_frozen_fixture_has_disjoint_splits_and_exact600_planned_pairs(suite):
    assert sum(len(suite[split]) * suite["pairs_per_direction"][split] for split in ("development", "validation", "blind")) == 600
    assert len(cases_for(suite, "blind")) == 12


def test_overlap_and_inconsistent_sample_denominators_are_rejected(suite, tmp_path):
    from scripts.evaluate_pilot import write_json

    changed = deepcopy(suite)
    changed["blind"][0] = "  " + suite["development"][0].upper() + "  "
    write_json(tmp_path / "overlap.json", changed)
    with pytest.raises(EvaluationError, match="disjoint"):
        load_suite(tmp_path / "overlap.json")
    changed = deepcopy(suite)
    changed["human_labeling"]["total_pairs"] = 999
    write_json(tmp_path / "count.json", changed)
    with pytest.raises(EvaluationError, match="counts"):
        load_suite(tmp_path / "count.json")


def test_missing_human_labels_are_unknown_not_correct(suite):
    manifest = manifest_for(suite)
    result = quality_metrics(manifest, [])
    assert result["precision_at_20_macro"] is None
    assert result["precision_macro_lower_bound"] == 0 and result["precision_macro_upper_bound"] == 1
    assert result["gates"]["precision_at_20_macro"] is None
    assert result["gates"]["all_human_labels_resolved"] is False


def test_real_human_label_denominator_includes_all_planned_directions(suite):
    manifest = manifest_for(suite)
    labels = labels_for(manifest)
    measured = quality_metrics(manifest, labels)
    assert measured["precision_at_20_macro"] == 1 and measured["successful_cases"] == 6
    manifest["outcomes"][0]["state"] = "failed"
    partial = quality_metrics(manifest, labels)
    assert partial["precision_at_20_macro"] is None
    assert partial["precision_macro_lower_bound"] == pytest.approx(5 / 6)
    assert partial["precision_macro_upper_bound"] == 1 and partial["successful_cases"] == 5
    assert partial["gates"]["all_planned_cases_finished"] is False


def test_omitted_case_cannot_disappear_from_denominator(suite):
    manifest = manifest_for(suite)
    manifest["cases"].pop()
    with pytest.raises(EvaluationError, match="denominator"):
        quality_metrics(manifest, [])
    manifest = manifest_for(suite)
    manifest["outcomes"].pop()
    result = quality_metrics(manifest, labels_for(manifest))
    assert result["planned_cases"] == 6 and result["successful_cases"] == 5


def test_uncertain_and_disagreeing_reviews_need_adjudication(suite):
    task = ReviewTask.model_validate(manifest_for(suite)["tasks"][0])
    primary = HumanLabel(task_id=task.task_id, reviewer_id="Reviewer A", label="relevant")
    secondary = HumanLabel(task_id=task.task_id, reviewer_id="Reviewer B", phase="secondary", label="irrelevant")
    resolved, stats = resolve_labels([task], [primary, secondary])
    assert resolved[task.task_id] is None and stats["disagreements"] == 1 and stats["adjudicated"] == 0
    adjudicated = HumanLabel(task_id=task.task_id, reviewer_id="Reviewer C", phase="adjudication", label="uncertain")
    resolved, stats = resolve_labels([task], [primary, secondary, adjudicated])
    assert resolved[task.task_id] is None and stats["adjudicated"] == 1


def test_fake_second_reviewer_and_duplicate_or_wrong_kind_labels_are_rejected(suite):
    task = ReviewTask.model_validate(manifest_for(suite)["tasks"][0])
    primary = HumanLabel(task_id=task.task_id, reviewer_id="Same person", label="relevant")
    secondary = HumanLabel(task_id=task.task_id, reviewer_id="Same person", phase="secondary", label="relevant")
    with pytest.raises(EvaluationError, match="different person"):
        resolve_labels([task], [primary, secondary])
    with pytest.raises(EvaluationError, match="Multiple labels"):
        resolve_labels([task], [primary, primary])
    with pytest.raises(EvaluationError, match="wrong label"):
        resolve_labels([task], [HumanLabel(task_id=task.task_id, reviewer_id="Reviewer", label="fabricated")])
    with pytest.raises(ValidationError):
        HumanLabel(task_id=task.task_id, reviewer_id="Reviewer", label="probably good")


def test_same_study_cannot_fill_multiple_top20_positions(suite):
    manifest = manifest_for(suite)
    manifest["tasks"][1]["payload"]["study_id"] = manifest["tasks"][0]["payload"]["study_id"]
    with pytest.raises(EvaluationError, match="multiple retrieval"):
        quality_metrics(manifest, [])


def test_blind_run_requires_explicit_flags_and_preselection_without_starting_app(suite, monkeypatch, tmp_path):
    monkeypatch.setattr("app.pilot.service.PilotService", lambda *args, **kwargs: pytest.fail("No app should start"))
    args = argparse.Namespace(execute=False, split="development")
    with pytest.raises(EvaluationError, match="--execute"):
        execute_split(args, suite)
    args = argparse.Namespace(execute=True, split="blind", allow_blind=False, preselect_confirmed_case=[])
    with pytest.raises(EvaluationError, match="preselect"):
        execute_split(args, suite)


def test_blank_csv_rows_are_missing_labels_and_unknown_payload_fields_rejected(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("task_id,reviewer_id,phase,label\nunknown,,primary,\n")
    assert load_labels(path) == []
    from scripts.evaluate_pilot import write_json
    write_json(tmp_path / "wrong.json", [{"task_id": "task", "reviewer_id": "Reviewer", "label": "relevant", "approved_by_ai": True}])
    with pytest.raises(ValidationError):
        load_labels(tmp_path / "wrong.json")


def test_explicit_no_credential_evaluation_records_not_run_without_fabricated_results(suite, tmp_path, monkeypatch):
    from app.runtime.credentials import CredentialStore
    from scripts.evaluate_pilot import read_json

    store = CredentialStore()
    monkeypatch.setattr(store, "get", lambda _: None)
    monkeypatch.setattr("app.runtime.session.credentials", lambda: store)
    monkeypatch.setattr("app.pilot.service.PilotService.start", lambda *_: pytest.fail("Missing AI must never synthesize a Russian plan"))
    args = argparse.Namespace(execute=True, split="development", allow_blind=False, preselect_confirmed_case=[],
        total_cost_micro=100000, timeout_seconds=30, data_dir=tmp_path / "data", model_dir=None, output=tmp_path / "evaluation")
    output = execute_split(args, suite)
    manifest = read_json(output / "run-manifest.json")
    assert manifest["status"] == "not_evaluated" and manifest["tasks"] == []
    assert len(manifest["outcomes"]) == 6 and all(row["state"] == "not_run" for row in manifest["outcomes"])
    assert manifest["accounted_cost_micro"] == 0
    assert not list(output.rglob("raw-result.json"))


def test_budget_reader_retains_unknown_holds_instead_of_treating_them_as_free():
    from scripts.evaluate_pilot import _budget_for_run

    class Runtime:
        def budget_status(self, **kwargs):
            return {"unknown_total": 1, "reconciliation_required": 0, "restore_pending": False,
                "scopes": [{"scope_id": "run/abc", "used": {"cost_micro": 123456, "calls": 1}}], "next_scope_after": None}

    assert _budget_for_run(Runtime(), "abc") == {"cost_micro": 123456, "calls": 1, "requires_reconciliation": True}


def candidate_manifest(suite):
    manifest = manifest_for(suite)
    for outcome in manifest["outcomes"]:
        outcome["cards_count"] = 3
        outcome["confirmed_count"] = 0
    for case in manifest["cases"]:
        for rank in range(1, 4):
            task = ReviewTask(task_id=digest({"candidate": rank, "case": case["case_id"]}), **case,
                kind="candidate", rank=rank, secondary_required=False,
                payload={"candidate_id": f"candidate-{rank}"})
            manifest["tasks"].append(task.model_dump(mode="json"))
    return manifest


def test_candidate_lifecycle_precision_is_not_document_relevance(suite):
    manifest = candidate_manifest(suite)
    pair_manifest = manifest_for(suite)
    labels = labels_for(pair_manifest)
    for task in manifest["tasks"]:
        if task["kind"] == "candidate":
            labels.append(HumanLabel(task_id=task["task_id"], reviewer_id="Lifecycle expert",
                label={1: "early_weak_signal", 2: "emerging", 3: "mainstream"}[task["rank"]]))
    result = quality_metrics(manifest, labels)
    assert result["document_relevance_precision_at_20_macro"] == 1
    scientific = result["candidate_metrics"]
    assert scientific["weak_signal_precision_macro"] == pytest.approx(1 / 3)
    assert scientific["emerging_precision_macro"] == pytest.approx(2 / 3)
    assert scientific["emerging_yield_macro"] == pytest.approx(2 / 15)
    assert scientific["return_yield_macro"] == pytest.approx(3 / 15)
    assert "preselected_confirmed_areas" not in result["gates"]
    assert result["gates"]["candidate_lifecycle_evaluated"] is True
    assert result["gates"]["emerging_precision_at_15"] is False
    assert scientific["recall"] is None


def test_empty_or_unreviewed_candidates_never_prove_weak_signal_quality(suite):
    manifest = manifest_for(suite)
    empty = quality_metrics(manifest, labels_for(manifest))["candidate_metrics"]
    assert empty["weak_signal_precision_macro"] is None
    assert empty["emerging_precision_macro"] is None
    assert empty["return_yield_macro"] == 0
    assert empty["all_returned_candidates_labeled"] is False
    pending = quality_metrics(candidate_manifest(suite), labels_for(manifest))["candidate_metrics"]
    assert pending["weak_signal_precision_macro"] is None
    assert pending["all_returned_candidates_labeled"] is False


def test_candidate_sample_cannot_hide_omitted_or_duplicate_cards(suite):
    manifest = candidate_manifest(suite)
    manifest["tasks"].pop()
    with pytest.raises(EvaluationError, match="All returned"):
        quality_metrics(manifest, [])
    manifest = candidate_manifest(suite)
    candidates = [task for task in manifest["tasks"] if task["kind"] == "candidate"]
    candidates[1]["payload"]["candidate_id"] = candidates[0]["payload"]["candidate_id"]
    with pytest.raises(EvaluationError, match="unique IDs"):
        quality_metrics(manifest, [])


def test_assisted_and_automatic_evaluation_are_distinct(suite):
    manifest = candidate_manifest(suite)
    manifest["evaluation_mode"] = "assisted"
    result = quality_metrics(manifest, [])
    assert result["evaluation_mode"] == "assisted"
    assert result["gates"]["assisted_reviews_traceable"] is False
    manifest["evaluation_mode"] = "automatic"
    assert "assisted_reviews_traceable" not in quality_metrics(manifest, [])["gates"]


def test_v2_protocol_is_candidate_focused_and_does_not_claim_published_cases_are_blind(tmp_path, monkeypatch):
    suite = load_suite(Path(__file__).parents[1] / "tests/evaluation/pilot-v2.json")
    assert suite["protocol_version"] == 2
    assert suite["blind_status"] == "published_proposals_unsealed"
    assert "квантовое зондирование живых систем" in suite["development"]
    assert "early_weak_signal" in suite["human_labeling"]["candidate_labels"]
    assert suite["gates"]["emerging_precision_at_15"] == 0.80
    monkeypatch.setattr("app.pilot.service.PilotService", lambda *args, **kwargs: pytest.fail("Unsealed evaluation cannot start"))
    args = argparse.Namespace(execute=True, split="blind", allow_blind=True, preselect_confirmed_case=[])
    with pytest.raises(EvaluationError, match="not a sealed"):
        execute_split(args, suite)


def sampling_payload(top_ids):
    cards = [{"candidate": {"candidate_id": f"candidate-{index}", "label": f"Mechanism {index}",
        "definition": "A concrete mechanism under independent review", "synonyms": [f"mechanism {index}"]},
        "claims": [], "evidence": []} for index in range(20)]
    return {"result": {"cards": cards, "top_trend_ids": top_ids}}


def test_lifecycle_scope_specificity_and_duplicate_tasks_follow_actual_top_order(suite):
    from scripts.evaluate_pilot import make_tasks

    case = cases_for(suite, "development")[0]
    shown = ["candidate-19", "candidate-2", "candidate-17"]
    tasks = make_tasks(case, sampling_payload(shown), {"studies": [], "relevance": []}, None, None, 20)
    lifecycle = [task for task in tasks if task.kind == "candidate"]
    assert [(task.rank, task.payload["candidate_id"]) for task in lifecycle] == list(enumerate(shown, 1))
    for kind in ("scope", "specificity"):
        assert [task.payload["candidate_id"] for task in tasks if task.kind == kind] == shown
    pairs = [task.payload for task in tasks if task.kind == "duplicate"]
    assert [(item["first"]["candidate_id"], item["second"]["candidate_id"]) for item in pairs] == [
        (shown[0], shown[1]), (shown[0], shown[2]), (shown[1], shown[2])]


def test_explicit_empty_top_is_not_replaced_by_retained_passports(suite):
    from scripts.evaluate_pilot import make_tasks

    case = cases_for(suite, "development")[0]
    tasks = make_tasks(case, sampling_payload([]), {"studies": [], "relevance": []}, None, None, 20)
    assert not [task for task in tasks if task.kind in {"candidate", "scope", "specificity", "duplicate"}]
    # Claim/fabrication quality intentionally samples all saved passports;
    # the manifest declares that separate population explicitly.
    assert len([task for task in tasks if task.kind == "fabrication"]) == 5


def test_legacy_payload_without_selection_retains_first15_sampling(suite):
    from scripts.evaluate_pilot import make_tasks

    case = cases_for(suite, "development")[0]
    raw = sampling_payload(None)
    del raw["result"]["top_trend_ids"]
    tasks = make_tasks(case, raw, {"studies": [], "relevance": []}, None, None, 20)
    assert [task.payload["candidate_id"] for task in tasks if task.kind == "candidate"] == [
        f"candidate-{index}" for index in range(15)]


@pytest.mark.parametrize("ids", [["candidate-2", "candidate-2"], ["missing"], "candidate-2"])
def test_invalid_displayed_selection_is_not_silently_repaired(ids):
    from scripts.evaluate_pilot import shown_top_cards

    with pytest.raises(EvaluationError, match="unique existing"):
        shown_top_cards(sampling_payload(ids)["result"])


def test_scientific_denominator_uses_actual_top_count_not_archive_size(suite):
    manifest = candidate_manifest(suite)
    for outcome in manifest["outcomes"]:
        outcome.update(cards_count=45, candidate_returned=3,
            shown_top_ids=["candidate-1", "candidate-2", "candidate-3"])
    result = quality_metrics(manifest, [])
    assert result["candidate_metrics"]["return_yield_macro"] == pytest.approx(3 / 15)
    manifest["outcomes"][0]["shown_top_ids"] = ["candidate-3", "candidate-2", "candidate-1"]
    with pytest.raises(EvaluationError, match="identities and rank"):
        quality_metrics(manifest, [])


def test_successful_explicit_empty_top_has_zero_yield_and_unknown_precision(suite):
    manifest = manifest_for(suite)
    for outcome in manifest["outcomes"]:
        outcome.update(cards_count=45, candidate_returned=0, shown_top_ids=[])
    result = quality_metrics(manifest, labels_for(manifest))
    assert result["candidate_metrics"]["return_yield_macro"] == 0
    assert result["candidate_metrics"]["weak_signal_precision_macro"] is None
    assert result["candidate_metrics"]["emerging_precision_macro"] is None
    assert result["gates"]["candidate_yield_at_15"] is False
    assert "assisted_reviews_traceable" not in result["gates"]


def test_adjudication_requires_a_third_normalized_human_identity(suite):
    task = ReviewTask.model_validate(manifest_for(suite)["tasks"][0])
    primary = HumanLabel(task_id=task.task_id, reviewer_id="Reviewer A", label="relevant")
    secondary = HumanLabel(task_id=task.task_id, reviewer_id="Reviewer B", phase="secondary", label="irrelevant")
    adjudication = HumanLabel(task_id=task.task_id, reviewer_id=" reviewer a ", phase="adjudication", label="relevant")
    with pytest.raises(EvaluationError, match="third independent"):
        resolve_labels([task], [primary, secondary, adjudication])
    secondary = secondary.model_copy(update={"reviewer_id": " reviewer a "})
    with pytest.raises(EvaluationError, match="different person"):
        resolve_labels([task], [primary, secondary])


def test_scoring_cannot_relabel_published_blind_as_sealed_or_change_suite_hash(suite):
    manifest = manifest_for(suite)
    manifest["suite"]["gates"]["precision_at_20_macro"] = 0
    with pytest.raises(EvaluationError, match="suite hash"):
        quality_metrics(manifest, [])
    manifest = manifest_for(suite)
    manifest.update(split="blind", cases=cases_for(suite, "blind"), tasks=[], outcomes=[])
    measured = quality_metrics(manifest, [])
    assert measured["gates"]["blind_protocol_sealed"] is False


def test_scope_and_specificity_population_must_cover_all_shown_candidates(suite):
    manifest = candidate_manifest(suite)
    measured = quality_metrics(manifest, [])
    assert measured["gates"]["scope_population_complete"] is False
    assert measured["gates"]["specificity_population_complete"] is False
    assert measured["gates"]["duplicate_pair_population_complete"] is False
    assert measured["gates"]["fabrication_population_complete"] is False


def test_claim_population_cannot_omit_a_sampled_substantive_claim(suite):
    from scripts.evaluate_pilot import review_population_coverage
    case = cases_for(suite, "development")[0]
    claim = {"role": "problem", "text": "Needs real source review", "evidence_ids": []}
    card = ReviewTask(task_id="sample", **case, kind="fabrication", secondary_required=False,
        payload={"candidate_id": "candidate", "claims": [claim]})
    outcome = {case["case_id"]: {"cards_count": 1, "sampled_card_ids": ["candidate"]}}
    measured = review_population_coverage([case], outcome, [card])
    assert measured["fabrication_population_complete"] is True
    assert measured["claim_population_complete"] is False
    task = ReviewTask(task_id="claim", **case, kind="claim", secondary_required=False,
        payload={"candidate_id": "candidate", "claim": claim})
    assert review_population_coverage([case], outcome, [card, task])["claim_population_complete"] is True
