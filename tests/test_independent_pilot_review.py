"""Synthetic protocol fixtures test tooling only; these are never expert labels."""
from copy import deepcopy
import csv
import json
import math
from pathlib import Path

import pytest

from app.backend.contracts import DocumentRecord
from scripts.evaluate_pilot import EvaluationError, cases_for, digest, load_suite, read_json, write_json
from scripts.prepare_independent_review import (
    COLUMNS, family_preflight, file_sha, inventory_template, prepare, score, visible_payload, write_csv,
)


@pytest.fixture
def packet_inputs(tmp_path):
    suite = load_suite(Path(__file__).parents[1] / "tests/evaluation/pilot-v2.json")
    suite.update(development=["Synthetic development"], validation=["Synthetic validation"], blind=["Synthetic blind"],
                 blind_status="sealed_external", frozen_at="2026-01-01T00:00:00+00:00")
    suite["pairs_per_direction"] = dict.fromkeys(("development", "validation", "blind"), 20)
    suite["human_labeling"].update(total_pairs=60, card_sample=[1, 5])
    inventory = inventory_template(suite)
    inventory.update(status="sealed_external", custodian_id="Custodian", sealed_at="2026-01-03T00:00:00+00:00",
                     exposure_inventory_complete=True)
    rows = inventory["corpora"] + [{"case_id": "exposure-prior", "split": "development", "query": "Prior exposed"}]
    documents_by_case = {}
    for number, row in enumerate(rows):
        documents = [DocumentRecord(source="crossref", source_id=f"doc-{number}-{index}",
            doi=f"10.9999/test.{number}.{index}", title=f"Synthetic source {number} {index}",
            abstract="Frozen abstract. However, no improvement was demonstrated.", url=f"https://example.org/{number}/{index}",
            fetched_at="2026-01-02T00:00:00Z").model_dump(mode="json") for index in range(20)]
        corpus = {"documents": documents}
        path = tmp_path / f"corpus-{number}.json"
        write_json(path, corpus)
        row.update(path=path.name, sha256=file_sha(path), families=[{"family_id": f"family-{number}",
            "identity_keys": [key for doc in documents for key in ("doi:" + doc["doi"], "crossref:" + doc["source_id"])],
            "revision_ids": [digest(doc) for doc in documents]}])
        documents_by_case[row["case_id"]] = (documents, digest(corpus))
    inventory["prior_exposure_corpora"] = rows[-1:]
    cases = cases_for(suite, "blind")
    case = cases[0]
    docs, corpus_hash = documents_by_case[case["case_id"]]
    tasks = []

    def add(kind, payload, rank=None):
        tasks.append({"task_id": "secret-task-" + digest([kind, payload]), **case, "kind": kind,
                      "rank": rank, "secondary_required": False, "payload": payload})

    for rank, doc in enumerate(docs, 1):
        add("pair", {"study_id": f"secret-study-{rank}", "revision_id": digest(doc),
            "source_url": doc["url"], "title": doc["title"], "abstract": doc["abstract"],
            "score": 0.987, "status": "private-model-status"}, rank)
    for rank in range(1, 9):
        evidence = [{"evidence_id": f"secret-evidence-{rank}", "revision_id": digest(docs[0]),
            "study_id": "secret-study-1", "source_url": docs[0]["url"], "quote": "Frozen abstract",
            "text_field": "abstract", "retrieved_at": "2026-01-02T00:00:00Z", "score": 0.123}]
        claims = [{"role": "problem", "text": f"Test mechanism {rank} claim", "evidence_ids": [f"secret-evidence-{rank}"]}]
        payload = {"candidate_id": f"secret-candidate-{rank}", "label": f"Test mechanism {rank}",
            "definition": f"Mechanism {rank} under review", "claims": claims, "evidence": evidence,
            "rank": rank, "category": "confirmed_trend", "score": 0.567}
        add("candidate", payload, rank)
        for kind in ("scope", "specificity"):
            add(kind, {key: payload[key] for key in ("candidate_id", "label", "definition", "evidence")})
        if rank <= 5:
            add("fabrication", payload)
            add("claim", {"candidate_id": payload["candidate_id"], "claim": claims[0], "evidence": evidence})
    for first in range(1, 9):
        for second in range(first + 1, 9):
            add("duplicate", {side: {"candidate_id": f"secret-candidate-{number}", "label": f"Test mechanism {number}",
                "definition": f"Mechanism {number} under review"} for side, number in (("first", first), ("second", second))})
    manifest = {"suite": suite, "suite_hash": digest(suite), "split": "blind", "cases": cases, "tasks": tasks,
        "workflow_version": "test-fixture", "encoder_spec": {"test": "no-model"}, "code_hashes": {"test": "synthetic"},
        "created_at": "2026-01-02T12:00:00+00:00", "evaluation_mode": "automatic",
        "outcomes": [{**case, "state": "succeeded", "cards_count": 8, "candidate_returned": 8,
            "shown_top_ids": [f"secret-candidate-{rank}" for rank in range(1, 9)],
            "sampled_card_ids": [f"secret-candidate-{rank}" for rank in range(1, 6)],
            "review_corpus_hash": corpus_hash, "as_of": "2026-01-02", "numeric_provenance_verified": True,
            "truncated_confirmed": 0}]}
    baseline, corrected, registry = (tmp_path / name for name in ("baseline.json", "corrected.json", "inventory.json"))
    write_json(baseline, manifest)
    fixed = deepcopy(manifest)
    for task in fixed["tasks"]:
        task["task_id"] = task["task_id"].replace("secret-task-", "secret-corrected-task-")
    write_json(corrected, fixed)
    write_json(registry, inventory)
    return baseline, corrected, registry, suite, inventory


def make_packet(inputs, tmp_path, name="packet", seed="a" * 64):
    baseline, corrected, registry, _, _ = inputs
    return prepare(baseline, corrected, registry, tmp_path / name, seed)


def test_packets_hide_arm_rank_status_score_and_internal_ids_and_merge_identical_tasks(packet_inputs, tmp_path):
    output = make_packet(packet_inputs, tmp_path)
    private = read_json(output / "private/manifest.json")
    assert all(len(values) == 2 for values in private["mapping"].values())
    for path in (output / "reviewer").iterdir():
        text = path.read_text(encoding="utf-8-sig")
        for marker in ("secret-", "0.987", "0.123", "0.567", '"rank"', '"score"', '"status"',
                       "baseline", "corrected", "confirmed_trend", "private-model-status"):
            assert marker not in text
    with (output / "reviewer/primary.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(not row["label"] and not row["reviewer_id"] for row in rows)
    claim_payload = json.loads(next(row["payload"] for row in rows if row["kind"] == "claim"))
    assert "publication_year" in claim_payload["evidence"][0]["source_document"]
    assert claim_payload["evidence"][0]["source_document"]["abstract"] == (
        "Frozen abstract. However, no improvement was demonstrated.")
    assert set(rows[0]) == set(COLUMNS)
    for kinds in private["double_review"].values():
        assert all(value["secondary"] >= math.ceil(value["tasks"] * .25) for value in kinds.values())


def test_seed_is_reproducible_private_and_changes_order(packet_inputs, tmp_path):
    first = make_packet(packet_inputs, tmp_path, "one")
    same = make_packet(packet_inputs, tmp_path, "two")
    changed = make_packet(packet_inputs, tmp_path, "three", "b" * 64)
    assert (first / "reviewer/primary.csv").read_bytes() == (same / "reviewer/primary.csv").read_bytes()
    assert (first / "reviewer/primary.csv").read_bytes() != (changed / "reviewer/primary.csv").read_bytes()
    assert (first / "private/manifest.json").read_bytes() == (same / "private/manifest.json").read_bytes()
    with pytest.raises(EvaluationError, match="refusing overwrite"):
        make_packet(packet_inputs, tmp_path, "one")


@pytest.mark.parametrize("mutation,reason", [
    ("same_family", "crosses splits"), ("same_alias", "crosses splits"),
    ("missing_revision", "omits corpus revisions"), ("missing_alias", "omit an actual"),
    ("missing_case", "All development"), ("missing_exposure", "prior exposure"),
    ("wrong_sha", "SHA-256 mismatch"), ("exposures_unknown", "prior exposed")])
def test_family_preflight_rejects_leakage_incomplete_inventory_and_hash_mismatch(packet_inputs, tmp_path, mutation, reason):
    _, _, _, suite, inventory = packet_inputs
    changed = deepcopy(inventory)
    rows = changed["corpora"]
    if mutation == "same_family":
        rows[1]["families"][0]["family_id"] = rows[0]["families"][0]["family_id"]
    elif mutation == "same_alias":
        rows[1]["families"][0]["identity_keys"].append(rows[0]["families"][0]["identity_keys"][0])
    elif mutation == "missing_revision":
        rows[0]["families"][0]["revision_ids"].pop()
    elif mutation == "missing_alias":
        rows[0]["families"][0]["identity_keys"].pop()
    elif mutation == "missing_case":
        rows.pop()
    elif mutation == "missing_exposure":
        changed["prior_exposure_corpora"] = []
    elif mutation == "wrong_sha":
        rows[0]["sha256"] = "0" * 64
    else:
        changed["exposure_inventory_complete"] = False
    with pytest.raises(EvaluationError, match=reason):
        family_preflight(suite, changed, tmp_path)


def roster_for(packet, tmp_path):
    path = tmp_path / "roster.json"
    roster = {"status": "attested_external", "attested_at": "2026-01-04T00:00:00+00:00",
        "packet_sha256": file_sha(packet / "private/manifest.json"),
        "blinded_before_primary_submission": True, "label_origin_human_only": True,
        "reviewers": [{"reviewer_id": phase + " test human", "phases": [phase],
            "external_human": True, "independent_of_implementation": True, "subject_expertise": "synthetic fixture"}
            for phase in ("primary", "secondary", "adjudication")]}
    write_json(path, roster)
    return path


def filled_answers(packet, tmp_path):
    """Only test-generated labels; never used on actual scientific output."""
    positive = dict(pair="relevant", candidate="emerging", scope="relevant", specificity="specific",
                    claim="supported", fabrication="no_fabrication", duplicate="distinct")
    answers = []
    for phase in ("primary", "secondary"):
        with (packet / f"reviewer/{phase}.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row.update(reviewer_id=phase + " test human", label=positive[row["kind"]])
        path = tmp_path / f"filled-{phase}.csv"
        write_csv(path, rows)
        answers.append(path)
    return answers


def test_empty_labels_cannot_pass_and_test_only_full_labels_are_mapped_without_loss(packet_inputs, tmp_path):
    packet = make_packet(packet_inputs, tmp_path)
    roster = roster_for(packet, tmp_path)
    blank = score(packet / "private/manifest.json", [packet / "reviewer/primary.csv", packet / "reviewer/secondary.csv"],
                  roster, tmp_path / "blank-score")
    assert blank["status"] == "not_accepted"
    assert blank["gates"]["all_review_labels_resolved"] is False
    assert blank["arms"]["corrected"]["candidate_metrics"]["emerging_precision_macro"] is None
    measured = score(packet / "private/manifest.json", filled_answers(packet, tmp_path), roster, tmp_path / "test-only-score")
    assert measured["status"] == "passed"
    assert measured["arms"]["corrected"]["planned_cases"] == 1
    assert measured["corrected_minus_baseline"]["emerging_precision_macro"] == 0
    with pytest.raises(EvaluationError, match="refusing overwrite"):
        score(packet / "private/manifest.json", [], roster, tmp_path / "test-only-score")


def test_readonly_review_metadata_cannot_be_changed(packet_inputs, tmp_path):
    packet = make_packet(packet_inputs, tmp_path)
    roster = roster_for(packet, tmp_path)
    answers = filled_answers(packet, tmp_path)
    text = answers[0].read_text(encoding="utf-8-sig").replace("Synthetic blind", "Changed direction", 1)
    answers[0].write_text(text, encoding="utf-8-sig")
    with pytest.raises(EvaluationError, match="modified read-only"):
        score(packet / "private/manifest.json", answers, roster, tmp_path / "score")


def test_changed_frozen_input_and_direct_inference_manifest_are_rejected(packet_inputs, tmp_path):
    packet = make_packet(packet_inputs, tmp_path)
    roster = roster_for(packet, tmp_path)
    packet_inputs[0].write_text("{}")
    with pytest.raises(EvaluationError, match="Frozen input changed"):
        score(packet / "private/manifest.json", [], roster, tmp_path / "score")
    with pytest.raises((EvaluationError, KeyError)):
        make_packet(packet_inputs, tmp_path, "second")


def test_published_protocol_is_unsealed_and_template_never_claims_completeness():
    suite = load_suite(Path(__file__).parents[1] / "tests/evaluation/pilot-v2.json")
    template = inventory_template(suite)
    assert template["status"] == "pending_external"
    assert template["exposure_inventory_complete"] is False
    assert all(row["path"] is None and not row["families"] for row in template["corpora"])


def test_unknown_claim_source_is_never_silently_dropped(packet_inputs):
    from scripts.evaluate_pilot import ReviewTask
    manifest = read_json(packet_inputs[0])
    task = next(row for row in manifest["tasks"] if row["kind"] == "claim")
    task["payload"]["claim"]["evidence_ids"] = ["missing"]
    with pytest.raises(EvaluationError, match="absent reviewer evidence"):
        visible_payload(ReviewTask.model_validate(task))


def test_unsealed_preview_cannot_pass_even_when_test_labels_meet_every_quality_gate(packet_inputs, tmp_path):
    baseline, corrected, registry, _, _ = packet_inputs
    for path in (baseline, corrected):
        manifest = read_json(path)
        manifest["suite"].update(blind_status="published_proposals_unsealed", frozen_at=None)
        manifest["suite_hash"] = digest(manifest["suite"])
        path.write_text(json.dumps(manifest))
    inventory = read_json(registry)
    inventory["suite_hash"] = read_json(baseline)["suite_hash"]
    registry.write_text(json.dumps(inventory))
    packet = make_packet(packet_inputs, tmp_path)
    result = score(packet / "private/manifest.json", filled_answers(packet, tmp_path), roster_for(packet, tmp_path), tmp_path / "score")
    assert result["status"] == "not_accepted"
    assert result["gates"]["external_protocol_sealed"] is False
    assert result["gates"]["protocol_frozen_before_runs"] is False
    assert result["arms"]["corrected"]["gates"]["blind_protocol_sealed"] is False


def test_missing_source_revision_is_rejected_before_creating_packet(packet_inputs, tmp_path):
    corrected = packet_inputs[1]
    manifest = read_json(corrected)
    manifest["tasks"][0]["payload"]["revision_id"] = "0" * 64
    corrected.write_text(json.dumps(manifest))
    with pytest.raises(EvaluationError, match="absent from the bound corpus"):
        make_packet(packet_inputs, tmp_path)
    assert not (tmp_path / "packet").exists()


def test_real_disagreement_produces_blank_adjudication_and_requires_third_review(packet_inputs, tmp_path):
    packet = make_packet(packet_inputs, tmp_path)
    roster = roster_for(packet, tmp_path)
    answers = filled_answers(packet, tmp_path)
    with answers[1].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    disagreement = next(row for row in rows if row["kind"] == "pair")
    disagreement["label"] = "irrelevant"
    alternative = tmp_path / "disagreeing-secondary.csv"
    write_csv(alternative, rows)
    incomplete = score(packet / "private/manifest.json", [answers[0], alternative], roster, tmp_path / "disagreement")
    assert incomplete["status"] == "not_accepted"
    assert incomplete["labeling"]["disagreements"] == 1
    with (tmp_path / "disagreement/adjudication.csv").open(encoding="utf-8-sig", newline="") as stream:
        adjudication = list(csv.DictReader(stream))
    assert len(adjudication) == 1 and not adjudication[0]["label"] and not adjudication[0]["reviewer_id"]
    adjudication[0].update(label="relevant", reviewer_id="adjudication test human")
    adjudicated = tmp_path / "test-adjudication.csv"
    write_csv(adjudicated, adjudication)
    complete = score(packet / "private/manifest.json", [answers[0], alternative, adjudicated], roster, tmp_path / "adjudicated")
    assert complete["status"] == "passed" and complete["labeling"]["adjudicated"] == 1


def test_nonexternal_reviewer_or_roster_for_another_packet_is_rejected(packet_inputs, tmp_path):
    packet = make_packet(packet_inputs, tmp_path)
    roster_path = roster_for(packet, tmp_path)
    roster = read_json(roster_path)
    roster["reviewers"][0]["independent_of_implementation"] = False
    roster_path.write_text(json.dumps(roster))
    answers = filled_answers(packet, tmp_path)
    with pytest.raises(EvaluationError, match="attested external"):
        score(packet / "private/manifest.json", answers, roster_path, tmp_path / "score")
    roster["packet_sha256"] = "0" * 64
    roster_path.write_text(json.dumps(roster))
    with pytest.raises(EvaluationError, match="exact private packet"):
        score(packet / "private/manifest.json", [], roster_path, tmp_path / "score")


def test_duplicate_implicit_case_is_rejected_but_distinct_explicit_variants_are_bound(packet_inputs, tmp_path):
    _, _, _, suite, inventory = packet_inputs
    changed = deepcopy(inventory)
    changed["corpora"].append(deepcopy(changed["corpora"][0]))
    with pytest.raises(EvaluationError, match="Duplicate case"):
        family_preflight(suite, changed, tmp_path)
    changed["corpora"][0]["variant_id"] = "baseline"
    changed["corpora"][-1]["variant_id"] = "corrected"
    result = family_preflight(suite, changed, tmp_path)
    assert result["family_disjoint"] is True
