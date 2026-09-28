"""Explicit, cost-bounded frozen evaluation using the desktop PilotService.

This module is never imported by the application. Blind execution requires an
extra flag. Scientific lifecycle is evaluated by blinded human labels, not
by the application's own confirmation count. Human labels are never filled
by an AI, fabricated from successful execution, or inferred from exact hashes.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, UTC
import hashlib
import io
import json
import math
from multiprocessing import freeze_support
from pathlib import Path
import time
from typing import Any, Literal
import unicodedata
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

SPLITS = ("development", "validation", "blind")
VERSION = "frozen-pilot-evaluation/2.2.0"
MAX_JSON_BYTES = 25_000_000


class EvaluationError(ValueError):
    pass


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    from app.runtime.backup import strict_json
    from app.pilot.reports import open_local_regular

    with open_local_regular(path) as stream:
        data = stream.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES:
        raise EvaluationError("Evaluation JSON exceeds 25 MB")
    return strict_json(data)


def write_json(path: Path, value: Any) -> None:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    if len(data) > MAX_JSON_BYTES:
        raise EvaluationError("Evaluation JSON exceeds 25 MB")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def load_suite(path: Path) -> dict[str, Any]:
    suite = read_json(path)
    if not isinstance(suite, dict) or suite.get("schema_version") != 1 or suite.get("application_must_not_load_this_file") is not True:
        raise EvaluationError("Not a frozen evaluation protocol")
    seen: set[str] = set()
    total = 0
    for split in SPLITS:
        values = suite.get(split)
        if not isinstance(values, list) or not values or len(values) > 100:
            raise EvaluationError("Each split requires 1–100 directions")
        for value in values:
            if not isinstance(value, str) or not 2 <= len(value.strip()) <= 500:
                raise EvaluationError("Invalid evaluation direction")
            normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
            if normalized in seen:
                raise EvaluationError("Development, validation and blind splits must be disjoint")
            seen.add(normalized)
        count = suite["pairs_per_direction"].get(split)
        if type(count) is not int or not 20 <= count <= 100:
            raise EvaluationError("Pair sample requires at least 20 real ranked documents")
        total += count * len(values)
    if suite["human_labeling"]["total_pairs"] != total or suite["human_labeling"].get("missing_labels_are_not_passes") is not True:
        raise EvaluationError("Human sample counts or missing-label policy disagree with the protocol")
    fraction = suite["human_labeling"]["second_reviewer_fraction"]
    if type(fraction) not in (int, float) or not 0 < fraction <= 1:
        raise EvaluationError("Invalid second-review fraction")
    return suite


def cases_for(suite: dict[str, Any], split: str) -> list[dict[str, str]]:
    if split not in SPLITS:
        raise EvaluationError("Choose one explicit evaluation split")
    return [{"case_id": digest({"split": split, "query": query})[:20], "query": query} for query in suite[split]]


class ReviewTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    case_id: str
    query: str
    kind: Literal["pair", "candidate", "scope", "specificity", "claim", "fabrication", "duplicate"]
    rank: int | None = Field(default=None, ge=1, le=20)
    secondary_required: bool = Field(strict=True)
    payload: dict[str, Any]


class HumanLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str = Field(min_length=1, max_length=100)
    reviewer_id: str = Field(min_length=2, max_length=200)
    phase: Literal["primary", "secondary", "adjudication"] = "primary"
    label: Literal["relevant", "irrelevant", "uncertain", "supported", "unsupported", "no_fabrication", "fabricated", "distinct", "duplicate", "early_weak_signal", "emerging", "established_research", "mainstream", "renewed_interest", "false_positive", "insufficient_evidence", "specific", "broad"]


_ALLOWED = {"candidate": {"early_weak_signal", "emerging", "established_research", "mainstream", "renewed_interest", "false_positive", "insufficient_evidence", "uncertain"},
            "scope": {"relevant", "irrelevant", "uncertain"}, "specificity": {"specific", "broad", "uncertain"}, "pair": {"relevant", "irrelevant", "uncertain"}, "claim": {"supported", "unsupported", "uncertain"},
            "fabrication": {"no_fabrication", "fabricated", "uncertain"}, "duplicate": {"distinct", "duplicate", "uncertain"}}


def resolve_labels(tasks: list[ReviewTask], labels: list[HumanLabel]) -> tuple[dict[str, str | None], dict[str, int]]:
    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise EvaluationError("Duplicate review tasks")
    grouped: dict[str, dict[str, HumanLabel]] = {}
    for label in labels:
        task = by_id.get(label.task_id)
        if task is None or label.label not in _ALLOWED[task.kind] or not label.reviewer_id.strip():
            raise EvaluationError("Unknown task, wrong label vocabulary or missing reviewer")
        values = grouped.setdefault(label.task_id, {})
        if label.phase in values:
            raise EvaluationError("Multiple labels for the same task and reviewer phase")
        values[label.phase] = label
    resolved: dict[str, str | None] = {}
    stats = {"primary": 0, "secondary": 0, "disagreements": 0, "adjudicated": 0, "missing_or_uncertain": 0}
    for task in tasks:
        values = grouped.get(task.task_id, {})
        primary, secondary, adjudication = (values.get(phase) for phase in ("primary", "secondary", "adjudication"))
        if (secondary or adjudication) and not primary:
            raise EvaluationError("Second review or adjudication has no primary label")
        if primary and secondary and primary.reviewer_id.strip().casefold() == secondary.reviewer_id.strip().casefold():
            raise EvaluationError("The second reviewer must be a different person")
        if adjudication and (not secondary or not primary or primary.label == secondary.label):
            raise EvaluationError("Adjudication requires an actual recorded disagreement")
        if adjudication and primary and secondary and adjudication.reviewer_id.strip().casefold() in {
                primary.reviewer_id.strip().casefold(), secondary.reviewer_id.strip().casefold()}:
            raise EvaluationError("Adjudication requires a third independent reviewer")
        stats["primary"] += primary is not None
        stats["secondary"] += secondary is not None
        value = primary.label if primary else None
        if primary and secondary and primary.label != secondary.label:
            stats["disagreements"] += 1
            value = adjudication.label if adjudication else None
            stats["adjudicated"] += adjudication is not None
        elif task.secondary_required and secondary is None:
            value = None
        if value == "uncertain":
            value = None
        resolved[task.task_id] = value
        stats["missing_or_uncertain"] += value is None
    return resolved, stats


def quality_metrics(manifest: dict[str, Any], labels: list[HumanLabel]) -> dict[str, Any]:
    if manifest.get("suite_hash") != digest(manifest["suite"]):
        raise EvaluationError("Frozen suite hash does not match")
    tasks = [ReviewTask.model_validate(item) for item in manifest["tasks"]]
    outcomes = manifest["outcomes"]
    cases = manifest["cases"]
    if cases != cases_for(manifest["suite"], manifest["split"]):
        raise EvaluationError("All frozen split cases must remain in the denominator")
    expected = {case["case_id"] for case in cases}
    by_case = {outcome["case_id"]: outcome for outcome in outcomes}
    if len(expected) != len(cases) or len(by_case) != len(outcomes) or set(by_case) - expected:
        raise EvaluationError("Duplicate or unknown evaluation cases")
    if any(task.case_id not in expected for task in tasks):
        raise EvaluationError("Review task belongs to another split")
    queries = {case["case_id"]: case["query"] for case in cases}
    if any(task.query != queries[task.case_id] for task in tasks):
        raise EvaluationError("Review query differs from the frozen case")
    if any(outcome["state"] not in {"succeeded", "failed", "not_run", "cancelled", "interrupted", "queued", "running"}
           for outcome in outcomes):
        raise EvaluationError("Unknown execution outcome")
    pair_ids = [(task.case_id, task.payload.get("study_id")) for task in tasks if task.kind == "pair"]
    if len(pair_ids) != len(set(pair_ids)) or any(not identity for _, identity in pair_ids):
        raise EvaluationError("A study cannot fill multiple retrieval positions for one direction")
    values, label_stats = resolve_labels(tasks, labels)
    rows: list[dict[str, Any]] = []
    for case in cases:
        outcome = by_case.get(case["case_id"], {"state": "not_run"})
        ranked = [task for task in tasks if task.case_id == case["case_id"] and task.kind == "pair" and task.rank is not None]
        ranks = [task.rank for task in ranked if task.rank is not None]
        if len(ranks) != len(set(ranks)) or sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise EvaluationError("P@20 requires unique consecutive actual retrieval ranks")
        succeeded = outcome["state"] == "succeeded"
        relevant = sum(values[task.task_id] == "relevant" for task in ranked)
        unknown = sum(values[task.task_id] is None for task in ranked)
        # Empty positions after a successful short return are explicit retrieval
        # abstentions. Failed/unrun cases have no measured retrieval outcome.
        upper = (relevant + unknown) / 20 if succeeded else 1.0
        point = relevant / 20 if succeeded and unknown == 0 else None
        rows.append({**case, "state": outcome["state"], "returned_at_20": len(ranked), "known_relevant": relevant,
            "unknown_labels": unknown, "precision_at_20": point,
            "lower_bound": relevant / 20 if succeeded else 0.0, "upper_bound": upper})
    macro = sum(row["precision_at_20"] for row in rows) / len(rows) if all(row["precision_at_20"] is not None for row in rows) else None
    claim_tasks = [task for task in tasks if task.kind == "claim"]
    known_supported = sum(values[task.task_id] == "supported" for task in claim_tasks)
    unknown_claims = sum(values[task.task_id] is None for task in claim_tasks)
    claim_support = known_supported / len(claim_tasks) if claim_tasks and not unknown_claims else None
    fabricated = [task for task in tasks if task.kind == "fabrication"]
    fabricated_count = sum(values[task.task_id] == "fabricated" for task in fabricated)
    duplicates = [task for task in tasks if task.kind == "duplicate"]
    duplicate_counts = {case_id: sum(task.case_id == case_id and values[task.task_id] == "duplicate" for task in duplicates) for case_id in expected}
    all_succeeded = all(row["state"] == "succeeded" for row in rows)
    gates = manifest["suite"]["gates"]
    results: dict[str, bool | None] = {
        "all_planned_cases_finished": all_succeeded,
        "precision_at_20_macro": macro >= gates["precision_at_20_macro"] if macro is not None else None,
        "minimum_per_blind_direction": (all(row["precision_at_20"] >= gates["minimum_per_blind_direction"] for row in rows)
            if manifest["split"] == "blind" and macro is not None else None),
        "substantive_claim_support": claim_support >= gates["substantive_claim_support"] if claim_support is not None else None,
        "fabricated_entities_or_sources": (fabricated_count == 0 if fabricated and all(values[task.task_id] is not None for task in fabricated) else None),
        "duplicate_pairs_per_top15": (all(count <= gates["maximum_duplicate_pairs_per_top15"] for count in duplicate_counts.values())
            if duplicates and all(values[task.task_id] is not None for task in duplicates) else None),
        "second_reviewer_coverage": (label_stats["secondary"] / len(tasks) >= manifest["suite"]["human_labeling"]["second_reviewer_fraction"]
            if tasks else None),
        "all_disagreements_adjudicated": label_stats["adjudicated"] == label_stats["disagreements"],
        "all_human_labels_resolved": label_stats["missing_or_uncertain"] == 0 and bool(tasks),
        "numeric_provenance": (all(outcome.get("numeric_provenance_verified") is True for outcome in outcomes) if all_succeeded else None),
        "truncated_histories_called_confirmed": (all(outcome.get("truncated_confirmed") == 0 for outcome in outcomes) if all_succeeded else None),
        "planned_pair_sample_complete": all(sum(task.kind == "pair" and task.case_id == case["case_id"] for task in tasks)
            == manifest["suite"]["pairs_per_direction"][manifest["split"]] for case in cases),
        "card_sample_size": (manifest["suite"]["human_labeling"]["card_sample"][0] <= len(fabricated) <= manifest["suite"]["human_labeling"]["card_sample"][1]
            if manifest["split"] == "blind" else None),
    }
    # A high score on a subset must not conceal missing scope, specificity,
    # duplicate-pair or sampled-claim judgments from the returned population.
    results.update(review_population_coverage(cases, by_case, tasks))
    if manifest["split"] == "blind":
        results["blind_protocol_sealed"] = bool(manifest["suite"].get("blind_status") == "sealed_external"
                                               and manifest["suite"].get("frozen_at"))
    mode = manifest.get("evaluation_mode", "automatic")
    if mode not in {"automatic", "assisted"}:
        raise EvaluationError("Unknown evaluation mode")
    scientific = candidate_metrics(cases, by_case, tasks, values)
    candidate_threshold = gates.get("emerging_precision_at_15", 0.80)
    results["candidate_lifecycle_evaluated"] = scientific["all_returned_candidates_labeled"]
    results["emerging_precision_at_15"] = (scientific["emerging_precision_macro"] >= candidate_threshold
        if scientific["emerging_precision_macro"] is not None else None)
    for key in ("scope_precision", "specificity_precision"):
        score = scientific[key]
        results[key] = score >= gates.get(key, 0.95) if score is not None else None
    if manifest["split"] != "blind":
        results.pop("minimum_per_blind_direction")
        results.pop("card_sample_size")
    # Fixed-slot yield prevents perfect precision by returning one safe card.
    results["candidate_yield_at_15"] = (scientific["emerging_yield_macro"] >= gates.get("emerging_yield_at_15", 0.50)
        if scientific["emerging_yield_macro"] is not None else None)
    if mode == "assisted":
        results["assisted_reviews_traceable"] = bool(manifest.get("assisted_review_ids"))
    result_status = "failed" if any(value is False for value in results.values()) else "pending" if any(value is None for value in results.values()) else "passed"
    return {"schema_version": 1, "version": VERSION, "split": manifest["split"], "manifest_hash": digest(manifest),
        "status": result_status, "evaluation_mode": mode, "candidate_metrics": scientific, "planned_cases": len(cases), "successful_cases": sum(row["state"] == "succeeded" for row in rows),
        "precision_at_20_macro": macro, "document_relevance_precision_at_20_macro": macro, "precision_macro_lower_bound": sum(row["lower_bound"] for row in rows) / len(rows),
        "precision_macro_upper_bound": sum(row["upper_bound"] for row in rows) / len(rows), "per_direction": rows,
        "substantive_claim_support": claim_support, "claims_sampled": len(claim_tasks), "cards_sampled": len(fabricated),
        "fabricated_entities_observed": fabricated_count, "duplicate_pairs_observed": duplicate_counts,
        "labeling": label_stats, "gates": results,
        "limitations": ["Missing and uncertain human labels are never treated as passes.",
            "A failed or unrun case remains in the planned denominator; lower and upper bounds are not measured accuracy.",
            "P@20 uses 20 fixed retrieval positions; a successful short return has explicit empty positions, never fabricated documents.",
            "P@20 measures document relevance only; emerging and weak-signal precision use separate candidate lifecycle labels.",
            "Candidate rank, model score and assigned lifecycle are hidden in the reviewer CSV; ranks remain in the frozen manifest for metric computation.",
            "Current audited cases are development regressions, never independent blind evidence.",
            "Hash/numerical checks prove reproducibility, not scientific truth or an independent expert review.",
            "If blind failures influence tuning, this split becomes validation; use a new sealed set."]}


def review_population_coverage(cases: list[dict[str, str]], outcomes: dict[str, Any],
                               tasks: list[ReviewTask]) -> dict[str, bool]:
    scope_ok = specificity_ok = duplicates_ok = fabrication_ok = claims_ok = True
    for case in cases:
        selected = [task for task in tasks if task.case_id == case["case_id"]]
        top = {task.payload.get("candidate_id") for task in selected if task.kind == "candidate"}
        for kind in ("scope", "specificity"):
            identifiers = [task.payload.get("candidate_id") for task in selected if task.kind == kind]
            complete = len(identifiers) == len(set(identifiers)) and set(identifiers) == top
            if kind == "scope":
                scope_ok &= complete
            else:
                specificity_ok &= complete
        expected_pairs = {frozenset((first, second)) for first in top for second in top if first != second}
        actual_pairs = [frozenset(task.payload.get(side, {}).get("candidate_id") for side in ("first", "second"))
                        for task in selected if task.kind == "duplicate"]
        duplicates_ok &= len(actual_pairs) == len(set(actual_pairs)) and set(actual_pairs) == expected_pairs
        cards = [task for task in selected if task.kind == "fabrication"]
        identifiers = [task.payload.get("candidate_id") for task in cards]
        outcome = outcomes.get(case["case_id"], {})
        expected_cards = outcome.get("sampled_card_ids")
        fabrication_ok &= (isinstance(expected_cards, list) and len(identifiers) == len(set(identifiers))
                           and set(identifiers) == set(expected_cards)
                           and len(identifiers) == min(5, outcome.get("cards_count", 0)))
        expected_claims = [(task.payload.get("candidate_id"), digest(claim)) for task in cards
                           for claim in task.payload.get("claims", ())]
        actual_claims = [(task.payload.get("candidate_id"), digest(task.payload.get("claim")))
                         for task in selected if task.kind == "claim"]
        claims_ok &= (len(actual_claims) == len(set(actual_claims))
                      and set(actual_claims) == set(expected_claims))
    return {"scope_population_complete": bool(scope_ok), "specificity_population_complete": bool(specificity_ok),
            "duplicate_pair_population_complete": bool(duplicates_ok),
            "fabrication_population_complete": bool(fabrication_ok), "claim_population_complete": bool(claims_ok)}


def candidate_metrics(cases: list[dict[str, str]], outcomes: dict[str, Any],
                      tasks: list[ReviewTask], values: dict[str, str | None]) -> dict[str, Any]:
    """Distinct lifecycle precision and yield; an empty TOP is never 100% accurate."""
    rows: list[dict[str, Any]] = []
    confusion: dict[str, int] = {}
    for case in cases:
        ranked = sorted((task for task in tasks if task.case_id == case["case_id"] and task.kind == "candidate"),
                        key=lambda task: task.rank or 0)
        ranks = [task.rank for task in ranked]
        ids = [task.payload.get("candidate_id") for task in ranked]
        if (ranks != list(range(1, len(ranked) + 1)) or len(ranked) > 15
                or len(ids) != len(set(ids)) or any(not item for item in ids)):
            raise EvaluationError("Candidate TOP requires unique IDs and consecutive actual ranks up to 15")
        outcome = outcomes.get(case["case_id"], {})
        expected = (outcome["candidate_returned"] if "candidate_returned" in outcome else
                    min(outcome.get("cards_count", len(ranked)), 15))
        if type(expected) is not int or not 0 <= expected <= 15:
            raise EvaluationError("Returned TOP count must be an integer from 0 to 15")
        if expected != len(ranked):
            raise EvaluationError("All returned TOP-15 candidates must be independently labeled")
        if "shown_top_ids" in outcome and outcome["shown_top_ids"] != ids:
            raise EvaluationError("Candidate tasks must match the displayed TOP identities and rank order")
        succeeded = outcome.get("state") == "succeeded"
        unknown = sum(values[task.task_id] is None for task in ranked)
        ws = sum(values[task.task_id] == "early_weak_signal" for task in ranked)
        emerging = sum(values[task.task_id] in {"early_weak_signal", "emerging"} for task in ranked)
        for task in ranked:
            label = values[task.task_id] or "unresolved"
            confusion[label] = confusion.get(label, 0) + 1
        measured = succeeded and unknown == 0 and bool(ranked)
        rows.append({**case, "returned_at_15": len(ranked), "unknown_lifecycle_labels": unknown,
            "early_weak_signals": ws, "emerging_including_weak": emerging,
            "weak_signal_precision": ws / len(ranked) if measured else None,
            "emerging_precision": emerging / len(ranked) if measured else None,
            "emerging_yield_at_15": emerging / 15 if succeeded and not unknown else None,
            "return_yield_at_15": len(ranked) / 15 if succeeded else None,
            "labels_complete": succeeded and bool(ranked) and not unknown})
    def macro(key: str) -> float | None:
        return sum(row[key] for row in rows) / len(rows) if rows and all(row[key] is not None for row in rows) else None
    return {"per_direction": rows, "weak_signal_precision_macro": macro("weak_signal_precision"),
        "emerging_precision_macro": macro("emerging_precision"), "emerging_yield_macro": macro("emerging_yield_at_15"),
        "return_yield_macro": macro("return_yield_at_15"), "lifecycle_label_counts": confusion,
        "scope_precision": _labeled_fraction(tasks, values, "scope", "relevant"),
        "specificity_precision": _labeled_fraction(tasks, values, "specificity", "specific"),
        "all_returned_candidates_labeled": bool(rows) and all(row["labels_complete"] for row in rows),
        "recall": None, "recall_limitation": "No independently enumerated candidate universe; discarded-document sampling is not full recall."}


def _labeled_fraction(tasks: list[ReviewTask], values: dict[str, str | None], kind: str, positive: str) -> float | None:
    selected = [task for task in tasks if task.kind == kind]
    return (sum(values[task.task_id] == positive for task in selected) / len(selected)
            if selected and all(values[task.task_id] is not None for task in selected) else None)


def shown_top_cards(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Use the displayed selection, including an explicit empty TOP.

    Older payloads without a selection retain their original first-15 sampling
    convention. This helper never promotes other passports to fill an empty TOP.
    """
    cards = result["cards"]
    top_ids = result.get("top_trend_ids")
    if top_ids is None:
        return list(cards[:15])
    by_id = {card["candidate"]["candidate_id"]: card for card in cards}
    if (len(by_id) != len(cards) or not isinstance(top_ids, (list, tuple))
            or len(top_ids) > 15 or any(not isinstance(item, str) or item not in by_id for item in top_ids)
            or len(set(top_ids)) != len(top_ids)):
        raise EvaluationError("Displayed TOP requires unique existing candidate identities")
    return [by_id[identifier] for identifier in top_ids]


def make_tasks(case: dict[str, str], raw: dict[str, Any], discovered: dict[str, Any], snapshot: Any,
               archive: Any, pair_count: int, *, card_count: int = 5) -> list[ReviewTask]:
    studies = {item["study_id"]: item for item in discovered["studies"]}
    relevance = discovered["relevance"]
    if len({item["study_id"] for item in relevance}) != len(relevance):
        raise EvaluationError("Duplicate relevance identities")
    if any(type(item["score"]) not in (int, float) or not math.isfinite(item["score"]) for item in relevance):
        raise EvaluationError("Invalid relevance score")
    ordered = sorted(relevance, key=lambda item: (-item["score"], item["study_id"]))
    retained = [item for item in ordered if item["decision"] == "retained"][:20]
    selected_ids = {item["study_id"] for item in retained}
    selected = retained + [item for item in ordered if item["study_id"] not in selected_ids][:max(0, pair_count - len(retained))]
    tasks: list[ReviewTask] = []

    def add(kind: str, payload: dict[str, Any], rank: int | None = None) -> None:
        identifier = digest({"case": case["case_id"], "kind": kind, "payload": payload})
        tasks.append(ReviewTask.model_validate(dict(task_id=identifier, **case, kind=kind, rank=rank,
            secondary_required=False, payload=payload)))

    for number, item in enumerate(selected):
        study = studies[item["study_id"]]
        index = study["representative_index"]
        if type(index) is not int or not 0 <= index < len(snapshot.documents):
            raise EvaluationError("Representative is absent from frozen discovery corpus")
        reference = snapshot.documents[index]
        document = archive.get(reference.revision_id)
        add("pair", dict(study_id=item["study_id"], revision_id=reference.revision_id, source_url=document.url,
            title=document.title, abstract=(document.abstract or "")[:3000], abstract_truncated=len(document.abstract or "") > 3000,
            ), number + 1 if number < len(retained) else None)
    cards = raw["result"]["cards"]
    shown = shown_top_cards(raw["result"])
    for position, card in enumerate(shown, 1):
        candidate = card["candidate"]
        payload = {"candidate_id": candidate["candidate_id"], "label": candidate["label"],
            "definition": candidate["definition"], "synonyms": candidate["synonyms"],
            "claims": [{key: claim[key] for key in ("role", "text", "evidence_ids")} for claim in card["claims"]],
            "evidence": card["evidence"],
            "rubric": "Assess technology identity, novelty, niche scope, earlyness and independently dated development. Low volume or new terminology alone is not a weak signal. Labels describe the candidate at its displayed granularity."}
        add("candidate", payload, position)
        add("scope", {key: payload[key] for key in ("candidate_id", "label", "definition", "evidence")})
        add("specificity", {key: payload[key] for key in ("candidate_id", "label", "definition", "evidence")})
    chosen = sorted(cards, key=lambda card: digest({"case": case["case_id"], "candidate": card["candidate"]["candidate_id"]}))[:card_count]
    for card in chosen:
        substantive_claims = [claim for claim in card["claims"] if claim["role"] in {"problem", "advantage", "case"}]
        substantive_ids = {identifier for claim in substantive_claims for identifier in claim["evidence_ids"]}
        add("fabrication", {"candidate_id": card["candidate"]["candidate_id"], "label": card["candidate"]["label"],
            "definition": card["candidate"]["definition"], "claims": [{key: claim[key] for key in ("role", "text", "evidence_ids")} for claim in substantive_claims],
            "evidence": [item for item in card["evidence"] if item["evidence_id"] in substantive_ids]})
        evidence = {item["evidence_id"]: item for item in card["evidence"]}
        for claim in card["claims"]:
            if claim["role"] in {"problem", "advantage", "case"}:
                add("claim", {"candidate_id": card["candidate"]["candidate_id"], "claim": {key: claim[key] for key in ("role", "text", "evidence_ids")},
                    "evidence": [evidence[identifier] for identifier in claim["evidence_ids"]]})
    for index, card in enumerate(shown):
        for other in shown[index + 1:]:
            add("duplicate", {label: {key: value["candidate"][key] for key in ("candidate_id", "label", "definition", "synonyms")}
                              for label, value in (("first", card), ("second", other))})
    return tasks


def _budget_for_run(runtime: Any, run_id: str) -> dict[str, Any]:
    # Use the same fixed application-owned maintenance operation as the UI.
    # Scripts cannot submit arbitrary writer callables to Coordinator.
    scope = "run/" + run_id
    report = runtime.budget_status(scope_after=scope[:-1])
    row = next((item for item in report["scopes"] if item["scope_id"] == scope), None)
    needs_review = bool(report["unknown_total"] or report["reconciliation_required"] or report["restore_pending"])
    if row is None:
        if report["next_scope_after"]:
            raise EvaluationError("Budget scope was not found in its bounded page")
        return {"cost_micro": 0, "requires_reconciliation": needs_review, "calls": 0}
    return {"cost_micro": row["used"]["cost_micro"], "requires_reconciliation": needs_review,
            "calls": row["used"]["calls"]}


def _budget_is_blocked(runtime: Any) -> bool:
    status = runtime.budget_status()
    return bool(status["unknown_total"] or status["reconciliation_required"] or status["restore_pending"])


def execute_split(args: argparse.Namespace, suite: dict[str, Any]) -> Path:
    from app.pilot.contracts import CorpusSnapshot
    from app.pilot.encoder import load_spec
    from app.pilot.export import export_result
    from app.pilot.contracts import AnalysisResult
    from app.pilot.methodology import AssessmentArtifact
    from app.pilot.service import PilotService, WORKFLOW_VERSION
    from app.pilot.settings import load_settings, PilotSettings
    from app.runtime.session import credentials

    if not args.execute:
        raise EvaluationError("Execution requires --execute; no sources or models were called")
    cases = cases_for(suite, args.split)
    preselected = args.preselect_confirmed_case or []
    if args.split == "blind" and not args.allow_blind:
        raise EvaluationError("Blind execution requires --allow-blind; preselection does not certify scientific quality")
    if args.split == "blind" and suite.get("protocol_version", 1) >= 2 and suite.get("blind_status") != "sealed_external":
        raise EvaluationError("Published proposed directions are not a sealed independent blind set")
    if preselected and not set(preselected).issubset({case["case_id"] for case in cases}):
        raise EvaluationError("Preselected case is outside the frozen split")
    if type(args.total_cost_micro) is not int or args.total_cost_micro < 0 or not 1 <= args.timeout_seconds <= 7200:
        raise EvaluationError("Explicit nonnegative batch cost and 1–7200 second case timeout required")
    code_root = Path(__file__).resolve().parents[1]
    code_hashes = {str(path.relative_to(code_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((code_root / "app").rglob("*")) if path.is_file() and path.suffix in {".py", ".json"}}
    code_hashes["scripts/evaluate_pilot.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    original = load_settings(args.data_dir)
    output: Path = args.output / args.split / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {"schema_version": 1, "version": VERSION, "created_at": datetime.now(UTC).isoformat(),
        "suite": suite, "suite_hash": digest(suite), "split": args.split, "cases": cases,
        "preselected_confirmed_cases": preselected, "evaluation_mode": "automatic", "workflow_version": WORKFLOW_VERSION,
        "code_hashes": code_hashes, "encoder_spec": load_spec(), "settings": original.model_dump(mode="json"),
        "total_cost_cap_micro": args.total_cost_micro, "currency": original.currency,
        "sampling_version": "top20-retained-plus-next-score-v1; explicit-shown-top-identities-v2; all shown-top pairs",
        "claim_sampling_population": "five hash-selected passports per direction from all archived cards, including outside TOP",
        "legacy_candidate_sampling": "first 15 cards only when top_trend_ids is absent or null",
        "outcomes": [], "tasks": [], "status": "running"}
    write_json(output / "protocol.json", {key: value for key, value in manifest.items() if key not in {"outcomes", "tasks", "status"}})
    runtime = PilotService(args.data_dir, credentials(), model_dir=args.model_dir)
    accounted = 0
    stopped = False
    configured = False
    tasks: list[ReviewTask] = []
    try:
        key_name = "deepseek_api_key" if original.provider == "deepseek" else "yandex_api_key"
        try:
            connected = bool(runtime.credentials.get(key_name))
        except Exception:
            connected = False
        if connected:
            stopped = _budget_is_blocked(runtime)
        for case in cases:
            started = time.monotonic()
            outcome: dict[str, Any] = {**case, "state": "not_run", "run_id": None, "seconds": None,
                "accounted_cost_micro": 0, "requires_reconciliation": False}
            if stopped or not connected or accounted >= args.total_cost_micro:
                outcome["reason"] = "batch_stopped_or_budget_exhausted" if connected else "AI credential required for automatic Russian query; no translation fallback used"
                manifest["outcomes"].append(outcome)
                continue
            run_id = None
            try:
                cap = min(original.run_cost_micro, args.total_cost_micro - accounted)
                settings = PilotSettings.model_validate(original.model_dump() | {"run_cost_micro": cap})
                runtime.configure(settings.model_dump())
                configured = True
                run_id = runtime.start(case["query"])
                outcome["run_id"] = run_id
                while True:
                    row = runtime.get(run_id)
                    if row["state"] not in {"queued", "running"}:
                        break
                    if time.monotonic() - started > args.timeout_seconds:
                        runtime.cancel(run_id)
                        runtime.coordinator.wait(timeout=45)
                        outcome["reason"] = "case_timeout"
                        break
                    time.sleep(0.25)
                row = runtime.get(run_id)
                outcome["state"] = row["state"]
                outcome["reason"] = row.get("error") or outcome.get("reason")
                spending = _budget_for_run(runtime, run_id)
                outcome["accounted_cost_micro"] = spending["cost_micro"]
                outcome["requires_reconciliation"] = spending["requires_reconciliation"]
                accounted += spending["cost_micro"]
                if spending["requires_reconciliation"] or row["state"] in {"queued", "running"}:
                    stopped = True
                if row["state"] == "succeeded":
                    raw = runtime.result(run_id)
                    discovered = runtime.coordinator.checkpoint_value(run_id, "candidates")
                    if not isinstance(discovered, dict):
                        raise EvaluationError("Frozen discovery checkpoint is missing")
                    snapshot = CorpusSnapshot.model_validate(runtime.coordinator.checkpoint_value(run_id, "discovery"))
                    result = AnalysisResult.model_validate(raw["result"])
                    artifacts = tuple(AssessmentArtifact.model_validate(item) for item in raw["assessments"])
                    case_dir = output / case["case_id"]
                    write_json(case_dir / "raw-result.json", raw)
                    write_json(case_dir / "discovery-checkpoint.json", discovered)
                    export_result(case_dir / "sources-and-result.zip", result, runtime.archive, artifacts)
                    shown = shown_top_cards(raw["result"])
                    case_tasks = make_tasks(case, raw, discovered, snapshot, runtime.archive,
                        suite["pairs_per_direction"][args.split], card_count=5)
                    references = {reference.revision_id for saved in result.snapshots for reference in saved.documents}
                    review_corpus = {"documents": [runtime.archive.get(identifier).model_dump(mode="json")
                                                   for identifier in sorted(references)]}
                    write_json(case_dir / "review-corpus.json", review_corpus)
                    outcome.update(numeric_provenance_verified=True,
                        confirmed_count=sum(card.category == "confirmed_trend" for card in result.cards),
                        shown_confirmed_count=sum(card["category"] == "confirmed_trend" for card in shown),
                        truncated_confirmed=sum(card.category == "confirmed_trend" and card.quality != "complete" for card in result.cards),
                        cards_count=len(result.cards), candidate_returned=len(shown),
                        sampled_card_ids=[task.payload["candidate_id"] for task in case_tasks if task.kind == "fabrication"],
                        review_corpus_hash=digest(review_corpus), as_of=result.query_plan.as_of.isoformat(),
                        shown_top_ids=[card["candidate"]["candidate_id"] for card in shown],
                        query_plan_hash=result.query_plan.plan_hash)
                    tasks.extend(case_tasks)
            except Exception as error:
                outcome["state"] = "failed"
                outcome["reason"] = "Evaluation failed: " + type(error).__name__
                # If a dispatched run cannot be accounted for, stop the batch;
                # never treat an unknown paid request as zero cost and continue.
                if run_id:
                    try:
                        runtime.cancel(run_id)
                        runtime.coordinator.wait(timeout=45)
                        spending = _budget_for_run(runtime, run_id)
                        if not outcome["accounted_cost_micro"]:
                            accounted += spending["cost_micro"]
                        outcome["accounted_cost_micro"] = spending["cost_micro"]
                        outcome["requires_reconciliation"] = spending["requires_reconciliation"]
                    except Exception:
                        outcome["accounted_cost_micro"] = None
                        outcome["requires_reconciliation"] = True
                    finally:
                        stopped = True
            finally:
                outcome["seconds"] = time.monotonic() - started
            manifest["outcomes"].append(outcome)
            write_json(output / (case["case_id"] + "-outcome.json"), outcome)
            print(json.dumps({"case_id": case["case_id"], "state": outcome["state"], "seconds": outcome["seconds"]}), flush=True)
    finally:
        try:
            if configured:
                runtime.configure(original.model_dump())
        finally:
            runtime.close()
    # Exactly ceil(25%) of all sampled tasks are selected deterministically for
    # a second named reviewer, before any labels are seen.
    fraction = suite["human_labeling"]["second_reviewer_fraction"]
    double: set[str] = set()
    for kind in _ALLOWED:
        selected_kind = sorted((task for task in tasks if task.kind == kind), key=lambda task: task.task_id)
        double.update(task.task_id for task in selected_kind[:math.ceil(len(selected_kind) * fraction)])
    tasks = [task.model_copy(update={"secondary_required": task.task_id in double}) for task in tasks]
    total_known = all(outcome.get("accounted_cost_micro") is not None for outcome in manifest["outcomes"])
    manifest.update(tasks=[task.model_dump(mode="json") for task in tasks], accounted_cost_micro=accounted if total_known else None,
        status="awaiting_human_labels" if tasks else "not_evaluated")
    write_json(output / "run-manifest.json", manifest)
    write_json(output / "label-vocabulary.json", {"labels_by_task_kind": {key: sorted(values) for key, values in _ALLOWED.items()},
        "phases": ["primary", "secondary", "adjudication"],
        "rule": "Use named human reviewers. Leave unreviewed rows blank. Adjudication follows an actual recorded disagreement."})
    with (output / "human-review.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=("task_id", "case_id", "query", "kind", "rank", "payload", "reviewer_id", "phase", "label"))
        writer.writeheader()
        for task in sorted(tasks, key=lambda item: item.task_id):
            for phase in (("primary", "secondary") if task.secondary_required else ("primary",)):
                writer.writerow({"task_id": task.task_id, "case_id": task.case_id, "query": task.query, "kind": task.kind,
                    "rank": "", "payload": json.dumps(task.payload, ensure_ascii=False), "reviewer_id": "", "phase": phase, "label": ""})
    write_json(output / "quality-unlabeled.json", quality_metrics(manifest, []))
    return output


def load_labels(path: Path) -> list[HumanLabel]:
    if path.suffix.casefold() == ".csv":
        from app.pilot.reports import open_local_regular
        with open_local_regular(path) as binary:
            data = binary.read(MAX_JSON_BYTES + 1)
        if len(data) > MAX_JSON_BYTES:
            raise EvaluationError("Label CSV exceeds 25 MB")
        csv.field_size_limit(2_000_000)
        with io.StringIO(data.decode("utf-8-sig"), newline="") as stream:
            rows = [{key: row[key] for key in ("task_id", "reviewer_id", "phase", "label")}
                    for row in csv.DictReader(stream) if row.get("label", "").strip()]
    else:
        rows = read_json(path)
    if not isinstance(rows, list) or len(rows) > 30000:
        raise EvaluationError("Labels must be a bounded list")
    return [HumanLabel.model_validate(row) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--suite", type=Path, default=Path(__file__).resolve().parents[1] / "tests/evaluation/pilot-v2.json")
    run.add_argument("--split", choices=SPLITS, required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--allow-blind", action="store_true")
    run.add_argument("--preselect-confirmed-case", action="append")
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--model-dir", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--total-cost-micro", type=int, required=True)
    run.add_argument("--timeout-seconds", type=int, default=1800)
    score = commands.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--labels", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    inventory = commands.add_parser("cases")
    inventory.add_argument("--suite", type=Path, default=Path(__file__).resolve().parents[1] / "tests/evaluation/pilot-v2.json")
    inventory.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args()
    try:
        if args.command == "run":
            print(execute_split(args, load_suite(args.suite)))
        elif args.command == "cases":
            print(json.dumps(cases_for(load_suite(args.suite), args.split), ensure_ascii=False, indent=2))
        else:
            manifest = read_json(args.manifest)
            if manifest.get("suite_hash") != digest(manifest["suite"]):
                raise EvaluationError("Frozen suite hash does not match")
            write_json(args.output, quality_metrics(manifest, load_labels(args.labels)))
        return 0
    except (EvaluationError, ValueError, OSError) as error:
        print("Evaluation did not complete: " + type(error).__name__)
        return 2


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
