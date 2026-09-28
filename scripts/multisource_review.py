"""Human-only, blinded review and conservative scoring for multisource cuts.

No application module imports this evaluator. Packet contents come exclusively
from transitive-verified portable artifacts and are regenerated before scoring.
"""

from __future__ import annotations

import csv
import io
import math
import tempfile
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from scripts.evaluate_pilot import EvaluationError, digest, read_json, write_json
from scripts.evaluate_multisource import (ARMS, CutCapture, _contained_file, _sha256,
                                          replay, verified_manifest)

_FIELDS = ("task_id", "phase", "reviewer_id", "concept_specific", "scope_relevant",
           "factual_claims_supported", "not_merely_renamed", "next_step_useful",
           "concept_useful_now", "payload_supported", "critical_unsupported",
           "explanation", "refs", "label_available_at")
_VALUES = {"yes", "no", "unknown"}
_CONCEPT_FIELDS = ("concept_specific", "scope_relevant", "factual_claims_supported",
                   "not_merely_renamed", "next_step_useful", "concept_useful_now")
_PAYLOAD_FIELDS = ("factual_claims_supported", "next_step_useful", "payload_supported",
                   "critical_unsupported")
_OUTCOME_FIELDS = ("area_id", "family_id", "independent_confirmation_180d",
                   "reviewer_id", "label_available_at", "independent_source_ref")


def _csv_rows(path: Path, fields: tuple[str, ...], *, limit: int = 2000) -> list[dict[str, str]]:
    from app.pilot.reports import open_local_regular

    with open_local_regular(path) as handle:
        raw = handle.read(4_000_001)
    if len(raw) > 4_000_000:
        raise EvaluationError("Human-label CSV exceeds 4 MB")
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig", errors="strict")), strict=True)
        if reader.fieldnames != list(fields):
            raise EvaluationError("Human-label CSV columns differ from the frozen template")
        rows = list(reader)
    except (UnicodeError, csv.Error):
        raise EvaluationError("Human-label CSV is malformed") from None
    if len(rows) > limit or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise EvaluationError("Human-label CSV has too many or malformed rows")
    return rows


def _material(manifest: dict[str, Any], capture: CutCapture) -> tuple[dict[tuple[str, str], list[dict[str, Any]]],
                                                                      dict[tuple[str, str, str], dict[str, Any]]]:
    """Extract reviewer evidence from packages, never from untrusted display text."""
    from app.pilot.export import read_result_package
    from app.pilot.multisource.contracts import (CapitalEvent, FundingMetric, SearchMetric,
                                                 SearchObservation, TechnologyConcept)
    from app.pilot.multisource.export import read_signal_package
    from app.runtime.jobs import TaskFailure

    neutral: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    payloads: dict[tuple[str, str, str], dict[str, Any]] = {}
    capture_dir = Path(manifest["capture_path"]).parent
    for case_row, case_capture in zip(manifest["cases"], capture.cases, strict=True):
        if case_row["area_id"] != case_capture.area_id:
            raise EvaluationError("Review case and capture area differ")
        area = case_row["area_id"]
        allowed = set(case_row["pool_family_ids"])
        for output in case_row["outputs"]:
            if output["state"] != "succeeded":
                continue
            artifact = _contained_file(capture_dir, output["artifact"])
            if _sha256(artifact) != output["artifact_sha256"]:
                raise EvaluationError("Review source changed after verified replay")
            ranked = {item["source_id"]: item for item in output["ranked"]}
            if output["arm"] == "C":
                with read_result_package(artifact) as package:
                    for card in package.result.cards:
                        identity = card.candidate.candidate_id
                        family = case_capture.family_map[identity]
                        if family not in allowed:
                            continue
                        evidence = [{"ref": item.revision_id, "url": item.source_url,
                                     "quote": item.quote, "source": item.source}
                                    for item in card.evidence]
                        neutral[(area, family)].append({"label": card.candidate.label,
                            "definition": card.candidate.definition, "evidence": evidence})
                        if identity in ranked:
                            content = {"label": card.candidate.label, "definition": card.candidate.definition,
                                       "claims": [item.model_dump(mode="json") for item in card.claims],
                                       "evidence": evidence, "limitations": list(card.limitations)}
                            key = (area, family, ranked[identity]["payload_hash"])
                            if digest(card.model_dump(mode="json")) != key[2]:
                                raise EvaluationError("Scientific review payload changed after replay")
                            payloads[key] = content
            else:
                with read_signal_package(artifact) as package:
                    concepts = {str(item.concept_id): item for item in (
                        package.store.get_object(ref, TechnologyConcept)
                        for ref in package.profile.concept_artifact_hashes)}
                    findings = {str(item.concept_id): item for item in package.profile.findings}
                    for identity, concept in concepts.items():
                        mapped_family = case_capture.family_map.get(identity)
                        if mapped_family is None or mapped_family not in allowed:
                            continue
                        finding = findings.get(identity)
                        references: list[dict[str, Any]] = []
                        if finding is not None:
                            for ref in finding.observation_hashes[:30]:
                                for kind in (SearchObservation, CapitalEvent):
                                    try:
                                        observation = package.store.get_object(ref, kind)
                                        references.append({"ref": ref, "type": kind.__name__,
                                                           "data": observation.model_dump(mode="json")})
                                        break
                                    except TaskFailure:
                                        continue
                                else:
                                    try:
                                        document = package.archive.get(ref)
                                    except TaskFailure:
                                        references.append({"ref": ref, "type": "other_verified_evidence"})
                                    else:
                                        references.append({"ref": ref, "type": "scientific_document",
                                                           "title": document.title,
                                                           "url": document.url})
                        neutral[(area, mapped_family)].append({"label": concept.label,
                            "definition": concept.definition, "evidence": references,
                            "total_observation_refs": len(finding.observation_hashes) if finding else 0})
                        if output["arm"] != "B" and identity in ranked:
                            if finding is None:
                                raise EvaluationError("Ranked signal has no verified finding")
                            key = (area, mapped_family, ranked[identity]["payload_hash"])
                            if digest(finding.model_dump(mode="json")) != key[2]:
                                raise EvaluationError("Signal review payload changed after replay")
                            metrics = []
                            for ref in finding.metric_hashes:
                                for metric_kind in (SearchMetric, FundingMetric):
                                    try:
                                        metric = package.store.get_object(ref, metric_kind)
                                        metrics.append({"ref": ref, "data": metric.model_dump(mode="json")})
                                        break
                                    except TaskFailure:
                                        continue
                            payloads[key] = {"label": concept.label, "definition": concept.definition,
                                             "finding": finding.model_dump(mode="json"),
                                             "metrics": metrics, "evidence": references,
                                             "evidence_truncated_to": 30}
    return neutral, payloads


def _packets(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest["mode"] != "end-to-end":
        raise EvaluationError("Blinded review packets are based on the end-to-end discovery pool")
    capture = CutCapture.model_validate(read_json(Path(manifest["capture_path"])))
    neutral, payloads = _material(manifest, capture)
    tasks: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        area = case["area_id"]
        for family in case["pool_family_ids"]:
            variants = neutral.get((area, family), [])
            if not variants:
                raise EvaluationError("Review pool contains a concept without a verified evidence packet")
            distinct = {digest(value): value for value in variants}
            tasks.append({"task_id": digest({"capture": manifest["capture_sha256"], "area": area,
                "family": family, "kind": "concept"})[:24], "kind": "concept", "area_id": area,
                "family_id": family, "material": sorted(distinct.values(), key=digest)})
        expected_payloads = {(area, row["family_id"], row["payload_hash"])
            for output in case["outputs"] if output["arm"] != "B" for row in output["ranked"]}
        if not expected_payloads.issubset(payloads):
            raise EvaluationError("Some shown payloads have no independently reviewable material")
        for key in sorted(expected_payloads):
            _, family, payload_hash = key
            tasks.append({"task_id": digest({"capture": manifest["capture_sha256"], "area": area,
                "family": family, "kind": "payload", "payload": payload_hash})[:24],
                "kind": "payload", "area_id": area, "family_id": family,
                "payload_hash": payload_hash, "material": payloads[key]})
    if len(tasks) > 195 or len({item["task_id"] for item in tasks}) != len(tasks):
        raise EvaluationError("Review population exceeds the predeclared per-cut cap or has duplicate IDs")
    seed = manifest["suite_hash"] + manifest["capture_sha256"]
    tasks.sort(key=lambda item: digest({"seed": seed, "task_id": item["task_id"]}))
    by_stratum: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item in tasks:
        by_stratum[(item["area_id"], item["kind"])].append(item["task_id"])
    secondary = sorted(identifier for _, ids in sorted(by_stratum.items())
                       for identifier in sorted(ids, key=lambda value: digest({"second": seed, "id": value}))
                       [:math.ceil(len(ids) * .25)])
    return {"schema_version": 1, "protocol_version": "multisource-evaluation/1.0.0",
            "manifest_hash": digest(manifest), "suite_hash": manifest["suite_hash"],
            "capture_sha256": manifest["capture_sha256"], "cut": manifest["cut"],
            "mode": "end-to-end", "knowledge_cutoff": manifest["knowledge_cutoff"],
            "status": "awaiting_independent_human_labels", "tasks": tasks,
            "secondary_task_ids": secondary,
            "instructions": "Review without arm/rank/score. Enter yes, no or unknown with explanation and evidence refs. "
                            "Secondary reviews must be independent; disagreements need a third reviewer."}


def review_packets(manifest_path: Path, output: Path, *, allow_blind: bool = False) -> Path:
    manifest = verified_manifest(manifest_path, allow_blind=allow_blind)
    packet = _packets(manifest)
    destination = output / "packets.json"
    template = output / "labels.csv"
    if destination.exists() or template.exists():
        raise EvaluationError("Review files already exist; never overwrite human work")
    write_json(destination, packet)
    with template.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_FIELDS)
        writer.writeheader()
        for task in packet["tasks"]:
            writer.writerow({"task_id": task["task_id"], "phase": "primary"})
            if task["task_id"] in packet["secondary_task_ids"]:
                writer.writerow({"task_id": task["task_id"], "phase": "secondary"})
    return destination


def _utc(value: str) -> datetime:
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        raise EvaluationError("Label timestamp is not ISO-8601") from None
    if moment.utcoffset() != timedelta(0) or moment > datetime.now(UTC):
        raise EvaluationError("Human label must have an available UTC timestamp no later than now")
    return moment


def _labels(path: Path, packet: dict[str, Any]) -> tuple[dict[str, dict[str, str | None]], dict[str, int]]:
    tasks = {item["task_id"]: item for item in packet["tasks"]}
    selected = set(packet["secondary_task_ids"])
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in _csv_rows(path, _FIELDS):
        task_id, phase = row["task_id"], row["phase"]
        if task_id not in tasks or phase not in {"primary", "secondary", "adjudication"}:
            raise EvaluationError("Human label refers to an unknown task or phase")
        if phase in grouped[task_id]:
            raise EvaluationError("Duplicate reviewer phase for one task")
        if not row["reviewer_id"].strip():
            if any(row[field].strip() for field in _FIELDS[3:]):
                raise EvaluationError("A filled human label has no reviewer identity")
            continue
        if phase == "secondary" and task_id not in selected:
            raise EvaluationError("Secondary review was not selected before labels were seen")
        needed = _CONCEPT_FIELDS if tasks[task_id]["kind"] == "concept" else _PAYLOAD_FIELDS
        excluded = set(_CONCEPT_FIELDS if tasks[task_id]["kind"] == "payload" else _PAYLOAD_FIELDS) - set(needed)
        if (any(row[field] not in _VALUES for field in needed)
                or any(row[field].strip() for field in excluded)
                or not row["explanation"].strip() or not row["label_available_at"].strip()):
            raise EvaluationError("Human review rubric is incomplete or used for the wrong task kind")
        _utc(row["label_available_at"])
        if any(row[field] != "unknown" for field in needed) and not row["refs"].strip():
            raise EvaluationError("Decisive human labels require evidence references")
        if (tasks[task_id]["kind"] == "concept" and row["concept_useful_now"] == "yes"
                and any(row[field] != "yes" for field in
                        ("concept_specific", "scope_relevant", "not_merely_renamed"))):
            raise EvaluationError("A useful concept must pass specificity, scope and novelty review")
        if (tasks[task_id]["kind"] == "payload" and row["payload_supported"] == "yes"
                and (row["factual_claims_supported"] != "yes" or row["critical_unsupported"] != "no")):
            raise EvaluationError("A supported payload cannot contain unsupported decisive claims")
        grouped[task_id][phase] = row
    resolved: dict[str, dict[str, str | None]] = {}
    stats = {"planned_tasks": len(tasks), "primary": 0, "secondary_selected": len(selected),
             "secondary": 0, "disagreements": 0, "adjudicated": 0, "unresolved_tasks": 0}
    for task_id, task in tasks.items():
        phases = grouped.get(task_id, {})
        first, second, third = (phases.get(phase) for phase in ("primary", "secondary", "adjudication"))
        if (second is not None or third is not None) and first is None:
            raise EvaluationError("Secondary or adjudicated review has no primary review")
        people = [item["reviewer_id"].strip().casefold() for item in (first, second, third) if item is not None]
        if len(set(people)) != len(people):
            raise EvaluationError("Independent review phases require distinct people")
        needed = _CONCEPT_FIELDS if task["kind"] == "concept" else _PAYLOAD_FIELDS
        disagreement = bool(first and second and any(first[field] != second[field] for field in needed))
        if third is not None and (second is None or not disagreement):
            raise EvaluationError("Adjudication requires a recorded disagreement")
        stats["primary"] += first is not None
        stats["secondary"] += second is not None
        stats["disagreements"] += disagreement
        stats["adjudicated"] += third is not None
        chosen = third if disagreement and third is not None else (
            first if first is not None and (task_id not in selected or second is not None)
            and not disagreement else None)
        values = {field: (chosen[field] if chosen and chosen[field] != "unknown" else None)
                  for field in needed}
        if any(value is None for value in values.values()):
            stats["unresolved_tasks"] += 1
        resolved[task_id] = values
    return resolved, stats


def _longitudinal(path: Path | None, packet: dict[str, Any], *, cutoff: datetime) -> dict[str, Any]:
    families = {(task["area_id"], task["family_id"]) for task in packet["tasks"]
                if task["kind"] == "concept"}
    horizon = cutoff + timedelta(days=180)
    matured = datetime.now(UTC) >= horizon
    rows = _csv_rows(path, _OUTCOME_FIELDS) if path is not None else []
    values: dict[tuple[str, str], str] = {}
    for row in rows:
        key = row["area_id"], row["family_id"]
        if key not in families or key in values or row["independent_confirmation_180d"] not in _VALUES:
            raise EvaluationError("Duplicate, unknown or invalid independent outcome")
        if (not row["reviewer_id"].strip() or not row["independent_source_ref"].strip()
                or _utc(row["label_available_at"]) < horizon):
            raise EvaluationError("Outcome needs an independent source and mature available label")
        values[key] = row["independent_confirmation_180d"]
    if rows and not matured:
        raise EvaluationError("180-day outcomes cannot be labeled before horizon maturity")
    return {"status": "censored" if not matured else "pending" if len(values) < len(families)
            or any(value == "unknown" for value in values.values()) else "reviewed",
            "horizon_at": horizon.isoformat(), "concept_families": len(families),
            "censored": len(families) if not matured else 0,
            "known_yes": sum(value == "yes" for value in values.values()) if matured else 0,
            "known_no": sum(value == "no" for value in values.values()) if matured else 0,
            "unknown": len(families) - sum(value in {"yes", "no"} for value in values.values()) if matured else 0}


def score(manifest_path: Path, labels: Path, output: Path, *, outcomes: Path | None = None,
          allow_blind: bool = False) -> Path:
    manifest = verified_manifest(manifest_path, allow_blind=allow_blind)
    if manifest["mode"] == "end-to-end":
        primary_manifest = manifest
    else:
        with tempfile.TemporaryDirectory(prefix="trendanalyser-review-pool-") as temporary:
            path = replay(Path(manifest["suite_path"]), Path(manifest["capture_path"]),
                          manifest["cut"], "end-to-end", Path(temporary), offline=True,
                          allow_blind=allow_blind)
            primary_manifest = read_json(path)
    expected = _packets(primary_manifest)
    packet = read_json(labels.parent / "packets.json")
    if packet != expected:
        raise EvaluationError("Reviewer packet was edited or belongs to another frozen capture")
    values, stats = _labels(labels, packet)
    concepts = {(task["area_id"], task["family_id"]): values[task["task_id"]]["concept_useful_now"]
                for task in packet["tasks"] if task["kind"] == "concept"}
    payloads = {(task["area_id"], task["family_id"], task["payload_hash"]): values[task["task_id"]]
                for task in packet["tasks"] if task["kind"] == "payload"}
    per_area: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        arms = []
        for arm, item in zip(ARMS, case["outputs"], strict=True):
            if item["arm"] != arm:
                raise EvaluationError("Arm output order changed")
            ranked = item["ranked"]
            known = sum(concepts.get((case["area_id"], row["family_id"])) == "yes" for row in ranked)
            unknown = sum(concepts.get((case["area_id"], row["family_id"])) is None for row in ranked)
            succeeded = item["state"] == "succeeded"
            support = [payloads.get((case["area_id"], row["family_id"], row["payload_hash"]))
                       for row in ranked]
            arms.append({"arm": arm, "state": item["state"], "returned_at_5": len(ranked),
                "known_useful": known, "unknown_returned": unknown,
                "useful_slots_at_5": known / 5 if succeeded and not unknown else None,
                "lower_bound": known / 5 if succeeded else 0.0,
                "upper_bound": (known + unknown) / 5 if succeeded else 1.0,
                "payloads_reviewed": sum(value is not None and value.get("payload_supported") is not None
                                         for value in support),
                "payloads_supported": sum(value is not None and value.get("payload_supported") == "yes"
                                          for value in support),
                "payloads_unsupported": sum(value is not None and value.get("payload_supported") == "no"
                                            for value in support),
                "critical_unsupported": sum(value is not None and value.get("critical_unsupported") == "yes"
                                            for value in support),
                "next_steps_useful": sum(value is not None and value.get("next_step_useful") == "yes"
                                         for value in support),
                "payloads_unresolved": sum(value is None or value.get("payload_supported") is None
                                           for value in support),
                "reported_spend_micro": item["reported_spend_micro"],
                "reported_external_calls": item["reported_external_calls"]})
        per_area.append({"area_id": case["area_id"], "arms": arms})
    macro = []
    for index, arm in enumerate(ARMS):
        selected = [case["arms"][index] for case in per_area]
        macro.append({"arm": arm, "planned_areas": len(selected),
            "successful_areas": sum(item["state"] == "succeeded" for item in selected),
            "mean_useful_slots_at_5": sum(item["useful_slots_at_5"] for item in selected) / len(selected)
            if all(item["useful_slots_at_5"] is not None for item in selected) else None,
            "lower_bound": sum(item["lower_bound"] for item in selected) / len(selected),
            "upper_bound": sum(item["upper_bound"] for item in selected) / len(selected)})
    complete = (stats["planned_tasks"] > 0 and stats["primary"] == stats["planned_tasks"]
                and stats["secondary"] == stats["secondary_selected"]
                and stats["unresolved_tasks"] == 0)
    try:
        manifest_file = str(manifest_path.resolve().relative_to(output.parent.resolve()))
        labels_file = str(labels.resolve().relative_to(output.parent.resolve()))
    except ValueError:
        raise EvaluationError("Keep the frozen manifest and labels below the quality report directory") from None
    labeled_at = [_utc(row["label_available_at"]) for row in _csv_rows(labels, _FIELDS)
                  if row["reviewer_id"].strip()]
    review_completed_at = max(labeled_at).isoformat() if complete else None
    report = {"schema_version": 1, "protocol_version": "multisource-evaluation/1.0.0",
              "status": "review_complete" if complete else "not_evaluated" if stats["primary"] == 0
              else "review_incomplete", "manifest_hash": digest(manifest),
              "packet_hash": digest(packet), "cut": manifest["cut"], "mode": manifest["mode"],
              "suite_hash": manifest["suite_hash"], "capture_sha256": manifest["capture_sha256"],
              "manifest_file": manifest_file, "labels_file": labels_file,
              "labels_sha256": _sha256(labels), "review_completed_at": review_completed_at,
              "knowledge_cutoff": manifest["knowledge_cutoff"], "human_R1": stats,
              "L2": _longitudinal(outcomes, packet, cutoff=datetime.fromisoformat(manifest["knowledge_cutoff"])),
              "per_area": per_area, "macro": macro,
              "default_screen_change_eligible": False,
              "limitations": ["Empty TOP slots remain in the denominator; failed cases retain [0,1] bounds.",
                "Concept labels and each distinct shown payload are reviewed separately.",
                "Review identity and independent source truth require human governance; hashes prove only integrity.",
                "A complete review alone does not establish the T1/T2 and timed UX gates.",
                "The scientific desktop view remains the default until those gates are independently met."]}
    write_json(output, report)
    return output


def verify_t1_completion(review_path: Path, report: dict[str, Any]) -> datetime:
    """Recheck T1 reviewer evidence before accepting a frozen T2 policy."""
    if (report.get("schema_version") != 1 or report.get("cut") != "T1"
            or report.get("mode") != "end-to-end" or report.get("status") != "review_complete"
            or not isinstance(report.get("manifest_file"), str)
            or not isinstance(report.get("labels_file"), str)):
        raise EvaluationError("T1 human review is incomplete or not the primary end-to-end review")
    manifest_path = _contained_file(review_path.parent, report["manifest_file"])
    labels_path = _contained_file(review_path.parent, report["labels_file"])
    if _sha256(labels_path) != report.get("labels_sha256"):
        raise EvaluationError("T1 human labels changed after its quality report")
    manifest = verified_manifest(manifest_path)
    packet = _packets(manifest)
    values, stats = _labels(labels_path, packet)
    complete = (stats["planned_tasks"] > 0 and stats["primary"] == stats["planned_tasks"]
                and stats["secondary"] == stats["secondary_selected"]
                and stats["unresolved_tasks"] == 0
                and all(value is not None for fields in values.values() for value in fields.values()))
    moments = [_utc(row["label_available_at"]) for row in _csv_rows(labels_path, _FIELDS)
               if row["reviewer_id"].strip()]
    if (not complete or not moments or report.get("review_completed_at") != max(moments).isoformat()
            or report.get("manifest_hash") != digest(manifest)
            or report.get("packet_hash") != digest(packet)
            or report.get("suite_hash") != manifest["suite_hash"]
            or report.get("capture_sha256") != manifest["capture_sha256"]
            or report.get("knowledge_cutoff") != manifest["knowledge_cutoff"]
            or report.get("human_R1") != stats):
        raise EvaluationError("T1 completion evidence does not reproduce from its human labels")
    return max(moments)
