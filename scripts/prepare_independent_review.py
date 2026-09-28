"""Offline, private-map preparation and scoring of external paired human review.

No model, credential store, API, or production service is instantiated. The
custodian keeps private/ and supplies only reviewer/ to independent humans.
See docs/methodology/independent-validation.md for the sealing protocol.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime
import hashlib
import hmac
import io
import json
import math
from pathlib import Path
import random
import re
from typing import Any

from app.backend.contracts import DocumentRecord
from app.pilot.reports import open_local_regular
from scripts.evaluate_pilot import (
    MAX_JSON_BYTES, SPLITS, EvaluationError, HumanLabel, ReviewTask, _ALLOWED,
    cases_for, digest, load_suite, quality_metrics, read_json, resolve_labels, write_json,
)

VERSION = "independent-pilot-review/1.0.0"
COLUMNS = ("task_id", "query", "as_of", "kind", "payload", "phase", "reviewer_id", "label")
IMMUTABLE = COLUMNS[:6]


def file_sha(path: Path) -> str:
    with open_local_regular(path) as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise EvaluationError("Input exceeds 25 MB")
    return hashlib.sha256(raw).hexdigest()


def inventory_template(suite: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "suite_hash": digest(suite), "status": "pending_external",
        "custodian_id": "", "sealed_at": None, "exposure_inventory_complete": False,
        "prior_exposure_corpora": [],
        "corpora": [{**case, "split": split, "path": None, "sha256": None, "families": []}
                    for split in SPLITS for case in cases_for(suite, split)]}


def _timestamp(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError) as error:
        raise EvaluationError("A frozen timestamp with timezone is required") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EvaluationError("A frozen timestamp with timezone is required")
    return parsed


def family_preflight(suite: dict[str, Any], inventory: dict[str, Any], base: Path) -> dict[str, Any]:
    """Check every frozen revision and declared alias, including exposed corpora.

    Family assignment and completeness of the exposure list are external
    attestations: hashing cannot establish publication equivalence or secrecy.
    """
    if inventory.get("schema_version") != 1 or inventory.get("suite_hash") != digest(suite):
        raise EvaluationError("Inventory must bind the exact frozen suite")
    if inventory.get("exposure_inventory_complete") is not True:
        raise EvaluationError("The custodian must inventory all prior exposed corpora")
    expected = {case["case_id"]: {**case, "split": split} for split in SPLITS for case in cases_for(suite, split)}
    corpora, exposed = inventory.get("corpora"), inventory.get("prior_exposure_corpora")
    if not isinstance(corpora, list) or not isinstance(exposed, list) or not exposed or len(corpora + exposed) > 1000:
        raise EvaluationError("Bounded corpus inventory and explicit prior exposure corpora required")
    seen_cases: set[str] = set()
    seen_variants: set[tuple[str, str]] = set()
    partitions: dict[str, tuple[str, str]] = {}
    bindings: list[dict[str, Any]] = []
    files: list[dict[str, str]] = []
    revision_count = 0
    for row in corpora + exposed:
        is_exposed = row in exposed
        case_id = row.get("case_id")
        split = "development" if is_exposed else row.get("split")
        if not is_exposed:
            if case_id not in expected or {key: row.get(key) for key in ("case_id", "query", "split")} != expected[case_id]:
                raise EvaluationError("Unknown or changed corpus case/split/query")
            variant = row.get("variant_id", "shared")
            if not isinstance(variant, str) or not variant.strip() or (case_id, variant) in seen_variants:
                raise EvaluationError("Duplicate case/variant corpus inventory")
            if case_id in seen_cases and (variant == "shared" or (case_id, "shared") in seen_variants):
                raise EvaluationError("Duplicate cases require distinct explicit corpus variant_id values")
            seen_variants.add((case_id, variant))
            seen_cases.add(case_id)
        elif not isinstance(case_id, str) or not case_id.startswith("exposure-"):
            raise EvaluationError("Prior exposures require an exposure-* case_id")
        path_value = row.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise EvaluationError("Every corpus needs an existing read-only path")
        path = base / path_value
        actual_sha = file_sha(path)
        if actual_sha != row.get("sha256"):
            raise EvaluationError("Corpus SHA-256 mismatch")
        raw = read_json(path)
        if not isinstance(raw, dict) or not isinstance(raw.get("documents"), list) or not 1 <= len(raw["documents"]) <= 30000:
            raise EvaluationError("Corpus requires 1–30000 normalized documents")
        docs = [DocumentRecord.model_validate(item) for item in raw["documents"]]
        revisions = {digest(document.model_dump(mode="json")): document for document in docs}
        if len(revisions) != len(docs):
            raise EvaluationError("Duplicate corpus revisions")
        families = row.get("families")
        if not isinstance(families, list) or not families or len(families) > len(docs):
            raise EvaluationError("Every corpus revision needs an independently assigned family")
        assigned: set[str] = set()
        for family in families:
            family_id, revision_ids, aliases = (family.get(key) for key in ("family_id", "revision_ids", "identity_keys"))
            if (not isinstance(family_id, str) or not family_id.strip() or not isinstance(revision_ids, list)
                    or not revision_ids or len(revision_ids) != len(set(revision_ids))
                    or not isinstance(aliases, list) or not aliases
                    or any(not isinstance(alias, str) or not alias.strip() for alias in aliases)):
                raise EvaluationError("Invalid family identity/revision/alias mapping")
            normalized_aliases = {alias.strip().casefold() for alias in aliases}
            for revision in revision_ids:
                if revision not in revisions or revision in assigned:
                    raise EvaluationError("Each corpus revision must belong to exactly one family")
                document = revisions[revision]
                required = {document.document_key.casefold(), f"{document.source}:{document.source_id}".casefold()}
                if document.patent_family_id:
                    required.add("patent-family:" + document.patent_family_id.casefold())
                if not required <= normalized_aliases:
                    raise EvaluationError("Family aliases omit an actual DOI/source/patent identity")
                assigned.add(revision)
            keys = {"family:" + family_id, *("alias:" + alias for alias in normalized_aliases),
                    *("revision:" + revision for revision in revision_ids)}
            for key in keys:
                previous = partitions.get(key)
                if previous and previous != (split, family_id):
                    raise EvaluationError("Study family/alias/revision crosses splits or has inconsistent family IDs")
                partitions[key] = (split, family_id)
        if assigned != set(revisions):
            raise EvaluationError("Family inventory omits corpus revisions")
        revision_count += len(revisions)
        if revision_count > 1_000_000:
            raise EvaluationError("Combined inventory exceeds one million revisions")
        files.append({"path": str(path.resolve()), "sha256": actual_sha})
        if not is_exposed:
            bindings.append({"case_id": case_id, "corpus_hash": digest(raw), "revision_ids": sorted(revisions), "path": str(path.resolve())})
    if seen_cases != set(expected):
        raise EvaluationError("All development, validation and blind cases need frozen corpus inventories")
    sealed = (inventory.get("status") == "sealed_external" and bool(str(inventory.get("custodian_id", "")).strip())
              and suite.get("blind_status") == "sealed_external")
    if sealed:
        sealed = _timestamp(suite.get("frozen_at")) <= _timestamp(inventory.get("sealed_at"))
    return {"family_disjoint": True, "externally_sealed": bool(sealed), "bindings": bindings,
        "files": files, "revision_count": revision_count, "inventory_hash": digest(inventory),
        "limitation": "Declared family equivalence, external independence and exposure completeness require human custody; software validates references and declared separation only."}


def visible_payload(task: ReviewTask, documents: dict[str, Any] | None = None) -> dict[str, Any]:
    """Whitelist scientific content, never recursively copy model output metadata."""
    source = task.payload
    if task.kind == "pair":
        return {key: source[key] for key in ("title", "abstract", "abstract_truncated", "source_url") if key in source}
    if task.kind == "duplicate":
        pair = [{key: source[side][key] for key in ("label", "definition", "synonyms") if key in source[side]}
                for side in ("first", "second")]
        pair.sort(key=digest)
        return dict(zip(("first", "second"), pair, strict=True))
    evidence = source.get("evidence", [])
    by_id: dict[str, str] = {}
    visible: dict[str, dict[str, Any]] = {}
    for item in evidence:
        text = {key: item[key] for key in ("source_url", "quote", "text_field", "retrieved_at") if key in item}
        if documents is not None:
            document = documents[item["revision_id"]]
            text["source_document"] = {key: document[key] for key in (
                "title", "abstract", "authors", "publication_year", "publication_month", "publication_date", "date_precision", "url")}
        identifier = "E-" + digest(text)[:24]
        if item["evidence_id"] in by_id:
            raise EvaluationError("Duplicate evidence ID")
        by_id[item["evidence_id"]] = identifier
        visible[identifier] = {"reference": identifier, **text}

    def claim(value):
        refs = value.get("evidence_ids", [])
        if any(ref not in by_id for ref in refs):
            raise EvaluationError("Claim refers to absent reviewer evidence")
        return {"role": value["role"], "text": value["text"], "references": sorted(by_id[ref] for ref in refs)}

    result: dict[str, Any] = {key: source[key] for key in ("label", "definition", "synonyms", "rubric") if key in source}
    if "claim" in source:
        result["claim"] = claim(source["claim"])
    if "claims" in source:
        result["claims"] = [claim(value) for value in source["claims"]]
    if "evidence" in source:
        result["evidence"] = [visible[key] for key in sorted(visible)]
    return result


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    raw = stream.getvalue().encode("utf-8-sig")
    if len(raw) > MAX_JSON_BYTES:
        raise EvaluationError("Reviewer CSV exceeds 25 MB")
    with path.open("xb") as output:
        output.write(raw)


def _cell(value: str) -> str:
    # CSV is often opened in spreadsheet software; keep content inert.
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


def prepare(baseline: Path, corrected: Path, inventory_path: Path, output: Path, seed: str) -> Path:
    if output.exists() or output.is_symlink():
        raise EvaluationError("Output directory already exists; refusing overwrite")
    if not re.fullmatch(r"[a-f0-9]{64}", seed):
        raise EvaluationError("Private seed must be exactly 64 lowercase hexadecimal characters")
    manifests = {arm: read_json(path) for arm, path in (("baseline", baseline), ("corrected", corrected))}
    first, second = manifests.values()
    for manifest in manifests.values():
        quality_metrics(manifest, [])
        if not manifest.get("code_hashes") or not manifest.get("encoder_spec") or not manifest.get("workflow_version"):
            raise EvaluationError("Both runs require frozen code, workflow and model provenance")
    if (first["suite_hash"], first["split"], first["cases"], first.get("evaluation_mode", "automatic")) != (
            second["suite_hash"], second["split"], second["cases"], second.get("evaluation_mode", "automatic")):
        raise EvaluationError("Paired runs must share the frozen suite, split, cases and evaluation mode")
    inventory = read_json(inventory_path)
    preflight = family_preflight(first["suite"], inventory, inventory_path.parent)
    bindings = {(item["case_id"], item["corpus_hash"]): set(item["revision_ids"]) for item in preflight["bindings"]}
    source_files = [{"path": str(path.resolve()), "sha256": file_sha(path)} for path in (baseline, corrected, inventory_path)]
    rng = random.Random(seed)
    key = bytes.fromhex(seed)
    merged: dict[str, dict[str, Any]] = {}
    mapping: dict[str, list[dict[str, str]]] = {}
    arm_ids: dict[str, dict[str, list[str]]] = {}
    documents = {}
    for item in preflight["bindings"]:
        normalized = [DocumentRecord.model_validate(raw).model_dump(mode="json") for raw in read_json(Path(item["path"]))["documents"]]
        documents.update({digest(raw): raw for raw in normalized})
    dates: dict[str, str] = {}
    for arm, manifest in manifests.items():
        outcomes = {item["case_id"]: item for item in manifest["outcomes"]}
        arm_ids[arm] = {}
        for task_raw in manifest["tasks"]:
            task = ReviewTask.model_validate(task_raw)
            outcome = outcomes[task.case_id]
            corpus = bindings.get((task.case_id, outcome.get("review_corpus_hash")))
            if corpus is None or not outcome.get("as_of"):
                raise EvaluationError("Every review task must bind a frozen corpus and as_of date")
            if task.case_id in dates and dates[task.case_id] != outcome["as_of"]:
                raise EvaluationError("Paired cases have different as_of dates")
            dates[task.case_id] = outcome["as_of"]
            references = [item.get("revision_id") for item in task.payload.get("evidence", [])]
            if task.kind == "pair":
                references.append(task.payload.get("revision_id"))
            if any(ref not in corpus for ref in references):
                raise EvaluationError("Review source reference is absent from the bound corpus")
            public = {"query": task.query, "as_of": outcome["as_of"], "kind": task.kind, "payload": visible_payload(task, documents)}
            identifier = "R-" + hmac.new(key, digest(public).encode(), hashlib.sha256).hexdigest()[:32]
            merged[identifier] = {"task_id": identifier, **public}
            mapping.setdefault(identifier, []).append({"arm": arm, "task_id": task.task_id})
            arm_ids[arm].setdefault(task.kind, []).append(identifier)
    if not merged or len(merged) > 30000:
        raise EvaluationError("A packet requires 1–30000 real review tasks")
    double: set[str] = set()
    fraction = max(0.25, first["suite"]["human_labeling"]["second_reviewer_fraction"])
    for arm in sorted(arm_ids):
        for _kind, identifiers in sorted(arm_ids[arm].items()):
            ordered = sorted(identifiers)
            rng.shuffle(ordered)
            double.update(ordered[:math.ceil(len(ordered) * fraction)])
    order = sorted(merged)
    rng.shuffle(order)
    rows = []
    for identifier in order:
        public_task = merged[identifier]
        for phase in (("primary", "secondary") if identifier in double else ("primary",)):
            rows.append({"task_id": identifier, "query": _cell(public_task["query"]), "as_of": public_task["as_of"],
                "kind": public_task["kind"], "payload": json.dumps(public_task["payload"], ensure_ascii=False, sort_keys=True),
                "phase": phase, "reviewer_id": "", "label": ""})
    adjusted = deepcopy(manifests)
    selected_original = {(item["arm"], item["task_id"]) for identifier in double for item in mapping[identifier]}
    for arm, manifest in adjusted.items():
        for task in manifest["tasks"]:
            task["secondary_required"] = (arm, task["task_id"]) in selected_original
    private = {"schema_version": 1, "version": VERSION, "seed": seed, "source_files": source_files,
        "preflight": preflight, "custodian_id": inventory.get("custodian_id"), "mapping": mapping,
        "original_manifest_hashes": {arm: digest(manifest) for arm, manifest in manifests.items()},
        "manifests": adjusted, "rows": rows,
        "double_review": {arm: {kind: {"tasks": len(ids), "secondary": sum(identifier in double for identifier in ids)}
            for kind, ids in kinds.items()} for arm, kinds in arm_ids.items()},
        "status": "awaiting_external_humans"}
    # Recheck every file used after all reads, before publishing a review packet.
    for item in source_files + preflight["files"]:
        if file_sha(Path(item["path"])) != item["sha256"]:
            raise EvaluationError("Input changed during preparation")
    output.mkdir(parents=True, exist_ok=False)
    (output / "private").mkdir(mode=0o700)
    (output / "reviewer").mkdir()
    write_json(output / "private/manifest.json", private)
    write_json(output / "reviewer/vocabulary.json", {kind: sorted(labels) for kind, labels in _ALLOWED.items()})
    write_csv(output / "reviewer/primary.csv", [row for row in rows if row["phase"] == "primary"])
    secondary = [row for row in rows if row["phase"] == "secondary"]
    rng.shuffle(secondary)
    write_csv(output / "reviewer/secondary.csv", secondary)
    return output


def score(packet: Path, answers: list[Path], roster_path: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise EvaluationError("Score output exists; refusing overwrite")
    private = read_json(packet)
    if private.get("version") != VERSION:
        raise EvaluationError("Unknown private review packet")
    for item in private["source_files"] + private["preflight"]["files"]:
        if file_sha(Path(item["path"])) != item["sha256"]:
            raise EvaluationError("Frozen input changed after packet preparation")
    roster = read_json(roster_path)
    if roster.get("packet_sha256") != file_sha(packet):
        raise EvaluationError("Roster must attest the exact private packet SHA-256")
    reviewers = roster.get("reviewers", [])
    by_reviewer = {item.get("reviewer_id", "").strip().casefold(): item for item in reviewers}
    if len(by_reviewer) != len(reviewers) or "" in by_reviewer:
        raise EvaluationError("Reviewer roster IDs must be unique named humans")
    expected = {(row["task_id"], row["phase"]): row for row in private["rows"]}
    seen: set[tuple[str, str]] = set()
    labels = []
    for path in answers:
        with open_local_regular(path) as stream:
            raw = stream.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise EvaluationError("Answer CSV exceeds 25 MB")
        csv.field_size_limit(2_000_000)
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
        if reader.fieldnames != list(COLUMNS):
            raise EvaluationError("Answer CSV columns changed")
        for row in reader:
            if len(seen) >= 60000:
                raise EvaluationError("Too many answer rows")
            identity = (row["task_id"], row["phase"])
            original = expected.get(identity)
            if row["phase"] == "adjudication":
                original = expected.get((row["task_id"], "primary"))
            if original is None or identity in seen or any(row[key] != original[key] for key in IMMUTABLE if key != "phase"):
                raise EvaluationError("Unknown, duplicate or modified read-only review row")
            seen.add(identity)
            if not row["label"].strip():
                continue
            reviewer = by_reviewer.get(row["reviewer_id"].strip().casefold())
            if (reviewer is None or reviewer.get("external_human") is not True
                    or reviewer.get("independent_of_implementation") is not True
                    or not reviewer.get("subject_expertise", "").strip()
                    or row["phase"] not in reviewer.get("phases", [])
                    or row["reviewer_id"].strip().casefold() == str(private.get("custodian_id", "")).strip().casefold()):
                raise EvaluationError("Labels require an attested external subject expert distinct from the custodian")
            labels.append(HumanLabel.model_validate({key: row[key] for key in ("task_id", "phase", "reviewer_id", "label")}))
    public_tasks = [ReviewTask(task_id=row["task_id"], case_id="private", query=row["query"], kind=row["kind"],
                              secondary_required=(row["task_id"], "secondary") in expected, payload={})
                    for row in private["rows"] if row["phase"] == "primary"]
    resolved, statistics = resolve_labels(public_tasks, labels)
    mapped: dict[str, list[HumanLabel]] = {"baseline": [], "corrected": []}
    for label in labels:
        for item in private["mapping"][label.task_id]:
            mapped[item["arm"]].append(label.model_copy(update={"task_id": item["task_id"]}))
    metrics = {arm: quality_metrics(manifest, mapped[arm]) for arm, manifest in private["manifests"].items()}
    conflict_ids = {label.task_id for label in labels if label.phase == "secondary"}
    conflicts = []
    for identifier in conflict_ids:
        values = {label.phase: label.label for label in labels if label.task_id == identifier}
        if values.get("primary") != values.get("secondary") and "adjudication" not in values:
            conflicts.append({**expected[(identifier, "primary")], "phase": "adjudication"})
    gates = {"external_protocol_sealed": private["preflight"]["externally_sealed"],
        "family_disjoint": private["preflight"]["family_disjoint"],
        "blind_split": all(manifest["split"] == "blind" for manifest in private["manifests"].values()),
        "roster_attested": roster.get("status") == "attested_external" and bool(roster.get("attested_at"))
            and roster.get("blinded_before_primary_submission") is True and roster.get("label_origin_human_only") is True,
        "all_review_labels_resolved": bool(resolved) and all(value is not None for value in resolved.values()),
        "corrected_quality_gates": metrics["corrected"]["status"] == "passed"}
    if gates["roster_attested"]:
        _timestamp(roster["attested_at"])
    gates["protocol_frozen_before_runs"] = private["preflight"]["externally_sealed"] and all(
        _timestamp(manifest["suite"].get("frozen_at")) <= _timestamp(manifest.get("created_at"))
        for manifest in private["manifests"].values())
    deltas = {}
    for key in ("emerging_precision_macro", "weak_signal_precision_macro", "emerging_yield_macro", "scope_precision", "specificity_precision"):
        before, after = (metrics[arm]["candidate_metrics"][key] for arm in ("baseline", "corrected"))
        deltas[key] = after - before if before is not None and after is not None else None
    result = {"version": VERSION, "packet_sha256": file_sha(packet), "roster_sha256": file_sha(roster_path),
        "answer_files": [{"sha256": file_sha(path), "name": path.name} for path in answers],
        "status": "passed" if all(gates.values()) else "not_accepted", "gates": gates,
        "labeling": statistics, "arms": metrics,
        "same_corpus_by_case": {case["case_id"]: len({outcome.get("review_corpus_hash")
            for manifest in private["manifests"].values() for outcome in manifest["outcomes"]
            if outcome["case_id"] == case["case_id"]}) == 1 for case in private["manifests"]["baseline"]["cases"]}, "corrected_minus_baseline": deltas,
        "limitations": ["No automatic or AI human labels; blank, uncertain and unresolved labels cannot pass.",
            "Difference is descriptive, without confidence intervals or proof of recall/forecast accuracy.",
            "External identity and the absence of unrecorded exposure are custodian attestations, not software-verifiable facts.",
            "Acceptance applies to corrected absolute frozen gates, not merely improvement over baseline."]}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "comparison.json", result)
    write_csv(output / "adjudication.csv", sorted(conflicts, key=lambda row: row["task_id"]))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template")
    template.add_argument("--suite", type=Path, required=True)
    template.add_argument("--output", type=Path, required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--suite", type=Path, required=True)
    preflight.add_argument("--inventory", type=Path, required=True)
    prep = commands.add_parser("prepare")
    for option in ("baseline", "corrected", "inventory", "output"):
        prep.add_argument("--" + option, type=Path, required=True)
    prep.add_argument("--seed", required=True)
    scoring = commands.add_parser("score")
    for option in ("packet", "roster", "output"):
        scoring.add_argument("--" + option, type=Path, required=True)
    scoring.add_argument("--answers", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "template":
            write_json(args.output, inventory_template(load_suite(args.suite)))
        elif args.command == "preflight":
            result = family_preflight(load_suite(args.suite), read_json(args.inventory), args.inventory.parent)
            print(json.dumps({key: result[key] for key in ("family_disjoint", "externally_sealed", "revision_count")}))
        elif args.command == "prepare":
            prepare(args.baseline, args.corrected, args.inventory, args.output, args.seed)
        else:
            result = score(args.packet, args.answers, args.roster, args.output)
            print(json.dumps({"status": result["status"], "gates": result["gates"]}))
            return 0 if result["status"] == "passed" else 2
    except (EvaluationError, ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, f"Review preparation/scoring failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
