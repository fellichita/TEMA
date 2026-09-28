"""Replay real production discovery locally, without downloads, AI or user keys.

Usage: python -m tools.verify_pilot_runtime --input snapshot.json --model-dir MODEL
       --output-dir NEW_DIRECTORY [--baseline-result discovery.json]

Only discovery.task is executed, through the production process boundary. Saved
documents are replay data, not a new search. A successful run measures execution
and provenance; it does not measure scientific accuracy. Cancellation accepts
SIGINT/SIGTERM or creation of --cancel-file. Results are never overwritten.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
from threading import Event
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.runtime.credentials import CredentialStore, LEGACY_ENVIRONMENT
from app.runtime.worker import WorkerCancelled, WorkerError, WorkerTimeout, run_in_process

VERSION = "pilot-runtime-verifier/2"
PROJECT = Path(__file__).resolve().parents[1]
MAX_INPUT_BYTES = 100_000_000
MAX_RESULT_BYTES = 25_000_000
STATUS_DISCOVERY_VERSION = "semantic-discovery-v5-publication-status"
# v6 keeps the v5 study identity and publication-status evidence; it only adds
# title-phrase groups to the hierarchy and the candidate list.
STATUS_DISCOVERY_VERSIONS = {STATUS_DISCOVERY_VERSION, "semantic-discovery-v6-title-phrase-groups"}
KNOWN_VERSIONS = {"semantic-discovery-v1", "semantic-discovery-v3", "semantic-discovery-v4-primary-units",
                  *STATUS_DISCOVERY_VERSIONS}
RUNTIME_ENVIRONMENT = {
    "PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "COMSPEC", "APPDATA", "LOCALAPPDATA",
    "TEMP", "TMP", "TMPDIR", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ", "PYTHONIOENCODING", "PYTHONUTF8",
    "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
}


class VerificationError(ValueError):
    """Value-free validation error; never format document contents or secrets."""


def scrub_environment():
    """Use the diagnostic environment policy, plus the application's credential map."""
    from tools.runtime_environment import diagnostic_environment

    # A runtime allowlist also excludes credential-bearing proxy URLs, custom
    # service credentials and user authentication sockets outside the API map.
    clean = {name: value for name, value in diagnostic_environment().items() if name.upper() in RUNTIME_ENVIRONMENT}
    for name in LEGACY_ENVIRONMENT:
        clean.pop(name, None)
    clean.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", ORT_DISABLE_TELEMETRY="1", PYTHONNOUSERSITE="1")
    os.environ.clear()
    os.environ.update(clean)


class IsolatedCredentialStore(CredentialStore):
    """No Keychain backend and no imported credentials, including before spawn."""

    def __init__(self):
        scrub_environment()
        super().__init__()
        # The production bootstrap now sees an empty credential environment.
        # Its later spawn-time call removes newly added names without importing.
        self.import_legacy_environment()

    def _load_backend(self):
        self._check_owner()
        return None


class Cancellation:
    def __init__(self, cancel_file=None):
        self.event = Event()
        self.cancel_file = Path(cancel_file) if cancel_file else None

    def set(self):
        self.event.set()

    def is_set(self):
        return self.event.is_set() or self.cancel_file is not None and self.cancel_file.exists()


@contextmanager
def cancellation_signals(cancel):
    previous = {}
    try:
        for name in (signal.SIGINT, signal.SIGTERM):
            previous[name] = signal.signal(name, lambda *_: cancel.set())
        yield
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError("Duplicate JSON object key")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise VerificationError("Non-finite JSON number")


def read_json(path, maximum=MAX_INPUT_BYTES):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise VerificationError("Input must be an existing regular file, not a symlink")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if not 0 < len(raw) <= maximum:
        raise VerificationError("JSON file is empty or exceeds its byte limit")
    try:
        value = json.loads(raw, object_pairs_hook=_unique, parse_constant=_invalid_constant)
    except (UnicodeError, RecursionError, json.JSONDecodeError):
        raise VerificationError("Invalid JSON encoding or nesting") from None
    pending, nodes = [iter([value])], 0
    while pending:
        try:
            item = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        nodes += 1
        if len(pending) > 64 or nodes > 1_000_000:
            raise VerificationError("JSON exceeds structural limits")
        if isinstance(item, float) and not math.isfinite(item):
            raise VerificationError("Non-finite JSON number")
        if isinstance(item, dict):
            pending.append(iter(item.values()))
        elif isinstance(item, list):
            pending.append(iter(item))
    if not isinstance(value, dict):
        raise VerificationError("JSON root must be an object")
    return value, raw


def validate_input(value):
    from app.backend.contracts import DocumentRecord
    from app.pilot.contracts import QueryPlan
    from app.pilot.discovery import DiscoveryOptions

    if set(value) - {"query_plan", "documents", "discovery_snapshot_id", "model_dir", "cache_dir", "options"}:
        raise VerificationError("Unknown discovery input fields")
    if not isinstance(value.get("documents"), list) or not 1 <= len(value["documents"]) <= 10000:
        raise VerificationError("Input requires 1–10000 document records")
    if not isinstance(value.get("discovery_snapshot_id"), str) or not re.fullmatch(r"[0-9a-f]{64}", value["discovery_snapshot_id"]):
        raise VerificationError("Input requires a SHA-256 discovery snapshot identifier")
    try:
        plan = QueryPlan.model_validate(value["query_plan"])
        for document in value["documents"]:
            DocumentRecord.model_validate(document)
        options = DiscoveryOptions(**value.get("options", {}))
    except (KeyError, TypeError, ValueError):
        raise VerificationError("Invalid production query, document or discovery options") from None
    return plan, asdict(options)


def claim_output_directory(path, model_dir):
    path, model_dir = Path(path).absolute(), Path(model_dir).resolve()
    if path.resolve().is_relative_to(model_dir):
        raise VerificationError("Output cannot be inside the read-only model directory")
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        raise VerificationError("Output directory already exists; choose a new directory") from None
    return path.resolve()


def write_new(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > MAX_INPUT_BYTES:
        raise VerificationError("Audit artifact exceeds its byte limit")
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _rows(result, key):
    rows = result.get(key)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise VerificationError("Unknown discovery result structure")
    return rows


def publication_status_evidence(result, documents=None):
    """Validate v5 status links, including notices targeting absent papers."""
    if result.get("discovery_version") not in STATUS_DISCOVERY_VERSIONS:
        if "publication_status_version" in result or "status_evidence" in result:
            raise VerificationError("Mixed publication-status and legacy result formats")
        return []
    if result.get("publication_status_version") != "publication-status/2.0.0":
        raise VerificationError("Unsupported publication-status rule version")
    rows = _rows(result, "status_evidence")
    documents_by_hash = None
    if documents is not None:
        from app.backend.contracts import DocumentRecord
        from app.pilot.contracts import content_hash

        records = [DocumentRecord.model_validate(raw) for raw in documents]
        documents_by_hash = {content_hash(record): record for record in records}
    seen = set()
    for row in rows:
        base_fields = {"kind", "source_key", "target_key", "source", "source_id", "revision_id"}
        kind = row.get("kind")
        if kind not in {"explicit_publication_status", "publisher_retraction_notice"}:
            raise VerificationError("Unknown publication-status evidence kind")
        allowed = base_fields | ({"metadata_path", "status"} if kind == "publisher_retraction_notice" else set())
        if (set(row) != allowed or any(not isinstance(row[key], str) or not row[key] for key in allowed)
                or row["source"] not in {"openalex", "crossref", "arxiv", "epo", "report"}
                or not re.fullmatch(r"[a-f0-9]{64}", row["revision_id"])):
            raise VerificationError("Invalid publication-status evidence structure")
        if kind == "publisher_retraction_notice" and (
                not re.fullmatch(r"update-to\[[0-9]+\]", row["metadata_path"])
                or row["status"] not in {"retraction", "retraction-notice", "retracted-article", "withdrawal"}):
            raise VerificationError("Invalid publisher-notice evidence")
        digest = json.dumps(row, sort_keys=True)
        if digest in seen:
            raise VerificationError("Duplicate publication-status evidence")
        seen.add(digest)
        if documents_by_hash is not None:
            from app.pilot.retractions import retraction_status_evidence

            document = documents_by_hash.get(row["revision_id"])
            expected = {key: value for key, value in row.items() if key != "revision_id"}
            if (document is None or expected not in retraction_status_evidence(
                    document, rules_version="publication-status/2.0.0")):
                raise VerificationError("Publication-status evidence does not match its input revision")
    return rows


def study_trace(result, documents=None):
    """Normalize known emitted shapes; never equate v1 orphan IDs with a queue."""
    if not isinstance(result, dict):
        raise VerificationError("Discovery result must be an object")
    version = result.get("discovery_version")
    if result.get("schema_version") != 3 or version not in KNOWN_VERSIONS:
        raise VerificationError("Unsupported discovery result version")
    status_evidence = publication_status_evidence(result, documents)
    studies, relevance = _rows(result, "studies"), _rows(result, "relevance")
    candidates, clusters = _rows(result, "candidates"), _rows(result, "clusters")
    excluded = _rows(result, "excluded_studies")
    if version == "semantic-discovery-v1":
        if "review_queue" in result or "unassigned_study_ids" in result:
            raise VerificationError("Mixed v1 and review-queue result formats")
        queue = []
        unassigned = result.get("early_signal_study_ids")
    else:
        queue = _rows(result, "review_queue")
        _rows(result, "review_queue_metadata")
        _rows(result, "hierarchy")
        unassigned = result.get("unassigned_study_ids")
    if not isinstance(unassigned, list) or any(not isinstance(sid, str) for sid in unassigned):
        raise VerificationError("Missing unassigned-study list")
    by_id, scores = {}, {}
    for collection, target in ((studies, by_id), (relevance, scores)):
        for row in collection:
            sid = row.get("study_id")
            if not isinstance(sid, str) or not sid or sid in target:
                raise VerificationError("Missing or duplicate study identity")
            target[sid] = row
    if version in STATUS_DISCOVERY_VERSIONS and any(
            row.get("identity_version") != "study-families-v2-publication-units"
            or not isinstance(row.get("identity_keys"), list)
            or any(not isinstance(key, str) or not key for key in row["identity_keys"])
            or len(row["identity_keys"]) != len(set(row["identity_keys"]))
            or row["study_id"] not in row["identity_keys"]
            for row in studies):
        raise VerificationError("Unsupported or missing v5 study-identity rule")
    if version in STATUS_DISCOVERY_VERSIONS:
        for study in studies:
            links = study.get("family_links")
            if not isinstance(links, list) or any(not isinstance(link, dict) for link in links):
                raise VerificationError("Missing v5 study-family evidence")
            revisions = {item["revision_id"]: item for item in study.get("revisions", ())}
            for link in links:
                reason = link.get("reason")
                if reason not in {"explicit_version_relation", "exact_title_author_preprint_crosswalk"}:
                    raise VerificationError("Unknown v5 study-family reason")
                sides = ("source", "target") if reason == "exact_title_author_preprint_crosswalk" else ("source",)
                for side in sides:
                    revision = revisions.get(link.get(side + "_revision_id"))
                    if revision is None or revision.get("document_key") != link.get(side + "_key"):
                        raise VerificationError("Study-family evidence has no matching archived revision")
    if set(scores) != set(by_id):
        raise VerificationError("Study/relevance identities do not agree")
    if (type(result.get("input_records")) is not int or not 0 <= result["input_records"] <= 10000
            or type(result.get("unique_studies")) is not int or result["unique_studies"] != len(studies)
            or type(result.get("retained_studies")) is not int
            or result["retained_studies"] != sum(row.get("decision") == "retained" for row in relevance)):
        raise VerificationError("Discovery record/study/retained counts do not agree")
    memberships, proposals, queued, excluded_reasons = (defaultdict(list) for _ in range(4))
    for records, field, target in ((clusters, "study_ids", memberships),
                                    (candidates, "discovery_study_ids", proposals),
                                    (queue, "discovery_study_ids", queued)):
        for row in records:
            members = row.get(field)
            if not isinstance(row.get("candidate_id"), str) or not isinstance(members, list):
                raise VerificationError("Invalid candidate membership structure")
            for sid in members:
                if not isinstance(sid, str) or sid not in by_id:
                    raise VerificationError("Candidate references an unknown study")
                target[sid].append(row["candidate_id"])
    for row in excluded:
        if not isinstance(row.get("study_id"), str) or not isinstance(row.get("reason"), str):
            raise VerificationError("Invalid exclusion structure")
        excluded_reasons[row["study_id"]].append(row["reason"])
    if set(unassigned) - set(by_id):
        raise VerificationError("Unassigned list references an unknown study")
    trace = []
    for sid in sorted(set(by_id) | set(excluded_reasons)):
        study, score = by_id.get(sid), scores.get(sid, {})
        representative = None
        if study is not None:
            index = study.get("representative_index")
            indices = study.get("input_indices")
            if (type(index) is not int or index < 0 or not isinstance(indices, list)
                    or index not in indices or any(type(i) is not int or i < 0 for i in indices)):
                raise VerificationError("Invalid representative input indices")
            if (not isinstance(score.get("decision"), str) or type(score.get("score")) not in (int, float)
                    or not math.isfinite(score["score"]) or not -1 <= score["score"] <= 1):
                raise VerificationError("Invalid relevance decision or score")
            representative = {"input_index": index, "input_indices": indices, "revisions": study.get("revisions", [])}
            if documents is not None:
                if max(indices) >= len(documents):
                    raise VerificationError("Representative index exceeds input document count")
                doc = documents[index]
                representative.update(title=doc["title"], has_abstract=bool(doc.get("abstract")),
                                      source=doc["source"], source_id=doc["source_id"])
        trace.append({"study_id": sid, "representative": representative,
            "identity_keys": [] if study is None else study.get("identity_keys", [sid]),
            "identity_version": None if study is None else study.get("identity_version"),
            "family_links": [] if study is None else study.get("family_links", []),
            "supplementary_study_ids": [] if study is None else study.get("supplementary_study_ids", []),
            "publication_status_version": result.get("publication_status_version"),
            "publication_status_evidence": [item for item in status_evidence
                if {item["source_key"], item["target_key"]}.intersection(
                    [sid] if study is None else study.get("identity_keys", [sid]))],
            "decision": score.get("decision", "excluded_before_embedding"), "score": score.get("score"),
            "exclusion_score": score.get("exclusion_score"), "exclusion_reasons": sorted(excluded_reasons[sid]),
            "cluster_ids": sorted(set(memberships[sid])), "candidate_ids": sorted(set(proposals[sid])),
            "queue_candidate_ids": sorted(set(queued[sid])), "unassigned": sid in unassigned,
            "queue_format": "no_explicit_queue_v1" if version == "semantic-discovery-v1" else "explicit_review_queue"})
    return trace


def compare_results(baseline, current):
    before = {row["study_id"]: row for row in study_trace(baseline)}
    after = {row["study_id"]: row for row in study_trace(current)}
    identity_fields = ("plan_hash", "discovery_snapshot_id", "input_records")
    matching_identity = all(baseline.get(key) == current.get(key) and key in current for key in identity_fields)
    changes = [{"study_id": sid, "before": before[sid], "after": after[sid]}
               for sid in sorted(before.keys() & after.keys()) if before[sid] != after[sid]]
    return {"baseline_version": baseline["discovery_version"], "current_version": current["discovery_version"],
        "identity_fields_match": matching_identity, "exact_same_input_proven": False,
        "added_study_ids": sorted(after.keys() - before.keys()), "removed_study_ids": sorted(before.keys() - after.keys()),
        "changed_studies": changes,
        "limitations": ["Raw discovery identity fields do not prove byte-identical inputs; compare verifier input SHA-256 manifests.",
                        "Candidate IDs may change with methodology; structural differences are not measured scientific improvements."]}


def resource_snapshot():
    """Native high-water RSS, not a subtraction or a sampled exact-worker peak."""
    if sys.platform not in {"darwin", "linux"}:
        return {"status": "unavailable", "unit": "bytes", "reason": "No verified native RSS unit mapping on this platform"}
    try:
        import resource
    except ImportError:
        return {"status": "unavailable", "unit": "bytes", "reason": "resource module unavailable on this platform"}
    factor = 1 if sys.platform == "darwin" else 1024
    return {"status": "available", "unit": "bytes", "native_unit": "bytes" if factor == 1 else "KiB",
        "parent_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * factor,
        "children_peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * factor,
        "scope": "Process-lifetime high-water marks; children include git helpers and the discovery worker. Not a per-worker delta."}


def executed_versions():
    from app.pilot.contracts import SIGNAL_METHODOLOGY_VERSION
    from app.pilot.discovery import DISCOVERY_VERSION

    return {"discovery": DISCOVERY_VERSION, "methodology": None,
            "methodology_scope": "not_executed_discovery_only",
            "service_methodology": SIGNAL_METHODOLOGY_VERSION}


def verify(input_path, model_dir, output_dir, *, baseline_result=None, deadline_seconds=900, cancel=None):
    if (type(deadline_seconds) not in (int, float) or not math.isfinite(deadline_seconds)
            or not 0 < deadline_seconds <= 86400):
        raise VerificationError("Deadline must be finite and within 0–86400 seconds")
    started = time.monotonic()
    cancel = Cancellation() if cancel is None else cancel

    def remaining():
        if cancel.is_set():
            raise WorkerCancelled()
        left = deadline_seconds - (time.monotonic() - started)
        if left <= 0:
            raise WorkerTimeout()
        return left

    remaining()
    value, raw = read_json(input_path)
    plan, options = validate_input(value)
    baseline, baseline_raw = read_json(baseline_result, MAX_RESULT_BYTES) if baseline_result else (None, None)
    if baseline is not None:
        study_trace(baseline)
    model_dir = Path(model_dir).resolve(strict=True)
    if not model_dir.is_dir():
        raise VerificationError("Model directory does not exist")
    remaining()
    destination = claim_output_directory(output_dir, model_dir)
    from app.pilot import discovery
    from app.pilot.encoder import load_spec, spec_fingerprint
    from tools.run_checks import source_environment, source_fingerprint

    # No existing cache/model paths in the input are trusted as write locations.
    effective = dict(value, model_dir=str(model_dir), cache_dir=str(destination / "embedding-cache"), options=options)
    with (destination / "source-input.json").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    write_new(destination / "input.json", effective)
    manifest = {"verifier_version": VERSION, "state": "running", "created_at": datetime.now(UTC).isoformat(),
        "input": {"path": str(Path(input_path).resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                  "bytes": len(raw), "records": len(value["documents"]), "plan_hash": plan.plan_hash},
        "effective_input_sha256": hashlib.sha256((destination / "input.json").read_bytes()).hexdigest(),
        "model": {"directory": str(model_dir), "spec_fingerprint": spec_fingerprint(), "spec": load_spec(),
                  "artifact_verification": "performed_by_production_encoder_in_worker"},
        "versions": executed_versions(),
        "limits": {"deadline_seconds": deadline_seconds, "max_input_bytes": MAX_INPUT_BYTES,
                   "max_result_bytes": MAX_RESULT_BYTES, "max_documents": 10000},
        "credential_policy": "isolated_store_no_keychain_no_environment_import; production scrub plus runtime environment allowlist before spawn",
        "scope": "Fresh local inference on supplied documents; no source collection, downloads or paid AI.",
        "baseline": None if baseline is None else {"path": str(Path(baseline_result).resolve()),
                                                  "sha256": hashlib.sha256(baseline_raw).hexdigest()}}
    credentials = IsolatedCredentialStore()
    try:
        manifest["source_before"] = source_environment()
        manifest["rss_before"] = resource_snapshot()
        write_new(destination / "started.json", manifest)
        run = run_in_process(discovery.task, destination / "input.json", destination / "current-result.json", cancel,
                             credentials=credentials, timeout_seconds=remaining(),
                             max_input_bytes=MAX_INPUT_BYTES, max_output_bytes=MAX_RESULT_BYTES)
        remaining()
        current, current_raw = read_json(run.output_path, MAX_RESULT_BYTES)
        if (current.get("input_records") != len(value["documents"])
                or current.get("plan_hash") != plan.plan_hash
                or current.get("discovery_snapshot_id") != value["discovery_snapshot_id"]):
            raise VerificationError("Worker result does not match the supplied input identity")
        if current.get("encoder_fingerprint") != manifest["model"]["spec_fingerprint"]:
            raise VerificationError("Worker encoder fingerprint differs from the frozen model specification")
        trace = study_trace(current, value["documents"])
        write_new(destination / "study-trace.json", trace)
        if current["discovery_version"] in STATUS_DISCOVERY_VERSIONS:
            write_new(destination / "publication-status.json", {"version": current["publication_status_version"],
                "evidence": current["status_evidence"], "verified_against_input_revisions": True,
                "scope": "Includes status notices whose target has no input study; those targets are not invented as corpus studies."})
        if baseline is not None:
            write_new(destination / "baseline-diff.json", compare_results(baseline, current))
        remaining()
        manifest["source_after_sha256"] = source_fingerprint(PROJECT)
        _, final_input_raw = read_json(input_path)
        manifest["input_unchanged"] = hashlib.sha256(final_input_raw).hexdigest() == manifest["input"]["sha256"]
        if not manifest["input_unchanged"] or manifest["source_before"]["source_tree_sha256"] != manifest["source_after_sha256"]:
            raise VerificationError("Input or source tree changed during verification")
        remaining()
        manifest.update(state="succeeded", result_sha256=hashlib.sha256(current_raw).hexdigest(),
                        worker_wall_seconds=run.elapsed_seconds, traced_studies=len(trace))
        manifest["model"]["artifact_verification"] = "passed_by_production_encoder"
    except Exception as error:
        manifest.update(state="cancelled" if isinstance(error, WorkerCancelled) else "timeout" if isinstance(error, WorkerTimeout) else "failed",
                        error_code=getattr(error, "code", type(error).__name__))
    finally:
        credentials.close()
        manifest["wall_seconds"] = time.monotonic() - started
        manifest["rss_after"] = resource_snapshot()
        write_new(destination / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path)
    parser.add_argument("--deadline-seconds", type=float, default=900)
    parser.add_argument("--cancel-file", type=Path)
    args = parser.parse_args(argv)
    cancel = Cancellation(args.cancel_file)
    try:
        with cancellation_signals(cancel):
            report = verify(args.input, args.model_dir, args.output_dir, baseline_result=args.baseline_result,
                            deadline_seconds=args.deadline_seconds, cancel=cancel)
    except (VerificationError, WorkerError, OSError) as error:
        print("Verification did not start: " + getattr(error, "code", type(error).__name__), file=sys.stderr)
        return 130 if isinstance(error, WorkerCancelled) else 124 if isinstance(error, WorkerTimeout) else 1
    print(json.dumps({"state": report["state"], "output_dir": str(args.output_dir.resolve()),
                      "wall_seconds": report["wall_seconds"]}, ensure_ascii=False))
    return {"succeeded": 0, "cancelled": 130, "timeout": 124}.get(report["state"], 1)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    raise SystemExit(main())
