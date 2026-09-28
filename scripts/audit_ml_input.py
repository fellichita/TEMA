"""Read-only audit of saved history/batches JSON; no ML or network dependencies.

python -m scripts.audit_ml_input SNAPSHOT --output-dir NEW_DIRECTORY
Counts describe the saved export, not the complete literature of a technology.
"""

import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re
import unicodedata


VERSION = "ml-input-audit-0.1"
SAMPLE_SEED = "photonic-input-audit-2026-09-08"
FIELDS = ("publication_year", "publication_date", "title", "abstract", "document_type", "doi", "source_id")


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def year_of(document):
    year = document.get("publication_year")
    return year if type(year) is int and 1000 <= year <= 9999 else None


def canonical_doi(value):
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", "", value.strip(), flags=re.I).casefold()


def normalized_title(value):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))


def missing(value):
    return not isinstance(value, str) or not value.strip()


def field_stats(documents):
    return {
        "count": len(documents),
        "missing_title": sum(missing(d.get("title")) for d in documents),
        "missing_abstract": sum(missing(d.get("abstract")) for d in documents),
        "short_nonempty_abstract_under_40_words": sum(
            not missing(d.get("abstract")) and len(d["abstract"].split()) < 40 for d in documents),
        "missing_or_invalid_year": sum(year_of(d) is None for d in documents),
        "missing_doi": sum(missing(d.get("doi")) for d in documents),
        "missing_url": sum(missing(d.get("url")) for d in documents),
        "document_types": dict(sorted(Counter(d.get("document_type") or "unknown" for d in documents).items())),
        "languages": dict(sorted(Counter(d.get("language") or "unknown" for d in documents).items())),
    }


def full_year_covered(periods, year):
    cursor, end = date(year, 1, 1), date(year, 12, 31)
    for period in sorted(periods, key=lambda p: p["from_date"]):
        left, right = date.fromisoformat(period["from_date"]), date.fromisoformat(period["until_date"])
        if right < cursor or left > end:
            continue
        if left > cursor:
            return False
        if right >= end:
            return True
        cursor = right + timedelta(days=1)
    return False


def audit(snapshot, *, source="openalex", start_year=2020, end_year=2025, sample_per_year=4):
    if not 1000 <= start_year <= end_year <= 9999 or sample_per_year < 1:
        raise ValueError("Invalid audit years or sample size")
    history = snapshot["history"]
    if snapshot.get("schema_version") != 1 or history.get("contract_version", 1) not in (1, 2):
        raise ValueError("Unsupported saved snapshot version")
    periods = [p for p in history["periods"] if p["source"] == source and p["state"] != "split"]
    if not periods:
        raise ValueError("No leaf periods for the selected source")
    batches = {}
    for batch in snapshot["batches"]:
        if batch["job_id"] in batches:
            raise ValueError("Duplicate batch for one job")
        batches[batch["job_id"]] = batch
    groups, period_rows = defaultdict(list), []
    for period in periods:
        job = period.get("job") or {}
        batch = batches.get(job.get("id"))
        entries = batch["documents"] if batch else []
        issues = []
        if batch is None:
            issues.append("missing_export_batch")
        elif batch.get("total") != len(entries) or job.get("stored") != len(entries):
            issues.append("export_count_mismatch")
        if period["state"] != "complete" or job.get("state") != "succeeded":
            issues.append("period_not_complete")
        if not job.get("source_exhausted"):
            issues.append("source_not_exhausted")
        if job.get("skipped") != 0:
            issues.append("source_records_skipped")
        available, scanned = job.get("total_available"), job.get("scanned")
        if available is not None and (scanned is None or scanned < available):
            issues.append("source_count_not_reached")
        request = job.get("request") or {}
        expected = {"topic": history["request"]["topic"], "source": source,
                    "from_date": period["from_date"], "until_date": period["until_date"]}
        if any(request.get(k) != v for k, v in expected.items()) or (
                request.get("primary_topic_ids", []) != history["request"].get("primary_topic_ids", [])):
            issues.append("request_mismatch")
        outside = 0
        for entry in entries:
            document = entry["document"]
            if document["source"] != source:
                issues.append("document_source_mismatch")
            year = year_of(document)
            day = document.get("publication_date")
            if day:
                try:
                    parsed = date.fromisoformat(day)
                    outside += not (period["from_date"] <= parsed.isoformat() <= period["until_date"])
                    if parsed.year != year:
                        issues.append("inconsistent_document_date")
                except ValueError:
                    issues.append("invalid_document_date")
            elif year is not None:
                outside += not (int(period["from_date"][:4]) <= year <= int(period["until_date"][:4]))
            else:
                issues.append("unknown_document_year")
            groups[entry["document_key"]].append(entry)
        if outside:
            issues.append("documents_outside_period")
        period_rows.append({
            "period_id": period["id"], "job_id": job.get("id"),
            "from_date": period["from_date"], "until_date": period["until_date"],
            "backend_state": period["state"], "backend_reason": period.get("incomplete_reason"),
            "scanned": scanned, "stored": job.get("stored"), "skipped": job.get("skipped"),
            "total_available": available, "source_exhausted": job.get("source_exhausted"),
            "request": request, "source_count_unaccounted_after_stored_and_skipped":
                scanned - job["stored"] - job["skipped"] if scanned is not None else None,
            "fields": field_stats([e["document"] for e in entries]),
            "documents_outside_period": outside, "issues": sorted(set(issues)),
        })

    representatives = {key: max(entries, key=lambda e: (e["document"].get("fetched_at") or "", e["revision_id"]))
                       for key, entries in groups.items()}
    conflicts, alias_owners = [], defaultdict(set)
    for key, entries in sorted(groups.items()):
        changed = [f for f in FIELDS if len({json.dumps(e["document"].get(f), sort_keys=True) for e in entries}) > 1]
        if changed:
            conflicts.append({"document_key": key, "fields": changed, "occurrences": len(entries),
                              "versions": [{k: e["document"].get(k) for k in
                                            ("source_id", "publication_year", "publication_date", "doi", "title")}
                                           for e in entries]})
        for entry in entries:
            d = entry["document"]
            if d.get("doi"):
                alias_owners["doi:" + canonical_doi(d["doi"])].add(key)
            if d.get("source_id"):
                alias_owners[d["source"] + ":" + d["source_id"]].add(key)
    collisions = [{"alias": alias, "document_keys": sorted(keys)}
                  for alias, keys in sorted(alias_owners.items()) if len(keys) > 1]
    quarantine = {c["document_key"] for c in conflicts
                  if {"publication_year", "publication_date", "doi"}.intersection(c["fields"])}
    quarantine.update(key for c in collisions for key in c["document_keys"])
    stable = {key: entry for key, entry in representatives.items() if key not in quarantine}
    title_groups = defaultdict(list)
    for key, entry in representatives.items():
        title = normalized_title(entry["document"].get("title") or "")
        if title:
            title_groups[title].append(key)
    repeated_titles = [{"title": title, "document_keys": sorted(keys)}
                       for title, keys in sorted(title_groups.items()) if len(keys) > 1]
    yearly, sample = [], []
    for year in range(start_year, end_year + 1):
        entries = [e for e in stable.values() if year_of(e["document"]) == year]
        raw_entries = [e for values in groups.values() for e in values if year_of(e["document"]) == year]
        overlapping = [p for p in period_rows if int(p["from_date"][:4]) <= year <= int(p["until_date"][:4])]
        reasons = sorted({issue for p in overlapping for issue in p["issues"]})
        if not full_year_covered(overlapping, year):
            reasons.append("calendar_gap")
        conflicting_keys = [key for key in quarantine if any(year_of(e["document"]) == year for e in groups[key])]
        if conflicting_keys:
            reasons.append("identity_or_date_conflicts")
        yearly.append({"year": year, "exported_occurrences": len(raw_entries),
                       "stable_identity_fields": field_stats([e["document"] for e in entries]),
                       "quarantined_identities_touching_year": len(conflicting_keys),
                       "coverage_issues": sorted(set(reasons))})
        chosen = sorted(entries, key=lambda e: (digest(SAMPLE_SEED + "|" + e["document_key"]), e["document_key"]))
        for e in chosen[:sample_per_year]:
            d = e["document"]
            sample.append({"sample_id": f"{year}-{sum(x['year'] == year for x in sample) + 1:02d}",
                           "year": year, "document_key": e["document_key"], "revision_id": e["revision_id"],
                           **{k: d.get(k) for k in ("source_id", "title", "abstract", "doi", "url", "document_type")},
                           "text_sha256": digest((d.get("title") or "") + "\n" + (d.get("abstract") or ""))})
    focus = [e["document"] for e in stable.values()
             if year_of(e["document"]) is not None and start_year <= year_of(e["document"]) <= end_year]
    observed_dates = [e["document"].get("fetched_at") for values in groups.values() for e in values
                      if e["document"].get("fetched_at")]
    report = {
        "audit_version": VERSION, "history_id": history["id"], "request": history["request"],
        "history_contract_version": history.get("contract_version", 1),
        "history_state": history["state"], "history_coverage_complete": history["coverage_complete"],
        "source": source, "focus_years": [start_year, end_year],
        "fetched_at_range": [min(observed_dates), max(observed_dates)] if observed_dates else None,
        "identity_policy": "backend document_key; latest fetched_at/revision_id for field audit; date/DOI conflicts and cross-key aliases quarantined",
        "all_exported_occurrences": sum(map(len, groups.values())), "backend_identities": len(groups),
        "repeated_identity_groups": sum(len(v) > 1 for v in groups.values()),
        "extra_occurrences": sum(len(v) - 1 for v in groups.values()),
        "all_identity_fields_latest_version": field_stats([e["document"] for e in representatives.values()]),
        "quarantined_identities": sorted(quarantine), "focus_stable_fields": field_stats(focus),
        "conflict_counts": {f: sum(f in c["fields"] for c in conflicts) for f in FIELDS},
        "identity_conflicts": conflicts, "cross_key_alias_collisions": collisions,
        "same_normalized_title_groups": len(repeated_titles),
        "same_normalized_title_identities": sum(len(g["document_keys"]) for g in repeated_titles),
        "same_normalized_title_examples": repeated_titles[:10],
        "periods": period_rows, "years": yearly,
        "relevance_sampling": {"seed": SAMPLE_SEED, "per_year": sample_per_year, "size": len(sample),
                               "population": "all stable identities by publication year; no text/type filter",
                               "meaning": "deterministic year-stratified diagnostic sample; not an expert benchmark"},
        "limitations": ["saved export only; relevance is not inferred by this script",
                        "matching titles are possible versions, not proven duplicates; no automatic merge",
                        "absence/shortness of abstract is a quality diagnostic, not a relevance decision"],
    }
    return report, sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory; existing paths are refused")
    parser.add_argument("--source", default="openalex")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--sample-per-year", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new path")
    payload = args.snapshot.read_bytes()
    report, sample = audit(json.loads(payload), source=args.source, start_year=args.start_year,
                           end_year=args.end_year, sample_per_year=args.sample_per_year)
    report["input"] = {"path": str(args.snapshot), "bytes": len(payload),
                       "sha256": hashlib.sha256(payload).hexdigest()}
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # Nothing is written until the entire input audit has succeeded.
    args.output_dir.mkdir(parents=True)
    for name, value in (("metrics.json", report), ("relevance-sample.json", sample)):
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "identities": report["backend_identities"],
                      "focus_stable": report["focus_stable_fields"], "sample_size": len(sample)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
