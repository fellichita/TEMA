"""Freeze fresh, blinded scope-review samples; never use the labels to tune the app."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from app.ml.corpus import read_snapshot
from app.ml.engine import TYPES, analysis_entries
from app.ml.text import PROCEEDINGS_TITLE, SERVICE_TITLE, clean, safe_url, scope_check, title_key
from tools.audit_ml_result import BUCKETS, _cell


SALT = "trendanalizer-fresh-scope-review-2026-09-08-v1"


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def identity(entry):
    doi = (entry["document"].get("doi") or "").casefold().removeprefix("https://doi.org/")
    return "doi:" + doi if doi else entry["document_key"]


def title_author(document):
    title = title_key(document["title"])
    authors = document.get("authors") or []
    author = title_key(authors[0]).split()[-1:] if authors else []
    return (title, tuple(author)) if len(title.split()) >= 6 and author else None


def _blocked(result, all_members):
    keys, revisions = set(), set()
    for bucket in BUCKETS:
        for card in result[bucket]:
            if all_members:
                keys.update(card["study_ids"])
            for source in card["sources"]:
                keys.add(source["id"])
                for version in source["versions"]:
                    keys.add(version["document_key"])
                    revisions.add(version["revision_id"])
            for quote in card["card"].values():
                if quote:
                    keys.add(quote["study_id"])
            for support in (card.get("direction_guard") or {}).get("supporting_documents", []):
                keys.add(support["study_id"])
    return keys, revisions


def build_control(corpus, result, direction_name):
    entries, input_selection = analysis_entries(corpus["entries"], result["options"]["end_year"])
    by_identity = defaultdict(list)
    for entry in entries:
        by_identity[identity(entry)].append(entry)
    eligible = []
    rejected_quality = defaultdict(int)
    for key, versions in sorted(by_identity.items()):
        years = {e["document"].get("publication_year") for e in versions}
        if len(years) != 1:
            rejected_quality["conflicting_year"] += 1
            continue
        chosen = max(versions, key=lambda e: (len(e["document"].get("abstract") or ""),
                                              e["document"].get("fetched_at") or "", e["revision_id"]))
        document = chosen["document"]
        year = document.get("publication_year")
        title, abstract = clean(document["title"]), clean(document.get("abstract"))
        if type(year) is not int or not 1900 <= year <= result["options"]["end_year"]:
            rejected_quality["invalid_year"] += 1
        elif document.get("document_type") not in TYPES:
            rejected_quality["unsupported_type"] += 1
        elif len(title.split()) < 3 or SERVICE_TITLE.search(title) or PROCEEDINGS_TITLE.search(title):
            rejected_quality["service_or_short_title"] += 1
        elif not 15 <= len(abstract.split()) <= 3000:
            rejected_quality["invalid_abstract_length"] += 1
        elif not safe_url(document.get("url")):
            rejected_quality["invalid_url"] += 1
        else:
            eligible.append({"id": key, "document": document, "title": title, "abstract": abstract,
                             "revision_id": chosen["revision_id"],
                             "versions": [{"document_key": v["document_key"], "revision_id": v["revision_id"]}
                                          for v in versions]})
    # Collapse the same conservative long-title/first-author duplicate groups as preparation.
    by_title = defaultdict(list)
    for entry in eligible:
        by_title[title_author(entry["document"]) or (entry["id"],)].append(entry)
    representatives = []
    for members in by_title.values():
        selected = max(members, key=lambda e: (len(e["abstract"]), e["id"]))
        representative = dict(selected)
        representative["id"] = min(e["id"] for e in members)
        representative["versions"] = [v for e in members for v in e["versions"]]
        representatives.append(representative)
    pools_by_strategy = {}
    scope_predictions = {}
    for strategy, all_members in (("all_release_members", True), ("reviewed_sources_and_guard_evidence", False)):
        keys, revisions = _blocked(result, all_members)
        blocked_titles = {title_author(e["document"]) for e in entries
                          if identity(e) in keys or e["document_key"] in keys or e["revision_id"] in revisions} - {None}
        pools = {True: [], False: []}
        for entry in representatives:
            if (entry["id"] in keys or title_author(entry["document"]) in blocked_titles
                    or any(v["document_key"] in keys or v["revision_id"] in revisions for v in entry["versions"])):
                continue
            if entry["id"] not in scope_predictions:
                scope_predictions[entry["id"]] = scope_check(corpus["topic"], entry["title"], entry["abstract"])
            prediction = scope_predictions[entry["id"]]
            pools[prediction == "direct_lexical_signal"].append({**entry, "prediction": prediction})
        pools_by_strategy[strategy] = pools
    strategy = next((name for name, pools in pools_by_strategy.items() if all(len(pool) >= 10 for pool in pools.values())), None)
    if strategy is None:
        counts = {name: {str(k): len(v) for k, v in pools.items()} for name, pools in pools_by_strategy.items()}
        raise ValueError(f"Недостаточно незнакомых документов для 10+10: {counts}")
    pools = pools_by_strategy[strategy]
    selected = [entry for pool in pools.values()
                for entry in sorted(pool, key=lambda e: (digest(SALT + "|select|" + e["id"]), e["id"]))[:10]]
    selected.sort(key=lambda e: digest(SALT + "|blind-order|" + e["id"]))
    blind, predictions = [], []
    for entry in selected:
        sample_id = direction_name + "-" + digest(SALT + "|sample|" + entry["id"])[:12]
        blind.append({"sample_id": sample_id, "title": entry["title"], "abstract": entry["abstract"],
                      "publication_year": entry["document"]["publication_year"], "url": entry["document"]["url"],
                      "review_label": None, "review_reason": None})
        predictions.append({"sample_id": sample_id, "study_id": entry["id"],
                            "scope_check": entry["prediction"], "admitted": entry["prediction"] == "direct_lexical_signal",
                            "chosen_revision_id": entry["revision_id"], "versions": entry["versions"],
                            "text_sha256": digest(entry["title"] + "\n" + entry["abstract"])})
    blinded = {"schema_version": 1, "direction": corpus["topic"], "documents": blind,
               "instructions": "Оцените тематическую связь по исходному тексту. Запишите решение и краткое основание. "
                               "Не открывайте файл прогнозов до завершения разметки."}
    key = {"schema_version": 1, "salt": SALT, "direction": corpus["topic"], "exclusion_strategy": strategy,
           "scope_decision": "scope_check == direct_lexical_signal", "sample_unit": "identity plus conservative version links",
           "pool_counts": {"admitted": len(pools[True]), "rejected": len(pools[False])},
           "all_strategy_pool_counts": {name: {"admitted": len(p[True]), "rejected": len(p[False])}
                                        for name, p in pools_by_strategy.items()},
           "sample_counts": {"admitted": 10, "rejected": 10}, "predictions": predictions,
           "input_provenance": corpus["provenance"], "release_fingerprint": result["fingerprint"],
           "input_selection": input_selection, "quality_exclusions": dict(rejected_quality),
           "limitations": ["Fresh frozen review after rule development on the same local corpora, not an external representative estimate.",
                           "Balanced predicted-positive/negative sampling changes prevalence; pooled recall/accuracy need appropriate weighting.",
                           "Labels are unset; predictions must remain hidden from annotators."]}
    return blinded, key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--direction-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = Path(args.result).read_bytes()
    result = json.loads(payload)
    blind, key = build_control(read_snapshot(args.snapshot), result, args.direction_name)
    key["result_sha256"] = hashlib.sha256(payload).hexdigest()
    key["blind_sha256"] = digest(json.dumps(blind, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    key["preparation_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {kind: args.output_dir / (args.direction_name + "-" + kind + suffix)
             for kind, suffix in (("blind", ".json"), ("predictions", ".json"), ("review", ".md"))}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Контрольная выборка уже существует; выберите новый каталог.")
    markdown = [f"# Слепая контрольная выборка: {_cell(blind['direction'])}", "", blind["instructions"], ""]
    for number, doc in enumerate(blind["documents"], 1):
        markdown += [f"## {number}. `{doc['sample_id']}`", "", f"**{_cell(doc['title'])}**", "",
                     f"Год: {doc['publication_year']}. [Источник]({doc['url']})", "", _cell(doc["abstract"]), "",
                     "Решение: __________", "Основание: __________", ""]
    for kind, value in (("blind", blind), ("predictions", key)):
        with paths[kind].open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    with paths["review"].open("x", encoding="utf-8") as handle:
        handle.write("\n".join(markdown))
    print(json.dumps({"direction": args.direction_name, "strategy": key["exclusion_strategy"],
                      "pool_counts": key["pool_counts"], "documents": len(blind["documents"]),
                      "outputs": {k: str(v) for k, v in paths.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
