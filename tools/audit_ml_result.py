"""Audit a saved result against its corpus without fitting NMF or claiming semantic quality."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

from app.ml.contracts import AnalysisInputError, AnalysisOptions
from app.ml.corpus import read_snapshot
from app.ml import engine
from app.ml.directions import direction_profile, profile_fingerprint
from app.ml.engine import analysis_entries, coverage, growth_assessment, metrics, prepare
from app.ml.evidence import supported_card
from app.ml.provenance import implementation_fingerprint
from app.ml.selection import partition_candidates, ranking_key, selection_for
from app.ml.semantic_contracts import SemanticGroup
from app.ml.text import (
    _topic_axis_matches, card_topic_guard, evidence_card, execution_evidence, resolve_topic, sentences,
)


BUCKETS = ("candidates", "preliminary_signals", "established", "excluded_off_direction")
STATUSES = dict(zip(BUCKETS, ("growth_candidate", "exploratory_candidate", "established", "off_direction"), strict=True))


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _nonfinite(value, path="result"):
    if isinstance(value, float) and not math.isfinite(value):
        yield path
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _nonfinite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _nonfinite(child, f"{path}[{index}]")


def _passage_is_original(text, mode, study):
    if not isinstance(text, str) or not text:
        return False
    if mode == "study_title":
        return text == study["title"]
    parts = sentences(study["abstract"])
    if mode == "source_excerpt":
        return text in parts and text in study["abstract"]
    if mode == "neighboring_excerpts":
        return text in {left + "\n" + right for left, right in zip(parts, parts[1:], strict=False)}
    return False


def _axis_matches(text, profile):
    if not profile:
        return {"carrier": [], "compute": []}
    if profile["id"] == "photonic_neuromorphic":
        return _topic_axis_matches([text])
    return {axis: sorted({match.group(0) for pattern in patterns
                          for match in re.finditer(pattern, text, re.I)})
            for axis, patterns in profile["axes"].items()}


def _normalised_guard(guard):
    if guard is None:
        return None
    guard = deepcopy(guard)
    guard["supporting_documents"] = sorted(guard["supporting_documents"], key=lambda d: (d["study_id"], d["text"]))
    return guard


def audit_result(corpus, result, *, replay_axis=True, model_dir=None):
    """Read-only structural replay. Errors never mean an automatic semantic verdict."""
    errors, checks, rows, limitations = [], Counter(), [], [
        "This is structural/provenance replay, not semantic acceptance of problem/advantage/example.",
        "NMF is not refitted: original assignments and raw coherence cannot be independently established.",
        "Metric derivatives are replayed from corpus membership and saved raw score_components.coherence.",
        "Source order is checked for internal consistency, not independently proven to be centroid order.",
    ]

    def check(ok, code, location):
        checks[code] += 1
        if not ok:
            errors.append({"code": code, "location": location})

    def finish():
        return {"schema_version": 1, "kind": "structural_ml_result_audit", "ok": not errors,
                "semantic_acceptance": "not_performed", "errors": errors, "checks": dict(checks),
                "rows": rows, "limitations": limitations, "axis_replay_enabled": replay_axis,
                "input_provenance": corpus.get("provenance"),
                "result_fingerprint": result.get("fingerprint") if isinstance(result, dict)
                and isinstance(result.get("fingerprint"), str) else None}

    check(isinstance(result, dict), "result_object", "result")
    if not isinstance(result, dict):
        return finish()
    for path in _nonfinite(result):
        check(False, "finite_numbers", path)
    if errors:
        return finish()
    check(result.get("schema_version") == 2, "schema_version", "schema_version")
    check(isinstance(result.get("fingerprint"), str) and bool(re.fullmatch(r"[0-9a-f]{64}", result["fingerprint"])),
          "fingerprint_format", "fingerprint")
    check(result.get("status") in {"ranked", "no_groups", "insufficient_data"}, "result_status", "status")
    try:
        options = AnalysisOptions.model_validate(result["options"])
        profile = direction_profile(options.topic)
        photonic = bool(profile and profile["id"] == "photonic_neuromorphic")
        dated, input_selection = analysis_entries(corpus["entries"], options.end_year)
        if options.relevance_mode == "semantic":
            try:
                studies, preparation, semantic_relevance = engine.prepare_analysis(
                    dated, options, model_dir=model_dir)
            except (AnalysisInputError, OSError, ImportError) as error:
                check(False, "semantic_replay_unavailable", "semantic_relevance")
                errors[-1]["message"] = ("Для проверки нужны локальные веса и requirements/semantic.lock. "
                                         + str(error))
                limitations.append("Semantic replay could not run; no conclusion about corpus validity was reached.")
                return finish()
            limitations.append("Semantic scope is replayed with verified local weights; similarity is not a relevance probability.")
        else:
            studies, preparation = prepare(dated, options)
            semantic_relevance = None
        if photonic:
            studies = [{**s, "execution": execution_evidence(s["title"], s["abstract"])} for s in studies]
        by_id = {s["id"]: s for s in studies}
        years = list(range(options.start_year, options.end_year + 1))
        totals = Counter(s["year"] for s in studies)
        expected_coverage = coverage(corpus["periods"], years)
        assessment = growth_assessment(corpus["periods"], years, {y: totals[y] for y in years},
            preparation=preparation, selection=input_selection,
            undated_records=sum(type(e["document"].get("publication_year")) is not int for e in corpus["entries"]))
        comparable = assessment["growth_data_comparable"]
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        check(False, "valid_options_and_corpus", type(error).__name__)
        return finish()
    check(resolve_topic(options.topic).casefold() == resolve_topic(corpus["topic"]).casefold(),
          "same_topic", "options.topic")
    check(result.get("source") == corpus["source"], "same_source", "source")
    check(result.get("provenance") == corpus["provenance"], "same_provenance", "provenance")
    check(result.get("preparation") == preparation, "preparation_replay", "preparation")
    check(result.get("semantic_relevance") == semantic_relevance,
          "semantic_scope_replay", "semantic_relevance")
    semantic_decisions = {row["study_id"]: row for row in
                          (semantic_relevance or {}).get("study_decisions", [])}
    check(result.get("temporal_selection") == input_selection, "input_selection_replay", "temporal_selection")
    check(result.get("source_relations_fingerprint") == input_selection["source_relations_fingerprint"],
          "source_relations_fingerprint", "source_relations_fingerprint")
    check(result.get("coverage") == expected_coverage, "coverage_replay", "coverage")
    check(result.get("growth_data_comparable") is comparable, "comparability_replay", "growth_data_comparable")
    for field in ("growth_comparability", "data_quality_notes"):
        check(result.get(field) == assessment[field], field + "_replay", field)
    check(isinstance(result.get("warnings"), list) and all(
          note in result["warnings"] for note in assessment["data_quality_notes"]),
          "quality_warnings_visible", "warnings")
    check(result.get("direction_counts") == [{"year": y, "documents": totals[y]} for y in years],
          "direction_counts", "direction_counts")
    hash_text = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    implementation_hash = implementation_fingerprint()
    configuration_hash = profile_fingerprint(options.topic)
    expected_fingerprint = hash_text(repr((engine.VERSION, options.model_dump(), implementation_hash,
                                          configuration_hash, input_selection,
                                          [(s["id"], s["year"], s["title"], s["abstract"], s["versions"])
                                           for s in studies], corpus["periods"], corpus["provenance"])))
    if semantic_relevance is not None:
        expected_fingerprint = hash_text(expected_fingerprint + json.dumps(
            semantic_relevance, sort_keys=True, ensure_ascii=False, allow_nan=False))
    check(result.get("pipeline_version") == engine.VERSION, "pipeline_version", "pipeline_version")
    check(result.get("implementation_fingerprint") == implementation_hash,
          "implementation_fingerprint", "implementation_fingerprint")
    check(result.get("direction_profile_fingerprint") == configuration_hash,
          "direction_profile_fingerprint", "direction_profile_fingerprint")
    check(result.get("fingerprint") == expected_fingerprint, "result_fingerprint", "fingerprint")
    if not replay_axis:
        limitations.append("Axis decision replay was explicitly disabled; axis source provenance is still checked.")
    all_groups, seen_groups, seen_studies = [], set(), set()
    originals = {entry["revision_id"]: entry["document"] for entry in dated}
    for bucket in BUCKETS:
        groups = result.get(bucket)
        check(isinstance(groups, list), "bucket_list", bucket)
        if not isinstance(groups, list):
            continue
        for rank, group in enumerate(groups, 1):
            location = f"{bucket}[{rank - 1}]"
            try:
                identifier, ids = group["id"], group["study_ids"]
                check(identifier not in seen_groups, "unique_group_id", location)
                seen_groups.add(identifier)
                check(isinstance(ids, list) and len(ids) == len(set(ids)) and ids == sorted(ids),
                      "unique_sorted_member_ids", location)
                check(not (set(ids) & seen_studies), "disjoint_group_members", location)
                seen_studies.update(ids)
                check(all(key in by_id for key in ids), "member_in_prepared_corpus", location)
                if not ids or any(key not in by_id for key in ids):
                    continue
                expected_id = hashlib.sha256("|".join(sorted(ids)).encode()).hexdigest()[:16]
                check(identifier == expected_id, "member_identity_hash", location)
                check(group["study_count"] == len(ids), "study_count", location)
                members = [by_id[key] for key in ids]
                if semantic_relevance is not None:
                    try:
                        semantic_group = SemanticGroup.model_validate(group.get("semantic_relevance")).model_dump()
                    except ValueError:
                        check(False, "semantic_group_replay", location)
                        continue
                    decision_rows = semantic_group.get("study_decisions", [])
                    semantic_only = sum(semantic_decisions[key]["semantic_only"] for key in ids)
                    check(sorted(row["study_id"] for row in decision_rows) == ids
                          and all(row == semantic_decisions.get(row["study_id"]) for row in decision_rows)
                          and semantic_group.get("semantic_only_documents") == semantic_only
                          and semantic_group.get("corpus_requires_review") is (semantic_relevance["semantic_only_studies"] > 0)
                          and semantic_group.get("requires_review") is bool(
                              semantic_only or semantic_relevance["semantic_only_studies"] > 0),
                          "semantic_group_replay", location)
                else:
                    check(group.get("semantic_relevance") is None, "semantic_group_replay", location)
                source_ids = [s["id"] for s in group["sources"]]
                check(len(source_ids) == min(12, len(ids)) and len(set(source_ids)) == len(source_ids),
                      "source_count", location)
                check(set(source_ids) <= set(ids), "source_membership", location)
                if not set(source_ids) <= set(ids):
                    continue
                if photonic:
                    evidence_ids = [s["study_id"] for s in group["document_evidence"]]
                    check(len(evidence_ids) == len(ids) and set(evidence_ids) == set(ids),
                          "all_member_execution", location)
                    if len(evidence_ids) != len(ids) or set(evidence_ids) != set(ids):
                        continue
                    check(source_ids == evidence_ids[:12], "source_order", location)
                    ordered = [by_id[key] for key in evidence_ids]
                else:
                    check(group.get("document_evidence") == [], "no_unconfigured_execution", location)
                    ordered = [by_id[key] for key in source_ids] + [s for s in members if s["id"] not in source_ids]
                expected_metrics = metrics(members, totals, years, group["metrics"]["score_components"]["coherence"])
                check(group["metrics"] == expected_metrics, "metrics_replay", location)
                check(group["title"] == " / ".join(group["keywords"][:2]), "title_keywords", location)
                keyword_sources = []
                for term in group["keywords"]:
                    matching = [s for s in members if term in s["title"].casefold()]
                    check(len(matching) >= 2, "keyword_in_multiple_source_titles", location + ":" + term)
                    keyword_sources.append({"term": term, "matches": _axis_matches(term, profile),
                                            "sources": [{"study_id": s["id"], "title": s["title"], "url": s["url"]}
                                                        for s in matching]})
                for source in group["sources"]:
                    study = by_id[source["id"]]
                    expected_source = {key: study[key] for key in ("id", "title", "url", "year", "doi", "versions")}
                    expected_source["evidence_level"] = study.get("execution", {}).get("evidence_level", "not_applicable")
                    check(source == expected_source, "source_metadata_versions", location + ":" + study["id"])
                for study in members:
                    for version in study["versions"]:
                        document = originals.get(version["revision_id"])
                        check(document is not None and document["publication_year"] == version["year"],
                              "member_version_provenance", location + ":" + version["revision_id"])
                expected_card, annotations, explanations, rejected = supported_card(ordered, evidence_card(ordered), group["keywords"])
                check(group["card"] == expected_card, "quote_selection_replay", location)
                check(group["evidence_annotations"] == annotations, "quote_modality_replay", location)
                check(group["explanations"] == explanations, "explanation_replay", location)
                check(group["rejected_quotes"] == rejected, "rejected_quotes_replay", location)
                quote_checks = []
                for field in ("problem", "advantage", "example"):
                    quote = group["card"].get(field)
                    if quote is None:
                        quote_checks.append({"field": field, "present": False})
                        continue
                    study = by_id.get(quote["study_id"])
                    check(study is not None and quote["study_id"] in source_ids, "quote_shown_source", location + ":" + field)
                    if study is None:
                        continue
                    original = _passage_is_original(quote["text"], quote["mode"], study)
                    check(original, "quote_exact_source", location + ":" + field)
                    check(quote["url"] == study["url"] and quote["title"] == study["title"],
                          "quote_url_title", location + ":" + field)
                    quote_checks.append({"field": field, "present": True, "original": original, **quote})
                if photonic:
                    for actual, study in zip(group["document_evidence"], ordered, strict=False):
                        expected = {"study_id": study["id"], "url": study["url"], **study["execution"]}
                        check(actual == expected, "execution_replay", location + ":" + study["id"])
                        for passage in actual["evidence"]:
                            check(_passage_is_original(passage["text"], passage["mode"], study),
                                  "execution_exact_source", location + ":" + study["id"])
                    direct = sum(s["execution"]["evidence_level"] == "direct" for s in members)
                    nominal = sum(s["execution"]["evidence_level"] == "nominal" for s in members)
                    expected_execution = {"evidence_level": "direct" if direct else "nominal",
                                          "direct_documents": direct, "nominal_documents": nominal}
                else:
                    expected_execution = {"evidence_level": "not_applicable", "direct_documents": 0, "nominal_documents": 0}
                check(group["execution"] == expected_execution, "execution_summary", location)
                guard = group.get("direction_guard")
                if replay_axis:
                    check(_normalised_guard(guard) == _normalised_guard(card_topic_guard(options.topic, group["keywords"], ordered)),
                          "axis_replay", location)
                if guard:
                    for supporting in guard["supporting_documents"]:
                        study = by_id.get(supporting["study_id"])
                        check(study is not None and study["id"] in ids, "axis_source_membership", location)
                        if study is not None:
                            check(_passage_is_original(supporting["text"], supporting["mode"], study),
                                  "axis_exact_source", location + ":" + study["id"])
                            check(supporting["url"] == study["url"] and supporting["title"] == study["title"],
                                  "axis_url_title", location + ":" + study["id"])
                check(group["selection"] == selection_for(group, comparable), "selection_reasons", location)
                check(group["selection"]["bucket"] == bucket and group["status"] == STATUSES[bucket],
                      "bucket_status", location)
                check(group["stage"] == "requires_review", "no_unverified_confirmed_stage", location)
                check(not (bucket == "candidates" and not comparable), "no_confirmed_incomplete_growth", location)
                witnesses = {}
                for field in ("first_observed_year_in_corpus", "first_observed_year_in_window"):
                    year = expected_metrics[field]
                    witnesses[field] = {"year": year, "sources": [
                        {"study_id": s["id"], "revision_id": v["revision_id"],
                         "title": originals[v["revision_id"]]["title"], "url": originals[v["revision_id"]]["url"]}
                        for s in members if s["year"] == year for v in s["versions"] if v["year"] == year
                        and v["revision_id"] in originals]}
                rows.append({"bucket": bucket, "rank": rank, "id": identifier, "title": group["title"],
                             "study_count": len(ids), "cluster_matches": _axis_matches(" ".join(group["keywords"]), profile),
                             "keyword_sources": keyword_sources, "direction_guard": guard,
                             "axis_check": guard["axis_check"] if guard else "not_applied",
                             "source_matches": guard["supporting_documents"] if guard else [],
                             "selection": group["selection"], "quote_provenance": quote_checks,
                             "first_year_witnesses": witnesses})
                all_groups.append(group)
            except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
                check(False, "valid_card_structure", location + ":" + type(error).__name__)
        try:
            check(groups == sorted(groups, key=ranking_key), "bucket_ranking_order", bucket)
        except (KeyError, TypeError, ValueError, AttributeError):
            check(False, "bucket_ranking_order", bucket)
    try:
        summary = result["selection_summary"]
        for bucket in BUCKETS:
            check(summary[bucket] == len(result[bucket]), "bucket_summary_counts", bucket)
        check(summary["requested_top_k"] == options.top_k and len(result["candidates"]) <= options.top_k,
              "top_limit", "selection_summary")
        check(summary["shortfall"] == options.top_k - len(result["candidates"]), "top_shortfall", "selection_summary")
        hidden = summary["below_top_ids"] + summary["duplicate_label_ids"]
        check(len(hidden) == len(set(hidden)) and not set(hidden) & seen_groups,
              "hidden_ids_disjoint", "selection_summary")
        check(summary["eligible_before_limit"] == len(result["candidates"]) + len(summary["below_top_ids"]),
              "eligible_before_limit", "selection_summary")
        check(len(result["candidates"]) == min(options.top_k, summary["eligible_before_limit"]),
              "eligible_fill_before_hiding", "selection_summary")
        expected_status = ("insufficient_data" if result.get("model") is None else
                           "ranked" if any(result[b] for b in BUCKETS[:3]) else "no_groups")
        check(result["status"] == expected_status, "status_matches_result", "status")
        if result.get("model") is None:
            check(not hidden and not any(result[b] for b in BUCKETS),
                  "no_groups_without_model", "model")
        if hidden:
            limitations.append("Below-top and duplicate-label cards are not serialized; their decisions cannot be replayed.")
        elif len(all_groups) == sum(len(result[b]) for b in BUCKETS):
            expected_buckets, expected_summary = partition_candidates(deepcopy(all_groups), comparable, options.top_k)
            check(expected_summary == summary, "full_selection_summary_replay", "selection_summary")
            for bucket in BUCKETS:
                check([g["id"] for g in expected_buckets[bucket]] == [g["id"] for g in result[bucket]],
                      "full_selection_order_replay", bucket)
        if result.get("model") is not None:
            model = result["model"]
            check(model["type"] == "TF-IDF + NMF", "model_type", "model.type")
            check(model["groups_before_top"] == sum(len(result[b]) for b in BUCKETS) + len(hidden),
                  "groups_before_top", "model")
            check(model["excluded_groups"].get("off_direction", 0) == len(result["excluded_off_direction"]),
                  "excluded_model_count", "model.excluded_groups.off_direction")
            for key, value in {"random_state": 42, "max_iter": 250, "tol": .001,
                               "concentration_threshold": .2, "topic_limit": 32}.items():
                check(model[key] == value, "frozen_model_parameters", "model." + key)
            upper_bound = min(32, max(2, int(math.sqrt(len(studies)))), len(studies) - 1)
            check(type(model["topics"]) is int and 1 <= model["topics"] <= upper_bound,
                  "model_topics_bound", "model.topics")
            check(type(model["iterations"]) is int and 1 <= model["iterations"] <= 250,
                  "model_iterations_bound", "model.iterations")
    except (KeyError, TypeError, ValueError) as error:
        check(False, "valid_selection_structure", type(error).__name__)
    return finish()


def _cell(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace("|", "\\|").replace("\n", "<br>")


def markdown_report(report):
    lines = ["# Структурная проверка ML-выдачи", "",
             f"Результат: {'проверки пройдены' if report['ok'] else 'обнаружены ошибки'}. "
             f"Карточек: {len(report['rows'])}; ошибок: {len(report['errors'])}.", "",
             "**Это проверка происхождения и согласованности данных, не содержательная приёмка цитат.**", "",
             "Все совпадения и источники представлены в сопутствующем JSON. В таблице для каждой оси "
             "показан первый подходящий подтверждающий фрагмент, если guard обращался к документам. "
             "`supported` подтверждает лексические оси, а не оптического исполнителя.", "",
             "| Раздел / № | Карточка | Ось носителя | Ось вычисления | Откуда совпадения, источник | Решение / причины | Происхождение цитат |",
             "|---|---|---|---|---|---|---|"]
    for row in report["rows"]:
        details = []
        document_matches = {axis: sorted({match for source in row["source_matches"]
                                         for match in source["matches"][axis]}) for axis in ("carrier", "compute")}
        for item in row["keyword_sources"]:
            if item["sources"]:
                source = item["sources"][0]
                details.append(f"keywords: «{_cell(item['term'])}» ← [{_cell(source['title'])}]({source['url']})")
        if row["source_matches"]:
            selected = []
            for axis in ("carrier", "compute"):
                source = next((s for s in row["source_matches"] if s["matches"][axis]), row["source_matches"][0])
                if source not in selected:
                    selected.append(source)
            for source in selected:
                details.append(f"Документ ({len(row['source_matches'])} подтверждений всего): «{_cell(source['text'])}» "
                               f"— [{_cell(source['title'])}]({source['url']}); "
                               f"носитель={_cell(source['matches']['carrier'])}, вычисление={_cell(source['matches']['compute'])}")
        present = [q for q in row["quote_provenance"] if q["present"]]
        verified = sum(q["original"] for q in present)
        lines.append("| " + " | ".join([
            _cell(f"{row['bucket']} / {row['rank']}"), _cell(row["title"]),
            _cell("keywords: " + (", ".join(row["cluster_matches"]["carrier"]) or "нет") +
                  "; документы: " + (", ".join(document_matches["carrier"]) or "нет лексических совпадений")),
            _cell("keywords: " + (", ".join(row["cluster_matches"]["compute"]) or "нет") +
                  "; документы: " + (", ".join(document_matches["compute"]) or "нет лексических совпадений")),
            "<br>".join(details), _cell(row["axis_check"] + "; " + ", ".join(row["selection"]["reasons"])),
            f"{verified}/{len(present)} дословных; пустых полей: {3 - len(present)}",
        ]) + " |")
    lines += ["", "## Ограничения", ""] + ["- " + item for item in report["limitations"]]
    if report["errors"]:
        lines += ["", "## Ошибки", ""] + [f"- `{e['code']}`: {_cell(e['location'])}" for e in report["errors"]]
    return "\n".join(lines) + "\n"


def write_report_pair(report, output_base):
    """Stage both files; publish by exclusive hard link so existing files cannot be replaced."""
    base = Path(output_base)
    paths = [Path(str(base) + ".json"), Path(str(base) + ".md")]
    if any(p.exists() for p in paths):
        raise FileExistsError("Отчёт уже существует; выберите новое имя.")
    payloads = [json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", markdown_report(report)]
    temporary, published = [], []
    try:
        for path, payload in zip(paths, payloads, strict=True):
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
                temporary.append(Path(handle.name))
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for source, target in zip(temporary, paths, strict=True):
            os.link(source, target)
            published.append((source, target))
    except BaseException:
        for source, target in published:
            if target.exists() and os.path.samefile(source, target):
                target.unlink()
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--model-dir", help="Каталог локальных весов для replay смыслового режима")
    args = parser.parse_args()
    corpus = read_snapshot(args.snapshot)
    payload = Path(args.result).read_bytes()
    result = json.loads(payload)
    report = audit_result(corpus, result, model_dir=args.model_dir)
    report["result_sha256"] = hashlib.sha256(payload).hexdigest()
    report["audit_fingerprint"] = _digest(report)
    paths = write_report_pair(report, args.output_base)
    for error in report["errors"]:
        if error.get("message"):
            print(error["message"], file=sys.stderr)
    print(json.dumps({"ok": report["ok"], "errors": len(report["errors"]),
                      "cards": len(report["rows"]), "outputs": [str(p) for p in paths]}, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
