"""Deterministic local NMF baseline and transparent candidate scoring."""

from collections import Counter, defaultdict
from datetime import date, timedelta
import hashlib
import math
import json
import re
from pathlib import Path

from app.input_safety import MAX_TITLE_CHARACTERS

from app.ml.contracts import AnalysisInputError, AnalysisOptions
from app.ml.corpus import checkpoint, normalize_snapshot_doi
from app.ml.model import fit_topics
from app.ml.evidence import supported_card, LATEX
from app.ml.selection import partition_candidates
from app.ml.provenance import implementation_fingerprint
from app.ml.study_relations import acs_supplement_parent_candidate, first_author_surname, source_relations_fingerprint, supplement_descendants, supplement_relations
from app.ml.text import PROCEEDINGS_TITLE, SERVICE_TITLE, card_topic_guard, clean, document_text_issue, evidence_card, resolve_topic, safe_url, scope_check, title_key


VERSION = "local-mvp-nmf-1.6.0"
TYPES = {"article", "preprint", "conference-paper", "review", "publication"}
GENERIC_PHRASES = {"long term", "short term", "term memory", "machine learning", "deep learning", "neural networks",
                   "neural network", "pattern recognition", "data processing", "energy consumption", "power consumption"}


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _document_identity(entry):
    doi = entry["document"].get("doi")
    return "doi:" + normalize_snapshot_doi(doi) if doi else entry["document_key"]


def _explicitly_retracted(document):
    metadata = document.get("raw_metadata") or {}
    # Oversized direct in-memory input is ineligible later. Inspect a bounded
    # prefix for an explicit withdrawal marker without manufacturing source text.
    title = (document.get("title") or "")[:MAX_TITLE_CHARACTERS]
    return bool((isinstance(metadata, dict) and metadata.get("is_retracted") is True)
                or re.match(r"^\s*(?:retracted|retraction)\s*:", clean(title), re.I))


def _retracted_identities(entries, end_year, cancel=None, *, relations=None):
    # Source keys persist when backend enriches a saved record with a DOI.
    # Join only explicit identity aliases, never titles or fuzzy similarities.
    parents, identities, withdrawn = {}, set(), set()

    def find(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for entry in entries:
        checkpoint(cancel)
        document = entry["document"]
        year = document.get("publication_year")
        if type(year) is int and year > end_year:
            continue
        identity = _document_identity(entry)
        identities.add(identity)
        parents[find(("key", entry["document_key"]))] = find(("identity", identity))
        if _explicitly_retracted(document):
            withdrawn.add(identity)
    roots = {find(("identity", identity)) for identity in withdrawn}
    withdrawn_ids = {identity for identity in identities if find(("identity", identity)) in roots}
    if relations is None:
        relations, _ = supplement_relations(entries, end_year, cancel)
    # A withdrawn parent cannot regain independent evidence via its materials.
    # Withdrawal of a supplement does not retract the parent publication.
    return withdrawn_ids | supplement_descendants(relations, withdrawn_ids)


def analysis_entries(entries, end_year, cancel=None):
    """Current-corpus eligibility before the protected version preparation."""
    reviews_path = Path(__file__).with_name("source_reviews.json")
    reviews = json.loads(reviews_path.read_text(encoding="utf-8"))["reviews"]
    retained, future, retracted, disputed = [], 0, [], []
    relations, _ = supplement_relations(entries, end_year, cancel)
    retracted_ids = _retracted_identities(entries, end_year, cancel, relations=relations)
    disputed_ids = set()
    for entry in entries:
        checkpoint(cancel)
        document = entry["document"]
        year = document.get("publication_year")
        if type(year) is int and year > end_year:
            future += 1
            continue
        doi = normalize_snapshot_doi(document["doi"]) if document.get("doi") else ""
        identity = "doi:" + doi if document.get("doi") else entry["document_key"]
        review = reviews.get(doi)
        if review and year == review["affected_corpus_year"]:
            disputed.append({"document_key": entry["document_key"], "doi": doi, **review})
            disputed_ids.add(identity)
            continue
        if identity in retracted_ids:
            retracted.append(entry["document_key"])
            continue
        retained.append((entry, identity))
    disputed_materials = supplement_descendants(relations, disputed_ids)
    related_exclusions = sorted({entry["document_key"] for entry, identity in retained
                                 if identity in disputed_materials})
    retained = [entry for entry, identity in retained if identity not in disputed_materials]
    return retained, {"raw_occurrences": len(entries), "excluded_after_end_year": future,
                      "excluded_retracted_occurrences": len(retracted), "retracted_document_keys": sorted(set(retracted)),
                      "excluded_disputed_date_occurrences": len(disputed), "date_reviews": disputed,
                      "excluded_disputed_parent_materials": related_exclusions,
                      "source_relations_fingerprint": source_relations_fingerprint(),
                      "source_reviews_fingerprint": _hash(reviews_path.read_text(encoding="utf-8"))}


def prepare(entries, options, cancel=None):
    groups, rejected = defaultdict(list), Counter()
    for entry in entries:
        checkpoint(cancel)
        groups[_document_identity(entry)].append(entry)
    candidates, conflicts = [], []
    retracted_ids = _retracted_identities(entries, options.end_year, cancel)
    materials, relation_conflicts = supplement_relations(entries, options.end_year, cancel)
    material_diagnostics = []
    for key, records in sorted(groups.items()):
        checkpoint(cancel)
        if key in retracted_ids:
            rejected["retracted_study"] += 1
            continue
        if key in relation_conflicts:
            rejected[relation_conflicts[key]] += 1
            material_diagnostics.append({"supplement_id": key, "status": relation_conflicts[key]})
            continue
        if key in materials:
            continue
        years = {e["document"].get("publication_year") for e in records}
        if len(years) > 1:
            rejected["conflicting_year"] += 1
            conflicts.append(key)
            continue
        # Try the same preference order, retaining all versions as provenance.
        # A longer unusable record must not hide a usable version of this study.
        ordered = sorted(records, key=lambda e: (len(e["document"].get("abstract") or ""),
                                                 e["document"].get("fetched_at") or "", e["revision_id"]), reverse=True)
        reasons = []
        for e in ordered:
            checkpoint(cancel)
            d = e["document"]
            oversized = document_text_issue(d)
            if oversized:
                reasons.append(oversized)
                continue
            year = d.get("publication_year")
            title, abstract = clean(d.get("title")), clean(d.get("abstract"))
            checkpoint(cancel)
            reason = None
            if type(year) is not int or not 1900 <= year <= options.end_year:
                reason = "unknown_or_future_year"
            elif d.get("document_type") not in TYPES:
                reason = "unsupported_document_type"
            elif SERVICE_TITLE.search(title) or PROCEEDINGS_TITLE.search(title) or len(title.split()) < 3:
                reason = "service_or_short_title"
            elif len(abstract.split()) < 15:
                reason = "missing_or_short_abstract"
            elif len(abstract.split()) > 3000:
                reason = "oversized_abstract_requires_review"
            elif not safe_url(d.get("url")):
                reason = "invalid_source_url"
            else:
                signal = scope_check(options.topic, title, abstract)
                if signal != "direct_lexical_signal":
                    reason = signal
            if reason is None:
                break
            reasons.append(reason)
        else:
            rejected[reasons[0]] += 1
            continue
        candidates.append({"id": key, "year": year, "title": title, "abstract": abstract, "url": d["url"],
                           "doi": d.get("doi"), "type": d["document_type"], "authors": list(d.get("authors") or []),
                           "versions": [{"document_key": r["document_key"], "revision_id": r["revision_id"],
                                         "source_id": r["document"].get("source_id"),
                                         "year": r["document"].get("publication_year")} for r in records]})
    parents = {study["id"]: study for study in candidates}
    attached = 0
    for key, link in sorted(materials.items()):
        checkpoint(cancel)
        if key not in groups or key in retracted_ids:
            continue
        parent = parents.get(link["parent_id"])
        status = "attached" if parent is not None else "parent_unavailable"
        material_diagnostics.append({"supplement_id": key, **link, "status": status})
        if parent is None:
            rejected["supplementary_material_without_eligible_parent"] += 1
            continue
        attached += 1
        rejected["supplementary_material"] += 1
        parent["versions"].extend({
            "document_key": record["document_key"], "revision_id": record["revision_id"],
            "source_id": record["document"].get("source_id"),
            "year": record["document"].get("publication_year"),
            "relation": "supplement", "parent_identity": link["parent_id"],
            "direct_parent_identity": link["direct_parent_id"],
            "relation_evidence": link["evidence"],
        } for record in groups[key] if not (type(record["document"].get("publication_year")) is int
                                           and record["document"]["publication_year"] > options.end_year))
    # Long identical titles with the same first-author surname: possible versions.
    # No fuzzy merge and no merging of generic conference headings.
    title_groups = defaultdict(list)
    for study in candidates:
        key = title_key(study["title"])
        author = first_author_surname(study["authors"])
        # An unconfirmed ACS material must not bypass the stricter relationship
        # checks through the older title/surname possible-version heuristic.
        merge_key = ((key, tuple(author)) if len(key.split()) >= 6 and author
                     and not acs_supplement_parent_candidate(study.get("doi")) else (study["id"],))
        title_groups[merge_key].append(study)
    studies, merged = [], 0
    for members in title_groups.values():
        selected = max(members, key=lambda s: (len(s["abstract"]), s["id"]))
        study = dict(selected)
        study["id"] = min(s["id"] for s in members)
        study["year"] = min(s["year"] for s in members)
        study["versions"] = sorted([v for s in members for v in s["versions"]], key=lambda v: v["revision_id"])
        study["possible_versions_merged"] = len(members) > 1
        merged += len(members) - 1
        studies.append(study)
    studies.sort(key=lambda s: s["id"])
    return studies, {"input_occurrences": len(entries), "source_identities": len(groups),
                     "repeated_occurrences": len(entries) - len(groups), "rejected": dict(rejected),
                     "conflicting_identity_ids": conflicts, "possible_versions_merged": merged,
                     "supplementary_materials_attached": attached,
                     "supplementary_relations": material_diagnostics,
                     "retained_studies": len(studies)}


def coverage(periods, years):
    result = {}
    for year in years:
        cursor, end = date(year, 1, 1), date(year, 12, 31)
        issues = set()
        for p in sorted(periods, key=lambda p: p["from_date"]):
            left, right = date.fromisoformat(p["from_date"]), date.fromisoformat(p["until_date"])
            if right < date(year, 1, 1) or left > end:
                continue
            issues.update(p["issues"])
            if left > cursor:
                issues.add("Пробел в календарном покрытии")
            cursor = max(cursor, min(right, end) + timedelta(days=1))
        if cursor <= end:
            issues.add("Пробел в календарном покрытии")
        result[str(year)] = sorted(issues)
    return result


def growth_assessment(periods, years, totals, *, preparation=None, selection=None, undated_records=0):
    """Compare the retained sample; preserve raw coverage and quality separately."""
    years = list(years)
    hard_periods = [{**p, "issues": [issue for issue in p["issues"]
                                    if issue not in p.get("nonblocking_issues", [])]} for p in periods]
    blocking = coverage(hard_periods, years)
    calendar = coverage([{**p, "issues": []} for p in periods], years)
    located_undated = sum(p.get("undated_records", 0) for p in periods)
    if undated_records > located_undated:
        # Legacy in-memory inputs may lack the original batch association.
        # Do not invent a year or silently drop their temporal uncertainty.
        for year in years:
            blocking[str(year)].append("Неизвестный год публикации у записей без установленного периода: "
                                       f"{undated_records - located_undated}.")
    denominators, notes = [], []
    for year in years:
        value = totals.get(year)
        state = ("unknown" if value is None else
                 "invalid" if type(value) is not int or value < 0 else
                 "positive" if value > 0 else "zero")
        denominators.append({"year": year, "documents": value if state in {"zero", "positive"} else None,
                             "state": state})
        if state != "positive":
            message = {"zero": "Нулевой знаменатель направления: нет пригодных исследований.",
                       "unknown": "Знаменатель направления неизвестен.",
                       "invalid": "Некорректный знаменатель направления."}[state]
            blocking[str(year)] = sorted(set([*blocking[str(year)], message]))
    for period in sorted(periods, key=lambda p: p["from_date"]):
        if not any(period["from_date"] <= f"{year}-12-31" and period["until_date"] >= f"{year}-01-01"
                   for year in years):
            continue
        notes.extend(f"{period['from_date']} — {period['until_date']}: {note}"
                     for note in period.get("data_quality_notes", []))
    selection, preparation = selection or {}, preparation or {}
    rejected = preparation.get("rejected", {})
    reasons = [
        (selection.get("excluded_retracted_occurrences", 0) or rejected.get("retracted_study", 0),
         "Исключены явно отозванные публикации"),
        (selection.get("excluded_disputed_date_occurrences", 0), "Исключены публикации с проверенным расхождением дат"),
        (rejected.get("conflicting_year", 0), "Исключены идентичности с конфликтующими годами"),
        (preparation.get("possible_versions_merged", 0), "Объединены вероятные версии исследований"),
        (preparation.get("supplementary_materials_attached", 0),
         "Дополнительные материалы сохранены при основных публикациях и не считаются независимыми исследованиями"),
        (rejected.get("supplementary_material_without_eligible_parent", 0),
         "Исключены дополнительные материалы без пригодной основной публикации"),
        (rejected.get("missing_or_short_abstract", 0), "Исключены записи без достаточной аннотации"),
    ]
    notes.extend(f"{message}: {count}. Это ограничение сохранённой пригодной выборки."
                 for count, message in reasons if count)
    return {"growth_data_comparable": not any(blocking.values()),
            "growth_comparability": {"scope": "retained_studies_in_saved_corpus",
                                     "calendar_complete": not any(calendar.values()),
                                     "blocking_issues": blocking, "denominators": denominators},
            "data_quality_notes": list(dict.fromkeys(notes))}


def metrics(studies, totals, years, coherence):
    counts = Counter(s["year"] for s in studies)
    rows = [{"year": year, "documents": counts[year], "direction_documents": totals.get(year, 0),
             "share": counts[year] / totals[year] if totals.get(year, 0) > 0 else None} for year in years]
    old, recent = rows[:-3], rows[-3:]
    old_n, old_total = sum(r["documents"] for r in old), sum(r["direction_documents"] for r in old)
    new_n, new_total = sum(r["documents"] for r in recent), sum(r["direction_documents"] for r in recent)
    old_known = bool(old) and all(r["share"] is not None for r in old)
    new_known = bool(recent) and all(r["share"] is not None for r in recent)
    old_share = old_n / old_total if old_known else None
    new_share = new_n / new_total if new_known else None
    raw_ratio = new_share / old_share if old_share and new_share is not None else None
    # Smooth both proportions, including a measured zero; missing data stay undefined.
    alpha = 0.5
    ratio = ((new_n + alpha) / (new_total + 2 * alpha)) / ((old_n + alpha) / (old_total + 2 * alpha)) \
        if old_known and new_known and old_n + new_n > 0 else None
    # A measured disappearance must not gain growth from the smoothing prior.
    if ratio is not None and old_n > 0 and new_n == 0:
        ratio = 0.0
    delta = 100 * (new_share - old_share) if old_share is not None and new_share is not None else None
    transitions = sum(a["share"] is not None and b["share"] is not None and b["share"] > a["share"]
                      for a, b in zip(recent, recent[1:], strict=False))
    first = min(s["year"] for s in studies)
    zero_baseline = old_known and old_n == 0
    new_topic = zero_baseline and first >= recent[0]["year"]
    # A new topic needs sustained evidence since its first observation, for at least two years.
    active = [r for r in recent if r["year"] >= first] if new_topic else recent
    sustained = (len(active) >= 2 and all(r["documents"] >= 2 for r in active) and
                 all(a["share"] is not None and b["share"] is not None and b["share"] > a["share"]
                     for a, b in zip(active, active[1:], strict=False)) and
                 active[-1]["documents"] >= active[-2]["documents"])
    # Heuristic discovery score; weights and other components remain unchanged.
    components = {"growth": min(1.0, max(0.0, math.log2(ratio) / 3)) if ratio else 0.0,
                  "recency": max(0.0, 1 - (years[-1] - first) / 6),
                  "persistence": transitions / 2,
                  "coherence": min(1.0, max(0.0, coherence)),
                  "support": min(1.0, math.log1p(new_n) / math.log(51))}
    weights = {"growth": 35, "recency": 20, "persistence": 15, "coherence": 20, "support": 10}
    score = round(sum(components[key] * weights[key] for key in weights), 2)
    growing = bool(ratio is not None and ratio >= 1.5 and new_n >= 10 and sustained)
    return {"years": rows, "first_observed_year": first,
            "first_observed_year_in_corpus": first,
            "first_observed_year_in_window": min((r["year"] for r in rows if r["documents"]), default=None),
            "baseline_share": old_share,
            "recent_share": new_share, "growth_ratio": ratio, "raw_growth_ratio": raw_ratio,
            "growth_smoothing": alpha, "zero_baseline": zero_baseline, "new_topic_in_window": new_topic,
            "share_change_pp": delta,
            "recent_documents": new_n, "increasing_transitions": transitions, "coherence": round(coherence, 4),
            "growth_pattern": growing, "score": score, "score_components": components, "score_weights": weights}


def prepare_analysis(entries, options, *, model_dir=None, cancel=None, progress=None):
    """Reuse version preparation, with an optional per-run semantic scope adapter.

    Guarded directions retain their original admission and only score prepared
    studies. Generic directions precompute a lookup for the same version selector.
    Nothing is written to the source corpus, and no model is loaded in lexical mode.
    """
    if options.relevance_mode == "lexical":
        studies, preparation = prepare(entries, options, cancel)
        return studies, preparation, None
    from app.ml.local_encoder import LocalEncoder
    from app.ml.semantic import (
        build_semantic_policy, build_semantic_policy_from_studies, guarded_direction, validate_query,
    )

    checkpoint(cancel)
    validate_query(options.topic)
    encoder = LocalEncoder(model_dir)
    checkpoint(cancel)

    def report(completed, total):
        if progress is not None:
            progress(15 + int(13 * completed / max(1, total)),
                     f"Смысловая проверка локальной моделью: {completed}/{total}")

    if guarded_direction(options.topic):
        studies, preparation = prepare(entries, options, cancel)
        policy = build_semantic_policy_from_studies(
            studies, options.topic, encoder, cancel=cancel, progress=report,
            end_year=options.end_year, source_entries=entries)
    else:
        policy = build_semantic_policy(entries, options.topic, encoder, cancel=cancel, progress=report,
                                       end_year=options.end_year)
        with policy.context():
            studies, preparation = prepare(entries, options, cancel)
    checkpoint(cancel)
    return studies, preparation, policy.summary(studies)


def analyze(corpus, options, *, model_dir=None, cancel=None, progress=None):
    import numpy as np
    from app.ml.text import execution_evidence
    from app.ml.directions import direction_profile, profile_fingerprint
    from sklearn.preprocessing import normalize

    options = AnalysisOptions.model_validate(options)
    progress = progress or (lambda percent, message: None)
    checkpoint(cancel)
    if resolve_topic(options.topic).casefold() != resolve_topic(corpus["topic"]).casefold():
        raise AnalysisInputError("Корпус собран по другому направлению. Выберите его тему или соберите новую историю.")
    progress(10, "Подготовка документов и проверка направления")
    # Later versions must not change conflicts, merges or vocabulary of an earlier window.
    dated_entries, input_selection = analysis_entries(corpus["entries"], options.end_year, cancel)
    studies, preparation, semantic_relevance = prepare_analysis(
        dated_entries, options, model_dir=model_dir, cancel=cancel, progress=progress)
    profile = direction_profile(options.topic)
    photonic = bool(profile and profile["id"] == "photonic_neuromorphic")
    # Annotate retained documents after admission; evidence rules are unchanged.
    if photonic:
        studies = [{**study, "execution": execution_evidence(study["title"], study["abstract"])} for study in studies]
    years = list(range(options.start_year, options.end_year + 1))
    totals = Counter(s["year"] for s in studies)
    year_coverage = coverage(corpus["periods"], years)
    warnings = ["Экспериментальные кандидаты. Новизна и стадия развития требуют содержательной проверки.",
                "Карточки содержат фрагменты источников на исходном языке; автоматический перевод не выполняется.",
                "Фильтр направления основан на словах и связях в предложениях; его точность ещё не измерена."]
    if profile is None:
        warnings.append("Профиль направления не настроен, проверка релевантности ограничена.")
    if semantic_relevance is not None:
        warnings.extend([
            "Используется локальная модель E5 для английских текстов. Сходство не является вероятностью релевантности.",
            "Порог смыслового сходства экспериментальный, независимая калибровка не выполнена. "
            "При добавлении документов только по сходству все группы расширенного корпуса остаются предварительными.",
            "Предметные фильтры известных направлений сохраняются; смысловое сходство не подтверждает исполнителя и цитаты.",
        ])
    assessment = growth_assessment(corpus["periods"], years, {year: totals[year] for year in years},
        preparation=preparation, selection=input_selection,
        undated_records=sum(type(e["document"].get("publication_year")) is not int for e in corpus["entries"]))
    reliable = assessment["growth_data_comparable"]
    warnings.append("Динамика и сопоставимость относятся к пригодным исследованиям сохранённого корпуса, "
                    "а не ко всем публикациям направления в мире.")
    warnings.extend(assessment["data_quality_notes"])
    if not reliable:
        warnings.append("Сопоставимость роста не подтверждена: проверьте блокирующие причины сбора "
                        "и годовые знаменатели направления в диагностике.")
    implementation_hash = implementation_fingerprint()
    config_hash = profile_fingerprint(options.topic)
    fingerprint = _hash(repr((VERSION, options.model_dump(), implementation_hash, config_hash, input_selection,
                             [(s["id"], s["year"], s["title"], s["abstract"], s["versions"]) for s in studies],
                             corpus["periods"], corpus["provenance"])))
    if semantic_relevance is not None:
        fingerprint = _hash(fingerprint + json.dumps(semantic_relevance, sort_keys=True,
                                                    ensure_ascii=False, allow_nan=False))
    result = {"schema_version": 2, "pipeline_version": VERSION, "fingerprint": fingerprint,
              "implementation_fingerprint": implementation_hash, "direction_profile_fingerprint": config_hash,
              "source_relations_fingerprint": input_selection["source_relations_fingerprint"],
              "direction_profile": profile["id"] if profile else None,
              "options": options.model_dump(), "source": corpus["source"], "provenance": corpus["provenance"],
              "preparation": preparation, "coverage": year_coverage, **assessment,
              "temporal_selection": input_selection,
              "direction_counts": [{"year": y, "documents": totals[y]} for y in years],
              "warnings": warnings, "candidates": [], "preliminary_signals": [], "established": [],
              "excluded_off_direction": [], "status": "insufficient_data",
              "selection_summary": {"candidates": 0, "preliminary_signals": 0, "established": 0,
                                    "excluded_off_direction": 0, "requested_top_k": options.top_k,
                                    "shortfall": options.top_k, "eligible_before_limit": 0,
                                    "below_top_ids": [], "duplicate_label_ids": []}}
    if semantic_relevance is not None:
        result["semantic_relevance"] = semantic_relevance
    semantic_decisions = {row["study_id"]: row for row in
                          (semantic_relevance or {}).get("study_decisions", [])}
    if sum(totals[y] for y in years) < 8:
        result["warnings"].append("Для группировки нужно хотя бы 8 подходящих исследований в выбранном окне.")
        return result
    progress(30, "Выделение слов и устойчивых словосочетаний")
    try:
        progress(45, "Локальная тематическая модель")
        fitted = fit_topics(studies, cancel)
    except ValueError:
        result["warnings"].append("Недостаточно различающихся терминов для тематической модели.")
        return result
    matrix, model, memberships = (fitted[key] for key in ("matrix", "model", "memberships"))
    vectorizer, n_topics = fitted["vectorizer"], fitted["n_topics"]
    if fitted["convergence_warning"]:
        warnings.append("Модель достигла лимита итераций; устойчивость групп требует проверки.")
    checkpoint(cancel)
    progress(75, "Расчёт динамики и подбор фрагментов источников")
    features = vectorizer.get_feature_names_out()
    labels = memberships.argmax(axis=1)
    concentration = memberships.max(axis=1) / np.maximum(memberships.sum(axis=1), 1e-12)
    discarded, groups = 0, []
    excluded_groups = Counter()
    for label in range(n_topics):
        checkpoint(cancel)
        indexes = np.flatnonzero((labels == label) & (concentration >= 0.2))
        if len(indexes) < 6:
            discarded += len(indexes)
            excluded_groups["fewer_than_six_studies"] += 1
            continue
        center = normalize(np.asarray(matrix[indexes].mean(axis=0)))[0]
        similarities = np.asarray(matrix[indexes] @ center).ravel()
        ordered = indexes[np.argsort(-similarities, kind="stable")]
        members = [studies[int(i)] for i in ordered]
        if sum(options.start_year <= s["year"] <= options.end_year for s in members) < 4:
            continue
        terms = []
        term_candidates = np.argsort(-model.components_[label], kind="stable")[:250]
        title_frequency = {int(i): sum(str(features[i]) in s["title"].casefold() for s in members)
                           for i in term_candidates}
        term_candidates = sorted(term_candidates, key=lambda i: (
            -(float(model.components_[label, i]) * (1 + math.log1p(title_frequency[int(i)]))), str(features[i])))
        for index in term_candidates:
            term = str(features[index])
            if (" " not in term or term in GENERIC_PHRASES or title_frequency[int(index)] < 2
                    or any(term in t or t in term for t in terms)):
                continue
            terms.append(term)
            if len(terms) == 3:
                break
        if not terms:
            excluded_groups["no_shared_title_phrase"] += 1
            continue
        direction_guard = card_topic_guard(options.topic, terms, members)
        off_direction = direction_guard is not None and direction_guard["axis_check"] == "off_direction"
        values = metrics(members, totals, years, float(similarities.mean()))
        card, annotations, explanations, rejected_quotes = supported_card(members, evidence_card(members), terms)
        limitations = ["Тематическая группа требует проверки: название извлечено автоматически.",
                       "Стадия развития технологии не установлена автоматически."]
        if not reliable:
            limitations.append("Рост не подтверждён сопоставимым покрытием данных.")
        elif assessment["data_quality_notes"]:
            limitations.append("Рост рассчитан по пригодной сохранённой выборке; "
                               "ограничения сбора и подготовки перечислены в диагностике качества данных.")
        if values["growth_ratio"] is None:
            limitations.append("Коэффициент роста не определён: недостаточно данных в окнах сравнения.")
        elif values["zero_baseline"]:
            limitations.append("Нулевая база в сохранённой выборке: G рассчитан со сглаживанием α=0.5; это не доказательство новизны.")
        if values["first_observed_year"] <= options.start_year:
            limitations.append("Тема наблюдается у начала окна или раньше; дата зарождения не установлена.")
        for field, title in (("problem", "проблемы"), ("advantage", "преимущества")):
            if card[field] is None:
                limitations.append(f"Явного описания {title} в отобранных фрагментах не найдено.")
        direct = sum(s.get("execution", {}).get("evidence_level") == "direct" for s in members)
        nominal = sum(s.get("execution", {}).get("evidence_level") == "nominal" for s in members)
        execution = {"evidence_level": ("direct" if direct else "nominal") if photonic else "not_applicable",
                     "direct_documents": direct, "nominal_documents": nominal}
        if photonic and not direct:
            limitations.append("У документов группы есть тематическая связь, но прямой оптический исполнитель не подтверждён.")
        if LATEX.search(" ".join([*terms, *(q["text"] for q in card.values() if q)])):
            limitations.append("В исходных фрагментах есть научная разметка; цитаты сохранены без переписывания.")
        if "term plasticity" in terms:
            limitations.append("Фраза term plasticity неполная; для интерпретации используйте исходные названия работ.")
        sources = [{**{k: s[k] for k in ("id", "title", "url", "year", "doi", "versions")},
                    "evidence_level": s.get("execution", {}).get("evidence_level", "not_applicable")}
                   for s in members[:12]]
        candidate = {"id": _hash("|".join(sorted(s["id"] for s in members)))[:16],
                       "title": " / ".join(terms[:2]), "keywords": terms,
                       "study_count": len(members), "study_ids": sorted(s["id"] for s in members),
                       "metrics": values, "card": card, "sources": sources, "limitations": limitations,
                       "execution": execution,
                       "document_evidence": [{"study_id": s["id"], "url": s["url"], **s["execution"]}
                                             for s in members if "execution" in s],
                       "evidence_annotations": annotations, "explanations": explanations,
                       "rejected_quotes": rejected_quotes,
                       "status": "growth_candidate" if values["growth_pattern"] and reliable else "exploratory_candidate",
                       "stage": "requires_review"}
        if semantic_relevance is not None:
            semantic_only = sum(semantic_decisions[s["id"]]["semantic_only"] for s in members)
            # Additions affect the shared TF-IDF/NMF fit and yearly denominators,
            # including groups whose own members all passed lexical admission.
            corpus_requires_review = semantic_relevance["semantic_only_studies"] > 0
            candidate["semantic_relevance"] = {
                "semantic_only_documents": semantic_only,
                "corpus_requires_review": corpus_requires_review,
                "requires_review": bool(semantic_only or corpus_requires_review),
                "study_decisions": [semantic_decisions[s["id"]] for s in members],
            }
            if semantic_only:
                limitations.append("В группе есть документы, допущенные только по смысловому сходству; "
                                   "релевантность требует проверки до включения в основной TOP.")
            if corpus_requires_review:
                limitations.append("Смысловая проверка расширила общий корпус: измениться могут тематическая модель "
                                   "и годовые знаменатели всех групп. Выдача остаётся предварительной.")
        if direction_guard is not None:
            candidate["direction_guard"] = direction_guard
            if direction_guard["axis_check"] == "partial":
                limitations.append("В кратком названии нет всех признаков направления. "
                                   "Дополнительные признаки найдены в источниках; требуется проверка группы."
                                   if direction_guard["reason"] == "document_support_for_missing_cluster_axis"
                                   else "Для проверки направления группы недостаточно исходного текста.")
        if off_direction:
            excluded_groups["off_direction"] += 1
        groups.append(candidate)
    buckets, selection_summary = partition_candidates(groups, reliable, options.top_k)
    if selection_summary["shortfall"]:
        warnings.append(f"В основном списке {len(buckets['candidates'])} из запрошенных {options.top_k}: "
                        "остальные группы не прошли условия роста, доказательности или объёма. "
                        "Предварительные сигналы и причины исключения показаны отдельно.")
    result.update(**buckets, selection_summary=selection_summary,
                  status="ranked" if any(buckets[key] for key in ("candidates", "preliminary_signals", "established")) else "no_groups",
                  model={"type": "TF-IDF + NMF", "topics": n_topics, "random_state": 42,
                         "iterations": int(model.n_iter_), "max_iter": 250, "tol": 0.001,
                         "concentration_threshold": 0.2, "topic_limit": 32,
                         "low_assignment_documents": int((concentration < 0.2).sum()),
                         "small_group_documents": discarded, "groups_before_top": len(groups),
                         "excluded_groups": dict(excluded_groups)})
    progress(100, "Анализ завершён")
    return result
