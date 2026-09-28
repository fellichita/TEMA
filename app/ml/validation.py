"""Publication-date retrospective of the frozen NMF ranking, never a UI quality score.

The protocol is fixed before observing outcomes. Only training publications may
define features, topics, labels, eligible groups, scores and the baseline order.
"""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import warnings

from app.ml.contracts import AnalysisOptions
from app.ml.engine import GENERIC_PHRASES, VERSION, _document_identity, coverage, growth_assessment, metrics, prepare
from app.ml.model import texts_for_studies
from app.ml.text import title_key
from app.ml.study_relations import first_author_surname


PROTOCOL = {
    "version": "publication-date-retrospective-1",
    "train_publication_year_max": 2019,
    "train_scoring_years": list(range(2014, 2020)),
    "target_early_years": [2020, 2021],
    "target_recent_years": [2022, 2023],
    "target_smoothed_relative_growth_min": 1.5,
    "target_recent_documents_min": 10,
    "smoothing_alpha": 0.5,
    "requested_k": 15,
    "assignment_concentration_min": 0.2,
    "eligible_train_group_min": 6,
    "eligible_train_scoring_window_min": 4,
    "baseline": "smoothed absolute count growth 2017–2019 / 2014–2016",
    "eligibility": "train-only NMF groups with shared title phrases; no evidence-card or stage gate",
    "model_parameters": {"max_topics": 32, "topics_rule": "min(32, floor(sqrt(n)), n-1, features)",
                         "random_state": 42, "max_iter": 250, "tol": .001, "init": "nndsvda",
                         "max_features": 18000, "ngram_range": [1, 3], "min_df": 2, "max_df": .9},
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _version_keys(study):
    return {study["id"]} | {v["document_key"] for v in study.get("versions", [])}


def _title_author(study):
    key = title_key(study["title"])
    authors = study.get("authors") or []
    surname = first_author_surname(authors)
    return (key, tuple(surname)) if len(key.split()) >= 6 and surname else None


def exclude_known_versions(train, future, *, raw_train_entries=()):
    """Remove later copies of train work without retroactively changing train."""
    # A publication was already known even when its early annotation was too
    # short for modelling. Future fuller text must not make it a new work.
    known = list(train)
    for entry in raw_train_entries:
        document = entry["document"]
        known.append({"id": _document_identity(entry),
                      "title": document["title"], "authors": document.get("authors") or [],
                      "versions": [{"document_key": entry["document_key"]}]})
    identities = set().union(*(_version_keys(s) for s in known)) if known else set()
    titles = {_title_author(s) for s in known} - {None}
    retained, excluded = [], []
    for study in future:
        reason = ("train_identity" if _version_keys(study) & identities else
                  "train_long_title_first_author" if _title_author(study) in titles else None)
        if reason:
            excluded.append({"id": study["id"], "reason": reason})
        else:
            retained.append(study)
    return retained, excluded


def target_outcome(counts, totals):
    """Measured zero topic counts differ from an unobserved direction denominator."""
    early, recent = PROTOCOL["target_early_years"], PROTOCOL["target_recent_years"]

    def valid_number(value):
        try:
            return (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0)
        except OverflowError:
            return False

    if any(not valid_number(counts.get(y, 0)) for y in early + recent):
        return {"growing": None, "growth_ratio": None, "early_documents": None,
                "recent_documents": None, "reason": "invalid_topic_counts"}
    n0, n1 = sum(counts.get(y, 0) for y in early), sum(counts.get(y, 0) for y in recent)
    if not valid_number(n0) or not valid_number(n1):
        return {"growing": None, "growth_ratio": None, "early_documents": None,
                "recent_documents": None, "reason": "invalid_topic_counts"}
    if any(not valid_number(totals.get(y)) or totals[y] <= 0 for y in early + recent):
        return {"growing": None, "growth_ratio": None, "early_documents": n0,
                "recent_documents": n1, "reason": "unknown_direction_denominator"}
    d0, d1 = sum(totals[y] for y in early), sum(totals[y] for y in recent)
    if not valid_number(d0) or not valid_number(d1):
        return {"growing": None, "growth_ratio": None, "early_documents": n0,
                "recent_documents": n1, "reason": "unknown_direction_denominator"}
    if n0 == n1 == 0:
        return {"growing": False, "growth_ratio": None, "early_documents": n0,
                "recent_documents": n1, "reason": "absent_topic"}
    alpha = PROTOCOL["smoothing_alpha"]
    ratio = (((n1 + alpha) / (d1 + 2 * alpha)) / ((n0 + alpha) / (d0 + 2 * alpha)))
    if n0 and not n1:
        ratio = 0.0
    if not math.isfinite(ratio):
        return {"growing": None, "growth_ratio": None, "early_documents": n0,
                "recent_documents": n1, "reason": "growth_ratio_out_of_range"}
    return {"growing": bool(ratio >= PROTOCOL["target_smoothed_relative_growth_min"]
                            and n1 >= PROTOCOL["target_recent_documents_min"]),
            "growth_ratio": ratio, "early_documents": n0, "recent_documents": n1,
            "reason": "growth_in_saved_sample"}


def ranking_quality(ranking, outcomes, requested_k=15):
    selected = ranking[:requested_k]
    labels = [outcomes[key]["growing"] for key in selected]
    known = sum(v is not None for v in labels)
    hits = sum(v is True for v in labels)
    precision = hits / len(selected) if selected and known == len(selected) else None
    return {"k": len(selected), "requested_k": requested_k, "hits": hits,
            "outcomes_known": known, "precision_at_k": precision,
            "precision_at_15": precision if requested_k == 15 and len(selected) == 15 else None,
            "coverage_at_15": len(selected) / 15, "selected_ids": selected}


def compare_rankings(ranking, baseline_ranking, outcomes):
    proposed = ranking_quality(ranking, outcomes)
    baseline = ranking_quality(baseline_ranking, outcomes)
    p, b = proposed["precision_at_k"], baseline["precision_at_k"]
    return {"proposed": proposed, "baseline": baseline,
            "lift": p / b if p is not None and b is not None and b > 0 else None,
            "selected_overlap": len(set(proposed["selected_ids"]) & set(baseline["selected_ids"])),
            "ranking_can_discriminate_at_k": len(ranking) > proposed["k"]}


def _training_groups(studies, fitted, totals):
    import numpy as np
    from sklearn.preprocessing import normalize

    memberships, matrix = fitted["memberships"], fitted["matrix"]
    labels = memberships.argmax(axis=1)
    concentration = memberships.max(axis=1) / np.maximum(memberships.sum(axis=1), 1e-12)
    features, model = fitted["vectorizer"].get_feature_names_out(), fitted["model"]
    groups = []
    years = PROTOCOL["train_scoring_years"]
    for label in range(fitted["n_topics"]):
        indexes = np.flatnonzero((labels == label) & (concentration >= .2))
        if len(indexes) < 6:
            continue
        members = [studies[int(i)] for i in indexes]
        if sum(years[0] <= s["year"] <= years[-1] for s in members) < 4:
            continue
        center = normalize(np.asarray(matrix[indexes].mean(axis=0)))[0]
        similarities = np.asarray(matrix[indexes] @ center).ravel()
        terms = []
        candidates = np.argsort(-model.components_[label], kind="stable")[:250]
        frequency = {int(i): sum(str(features[i]) in s["title"].casefold() for s in members)
                     for i in candidates}
        candidates = sorted(candidates, key=lambda i: (
            -(float(model.components_[label, i]) * (1 + math.log1p(frequency[int(i)]))), str(features[i])))
        for i in candidates:
            term = str(features[i])
            if (" " not in term or term in GENERIC_PHRASES or frequency[int(i)] < 2
                    or any(term in t or t in term for t in terms)):
                continue
            terms.append(term)
            if len(terms) == 3:
                break
        if not terms:
            continue
        values = metrics(members, totals, years, float(similarities.mean()))
        counts = Counter(s["year"] for s in members)
        old = sum(counts[y] for y in years[:3])
        recent = sum(counts[y] for y in years[3:])
        baseline = (recent + .5) / (old + .5)
        if old and not recent:
            baseline = 0.0
        groups.append({"id": _digest(sorted(s["id"] for s in members))[:16], "topic_index": label,
                       "title": " / ".join(terms[:2]), "keywords": terms,
                       "study_ids": sorted(s["id"] for s in members), "study_count": len(members),
                       "metrics": values, "baseline_absolute_growth": baseline})
    return groups


def run_retrospective(corpus, *, fit_topics_fn=None):
    """Evaluate a frozen ranking on later publications; input must be a validated corpus."""
    import numpy as np
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits

    if fit_topics_fn is None:
        from app.ml.model import fit_topics
        fit_topics_fn = fit_topics
    train_entries, future_entries = [], []
    for entry in corpus["entries"]:
        year = entry["document"].get("publication_year")
        if type(year) is not int:
            continue
        if year <= 2019:
            train_entries.append(entry)
        elif 2020 <= year <= 2023:
            future_entries.append(entry)
    train, train_preparation = prepare(train_entries, AnalysisOptions(
        topic=corpus["topic"], start_year=2014, end_year=2019))
    future, future_preparation = prepare(future_entries, AnalysisOptions(
        topic=corpus["topic"], start_year=2020, end_year=2023))
    future, cross_window_versions = exclude_known_versions(train, future, raw_train_entries=train_entries)
    train_totals, future_totals = Counter(s["year"] for s in train), Counter(s["year"] for s in future)
    train_coverage = coverage(corpus["periods"], PROTOCOL["train_scoring_years"])
    future_coverage = coverage(corpus["periods"], range(2020, 2024))
    undated_records = sum(type(e["document"].get("publication_year")) is not int for e in corpus["entries"])
    train_assessment = growth_assessment(corpus["periods"], PROTOCOL["train_scoring_years"],
        {y: train_totals[y] for y in PROTOCOL["train_scoring_years"]}, preparation=train_preparation,
        undated_records=undated_records)
    future_assessment = growth_assessment(corpus["periods"], range(2020, 2024),
        {y: future_totals[y] for y in range(2020, 2024)}, preparation=future_preparation,
        undated_records=undated_records)
    metadata_reasons = []
    if any(train_totals[y] == 0 for y in PROTOCOL["train_scoring_years"]):
        metadata_reasons.append("train:unknown_direction_denominator")
    if any(future_totals[y] == 0 for y in range(2020, 2024)):
        metadata_reasons.append("holdout:unknown_direction_denominator")
    for name, diagnostic in (("train", train_preparation), ("holdout", future_preparation)):
        if diagnostic["possible_versions_merged"]:
            metadata_reasons.append(name + ":possible_versions_merged")
        for reason in ("conflicting_year", "missing_or_short_abstract"):
            if diagnostic["rejected"].get(reason):
                metadata_reasons.append(name + ":" + reason)
    fetched = sorted(e["document"]["fetched_at"] for e in train_entries + future_entries
                     if isinstance(e["document"].get("fetched_at"), str))
    implementation = {"pipeline_version": VERSION, "numpy_version": np.__version__,
                      "sklearn_version": sklearn.__version__}
    train_fingerprint = _digest({"protocol": PROTOCOL, "studies": train, "coverage": train_coverage,
                                 "implementation": implementation})
    report = {
        "schema_version": 1, "status": "exploratory_retrospective", "protocol": deepcopy(PROTOCOL),
        "protocol_sha256": _digest(PROTOCOL), "implementation": implementation,
        "scope": "NMF ranking-only pilot, not evaluation of the complete application or evidence cards",
        "topic": corpus["topic"], "source": corpus["source"], "provenance": corpus["provenance"],
        "train_fingerprint": train_fingerprint, "train_preparation": train_preparation,
        "holdout_preparation": future_preparation, "holdout_train_versions_excluded": cross_window_versions,
        "train_counts": {str(y): train_totals[y] for y in range(2012, 2020)},
        "holdout_counts": {str(y): future_totals[y] for y in range(2020, 2024)},
        "train_coverage": train_coverage, "holdout_coverage": future_coverage,
        "data_comparable": train_assessment["growth_data_comparable"] and future_assessment["growth_data_comparable"],
        "growth_comparability": {"train": train_assessment["growth_comparability"],
                                  "holdout": future_assessment["growth_comparability"]},
        "data_quality_notes": [f"{name}: {note}" for name, assessment in
                               (("train", train_assessment), ("holdout", future_assessment))
                               for note in assessment["data_quality_notes"]],
        "metadata_limitations": metadata_reasons,
        "fetched_at_range": [fetched[0], fetched[-1]] if fetched else None,
        "limitations": [
            "Retrospective by publication year; metadata were not archived as of the training cutoff.",
            "Partial coverage and version or text selection can distort rare-topic outcomes.",
            "Shared vocabulary and frozen NMF cannot predict topics absent from the training vocabulary.",
            "Rules were developed using later literature; this is not an untouched prospective test.",
            "Small K and a single direction do not establish predictive ability or statistical significance.",
            "Outcome growth in the saved sample does not establish novelty, emerging stage or useful citations.",
        ],
        "groups": [], "model": None, "training_ranking": [], "baseline_ranking": [], "outcomes": {},
    }
    if len(train) < 8:
        report["diagnostic"] = "insufficient_training_studies"
    else:
        try:
            fitted = fit_topics_fn(train)
        except ValueError:
            fitted = None
            report["diagnostic"] = "insufficient_training_vocabulary"
        if fitted is not None:
            groups = _training_groups(train, fitted, train_totals)
            report["groups"] = groups
            report["training_ranking"] = [g["id"] for g in sorted(groups, key=lambda g: (
                -g["metrics"]["score"], -g["metrics"]["recent_documents"], g["id"]))]
            report["baseline_ranking"] = [g["id"] for g in sorted(groups, key=lambda g: (
                -g["baseline_absolute_growth"], -g["metrics"]["recent_documents"], g["id"]))]
            report["model"] = {"n_topics": fitted["n_topics"], "train_studies": len(train),
                               "eligible_train_groups": len(groups),
                               "vocabulary": sorted(fitted["vectorizer"].get_feature_names_out().tolist()),
                               "training_convergence_warning": bool(fitted["convergence_warning"]),
                               "holdout_convergence_warning": False,
                               "holdout_low_assignment": 0, "holdout_zero_features": 0}
            counts = {g["topic_index"]: Counter() for g in groups}
            if future:
                transformed = fitted["vectorizer"].transform(texts_for_studies(future))
                zero_rows = np.asarray(transformed.getnnz(axis=1)) == 0
                with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    memberships = fitted["model"].transform(transformed)
                report["model"]["holdout_convergence_warning"] = any(
                    issubclass(w.category, ConvergenceWarning) for w in caught)
                labels = memberships.argmax(axis=1)
                concentration = memberships.max(axis=1) / np.maximum(memberships.sum(axis=1), 1e-12)
                report["model"]["holdout_low_assignment"] = int((concentration < .2).sum())
                report["model"]["holdout_zero_features"] = int(zero_rows.sum())
                for i, study in enumerate(future):
                    label = int(labels[i])
                    if not zero_rows[i] and concentration[i] >= .2 and label in counts:
                        counts[label][study["year"]] += 1
            report["outcomes"] = {g["id"]: target_outcome(counts[g["topic_index"]], future_totals) |
                                   {"counts": {str(y): counts[g["topic_index"]][y] for y in range(2020, 2024)}}
                                   for g in groups}
    report["evaluation"] = compare_rankings(report["training_ranking"], report["baseline_ranking"], report["outcomes"])
    report["fingerprint"] = _digest(report)
    return report
