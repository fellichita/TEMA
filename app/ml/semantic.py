"""Optional, local semantic additions to generic lexical scope admission.

Callers supply validated, temporally selected entries. Embeddings never replace
the source text or the protected version preparation. Similarity is a retrieval
signal, not a relevance probability or evidence of a scientific claim.
"""

import math
import unicodedata
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType

from app.ml.contracts import AnalysisInputError
from app.ml.corpus import checkpoint
from app.ml.directions import direction_profile
from app.ml.scope_context import ScopeDecision, semantic_scope
from app.ml.text import PROCEEDINGS_TITLE, SERVICE_TITLE, _lexical_scope_check, clean, document_text_issue, resolve_topic, safe_url

SEMANTIC_POLICY_VERSION = "generic-lexical-or-semantic-2"
ABSTRACT_CHARACTER_LIMIT = 6000
MAX_NON_LATIN_LETTER_SHARE = 0.6


def semantic_key(topic, title, abstract):
    """Exact selected-text key, independent of occurrence order and DOI aliases."""
    return resolve_topic(topic).casefold(), clean(title), clean(abstract)


def guarded_direction(topic):
    """Preserve all existing profile guards, including legacy photonic dispatch."""
    query = resolve_topic(topic).casefold()
    return bool(direction_profile(topic) is not None or
                ("photon" in query and ("neuro" in query or "neural" in query)))


def validate_query(topic):
    """Validate script/known aliases before model loading; not language detection."""
    query = resolve_topic(topic)
    if not any("LATIN" in unicodedata.name(char, "") for char in query):
        raise AnalysisInputError("Для семантического анализа укажите направление на английском языке.")
    if any(
            char.isalpha() and "LATIN" not in unicodedata.name(char, "") for char in query):
        raise AnalysisInputError("Неизвестное направление нужно задать на английском языке; перевод не выполняется.")
    return query


def _document_skip_reason(document, end_year, cancel=None):
    # Reuse the preparation constants, without reproducing its identity/version
    # handling or changing which source version prepare ultimately chooses.
    from app.ml.engine import TYPES

    oversized = document_text_issue(document)
    if oversized:
        return oversized

    year = document.get("publication_year")
    title, abstract = clean(document.get("title")), clean(document.get("abstract"))
    checkpoint(cancel)
    if type(year) is not int or not 1900 <= year <= end_year:
        return "unknown_or_future_year"
    if document.get("document_type") not in TYPES:
        return "unsupported_document_type"
    if len(title.split()) < 3 or SERVICE_TITLE.search(title) or PROCEEDINGS_TITLE.search(title):
        return "service_or_short_title"
    if len(abstract.split()) < 15:
        return "missing_or_short_abstract"
    if len(abstract.split()) > 3000:
        return "oversized_abstract_requires_review"
    if not safe_url(document.get("url")):
        return "invalid_source_url"
    return _language_skip_reason(document, abstract)


def _declared_nonenglish(language):
    if language is None:
        return False
    if not isinstance(language, str):
        return True
    value = language.strip().casefold().replace("_", "-")
    return value not in {"", "und", "unknown", "en", "eng", "english"} and not value.startswith("en-")


def _language_skip_reason(document, abstract):
    if _declared_nonenglish(document.get("language")):
        return "declared_non_english_language"
    if not abstract.isascii():
        letters = [char for char in abstract if char.isalpha()]
        non_latin = sum("LATIN" not in unicodedata.name(char, "") for char in letters)
        if letters and non_latin / len(letters) > MAX_NON_LATIN_LETTER_SHARE:
            return "predominantly_non_latin_abstract"
    return None


class SemanticDecision(ScopeDecision):
    unscored_reason: str | None


@dataclass(frozen=True)
class SemanticPolicy:
    topic: str
    threshold: float
    guarded: bool
    scores: Mapping[tuple[str, str, str], float]
    model_manifest: dict
    input_occurrences: int
    eligible_occurrences: int
    unscored_reasons: Mapping[tuple[str, str, str], str]
    skipped_occurrences: Mapping[str, int]
    end_year: int

    def context(self):
        return semantic_scope(self)

    def decision(self, topic: str, title: str, abstract: str) -> SemanticDecision:
        key = semantic_key(topic, title, abstract)
        lexical = _lexical_scope_check(topic, key[1], key[2]) == "direct_lexical_signal"
        similarity = self.scores.get(key) if key[0] == self.topic else None
        semantic_only = bool(not lexical and not self.guarded and key[0] == self.topic
                             and similarity is not None and similarity >= self.threshold)
        return {"admitted": lexical or semantic_only,
                "route": "lexical" if lexical else "semantic" if semantic_only else "rejected",
                "lexical_admitted": lexical, "semantic_only": semantic_only,
                "similarity": similarity, "scored": similarity is not None,
                "requires_review": semantic_only,
                "unscored_reason": None if similarity is not None else self.unscored_reasons.get(key, "text_not_scored")}

    def summary(self, studies):
        decisions = [{"study_id": study["id"], **self.decision(
            self.topic, study["title"], study["abstract"])} for study in sorted(studies, key=lambda s: s["id"])]
        routes = Counter(item["route"] for item in decisions)
        return {"mode": "semantic_assisted", "policy": "lexical_or_semantic_generic",
                "policy_version": SEMANTIC_POLICY_VERSION, "threshold": self.threshold,
                "calibrated": False, "similarity_kind": "cosine", "guarded_direction": self.guarded,
                "model": deepcopy(self.model_manifest),
                "text_policy": {"title": "once", "abstract_characters": ABSTRACT_CHARACTER_LIMIT},
                "language_policy": {"supported_language": "English",
                                    "uses_declared_language": True, "language_autodetection": False,
                                    "unknown_latin_language": "not_verified",
                                    "max_non_latin_abstract_letter_share": MAX_NON_LATIN_LETTER_SHARE},
                "publication_year_bounds": {"minimum": 1900, "maximum": self.end_year},
                "input_occurrences": self.input_occurrences, "eligible_occurrences": self.eligible_occurrences,
                "skipped_occurrences_by_reason": dict(sorted(self.skipped_occurrences.items())),
                "blocked_texts_by_reason": dict(sorted(Counter(self.unscored_reasons.values()).items())),
                "scored_texts": len(self.scores), "retained_studies": len(decisions),
                "lexical_studies": routes["lexical"], "semantic_only_studies": routes["semantic"],
                "unscored_studies": sum(not item["scored"] for item in decisions),
                "study_decisions": decisions}


def _checked_vectors(values, count):
    import numpy as np

    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AnalysisInputError("Локальный encoder вернул некорректные эмбеддинги.") from error
    if matrix.shape != (count, 384) or not np.isfinite(matrix).all():
        raise AnalysisInputError("Локальный encoder вернул неверную размерность или нечисловые эмбеддинги.")
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=0.001, rtol=0):
        raise AnalysisInputError("Локальный encoder вернул ненормализованные эмбеддинги.")
    return matrix


def build_semantic_policy(entries, topic, encoder, threshold=0.82, cancel=None, progress=None, *, end_year=None):
    """Encode unique eligible texts once; callers must exclude future entries first.

    ``encoder`` implements encode(texts, kind=..., cancel=..., progress=...) and
    manifest(). No fallback or model download is performed here. Scope admission
    uses an exact cleaned key even when the encoded abstract is length limited.
    """
    checkpoint(cancel)
    query = validate_query(topic)
    end_year = datetime.now(UTC).year - 1 if end_year is None else end_year
    if type(end_year) is not int or not 1900 <= end_year <= 9998:
        raise AnalysisInputError("Некорректный последний год семантического анализа.")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise AnalysisInputError("Порог семантического сходства должен быть конечным числом от 0 до 1.")
    key_set: set[tuple[str, str, str]] = set()
    eligible_keys: Counter[tuple[str, str, str]] = Counter()
    unscored: dict[tuple[str, str, str], str] = {}
    language_blocked: set[tuple[str, str, str]] = set()
    skipped: Counter[str] = Counter()
    occurrences = 0
    for entry in entries:
        checkpoint(cancel)
        occurrences += 1
        document = entry["document"]
        oversized = document_text_issue(document)
        if oversized:
            skipped[oversized] += 1
            continue
        key = semantic_key(topic, document.get("title"), document.get("abstract"))
        checkpoint(cancel)
        reason = _document_skip_reason(document, end_year, cancel)
        if reason is None:
            key_set.add(key)
            eligible_keys[key] += 1
        else:
            skipped[reason] += 1
            unscored[key] = min(reason, unscored.get(key, reason))
            if reason == "declared_non_english_language":
                language_blocked.add(key)
    # scope_check has text arguments, not a source identity. If identical text
    # carries conflicting language metadata, do not silently admit either version
    # semantically. The legacy lexical rule and source records remain unchanged.
    for key in language_blocked & key_set:
        skipped["conflicting_language_metadata"] += eligible_keys[key]
        unscored[key] = "conflicting_language_metadata"
    scored_keys = key_set - language_blocked
    keys = sorted(scored_keys)
    eligible = sum(eligible_keys[key] for key in keys)
    unscored = {key: reason for key, reason in unscored.items() if key not in scored_keys}
    scores = {}
    if keys:
        query_vector = _checked_vectors(encoder.encode([query], kind="query", cancel=cancel, progress=None), 1)[0]
        checkpoint(cancel)
        passages = [title + ". " + abstract[:ABSTRACT_CHARACTER_LIMIT] for _, title, abstract in keys]
        vectors = _checked_vectors(encoder.encode(passages, kind="passage", cancel=cancel, progress=progress), len(keys))
        checkpoint(cancel)
        scores = {key: max(-1.0, min(1.0, float(vector @ query_vector)))
                  for key, vector in zip(keys, vectors, strict=True)}
    checkpoint(cancel)
    return SemanticPolicy(resolve_topic(topic).casefold(), float(threshold), guarded_direction(topic),
                          MappingProxyType(scores), deepcopy(encoder.manifest()), occurrences, eligible,
                          MappingProxyType(unscored), MappingProxyType(dict(skipped)), end_year)


def build_semantic_policy_from_studies(studies, topic, encoder, threshold=0.82, cancel=None, progress=None, *,
                                       end_year=None, source_entries=None):
    """Score already prepared studies for guarded-profile diagnostics only."""
    studies = list(studies)
    selected = {semantic_key(topic, study["title"], study["abstract"]) for study in studies}
    languages = {}
    for entry in source_entries or ():
        checkpoint(cancel)
        document = entry["document"]
        if not document_text_issue(document) and _declared_nonenglish(document.get("language")):
            key = semantic_key(topic, document.get("title"), document.get("abstract"))
            if key in selected:
                languages[key] = document["language"]
    entries = ({"document": {"title": study["title"], "abstract": study["abstract"],
                              "publication_year": study.get("year"),
                              "language": languages.get(semantic_key(topic, study["title"], study["abstract"]),
                                                        study.get("language")),
                              "document_type": study["type"], "url": study["url"]}} for study in studies)
    return build_semantic_policy(entries, topic, encoder, threshold, cancel, progress, end_year=end_year)
