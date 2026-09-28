"""Versioned local-encoder relevance scores for the unified publication TOP.

An E5 cosine is mapped to a bounded ordinal scale. It is useful for ordering
publications within this search, not a calibrated probability of correctness
or the methodology's separate confidence in a technology trend.
"""

from __future__ import annotations

from concurrent.futures import CancelledError
import hashlib
import json
import math
import re
from threading import Event, Timer
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.pilot.encoder import EncoderError

SCORING_VERSION = "publication-e5-cosine/1.1.0"
MAX_PUBLICATIONS = 15
SCORING_TIMEOUT_SECONDS = 120
COSINE_ZERO = 0.75
COSINE_HUNDRED = 0.90


class PublicationConfidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    score: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1, max_length=500)
    evidence_quote: str = Field(min_length=1, max_length=300)
    basis: Literal["title", "title_and_summary"]


class _Cancellation(Protocol):
    def is_set(self) -> bool: ...


class _ScoreCancellation:
    def __init__(self, run_cancel: _Cancellation, deadline: Event):
        self.run_cancel = run_cancel
        self.deadline = deadline

    def is_set(self) -> bool:
        return self.run_cancel.is_set() or self.deadline.is_set()


def _normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def ordinal_score(cosine: float) -> int | None:
    """Fixed linear scale: cosine .75 -> 0 and .90 -> 100, clipped outside.

    Rounding is half-up for nonnegative scaled values. The fixed anchors make
    scores comparable across searches using this exact encoder and version.
    """
    if isinstance(cosine, bool) or not isinstance(cosine, (int, float)):
        return None
    similarity = float(cosine)
    if not math.isfinite(similarity) or not -1 <= similarity <= 1:
        return None
    scaled = max(0.0, min(100.0, (similarity - COSINE_ZERO) * 100 / (COSINE_HUNDRED - COSINE_ZERO)))
    return math.floor(scaled + 0.5)


def validate_confidence(value: object, title: object, summary: object) -> dict[str, Any] | None:
    """Show only structurally valid scores with an actual source excerpt."""
    if not isinstance(title, str) or not title.strip():
        return None
    try:
        confidence = PublicationConfidence.model_validate(value)
    except (ValidationError, ValueError, TypeError):
        return None
    expected_basis = "title_and_summary" if isinstance(summary, str) and summary.strip() else "title"
    if confidence.basis != expected_basis:
        return None
    quote = _normalized(confidence.evidence_quote)
    if not quote or not any(quote in _normalized(text) for text in (title, summary)
                            if isinstance(text, str) and text.strip()):
        return None
    return confidence.model_dump(mode="json")


def _input_hash(query: str, english_query: str | None, publication: dict[str, Any]) -> str:
    material = {"version": SCORING_VERSION, "query": query, "english_query": english_query,
                "publication_id": publication["publication_id"],
                "title": publication["title"], "summary": publication.get("summary")}
    return hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _checkpoint_stage(publication_id: str) -> str:
    return "publication_confidence_" + publication_id[:16]


def _reason(cosine: float) -> str:
    return (f"Смысловая близость по локальной модели E5: {cosine:.3f} "
            "(лучшее совпадение с исходным или английским запросом). "
            "Порядковая шкала: 0,75 → 0; 0,90 → 100. Это не вероятность.")


def _prepared_scores(selected: list[dict[str, Any]], query: str, english_query: str | None,
                     context: Any, fingerprint: str | None
                     ) -> tuple[dict[str, dict[str, Any]], list[tuple[dict[str, Any], str, str]]]:
    output: dict[str, dict[str, Any]] = {}
    pending: list[tuple[dict[str, Any], str, str]] = []
    for publication in selected:
        publication_id = publication.get("publication_id")
        if not isinstance(publication_id, str) or re.fullmatch(r"[a-f0-9]{64}", publication_id) is None:
            continue
        if not isinstance(publication.get("title"), str) or not publication["title"].strip():
            continue
        digest = _input_hash(query, english_query, publication)
        stage = _checkpoint_stage(publication_id)
        saved = context.load_checkpoint(stage)
        if (isinstance(saved, dict) and saved.get("input_hash") == digest
                and (fingerprint is None or saved.get("model_fingerprint") == fingerprint)):
            confidence = validate_confidence(saved.get("model_confidence"),
                                             publication["title"], publication.get("summary"))
            if confidence is not None:
                output[publication_id] = confidence
                continue
        pending.append((publication, digest, stage))
    return output, pending


def completed_cached_scores(publications: list[dict[str, Any]], *, query: str,
                            english_query: str | None, expected_fingerprint: str,
                            context: Any) -> dict[str, dict[str, Any]] | None:
    """Return a complete verified CPU checkpoint without loading model weights."""
    selected = publications[:MAX_PUBLICATIONS]
    # Most analyses have no saved scores. Avoid hashing and reading every
    # checkpoint on that cold path; the normal scorer will read them once.
    for publication in selected:
        publication_id = publication.get("publication_id")
        title = publication.get("title")
        if (isinstance(publication_id, str) and re.fullmatch(r"[a-f0-9]{64}", publication_id)
                and isinstance(title, str) and title.strip()):
            if context.load_checkpoint(_checkpoint_stage(publication_id)) is None:
                return None
            break
    else:
        return {}
    output, pending = _prepared_scores(selected, query, english_query,
                                       context, expected_fingerprint)
    return output if not pending else None


def score_publications(publications: list[dict[str, Any]], *, query: str,
                       english_query: str | None = None, encoder: Any | None,
                       context: Any) -> dict[str, dict[str, Any]]:
    """Score at most 15 preliminary TOP items in one bounded, local batch.

    Every completed item is checkpointed. A resume reuses only scores for the
    same query/title/summary/version and the same encoder fingerprint.
    """
    selected = publications[:MAX_PUBLICATIONS]
    fingerprint = getattr(encoder, "fingerprint", None)
    output, pending = _prepared_scores(selected, query, english_query, context, fingerprint)
    if encoder is None or not pending:
        return output

    deadline = Event()
    timer = Timer(SCORING_TIMEOUT_SECONDS, deadline.set)
    timer.daemon = True
    timer.start()
    cancellation = _ScoreCancellation(context.cancel_event, deadline)
    completed = len(selected) - len(pending)
    try:
        context.progress("publish", f"Оцениваем публикации: {completed} из {len(selected)}",
                         completed, len(selected))
        passages = []
        for publication, _, _ in pending:
            title = publication["title"][:1000]
            summary = publication.get("summary")
            summary = summary[:1000] if isinstance(summary, str) and summary.strip() else None
            passages.append(title + (". " + summary if summary else ""))
        query_texts = list(dict.fromkeys(text.strip()[:2000] for text in (query, english_query)
                                         if isinstance(text, str) and text.strip()))
        if not query_texts:
            return output
        query_vectors = encoder.encode(query_texts, kind="query", cancel=cancellation)
        publication_vectors = encoder.encode(passages, kind="passage", cancel=cancellation)
        context.check_cancelled()
        if deadline.is_set():
            return output
        for (publication, digest, stage), vector in zip(pending, publication_vectors, strict=True):
            context.check_cancelled()
            similarity = max(sum(float(left) * float(right)
                                 for left, right in zip(query_vector, vector, strict=True))
                             for query_vector in query_vectors)
            score = ordinal_score(similarity)
            if score is None:
                continue
            title = publication["title"]
            summary = publication.get("summary")
            candidate = {"score": score, "reason": _reason(similarity),
                         "evidence_quote": title.strip()[:300].strip(),
                         "basis": "title_and_summary" if isinstance(summary, str) and summary.strip() else "title"}
            confidence = validate_confidence(candidate, title, summary)
            if confidence is None:
                continue
            context.checkpoint(stage, {"input_hash": digest, "model_fingerprint": fingerprint,
                                       "model_confidence": confidence})
            output[publication["publication_id"]] = confidence
            completed += 1
            context.progress("publish", f"Оцениваем публикации: {completed} из {len(selected)}",
                             completed, len(selected))
    except CancelledError:
        context.check_cancelled()  # User cancellation becomes TaskCancelled.
        # Otherwise our per-stage timeout expired; preserve completed scores.
    except (EncoderError, OSError, ValueError, TypeError):
        context.check_cancelled()
    finally:
        timer.cancel()
    return output
