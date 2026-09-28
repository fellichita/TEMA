"""Technology identity lookup: exact reviewed aliases, then review-only proposals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import Event
from typing import Protocol
from uuid import UUID

import numpy as np

from app.pilot.multisource.contracts import TechnologyConcept
from app.pilot.multisource.queries import normalize_term
from app.runtime.jobs import TaskCancelled, TaskFailure

MAX_CONCEPTS = 100
MAX_PROPOSALS = 5


class TextEncoder(Protocol):
    def encode(self, texts: list[str] | tuple[str, ...], *, kind: str = "passage",
               cancel: Event | None = None, progress=None) -> object: ...


@dataclass(frozen=True)
class ExactMatch:
    state: str
    concept_id: UUID | None
    candidate_ids: tuple[UUID, ...]


@dataclass(frozen=True)
class SemanticProposal:
    concept_id: UUID
    similarity: float
    status: str = "proposed"


def _key(value: str) -> str:
    return normalize_term(value).casefold()


def exact_confirmed_match(term: str, concepts: tuple[TechnologyConcept, ...], *,
                          cutoff: datetime | None = None) -> ExactMatch:
    """Acronyms and aliases are never inferred from spelling alone."""
    if len(concepts) > MAX_CONCEPTS:
        raise TaskFailure("Слишком много технологий для сопоставления.")
    key = _key(term)
    found = set()
    for concept in concepts:
        if concept.identity_status != "confirmed":
            continue
        if cutoff is not None and (concept.confirmed_at is None or concept.confirmed_at > cutoff):
            continue
        if key == _key(concept.label) or any(alias.status == "confirmed" and key == _key(alias.text)
                                             and (cutoff is None or alias.confirmed_at is not None
                                                  and alias.confirmed_at <= cutoff) for alias in concept.aliases):
            found.add(concept.concept_id)
    candidates = tuple(sorted(found, key=str))
    if len(candidates) == 1:
        return ExactMatch("exact_confirmed", candidates[0], candidates)
    return ExactMatch("ambiguous" if candidates else "none", None, candidates)


def propose_semantic_matches(term: str, concepts: tuple[TechnologyConcept, ...], encoder: TextEncoder,
                             *, cancel: Event | None = None) -> tuple[SemanticProposal, ...]:
    """Local embeddings rank review suggestions; their scores never confirm identity."""
    if len(concepts) > MAX_CONCEPTS:
        raise TaskFailure("Слишком много технологий для сопоставления.")
    if cancel is not None and cancel.is_set():
        raise TaskCancelled()
    if not concepts:
        return ()
    phrase = normalize_term(term)
    ordered = tuple(sorted(concepts, key=lambda item: str(item.concept_id)))
    inputs = [phrase, *(item.label + ". " + item.definition[:500] for item in ordered)]
    vectors = np.asarray(encoder.encode(inputs, kind="query", cancel=cancel), dtype=np.float32)
    if vectors.shape != (len(inputs), 384) or not np.isfinite(vectors).all():
        raise TaskFailure("Модель сопоставления вернула некорректные векторы.")
    norms = np.linalg.norm(vectors, axis=1)
    if (norms <= 0).any():
        raise TaskFailure("Модель сопоставления вернула пустой вектор.")
    scores = (vectors[1:] @ vectors[0]) / (norms[1:] * norms[0])
    if cancel is not None and cancel.is_set():
        raise TaskCancelled()
    ranked = sorted(zip(ordered, scores, strict=True), key=lambda pair: (-float(pair[1]),
                                                                          str(pair[0].concept_id)))
    return tuple(SemanticProposal(item.concept_id, float(score)) for item, score in ranked[:MAX_PROPOSALS])
