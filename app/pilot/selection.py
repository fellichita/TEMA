"""A single reproducible TOP policy for generation, review, import and display."""

from __future__ import annotations

from math import isfinite

from app.pilot.contracts import TrendCard
from app.pilot.evidence import TITLE_ADMISSION_VERSION, normalize_title

SIGNAL_CATEGORIES = frozenset({"early_signal", "confirmed_trend", "weak_signal_candidate", "emerging_candidate"})
SELECTION_VERSION = "evidence-gated-top/3.3.0"


def _name(text: str) -> str:
    return " ".join(word[:-1] if len(word) > 4 and word.endswith("s")
                    and not word.endswith(("ss", "us", "is")) else word
                    for word in normalize_title(text).split())


def concept_keys(card: TrendCard) -> set[str]:
    """Only full, frozen technology names; shared words never merge siblings."""
    candidate = card.candidate
    keys = {_name(candidate.label)}
    if candidate.specificity == "specific_technology" and candidate.admission_rule_version == TITLE_ADMISSION_VERSION:
        keys.update(_name(phrase) for phrase in candidate.synonyms if len(normalize_title(phrase).split()) >= 2)
    return keys


def select_top(cards, artifacts, *, limit: int = 15) -> tuple[str, ...]:
    if type(limit) is not int or not 1 <= limit <= 15:
        raise ValueError("TOP limit must be an integer between 1 and 15")
    assessments = {item.assessment.candidate_id: item.assessment for item in artifacts}
    candidates = []
    stages = {"early_signal": 0, "weak_signal_candidate": 1, "confirmed_trend": 2, "emerging_candidate": 3}
    for card in cards:
        assessed = assessments.get(card.candidate.candidate_id)
        if (card.category not in SIGNAL_CATEGORIES or card.methodology_version not in {"3.2.0", "3.3.0", "3.4.0"}
                or card.candidate.specificity != "specific_technology" or assessed is None
                or assessed.methodology_version != card.methodology_version or assessed.category != card.category
                or card.assessment_hash != assessed.assessment_hash or assessed.signal_priority is None
                or not isfinite(assessed.signal_priority)):
            continue
        candidates.append((card, assessed))
    candidates.sort(key=lambda pair: (stages[pair[0].category], -pair[1].signal_priority,
                                      pair[0].candidate.candidate_id))
    identifiers: list[str] = []
    seen: set[str] = set()
    for card, _ in candidates:
        keys = concept_keys(card)
        if keys & seen:
            continue
        seen.update(keys)
        identifiers.append(card.candidate.candidate_id)
        if len(identifiers) == limit:
            break
    return tuple(identifiers)
