"""Describe completed computations without declaring scientific success.

Derived from saved cards, so old packages remain readable and later expert
reviews cannot leave a stale persisted completion flag. A negative, fully
assessed TOP is a valid outcome; an unprocessed hypothesis queue is separate.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal


REJECTION_REASONS = {
    "off_scope_or_mixed": "Именование отклонено: группа вне области запроса или объединяет разные области.",
    "scope_evidence_missing": "Именование отклонено: связь всех представителей с областью не подтверждена цитатами.",
    "definition_rejected": "Предложенное определение не прошло проверку; гипотеза сохранена для уточнения.",
}


def next_candidate_batch(discovered: Mapping[str, Any], processed_ids: set[str] | frozenset[str], *,
                         limit: int = 30) -> tuple[Any, ...]:
    """A deterministic bounded continuation after rejected or incomplete candidates.

    This function allocates no network/model work. The caller must retain its
    overall document/call budget and checkpoint completed IDs before continuing.
    A single-paper mechanism and an overflow cluster share access to review.
    """
    from itertools import zip_longest
    from app.pilot.contracts import Candidate

    if type(limit) is not int or not 1 <= limit <= 30:
        raise ValueError("A candidate batch must contain between 1 and 30 entries")
    groups = list(discovered.get("candidates", ()))
    queued = discovered.get("review_queue", ())
    rare = [item for item in queued if len(item.get("discovery_study_ids", ())) == 1]
    groups.extend(item for item in queued if len(item.get("discovery_study_ids", ())) != 1)
    batch: dict[str, Candidate] = {}
    for pair in zip_longest(groups, rare):
        for raw in pair:
            if raw is None or raw.get("candidate_id") in processed_ids:
                continue
            candidate = Candidate.model_validate(raw)
            batch.setdefault(candidate.candidate_id, candidate)
            if len(batch) == limit:
                return tuple(batch.values())
    return tuple(batch.values())


@dataclass(frozen=True)
class EvaluationProgress:
    state: Literal["no_candidates", "not_assessed", "partially_assessed", "assessed"]
    cards: int
    history_assessed: int
    history_required: int
    pending: int
    unresolved_mechanisms: int
    insufficient_evidence: int
    queued_count: int = 0
    rejected_count: int = 0

    @property
    def message(self) -> str:
        messages = {
            "no_candidates": "Результат сохранён. Карточки кандидатов не сформированы.",
            "not_assessed": "Результат сохранён. Научная оценка кандидатов не завершена.",
            "partially_assessed": "Результат сохранён. Автоматическая проверка выполнена частично.",
            "assessed": "Результат сохранён. Автоматическая проверка выполнена; это не независимая научная валидация.",
        }
        return messages[self.state]

    @property
    def details(self) -> str:
        parts = ([f"Исторические оценки: {self.history_assessed} из {self.history_required}."]
                 if self.cards else [])
        if self.unresolved_mechanisms:
            parts.append(f"Границы механизма требуют проверки: {self.unresolved_mechanisms}.")
        if self.pending:
            parts.append(f"Не завершена проверка кандидатов: {self.pending}; их нельзя считать слабыми сигналами.")
        elif self.insufficient_evidence:
            parts.append(f"Для вывода о стадии недостаточно доказательств у {self.insufficient_evidence} кандидатов.")
        if self.queued_count:
            parts.append(f"Гипотез без паспорта в очереди: {self.queued_count}; проверка всей очереди не завершена.")
            if self.rejected_count:
                parts.append(f"Из них именование отклонено: {self.rejected_count}; причины сохранены.")
        return " ".join(parts)


def evaluation_progress(result: Mapping[str, Any]) -> EvaluationProgress:
    """Count saved historical assessments, not scores, TOP size or popularity.

    Off-scope rejection does not need a history; insufficient evidence after a
    performed assessment is not an execution failure. Coverage remains in the
    passports and does not by itself mean that processing was interrupted.
    """
    cards = result.get("cards", ())
    history = required = pending = unresolved = insufficient = 0
    for card in cards:
        if card.get("category") == "off_scope":
            continue
        required += 1
        assessed = bool(card.get("assessment_hash") and card.get("historical_snapshot_id"))
        history += assessed
        uncertain = card.get("candidate", {}).get("specificity") in {"uncertain", "broad_topic"}
        unresolved += uncertain
        pending += not assessed or uncertain or card.get("category") == "unassessed_cluster"
        insufficient += card.get("category") == "insufficient_evidence"
    state: Literal["no_candidates", "not_assessed", "partially_assessed", "assessed"]
    if not cards:
        state = "no_candidates"
    elif pending:
        state = "not_assessed" if pending == len(cards) and not history else "partially_assessed"
    else:
        state = "assessed"
    queued_ids = {candidate.get("candidate_id") for candidate in result.get("candidate_queue", ())}
    rejected = sum(item.get("outcome") == "definition_rejected" and item.get("candidate_id") in queued_ids
                   for item in result.get("candidate_review_states") or ())
    return EvaluationProgress(state, len(cards), history, required, pending, unresolved, insufficient,
                              len(result.get("candidate_queue", ())), rejected)


def label_stage_limitations(checkpoints: Iterable[Mapping[str, Any]],
                            unresolved_ids: set[str]) -> tuple[str, ...]:
    """Aggregate safe failure codes, never echo arbitrary remote error text.

    New runs record exact fallback IDs. Older checkpoints can be read using
    their frozen candidates; retained, grounded definitions are not failures.
    """
    groups: dict[str, set[str]] = {}
    for checkpoint in checkpoints:
        fallback = checkpoint.get("fallback_candidate_ids")
        if fallback is None:
            fallback = [item["candidate_id"] for item in checkpoint.get("candidates", ())
                        if item.get("specificity") != "specific_technology"]
        affected = unresolved_ids.intersection(fallback)
        if not affected:
            continue
        code = checkpoint.get("failure_code")
        if code in {"budget_exceeded", "budget_state_error"}:
            reason = "AI-именование не выполнено из-за ограничения доступного бюджета; проверьте лимиты и незавершённые резервы в разделе расходов"
        elif code in {"read_timeout", "connect_timeout", "write_timeout", "pool_timeout", "response_deadline"}:
            reason = "AI-именование не завершено: превышено время ожидания ответа"
        elif code in {"transport_failed", "transport_unavailable"}:
            reason = "AI-именование не завершено: полный ответ провайдера не получен"
        elif code in {"invalid_response", "response_too_large"}:
            reason = "AI-именование не подтверждено: ответ провайдера не прошёл проверку"
        elif code == "evidence_rejected":
            reason = "Связь механизма с областью запроса не подтверждена текстом источников; требуется уточнение гипотезы"
        elif code == "mechanism_alias_required":
            reason = "Для истории требуется название конкретного механизма; полное название единственной статьи не принято как имя технологии"
        elif code == "external_ai_not_allowed":
            reason = "Передача документов AI не разрешена; технологические границы требуют локальной проверки"
        else:
            reason = "Технологические границы не подтверждены: использованы непроверенные локальные названия"
        groups.setdefault(reason, set()).update(affected)
    return tuple(f"В этом запуске: {reason}. Кандидатов: {len(identifiers)}."
                 for reason, identifiers in groups.items())
