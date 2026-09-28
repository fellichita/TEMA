"""One complete desktop analysis workflow, with real sources and inspectable evidence."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, wait as wait_futures
from collections.abc import Callable
from datetime import date, datetime, timedelta, UTC
import json
import os
from pathlib import Path
import re
import tempfile
from threading import Event, Lock
from typing import Any, Literal, Protocol, cast
import unicodedata
from uuid import uuid4

from pydantic import ValidationError

from app.pilot.archive import DocumentArchive
from app.pilot.contracts import AnalysisResult, Candidate, CorpusSnapshot, QueryLimits, QueryPlan, content_hash
from app.pilot.llm import LlmCancelled, LlmClient, LlmError
from app.pilot.query import (EXPLICIT_QUERY_VERSION, LOCAL_TRANSLATION_QUERY_VERSION, QUERY_PROMPT_VERSION,
                             QueryClarificationRequired, QueryError, normalize_query, plan_query,
                             scope_is_unrelated)
from app.pilot.settings import PROFILE_HISTORY_LIMITS, CollectionProfile, PilotSettings, apply_collection_profile, load_settings, save_settings
from app.pilot.sources import collect_snapshot
from app.runtime.budget import BudgetError, BudgetLimits, BudgetService
from app.runtime.credentials import CredentialStore, CredentialUnavailable
from app.runtime.jobs import Coordinator, RunContext, TaskCancelled, TaskFailure
from app.runtime.worker import WorkerCancelled, WorkerError, run_in_process

WORKFLOW_VERSION = "pilot-v3.8-20260924-crossref-search"
# 1.1.0: source records are stored without third-party link tokens; artifacts
# cached before that still carry them and fail the final credential check.
SHARED_SOURCE_CACHE_VERSION = "exact-source-artifacts/1.1.0"
SHARED_SOURCE_CACHE_TTL = timedelta(hours=6)
# Записей на один дополнительный источник в глубоком режиме (быстрый — около 35).
DEEP_SOURCE_CAP = 100
# Кандидатов радара в глубоком режиме (быстрый — 60). Кандидаты идут по частоте,
# а самые частые фразы почти всегда зрелые: 28.09.2026 из 60 зрелыми были 53.
DEEP_RADAR_CANDIDATES = 120
# A local answer is free, so money is the wrong fence for it: what the paid caps
# really bound is the number of requests a provider needs. The paid naming stage
# describes four clusters per request, the local one a single cluster plus a
# possible corrective retry, so the same 24 calls covered only a third of the
# bounded 60 attempts and left the rest with an unverified lexical name. An
# unnamed candidate is never specific_technology, never receives a historical
# assessment and can never enter the TOP, which is why a local run reported an
# empty list. Requests and tokens stay bounded; the user's cost caps are kept
# unchanged and simply never bind, because the local tariff is zero.
LOCAL_RUN_CALLS, LOCAL_RUN_INPUT_TOKENS, LOCAL_RUN_OUTPUT_TOKENS = 400, 8_000_000, 500_000
LOCAL_DAY_CALLS, LOCAL_DAY_INPUT_TOKENS, LOCAL_DAY_OUTPUT_TOKENS = 2400, 48_000_000, 3_000_000


def _run_budget_limits(provider: str, limits: QueryLimits, cost_micro: int) -> BudgetLimits:
    """Fence one analysis by what its provider actually costs the user.

    A saved plan that admits no AI work at all keeps its zero caps whatever the
    provider is: a replay or refinement of such a result must not start asking
    a model now.
    """
    if provider == "local" and all((limits.llm_calls, limits.input_tokens, limits.output_tokens)):
        return BudgetLimits(LOCAL_RUN_CALLS, LOCAL_RUN_INPUT_TOKENS, LOCAL_RUN_OUTPUT_TOKENS, cost_micro)
    return BudgetLimits(limits.llm_calls, limits.input_tokens, limits.output_tokens, cost_micro)


def _day_budget_limits(provider: str, cost_micro: int) -> BudgetLimits:
    """Fence one calendar day of the same provider, which owns its own scope."""
    if provider == "local":
        return BudgetLimits(LOCAL_DAY_CALLS, LOCAL_DAY_INPUT_TOKENS, LOCAL_DAY_OUTPUT_TOKENS, cost_micro)
    return BudgetLimits(96, 800000, 120000, cost_micro)


def _source_snapshot_cacheable(snapshot: CorpusSnapshot) -> bool:
    """Do not turn a temporary provider or credential failure into reusable data."""
    return not any(reason.startswith("source_") or reason == "credential_storage_unavailable"
                   for coverage in snapshot.coverage for reason in coverage.reasons)


def _stored_signal_base(base: dict | None, expected_id: str | None, requested_id: str) -> dict:
    if base is None or requested_id != expected_id:
        raise TaskFailure("Связанный научный результат отсутствует.")
    return base


def _signal_transfer_rights(share_confirmed: bool, license_ref: str | None) -> tuple[Literal["share_allowed", "local_only"], str | None]:
    from app.pilot.multisource.contracts import validate_import_export_right

    if type(share_confirmed) is not bool or license_ref is not None and not isinstance(license_ref, str):
        raise TaskFailure("Некорректное подтверждение права передачи источника.")
    right: Literal["share_allowed", "local_only"] = "share_allowed" if share_confirmed else "local_only"
    reference = license_ref.strip() if share_confirmed and license_ref is not None else None
    if not share_confirmed and license_ref not in (None, ""):
        raise TaskFailure("Для права передачи требуется отдельное подтверждение.")
    try:
        validate_import_export_right(right, reference)
    except ValueError:
        raise TaskFailure("Для передачи источника подтвердите право и укажите ссылку или основание лицензии.") from None
    return right, reference
_REJECTED_REFINEMENT = (
    "Повторное уточнение этой гипотезы не дало подтверждённого определения. "
    "Сохранены прежние определение, доказательства и оценка; нового подтверждения нет."
)
_LEGACY_REJECTED_REFINEMENT = (
    "Повторное именование отклонило выбранную группу. Её прежняя оценка сохранена без нового подтверждения; исходный результат не изменён."
)


def _reviewed_refinement_notice(card):
    # Bind the notice without modifying the exact card attributed to an expert.
    identity = content_hash({"candidate_id": card.candidate.candidate_id})[:16]
    return f"Карточка «{card.candidate.label}» [{identity}]: " + _REJECTED_REFINEMENT


def _retain_refinement_notices(source, cards, replaced, rejected_id, reviewed_ids):
    """Carry failed-refinement provenance with the retained candidate, not the job.

    Expert cards are immutable ReviewRecord subjects. Their same static notice
    stays result-level and candidate-bound; other cards can carry it directly.
    No remote failure text or prior scientific decisions are rewritten.
    """
    notices = []
    retained = []
    for card in cards:
        identifier = card.candidate.candidate_id
        if identifier not in replaced:
            notice = _reviewed_refinement_notice(card)
            if notice in source.limitations:
                notices.append(notice)
            if identifier == rejected_id:
                if identifier in reviewed_ids:
                    notices.append(notice)
                else:
                    card = card.model_copy(update={"limitations": tuple(dict.fromkeys(
                        (*card.limitations, _REJECTED_REFINEMENT)))})
        retained.append(card)
    return retained, tuple(dict.fromkeys(notices))


def _day_budget_scope(budget: BudgetService, settings: PilotSettings) -> str:
    """Resolve today's scope on the coordinator without resetting saved headroom."""
    scope = "day/" + settings.provider + "/" + datetime.now(UTC).date().isoformat()
    limits = _day_budget_limits(settings.provider, settings.day_cost_micro)
    row = budget.connection.execute("SELECT currency,max_calls,max_input,max_output FROM pilot_budget_scopes "
                                    "WHERE scope_id=?", (scope,)).fetchone()
    if row is None:
        budget.create_scope(scope, limits, currency=settings.currency)
    elif row[0] != settings.currency:
        raise TaskFailure("Валюта сохранённого дневного бюджета отличается от настроек.")
    elif settings.provider == "local" and tuple(row[1:]) != (limits.calls, limits.input_tokens, limits.output_tokens):
        # A day opened under the paid request fence must not keep it: no money
        # was ever at stake here, and the old cap silently truncates naming.
        budget.update_limits(scope, limits)
    return scope


def _configure_day_budget(connection, provider: str, currency: str, cost_micro: int):
    """Only an explicit changed preference may increase an existing day's cap."""
    scope = "day/" + provider + "/" + datetime.now(UTC).date().isoformat()
    row = connection.execute("SELECT currency FROM pilot_budget_scopes WHERE scope_id=?", (scope,)).fetchone()
    if row is not None:
        if row[0] != currency:
            raise TaskFailure("Нельзя изменить валюту существующего бюджетного периода.")
        BudgetService(connection).update_limits(scope, _day_budget_limits(provider, cost_micro))


class _Cancellation(Protocol):
    def is_set(self) -> bool: ...


class _StartCancellation:
    def __init__(self, requested: Event, shutdown: Event):
        self.requested, self.shutdown = requested, shutdown

    def is_set(self) -> bool:
        return self.requested.is_set() or self.shutdown.is_set()


class OfflineContext:
    def __init__(self, cancel: _Cancellation):
        self.cancel_event = cancel

    def check_cancelled(self):
        if self.cancel_event.is_set():
            raise TaskCancelled()

    def progress(self, *_):
        self.check_cancelled()


def _rank_cards(cards, artifacts):
    assessments = {item.assessment.candidate_id: item.assessment for item in artifacts}
    categories = {"early_signal": 0, "weak_signal_candidate": 1, "confirmed_trend": 2, "emerging_candidate": 3,
                  "insufficient_evidence": 4, "unassessed_cluster": 5, "renewed_interest": 6,
                  "established_topic": 7, "transient_burst": 8, "declining": 9, "off_scope": 10}
    confidence = {"high": 0, "medium": 1, "low": 2}

    def key(card):
        assessed = assessments.get(card.candidate.candidate_id)
        priority = (assessed.signal_priority if assessed and assessed.methodology_version in {"3.2.0", "3.3.0", "3.4.0"}
                    and assessed.signal_priority is not None else assessed.priority_lower_bound if assessed else 0)
        return (categories[card.category], -priority,
                confidence[assessed.confidence] if assessed else 3, card.candidate.candidate_id)

    cards.sort(key=key)


def _candidate_pool(discovered: dict, limit: int) -> tuple[Candidate, ...]:
    """Give clustered and rare hypotheses equal access before expensive review.

    The remainder is retained in AnalysisResult.candidate_queue, not discarded or
    relabelled as weak signals. A bounded pool limits network/model costs.
    """
    from itertools import zip_longest

    groups = list(discovered.get("candidates", []))
    queued = discovered.get("review_queue", [])
    # Overflow clusters must not consume the paper-level lane before individual
    # mechanisms receive review. A paper inside a cluster has the same access as
    # one labelled noise; neither is a positive weak signal at discovery time.
    rare = [item for item in queued if len(item.get("discovery_study_ids", ())) == 1]
    groups.extend(item for item in queued if len(item.get("discovery_study_ids", ())) != 1)
    pool: dict[str, Candidate] = {}
    for pair in zip_longest(groups, rare):
        for raw in pair:
            if raw is not None:
                candidate = Candidate.model_validate(raw)
                pool.setdefault(candidate.candidate_id, candidate)
                if len(pool) >= limit:
                    return tuple(pool.values())
    return tuple(pool.values())


def _history_allocations(candidates: tuple[Candidate, ...], total: int, per_candidate: int, *,
                         include_unresolved: bool = False) -> dict[str, int]:
    """Stable shares cannot depend on the incoming order or the first cluster size."""
    eligible = sorted(item.candidate_id for item in candidates
                      if include_unresolved or item.specificity == "specific_technology")
    if not eligible:
        return {}
    share, extra = divmod(total, len(eligible))
    return {identifier: min(per_candidate, share + (index < extra)) for index, identifier in enumerate(eligible)}


def _history_subbudgets(allocation: int) -> tuple[int, int]:
    """Automatic antecedents and the recent series share one bounded allocation."""
    if allocation < 2:
        return (0, 0)
    earlier = min(500, max(1, allocation // 4))
    return earlier, allocation - earlier


def _distinct_cards(cards):
    """Merge whole-name spelling/plural variants, preserving all modifiers/order.

    A shared parent word or an alias bag is not proof that mechanisms coincide.
    Only new, specifically admitted technology names receive plural folding.
    """
    from app.pilot.evidence import TITLE_ADMISSION_VERSION, normalize_title

    kept, seen = [], set()
    for card in cards:
        candidate = card.candidate
        words = normalize_title(candidate.label).split()
        if candidate.specificity == "specific_technology" and candidate.admission_rule_version == TITLE_ADMISSION_VERSION:
            words = [word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is"))
                     else word for word in words]
        name = " ".join(words)
        if name in seen:
            continue
        seen.add(name)
        kept.append(card)
    return kept


def _passport_ai_plan(candidates, snapshot, archive, context, budget, scope_ids, config):
    """Choose feasible AI passports before iteration, across support strata.

    Plan with conservative per-request upper bounds in every run/day dimension;
    dispatch still performs the authoritative atomic reservation. Cached work,
    uncertain groups and documents without public AI permission consume no slot.
    Unspent headroom is not assigned opportunistically to later input positions.
    """
    from app.pilot.evidence import PassportDraft, member_documents, representative_documents
    from app.pilot.hierarchy import balanced_groups
    from app.runtime.budget import RequestAllowance
    if config.provider == "local":
        from app.pilot.local_client import LOCAL_ANSWER_TOKENS

        # LocalLlmClient caps this same request at LOCAL_ANSWER_TOKENS. Its
        # ledger reserves the capped amount, so a paid-provider estimate would
        # unnecessarily remove otherwise feasible passports from the plan.
        output_cap = min(3000, LOCAL_ANSWER_TOKENS)
    else:
        output_cap = 3000

    snapshots = [budget.snapshot(scope) for scope in scope_ids]
    if not snapshots or any(item.requires_reconciliation or item.currency != config.currency for item in snapshots):
        return {}
    names = ("calls", "input_tokens", "output_tokens", "cost_micro")
    remaining = [min(getattr(item.remaining, name) for item in snapshots) for name in names]
    schema_text = json.dumps(PassportDraft.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
    groups = [{"members": candidate.discovery_study_ids, "candidate": candidate}
              for candidate in candidates if candidate.specificity == "specific_technology"]
    ordered = balanced_groups(groups, score=lambda _: 0, key=lambda group: group["candidate"].candidate_id)
    selected = {}
    for group in ordered:
        context.check_cancelled()
        candidate = group["candidate"]
        stage = "passport_" + content_hash({"id": candidate.candidate_id})[:20]
        if context.load_checkpoint(stage) is not None:
            continue
        documents = representative_documents(member_documents(candidate, snapshot, archive, context), limit=5)
        if not all(document.source in {"openalex", "crossref", "arxiv"} for _, document in documents):
            continue
        payload = {"candidate": candidate.label, "definition": candidate.definition,
                   "documents": [{"revision_id": reference.revision_id, "title": document.title[:1500],
                                  "abstract": (document.abstract or "")[:3500]} for reference, document in documents]}
        # Account for the second JSON escaping of user/schema strings in chat
        # messages too (quotes and backslashes may dominate real source text).
        # 8192 bytes additionally bound the system instruction and 1024-byte
        # framing reserve; dispatch retains the authoritative budget fence.
        messages = [{"role": "system", "content": schema_text},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        input_cap = len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 8192
        allowance = RequestAllowance(input_cap, output_cap, config.quote(input_cap, output_cap))
        required = (1, allowance.input_tokens, allowance.output_tokens, allowance.cost_micro)
        if all(amount <= available for amount, available in zip(required, remaining, strict=True)):
            selected[candidate.candidate_id] = allowance
            remaining = [available - amount for amount, available in zip(required, remaining, strict=True)]
    return selected


def _passport_ai_selection(candidates, snapshot, archive, context, budget, scope_ids, config, *, settings_hash,
                           checkpoint_stage="passport_ai_plan"):
    """Freeze paid-selection identity across interruption/resume without new slots."""
    policy = "balanced-feasible-budget/1.0.0"
    input_hash = content_hash({"policy": policy, "snapshot": snapshot.snapshot_hash, "settings": settings_hash,
        "candidates": [candidate.model_dump(mode="json") for candidate in
                       sorted(candidates, key=lambda item: item.candidate_id)]})
    saved = context.load_checkpoint(checkpoint_stage)
    if saved is not None:
        identifiers = saved.get("selected_candidate_ids")
        if (saved.get("policy") != policy or saved.get("input_hash") != input_hash
                or not isinstance(identifiers, list) or not all(isinstance(item, str) for item in identifiers)
                or len(identifiers) != len(set(identifiers))
                or not set(identifiers).issubset(candidate.candidate_id for candidate in candidates
                                               if candidate.specificity == "specific_technology")):
            raise TaskFailure("Сохранённый план AI-паспортов относится к другим кандидатам или настройкам.")
        return frozenset(identifiers)
    chosen = (_passport_ai_plan(candidates, snapshot, archive, context, budget, scope_ids, config)
              if config is not None else {})
    identifiers = sorted(chosen)
    context.checkpoint(checkpoint_stage, {"policy": policy, "input_hash": input_hash,
                                          "selected_candidate_ids": identifiers})
    return frozenset(identifiers)


class _SharedLocalModel:
    """Одна загруженная локальная модель на все анализы процесса.

    Одновременные анализы не держат по копии весов в видеопамяти: вычисления
    идут по очереди, а сетевые этапы соседнего анализа идут в это время. Закрыть
    её может только владелец (служба), не отдельный анализ.
    """

    def __init__(self, model: Any):
        self._model = model
        self._lock = Lock()
        self.spec = model.spec
        self.provider = model.provider

    @property
    def fingerprint(self) -> str:
        return self._model.fingerprint

    @property
    def model_id(self) -> str:
        return self._model.model_id

    def prompt_tokens(self, *args: Any, **kwargs: Any) -> list[int]:
        with self._lock:
            return self._model.prompt_tokens(*args, **kwargs)

    def generate(self, **kwargs: Any) -> Any:
        with self._lock:
            return self._model.generate(**kwargs)

    def generate_many(self, requests: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._model.generate_many(requests, **kwargs)

    def prepare(self, requests: Any, **kwargs: Any) -> None:
        with self._lock:
            self._model.prepare(requests, **kwargs)

    def close(self) -> None:
        """Анализ отпускает модель, но выгружает её только служба."""

    def unload(self) -> None:
        with self._lock:
            self._model.close()


class _BackgroundContext:
    """Контекст фоновой задачи запуска: общая отмена, но ни строчки в базу.

    Базу анализов пишет только поток координатора, поэтому фоновая работа не
    сообщает ход и не сохраняет этапы — это делает основной поток, когда
    забирает её результат. Её собственная отмена не отменяет весь анализ.
    """

    def __init__(self, context: RunContext):
        self.run_id, self.attempt = context.run_id, context.attempt
        self._main = context.cancel_event
        self.cancel_event = Event()

    def check_cancelled(self) -> None:
        if self._main.is_set() or self.cancel_event.is_set():
            raise TaskCancelled()

    def progress(self, stage: str, message: str, completed: int = 0, total: int = 0) -> None:
        self.check_cancelled()


def _await_background(context: RunContext, future: Future) -> Any:
    """Результат фоновой задачи или None, если она не удалась; анализ можно отменить."""
    while not wait_futures((future,), timeout=0.5).done:
        context.check_cancelled()
    try:
        return future.result()
    except TaskCancelled:
        context.check_cancelled()
        return None
    except Exception:
        return None


def _await_radar(context: RunContext, future: Future, progress: list[int]) -> dict:
    """Дождаться фонового ТОПа технологий, показывая его собственный ход.

    Сбой радара не отменяет научный результат: веб покажет, что ТОП недоступен.
    """
    shown = None
    while not wait_futures((future,), timeout=0.5).done:
        done, total = progress
        if (done, total) != shown:
            shown = done, total
            context.progress("radar", f"Досчитываем ТОП-15 технологий: {min(done, total)} из {total}"
                             if total else "Досчитываем ТОП-15 технологий", min(done, total), total)
        else:
            context.check_cancelled()
    try:
        return future.result()
    except TaskCancelled:
        raise
    except Exception:
        context.check_cancelled()
        return {"state": "unavailable", "message": "Не удалось собрать ТОП технологий."}


def _freeze_processing_checkpoint(context, stage, value):
    saved = context.load_checkpoint(stage)
    if saved is not None and saved != value:
        raise TaskFailure("Сохранённая очередь проверки относится к другим кандидатам или лимитам.")
    if saved is None:
        context.checkpoint(stage, value)
    return value


def _candidate_attempt_plan(discovered, plan, snapshot, context, *, settings_hash, attempts=60):
    """Reserve deterministic document shares for at most ``attempts`` (≤60) distinct attempts.

    Unnamed slots retain their shares for later batches. Neither rejections nor
    interruption move those shares to the first accepted passports; unused
    reservations cause no source requests or AI charges.
    """
    candidates = _candidate_pool(discovered, attempts)
    allocations = _history_allocations(candidates, plan.limits.new_historical_documents,
        plan.limits.historical_documents_per_candidate, include_unresolved=True)
    frozen = {"policy": "bounded-attempts/1.0.0", "input_hash": content_hash({
        "candidates": [item.model_dump(mode="json") for item in candidates],
        "plan": plan.plan_hash, "snapshot": snapshot.snapshot_hash, "settings": settings_hash}),
        "candidate_ids": [item.candidate_id for item in candidates], "history_allocations": allocations}
    _freeze_processing_checkpoint(context, "candidate_attempt_plan", frozen)
    return candidates, allocations, frozen["input_hash"]


def _candidate_batch_outcome(context, candidates, accepted, *, offset, plan_hash):
    """Bind safe portable outcomes to the persisted naming checkpoints."""
    from app.pilot.completion import REJECTION_REASONS

    original = {item.candidate_id: item for item in candidates}
    kept = {item.candidate_id: item for item in accepted}
    if len(kept) != len(accepted) or not kept.keys() <= original.keys():
        raise TaskFailure("Именование изменило набор выбранных гипотез.")
    for identifier, item in kept.items():
        source = original[identifier]
        if (item.discovery_study_ids != source.discovery_study_ids
                or item.discovery_snapshot_id != source.discovery_snapshot_id or item.plan_hash != source.plan_hash):
            raise TaskFailure("Именование изменило исходные исследования гипотезы.")
    rejected, checkpoints = {}, []
    for index in range(offset, offset + len(candidates)):
        stage = f"labels_{index}"
        saved = context.load_checkpoint(stage)
        if saved is None:
            continue
        if not set(saved.get("input_candidate_ids", ())) <= original.keys():
            raise TaskFailure("Сохранённое именование относится к другой партии гипотез.")
        checkpoints.append({"stage": stage, "hash": content_hash(saved)})
        for item in saved.get("rejected", ()):
            if item["candidate_id"] in original:
                code = item.get("reason_code", "definition_rejected")
                rejected[item["candidate_id"]] = code if code in REJECTION_REASONS else "definition_rejected"
    outcome: dict[str, Any] = {"input_hash": content_hash({"plan": plan_hash, "offset": offset,
        "candidates": [item.model_dump(mode="json") for item in candidates]}),
        "input_candidate_ids": list(original), "accepted_candidate_ids": list(kept),
        "rejected": {identifier: rejected.get(identifier, "definition_rejected")
                     for identifier in original if identifier not in kept}, "label_checkpoints": checkpoints}
    stage = f"candidate_batch_outcome_{offset}"
    _freeze_processing_checkpoint(context, stage, outcome)
    return {identifier: {"candidate_id": identifier, "source_run_id": context.run_id,
        "attempt_ordinal": offset + index, "stage": stage, "stage_hash": content_hash(outcome),
        "input_hash": outcome["input_hash"],
        "outcome": "passport_created" if identifier in kept else "definition_rejected",
        "reason_code": outcome["rejected"].get(identifier)}
        for index, identifier in enumerate(original)}


def _top_trend_ids(cards, limit: int = 15) -> tuple[str, ...]:
    return tuple(card.candidate.candidate_id for card in cards if card.category == "confirmed_trend")[:limit]


def _antecedent_shortlist(cards, artifacts, eligible_ids: set[str], *, limit: int) -> tuple[str, ...]:
    """Rank cheap assessments before any older-reference source requests."""
    from app.pilot.selection import concept_keys

    assessments = {item.assessment.candidate_id: item.assessment for item in artifacts}
    ranked = []
    for card in cards:
        identifier = card.candidate.candidate_id
        assessment = assessments.get(identifier)
        if (identifier not in eligible_ids or assessment is None
                or card.candidate.specificity != "specific_technology"):
            continue
        unresolved = tuple(reason for reason in assessment.gate_failures
                           if reason != "earlier_search_not_complete")
        ranked.append((len(unresolved), -assessment.priority_upper_bound,
                       -assessment.priority_lower_bound, -assessment.recent_studies,
                       identifier, card))
    ranked.sort(key=lambda item: item[:-1])
    selected = []
    seen: set[str] = set()
    for *_, identifier, card in ranked:
        keys = concept_keys(card)
        if keys & seen:
            continue
        seen.update(keys)
        selected.append(identifier)
        if len(selected) == limit:
            break
    return tuple(selected)


def _retained_snapshots(snapshots, cards, queue, artifacts, *, patent_ids=()):
    """Keep live provenance; superseded refinement snapshots stay in the source result."""
    required = set(patent_ids)
    required.update(item.discovery_snapshot_id for item in queue)
    for card in cards:
        required.add(card.candidate.discovery_snapshot_id)
        if card.historical_snapshot_id is not None:
            required.add(card.historical_snapshot_id)
    for artifact in artifacts:
        if artifact.inputs.antecedents is not None:
            required.add(artifact.inputs.antecedents.snapshot.snapshot_id)
    by_id = {item.snapshot_id: item for item in snapshots}
    if not required:
        required.add(next(item.snapshot_id for item in snapshots if item.purpose == "discovery"))
    retained = [item for key, item in by_id.items() if key in required]
    covered = {ref.revision_id for item in retained for ref in item.documents}
    missing = {evidence.revision_id for card in cards for evidence in card.evidence} - covered
    missing.update(reference.revision_id for artifact in artifacts
                   for reference in artifact.inputs.publication_status_revisions if reference.revision_id not in covered)
    for item in reversed(tuple(by_id.values())):
        if missing.intersection(ref.revision_id for ref in item.documents):
            if item.snapshot_id not in required:
                retained.append(item)
                required.add(item.snapshot_id)
            missing.difference_update(ref.revision_id for ref in item.documents)
    if missing:
        raise TaskFailure("Сохранённые свидетельства не покрыты исходными снимками.")
    return tuple(retained)


def _reconcile_publication_status(cards, artifacts, snapshots, plan, archive, context):
    """Apply later source facts before publishing, keeping unaffected passports frozen."""
    if not any(card.methodology_version == "3.4.0" for card in cards):
        return artifacts
    from app.pilot.history import apply_publication_status, assess_snapshot
    from app.pilot.methodology import AssessmentArtifact
    from app.pilot.publication_status import collect_status_context

    statuses = collect_status_context(tuple(reference for snapshot in snapshots for reference in snapshot.documents),
                                      archive, context, as_of=plan.as_of)
    blocked = statuses.withdrawn | statuses.supporting
    by_candidate = {item["assessment"]["candidate_id"]: AssessmentArtifact.model_validate(item) for item in artifacts}
    by_snapshot = {item.snapshot_id: item for item in snapshots}
    for index, card in enumerate(cards):
        context.check_cancelled()
        context.progress("reconcile", f"Сверяем статусы публикаций: {index + 1} из {len(cards)}", index, len(cards))
        if card.methodology_version != "3.4.0":
            continue
        passport = apply_publication_status(card, statuses.withdrawn, statuses.supporting)
        artifact = by_candidate.get(card.candidate.candidate_id)
        if artifact is None:
            cards[index] = passport
            continue
        inputs = artifact.inputs
        observed = {study for year in inputs.history.observations for study in year.study_ids}
        observed.update(item.result.study_id for item in inputs.primary_observations)
        observed.update(item.novelty.study_id for item in inputs.source_novelty)
        if inputs.history.first_observed_study_id is not None:
            observed.add(inputs.history.first_observed_study_id)
        if passport == card and not observed.intersection(blocked):
            continue
        revised, revised_card, _ = assess_snapshot(card.candidate, plan, by_snapshot[card.historical_snapshot_id],
            archive, context, passport=passport, methodology_version="3.4.0", verified_novelty=inputs.novelty,
            field_exposure=inputs.field_exposure, antecedents=inputs.antecedents,
            source_novelty=inputs.source_novelty, primary_observations=inputs.primary_observations,
            publication_status_revisions=statuses.references)
        cards[index] = revised_card
        by_candidate[card.candidate.candidate_id] = revised
    context.progress("reconcile", "Статусы публикаций сверены", len(cards), len(cards))
    return [by_candidate[item["assessment"]["candidate_id"]].model_dump(mode="json") for item in artifacts]


class PilotService:
    def __init__(self, data_dir: Path, credentials: CredentialStore, *, model_dir: Path | None = None):
        from app.pilot.encoder import load_spec, model_directory
        from app.pilot.library import ResultLibrary
        from app.runtime.model_resources import resolve_model

        self.data_dir = data_dir.resolve()
        self.credentials = credentials
        self.credentials.import_legacy_environment()
        self.archive = DocumentArchive(self.data_dir / "revisions")
        self._library = ResultLibrary(self.data_dir, self.archive)
        self.model_location = resolve_model("multilingual-e5-small", load_spec()["revision"],
                                            explicit_dir=model_dir, development_default=model_directory(self.data_dir))
        self.model_dir = self.model_location.path
        # This profile's own weights. Without an explicit directory the model
        # would resolve to the default profile, and a test or a second profile
        # would silently load someone else's 1.8 GB copy.
        from app.pilot.local_llm import model_directory as local_llm_directory

        self.local_llm_dir = local_llm_directory(self.data_dir)
        self._model_status_cache: tuple[tuple[tuple[int, int, int, int, int] | None, ...], str, str | None] | None = None
        self._translator: object | None = None
        self._translator_lock = Lock()
        self.model_cancel = Event()
        self.view_cancel = Event()
        # Веб подключает сюда свой переводчик: ТОП технологий переводится в фоне анализа.
        self.radar_translator: Callable[[dict, Event], dict] | None = None
        # Веб держит локальную модель загруженной между анализами и делит её
        # между одновременными анализами; настольное приложение грузит её на
        # анализ и выгружает, освобождая видеопамять для энкодера.
        self.keep_llm_loaded = False
        self._shared_llm: _SharedLocalModel | None = None
        self._shared_llm_lock = Lock()
        # Веб не ждёт ТОП технологий, если он не успел к концу анализа: его
        # досчёт передаётся сюда и забирается сайтом (`take_radar`).
        self.radar_handoff = False
        self._handed_radars: dict[str, tuple[Future, list[int], Event]] = {}
        self._handed_lock = Lock()
        self.coordinator = Coordinator(self.data_dir, self._run, workflow_version=WORKFLOW_VERSION)

    def prepare_local_models(self) -> dict:
        """Place models this project already carries where this profile reads them.

        Called once when the interface opens a profile. The bytes come from the
        project's own verified stage, the same folder the native build packs, so
        nothing is fetched and nothing leaves the machine: a new profile opens
        ready instead of asking for an installation this computer already did.

        Only copying is automatic. A model staged nowhere stays missing and the
        interface keeps offering the explicit installation, because pulling
        hundreds of megabytes without being asked is not a startup step. A frozen
        build already carries its models and places nothing.

        Never raises: a profile that cannot be prepared is a missing model, which
        status() reports on its own.
        """
        from app.pilot.encoder import load_spec as scientific_spec, verify_artifacts as verify_scientific
        from app.pilot.translator import load_spec as translation_spec, model_directory as translation_directory
        from app.pilot.translator import verify_artifacts as verify_translation

        placed: dict[str, bool] = {}
        if self.model_location.origin != "development":
            return {"origin": self.model_location.origin, "placed": placed}
        try:
            from scripts.model_staging import install_from_staging

            for key, directory, spec, verify in (
                    ("multilingual-e5-small", self.model_dir, scientific_spec(), verify_scientific),
                    ("opus-mt-ru-en", translation_directory(self.data_dir), translation_spec(), verify_translation)):
                if directory.exists():
                    continue
                placed[key] = bool(install_from_staging(key, directory, spec, verify))
        except Exception:
            pass
        return {"origin": self.model_location.origin, "placed": placed}

    def status(self, *, verify_model: bool = False) -> dict:
        from app.pilot.encoder import EncoderError, load_spec, verify_artifacts

        files = load_spec()["files"]
        signature: list[tuple[int, int, int, int, int] | None] = []
        for item in files:
            try:
                info = (self.model_dir / item["name"]).stat()
                signature.append((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
            except OSError:
                signature.append(None)
        fingerprint = tuple(signature)
        cached = self._model_status_cache
        if cached is not None and cached[0] == fingerprint and not verify_model:
            model_state, model_error = cached[1:]
        else:
            try:
                verify_artifacts(self.model_dir, cancel=self.model_cancel)
                model_state, model_error = "ready", None
            except CancelledError:
                raise
            except EncoderError as error:
                model_state = "corrupt" if any((self.model_dir / item["name"]).exists() for item in files) else "missing"
                model_error = (("Встроенная модель отсутствует. Переустановите приложение." if
                                model_state == "missing" else
                                "Встроенная модель повреждена. Переустановите приложение.") if
                               self.model_location.origin == "bundled" else str(error))
            except (OSError, ValueError):
                model_state = "unavailable"
                model_error = ("Не удалось проверить встроенную модель. Переустановите приложение." if
                               self.model_location.origin == "bundled" else
                               "Не удалось проверить локальную модель.")
            else:
                if self.model_location.origin == "bundled":
                    from app.pilot.encoder import MultilingualEncoder

                    try:
                        encoder = MultilingualEncoder(self.model_dir, cancel=self.model_cancel)
                        del encoder
                    except CancelledError:
                        raise
                    except Exception:
                        model_state, model_error = ("unavailable",
                            "Встроенная модель не загружается. Переустановите приложение.")
            self._model_status_cache = (fingerprint, model_state, model_error)
        keys: dict[str, bool | None] = {}
        key_errors = []
        for name in ("deepseek_api_key", "yandex_api_key", "wordstat_api_key", "openalex_api_key", "epo_ops_key", "epo_ops_secret"):
            try:
                keys[name] = bool(self.credentials.get(name))
            except CredentialUnavailable:
                keys[name] = None
                key_errors.append(name)
        try:
            from app.pilot.local_llm import LocalModelError, artifacts_present

            local_llm_installed = artifacts_present(self.local_llm_dir)
        except (LocalModelError, OSError):
            # A damaged specification is the offline provider's problem, not a
            # reason to leave the interface without a status for everything else.
            local_llm_installed = False
        settings_error = None
        try:
            settings = load_settings(self.data_dir)
        except TaskFailure as error:
            settings = PilotSettings()
            settings_error = str(error)
        return {
            "settings": settings.model_dump(mode="json"),
            "settings_error": settings_error,
            "model_installed": model_state == "ready",
            "model_state": model_state,
            "model_error": model_error,
            "model_origin": self.model_location.origin,
            "model_dir": str(self.model_dir),
            "local_llm_installed": local_llm_installed,
            "keys": keys,
            "key_errors": key_errors,
            "signal_sources": {
                "scientific": "model_ready" if model_state == "ready" else "model_unavailable",
                "arxiv_atom": "local_import_ready",
                "wordstat_csv": "local_import_ready",
                "wordstat_api": ("disabled" if not settings.wordstat_api_enabled else
                                 "key_unavailable" if keys.get("wordstat_api_key") is None else
                                 "key_missing" if not keys.get("wordstat_api_key") else "configured_unverified"),
                "cordis_csv": "local_import_ready",
                "investment_csv": "local_import_ready",
            },
            "persistent_keys_available": self.credentials.storage_status.persistent_available,
        }

    def configure(self, values: dict, keys: dict[str, str] | None = None, *, persistent: bool = False) -> dict:
        settings = PilotSettings.model_validate(values)
        self.credentials.validate_updates(keys if keys is not None else {}, persistent=persistent)
        with self.coordinator.idle_operation():
            try:
                previous = load_settings(self.data_dir)
            except TaskFailure:
                previous = None
            if previous is None or (previous.day_cost_micro, previous.provider) != (settings.day_cost_micro, settings.provider):
                self.coordinator.maintenance(_configure_day_budget, settings.provider, settings.currency, settings.day_cost_micro)
            # Do not store masked placeholder text as a real key. Blank means unchanged.
            try:
                for name, value in (keys or {}).items():
                    if value:
                        self.credentials.set(name, value, persistent=persistent)
            except CredentialUnavailable:
                raise CredentialUnavailable("Не удалось завершить запись ключей в системное хранилище. "
                                            "Часть изменений могла сохраниться; проверьте ключи и журнал бюджета.") from None
            save_settings(self.data_dir, settings)
            return self.status()

    def install_model(self) -> dict:
        if self.model_location.origin == "bundled":
            return self.status(verify_model=True)
        from scripts.install_pilot_model import install

        OfflineContext(self.model_cancel).check_cancelled()
        install(self.model_dir, cancel=self.model_cancel)
        return self.status()

    def install_local_llm(self, progress=None) -> dict:
        """Install the offline AI weights on an explicit request, reporting bytes.

        This is the interface's own copy of `python -m scripts.install_local_llm`,
        so a user without a key does not need a terminal for the one download the
        application cannot do by itself. The analysis still never fetches these
        bytes, and `progress(received, total)` is called from this worker thread.
        """
        from scripts.install_local_llm import install

        OfflineContext(self.model_cancel).check_cancelled()
        install(self.local_llm_dir, cancel=self.model_cancel, progress=progress)
        return self.status()

    def start(self, query: str, english_query: str | None = None, *, cancel: Event | None = None,
              seed_arxiv_receipt_hash: str | None = None,
              collection_profile: CollectionProfile | None = None,
              english_source: Literal["user", "local_translation"] = "user",
              source_policy: dict | None = None) -> str:
        from app.pilot.approved_sources.catalog import SourcePolicy
        from app.pilot.encoder import verify_artifacts

        query = normalize_query(query)
        if source_policy is not None:
            try:
                source_policy = SourcePolicy.from_json(source_policy).to_json()
            except (TypeError, ValueError):
                raise TaskFailure("Некорректное правило источников.") from None
        if seed_arxiv_receipt_hash is not None and not re.fullmatch(r"[0-9a-f]{64}", seed_arxiv_receipt_hash):
            raise TaskFailure("Некорректный идентификатор экспорта arXiv.")
        requested = cancel if cancel is not None else Event()
        preflight: _Cancellation = (_StartCancellation(requested, self.model_cancel)
                                   if cancel is not None else self.model_cancel)
        with self.coordinator.idle_operation(for_run=True):
            OfflineContext(preflight).check_cancelled()
            settings = load_settings(self.data_dir)
            if collection_profile not in {None, "fast", "deep"}:
                raise TaskFailure("Неизвестный режим сбора данных.")
            if collection_profile is not None:
                settings = apply_collection_profile(settings, collection_profile)
            try:
                verify_artifacts(self.model_dir, cancel=preflight)
            except CancelledError:
                raise TaskCancelled() from None
            OfflineContext(preflight).check_cancelled()
            credential_name = "deepseek_api_key" if settings.provider == "deepseek" else "yandex_api_key"
            # The local model reads a Russian direction itself, so it needs no
            # English formulation and no key to start.
            if settings.provider != "local" and not english_query and not self.credentials.get(credential_name):
                if not re.search(r"[А-Яа-яЁё]", query) and re.search(r"[A-Za-z]", query):
                    english_query = query
                else:
                    raise TaskFailure("Для понимания русского запроса подключите AI в настройках. "
                                      "Ручной режим принимает вашу английскую поисковую формулировку.")
            OfflineContext(preflight).check_cancelled()
            if english_source not in {"user", "local_translation"}:
                raise TaskFailure("Неизвестное происхождение английской формулировки.")
            payload = {"query": query, "english_query": english_query,
                       "english_source": english_source if english_query else "user",
                       "as_of": date.today().isoformat(), "settings": settings.model_dump(mode="json")}
            if collection_profile is not None:
                payload["collection_profile"] = collection_profile
            if seed_arxiv_receipt_hash is not None:
                payload["seed_arxiv_receipt_hash"] = seed_arxiv_receipt_hash
            if source_policy is not None:
                # Правило владельца (страны, выключенные источники, обученные доли)
                # записывается в запуск: результат помнит, что было опрошено.
                payload["source_policy"] = source_policy
            return self.coordinator.submit(payload, cancel=requested)

    def start_signals(self, query_profile_hash: str, concept_hashes: tuple[str, ...], *,
                      wordstat_receipt_hash: str | None = None, arxiv_receipt_hash: str | None = None,
                      capital_receipt_hashes: tuple[str, ...] = (), association_hashes: tuple[str, ...] = (),
                      base_result_run_id: str | None = None,
                      scientific_links: tuple[dict, ...] = (),
                      cancel: Event | None = None) -> str:
        """Start the independent local evidence profile without changing a scientific run."""
        from app.pilot.multisource.contracts import load_policy
        from app.pilot.multisource.workflow import ScientificLink, SignalRunInput

        requested = cancel if cancel is not None else Event()
        with self.coordinator.idle_operation():
            OfflineContext(_StartCancellation(requested, self.model_cancel)).check_cancelled()
            _, policy_hash = load_policy()
            try:
                plan = SignalRunInput(query_profile_hash=query_profile_hash, concept_hashes=concept_hashes,
                                      wordstat_receipt_hash=wordstat_receipt_hash,
                                      arxiv_receipt_hash=arxiv_receipt_hash,
                                      capital_receipt_hashes=capital_receipt_hashes,
                                      association_hashes=association_hashes,
                                      base_result_run_id=base_result_run_id,
                                      scientific_links=tuple(ScientificLink.model_validate(item) for item in scientific_links),
                                      policy_hash=policy_hash,
                                      decision_at=datetime.now(UTC))
            except ValidationError:
                raise TaskFailure("Некорректные данные для профиля сигналов.") from None
            return self.coordinator.submit(plan.model_dump(mode="json"), cancel=requested)

    def create_signal_query(self, query: str, definition: str, phrase: str) -> dict:
        """Store one user-confirmed technology and its immutable search vocabulary."""
        from app.pilot.multisource.contracts import TechnologyConcept
        from app.pilot.multisource.queries import build_manual_profile
        from app.pilot.multisource.store import SignalStore

        with self.coordinator.idle_operation():
            when = datetime.now(UTC)
            profile = build_manual_profile(query, definition, seed_terms=(phrase,), primary_phrase=phrase,
                                           confirmed_at=when)
            store = SignalStore(self.data_dir)
            profile_hash = store.put_object(profile)
            concept = TechnologyConcept(concept_id=uuid4(), label=phrase, definition=profile.definition,
                                        identity_status="confirmed", confirmed_at=when,
                                        provenance_hashes=(profile_hash,))
            concept_hash = store.put_object(concept)
            return {"query_profile_hash": profile_hash, "concept_hash": concept_hash,
                    "concept_id": str(concept.concept_id), "phrase": phrase}

    def preview_signal_csv(self, path: str, encoding: str, delimiter: str) -> dict:
        from typing import cast
        from app.pilot.multisource.imports import CsvDelimiter, CsvEncoding, read_csv_document

        if encoding not in {"utf-8-sig", "cp1251"} or delimiter not in {";", ",", "\t"}:
            raise TaskFailure("Выберите кодировку и разделитель CSV.")
        return read_csv_document(Path(path), encoding=cast(CsvEncoding, encoding),
                                 delimiter=cast(CsvDelimiter, delimiter)).preview()

    def fetch_signal_wordstat(self, query_profile_hash: str, first_month: str, last_month: str) -> dict:
        """One user-requested paid GetDynamics call with a durable local budget fence."""
        from app.pilot.multisource.contracts import WordstatImportReceipt
        from app.pilot.multisource.store import SignalStore
        from app.pilot.multisource.wordstat_api import fetch_wordstat_dynamics

        settings = load_settings(self.data_dir)
        if not settings.wordstat_api_enabled:
            raise TaskFailure("Wordstat API выключен в настройках; можно импортировать CSV без ключа.")
        key = self.credentials.get("wordstat_api_key")
        if key is None:
            raise TaskFailure("Укажите отдельный ключ Wordstat API в настройках.")
        try:
            first = date.fromisoformat(first_month + "-01")
            last = date.fromisoformat(last_month + "-01")
            next_month = date(last.year + (last.month == 12), last.month % 12 + 1, 1)
            last_day = next_month - timedelta(days=1)
        except (TypeError, ValueError, OverflowError):
            raise TaskFailure("Укажите первый и последний месяц в формате ГГГГ-ММ.") from None
        with self.coordinator.idle_operation():
            store = SignalStore(self.data_dir)
            digest = fetch_wordstat_dynamics(store, query_profile_hash, folder_id=settings.wordstat_folder_id,
                                             api_key=key, from_date=first, to_date=last_day,
                                             hour_cap=settings.wordstat_hourly_cap,
                                             daily_cap=settings.wordstat_daily_cap,
                                             coordinator=self.coordinator)
            receipt = store.get_object(digest, WordstatImportReceipt)
            return {"receipt_hash": digest, "source": "wordstat", "accepted": receipt.accepted_count,
                    "rejected": receipt.rejected_count, "snapshot_hash": receipt.snapshot_hash}

    def import_signal_csv(self, query_profile_hash: str, path: str, source: str, mapping: dict,
                          encoding: str, delimiter: str, *, retention_confirmed: bool,
                          share_confirmed: bool = False, license_ref: str | None = None) -> dict:
        from app.pilot.multisource.capital import CordisMapping, InvestmentMapping, import_capital_csv
        from app.pilot.multisource.contracts import CapitalImportReceipt, WordstatImportReceipt
        from app.pilot.multisource.imports import CsvDelimiter, CsvEncoding
        from app.pilot.multisource.store import SignalStore
        from app.pilot.multisource.wordstat import DynamicsMapping, import_wordstat_csv
        from typing import cast

        if not retention_confirmed:
            raise TaskFailure("Подтвердите право хранить выбранную выгрузку локально.")
        export_right, license_ref = _signal_transfer_rights(share_confirmed, license_ref)
        if encoding not in {"utf-8-sig", "cp1251"} or delimiter not in {";", ",", "\t"}:
            raise TaskFailure("Выберите кодировку и разделитель CSV.")
        try:
            values = dict(mapping)
            specification: DynamicsMapping | CordisMapping | InvestmentMapping
            if source == "wordstat":
                for name in ("expected_from", "expected_to"):
                    if values.get(name):
                        values[name] = date.fromisoformat(values[name] + "-01")
                    else:
                        values[name] = None
                if not values.get("share_column"):
                    values["share_column"] = None
                specification = DynamicsMapping(**values)
            elif source == "cordis":
                specification = CordisMapping(**values)
            elif source == "investment_csv":
                specification = InvestmentMapping(**values)
            else:
                raise TaskFailure("Неизвестный источник CSV.")
        except (KeyError, TypeError, ValueError):
            raise TaskFailure("Карта колонок или дат CSV некорректна.") from None
        with self.coordinator.idle_operation():
            store = SignalStore(self.data_dir)
            if isinstance(specification, DynamicsMapping):
                digest = import_wordstat_csv(store, Path(path), query_profile_hash, kind="dynamics",
                                             mapping=specification, encoding=cast(CsvEncoding, encoding),
                                             delimiter=cast(CsvDelimiter, delimiter),
                                             retention="local_allowed", export_right=export_right,
                                             license_ref=license_ref)
                receipt = store.get_object(digest, WordstatImportReceipt)
                return {"receipt_hash": digest, "source": source, "accepted": receipt.accepted_count,
                        "rejected": receipt.rejected_count, "snapshot_hash": receipt.snapshot_hash}
            digest = import_capital_csv(store, Path(path), query_profile_hash, mapping=specification,
                                        encoding=cast(CsvEncoding, encoding), delimiter=cast(CsvDelimiter, delimiter),
                                        retention="local_allowed", export_right=export_right,
                                        license_ref=license_ref)
            receipt_c = store.get_object(digest, CapitalImportReceipt)
            return {"receipt_hash": digest, "source": source, "accepted": len(receipt_c.event_hashes),
                    "rejected": receipt_c.rejected_count, "snapshot_hash": receipt_c.snapshot_hash}

    def import_signal_atom(self, query_profile_hash: str, path: str, *, retention_confirmed: bool,
                           share_confirmed: bool = False, license_ref: str | None = None) -> dict:
        from app.pilot.multisource.arxiv import import_arxiv_discovery
        from app.pilot.multisource.contracts import ArxivImportReceipt
        from app.pilot.multisource.store import SignalStore

        if not retention_confirmed:
            raise TaskFailure("Подтвердите право хранить Atom-выгрузку локально.")
        export_right, license_ref = _signal_transfer_rights(share_confirmed, license_ref)
        with self.coordinator.idle_operation():
            store = SignalStore(self.data_dir)
            digest = import_arxiv_discovery(store, self.archive, Path(path), query_profile_hash,
                                            as_of=date.today(), retention="local_allowed",
                                            export_right=export_right, license_ref=license_ref)
            receipt = store.get_object(digest, ArxivImportReceipt)
            return {"receipt_hash": digest, "source": "arxiv", "accepted": len(receipt.selected_revision_ids),
                    "rejected": receipt.rejected, "snapshot_hash": receipt.snapshot_hash}

    def signal_arxiv_candidates(self, receipt_hash: str) -> list[dict]:
        from app.pilot.multisource.contracts import ArxivImportReceipt
        from app.pilot.multisource.store import SignalStore

        receipt = SignalStore(self.data_dir).get_object(receipt_hash, ArxivImportReceipt)
        return [{"revision_id": digest, "title": self.archive.get(digest).title,
                 "abstract": (self.archive.get(digest).abstract or "")[:1200]}
                for digest in receipt.selected_revision_ids]

    def signal_scientific_runs(self) -> list[dict]:
        """Offer existing completed scientific runs; selection remains explicit."""
        rows = self.list_runs(0, 50, source="local") + self.list_runs(0, 50, source="imported")
        result = []
        for row in rows:
            if row["state"] != "succeeded":
                continue
            try:
                query = json.loads(row["input_json"]).get("query", "")
            except (ValueError, TypeError):
                query = ""
            result.append({"run_id": row["id"], "query": str(query)[:200],
                           "created_at": row["created_at"], "imported": bool(row.get("imported"))})
        return sorted(result, key=lambda row: row["created_at"], reverse=True)

    def signal_scientific_cards(self, run_id: str) -> list[dict]:
        """Show only cards from a fully verified scientific result."""
        from app.pilot.multisource.workflow import _base_result

        result, _ = _base_result(self.result, run_id, self.archive, datetime.now(UTC))
        return [{"candidate_id": card.candidate.candidate_id, "label": card.candidate.label,
                 "category": card.category, "definition": card.candidate.definition[:1200]}
                for card in result.cards]

    def link_signal_arxiv(self, concept_hash: str, receipt_hash: str, revision_id: str) -> str:
        from app.pilot.multisource.contracts import ArxivImportReceipt, TechnologyConcept
        from app.pilot.multisource.store import SignalStore

        with self.coordinator.idle_operation():
            store = SignalStore(self.data_dir)
            concept = store.get_object(concept_hash, TechnologyConcept)
            receipt = store.get_object(receipt_hash, ArxivImportReceipt)
            if (revision_id not in receipt.selected_revision_ids or
                    receipt.query_profile_hash not in concept.provenance_hashes):
                raise TaskFailure("Препринт не относится к выбранной области.")
            self.archive.get(revision_id)
            provenance = tuple(dict.fromkeys((*concept.provenance_hashes, revision_id)))
            linked = TechnologyConcept.model_validate(concept.model_copy(update={
                "confirmed_at": datetime.now(UTC), "provenance_hashes": provenance,
            }).model_dump(mode="json"))
            return store.put_object(linked)

    def signal_associations(self, run_id: str) -> list[dict]:
        from app.pilot.multisource.contracts import CapitalDescription, TechnologyAssociation
        from app.pilot.multisource.store import SignalStore

        profile = self.signal_result(run_id)["profile"]
        store = SignalStore(self.data_dir)
        result = []
        for digest in profile["association_artifact_hashes"]:
            item = store.get_object(digest, TechnologyAssociation)
            description = (store.get_object(item.evidence_hashes[0], CapitalDescription)
                           if item.evidence_hashes else None)
            result.append({"hash": digest, "subject_id": item.subject_id, "subject_kind": item.subject_kind,
                           "relation": item.relation, "status": item.status,
                           "title": description.title if description else "",
                           "description": description.description if description else "",
                           "source_url": description.source_url if description else None})
        return result

    def confirm_signal_grant(self, run_id: str, association_hash: str, reviewer: str) -> str:
        from app.pilot.multisource.contracts import TechnologyAssociation
        from app.pilot.multisource.store import SignalStore

        if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 200:
            raise TaskFailure("Укажите имя проверяющего.")
        with self.coordinator.idle_operation():
            profile = self.signal_result(run_id)["profile"]
            if association_hash not in profile["association_artifact_hashes"]:
                raise TaskFailure("Эта связь отсутствует в выбранном профиле.")
            store = SignalStore(self.data_dir)
            item = store.get_object(association_hash, TechnologyAssociation)
            if (item.status != "proposed" or item.subject_kind != "project" or item.relation != "researches"
                    or not item.evidence_hashes):
                raise TaskFailure("Автоматическое подтверждение этой связи не поддерживается.")
            reviewed = item.model_copy(update={"status": "confirmed", "relation_at_event": "supported",
                                               "reviewer": reviewer.strip(), "reviewed_at": datetime.now(UTC)})
            return store.put_object(TechnologyAssociation.model_validate(reviewed.model_dump(mode="json")))

    def list_signal_runs(self, offset: int = 0, limit: int = 50) -> list[dict]:
        from app.pilot.library import catalogue_paths, read_artifact

        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise TaskFailure("Некорректная страница истории сигналов.")
        needed = offset + limit
        if needed > 10_000:
            raise TaskFailure("История сигналов превышает предел просмотра 10 000 записей.")
        local: list[dict] = []
        while len(local) < needed:
            page_size = min(100, needed - len(local))
            batch = self.coordinator.list_operation_runs("signals", limit=page_size, offset=len(local))
            local.extend(batch)
            if len(batch) < page_size:
                break
        imported_dir = self.data_dir / "signals" / "imported"
        imported = []
        for path in catalogue_paths(imported_dir):
            digest = path.stem
            record = read_artifact(imported_dir, digest)
            if set(record) != {"profile_hash", "base", "science_id"}:
                raise TaskFailure("Каталог импортированных сигналов повреждён.")
            from app.pilot.multisource.contracts import SignalProfile
            from app.pilot.multisource.store import SignalStore
            profile = SignalStore(self.data_dir).get_object(record["profile_hash"], SignalProfile)
            imported.append({"id": "signal-import-" + digest, "created_at": profile.decision_at.isoformat(),
                             "state": "succeeded", "source": "imported"})
        return sorted(local + imported, key=lambda row: row["created_at"], reverse=True)[offset:offset + limit]

    def export_signal(self, run_id: str, path: str) -> dict:
        from app.pilot.multisource.export import export_signal_package, read_imported_signal
        from app.pilot.multisource.store import SignalStore

        value = self.signal_result(run_id)
        read_base: Callable[[str], dict] = self.result
        if run_id.startswith("signal-import-"):
            _, profile, base, _ = read_imported_signal(self.data_dir, run_id)
            read_base = lambda base_id: _stored_signal_base(base, profile.base_result_run_id, base_id)
        result = export_signal_package(Path(path), SignalStore(self.data_dir), self.archive,
                                       value["profile_hash"], read_base)
        return {"path": str(result.path), "sha256": result.sha256, "files": result.files}

    def import_signal(self, path: str) -> dict:
        from app.pilot.multisource.export import import_signal_package

        with self.coordinator.idle_operation():
            return import_signal_package(Path(path), self.data_dir, self._library)

    def signal_result(self, run_id: str) -> dict:
        from app.pilot.multisource.contracts import (ArxivImportReceipt, CapitalImportReceipt,
                                                     TechnologyAssociation, TechnologyConcept,
                                                     WordstatImportReceipt)
        from app.pilot.multisource.export import read_imported_signal
        from app.pilot.multisource.store import SignalStore
        from app.pilot.multisource.workflow import verify_signal_profile

        OfflineContext(self.view_cancel).check_cancelled()
        read_base: Callable[[str], dict]
        if run_id.startswith("signal-import-"):
            imported_profile_hash, imported_profile, base, science_id = read_imported_signal(self.data_dir, run_id)
            value = {"kind": "signals", "profile_hash": imported_profile_hash,
                     "profile": imported_profile.model_dump(mode="json"), "imported": True,
                     "scientific_reuse_run_id": science_id}
            read_base = lambda base_id: _stored_signal_base(base, imported_profile.base_result_run_id, base_id)
        else:
            value = self.coordinator.result(run_id, cancel=self.view_cancel)
            read_base = self.result
        profile_hash = value.get("profile_hash")
        if value.get("kind") != "signals" or not isinstance(profile_hash, str):
            raise TaskFailure("Этот запуск не является профилем сигналов.")
        store = SignalStore(self.data_dir)
        profile = verify_signal_profile(store, self.archive, profile_hash, read_base)
        if value.get("profile") != profile.model_dump(mode="json"):
            raise TaskFailure("Опубликованный профиль отличается от сохранённого результата.")
        imports = {}
        for digest in profile.import_receipt_hashes:
            try:
                store.get_object(digest, WordstatImportReceipt)
                imports["wordstat"] = digest
            except TaskFailure:
                try:
                    store.get_object(digest, ArxivImportReceipt)
                    imports["arxiv"] = digest
                except TaskFailure:
                    receipt_c = store.get_object(digest, CapitalImportReceipt)
                    imports[receipt_c.source] = digest
        concepts = {str(item.concept_id): item.label for item in (
            store.get_object(digest, TechnologyConcept) for digest in profile.concept_artifact_hashes)}
        confirmed = tuple(digest for digest in profile.association_artifact_hashes
                          if store.get_object(digest, TechnologyAssociation).status == "confirmed")
        return value | {"view_id": run_id, "imports": imports, "concepts": concepts,
                        "confirmed_association_hashes": confirmed}

    def signal_finding_evidence(self, run_id: str, finding_id: str) -> dict:
        """Bounded local evidence view after the profile's full transitive verification."""
        from app.pilot.multisource.contracts import (CapitalDescription, CapitalEvent, FundingMetric,
                                                     SearchMetric, SearchObservation, SignalProfile, SourceSnapshot)
        from app.pilot.multisource.store import SignalStore

        view = self.signal_result(run_id)
        profile = SignalProfile.model_validate(view["profile"])
        finding = next((item for item in profile.findings if str(item.finding_id) == finding_id), None)
        if finding is None:
            raise TaskFailure("Карточка отсутствует в выбранном профиле.")
        store = SignalStore(self.data_dir)
        metrics = []
        for digest in finding.metric_hashes:
            item: SearchMetric | FundingMetric
            try:
                item = store.get_object(digest, SearchMetric)
            except TaskFailure:
                item = store.get_object(digest, FundingMetric)
            metrics.append(item.model_dump(mode="json") | {"metric_kind": type(item).__name__})
        observations = []
        for digest in finding.observation_hashes[:100]:
            value = None
            for expected in (SearchObservation, CapitalEvent, CapitalDescription):
                try:
                    value = store.get_object(digest, expected)
                    break
                except TaskFailure:
                    pass
            if value is None:
                document = self.archive.get(digest)
                observations.append({"kind": "arxiv_preprint", "id": digest, "title": document.title,
                                     "source_url": document.url})
            else:
                observations.append(value.model_dump(mode="json") | {"kind": type(value).__name__, "id": digest})
        snapshots = [store.get_object(digest, SourceSnapshot).model_dump(mode="json")
                     for digest in profile.source_snapshot_hashes]
        return {"finding_id": finding_id, "metrics": metrics, "observations": observations,
                "total_observations": len(finding.observation_hashes), "snapshots": snapshots}

    def signal_scenario(self, run_id: str, channel: str, concept_id: str | None = None,
                        event_kind: str | None = None, currency: str | None = None,
                        event_hash: str | None = None) -> dict:
        """Deterministic offline view; the saved result and source objects are unchanged."""
        from app.pilot.multisource.contracts import SignalProfile
        from app.pilot.multisource.profiles import exclude_channel
        from app.pilot.multisource.store import SignalStore
        from app.pilot.multisource.workflow import _base_result

        profile = SignalProfile.model_validate(self.signal_result(run_id)["profile"])
        order = {}
        if profile.base_result_run_id is not None:
            read_base: Callable[[str], dict] = self.result
            if run_id.startswith("signal-import-"):
                from app.pilot.multisource.export import read_imported_signal
                _, _, base, _ = read_imported_signal(self.data_dir, run_id)
                read_base = lambda base_id: _stored_signal_base(base, profile.base_result_run_id, base_id)
            scientific, _ = _base_result(read_base, profile.base_result_run_id, self.archive,
                                         profile.knowledge_cutoff)
            order = {card.candidate.candidate_id: index for index, card in enumerate(scientific.cards)}
        if channel not in {"exclude_wordstat", "exclude_funding", "exclude_science",
                           "exclude_largest_disclosed_event"}:
            raise TaskFailure("Неизвестный сценарий устойчивости.")
        from typing import cast
        from uuid import UUID
        from app.pilot.multisource.profiles import Channel
        try:
            selected_concept = UUID(concept_id) if concept_id is not None else None
        except ValueError:
            raise TaskFailure("Некорректный идентификатор технологии.") from None
        return exclude_channel(SignalStore(self.data_dir), profile, cast(Channel, channel), order,
                               concept_id=selected_concept, event_kind=event_kind,
                               currency=currency, event_hash=event_hash)

    def signal_largest_event_options(self, run_id: str, concept_id: str) -> tuple[dict, ...]:
        """Verified, untruncated tie choices for the selected financial scenario."""
        from uuid import UUID
        from app.pilot.multisource.contracts import SignalProfile
        from app.pilot.multisource.profiles import largest_disclosed_options
        from app.pilot.multisource.store import SignalStore

        profile = SignalProfile.model_validate(self.signal_result(run_id)["profile"])
        try:
            identifier = UUID(concept_id)
        except ValueError:
            raise TaskFailure("Некорректный идентификатор технологии.") from None
        return largest_disclosed_options(SignalStore(self.data_dir), profile, identifier)

    def signal_watch_state(self, run_id: str, concept_id: str) -> dict:
        """Read the latest explicit user choice without changing analysis queues."""
        from uuid import UUID
        from app.pilot.multisource.contracts import SignalProfile
        from app.pilot.multisource.store import SignalStore

        profile = SignalProfile.model_validate(self.signal_result(run_id)["profile"])
        try:
            identifier = UUID(concept_id)
        except ValueError:
            raise TaskFailure("Некорректный идентификатор технологии.") from None
        if identifier not in {item.concept_id for item in profile.findings}:
            raise TaskFailure("Технология отсутствует в выбранном профиле.")
        store = SignalStore(self.data_dir)
        matching = []
        for digest, record in store.watch_records():
            if record.concept_id != identifier:
                continue
            source_profile = store.get_object(record.profile_hash, SignalProfile)
            if (record.recorded_at < source_profile.knowledge_cutoff or
                    identifier not in {item.concept_id for item in source_profile.findings}):
                raise TaskFailure("Запись наблюдения относится к другой технологии или дате.")
            matching.append((digest, record))
        if not matching:
            return {"concept_id": concept_id, "watched": False, "record_hash": None,
                    "recorded_at": None, "note": None}
        digest, latest = matching[-1]
        return {"concept_id": concept_id, "watched": latest.action == "watch", "record_hash": digest,
                "recorded_at": latest.recorded_at.isoformat(), "note": latest.note}

    def set_signal_watch(self, run_id: str, concept_id: str, watch: bool,
                         note: str | None = None) -> dict:
        from uuid import UUID
        from app.pilot.multisource.contracts import SignalProfile, WatchRecord
        from app.pilot.multisource.store import SignalStore, object_digest

        if type(watch) is not bool or note is not None and (not note.strip() or len(note) > 500):
            raise TaskFailure("Некорректная отметка наблюдения или примечание.")
        profile = SignalProfile.model_validate(self.signal_result(run_id)["profile"])
        try:
            identifier = UUID(concept_id)
        except ValueError:
            raise TaskFailure("Некорректный идентификатор технологии.") from None
        if identifier not in {item.concept_id for item in profile.findings}:
            raise TaskFailure("Технология отсутствует в выбранном профиле.")
        record = WatchRecord(profile_hash=object_digest(profile), concept_id=identifier,
                             action="watch" if watch else "unwatch", recorded_at=datetime.now(UTC),
                             note=note.strip() if note is not None else None)
        SignalStore(self.data_dir).put_watch_record(record)
        return self.signal_watch_state(run_id, concept_id)

    def signal_compare(self, first_run_id: str, second_run_id: str) -> dict:
        from app.pilot.multisource.contracts import SignalProfile
        from app.pilot.multisource.profiles import compare_profiles
        from app.pilot.multisource.store import SignalStore

        if first_run_id == second_run_id:
            raise TaskFailure("Выберите два разных запуска сигналов.")
        first = SignalProfile.model_validate(self.signal_result(first_run_id)["profile"])
        second = SignalProfile.model_validate(self.signal_result(second_run_id)["profile"])
        if second.decision_at < first.decision_at:
            first, second = second, first
        return compare_profiles(SignalStore(self.data_dir), first, second)

    def get(self, run_id: str) -> dict:
        row = self.coordinator.get(run_id)
        row["clarification"] = (None if row.get("persistence_error") else
                                self.coordinator.checkpoint_value(run_id, "clarification", cancel=self.view_cancel))
        return row

    def list_runs(self, offset: int = 0, limit: int = 50, source: str = "all") -> list[dict]:
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50 or source not in {"all", "local", "imported"}:
            raise ValueError("Некорректная страница истории.")
        library = self._library
        if source == "imported":
            return library.list_rows(offset=offset, limit=limit)
        analyses = self.coordinator.list_runs(limit=limit, offset=offset, analyses_only=True)
        if source == "local":
            return analyses
        # Compatibility summary; paged UI requests each source explicitly so
        # older imported dates cannot displace local pages or disappear.
        return sorted(analyses + library.list_rows(offset=offset, limit=limit),
                      key=lambda row: row["created_at"], reverse=True)

    def cancel(self, run_id: str) -> bool:
        return self.coordinator.cancel(run_id)

    def resume(self, run_id: str, *, cancel: Event | None = None) -> str:
        return self.coordinator.resume(run_id, cancel=cancel)

    def result(self, run_id: str) -> dict:
        OfflineContext(self.view_cancel).check_cancelled()
        if run_id.startswith("import-"):
            return self._library.read(run_id, cancel=self.view_cancel) | {"view_id": run_id}
        value = self.coordinator.result(run_id, cancel=self.view_cancel)
        if "result" not in value:
            raise TaskFailure("Этот запуск не является научным анализом.")
        AnalysisResult.model_validate(value["result"])
        return value | {"view_id": run_id}

    def import_result(self, path: str) -> dict:
        OfflineContext(self.view_cancel).check_cancelled()
        result = self._library.import_file(Path(path), cancel=self.view_cancel)
        result["payload"]["view_id"] = result["id"]
        return result

    def export_result(self, run_id: str, path: str) -> dict:
        from app.pilot.export import export_result
        from app.pilot.methodology import AssessmentArtifact
        from app.pilot.review import ReviewRecord

        OfflineContext(self.view_cancel).check_cancelled()
        payload = self.result(run_id)
        export_result(Path(path), AnalysisResult.model_validate(payload["result"]), self.archive,
                       tuple(AssessmentArtifact.model_validate(item) for item in payload.get("assessments", [])),
                       reviews=tuple(ReviewRecord.model_validate(item) for item in payload.get("reviews", [])),
                       cancel=self.view_cancel)
        return {"path": path}

    def documents(self, run_id: str, offset: int = 0, limit: int = 50) -> dict:
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Invalid pagination")
        if run_id.startswith("import-"):
            return self._library.documents(run_id, offset, limit, cancel=self.view_cancel)
        OfflineContext(self.view_cancel).check_cancelled()
        saved = self.coordinator.checkpoint_value(run_id, "discovery", cancel=self.view_cancel)
        if saved is None:
            return {"items": [], "total": 0}
        snapshot = CorpusSnapshot.model_validate(saved)
        items = []
        for reference in snapshot.documents[offset:offset + limit]:
            OfflineContext(self.view_cancel).check_cancelled()
            items.append(self.archive.get(reference.revision_id).model_dump(mode="json"))
        OfflineContext(self.view_cancel).check_cancelled()
        return {"total": len(snapshot.documents), "items": items}

    def close(self) -> None:
        self.model_cancel.set()
        self.view_cancel.set()
        self.coordinator.close(timeout=45)
        self._library.clear_cache()
        self.unload_llm()

    def supplemental_list(self, offset: int = 0, limit: int = 50):
        from app.pilot.supplemental import SupplementalLibrary

        return SupplementalLibrary(self.data_dir, self.archive).list_page(offset, limit, cancel=self.view_cancel)

    def budget_status(self, scope_after="", request_after=""):
        from app.runtime.budget_admin import BudgetAdmin

        return BudgetAdmin(self.coordinator).status(scope_after=scope_after, request_after=request_after)

    def budget_reconcile(self, request_id, values):
        from app.runtime.budget_admin import BudgetAdmin

        return BudgetAdmin(self.coordinator).reconcile_unknown(request_id, **values)

    def budget_acknowledge(self, scope_id, cost_micro, confirmed):
        from app.runtime.budget_admin import BudgetAdmin

        if type(cost_micro) is not int or cost_micro < 0 or cost_micro > 10_000_000_000:
            raise ValueError("Некорректный новый запас бюджета.")
        # A day scope names its own provider, so restoring headroom keeps the
        # request fence that provider is actually billed by. Every other scope,
        # and an acknowledgement without one, keeps the conservative paid fence.
        parts = scope_id.split("/") if isinstance(scope_id, str) else ()
        provider = parts[1] if len(parts) == 3 and parts[0] == "day" else ""
        allowance = _day_budget_limits(provider, cost_micro) if cost_micro else None
        return BudgetAdmin(self.coordinator).acknowledge_restore(scope_id, additional_allowance=allowance, confirmed=confirmed)

    def translate_query(self, text: str) -> dict:
        """Produce the English search formulation for a Russian direction locally.

        The model is a general translator, not a terminology authority. Its
        result starts a run without review, so `reviewed` stays false and the
        plan records the formulation as a machine translation. The loaded
        sessions are reused because verifying and loading 117 MB on every
        keystroke would be the slowest part of a sub-second translation.
        """
        from app.pilot.translator import RussianEnglishTranslator, TranslationError, model_directory

        if not isinstance(text, str) or not text.strip():
            raise TaskFailure("Введите направление исследования.")
        translation_dir = model_directory(self.data_dir)
        try:
            with self._translator_lock:
                if self._translator is None:
                    # The profile that owns this service, not the default one.
                    self._translator = RussianEnglishTranslator(translation_dir, cancel=self.view_cancel)
            translator = self._translator
            english = translator.translate(text, cancel=self.view_cancel)  # type: ignore[attr-defined]
        except TranslationError as error:
            raise TaskFailure(str(error)) from None
        if not english:
            raise TaskFailure("Локальная модель не смогла перевести этот запрос. Впишите формулировку сами.")
        return {"english_query": english, "model_id": translator.spec["model_id"],  # type: ignore[attr-defined]
                "revision": translator.spec["revision"],  # type: ignore[attr-defined]
                "reviewed": False}

    def delete_credential(self, name, persistent=False):
        with self.coordinator.idle_operation():
            self.credentials.delete(name, persistent=persistent)
            return self.status()

    def import_report(self, path, metadata):
        from app.pilot.supplemental import SupplementalLibrary

        OfflineContext(self.view_cancel).check_cancelled()
        return SupplementalLibrary(self.data_dir, self.archive).import_report(path, metadata,
            credentials=self.credentials, cancel=self.view_cancel)

    def import_arxiv(self, path):
        from app.pilot.supplemental import SupplementalLibrary

        OfflineContext(self.view_cancel).check_cancelled()
        return SupplementalLibrary(self.data_dir, self.archive).import_arxiv(path, cancel=self.view_cancel)

    def supplemental_matches(self, run_id, candidate_id):
        from app.pilot.supplemental import SupplementalLibrary

        result = AnalysisResult.model_validate(self.result(run_id)["result"])
        card = next((item for item in result.cards if item.candidate.candidate_id == candidate_id), None)
        if card is None:
            raise TaskFailure("Технология отсутствует в этом результате.")
        OfflineContext(self.view_cancel).check_cancelled()
        return SupplementalLibrary(self.data_dir, self.archive).match(card.candidate, OfflineContext(self.view_cancel))

    def sensitivity(self, run_id: str, candidate_id: str, scenario: dict) -> dict:
        from app.pilot.antecedents import AntecedentBundle
        from app.pilot.methodology import AssessmentArtifact
        from app.pilot.sensitivity import SensitivityScenario, evaluate_sensitivity, save_sensitivity

        payload = self.result(run_id)
        result = AnalysisResult.model_validate(payload["result"])
        card = next((item for item in result.cards if item.candidate.candidate_id == candidate_id), None)
        if card is None or card.historical_snapshot_id is None:
            raise TaskFailure("Сначала нужна историческая проверка выбранной технологии.")
        snapshot = next(item for item in result.snapshots if item.snapshot_id == card.historical_snapshot_id)
        assessment = next((AssessmentArtifact.model_validate(item) for item in payload.get("assessments", [])
                           if item["assessment"]["candidate_id"] == candidate_id), None)
        if assessment is None:
            raise TaskFailure("Сохранённый расчёт истории отсутствует.")
        OfflineContext(self.view_cancel).check_cancelled()
        context = OfflineContext(self.view_cancel)
        review = next((item for item in payload.get("reviews", [])
                       if item["decision"]["candidate_id"] == candidate_id), None)
        report = evaluate_sensitivity(card.candidate, result.query_plan, snapshot, self.archive, context,
            passport=card, scenario=SensitivityScenario.model_validate(scenario), verified_novelty=assessment.inputs.novelty,
            methodology_version=assessment.assessment.methodology_version,
            antecedents=AntecedentBundle.model_validate(review["bundle"]) if review else assessment.inputs.antecedents,
            field_exposure=assessment.inputs.field_exposure, source_novelty=assessment.inputs.source_novelty, primary_observations=assessment.inputs.primary_observations,
            publication_status_revisions=assessment.inputs.publication_status_revisions)
        save_sensitivity(report, self.data_dir / "sensitivity", context=context)
        return report.model_dump(mode="json")

    def begin_review(self, run_id, candidate_id):
        result = AnalysisResult.model_validate(self.result(run_id)["result"])
        card = next((item for item in result.cards if item.candidate.candidate_id == candidate_id), None)
        if card is None or card.historical_snapshot_id is None:
            raise TaskFailure("Для оценки новизны сначала нужна историческая проверка технологии.")
        row = self.coordinator.find_antecedents_run(run_id, candidate_id)
        if row is not None:
            if row["state"] in {"cancelled", "interrupted", "failed"}:
                return self.coordinator.resume(row["id"])
            return row["id"]
        return self.coordinator.submit({"operation": "antecedents", "source_run_id": run_id,
                                       "candidate_id": candidate_id, "query": result.query_plan.original_query})

    def refine_candidate(self, run_id: str, candidate_id: str, *, cancel: Event | None = None) -> str:
        """Re-examine one stored hypothesis without overwriting the original result."""
        with self.coordinator.idle_operation():
            requested = cancel if cancel is not None else Event()
            OfflineContext(_StartCancellation(requested, self.model_cancel)).check_cancelled()
            original = AnalysisResult.model_validate(self.result(run_id)["result"])
            pool = {item.candidate.candidate_id: item.candidate for item in original.cards}
            pool.update((item.candidate_id, item) for item in original.candidate_queue)
            if candidate_id not in pool:
                raise TaskFailure("Выбранная гипотеза отсутствует в сохранённом результате.")
            if len(original.cards) >= 60 and candidate_id not in {item.candidate.candidate_id for item in original.cards}:
                raise TaskFailure("В результате уже 60 паспортов. Запустите отдельный анализ выбранного механизма.")
            settings = load_settings(self.data_dir)
            return self.coordinator.submit({"operation": "refine_candidate", "source_run_id": run_id,
                "candidate_id": candidate_id, "query": original.query_plan.original_query,
                "english_query": original.query_plan.english_query, "as_of": original.query_plan.as_of.isoformat(),
                "settings": settings.model_dump(mode="json")}, cancel=requested)

    def _seed_refinement(self, context, payload):
        if payload.get("operation") != "refine_candidate":
            return None
        original = self.result(payload["source_run_id"])
        result = AnalysisResult.model_validate(original["result"])
        candidates = {item.candidate.candidate_id: item.candidate for item in result.cards}
        candidates.update((item.candidate_id, item) for item in result.candidate_queue)
        candidate = candidates.get(payload["candidate_id"])
        if candidate is None:
            raise TaskFailure("Исходная гипотеза не найдена.")
        snapshot = next(item for item in result.snapshots if item.snapshot_id == candidate.discovery_snapshot_id)
        seed = {"plan": result.query_plan.model_dump(mode="json"), "discovery": snapshot.model_dump(mode="json"),
                "candidates": {"candidates": [candidate.model_dump(mode="json")], "review_queue": []}}
        for name, value in seed.items():
            saved = context.load_checkpoint(name)
            if saved is not None and saved != value:
                raise TaskFailure("Сохранённое продолжение относится к другой исходной гипотезе.")
            if saved is None:
                context.checkpoint(name, value)
        return original

    def review_progress(self, job_id):
        row = self.coordinator.get(job_id)
        if row["state"] == "succeeded":
            row["review_data"] = self.coordinator.result(job_id, cancel=self.view_cancel)
        return row

    def _run_antecedents(self, context, payload):
        from app.pilot.antecedents import collect_antecedents

        result = AnalysisResult.model_validate(self.result(payload["source_run_id"])["result"])
        card = next(item for item in result.cards if item.candidate.candidate_id == payload["candidate_id"])
        bundle = collect_antecedents(card.candidate, result.query_plan, self.archive, self.credentials, context)
        source_context = {}
        for evidence in card.evidence:
            context.check_cancelled()
            if evidence.revision_id not in source_context:
                document = self.archive.get(evidence.revision_id)
                source_context[evidence.revision_id] = {"title": document.title, "abstract": document.abstract or ""}
        return {"kind": "antecedents", "source_run_id": payload["source_run_id"], "card": card.model_dump(mode="json"),
                "bundle": bundle.model_dump(mode="json"), "bundle_hash": bundle.bundle_hash,
                "search_complete": bundle.search_complete, "source_context": source_context}

    def apply_review(self, job_id, values):
        from app.pilot.antecedents import AntecedentBundle
        from app.pilot.history import assess_snapshot
        from app.pilot.methodology import AssessmentArtifact
        from app.pilot.review import ReviewDecision, ReviewRecord, apply_novelty_review, record_review

        prepared = self.coordinator.result(job_id)
        if prepared.get("kind") != "antecedents":
            raise TaskFailure("Этот запуск не содержит обзор ранних аналогов.")
        original = self.result(prepared["source_run_id"])
        result = AnalysisResult.model_validate(original["result"])
        bundle = AntecedentBundle.model_validate(prepared["bundle"])
        source_card = next(item for item in result.cards if item.candidate.candidate_id == bundle.candidate_id)
        decision = ReviewDecision.model_validate(dict(values, reviewed_at=datetime.now(UTC),
            candidate_id=source_card.candidate.candidate_id, admission_rule_hash=source_card.candidate.admission_rule_hash,
            bundle_hash=bundle.bundle_hash))
        context = OfflineContext(self.view_cancel)
        context.check_cancelled()
        from app.pilot.contracts import MethodologyVersion

        previous_artifact = next((AssessmentArtifact.model_validate(item) for item in original.get("assessments", [])
            if item["assessment"]["candidate_id"] == source_card.candidate.candidate_id), None)
        if previous_artifact is None:
            raise TaskFailure("Сохранённый расчёт истории отсутствует; сначала повторите проверку кандидата.")
        # Refinement may retain older passports inside a newer container. Review
        # their frozen extraction rules; changing the container is not a reanalysis.
        source_version = previous_artifact.assessment.methodology_version
        if source_version != (source_card.methodology_version or "3.0.0"):
            raise TaskFailure("Версия карточки не соответствует сохранённому расчёту истории.")
        version: MethodologyVersion = "3.1.0" if source_version == "3.0.0" else source_version
        container_version = max((result.methodology_version, version),
                                key=lambda value: tuple(int(part) for part in value.split(".")))
        passport, novelty = apply_novelty_review(decision, bundle, source_card, self.archive, context=context,
                                               methodology_version=version)
        historical = next(item for item in result.snapshots if item.snapshot_id == source_card.historical_snapshot_id)
        card_candidate = source_card.candidate
        artifact, card, _ = assess_snapshot(card_candidate, result.query_plan, historical,
            self.archive, context, passport=passport, verified_novelty=novelty, antecedents=bundle,
            methodology_version=version, field_exposure=previous_artifact.inputs.field_exposure,
            source_novelty=previous_artifact.inputs.source_novelty, primary_observations=previous_artifact.inputs.primary_observations,
            publication_status_revisions=previous_artifact.inputs.publication_status_revisions)
        record = record_review(self.data_dir / "reviews", decision, bundle, source_card=source_card,
            reviewed_card=card, artifact=artifact, historical_snapshot=historical, archive=self.archive, context=context)
        cards = [card if item.candidate.candidate_id == card_candidate.candidate_id else item for item in result.cards]
        artifacts = tuple(AssessmentArtifact.model_validate(item) for item in original.get("assessments", [])
                          if item["assessment"]["candidate_id"] != card_candidate.candidate_id) + (artifact,)
        _rank_cards(cards, artifacts)
        snapshots = tuple({item.snapshot_id: item for item in (*result.snapshots, bundle.snapshot)}.values())
        from app.pilot.selection import select_top

        top_ids = (select_top(cards, artifacts, limit=result.top_limit or 15)
                   if container_version in {"3.2.0", "3.3.0", "3.4.0"} else _top_trend_ids(cards))
        if container_version in {"3.2.0", "3.3.0", "3.4.0"}:
            snapshots = _retained_snapshots(snapshots, cards, result.candidate_queue, artifacts,
                patent_ids=tuple(item.snapshot_id for item in snapshots if item.purpose == "enrichment"))
        revised = AnalysisResult.model_validate(result.model_dump(mode="python") | dict(result_id=uuid4().hex,
            methodology_version=container_version, top_trend_ids=top_ids,
            created_at=datetime.now(UTC), cards=tuple(cards), snapshots=snapshots,
            quality="complete" if all(item.quality == "complete" for item in cards) else "partial",
            limitations=tuple(dict.fromkeys((*result.limitations, "Новизна отмеченных технологий оценена указанным экспертом; личность заявлена локально.")))))
        reviews = tuple(ReviewRecord.model_validate(item) for item in original.get("reviews", [])
                        if item["decision"]["candidate_id"] != card_candidate.candidate_id) + (record,)
        return self._library.save_result(revised, artifacts, reviews, cancel=self.view_cancel)

    def _cache_read(self, key: str, name: str) -> dict | None:
        from app.pilot.reports import open_local_regular

        path = self.data_dir / "cache" / "analyses" / key / (name + ".json")
        try:
            with open_local_regular(path) as handle:
                data = handle.read(25_000_001)
            if len(data) > 25_000_000:
                return None
            value = json.loads(data)
            return value if isinstance(value, dict) else None
        except (OSError, ValueError):
            return None

    def _cache_directory(self, *parts: str) -> Path:
        """Create only plain, app-owned cache directories below this profile."""
        directory = self.data_dir
        if directory.is_symlink():
            raise OSError("Cache root is a symbolic link")
        for part in ("cache", *parts):
            if not re.fullmatch(r"[a-z0-9_-]{1,80}", part):
                raise ValueError("Invalid cache path component")
            directory = directory / part
            if directory.is_symlink():
                raise OSError("Cache directory is a symbolic link")
            directory.mkdir(exist_ok=True)
            if directory.is_symlink() or not directory.is_dir():
                raise OSError("Unsafe cache directory")
        return directory

    def _cache_write(self, key: str, name: str, value: dict) -> None:
        if not re.fullmatch(r"[a-f0-9]{64}", key) or not re.fullmatch(r"[a-z0-9_-]{1,40}", name):
            raise ValueError("Invalid cache key")
        path = self._cache_directory("analyses", key) / (name + ".json")
        data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        temporary = path.with_suffix("." + uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _shared_cache_read(self, kind: str, key: str, *, now: datetime | None = None) -> dict | None:
        """Read a short-lived exact-input source artifact with an integrity fence."""
        from app.pilot.reports import open_local_regular

        path = self.data_dir / "cache" / "source-artifacts" / kind / (key + ".json")
        try:
            with open_local_regular(path) as handle:
                data = handle.read(25_000_001)
            if len(data) > 25_000_000:
                return None
            value = json.loads(data)
            if not isinstance(value, dict) or set(value) != {
                    "cache_version", "kind", "key", "created_at", "expires_at", "payload_hash", "payload"}:
                return None
            payload = value["payload"]
            if (value["cache_version"] != SHARED_SOURCE_CACHE_VERSION or value["kind"] != kind
                    or value["key"] != key or not isinstance(payload, dict)
                    or value["payload_hash"] != content_hash(payload)):
                return None
            created_at = datetime.fromisoformat(value["created_at"])
            expires_at = datetime.fromisoformat(value["expires_at"])
            if created_at.tzinfo is None or expires_at.tzinfo is None:
                return None
            current = (now or datetime.now(UTC)).astimezone(UTC)
            created_at = created_at.astimezone(UTC)
            expires_at = expires_at.astimezone(UTC)
            if (expires_at - created_at != SHARED_SOURCE_CACHE_TTL
                    or created_at > current or current >= expires_at):
                return None
            return payload
        except (OSError, TypeError, ValueError):
            return None

    def _shared_cache_write(self, kind: str, key: str, payload: dict, *,
                            now: datetime | None = None) -> None:
        """Atomically store only disposable source data; a failed cache never fails a run."""
        if not re.fullmatch(r"[a-f0-9]{64}", key):
            raise ValueError("Invalid cache key")
        created_at = (now or datetime.now(UTC)).astimezone(UTC)
        value = {
            "cache_version": SHARED_SOURCE_CACHE_VERSION,
            "kind": kind,
            "key": key,
            "created_at": created_at.isoformat(),
            "expires_at": (created_at + SHARED_SOURCE_CACHE_TTL).isoformat(),
            "payload_hash": content_hash(payload),
            "payload": payload,
        }
        temporary = None
        try:
            path = self._cache_directory("source-artifacts", kind) / (key + ".json")
            temporary = path.with_suffix("." + uuid4().hex + ".tmp")
            data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        except OSError:
            pass
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def shared_llm(self) -> _SharedLocalModel:
        """Общая загруженная модель; загружается при первом анализе."""
        from app.pilot.local_llm import LocalInstructModel, LocalModelError

        with self._shared_llm_lock:
            if self._shared_llm is None:
                try:
                    self._shared_llm = _SharedLocalModel(LocalInstructModel(self.local_llm_dir))
                except LocalModelError as error:
                    # Тот же отказ, что даёт клиент, загружающий модель сам.
                    raise LlmError("model_unavailable", str(error)) from None
            return self._shared_llm

    def llm_loaded(self) -> bool:
        with self._shared_llm_lock:
            return self._shared_llm is not None

    def unload_llm(self) -> bool:
        """Выгрузить общую модель из памяти (идущий расчёт дожидается своей очереди)."""
        with self._shared_llm_lock:
            shared, self._shared_llm = self._shared_llm, None
        if shared is None:
            return False
        shared.unload()
        return True

    def take_radar(self, run_id: str) -> tuple[Future, list[int], Event] | None:
        """Досчёт ТОПа технологий, переданный завершившимся анализом: (задача, ход, отмена)."""
        with self._handed_lock:
            return self._handed_radars.pop(run_id, None)

    def _hand_off_radar(self, run_id: str, future: Future, progress: list[int], cancel: Event) -> None:
        with self._handed_lock:
            self._handed_radars[run_id] = (future, progress, cancel)
            # Незабранные (сайт перезапущен, страница закрыта) не копятся.
            while len(self._handed_radars) > 8:
                self._handed_radars.pop(next(iter(self._handed_radars)))[2].set()

    def _technology_radar(self, query: str, plan: QueryPlan, discovery: dict,
                          approved_sources: Callable[[], dict | None], *, cancel: Event,
                          progress: Callable[[int, int], None], max_candidates: int | None = None) -> dict:
        """ТОП-15 технологий по той же выборке, что увидит веб: наука и одобренные источники."""
        from app.radar.pipeline import MAX_CANDIDATES, SHARED_CACHE, build_radar
        from app.web_api import radar_input

        pool, as_of, terms = radar_input(
            {"result": {"snapshots": [discovery], "query_plan": plan.model_dump(mode="json")},
             "approved_sources": approved_sources()}, archive=self.archive)
        if as_of is None or not pool:
            return {"state": "unavailable", "message": "Для ТОПа технологий нет найденных материалов."}
        # Радар стартует раньше смысловой оценки, поэтому тема проверяется по
        # словам плана и, если программа уже обучилась, её моделью. Иначе
        # кандидаты в технологии вырастали из заголовков чужих новостей.
        from app.relevance_learning import RelevanceModel
        from app.topic_relevance import TopicProfile, filter_pool

        profile = TopicProfile.from_plan(plan)
        pool = filter_pool(pool, profile, scorer=RelevanceModel.load(self.data_dir).scorer(profile, as_of))
        try:
            openalex_key = self.credentials.get("openalex_api_key")
        except Exception:
            openalex_key = None
        radar = build_radar(pool, query=query, query_terms=terms or [query], as_of=as_of,
                            cancel=cancel, progress=progress, openalex_key=openalex_key, cache=SHARED_CACHE,
                            max_candidates=max_candidates or MAX_CANDIDATES)
        # Русские выжимки делаются здесь же, пока анализ ещё идёт: переводчик
        # работает на процессоре, а основной поток в это время ждёт GPU и сеть.
        # Без переводчика (настольное приложение) ТОП остаётся в оригинале.
        translate = self.radar_translator
        if translate is not None:
            try:
                radar["translation"] = translate(radar, cancel)
            except Exception:
                pass
        return {"state": "ready", "result": radar}

    def _run(self, context: RunContext, payload: dict) -> dict:
        if os.environ.get("TRENDANALIZER_INFERENCE_PROVIDER", "").strip().lower() == "cuda":
            from app.runtime.inference import RequestedCudaUnavailable, execution_providers

            try:
                execution_providers()
            except RequestedCudaUnavailable as error:
                raise TaskFailure(str(error)) from None
        if payload.get("operation") == "signals":
            from app.pilot.multisource.workflow import run_signal_profile

            return run_signal_profile(context, payload, self.data_dir, self.archive, self.result)
        if payload.get("operation") == "antecedents":
            return self._run_antecedents(context, payload)
        client: LlmClient | None = None
        approved_pool: ThreadPoolExecutor | None = None
        approved_future: Future | None = None
        funding_future: Future | None = None
        save_approved_on_exit: Callable[[], None] | None = None
        save_funding_on_exit: Callable[[], None] | None = None
        radar_pool: ThreadPoolExecutor | None = None
        radar_future: Future | None = None
        radar_handed = False
        exposure_pool: ThreadPoolExecutor | None = None
        exposure_future: Future | None = None
        exposure_context: _BackgroundContext | None = None
        radar_cancel = Event()
        try:
            refined_from = self._seed_refinement(context, payload)
            settings = PilotSettings.model_validate(payload["settings"])
            collection_profile = payload.get("collection_profile")
            if collection_profile not in {None, "fast", "deep"}:
                raise TaskFailure("Неизвестный режим сбора данных.")
            as_of = date.fromisoformat(payload["as_of"])
            limits = QueryLimits(discovery_documents=settings.discovery_documents)
            if collection_profile is not None:
                per_candidate, new_documents = PROFILE_HISTORY_LIMITS[collection_profile]
                limits = QueryLimits(discovery_documents=settings.discovery_documents,
                                     historical_documents_per_candidate=per_candidate,
                                     new_historical_documents=new_documents)
            if refined_from is not None:
                limits = QueryPlan.model_validate(refined_from["result"]["query_plan"]).limits
            budget = BudgetService(context.connection)
            run_scope = "run/" + context.run_id
            run_exists = context.connection.execute("SELECT currency FROM pilot_budget_scopes WHERE scope_id=?", (run_scope,)).fetchone()
            if run_exists is None:
                budget.create_scope(run_scope, _run_budget_limits(settings.provider, limits,
                                    settings.run_cost_micro), currency=settings.currency)
            elif run_exists[0] != settings.currency:
                raise TaskFailure("Валюта сохранённого бюджета анализа отличается от настроек.")
            _day_budget_scope(budget, settings)
            scopes = (run_scope,)

            def scopes_for_dispatch(request_scopes):
                # The sole coordinator writer resolves the calendar at each
                # paid admission. Run and day allowances are then reserved in
                # the same BudgetService transaction; previous dates stay intact.
                context.check_cancelled()
                return (*request_scopes, _day_budget_scope(budget, settings))

            key_name = "deepseek_api_key" if settings.provider == "deepseek" else "yandex_api_key"
            local_model_notice = ""
            if settings.provider == "local":
                from app.pilot.local_client import LocalLlmClient, LocalProviderConfig

                try:
                    # The local client implements the same completion contract
                    # used by the planner and evidence routines.
                    client = cast(LlmClient, LocalLlmClient(
                        LocalProviderConfig(currency=settings.currency), budget,
                        model=cast(Any, self.shared_llm()) if self.keep_llm_loaded else None,
                        model_dir=self.local_llm_dir, scope_resolver=scopes_for_dispatch))
                except LlmError as error:
                    # Weights that are not installed must not cost the user the
                    # whole run: it still collects documents and groups them, and
                    # the result says plainly why no candidate was confirmed.
                    if (os.environ.get("TRENDANALIZER_INFERENCE_PROVIDER", "").strip().lower() == "cuda"
                            and error.code == "model_unavailable"):
                        raise TaskFailure(str(error)) from None
                    if error.code != "model_unavailable":
                        raise
                    local_model_notice = ("Локальная AI-модель не установлена, поэтому названия и границы "
                                          "технологий не проверялись: ни один кандидат не может попасть в TOP. "
                                          "Установите её командой python -m scripts.install_local_llm.")
            else:
                try:
                    has_key = bool(self.credentials.get(key_name))
                except CredentialUnavailable:
                    if not payload.get("english_query"):
                        raise
                    has_key = False
                if has_key:
                    client = LlmClient(settings.provider_config(), self.credentials, budget,
                                       scope_resolver=scopes_for_dispatch)
            cache_key = content_hash({"workflow": WORKFLOW_VERSION, "request": payload})
            context.progress("plan", "Определяем границы технологического направления")
            saved = context.load_checkpoint("plan")
            cached = QueryPlan.model_validate(saved) if saved else None
            if cached is None:
                reusable = self._cache_read(cache_key, "plan")
                try:
                    cached = QueryPlan.model_validate(reusable) if reusable else None
                except ValidationError:
                    cached = None  # Disposable cache; durable checkpoints remain strict.
                if (cached is not None and cached.planner_version.startswith("query-planner/")
                        and cached.planner_version != QUERY_PROMPT_VERSION):
                    # Search expansion rules changed. A new run must not reuse
                    # an old broad scope; durable checkpoints still resume as-is.
                    cached = None
                if (cached is not None and not payload.get("english_query")
                        and cached.planner_version in {EXPLICIT_QUERY_VERSION, LOCAL_TRANSLATION_QUERY_VERSION}):
                    # A literal translation is what a failed model answer left
                    # behind, not a plan worth replaying: ask the model again.
                    cached = None
                if cached is not None and scope_is_unrelated(payload["query"], cached.english_query,
                        cached.subdirections, tuple(item.text for item in cached.queries)):
                    # A scope about another field reached the cache once and was
                    # then replayed for every repeat of the request, so the run
                    # never asked the model again. Disposable: drop and re-plan.
                    cached = None
            plan_messages = ("Читаем запрос и границы периода", "Ожидаем ответ модели о границах направления",
                             "Проверяем предложенную область", "Область направления определена")
            english_query = payload.get("english_query")
            if client is None and not english_query:
                # Started under the local provider, whose weights turned out to be
                # missing: the search still needs an English formulation, and it
                # comes from the same two sources a run without any AI uses.
                english_query = (payload["query"] if not re.search(r"[А-Яа-яЁё]", payload["query"])
                                 else self.translate_query(payload["query"])["english_query"])
            try:
                plan = plan_query(payload["query"], client, as_of=as_of,
                                  request_id=f"{context.run_id}/plan/{uuid4().hex}", scope_ids=scopes,
                                  cancel=context.cancel_event, english_query=english_query,
                                  cached_plan=cached, limits=limits,
                                  english_source=payload.get("english_source", "user")
                                  if payload.get("english_query") else
                                  ("user" if english_query == payload["query"] else "local_translation"),
                                  progress=lambda completed, total: context.progress(
                                      "plan", plan_messages[completed], completed, total))
            except QueryClarificationRequired as error:
                context.checkpoint("clarification", {"options": list(error.options)})
                raise TaskFailure("Направление неоднозначно. Выберите предложенную трактовку и запустите новый анализ.") from None
            except LlmError as error:
                # The small local model sometimes cannot hold the whole scope
                # format. Losing the run over that would be worse than searching
                # by the translated direction alone, which is exactly what this
                # application does when no AI is configured at all. The plan
                # records that narrower origin; nothing else is assumed.
                if settings.provider != "local" or error.code not in {
                        "schema_failed", "invalid_json", "empty_response", "incomplete_response",
                        "unrelated_scope"}:
                    raise
                # An English request is already a search formulation; only a
                # Russian one needs the bundled translator. Sending English
                # through a Russian-to-English model would corrupt the subject
                # this fallback exists to preserve.
                russian = bool(re.search(r"[А-Яа-яЁё]", payload["query"]))
                translated = self.translate_query(payload["query"])["english_query"] if russian else payload["query"]
                plan = None
                if russian:
                    # The 1.5B model often answers a Russian request in Russian
                    # only («твердотельные аккумуляторы» everywhere). Asked about
                    # the translation it usually names the field and its methods,
                    # which a single literal phrase never does.
                    context.progress("plan", "Уточняем область по переводу запроса", 2, 3)
                    try:
                        plan = plan_query(payload["query"], client, as_of=as_of,
                                          request_id=f"{context.run_id}/plan/{uuid4().hex}", scope_ids=scopes,
                                          cancel=context.cancel_event, cached_plan=None, limits=limits,
                                          model_query=translated)
                    except LlmCancelled:
                        raise
                    except (LlmError, QueryError, BudgetError):
                        plan = None
                if plan is None:
                    context.progress("plan", "Локальная модель не описала область; ищем по формулировке запроса", 2, 3)
                    plan = plan_query(payload["query"], None, as_of=as_of,
                                      request_id=f"{context.run_id}/plan/{uuid4().hex}", scope_ids=scopes,
                                      cancel=context.cancel_event, english_query=translated,
                                      cached_plan=None, limits=limits,
                                      english_source="local_translation" if russian else "user")
            finally:
                if client and client.last_receipt:
                    context.checkpoint("plan_receipt", client.last_receipt.to_dict())
            assert plan is not None
            local_plan_fallback = (settings.provider == "local" and client is not None
                and not payload.get("english_query")
                and plan.planner_version in {EXPLICIT_QUERY_VERSION, LOCAL_TRANSLATION_QUERY_VERSION})
            context.checkpoint("plan", plan.model_dump(mode="json"))
            self._cache_write(cache_key, "plan", plan.model_dump(mode="json"))
            from app.pilot.approved_sources import (SourceSnapshot, collect_approved_sources,
                                                    unavailable_snapshot)

            # This independent stream is observational. It must never enter the
            # scientific corpus or historical baseline. Start it
            # before publication discovery so approved sources overlap with that
            # network work and the later AI stages.
            source_query = "".join(" " if unicodedata.category(char) in {"Cc", "Cf", "Cs"}
                                   else char for char in unicodedata.normalize("NFKC", plan.english_query))
            source_query = " ".join(source_query.split())[:500].rstrip() or plan.original_query[:500]
            approved_saved = context.load_checkpoint("approved_sources")
            approved_snapshot: SourceSnapshot | None = None
            if approved_saved is not None:
                try:
                    approved_snapshot = SourceSnapshot.model_validate(approved_saved)
                except ValidationError:
                    raise TaskFailure("Сохранённый этап дополнительных источников повреждён.") from None
                if approved_snapshot.query != source_query or approved_snapshot.as_of != as_of:
                    raise TaskFailure("Сохранённые наблюдения относятся к другому запросу.")
            else:
                from app.pilot.approved_sources.catalog import SourcePolicy

                try:
                    source_policy = SourcePolicy.from_json(payload.get("source_policy"))
                except (TypeError, ValueError):
                    raise TaskFailure("Сохранённое правило источников повреждено.") from None
                # Русскоязычные источники (Хабр, КиберЛенинка, русское издание
                # Google News) ищут по исходной русской формулировке.
                localized = ({"ru": plan.original_query} if plan.language == "ru"
                             and re.search(r"[А-Яа-яЁё]", plan.original_query) else {})
                # Глубокий режим смотрит у каждого источника до 100 записей —
                # столько почти все адаптеры отдают одним запросом, так что
                # сбор не становится дольше; быстрый делит прежние 900.
                from app.pilot.approved_sources.contracts import SOURCE_IDS

                approved_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="external-sources")
                if collection_profile == "deep":
                    approved_future = approved_pool.submit(collect_approved_sources, source_query,
                        as_of=as_of, cancel=context.cancel_event, policy=source_policy,
                        localized=localized, per_source_cap=DEEP_SOURCE_CAP,
                        max_observations=DEEP_SOURCE_CAP * len(SOURCE_IDS))
                else:
                    approved_future = approved_pool.submit(collect_approved_sources, source_query,
                        as_of=as_of, cancel=context.cancel_event, policy=source_policy,
                        localized=localized)

            # Public NIH project awards are a separate money signal. They do
            # not enter the scientific corpus, publication counts or trend
            # confidence. One bounded request overlaps the other sources only
            # for the main web analysis, which has an explicit collection mode.
            funding_payload: dict | None = None
            # NIH normalizes punctuation in its plain-term search. Keep the
            # checkpoint identity equal to the topic returned by its adapter.
            funding_terms = re.findall(r"[^\W_]+", source_query, flags=re.UNICODE)
            funding_query = (" ".join(funding_terms[:8]) or
                             " ".join(source_query.split()[:8]))[:180]
            month_index = as_of.year * 12 + as_of.month - 1 - 23
            funding_from = date(month_index // 12, month_index % 12 + 1, 1)
            if collection_profile is not None:
                funding_saved = context.load_checkpoint("funding_sources")
                if funding_saved is not None:
                    if (not isinstance(funding_saved, dict)
                            or funding_saved.get("source_id") != "nih_reporter"
                            or funding_saved.get("topic") != funding_query
                            or funding_saved.get("from_date") != funding_from.isoformat()
                            or funding_saved.get("to_date") != as_of.isoformat()):
                        raise TaskFailure("Сохранённые данные финансирования относятся к другому запросу.")
                    funding_payload = funding_saved
                else:
                    from app.pilot.funding_sources import fetch_nih_grants

                    if approved_pool is None:
                        approved_pool = ThreadPoolExecutor(max_workers=2,
                                                           thread_name_prefix="external-sources")
                    funding_future = approved_pool.submit(fetch_nih_grants, funding_query,
                                                          funding_from, as_of,
                                                          cancel=context.cancel_event)

            def save_approved_sources(*, wait_for_result: bool = False) -> None:
                nonlocal approved_snapshot, approved_future
                if approved_future is None or not wait_for_result and not approved_future.done():
                    return
                try:
                    candidate = approved_future.result()
                    approved_snapshot = SourceSnapshot.model_validate(
                        candidate.model_dump(mode="json") if isinstance(candidate, SourceSnapshot) else candidate)
                    if approved_snapshot.query != source_query or approved_snapshot.as_of != as_of:
                        raise ValueError("Observation source returned a different query")
                except TaskCancelled:
                    raise
                except Exception:
                    # The extra sources may fail without invalidating the
                    # scientific result. Their missing coverage stays visible.
                    context.check_cancelled()
                    approved_snapshot = unavailable_snapshot(source_query, as_of)
                context.checkpoint("approved_sources", approved_snapshot.model_dump(mode="json"))
                approved_future = None

            def save_funding_sources(*, wait_for_result: bool = False) -> None:
                nonlocal funding_payload, funding_future
                if funding_future is None or not wait_for_result and not funding_future.done():
                    return
                try:
                    candidate = funding_future.result()
                    if (candidate.topic != funding_query or candidate.from_date != funding_from
                            or candidate.to_date != as_of):
                        raise ValueError("Funding result belongs to another query")
                    funding_payload = candidate.to_dict()
                except TaskCancelled:
                    raise
                except Exception:
                    context.check_cancelled()
                    funding_payload = {
                        "source_id": "nih_reporter", "topic": funding_query,
                        "from_date": funding_from.isoformat(), "to_date": as_of.isoformat(),
                        "date_basis": "award_notice_date",
                        "amount_basis": "reported_fiscal_year_award_usd", "awards": [],
                        "total_available": None, "records_returned": 0,
                        "rejected_records": 0, "partial_coverage": True,
                        "coverage_reason": "source_unavailable",
                    }
                context.checkpoint("funding_sources", funding_payload)
                funding_future = None
            save_approved_on_exit = save_approved_sources
            save_funding_on_exit = save_funding_sources
            seed = None
            if payload.get("seed_arxiv_receipt_hash"):
                from app.pilot.multisource.arxiv import build_arxiv_corpus_snapshot
                from app.pilot.multisource.store import SignalStore

                seed = build_arxiv_corpus_snapshot(SignalStore(self.data_dir), self.archive,
                                                    payload["seed_arxiv_receipt_hash"], plan,
                                                    cancel=context.cancel_event)

            def validate_seed(candidate: CorpusSnapshot) -> CorpusSnapshot:
                if seed is not None:
                    if (seed.coverage[0] not in candidate.coverage or
                            not {item.revision_id for item in seed.documents}.issubset(
                                {item.revision_id for item in candidate.documents})):
                        raise TaskFailure("Сохранённая научная выборка потеряла препринты исходного сигнала.")
                return candidate

            from app.pilot.field_history import EXPOSURE_METHOD, FieldExposure, collect_field_exposure, verify_field_exposure

            # Годовой фон направления зависит только от плана, а собирается
            # полминуты: OpenAlex отвечает с паузами лимита. Веб-анализ начинает
            # его сразу, и он идёт, пока собираются публикации и модель называет
            # темы (замер 28.09.2026: 37 с последовательно после именования).
            # Сохраняет его и сообщает ход по-прежнему основной поток. Если
            # конкретных технологий не найдётся, десять запросов пропадут —
            # настольное приложение их бережёт и собирает фон только по нужде.
            exposure_cache_key = content_hash({
                "artifact": "field-exposure/1.0.0",
                "method_version": EXPOSURE_METHOD,
                "plan_hash": plan.plan_hash,
            })
            if (collection_profile is not None and settings.history_enabled
                    and context.load_checkpoint("field_exposure") is None
                    and self._shared_cache_read("field-exposure", exposure_cache_key) is None):
                exposure_context = _BackgroundContext(context)
                exposure_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="field-exposure-background")
                exposure_future = exposure_pool.submit(collect_field_exposure, plan, cast(RunContext, exposure_context),
                                                       self.credentials)

            discovery_cache_key = content_hash({
                "artifact": "publication-discovery/1.0.0",
                "plan_hash": plan.plan_hash,
                "seed_arxiv_receipt_hash": payload.get("seed_arxiv_receipt_hash"),
            })

            saved = context.load_checkpoint("discovery")
            snapshot = None
            snapshot_origin = None
            if saved is not None:
                snapshot = validate_seed(self._validate_snapshot(saved, plan, context))
                snapshot_origin = "checkpoint"
            else:
                cached_snapshot = self._cache_read(cache_key, "discovery")
                if cached_snapshot is not None:
                    try:
                        snapshot = validate_seed(self._validate_snapshot(cached_snapshot, plan, context))
                        if _source_snapshot_cacheable(snapshot):
                            snapshot_origin = "analysis_cache"
                        else:
                            snapshot = None
                    except (TaskFailure, ValidationError):
                        snapshot = None  # Recollect disposable cache; never mask broken checkpoints.
                if snapshot is None:
                    cached_snapshot = self._shared_cache_read("discovery", discovery_cache_key)
                    if cached_snapshot is not None:
                        try:
                            snapshot = validate_seed(self._validate_snapshot(cached_snapshot, plan, context))
                            if _source_snapshot_cacheable(snapshot):
                                snapshot_origin = "shared_cache"
                            else:
                                snapshot = None
                        except (TaskFailure, ValidationError):
                            snapshot = None
            if snapshot is not None:
                context.progress("discovery", "Используем проверенную выборку, полученную сегодня")
            else:
                snapshot = collect_snapshot(plan, context, self.archive, self.credentials)
                if seed is not None:
                    from app.pilot.multisource.arxiv import merge_arxiv_seed

                    snapshot = merge_arxiv_seed(snapshot, seed)
                snapshot_origin = "collected"
            snapshot_data = snapshot.model_dump(mode="json")
            cacheable = _source_snapshot_cacheable(snapshot)
            if snapshot_origin == "collected" and cacheable:
                self._cache_write(cache_key, "discovery", snapshot_data)
            if snapshot_origin in {"analysis_cache", "collected"} and cacheable:
                self._shared_cache_write("discovery", discovery_cache_key, snapshot_data)
            context.checkpoint("discovery", snapshot_data)
            save_approved_sources()
            save_funding_sources()
            # ТОП-15 технологий веб-анализа нужен только собранной выборке.
            # Он считается в фоне, пока локальная модель называет и проверяет
            # темы: это сетевые запросы, а не GPU, и к концу анализа он готов.
            radar_payload = context.load_checkpoint("radar") if collection_profile is not None else None
            radar_progress = [0, 0]
            if collection_profile is not None and radar_payload is None:
                pending_sources, known_sources = approved_future, approved_snapshot

                def radar_sources() -> dict | None:
                    source: object = known_sources
                    if pending_sources is not None:
                        try:
                            source = pending_sources.result()
                        except Exception:
                            return None
                    if isinstance(source, SourceSnapshot):
                        return source.model_dump(mode="json")
                    return source if isinstance(source, dict) else None

                def radar_step(done: int, total: int) -> None:
                    radar_progress[:] = done, total

                radar_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="technology-radar")
                radar_future = radar_pool.submit(self._technology_radar, payload["query"], plan, snapshot_data,
                                                 radar_sources, cancel=radar_cancel, progress=radar_step,
                                                 **({"max_candidates": DEEP_RADAR_CANDIDATES}
                                                    if collection_profile == "deep" else {}))
            discovered = context.load_checkpoint("candidates")
            if discovered is None:
                discovered = self._discover(context, snapshot, plan)
                context.checkpoint("candidates", discovered)
            save_approved_sources()
            save_funding_sources()
            from app.pilot.evidence import build_passport, label_candidates
            from app.pilot.history import assess_history
            from app.pilot.antecedents import collect_antecedents
            from app.pilot.signal_evidence import extract_signal_evidence, extract_primary_observations
            from app.pilot.contracts import Claim
            from app.pilot.methodology import AssessmentArtifact
            from app.pilot.selection import select_top

            settings_hash = content_hash(settings.model_dump(mode="json"))
            schedule, allocations, attempt_plan_hash = _candidate_attempt_plan(
                discovered, plan, snapshot, context, settings_hash=settings_hash,
                attempts=settings.candidate_attempts)
            attempt_positions = {item.candidate_id: index for index, item in enumerate(schedule)}
            evidence_client = client if settings.external_ai_allowed else None

            def naming_budget_unavailable() -> bool:
                # A refused large batch need not exhaust the budget for a
                # smaller later one. Stop only when no request can fit a scope.
                minimum_cost = evidence_client.config.quote(1, 1) if evidence_client is not None else 0
                for scope in scopes_for_dispatch(scopes):
                    remaining = budget.snapshot(scope)
                    allowance = remaining.remaining
                    if (remaining.requires_reconciliation or allowance.calls == 0
                            or allowance.input_tokens == 0 or allowance.output_tokens == 0
                            or allowance.cost_micro < minimum_cost):
                        return True
                return False

            field_exposure = None
            snapshots = [snapshot]
            cards, artifacts, patents = [], [], []
            deep_history_inputs = {}
            attempt_states = {}
            offset = 0
            while offset < len(schedule):
                context.check_cancelled()
                batch_size = min(30, 2 * settings.candidate_limit) if offset == 0 else 30
                pending = schedule[offset:offset + batch_size]
                _freeze_processing_checkpoint(context, f"candidate_batch_{offset}", {
                    "plan_hash": attempt_plan_hash, "offset": offset,
                    "candidate_ids": [item.candidate_id for item in pending]})
                candidates = label_candidates(pending, snapshot, self.archive, context,
                    client=evidence_client, scope_ids=scopes, query_plan=plan, stage_offset=offset,
                    budget_unavailable=naming_budget_unavailable)
                attempt_states.update(_candidate_batch_outcome(context, pending, candidates,
                    offset=offset, plan_hash=attempt_plan_hash))
                if (field_exposure is None and settings.history_enabled
                        and any(item.specificity == "specific_technology" for item in candidates)):
                    saved_exposure = context.load_checkpoint("field_exposure")
                    if saved_exposure is not None:
                        field_exposure = verify_field_exposure(FieldExposure.model_validate(saved_exposure), plan)
                    else:
                        cached_exposure = self._shared_cache_read("field-exposure", exposure_cache_key)
                        if cached_exposure is not None:
                            try:
                                field_exposure = verify_field_exposure(
                                    FieldExposure.model_validate(cached_exposure), plan)
                            except (ValidationError, ValueError):
                                field_exposure = None
                        if field_exposure is None and exposure_future is not None:
                            context.progress("history", "Дожидаемся годового фона технологического направления")
                            background = _await_background(context, exposure_future)
                            if isinstance(background, FieldExposure):
                                field_exposure = verify_field_exposure(background, plan)
                                if all(item.status == "observed" for item in field_exposure.years):
                                    self._shared_cache_write("field-exposure", exposure_cache_key,
                                                             field_exposure.model_dump(mode="json"))
                        if field_exposure is not None:
                            context.progress("history", "Используем проверенную статистику по годам")
                        else:
                            context.progress("history", "Проверяем годовой фон технологического направления")
                            field_exposure = collect_field_exposure(plan, context, self.credentials)
                            if all(item.status == "observed" for item in field_exposure.years):
                                self._shared_cache_write("field-exposure", exposure_cache_key,
                                                         field_exposure.model_dump(mode="json"))
                        context.checkpoint("field_exposure", field_exposure.model_dump(mode="json"))
                # The local model selects passport quotations that the grounding
                # rules then refuse: 0 of 140 accepted across every local run of
                # 19–24.09.2026, each costing 10–50 seconds before the same
                # extractive selection was used anyway. It is not asked.
                ai_passports = (frozenset() if settings.provider == "local" else
                    _passport_ai_selection(candidates, snapshot, self.archive, context, budget,
                    scopes_for_dispatch(scopes), evidence_client.config if evidence_client is not None else None,
                    settings_hash=settings_hash, checkpoint_stage=f"passport_ai_plan_{offset}"))
                for candidate in candidates:
                    number = attempt_positions[candidate.candidate_id]
                    context.progress("evidence", f"Проверяем доказательства: {number + 1} из {len(schedule)}",
                                     number, len(schedule))
                    passport_client = evidence_client if candidate.candidate_id in ai_passports else None
                    card = build_passport(candidate, snapshot, self.archive, context, client=passport_client, scope_ids=scopes, methodology_version="3.4.0")
                    card = card.model_copy(update={"methodology_version": "3.4.0"})
                    if settings.provider == "local":
                        card = card.model_copy(update={"limitations": tuple(dict.fromkeys((*card.limitations,
                            "Цитаты паспорта отобраны локальным извлечением без AI: локальная модель не выбирает цитаты, проходящие проверку.")))})
                    elif settings.external_ai_allowed and candidate.candidate_id not in ai_passports:
                        card = card.model_copy(update={"limitations": tuple(dict.fromkeys((*card.limitations,
                            "AI-проверка паспорта не выполнялась: кандидат не включён в зафиксированный план внешней проверки; использовано локальное извлечение цитат.")))})
                    allocation = allocations.get(candidate.candidate_id, 0)
                    earlier_limit, history_limit = _history_subbudgets(allocation)
                    if settings.history_enabled and allocation < 2 and candidate.specificity == "specific_technology":
                        card = card.model_copy(update={"limitations": (*card.limitations,
                            "Выделенного бюджета недостаточно для истории и ранних аналогов; сигнал не подтверждён.")})
                    if candidate.specificity == "specific_technology":
                        source_novelty = extract_signal_evidence(candidate, snapshot, self.archive, context, query_plan=plan, method_version="archived-author-novelty/3.0.0")
                        primary_observations = extract_primary_observations(candidate, snapshot, self.archive, context, query_plan=plan, method_version="archived-primary-result/2.0.0")
                        evidence = {item.evidence_id: item for item in card.evidence}
                        claims = list(card.claims)
                        for primary in primary_observations:
                            evidence[primary.result.evidence_id] = primary.result
                            if not any(claim.role == "case" and claim.support == "supported" and primary.result.evidence_id in claim.evidence_ids for claim in claims):
                                claims.append(Claim(claim_id="primary-" + content_hash(primary)[:24], role="case", support="supported", grounding_method="exact-contextual-quotation/4.0.0", text=primary.result.quote, evidence_ids=(primary.result.evidence_id,)))
                        for source in source_novelty:
                            evidence.update((item.evidence_id, item) for item in (source.novelty, source.experiment))
                            claims.append(Claim(claim_id="source-novelty-" + content_hash(source)[:24], role="novelty",
                                text=source.novelty.quote, support="unverified", grounding_method=source.method_version,
                                evidence_ids=tuple(dict.fromkeys((source.novelty.evidence_id, source.experiment.evidence_id)))))
                        card = card.model_copy(update={"evidence": tuple(evidence.values()), "claims": tuple(claims)})
                        history_passport = card
                        checked = assess_history(candidate, plan, snapshot, self.archive, self.credentials, context,
                                                 passport=card, remaining_new_documents=history_limit if settings.history_enabled else 0,
                                                 methodology_version="3.4.0", field_exposure=field_exposure,
                                                 source_novelty=source_novelty, antecedents=None,
                                                 primary_observations=primary_observations)
                        if settings.history_enabled and earlier_limit > 0:
                            deep_history_inputs[candidate.candidate_id] = (
                                candidate, history_passport, source_novelty, primary_observations,
                                earlier_limit, history_limit)
                        artifacts.append(checked.artifact.model_dump(mode="json"))
                        snapshots.append(checked.snapshot)
                        card = checked.card
                    if settings.patents_enabled and candidate.specificity == "specific_technology":
                        from app.pilot.enrichment import PatentSignal, collect_patent_signal

                        stage = "patents_" + str(number)
                        saved_patents = context.load_checkpoint(stage)
                        signal = (PatentSignal.model_validate(saved_patents) if saved_patents is not None else
                                  collect_patent_signal(candidate, plan, self.archive, self.credentials, context))
                        if signal.candidate_id != candidate.candidate_id:
                            raise TaskFailure("Сохранённая проверка патентов относится к другой технологии.")
                        context.checkpoint(stage, signal.model_dump(mode="json"))
                        patents.append(signal.model_dump(mode="json"))
                        snapshots.append(CorpusSnapshot(snapshot_id=content_hash(signal), plan_hash=plan.plan_hash,
                            purpose="enrichment", as_of=plan.as_of, created_at=datetime.now(UTC), documents=signal.documents,
                            coverage=(signal.coverage,), normalizer_version=signal.version, deduplication_version="epo-family-id-v1"))
                    cards.append(card)
                offset += len(pending)
                validated_artifacts = tuple(AssessmentArtifact.model_validate(item) for item in artifacts)
                if settings.history_enabled:
                    shortlisted = _antecedent_shortlist(cards, validated_artifacts,
                        set(deep_history_inputs), limit=settings.candidate_limit)
                    if len(shortlisted) >= settings.candidate_limit:
                        break
                elif len(select_top(cards, validated_artifacts,
                                    limit=settings.candidate_limit)) >= settings.candidate_limit:
                    break

            validated_artifacts = tuple(AssessmentArtifact.model_validate(item) for item in artifacts)
            shortlist = (_antecedent_shortlist(cards, validated_artifacts, set(deep_history_inputs),
                                                limit=settings.candidate_limit)
                         if settings.history_enabled else ())
            for position, identifier in enumerate(shortlist, start=1):
                context.check_cancelled()
                candidate, history_passport, source_novelty, primary_observations, earlier_limit, history_limit = (
                    deep_history_inputs[identifier])
                context.progress("antecedents",
                    f"Проверяем ранние аналоги финалистов: {position} из {len(shortlist)}",
                    position - 1, len(shortlist))
                antecedents = collect_antecedents(candidate, plan, self.archive, self.credentials, context,
                                                  max_documents=earlier_limit)
                checked = assess_history(candidate, plan, snapshot, self.archive, self.credentials, context,
                    passport=history_passport, remaining_new_documents=history_limit,
                    methodology_version="3.4.0", field_exposure=field_exposure,
                    source_novelty=source_novelty, antecedents=antecedents,
                    primary_observations=primary_observations)
                snapshots.append(antecedents.snapshot)
                snapshots.append(checked.snapshot)
                cards = [checked.card if item.candidate.candidate_id == identifier else item for item in cards]
                artifacts = [checked.artifact.model_dump(mode="json")
                    if item["assessment"]["candidate_id"] == identifier else item for item in artifacts]

            _rank_cards(cards, tuple(AssessmentArtifact.model_validate(item) for item in artifacts))
            # Keep every assessed passport. The selection layer suppresses TOP
            # duplicates without erasing their evidence or original provenance.
            selected_ids = {card.candidate.candidate_id for card in cards}
            artifacts = [item for item in artifacts if item["assessment"]["candidate_id"] in selected_ids]
            queue = {item["candidate_id"]: Candidate.model_validate(item) for item in
                     (*discovered.get("candidates", []), *discovered.get("review_queue", []))
                     if item["candidate_id"] not in selected_ids}
            reviews = []
            previous_review_states = {}
            refinement_rejected = False
            refinement_notices = ()
            if refined_from is not None:
                source_result = AnalysisResult.model_validate(refined_from["result"])
                previous_review_states = {item.candidate_id: item.model_dump(mode="json")
                                          for item in source_result.candidate_review_states or ()}
                replaced = {item.candidate.candidate_id for item in cards}
                refinement_rejected = payload["candidate_id"] not in replaced
                cards = [item for item in source_result.cards if item.candidate.candidate_id not in replaced] + cards
                artifacts = [item for item in refined_from.get("assessments", [])
                             if item["assessment"]["candidate_id"] not in replaced] + artifacts
                reviews = [item for item in refined_from.get("reviews", [])
                           if item["decision"]["candidate_id"] not in replaced]
                cards, refinement_notices = _retain_refinement_notices(source_result, cards, replaced,
                    payload["candidate_id"] if refinement_rejected else None,
                    {item["decision"]["candidate_id"] for item in reviews})
                patents = [item for item in refined_from.get("patent_signals", [])
                           if item["candidate_id"] not in replaced] + patents
                queue.update((item.candidate_id, item) for item in source_result.candidate_queue if item.candidate_id not in replaced)
                for retained_card in cards:
                    queue.pop(retained_card.candidate.candidate_id, None)
                snapshots = list({item.snapshot_id: item for item in (*source_result.snapshots, *snapshots)}.values())
                _rank_cards(cards, tuple(AssessmentArtifact.model_validate(item) for item in artifacts))
            # Bind the last attempt to the candidate actually retained after a
            # rejected refinement, without altering its old scientific passport.
            final_candidates = {item.candidate_id: item for item in queue.values()}
            final_candidates.update((card.candidate.candidate_id, card.candidate) for card in cards)
            for identifier, state in attempt_states.items():
                candidate = final_candidates[identifier]
                previous_review_states[identifier] = state | {
                    "candidate_hash": content_hash(candidate),
                    "discovery_snapshot_id": candidate.discovery_snapshot_id}
            artifacts = _reconcile_publication_status(cards, artifacts, snapshots, plan, self.archive, context)
            _rank_cards(cards, tuple(AssessmentArtifact.model_validate(item) for item in artifacts))
            snapshots = list(_retained_snapshots(snapshots, cards, tuple(queue.values()),
                tuple(AssessmentArtifact.model_validate(item) for item in artifacts),
                patent_ids=tuple(content_hash(item) for item in patents)))
            top_ids = select_top(cards, tuple(AssessmentArtifact.model_validate(item) for item in artifacts), limit=settings.candidate_limit)
            # These categories are evidence gates, never padding to exactly 15.
            quality = "insufficient_data" if not cards else "complete" if all(card.quality == "complete" for card in cards) else "partial"
            limitations = ["Выводы относятся к сохранённым источникам и зафиксированным правилам отбора.",
                           "Веса и пороги требуют независимой предметной оценки."]
            if local_model_notice:
                limitations.append(local_model_notice)
            elif settings.provider == "local":
                limitations.append("Область каждого кандидата подтверждена цитатой из одного представителя "
                                   "группы: локальная модель надёжно цитирует один документ. Внешний AI "
                                   "проверяет связь по нескольким представителям.")
            if local_plan_fallback:
                limitations.append("Локальная модель не составила проверяемый план области: поиск выполнен "
                                   "по исходной английской формулировке или её машинному переводу без расширения направления.")
            if collection_profile == "fast":
                limitations.append("Быстрый режим использует сокращённую выборку и исторический бюджет; "
                                   "результат предварительный и может не включать слабые сигналы.")
            if quality != "complete":
                limitations.append("Недостаток истории или доказательств показан в каждой карточке; размер TOP ограничен прошедшими допуск кандидатами.")
            limitations.append("Автоматические weak signal и emerging — проверяемые гипотезы; заявления авторов не заменяют независимую оценку новизны.")
            if refinement_rejected:
                limitations.append(_LEGACY_REJECTED_REFINEMENT)
            limitations.extend(refinement_notices)
            from app.pilot.completion import evaluation_progress, label_stage_limitations

            unresolved_ids = {card.candidate.candidate_id for card in cards
                              if card.candidate.specificity != "specific_technology"}
            # At most 60 distinct attempted candidates and disjoint label stages.
            # Persist safe explanations in portable results, but derive counts and
            # completion anew after loading/refinement/review instead of migrating
            # scientific states or treating an empty TOP as an execution failure.
            label_checkpoints = (context.load_checkpoint(f"labels_{index}") for index in range(60))
            limitations.extend(label_stage_limitations((item for item in label_checkpoints if item), unresolved_ids))
            if not settings.history_enabled and any(card.candidate.specificity == "specific_technology"
                    and not card.assessment_hash for card in cards):
                limitations.append("Историческая проверка выключена в настройках этого запуска; без неё стадия кандидатов не подтверждена.")
            result = AnalysisResult.model_validate(dict(
                result_id=uuid4().hex, run_id=context.run_id, query_plan=plan.model_dump(mode="json"),
                methodology_version="3.4.0", top_limit=settings.candidate_limit,
                created_at=datetime.now(UTC), quality=quality, snapshots=snapshots, cards=cards,
                top_trend_ids=top_ids,
                candidate_queue=tuple(queue.values()),
                candidate_review_states=tuple(previous_review_states.values()),
                limitations=limitations))
            from app.pilot.export import verify_result
            from app.pilot.review import ReviewRecord

            verify_result(result, self.archive, tuple(AssessmentArtifact.model_validate(item) for item in artifacts),
                          reviews=tuple(ReviewRecord.model_validate(item) for item in reviews), context=context)
            context.checkpoint("assessments", {"items": artifacts})
            if approved_future is not None and not approved_future.done():
                context.progress("external_sources", "Завершаем сбор дополнительных источников")
            save_approved_sources(wait_for_result=True)
            save_funding_sources(wait_for_result=True)
            if approved_snapshot is None:
                approved_snapshot = unavailable_snapshot(source_query, as_of)
            result_json = result.model_dump(mode="json")
            approved_json = approved_snapshot.model_dump(mode="json")
            # The web TOP is drawn from one deduplicated scientific + approved
            # source pool. Score exactly those preliminary 15 publications,
            # including observations whose rights permit local processing only.
            from app.pilot.publication_confidence import completed_cached_scores, score_publications
            from app.web_api import _publication_pool, web_result

            if settings.provider == "local" and client is not None:
                # The workflow is finished with Qwen. Release its weights
                # before loading the scientific encoder for a small batch.
                client.close()
                client = None
            # Тема каждой публикации оценивается до выбора ТОПа: сначала смысл
            # и слова, потом место в выдаче. Возобновлённый анализ берёт оценку
            # из контрольной точки.
            from app.relevance_learning import (RelevanceModel, discovery_decisions, run_samples,
                                                save_run_samples)

            pool = _publication_pool(result_json, approved_snapshot, self.archive, keep_studies=True)
            discovery_relevance = discovery_decisions(discovered.get("relevance") or ())
            publication_relevance = context.load_checkpoint("publication_relevance")
            relevance_encoder = None
            if publication_relevance is None:
                context.progress("publish", "Проверяем, какие публикации относятся к теме запроса")
                if pool and (self.model_dir / "tokenizer.json").is_file():
                    from app.pilot.encoder import EncoderError, MultilingualEncoder

                    try:
                        relevance_encoder = MultilingualEncoder(self.model_dir, cancel=context.cancel_event)
                    except CancelledError:
                        context.check_cancelled()
                    except (EncoderError, OSError):
                        relevance_encoder = None  # Без смысла тема оценивается по словам плана.
                from app.pilot.publication_relevance import assess_pool

                publication_relevance = assess_pool(
                    pool, plan, discovery=discovery_relevance, encoder=relevance_encoder,
                    cancel=context.cancel_event, model=RelevanceModel.load(self.data_dir),
                    progress=lambda done, total: context.progress(
                        "publish", f"Проверяем тему публикаций источников: {done} из {total}", done, total))
                context.checkpoint("publication_relevance", publication_relevance)
            try:
                # Примеры этого анализа — материал для самообучения следующих.
                save_run_samples(self.data_dir, run_samples(
                    context.run_id, plan.model_dump(mode="json"), pool,
                    relevance={publication_id: {"cosine": entry[3]} for publication_id, entry in
                               (publication_relevance.get("items") or {}).items() if isinstance(entry, list)
                               and len(entry) >= 4},
                    discovery=discovery_relevance, created_at=datetime.now(UTC).isoformat()))
            except (OSError, ValueError, TypeError):
                pass  # Обучение — дополнение: без примеров анализ остаётся верным.
            del pool
            preliminary_top = web_result({"result": result_json, "assessments": artifacts,
                                          "approved_sources": approved_json,
                                          "funding_sources": funding_payload,
                                          "publication_relevance": publication_relevance},
                                         archive=self.archive)["top_publications"]
            confidence_encoder = relevance_encoder
            cached_confidences = None
            if preliminary_top and confidence_encoder is None and (self.model_dir / "tokenizer.json").is_file():
                from app.pilot.encoder import EncoderError, MultilingualEncoder, spec_fingerprint
                from app.runtime.inference import execution_providers

                try:
                    # On a CPU-only runtime the model fingerprint is known from
                    # its pinned specification. A fully checkpointed resume can
                    # reuse the verified scores without loading ONNX weights.
                    if execution_providers() == ("CPUExecutionProvider",):
                        context.check_cancelled()
                        cached_confidences = completed_cached_scores(
                            preliminary_top, query=plan.original_query,
                            english_query=plan.english_query,
                            expected_fingerprint=spec_fingerprint(), context=context)
                    if cached_confidences is None:
                        confidence_encoder = MultilingualEncoder(self.model_dir, cancel=context.cancel_event)
                except CancelledError:
                    context.check_cancelled()
                except (EncoderError, OSError):
                    pass  # The scientific result remains valid without this optional score.
            publication_confidences = (cached_confidences if cached_confidences is not None else
                score_publications(preliminary_top, query=plan.original_query,
                                   english_query=plan.english_query, encoder=confidence_encoder,
                                   context=context))
            del confidence_encoder, relevance_encoder
            if radar_future is not None and not radar_future.done() and self.radar_handoff:
                # ТОП технологий упирается в правило arXiv (запрос раз в 3 секунды
                # на весь процесс) и может досчитываться дольше самого анализа.
                # Результат публикуется сразу, а досчёт забирает сайт.
                self._hand_off_radar(context.run_id, radar_future, radar_progress, radar_cancel)
                radar_handed = True
                radar_payload = {"state": "pending"}
            elif radar_future is not None:
                radar_payload = _await_radar(context, radar_future, radar_progress)
                context.checkpoint("radar", radar_payload)
            spending = budget.snapshot(run_scope)
            completion = evaluation_progress(result.model_dump(mode="python"))
            context.progress("complete", completion.message + " " + completion.details, 1, 1)
            return {"result": result_json, "assessments": artifacts,
                    "radar": radar_payload,
                    "reviews": reviews,
                    "patent_signals": patents,
                    "approved_sources": approved_json,
                    "funding_sources": funding_payload,
                    "publication_confidences": publication_confidences,
                    "publication_relevance": publication_relevance,
                    "discovery_summary": {key: discovered[key] for key in
                                          ("input_records", "unique_studies", "retained_studies", "quality", "limitations",
                                           "discovery_version", "hierarchy", "unassigned_study_ids", "review_queue_metadata")
                                          if key in discovered},
                    "budget": {"currency": settings.currency, "cost_micro": spending.used.cost_micro,
                               "calls": spending.used.calls, "requires_reconciliation": spending.requires_reconciliation}}
        except (LlmCancelled, WorkerCancelled):
            raise TaskCancelled() from None
        except (LlmError, BudgetError, QueryError, CredentialUnavailable, WorkerError) as error:
            raise TaskFailure(str(error)) from None
        finally:
            if not context.cancel_event.is_set():
                # A failed scientific stage can finish after the independent
                # sources have already returned. Preserve only finished work;
                # never wait for a network call or replace the original error.
                for future, save in ((approved_future, save_approved_on_exit),
                                     (funding_future, save_funding_on_exit)):
                    if future is not None and future.done() and save is not None:
                        try:
                            save()
                        except Exception:
                            pass
            if approved_pool is not None:
                # A failed or cancelled scientific run should not wait for
                # unrelated HTTP timeouts before its state is reported.
                approved_pool.shutdown(wait=all(future is None or future.done()
                                                for future in (approved_future, funding_future)),
                                       cancel_futures=True)
            if radar_pool is not None:
                # После сбоя или отмены радар бросает сетевые запросы сам;
                # переданный сайту досчёт продолжается.
                if not radar_handed:
                    radar_cancel.set()
                radar_pool.shutdown(wait=False, cancel_futures=not radar_handed)
            if exposure_pool is not None:
                # Ненужный (сбой, отмена или нет технологий) фон не держит завершение анализа.
                if exposure_context is not None and exposure_future is not None and not exposure_future.done():
                    exposure_context.cancel_event.set()
                exposure_pool.shutdown(wait=False, cancel_futures=True)
            if client is not None:
                client.close()

    def _validate_snapshot(self, value, plan, context):
        snapshot = CorpusSnapshot.model_validate(value)
        if snapshot.plan_hash != plan.plan_hash:
            raise TaskFailure("Сохранённая выборка относится к другой области анализа.")
        for reference in snapshot.documents:
            context.check_cancelled()
            self.archive.get(reference.revision_id)
        return snapshot

    def _discover(self, context: RunContext, snapshot: CorpusSnapshot, plan: QueryPlan) -> dict:
        from app.pilot.discovery import task

        context.progress("relevance", "Сопоставляем смысл запроса и публикаций, выделяем технологические группы")
        documents = []
        for reference in snapshot.documents:
            context.check_cancelled()
            document = self.archive.get(reference.revision_id)
            documents.append(document.model_dump(mode="json"))
        working = self.data_dir / "tmp"
        working.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="analysis-", dir=working) as directory:
            source, destination = Path(directory) / "input.json", Path(directory) / "output.json"
            source.write_text(json.dumps({"query_plan": plan.model_dump(mode="json"), "documents": documents,
                                          "discovery_snapshot_id": snapshot.snapshot_id,
                                          "model_dir": str(self.model_dir),
                                          "cache_dir": str(self.data_dir / "cache" / "embeddings")},
                                         ensure_ascii=False, allow_nan=False), encoding="utf-8")
            def report(completed: int, total: int) -> None:
                # Cancellation is owned by the worker loop; a stale UI update
                # must not turn into a task failure inside the computation.
                try:
                    context.progress("relevance", "Сопоставляем смысл запроса и публикаций: "
                                     f"{completed} из {total}", completed, total)
                except (TaskCancelled, ValueError):
                    pass

            from app.pilot.discovery import MAX_INPUT_BYTES, MAX_OUTPUT_BYTES

            # A ten-thousand-study corpus is minutes of embedding work, so the
            # budget is an hour rather than the quarter of one that a 1500-study
            # run needed. Cancellation still stops it immediately.
            run_in_process(task, source, destination, context.cancel_event, credentials=self.credentials,
                           timeout_seconds=3600, max_input_bytes=MAX_INPUT_BYTES,
                           max_output_bytes=MAX_OUTPUT_BYTES, on_progress=report)
            return json.loads(destination.read_text(encoding="utf-8"))
