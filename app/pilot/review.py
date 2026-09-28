"""Explicit attributed expert decisions, reproducible assessment and local journal.

Only the user-facing expert workflow should construct a ReviewDecision. No LLM
client, automatic approval or network operation exists here. The reviewer name
is locally declared, not authenticated or cryptographically certified. Content
hashes establish integrity and reproducibility, not scientific truth or identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Self
from uuid import uuid4

from pydantic import Field, model_validator

from app.pilot.antecedents import AntecedentBundle, verify_antecedents
from app.pilot.archive import DocumentArchive, RevisionSource
from app.pilot.contracts import (
    METHODOLOGY_VERSION, MethodologyVersion, Claim, Contract, CorpusSnapshot, TrendCard, Evidence, DocumentRevisionRef,
    content_hash, verify_evidence_text,
)
from app.pilot.evidence import EvidenceError, archived_field, quote_evidence
from app.pilot.history import CancellationContext, assess_snapshot
from app.pilot.methodology import AssessmentArtifact, NoveltyAssessment, NoveltyKind
from app.pilot.reports import open_local_regular
from app.runtime.jobs import TaskCancelled

if TYPE_CHECKING:
    from app.runtime.backup import ArchiveCancellation

REVIEW_VERSION: Final = "manual-novelty-review/1.0.0"
REVIEW_GROUNDING: Final = "reviewed-novelty/manual-v1"
MAX_REVIEW_BYTES = 25_000_000


class _LocalVerification:
    def __init__(self, cancel: ArchiveCancellation | None = None,
                 context: CancellationContext | None = None) -> None:
        self.deadline = time.monotonic() + 120
        self.cancel = cancel
        self.context = context

    def check_cancelled(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise TaskCancelled()
        if self.context is not None:
            self.context.check_cancelled()
        if time.monotonic() > self.deadline:
            raise EvidenceError("Проверка экспертного решения превысила допустимое время.")


class FieldReview(Contract):
    """Explicit semantic judgment; an archived quote never supplies this itself."""
    version: Literal["manual-claim-review/1.0.0"] = "manual-claim-review/1.0.0"
    role: Literal["problem", "advantage", "case", "application"]
    source_evidence_id: str = Field(min_length=1, max_length=2048)
    text_field: Literal["title", "abstract", "full_text"]
    quote: str = Field(min_length=10, max_length=1500)
    verdict: Literal["supported", "unverified", "contradicted"]
    rationale: str = Field(min_length=40, max_length=2000)
    context_checked: bool = Field(default=False, strict=True)
    application_kind: Literal["research", "demonstrator", "deployment"] | None = None

    @model_validator(mode="after")
    def valid_role(self) -> Self:
        if not self.context_checked or len(self.rationale.strip()) < 40:
            raise ValueError("Проверьте контекст, связь с механизмом и запишите содержательное обоснование.")
        if self.role == "application" and self.verdict == "supported" and self.application_kind is None:
            raise ValueError("Подтверждение применения требует явного указания стадии.")
        if self.role != "application" and self.application_kind is not None:
            raise ValueError("Стадия применения относится только к полю application.")
        if self.role in {"problem", "advantage", "application"} and self.text_field == "title":
            raise ValueError("Для содержательного вывода недостаточно названия: выберите цитату содержания.")
        return self


class ReviewDecision(Contract):
    version: Literal["manual-novelty-review/1.0.0"] = REVIEW_VERSION
    reviewer_name: str = Field(min_length=2, max_length=200)
    reviewed_at: datetime
    candidate_id: str = Field(min_length=1, max_length=2048)
    admission_rule_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    bundle_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    kind: NoveltyKind
    rationale: str = Field(min_length=40, max_length=2000)
    mechanism_comparison: str = Field(min_length=40, max_length=2000)
    terminology_review: str = Field(min_length=30, max_length=1500)
    current_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=10)
    earlier_evidence_ids: tuple[str, ...] = Field(default=(), max_length=10)
    earlier_analogues_checked: bool = Field(strict=True)
    terminology_changes_checked: bool = Field(strict=True)
    supersedes_review_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    field_reviews: tuple[FieldReview, ...] = Field(default=(), max_length=4, exclude_if=lambda value: not value)

    @property
    def decision_hash(self) -> str:
        return content_hash(self)

    @model_validator(mode="after")
    def distinct_references(self) -> Self:
        for identifiers in (self.current_evidence_ids, self.earlier_evidence_ids):
            if len(identifiers) != len(set(identifiers)):
                raise ValueError("Одно свидетельство нельзя выбрать несколько раз.")
        if len({review.role for review in self.field_reviews}) != len(self.field_reviews):
            raise ValueError("Для одного поля допустимо одно явное экспертное решение.")
        if len(self.reviewer_name.strip()) < 2:
            raise ValueError("Укажите имя эксперта; пробелы не удостоверяют авторство решения.")
        if self.reviewed_at > datetime.now(UTC) + timedelta(minutes=5):
            raise ValueError("Время экспертного решения не может быть будущим.")
        return self


def _reviewed_fields(decision: ReviewDecision, passport: TrendCard, archive: RevisionSource,
                     verification: CancellationContext) -> tuple[tuple[Claim, ...], tuple[Evidence, ...]]:
    sources = {item.evidence_id: item for item in passport.evidence}
    claims, evidence_items = [], []
    for review in decision.field_reviews:
        verification.check_cancelled()
        source = sources.get(review.source_evidence_id)
        if source is None or source.study_id not in passport.candidate.discovery_study_ids:
            raise EvidenceError("Экспертная цитата должна относиться к текущим исследованиям кандидата.")
        document = archive.get(source.revision_id)
        verify_evidence_text(source, archived_field(document, source.text_field))
        if source.source != document.source or source.source_url != document.url:
            raise EvidenceError("Архивный источник экспертного поля изменён.")
        reference = DocumentRevisionRef(revision_id=source.revision_id, study_id=source.study_id,
            source=source.source, source_id=document.source_id, text_hash=source.text_hash,
            observed_at=document.fetched_at, publication_year=document.publication_year)
        evidence = quote_evidence(reference, document, text_field=review.text_field, quote=review.quote)
        evidence_items.append(evidence)
        text = (f"Экспертная оценка поля ({decision.reviewer_name}; {decision.reviewed_at.date().isoformat()}): "
                f"{review.rationale}\nИсходная цитата: {review.quote}")
        method = (f"reviewed-application/{review.application_kind}/manual-v1" if review.role == "application"
                  and review.application_kind else "reviewed-evidence/manual-v1")
        claims.append(Claim(claim_id=f"review-field-{review.role}-" + decision.decision_hash,
            role=review.role, text=text, support=review.verdict, evidence_ids=(evidence.evidence_id,),
            grounding_method=method))
    return tuple(claims), tuple(evidence_items)


def apply_novelty_review(decision: ReviewDecision, bundle: AntecedentBundle, passport: TrendCard,
                         archive: RevisionSource, *, context: CancellationContext | None = None,
                         methodology_version: MethodologyVersion = METHODOLOGY_VERSION,
                         ) -> tuple[TrendCard, NoveltyAssessment]:
    """Apply a manually supplied, attributed decision; never infer one from AI."""
    verification = context if context is not None else _LocalVerification()
    verification.check_cancelled()
    decision = ReviewDecision.model_validate(decision.model_dump())
    bundle = AntecedentBundle.model_validate_json(bundle.model_dump_json())
    passport = TrendCard.model_validate_json(passport.model_dump_json())
    verification.check_cancelled()
    candidate = passport.candidate
    verify_antecedents(bundle, candidate, bundle.query_plan, archive, check=verification.check_cancelled)
    if (decision.candidate_id != candidate.candidate_id or decision.admission_rule_hash != candidate.admission_rule_hash
            or decision.bundle_hash != bundle.bundle_hash or decision.reviewed_at < bundle.snapshot.created_at):
        raise EvidenceError("Экспертное решение относится к другому определению, обзору или времени.")
    if not decision.earlier_analogues_checked or not decision.terminology_changes_checked:
        raise EvidenceError("Подтверждение требует отдельной проверки ранних аналогов и смены терминологии.")
    if decision.field_reviews and methodology_version == "3.0.0":
        raise EvidenceError("Содержательная проверка полей требует методологии 3.1.0; старый расчёт неизменяем.")
    field_claims, field_evidence = _reviewed_fields(decision, passport, archive, verification)
    current = {item.evidence_id: item for item in passport.evidence}
    earlier = {item.evidence_id: item for item in bundle.evidence}
    if not set(decision.current_evidence_ids).issubset(current) or not set(decision.earlier_evidence_ids).issubset(earlier):
        raise EvidenceError("Выбранное свидетельство отсутствует в текущем паспорте или обзоре аналогов.")
    selected_current = [current[identifier] for identifier in decision.current_evidence_ids]
    # A reviewer may quote a previously unselected abstract passage of an
    # already archived case. It is saved and included in the reproducible record.
    selected_current.extend(item for item in field_evidence if item.evidence_id not in decision.current_evidence_ids)
    if any(item.study_id not in candidate.discovery_study_ids for item in selected_current):
        raise EvidenceError("Текущие свидетельства должны относиться к исходным исследованиям кандидата.")
    if not any(item.text_field in {"abstract", "full_text"} for item in selected_current):
        raise EvidenceError("Название работы недостаточно для сравнения механизмов: выберите точную цитату содержания.")
    if decision.kind in {"established", "renamed"} and not decision.earlier_evidence_ids:
        raise EvidenceError("Старый или переименованный метод требует конкретного более раннего источника.")
    if decision.kind.startswith("new_"):
        if not bundle.search_complete:
            raise EvidenceError("Неполный поиск ранних аналогов не допускает подтверждение новизны; завершите или сузьте обзор.")
        if bundle.matched_study_ids and not decision.earlier_evidence_ids:
            raise EvidenceError("Найдены ранние совпадения: сравните механизм хотя бы с одним конкретным аналогом.")
    selected_earlier = [earlier[identifier] for identifier in decision.earlier_evidence_ids]
    for evidence in (*selected_current, *selected_earlier):
        verification.check_cancelled()
        document = archive.get(evidence.revision_id)
        verify_evidence_text(evidence, archived_field(document, evidence.text_field))
        if (evidence.source_url != document.url or evidence.source != document.source
                or evidence.study_id != document.document_key):
            raise EvidenceError("Экспертное свидетельство не соответствует архивному документу.")
    claim_id = "review-" + decision.decision_hash
    text = (f"Экспертная оценка ({decision.reviewer_name}; {decision.reviewed_at.date().isoformat()}): "
            f"{decision.rationale}\nСравнение механизмов: {decision.mechanism_comparison}\n"
            f"Проверка терминологии: {decision.terminology_review}")
    claim = Claim(claim_id=claim_id, role="novelty", text=text, support="supported",
        evidence_ids=tuple(dict.fromkeys((*decision.current_evidence_ids,
            *(item.evidence_id for item in field_evidence), *decision.earlier_evidence_ids))),
        grounding_method=REVIEW_GROUNDING)
    reviewed_roles = {item.role for item in field_claims}
    claims = tuple(item for item in passport.claims if item.role != "novelty" and item.role not in reviewed_roles) + field_claims + (claim,)
    all_evidence = tuple({item.evidence_id: item for item in (*passport.evidence, *selected_earlier, *field_evidence)}.values())
    if len(all_evidence) > 200 or len(claims) > 40:
        raise EvidenceError("Превышен бюджет доказательств паспорта; уменьшите число выбранных цитат.")
    limitations = tuple(dict.fromkeys((*passport.limitations,
        "Новизна является указанным экспертным суждением по доступным источникам, а не доказательством мировой научной новизны.",
        "Имя reviewer заявлено локально; журнал не удостоверяет личность и не заменяет независимую научную экспертизу.")))
    expanded = TrendCard.model_validate(passport.model_dump(mode="python") | dict(
        methodology_version=None if methodology_version == "3.0.0" else methodology_version,
        category="early_signal" if methodology_version == "3.0.0" else "insufficient_evidence",
        quality="partial", claims=claims, evidence=all_evidence,
        historical_snapshot_id=None, assessment_hash=None, limitations=limitations))
    return expanded, NoveltyAssessment(kind=decision.kind, claim_id=claim_id,
        earlier_analogues_checked=True, terminology_changes_checked=True)


class ReviewRecord(Contract):
    record_type: Literal["expert_novelty_review"] = "expert_novelty_review"
    version: Literal["manual-novelty-review/1.0.0"] = REVIEW_VERSION
    created_at: datetime
    decision: ReviewDecision
    bundle: AntecedentBundle
    source_card: TrendCard
    reviewed_card: TrendCard
    artifact: AssessmentArtifact
    historical_snapshot: CorpusSnapshot

    @property
    def review_id(self) -> str:
        return content_hash(self)

    @model_validator(mode="after")
    def consistent_record(self) -> Self:
        if (self.decision.bundle_hash != self.bundle.bundle_hash
                or self.source_card.candidate != self.reviewed_card.candidate
                or self.artifact.inputs.candidate != self.reviewed_card.candidate
                or self.reviewed_card.assessment_hash != self.artifact.assessment.assessment_hash
                or self.historical_snapshot.snapshot_id != self.artifact.inputs.history.snapshot_id):
            raise ValueError("Запись экспертного журнала содержит несогласованные результаты.")
        return self


def verify_review_record(record: ReviewRecord, archive: RevisionSource, *,
                         context: CancellationContext | None = None) -> None:
    """Repeat quotations, temporal admission, groups, gates and reviewed decision."""
    verification = context if context is not None else _LocalVerification()
    verification.check_cancelled()
    record = ReviewRecord.model_validate_json(record.model_dump_json())
    version = record.artifact.assessment.methodology_version
    if version in {"3.2.0", "3.3.0", "3.4.0"}:
        from app.pilot.field_history import verify_field_exposure
        from app.pilot.signal_evidence import verify_signal_sources, verify_primary_sources
        from app.pilot.evidence_versions import novelty_method, primary_method

        verify_signal_sources(record.artifact.inputs.source_novelty, record.source_card.candidate,
                              archive, verification, query_plan=record.bundle.query_plan,
                              method_version=novelty_method(version) if version == "3.4.0" else None)
        verify_primary_sources(record.artifact.inputs.primary_observations, record.source_card.candidate,
                               archive, verification, query_plan=record.bundle.query_plan,
                               method_version=primary_method(version) if version == "3.4.0" else None)
        if record.artifact.inputs.field_exposure is not None:
            verify_field_exposure(record.artifact.inputs.field_exposure, record.bundle.query_plan)
        if record.artifact.inputs.antecedents != record.bundle:
            raise EvidenceError("Экспертная запись относится к другому архивному поиску аналогов.")
    expanded, novelty = apply_novelty_review(record.decision, record.bundle, record.source_card, archive,
                                            context=verification, methodology_version=version)
    artifact, card, _ = assess_snapshot(expanded.candidate, record.bundle.query_plan,
        record.historical_snapshot, archive, verification, passport=expanded, verified_novelty=novelty,
        methodology_version=version, antecedents=record.bundle if version != "3.0.0" else None,
        field_exposure=record.artifact.inputs.field_exposure,
        source_novelty=record.artifact.inputs.source_novelty, primary_observations=record.artifact.inputs.primary_observations,
        publication_status_revisions=record.artifact.inputs.publication_status_revisions)
    if artifact != record.artifact or card != record.reviewed_card:
        raise EvidenceError("Экспертная запись не воспроизводится по сохранённым источникам и методике.")
    verification.check_cancelled()


def read_review(directory: Path, review_id: str, *, archive: DocumentArchive | None = None,
                 context: CancellationContext | None = None) -> ReviewRecord:
    verification = context if context is not None else _LocalVerification()
    verification.check_cancelled()
    if not re.fullmatch(r"[a-f0-9]{64}", review_id):
        raise EvidenceError("Некорректный идентификатор экспертного решения.")
    try:
        with open_local_regular(directory / (review_id + ".json")) as stream:
            data = stream.read(MAX_REVIEW_BYTES + 1)
        verification.check_cancelled()
        if len(data) > MAX_REVIEW_BYTES or hashlib.sha256(data).hexdigest() != review_id:
            raise ValueError("Review digest mismatch")
        record = ReviewRecord.model_validate_json(data)
        if record.review_id != review_id:
            raise ValueError("Review canonical digest mismatch")
    except (OSError, ValueError, RecursionError):
        raise EvidenceError("Экспертная запись повреждена или отсутствует.") from None
    if archive is not None:
        verify_review_record(record, archive, context=verification)
    verification.check_cancelled()
    return record


def record_review(directory: Path, decision: ReviewDecision, bundle: AntecedentBundle, *,
                   source_card: TrendCard, reviewed_card: TrendCard, artifact: AssessmentArtifact,
                   historical_snapshot: CorpusSnapshot, archive: DocumentArchive,
                   cancel: ArchiveCancellation | None = None,
                   context: CancellationContext | None = None) -> ReviewRecord:
    """Append a self-contained content-addressed journal entry after verification.

    ``directory`` is data_dir / 'reviews'. Existing entries are never edited;
    supersedes_review_id explicitly links a replacement decision to its history.
    """
    verification = _LocalVerification(cancel, context)
    verification.check_cancelled()
    record = ReviewRecord(created_at=datetime.now(UTC), decision=decision, bundle=bundle,
        source_card=source_card, reviewed_card=reviewed_card, artifact=artifact,
        historical_snapshot=historical_snapshot)
    verify_review_record(record, archive, context=verification)
    if decision.supersedes_review_id:
        old = read_review(directory, decision.supersedes_review_id, archive=archive, context=verification)
        if old.decision.candidate_id != decision.candidate_id:
            raise EvidenceError("Предыдущее экспертное решение относится к другому кандидату.")
    data = json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) > MAX_REVIEW_BYTES:
        raise EvidenceError("Экспертная запись превышает ограничение 25 МБ.")
    verification.check_cancelled()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (record.review_id + ".json")
    temporary = directory / (uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            for offset in range(0, len(data), 1024 * 1024):
                verification.check_cancelled()
                stream.write(data[offset:offset + 1024 * 1024])
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and refuses to replace any prior
        # journal entry. Temporary and target are on the same filesystem.
        verification.check_cancelled()
        try:
            os.link(temporary, target)
        except FileExistsError:
            read_review(directory, record.review_id, archive=archive, context=verification)
        if os.name != "nt":
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except OSError:
        raise EvidenceError("Не удалось надёжно сохранить экспертную запись; результат не опубликован.") from None
    finally:
        temporary.unlink(missing_ok=True)
    return record
