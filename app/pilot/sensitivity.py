"""Offline, immutable counterfactual views of a frozen historical assessment.

No source collection, reclustering or model call occurs here. Exclusions change
the evidence available to an assessment, never its candidate definition. The
reference catalogue is OpenAlex: removing it makes growth unknown, not zero.
Journal files are separate content-addressed artifacts; original snapshots and
document revisions remain untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Final, Literal, Self
from uuid import uuid4

from pydantic import Field, model_validator

from app.identity import validate_data_dir
from app.pilot.antecedents import AntecedentBundle, _derive_matches, verify_antecedents
from app.pilot.archive import DocumentArchive
from app.pilot.contracts import (
    METHODOLOGY_VERSION, MethodologyVersion, Candidate, Contract, CorpusSnapshot, Identifier, QueryPlan, Source, TrendCard,
    content_hash, require_unique, DocumentRevisionRef,
)
from app.pilot.evidence import EvidenceError
from app.pilot.history import CancellationContext, assess_snapshot
from app.pilot.methodology import AssessmentArtifact, NoveltyAssessment
from app.pilot.field_history import FieldExposure, verify_field_exposure
from app.pilot.signal_evidence import PrimaryStudyEvidence, SourceNoveltyEvidence, verify_signal_sources, verify_primary_sources

SENSITIVITY_VERSION: Final = "frozen-reference-exclusions/2.0.0"
LEGACY_SENSITIVITY_VERSION: Final = "frozen-reference-exclusions/1.0.0"
MAX_JOURNAL_BYTES = 50_000_000
CONDITIONAL_VIEW = (
    "Пересчёт выполнен по неизменному определению кандидата и сохранённому корпусу с явными исключениями. "
    "Покрытие и число просмотренных записей относятся к исходному сбору; новый поиск не выполнялся."
)
GROUP_SCOPE = (
    "Исключены только явно проверенные работы крупнейшей группы за три последних полных года. "
    "Принадлежность более ранних работ этой группе не предполагается; совпадение авторов или организаций "
    "не доказывает независимость экспериментов."
)


class SensitivityScenario(Contract):
    excluded_sources: tuple[Source, ...] = Field(default=(), max_length=5)
    excluded_study_ids: tuple[Identifier, ...] = Field(default=(), max_length=3000)
    exclude_largest_verified_group: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def valid_exclusions(self) -> Self:
        require_unique(self.excluded_sources, "excluded sources")
        require_unique(self.excluded_study_ids, "excluded studies")
        if not (self.excluded_sources or self.excluded_study_ids or self.exclude_largest_verified_group):
            raise ValueError("Sensitivity requires at least one explicit exclusion")
        return self


class SensitivityChange(Contract):
    baseline_studies_delta: int = Field(strict=True)
    recent_studies_delta: int = Field(strict=True)
    smoothed_growth_delta: float
    priority_score_delta: float | None = None
    growth_confirmation_changed: bool = Field(strict=True)
    category_changed: bool = Field(strict=True)


class SensitivityReport(Contract):
    method_version: Literal["frozen-reference-exclusions/1.0.0", "frozen-reference-exclusions/2.0.0"] = SENSITIVITY_VERSION
    scenario: SensitivityScenario
    query_plan: QueryPlan
    original_snapshot: CorpusSnapshot
    baseline: AssessmentArtifact
    baseline_card: TrendCard
    antecedents: AntecedentBundle | None = Field(default=None, exclude_if=lambda value: value is None)
    status: Literal["evaluated", "not_comparable", "unavailable"]
    after_snapshot: CorpusSnapshot | None = None
    after: AssessmentArtifact | None = None
    after_card: TrendCard | None = None
    excluded_study_ids: tuple[Identifier, ...] = Field(default=(), max_length=3000)
    selected_group_id: Identifier | None = None
    removed_evidence_ids: tuple[Identifier, ...] = Field(default=(), max_length=200)
    removed_claim_ids: tuple[Identifier, ...] = Field(default=(), max_length=40)
    limitations: tuple[Identifier, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def frozen_and_consistent(self) -> Self:
        for name in ("excluded_study_ids", "removed_evidence_ids", "removed_claim_ids"):
            require_unique(getattr(self, name), name)
        expected_method = LEGACY_SENSITIVITY_VERSION if self.baseline.assessment.methodology_version == "3.0.0" else SENSITIVITY_VERSION
        if self.method_version != expected_method:
            raise ValueError("Sensitivity method does not match its frozen methodology")
        candidate = self.baseline.inputs.candidate
        if (candidate.plan_hash != self.query_plan.plan_hash
                or self.original_snapshot.plan_hash != self.query_plan.plan_hash
                or self.baseline.inputs.history.snapshot_id != self.original_snapshot.snapshot_id
                or self.baseline_card.candidate != candidate
                or self.baseline_card.assessment_hash != self.baseline.assessment.assessment_hash
                or self.baseline_card.historical_snapshot_id != self.original_snapshot.snapshot_id
                or self.baseline_card.claims != self.baseline.inputs.claims
                or tuple(item.evidence_id for item in self.baseline_card.evidence) != self.baseline.inputs.evidence_ids):
            raise ValueError("Baseline references are inconsistent")
        if self.status == "unavailable":
            if any(value is not None for value in (self.after, self.after_card, self.after_snapshot, self.selected_group_id)):
                raise ValueError("Unavailable scenario cannot contain a simulated result")
            if self.excluded_study_ids or self.removed_evidence_ids or self.removed_claim_ids:
                raise ValueError("Unavailable scenario cannot claim applied exclusions")
            return self
        if self.after_card is None or self.after_card.candidate != candidate:
            raise ValueError("Sensitivity cannot change the candidate or discovery membership")
        if not set(self.scenario.excluded_study_ids).issubset(self.excluded_study_ids):
            raise ValueError("An explicit study exclusion was lost")
        if self.selected_group_id is not None:
            independence = self.baseline.inputs.independence
            group = next((item for item in independence.groups if item.group_id == self.selected_group_id), None) if independence else None
            largest = min(independence.groups, key=lambda item: (-len(item.study_ids), item.group_id)) if independence else None
            if (not self.scenario.exclude_largest_verified_group or group is None or group != largest
                    or independence is None or not independence.coverage_complete
                    or not self.baseline.inputs.history.coverage.complete_history
                    or not set(group.study_ids).issubset(self.excluded_study_ids)):
                raise ValueError("Selected group is not supported by the baseline")
        elif self.scenario.exclude_largest_verified_group:
            raise ValueError("Largest-group scenario requires an explicitly verified group")
        before_evidence = {item.evidence_id for item in self.baseline_card.evidence}
        after_evidence = {item.evidence_id for item in self.after_card.evidence}
        before_claims = {item.claim_id: item for item in self.baseline_card.claims}
        after_claims = {item.claim_id: item for item in self.after_card.claims}
        if (set(self.removed_evidence_ids) != before_evidence - after_evidence
                or set(self.removed_claim_ids) != set(before_claims) - set(after_claims)):
            raise ValueError("Removed evidence and claims must match the actual result")
        if any(before_claims.get(key) != value for key, value in after_claims.items()):
            raise ValueError("Sensitivity cannot invent or rewrite a claim")
        if any(item.source in self.scenario.excluded_sources or item.study_id in self.excluded_study_ids
               for item in self.after_card.evidence):
            raise ValueError("Excluded evidence reappeared in the result")
        if self.after is None:
            if (self.status != "not_comparable" or self.after_snapshot is not None
                    or self.after_card.assessment_hash is not None
                    or self.after_card.historical_snapshot_id is not None):
                raise ValueError("Unknown history cannot retain a historical assessment")
        else:
            if self.after_snapshot is None or "openalex" in self.scenario.excluded_sources:
                raise ValueError("A numerical result requires its reference snapshot")
            expected = tuple(item for item in self.original_snapshot.documents if item.study_id not in self.excluded_study_ids)
            if (self.after_snapshot.documents != expected or self.after_snapshot.coverage != self.original_snapshot.coverage
                    or self.after.inputs.candidate != candidate
                    or self.after.inputs.history.snapshot_id != self.after_snapshot.snapshot_id
                    or self.after_card.assessment_hash != self.after.assessment.assessment_hash
                    or self.after_card.historical_snapshot_id != self.after_snapshot.snapshot_id
                    or self.after_card.claims != self.after.inputs.claims
                    or tuple(item.evidence_id for item in self.after_card.evidence) != self.after.inputs.evidence_ids):
                raise ValueError("Conditional history does not match the explicit exclusions")
        if self.status == "evaluated" and (self.after is None or not self.baseline.inputs.history.coverage.complete_history
                                           or not self.after.inputs.history.coverage.complete_history):
            raise ValueError("Comparable sensitivity requires complete reference coverage")
        return self

    @property
    def report_hash(self) -> str:
        return content_hash(self)

    @property
    def change(self) -> SensitivityChange | None:
        """Never subtract unknown measurements or convert missing scores to zero."""
        if self.status != "evaluated" or self.after is None:
            return None
        before, after = self.baseline.assessment, self.after.assessment
        return SensitivityChange(
            baseline_studies_delta=after.baseline_studies - before.baseline_studies,
            recent_studies_delta=after.recent_studies - before.recent_studies,
            smoothed_growth_delta=after.smoothed_growth - before.smoothed_growth,
            priority_score_delta=(after.priority_score - before.priority_score
                if after.priority_score is not None and before.priority_score is not None else None),
            growth_confirmation_changed=after.growth_confirmed != before.growth_confirmed,
            category_changed=after.category != before.category,
        )


def _family_exclusions(excluded: set[str], snapshot: CorpusSnapshot, passport: TrendCard,
                       antecedents: AntecedentBundle | None, archive: DocumentArchive,
                       context: CancellationContext, *,
                       methodology_version: MethodologyVersion = METHODOLOGY_VERSION) -> set[str]:
    """Resolve identities before filtering, including earlier versions of a study."""
    from app.pilot.discovery import deduplicate

    revisions = {item.revision_id: item.study_id for item in snapshot.documents}
    revisions.update((item.revision_id, item.study_id) for item in passport.evidence)
    if antecedents is not None:
        revisions.update((item.revision_id, item.study_id) for item in antecedents.snapshot.documents)
    documents = []
    aliases = []
    for revision, study in sorted(revisions.items()):
        context.check_cancelled()
        document = archive.get(revision)
        documents.append(document)
        aliases.append({study, document.document_key})
    closure = set(excluded)
    for family in deduplicate(documents,
            version="study-families-v2-publication-units" if methodology_version == "3.4.0" else "study-families-v1"):
        context.check_cancelled()
        identities = set().union(*(aliases[index] for index in family["input_indices"]))
        if identities & excluded:
            closure.update(identities)
    return closure


def _conditional_antecedents(bundle: AntecedentBundle, candidate: Candidate, plan: QueryPlan,
                            scenario: SensitivityScenario, excluded: set[str], archive: DocumentArchive,
                            context: CancellationContext) -> AntecedentBundle:
    """A reproducible exclusion view, never a new search or an invention claim."""
    verify_antecedents(bundle, candidate, plan, archive, check=context.check_cancelled)
    references = tuple(item for item in bundle.snapshot.documents
        if item.source not in scenario.excluded_sources and item.study_id not in excluded)
    if references == bundle.snapshot.documents:
        return bundle
    values = bundle.snapshot.model_dump(mode="python") | {
        "snapshot_id": "sensitivity-antecedents-" + content_hash({"original": bundle.bundle_hash,
            "excluded": sorted(excluded), "sources": scenario.excluded_sources, "method": SENSITIVITY_VERSION}),
        "documents": references}
    # The original scan receipt remains immutable. Any counterfactual deletion
    # makes this view incomplete as a search, so it cannot certify absence or
    # restore an earlier-search confirmation gate after known evidence removal.
    values["coverage"] = tuple(type(item).model_validate(item.model_dump(mode="python") | {
        "state": "partial", "completed_years": (), "comparable": False,
        "reasons": tuple(dict.fromkeys((*item.reasons, "counterfactual_antecedent_exclusion")))})
        for item in bundle.snapshot.coverage)
    snapshot = CorpusSnapshot.model_validate(values)
    matched, conflicts = _derive_matches(candidate, snapshot, archive, bundle.first_year,
        plan.completed_years[0] - 1, context.check_cancelled)
    first = min(((document.publication_year, study) for study, (_, document) in matched.items()), default=None)
    evidence = tuple(item for item in bundle.evidence if item.source not in scenario.excluded_sources
                     and item.study_id in matched)
    return AntecedentBundle.model_validate(bundle.model_dump(mode="python") | {
        "snapshot": snapshot, "matched_study_ids": tuple(sorted(matched)), "conflicting_study_ids": conflicts,
        "earliest_observed_year": first[0] if first else None,
        "earliest_observed_study_id": first[1] if first else None,
        "operational_status": "earlier_matches_found" if matched else "incomplete_search", "evidence": evidence,
        "limitations": tuple(dict.fromkeys((*bundle.limitations,
            "Условный срез после исключений; исходный поиск не повторялся, отсутствие ранних работ не удостоверено.")))})


def evaluate_sensitivity(candidate: Candidate, plan: QueryPlan, historical_snapshot: CorpusSnapshot,
                         archive: DocumentArchive, context: CancellationContext, *, passport: TrendCard,
                         scenario: SensitivityScenario,
                         verified_novelty: NoveltyAssessment | None = None,
                         methodology_version: MethodologyVersion = METHODOLOGY_VERSION,
                         antecedents: AntecedentBundle | None = None,
                         field_exposure: FieldExposure | None = None,
                         source_novelty: tuple[SourceNoveltyEvidence, ...] = (),
                         primary_observations: tuple[PrimaryStudyEvidence, ...] = (),
                         publication_status_revisions: tuple[DocumentRevisionRef, ...] = ()) -> SensitivityReport:
    """Recompute a bounded exclusion scenario using existing archive bytes only.

    Pass the separately reviewed novelty object from the original artifact when
    present. Evidence removed by an exclusion revokes its dependent claims and
    novelty review; surviving aliases never remove a reference work implicitly.
    """
    context.check_cancelled()
    legacy = methodology_version == "3.0.0"
    if methodology_version in {"3.2.0", "3.3.0", "3.4.0"}:
        from app.pilot.evidence_versions import novelty_method, primary_method

        verify_signal_sources(source_novelty, candidate, archive, context, query_plan=plan,
                              method_version=novelty_method(methodology_version) if methodology_version == "3.4.0" else None)
        verify_primary_sources(primary_observations, candidate, archive, context, query_plan=plan,
                               method_version=primary_method(methodology_version) if methodology_version == "3.4.0" else None)
        if field_exposure is not None:
            verify_field_exposure(field_exposure, plan)
    method_version: Literal["frozen-reference-exclusions/1.0.0", "frozen-reference-exclusions/2.0.0"] = (
        LEGACY_SENSITIVITY_VERSION if legacy else SENSITIVITY_VERSION)
    if passport.category in ("confirmed_trend", "renewed_interest", "established_topic") and verified_novelty is None:
        raise EvidenceError("Для воспроизведения исходной категории нужна сохранённая оценка новизны.")
    baseline, baseline_card, _ = assess_snapshot(candidate, plan, historical_snapshot, archive, context,
        passport=passport, verified_novelty=verified_novelty, methodology_version=methodology_version,
        antecedents=antecedents, field_exposure=field_exposure, source_novelty=source_novelty,
        primary_observations=primary_observations, publication_status_revisions=publication_status_revisions)
    known_studies = {item.study_id for item in historical_snapshot.documents} | {item.study_id for item in baseline_card.evidence}
    if antecedents is not None and not legacy:
        known_studies.update(item.study_id for item in antecedents.snapshot.documents)
    if not set(scenario.excluded_study_ids).issubset(known_studies):
        raise EvidenceError("Исключаемое исследование отсутствует в сохранённой истории и доказательствах кандидата.")
    excluded = set(scenario.excluded_study_ids)
    limitations = [CONDITIONAL_VIEW]
    selected_group_id = None
    if scenario.exclude_largest_verified_group:
        independence = baseline.inputs.independence
        if independence is None or not independence.coverage_complete or not baseline.inputs.history.coverage.complete_history:
            context.check_cancelled()
            return SensitivityReport(method_version=method_version, scenario=scenario, query_plan=plan, original_snapshot=historical_snapshot,
                baseline=baseline, baseline_card=baseline_card, antecedents=antecedents,
                status="unavailable", limitations=(CONDITIONAL_VIEW,
                "Крупнейшая группа не определена: отсутствует полная история или проверенные ID авторов и организаций всех недавних работ."))
        selected = min(independence.groups, key=lambda item: (-len(item.study_ids), item.group_id))
        selected_group_id = selected.group_id
        excluded.update(selected.study_ids)
        limitations.append(GROUP_SCOPE)
    if excluded and not legacy:
        excluded = _family_exclusions(excluded, historical_snapshot, baseline_card, antecedents, archive, context,
                                      methodology_version=methodology_version)
    conditional_antecedents = antecedents
    if antecedents is not None and not legacy:
        conditional_antecedents = _conditional_antecedents(antecedents, candidate, plan, scenario, excluded, archive, context)
        if conditional_antecedents != antecedents:
            limitations.append("Изменён корпус ранних аналогов; исходная экспертная оценка новизны не переносится на условный срез.")
    if len(excluded) > 3000:
        raise EvidenceError("Сценарий превышает лимит 3000 исключаемых исследований.")
    evidence = []
    for item in baseline_card.evidence:
        context.check_cancelled()
        if item.source not in scenario.excluded_sources and item.study_id not in excluded:
            evidence.append(item)
    remaining_ids = {item.evidence_id for item in evidence}
    conditional_source_novelty = tuple(item for item in source_novelty
        if item.novelty.study_id not in excluded and item.experiment.study_id not in excluded
        and item.novelty.source not in scenario.excluded_sources and item.experiment.source not in scenario.excluded_sources
        and {item.novelty.evidence_id, item.experiment.evidence_id}.issubset(remaining_ids))
    if conditional_source_novelty != source_novelty:
        limitations.append("Архивное заявление о новом механизме или его эксперимент исключены; зависимая автоматическая гипотеза отозвана.")
    conditional_primary = tuple(item for item in primary_observations
        if item.result.study_id not in excluded and item.result.source not in scenario.excluded_sources
        and item.result.evidence_id in remaining_ids)
    if conditional_primary != primary_observations:
        limitations.append("Первичный результат исключён; зависимая гипотеза пересчитана без него.")
    claims = tuple(item for item in baseline_card.claims
                   if set(item.evidence_ids).issubset(remaining_ids) and item.kind != "numeric")
    if any(item.kind == "numeric" for item in baseline_card.claims):
        limitations.append("Производные числовые утверждения удалены; актуальные показатели находятся в отдельной пересчитанной оценке.")
    if (verified_novelty is not None and not legacy and conditional_antecedents != antecedents):
        claims = tuple(item for item in claims if item.claim_id != verified_novelty.claim_id)
    if verified_novelty is not None and verified_novelty.claim_id not in {item.claim_id for item in claims}:
        verified_novelty = None
        limitations.append("Доказательство отдельной оценки новизны исключено; прежнее подтверждение новизны отозвано.")
    filtered = TrendCard(candidate=candidate,
        methodology_version=None if methodology_version == "3.0.0" else methodology_version,
        category="early_signal" if methodology_version == "3.0.0" else "insufficient_evidence",
        quality="partial", claims=claims,
        evidence=tuple(evidence), limitations=tuple(dict.fromkeys((*baseline_card.limitations, *limitations))))
    after_snapshot = None
    after = None
    status: Literal["evaluated", "not_comparable", "unavailable"] = "not_comparable"
    if "openalex" in scenario.excluded_sources:
        limitations.append("OpenAlex — опорный исторический каталог — исключён. Динамика публикаций и изменение оценки неизвестны; нулевые значения не подставляются.")
        after_card = TrendCard.model_validate(filtered.model_dump(mode="python") | {
            "limitations": tuple(dict.fromkeys((*filtered.limitations, *limitations)))})
    else:
        # Query accounting remains the immutable collection receipt. This is a
        # conditional subset, not a claim to have run a new exhaustive query.
        references = tuple(item for item in historical_snapshot.documents if item.study_id not in excluded)
        after_snapshot = CorpusSnapshot.model_validate(historical_snapshot.model_dump(mode="python") | {
            "snapshot_id": "sensitivity-" + content_hash({"original": historical_snapshot.snapshot_hash,
                "scenario": scenario.model_dump(mode="json"), "excluded": sorted(excluded), "method": method_version}),
            "documents": references,
        })
        after, after_card, _ = assess_snapshot(candidate, plan, after_snapshot, archive, context,
            passport=filtered, verified_novelty=verified_novelty, methodology_version=methodology_version,
            antecedents=(antecedents if verified_novelty is not None else None) if legacy else conditional_antecedents,
            field_exposure=field_exposure, source_novelty=conditional_source_novelty, primary_observations=conditional_primary,
            publication_status_revisions=baseline.inputs.publication_status_revisions)
        if baseline.inputs.history.coverage.complete_history and after.inputs.history.coverage.complete_history:
            status = "evaluated"
        else:
            limitations.append("Исходная или пересчитанная reference-история неполна; изменение показателей не считается сопоставимым.")
    context.check_cancelled()
    after_evidence = {item.evidence_id for item in after_card.evidence}
    after_claims = {item.claim_id for item in after_card.claims}
    return SensitivityReport(method_version=method_version, scenario=scenario, query_plan=plan, original_snapshot=historical_snapshot,
        baseline=baseline, baseline_card=baseline_card, antecedents=antecedents,
        status=status, after_snapshot=after_snapshot, after=after, after_card=after_card,
        excluded_study_ids=tuple(sorted(excluded)), selected_group_id=selected_group_id,
        removed_evidence_ids=tuple(sorted(item.evidence_id for item in baseline_card.evidence if item.evidence_id not in after_evidence)),
        removed_claim_ids=tuple(sorted(item.claim_id for item in baseline_card.claims if item.claim_id not in after_claims)),
        limitations=tuple(dict.fromkeys(limitations)))


def verify_sensitivity(report: SensitivityReport, archive: DocumentArchive,
                       context: CancellationContext) -> None:
    """Replay archive admission and both assessments, not just a JSON checksum."""
    report = SensitivityReport.model_validate(report.model_dump(mode="python"))
    replayed = evaluate_sensitivity(report.baseline.inputs.candidate, report.query_plan,
        report.original_snapshot, archive, context, passport=report.baseline_card,
        scenario=report.scenario, verified_novelty=report.baseline.inputs.novelty,
        methodology_version=report.baseline.assessment.methodology_version, antecedents=report.antecedents,
        field_exposure=report.baseline.inputs.field_exposure, source_novelty=report.baseline.inputs.source_novelty,
        primary_observations=report.baseline.inputs.primary_observations,
        publication_status_revisions=report.baseline.inputs.publication_status_revisions)
    if replayed != report:
        raise EvidenceError("Проверка устойчивости не воспроизводится по архиву и исходному определению.")


def _journal_bytes(report: SensitivityReport) -> bytes:
    # Revalidate even a caller's unchecked model_copy/model_construct before IO.
    report = SensitivityReport.model_validate(report.model_dump(mode="python"))
    payload = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(payload) > MAX_JOURNAL_BYTES:
        raise EvidenceError("Отчёт устойчивости превышает размер отдельного журнала.")
    return payload


def save_sensitivity(report: SensitivityReport, journal_directory: Path, *,
                     context: CancellationContext | None = None) -> Path:
    """Atomically append an immutable report in a dedicated journal directory.

    The returned filename includes its complete SHA-256. Repeating the same
    evaluation is idempotent; an existing different/corrupt file is not replaced.
    """
    if context is not None:
        context.check_cancelled()
    payload = _journal_bytes(report)
    directory = validate_data_dir(journal_directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = directory / ("sensitivity-" + hashlib.sha256(payload).hexdigest() + ".json")
    temporary = directory / ("." + uuid4().hex + ".tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if context is not None:
            context.check_cancelled()
        try:
            # An atomic, exclusive same-filesystem link prevents a concurrent
            # writer or a corrupt existing entry from being overwritten.
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or load_sensitivity(target) != report:
                raise EvidenceError("Запись журнала устойчивости уже существует с другим содержимым.") from None
        if os.name != "nt":
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_sensitivity(path: Path) -> SensitivityReport:
    """Bounded read with exact-byte hash and structural consistency checks."""
    match = re.fullmatch(r"sensitivity-([a-f0-9]{64})\.json", path.name)
    if match is None or path.is_symlink():
        raise EvidenceError("Некорректное имя записи журнала устойчивости.")
    try:
        with path.open("rb") as handle:
            payload = handle.read(MAX_JOURNAL_BYTES + 1)
        if len(payload) > MAX_JOURNAL_BYTES or hashlib.sha256(payload).hexdigest() != match.group(1):
            raise ValueError("Invalid journal hash")
        report = SensitivityReport.model_validate_json(payload)
        if report.report_hash != match.group(1):
            raise ValueError("Noncanonical journal content")
        return report
    except (OSError, ValueError):
        raise EvidenceError("Запись журнала устойчивости отсутствует, повреждена или не соответствует контракту.") from None
