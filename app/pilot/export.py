"""Portable v3 results with exact document/quotation and numerical provenance.

No model, API key, SQLite spending ledger or network request is needed to open a
package. Verification proves internal integrity and repeatable arithmetic; a
package cannot authenticate its own author or establish semantic/scientific truth.
Reports/full text are not implicitly redistributable and are refused by this
metadata-only format until a separately versioned rights policy supports them.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import time
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from app.backend.contracts import DocumentRecord
from app.pilot.archive import ArchiveReading, DocumentArchive, document_text
from app.pilot.contracts import AnalysisResult, Claim, content_hash, verify_evidence_text
from app.pilot.grounding import verify_supported_source_claim as _verify_supported_source_claim
from app.pilot.evidence_versions import novelty_method, primary_method
from app.pilot.history import CancellationContext
from app.pilot.methodology import AssessmentArtifact
from app.pilot.review import ReviewRecord, verify_review_record
from app.runtime.backup import (
    ArchiveCancellation, ArchiveError, BackupResult, PackageManifest, assert_no_credentials, strict_json,
    unpack_package, write_package,
)
from app.runtime.jobs import TaskCancelled

_METRICS = frozenset({
    "baseline_years", "recent_years", "baseline_studies", "recent_studies", "active_recent_years",
    "first_observed_year", "smoothed_growth", "raw_growth", "recent_theil_sen_slope",
    "priority_score", "priority_lower_bound", "priority_upper_bound",
})
_FULL_TEXT_KEYS = frozenset({"fulltext", "fulltextcontent", "pdfbytes", "originaltext", "rawtext"})
_NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?(?!\w)")
_REVIEWED_METHODS = ("reviewed-novelty/", "reviewed-evidence/", "reviewed-application/")


class _VerificationContext:
    """One deadline/cancellation fence spans ZIP IO and evidence replay."""

    def __init__(self, cancel: ArchiveCancellation | None, parent: CancellationContext | None) -> None:
        self.cancel = cancel
        self.parent = parent
        # Verification rereads and rehashes every archived revision, and a
        # full corpus now holds ten thousand of them rather than fifteen hundred.
        self.deadline = time.monotonic() + 600

    def check_cancelled(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise TaskCancelled("Проверка пакета отменена.")
        if self.parent is not None:
            self.parent.check_cancelled()
        if time.monotonic() > self.deadline:
            raise ArchiveError("Проверка доказательств превысила допустимое время.")

    def is_set(self) -> bool:
        # ZIP helpers check an Event-like interface. Raising preserves the reason
        # (user cancellation or the shared deadline) through their cleanup.
        self.check_cancelled()
        return False

    def progress(self, stage: str, message: str, completed: int = 0, total: int = 0) -> None:
        """Forward a run's own progress signature, so nested contexts chain.

        Verification also runs from the library, an export and a backup, where
        no run is watching; there the report simply has nowhere to go.
        """
        report = getattr(self.parent, "progress", None)
        if report is not None:
            report(stage, message, completed, total)


def _verify_numeric_text(text: str, values: list[Any]) -> None:
    """Explicit numeric claims must display their referenced values, with rounding.

    Derived units/percentages need their own versioned metric; silently inventing
    a conversion would defeat provenance. Numbers in ordinary quoted evidence
    remain source quotations and are not treated as computed trend metrics.
    """
    allowed: list[Decimal] = []
    for value in values:
        for item in value if isinstance(value, tuple) else (value,):
            if type(item) in (int, float):
                allowed.append(Decimal(str(item)))
    tokens = _NUMBER.findall(text)
    if not tokens or not allowed:
        raise ValueError("Numeric claim has no displayed reproducible value")
    for token in tokens:
        observed = Decimal(token.replace(",", "."))
        exponent = observed.as_tuple().exponent
        if not isinstance(exponent, int) or abs(exponent) > 32:
            raise ValueError("Unbounded numeric precision")
        quantum = Decimal(1).scaleb(exponent)
        if not any(value.quantize(quantum, rounding=ROUND_HALF_UP) == observed for value in allowed):
            raise ValueError("Displayed number differs from referenced metrics")


def _public_document(document: DocumentRecord) -> None:
    # Crossref/OpenAlex "report" is a bibliographic genre (e.g. a DOE final
    # report), not evidence that this record contains the licensed report text.
    if (document.source not in {"openalex", "crossref", "epo", "arxiv"}
            or document.document_type == "full_text" or getattr(document, "full_text", None)):
        raise ArchiveError("Этот пакет поддерживает публичные библиографические записи. Отчёты и полный текст требуют отдельного разрешения.")
    payload = document.model_dump(mode="json")
    assert_no_credentials(payload)
    stack: list[Any] = [document.raw_metadata]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, nested in value.items():
                normalized = str(key).lower().replace("_", "").replace("-", "")
                positions = isinstance(nested, list) and all(type(position) is int for position in nested)
                if normalized in _FULL_TEXT_KEYS and not positions:
                    raise ArchiveError("Полный текст не включается в пакет метаданных автоматически.")
                stack.append(nested)
        elif isinstance(value, list):
            stack.extend(value)


def _validated_reviews(reviews: Sequence[ReviewRecord]) -> tuple[ReviewRecord, ...]:
    if len(reviews) > 60:
        raise ArchiveError("Пакет содержит слишком много экспертных решений.")
    records = tuple(ReviewRecord.model_validate_json(item.model_dump_json()) for item in reviews)
    if len(json.dumps([item.model_dump(mode="json") for item in records], ensure_ascii=False).encode("utf-8")) > 25 * 1024 * 1024:
        raise ArchiveError("Экспертные решения превышают ограничение 25 МБ.")
    return records


def _package_revision_ids(result: AnalysisResult, reviews: Sequence[ReviewRecord]) -> set[str]:
    revisions = {reference.revision_id for snapshot in result.snapshots for reference in snapshot.documents}
    for review in reviews:
        for snapshot in (review.bundle.snapshot, review.historical_snapshot):
            revisions.update(reference.revision_id for reference in snapshot.documents)
        for card in (review.source_card, review.reviewed_card):
            revisions.update(evidence.revision_id for evidence in card.evidence)
    return revisions


def verify_result(
    result: AnalysisResult, archive: DocumentArchive, assessments: Sequence[AssessmentArtifact] = (), *,
    reviews: Sequence[ReviewRecord] = (),
    cancel: ArchiveCancellation | None = None, context: CancellationContext | None = None,
) -> tuple[AnalysisResult, tuple[AssessmentArtifact, ...]]:
    """Revalidate frozen models too: model_copy must not bypass import boundaries."""
    try:
        replay_context = _VerificationContext(cancel, context)
        # One verification pass used to ask the archive for the same revisions
        # several times over: directly, again through publication status and
        # again through each per-card verifier. This view reads and verifies
        # every revision once and is dropped together with this call.
        reading = ArchiveReading(archive)
        replay_context.check_cancelled()
        result = AnalysisResult.model_validate_json(result.model_dump_json())
        version_order = {version: index for index, version in enumerate(("3.0.0", "3.1.0", "3.2.0", "3.3.0", "3.4.0"))}
        if any(version_order[card.methodology_version or "3.0.0"] > version_order[result.methodology_version]
               for card in result.cards):
            raise ArchiveError("Методика карточки новее версии контейнера результата; требуется совместимый формат.")
        if result.methodology_version not in {"3.2.0", "3.3.0", "3.4.0"} and any(card.methodology_version in {"3.2.0", "3.3.0", "3.4.0"} for card in result.cards):
            raise ValueError("Signal assessment cannot be exported under an older result selection policy")
        if len(assessments) > 60:
            raise ValueError("Too many assessments")
        validated_artifacts: list[AssessmentArtifact] = []
        for item in assessments:
            replay_context.check_cancelled()
            validated_artifacts.append(AssessmentArtifact.model_validate_json(item.model_dump_json()))
        artifacts = tuple(validated_artifacts)
        replay_context.check_cancelled()
        review_records = _validated_reviews(reviews)
        replay_context.check_cancelled()
        reviewed_candidates = {item.decision.candidate_id: item for item in review_records}
        if len(reviewed_candidates) != len(review_records):
            raise ValueError("Duplicate expert decisions for one result candidate")
        if any(item.bundle.query_plan != result.query_plan for item in review_records):
            raise ValueError("Expert decision belongs to a different query plan")
        if len(artifacts) > 60:
            raise ValueError("Too many assessments")
        by_hash = {item.assessment.assessment_hash: item for item in artifacts}
        if len(by_hash) != len(artifacts):
            raise ValueError("Duplicate assessments")
        snapshots = {item.snapshot_id: item for item in result.snapshots}
        documents: dict[str, DocumentRecord] = {}
        identities: dict[tuple[str, str], str] = {}
        for snapshot in result.snapshots:
            replay_context.check_cancelled()
            if snapshot.as_of != result.query_plan.as_of:
                raise ValueError("Snapshot date differs from plan")
            for reference in snapshot.documents:
                replay_context.check_cancelled()
                document = documents.get(reference.revision_id)
                if document is None:
                    document = reading.get(reference.revision_id)
                    _public_document(document)
                    documents[reference.revision_id] = document
                if (reference.source != document.source or reference.source_id != document.source_id
                        or reference.observed_at != document.fetched_at
                        or reference.publication_year != document.publication_year
                        or reference.publicly_available_at != document.publication_date
                        or reference.text_hash != hashlib.sha256(document_text(document).encode("utf-8")).hexdigest()):
                    raise ValueError("Document reference differs from its exact revision")
                for identity in ((document.source, document.source_id), ("document-key", document.document_key)):
                    if identity in identities and identities[identity] != reference.study_id:
                        raise ValueError("One document is being counted as multiple studies")
                    identities[identity] = reference.study_id
        used_assessments: set[str] = set()
        used_reviews: set[str] = set()
        for revision_id in _package_revision_ids(result, review_records) - documents.keys():
            replay_context.check_cancelled()
            document = reading.get(revision_id)
            _public_document(document)
            documents[revision_id] = document
        status_context = None
        if any(card.methodology_version == "3.4.0" for card in result.cards):
            from app.pilot.publication_status import collect_status_context

            status_context = collect_status_context(tuple(reference for snapshot in result.snapshots
                for reference in snapshot.documents), reading, replay_context, as_of=result.query_plan.as_of)
        snapshot_revision_ids = {reference.revision_id for snapshot in result.snapshots for reference in snapshot.documents}
        for position, card in enumerate(result.cards):
            replay_context.check_cancelled()
            replay_context.progress("verify", f"Проверяем доказательства карточек: {position + 1} из {len(result.cards)}",
                                    position, len(result.cards))
            review = reviewed_candidates.get(card.candidate.candidate_id)
            reviewed_claims = tuple(claim for claim in card.claims
                if (claim.grounding_method or "").startswith(_REVIEWED_METHODS))
            if reviewed_claims:
                if review is None:
                    raise ArchiveError("Экспертная оценка требует полного ReviewRecord с обзором аналогов и источниками.")
                if review.reviewed_card != card:
                    raise ValueError("Reviewed card differs from attributed review record")
                verify_review_record(review, reading, context=replay_context)
                used_reviews.add(review.decision.candidate_id)
            elif review is not None:
                raise ValueError("Expert record has no corresponding reviewed claim")
            evidence_by_id = {item.evidence_id: item for item in card.evidence}
            if card.methodology_version == "3.4.0":
                assert status_context is not None
                withdrawn = status_context.withdrawn
                if any(claim.support == "supported"
                    and any(evidence_by_id[identifier].study_id in withdrawn for identifier in claim.evidence_ids)
                    for claim in card.claims):
                    raise ArchiveError("Подтверждённая цитата относится к отозванной работе по сохранённым сведениям источников.")
                supporting = status_context.supporting
                if any(claim.support == "supported" and claim.role in {"case", "advantage", "application"}
                    and any(evidence_by_id[identifier].study_id in supporting for identifier in claim.evidence_ids)
                    for claim in card.claims):
                    raise ArchiveError("Приложение или набор данных не подтверждает собственный результат независимого исследования.")
            for evidence in card.evidence:
                replay_context.check_cancelled()
                document = documents[evidence.revision_id]
                if evidence.text_field == "title":
                    text = document.title
                elif evidence.text_field == "abstract":
                    text = document.abstract or ""
                elif evidence.text_field == "metadata":
                    text = json.dumps(document.raw_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                else:
                    raise ValueError("This archive has no full text")
                verify_evidence_text(evidence, text)
                if evidence.source != document.source or evidence.study_id != document.document_key:
                    raise ValueError("Evidence source or research identity differs from its archived document")
                urls = {document.url}
                if document.doi:
                    urls.add("https://doi.org/" + document.doi)
                if evidence.source_url not in urls or evidence.retrieved_at != document.fetched_at:
                    raise ValueError("Evidence link or observation time differs from the archived source")
            for claim in card.claims:
                replay_context.check_cancelled()
                if (claim.grounding_method in {"exact-contextual-quotation/4.0.0", "verified-application/research/2.0.0"}
                        and card.methodology_version != "3.4.0"):
                    raise ValueError("Новая проверка цитат требует методики 3.4.")
                _verify_supported_source_claim(claim, evidence_by_id, documents,
                    legacy=card.methodology_version in {None, "3.0.0"},
                    candidate_studies=set(card.candidate.discovery_study_ids),
                    methodology_version=card.methodology_version)
            if any(claim.support == "supported" and claim.role in {"advantage", "application"}
                   and claim.grounding_method in {"exact-contextual-quotation/4.0.0", "verified-application/research/2.0.0"}
                   for claim in card.claims):
                from app.pilot.evidence import candidate_claims_grounded, member_documents

                members = member_documents(card.candidate, snapshots[card.candidate.discovery_snapshot_id], reading, replay_context)
                if not candidate_claims_grounded(card.candidate, members):
                    raise ValueError("Преимущество отдельной работы нельзя приписать неподтверждённой группе технологий.")
            if card.assessment_hash is None:
                if any((claim.grounding_method or "").startswith("archived-author-novelty/") for claim in card.claims):
                    raise ValueError("Automatic novelty assertions require their reproducible assessment input")
                allowed = ({"early_signal"} if card.methodology_version in {None, "3.0.0"} else
                           {"unassessed_cluster", "insufficient_evidence", "off_scope"})
                if card.historical_snapshot_id is not None or card.category not in allowed or any(claim.kind == "numeric" for claim in card.claims):
                    raise ValueError("Historical or numerical card has no reproducible assessment")
                continue
            artifact = by_hash.get(card.assessment_hash)
            if artifact is None:
                raise ValueError("Missing assessment artifact")
            used_assessments.add(card.assessment_hash)
            inputs, assessment = artifact.inputs, artifact.assessment
            if assessment.methodology_version == "3.4.0" and status_context is not None:
                blocked = status_context.withdrawn | status_context.supporting
                counted = {study for year in inputs.history.observations for study in year.study_ids}
                if counted.intersection(blocked) or inputs.history.first_observed_study_id in blocked:
                    raise ArchiveError("Исторический показатель включает работу, отозванную или помеченную дополнительным материалом в другом снимке результата.")
            if (card.methodology_version or "3.0.0") != assessment.methodology_version:
                raise ValueError("Card methodology version differs from its reproducible assessment")
            antecedents = review.bundle if review else None
            if assessment.methodology_version in {"3.2.0", "3.3.0", "3.4.0"}:
                from app.pilot.antecedents import verify_antecedents
                from app.pilot.field_history import verify_field_exposure
                from app.pilot.signal_evidence import verify_signal_evidence

                antecedents = inputs.antecedents
                if review is not None and antecedents != review.bundle:
                    raise ValueError("Reviewed novelty and automatic assessment use different antecedent searches")
                if antecedents is not None:
                    if snapshots.get(antecedents.snapshot.snapshot_id) != antecedents.snapshot:
                        raise ValueError("Automatic antecedent archive is absent from the result snapshots")
                    verify_antecedents(antecedents, card.candidate, result.query_plan, reading,
                                       check=replay_context.check_cancelled)
                if inputs.field_exposure is not None:
                    verify_field_exposure(inputs.field_exposure, result.query_plan)
                discovery = snapshots[card.candidate.discovery_snapshot_id]
                verify_signal_evidence(inputs.source_novelty, card.candidate, discovery, reading,
                                        replay_context, query_plan=result.query_plan,
                                        method_version=novelty_method(assessment.methodology_version),
                                        excluded_study_ids=(status_context.withdrawn | status_context.supporting
                                            if assessment.methodology_version == "3.4.0" and status_context is not None else frozenset()))
                if assessment.methodology_version in {"3.3.0", "3.4.0"}:
                    from app.pilot.signal_evidence import verify_primary_observations
                    verify_primary_observations(inputs.primary_observations, card.candidate, discovery, reading, replay_context,
                                                query_plan=result.query_plan,
                                                method_version=primary_method(assessment.methodology_version),
                                                excluded_study_ids=(status_context.withdrawn | status_context.supporting
                                                    if assessment.methodology_version == "3.4.0" and status_context is not None else frozenset()))
                    for primary in inputs.primary_observations:
                        if evidence_by_id.get(primary.result.evidence_id) != primary.result:
                            raise ValueError("Primary observation is absent from the passport")
                for source_item in inputs.source_novelty:
                    if any(evidence_by_id.get(source.evidence_id) != source
                           for source in (source_item.novelty, source_item.experiment)):
                        raise ValueError("Automatic source assertion is absent or changed in the displayed passport")
                expected_claims = tuple(Claim(claim_id="source-novelty-" + content_hash(item)[:24],
                    role="novelty", support="unverified", grounding_method=item.method_version,
                    text=item.novelty.quote,
                    evidence_ids=tuple(dict.fromkeys((item.novelty.evidence_id, item.experiment.evidence_id))))
                    for item in inputs.source_novelty)
                automatic_claims = tuple(claim for claim in card.claims
                    if (claim.grounding_method or "").startswith("archived-author-novelty/"))
                if ((review is None and automatic_claims != expected_claims)
                        or (review is not None and automatic_claims)):
                    raise ValueError("Automatic novelty statements differ from their archived, unverified assertions")
            if inputs.novelty is not None:
                if review is None or not reviewed_claims:
                    raise ArchiveError("Оценка новизны не содержит воспроизводимого экспертного решения.")
                if review.artifact != artifact or review.historical_snapshot != snapshots[assessment.historical_snapshot_id]:
                    raise ValueError("Expert decision uses a different assessment or history snapshot")
            elif review is not None:
                raise ValueError("Unused expert novelty assessment")
            if (inputs.candidate != card.candidate or inputs.claims != card.claims
                    or set(inputs.evidence_ids) != {item.evidence_id for item in card.evidence}
                    or assessment.category != card.category or assessment.quality != card.quality
                    or assessment.historical_snapshot_id != card.historical_snapshot_id
                    or inputs.history.as_of != result.query_plan.as_of
                    or tuple(item.year for item in inputs.history.observations) != result.query_plan.completed_years):
                raise ValueError("Card differs from its assessed inputs")
            snapshot = snapshots[assessment.historical_snapshot_id]
            from app.pilot.evidence import TITLE_ADMISSION_VERSION
            from app.pilot.history import verify_history_artifact

            if card.candidate.admission_rule_version not in {TITLE_ADMISSION_VERSION, "title-phrase-admission/1.0.0"}:
                raise ArchiveError("Версия исторического отбора этого пакета не поддерживается. Откройте его в совместимой версии приложения.")
            if any(reference.revision_id not in snapshot_revision_ids for reference in inputs.publication_status_revisions):
                raise ValueError("Publication status proof is absent from result snapshots")
            verify_history_artifact(artifact, result.query_plan, snapshot, reading, card, replay_context,
                                    antecedents=antecedents,
                                    methodology_version=assessment.methodology_version)
            history_members = {(item.study_id, item.publication_year) for item in snapshot.documents if item.source == "openalex"}
            if assessment.methodology_version == "3.0.0":
                for observation in inputs.history.observations:
                    if any((study, observation.year) not in history_members for study in observation.study_ids):
                        raise ValueError("Historical study or year absent from source snapshot")
                if inputs.history.first_observed_study_id is not None:
                    if (inputs.history.first_observed_study_id, inputs.history.first_observed_year) not in history_members:
                        raise ValueError("First observation has no historical source")
            else:
                for observation in inputs.history.observations:
                    if any((study, observation.year) not in history_members for study in observation.study_ids):
                        raise ValueError("Historical family representative is absent from source snapshot")
                earlier_members = ({(item.study_id, item.publication_year) for item in antecedents.snapshot.documents}
                                   if antecedents is not None else set())
                if (inputs.history.first_observed_study_id is not None and
                    (inputs.history.first_observed_study_id, inputs.history.first_observed_year)
                        not in history_members | earlier_members):
                    raise ValueError("First observation has no historical or reviewed antecedent source")
            for claim in card.claims:
                metric_values: list[Any] = []
                for metric_reference in claim.metric_refs:
                    metric = metric_reference
                    if ":" in metric_reference:
                        digest, metric = metric_reference.split(":", 1)
                        if digest != card.assessment_hash:
                            raise ValueError("Metric references another assessment")
                    if metric not in _METRICS or getattr(assessment, metric) is None:
                        raise ValueError("Metric is absent, unknown or not a numerical assessment field")
                    metric_values.append(getattr(assessment, metric))
                if claim.kind == "numeric":
                    _verify_numeric_text(claim.text, metric_values)
        replay_context.progress("verify", "Доказательства карточек проверены",
                                len(result.cards), len(result.cards))
        if used_assessments != set(by_hash):
            raise ValueError("Unused assessments in result package")
        if used_reviews != set(reviewed_candidates):
            raise ValueError("Unused expert review records in result package")
        if result.methodology_version in {"3.2.0", "3.3.0", "3.4.0"}:
            from app.pilot.selection import select_top

            if result.top_limit is None or result.top_trend_ids != select_top(result.cards, artifacts, limit=result.top_limit):
                raise ValueError("Displayed TOP does not reproduce the verified ranking and selection limit")
        replay_context.check_cancelled()
        return result, artifacts
    except (ArchiveError, TaskCancelled):
        raise
    except Exception:
        raise ArchiveError("Результат не прошёл проверку: отсутствуют документы, точные цитаты или воспроизводимые расчёты.") from None


def export_result(
    path: Path, result: AnalysisResult, archive: DocumentArchive,
    assessments: Sequence[AssessmentArtifact] = (), *, reviews: Sequence[ReviewRecord] = (),
    cancel: ArchiveCancellation | None = None, context: CancellationContext | None = None,
) -> BackupResult:
    try:
        replay_context = _VerificationContext(cancel, context)
        replay_context.check_cancelled()
        review_records = _validated_reviews(reviews)
        result, artifacts = verify_result(result, archive, assessments, reviews=review_records, context=replay_context)
        with tempfile.TemporaryDirectory(prefix="trendanalyser-result-") as temporary:
            directory = Path(temporary)
            result_path = directory / "result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")
            assessments_path = directory / "assessments.json"
            assessments_path.write_text(json.dumps([item.model_dump(mode="json") for item in artifacts], ensure_ascii=False, allow_nan=False), encoding="utf-8")
            files = {"result.json": result_path, "assessments.json": assessments_path}
            if review_records:
                reviews_path = directory / "reviews.json"
                reviews_path.write_text(json.dumps([item.model_dump(mode="json") for item in review_records],
                    ensure_ascii=False, allow_nan=False), encoding="utf-8")
                files["reviews.json"] = reviews_path
            for digest in _package_revision_ids(result, review_records):
                replay_context.check_cancelled()
                files[f"revisions/{digest[:2]}/{digest}.json"] = archive.path(digest)
            return write_package(Path(path), files, kind="trendanalizer-result", cancel=replay_context)
    except (ArchiveError, TaskCancelled):
        raise
    except Exception:
        raise ArchiveError("Не удалось сохранить пакет результата.") from None


@dataclass
class ResultPackage:
    result: AnalysisResult
    assessments: tuple[AssessmentArtifact, ...]
    archive: DocumentArchive
    _context: AbstractContextManager[tuple[Path, PackageManifest]] = field(repr=False)
    reviews: tuple[ReviewRecord, ...] = ()
    _closed: bool = field(default=False, repr=False)

    @property
    def revision_ids(self) -> tuple[str, ...]:
        """Every verified revision, including review-only source material."""
        return tuple(sorted(_package_revision_ids(self.result, self.reviews)))

    def __enter__(self) -> ResultPackage:
        if self._closed:
            raise ArchiveError("Пакет результата уже закрыт.")
        return self

    def close(self) -> None:
        if not self._closed:
            self._context.__exit__(None, None, None)
            self._closed = True

    def __exit__(self, *args: object) -> None:
        self.close()


def read_result_package(
    path: Path, *, cancel: ArchiveCancellation | None = None, context: CancellationContext | None = None,
) -> ResultPackage:
    replay_context = _VerificationContext(cancel, context)
    replay_context.check_cancelled()
    package_context = unpack_package(Path(path), expected_kind="trendanalizer-result", cancel=replay_context)
    directory, manifest = package_context.__enter__()
    try:
        replay_context.check_cancelled()
        result_path, assessments_path = directory / "result.json", directory / "assessments.json"
        if result_path.stat().st_size > 25 * 1024 * 1024 or assessments_path.stat().st_size > 25 * 1024 * 1024:
            raise ValueError("Oversized result structure")
        result = AnalysisResult.model_validate(strict_json(result_path.read_bytes()))
        replay_context.check_cancelled()
        raw_assessments = strict_json(assessments_path.read_bytes())
        if not isinstance(raw_assessments, list) or len(raw_assessments) > 60:
            raise ValueError("Invalid assessments")
        validated_artifacts: list[AssessmentArtifact] = []
        for item in raw_assessments:
            replay_context.check_cancelled()
            validated_artifacts.append(AssessmentArtifact.model_validate(item))
        assessments = tuple(validated_artifacts)
        reviews_path = directory / "reviews.json"
        reviews: tuple[ReviewRecord, ...] = ()
        if reviews_path.exists():
            if reviews_path.stat().st_size > 25 * 1024 * 1024:
                raise ValueError("Oversized review structure")
            raw_reviews = strict_json(reviews_path.read_bytes())
            if not isinstance(raw_reviews, list) or not 1 <= len(raw_reviews) <= 60:
                raise ValueError("Invalid expert reviews")
            validated_reviews: list[ReviewRecord] = []
            for raw_review in raw_reviews:
                replay_context.check_cancelled()
                validated_reviews.append(ReviewRecord.model_validate(raw_review))
            reviews = tuple(validated_reviews)
        expected = {"result.json", "assessments.json"} | {
            f"revisions/{digest[:2]}/{digest}.json" for digest in _package_revision_ids(result, reviews)}
        if reviews:
            expected.add("reviews.json")
        if expected != {entry.path for entry in manifest.files}:
            raise ValueError("Unexpected files in result package")
        archive = DocumentArchive(directory / "revisions")
        result, assessments = verify_result(result, archive, assessments, reviews=reviews, context=replay_context)
        replay_context.check_cancelled()
        return ResultPackage(result, assessments, archive, package_context, reviews=reviews)
    except (ArchiveError, TaskCancelled):
        package_context.__exit__(None, None, None)
        raise
    except Exception:
        package_context.__exit__(None, None, None)
        raise ArchiveError("Пакет результата повреждён или его доказательства не прошли проверку.") from None
