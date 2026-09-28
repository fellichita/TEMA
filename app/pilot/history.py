"""Complete-reference history under a frozen TITLE-only admission definition.

Search coverage is coverage of the explicit query union, not worldwide recall.
The same title rule is applied to every year. Abstract availability and model
guesses cannot change membership or numerical growth. Source failure and caps
remain partial; unverified novelty never enters confirmed emerging TOP-15.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Literal, Protocol, cast

from app.backend.contracts import DocumentRecord
from app.backend.providers.base import DocumentProvider
from app.pilot.archive import ArchiveReading, DocumentArchive, RevisionSource
from app.pilot.contracts import METHODOLOGY_VERSION, MethodologyVersion, Candidate, CorpusSnapshot, Coverage, DocumentRevisionRef, Evidence, QueryPlan, SearchQuery, TrendCard, content_hash, verify_evidence_text
from app.pilot.evidence_versions import novelty_method, primary_method, publication_rules
from app.pilot.field_history import FieldExposure, verify_field_exposure
from app.pilot.signal_evidence import PrimaryStudyEvidence, SourceNoveltyEvidence, verify_signal_sources, verify_primary_sources
from app.pilot.evidence import TITLE_ADMISSION_VERSION, EvidenceError, admission_hash, archived_field, build_passport, quote_evidence, candidate_title_matches, coherent_aliases
from app.pilot.methodology import ApplicationKind, ApplicationAssessment, AssessmentArtifact, AssessmentInput, HistoricalSeries, IndependenceAssessment, IndependentGroup, NoveltyAssessment, YearStudies, evaluate_candidate
from app.pilot.sources import collect_snapshot, exclusion_reason
from app.runtime.credentials import CredentialStore
from app.runtime.jobs import RunContext, TaskFailure

if TYPE_CHECKING:
    from app.pilot.antecedents import AntecedentBundle

GROUP_METHOD_VERSION = "openalex-author-components-team-diversity/2.0.0"
LEGACY_GROUP_METHOD_VERSION = "openalex-author-institution-components/1.0.0"


class CancellationContext(Protocol):
    def check_cancelled(self) -> None: ...


@dataclass(frozen=True)
class HistoryResult:
    artifact: AssessmentArtifact
    card: TrendCard
    snapshot: CorpusSnapshot
    new_document_count: int
    rejected_by_admission: int


def _validated_rule(candidate: Candidate) -> None:
    if (candidate.admission_rule_version not in {TITLE_ADMISSION_VERSION, "title-phrase-admission/1.0.0"}
            or candidate.admission_rule_hash != admission_hash(candidate) or not candidate.synonyms):
        raise EvidenceError("Сначала зафиксируйте определение кандидата и единое правило отбора по названиям.")
    if candidate.admission_rule_version == TITLE_ADMISSION_VERSION and not coherent_aliases(candidate.synonyms):
        raise EvidenceError("Замороженные фразы должны обозначать одну конкретную технологию, а не объединение разных направлений.")
    for phrase in (*candidate.synonyms, *candidate.exclusions):
        if not phrase.strip() or len(phrase) > 180:
            raise EvidenceError("Некорректное замороженное поисковое определение.")


def _aggregate_coverage(snapshot: CorpusSnapshot, plan: QueryPlan, *, unresolved_studies: int = 0,
                        comparable: bool = True, admission_version: str = TITLE_ADMISSION_VERSION) -> Coverage:
    if not snapshot.coverage or any(item.source != "openalex" or item.purpose != "history" for item in snapshot.coverage):
        raise EvidenceError("История должна состоять только из reference-запросов OpenAlex.")
    expected_years = set(plan.completed_years)
    if any(set(item.requested_years) != expected_years for item in snapshot.coverage):
        raise EvidenceError("Годы исторического снимка не соответствуют плану.")
    all_complete = all(item.complete_history for item in snapshot.coverage)
    exhausted = all(item.pagination_exhausted for item in snapshot.coverage)
    limited = any(item.limit_reached for item in snapshot.coverage)
    reasons = [reason for item in snapshot.coverage for reason in item.reasons]
    scanned = sum(item.scanned_records for item in snapshot.coverage)
    accepted = sum(item.accepted_records for item in snapshot.coverage)
    rejected = sum(item.rejected_records for item in snapshot.coverage)
    unresolved = sum(item.unresolved_records for item in snapshot.coverage)
    if unresolved_studies:
        # Ambiguous accepted records become unresolved without altering the scan
        # denominator; do not fabricate an extra count per duplicated source.
        moved = min(accepted, unresolved_studies)
        accepted -= moved
        unresolved += moved
        reasons.append("conflicting_study_dates_or_membership")
    if not comparable:
        reasons.append("reference_membership_not_comparable")
    complete = all_complete and not unresolved_studies and comparable
    state: Literal["complete", "partial", "unavailable"] = "complete" if complete else "partial" if scanned else "unavailable"
    return Coverage(source="openalex", purpose="history", query_hash=content_hash({
        "queries": [item.query_hash for item in snapshot.coverage], "admission": admission_version}),
        state=state, requested_years=plan.completed_years, completed_years=plan.completed_years if complete else (),
        pagination_exhausted=exhausted, comparable=complete, scanned_records=scanned,
        accepted_records=accepted, rejected_records=rejected, unresolved_records=unresolved,
        limit_reached=limited, reasons=tuple(dict.fromkeys(reasons)) if reasons else (() if complete else ("incomplete_reference",)))


def _metadata_identities(document: DocumentRecord, *, require_institutions: bool = True) -> tuple[set[str], set[str]] | None:
    authorships = document.raw_metadata.get("authorships")
    if not isinstance(authorships, list) or not authorships:
        return None
    authors: set[str] = set()
    institutions: set[str] = set()
    for item in authorships:
        if not isinstance(item, dict) or not isinstance(item.get("author"), dict):
            return None
        identifier = item["author"].get("id")
        if not isinstance(identifier, str) or re.fullmatch(r"https://openalex\.org/A[0-9]{1,20}", identifier) is None:
            return None
        authors.add(identifier)
        affiliations = item.get("institutions")
        if not isinstance(affiliations, list) or not affiliations:
            if require_institutions:
                return None
            affiliations = []
        for institution in affiliations:
            identifier = institution.get("id") if isinstance(institution, dict) else None
            if not isinstance(identifier, str) or re.fullmatch(r"https://openalex\.org/I[0-9]{1,20}", identifier) is None:
                return None
            institutions.add(identifier)
    return authors, institutions


def _independent_groups(records: dict[str, tuple[DocumentRevisionRef, DocumentRecord]],
                        recent_ids: set[str], context: CancellationContext, *,
                        methodology_version: MethodologyVersion = METHODOLOGY_VERSION) -> tuple[IndependenceAssessment | None, tuple[Evidence, ...], str | None]:
    """Authorship-connected bibliometric teams; shared university is not one lab.

    These observations never claim independent experimental replication. Legacy
    artifacts retain their original author-or-institution graph for replay.
    """
    legacy = methodology_version == "3.0.0"
    method = LEGACY_GROUP_METHOD_VERSION if legacy else GROUP_METHOD_VERSION
    if not recent_ids:
        return None, (), "independent_groups_unknown_or_incomplete"
    ordered = sorted(recent_ids)
    parents = {study: study for study in ordered}

    def find(study: str) -> str:
        while parents[study] != study:
            parents[study] = parents[parents[study]]
            study = parents[study]
        return study

    identities: dict[str, str] = {}
    for study in ordered:
        context.check_cancelled()
        metadata = _metadata_identities(records[study][1], require_institutions=legacy)
        if metadata is None:
            return None, (), "Авторские или институциональные OpenAlex ID доступны не для всех недавних работ; независимость неизвестна."
        for identity in (metadata[0] | metadata[1] if legacy else metadata[0]):
            if identity in identities:
                left, right = find(study), find(identities[identity])
                parents[max(left, right)] = min(left, right)
            else:
                identities[identity] = study
    groups: dict[str, list[str]] = {}
    for study in ordered:
        groups.setdefault(find(study), []).append(study)
    if len(groups) > 150:
        return None, (), "Число групп превышает бюджет доказательств паспорта; независимость не оценена полностью."
    result = []
    evidence_items = []
    for studies in sorted(groups.values()):
        reference, document = records[studies[0]]
        text = json.dumps(document.raw_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        # Cite a source metadata object, retaining the full exact revision in the
        # archive. The algorithm can reproduce every edge using all study IDs.
        metadata = document.raw_metadata.get("authorships")
        if not isinstance(metadata, list) or not metadata:
            return None, (), "Отсутствуют проверяемые авторские метаданные."
        quote = json.dumps(metadata[0], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(quote) > 12000 or quote not in text:
            return None, (), "Метаданные авторской группы превышают границы цитирования; независимость неизвестна."
        evidence = quote_evidence(reference, document, text_field="metadata", quote=quote)
        evidence_items.append(evidence)
        result.append(IndependentGroup(group_id="group-" + content_hash({"studies": studies, "method": method}),
                                       study_ids=tuple(studies), evidence_ids=(evidence.evidence_id,)))
    return IndependenceAssessment(groups=tuple(result), method_version=method,
                                   coverage_complete=True), tuple(evidence_items), None


def _verified_application(passport: TrendCard, requested: ApplicationAssessment | None = None) -> ApplicationAssessment | None:
    """Only explicit grounded application claims supply this optional component."""
    kinds = {"research", "demonstrator", "deployment"}
    found = []
    for claim in passport.claims:
        if claim.role != "application" or claim.support != "supported":
            continue
        method = (claim.grounding_method or "").split("/")
        if len(method) >= 2 and method[0] in {"verified-application", "reviewed-application"} and method[1] in kinds:
            # Automated extraction currently verifies research only. An explicit
            # attributed review is needed for demonstrator/deployment maturity.
            if method[0] == "verified-application" and method[1] != "research":
                continue
            found.append(ApplicationAssessment(kind=cast(ApplicationKind, method[1]), claim_id=claim.claim_id))
    if requested is not None:
        if requested not in found:
            raise EvidenceError("Стадия применения требует отдельного подтверждённого утверждения и источника.")
        return requested
    return found[0] if len(found) == 1 else None


def _exclude_previously_published_families(accepted: dict[str, tuple[DocumentRevisionRef, DocumentRecord]],
        historical: dict[str, list[tuple[DocumentRevisionRef, DocumentRecord]]], bundle: AntecedentBundle,
        archive: RevisionSource, context: CancellationContext, *,
        methodology_version: MethodologyVersion = METHODOLOGY_VERSION,
        excluded_study_ids: frozenset[str] = frozenset()) -> set[str]:
    """A later journal manifestation cannot count as a new study after an older preprint.

    Include all historical manifestations to retain relation chains, even when
    their current representative is a different DOI. Cross-window contradictory
    dates of the same identity remain unresolved, not a clean first-publication claim.
    """
    from app.pilot.discovery import deduplicate

    earlier_ids = set(bundle.matched_study_ids) - excluded_study_ids
    if not earlier_ids or not accepted:
        return set()
    documents = [document for versions in historical.values() for _, document in versions]
    for reference in bundle.snapshot.documents:
        context.check_cancelled()
        if reference.study_id in earlier_ids:
            documents.append(archive.get(reference.revision_id))
    years: dict[str, set[int | None]] = {}
    for document in documents:
        years.setdefault(document.document_key, set()).add(document.publication_year)
    conflicts = {key for key in earlier_ids & set(historical) if len(years[key]) > 1}
    for family in deduplicate(documents,
            version="study-families-v2-publication-units" if methodology_version == "3.4.0" else "study-families-v1"):
        context.check_cancelled()
        identities = {documents[index].document_key for index in family["input_indices"]}
        if identities & earlier_ids:
            for identity in identities:
                accepted.pop(identity, None)
    return conflicts


def apply_publication_status(passport: TrendCard, withdrawn: frozenset[str],
                             supporting: frozenset[str]) -> TrendCard:
    """Retain quotations, but revoke automatic support contradicted by known status."""
    evidence = {item.evidence_id: item for item in passport.evidence}
    claims = []
    for claim in passport.claims:
        blocked = withdrawn | supporting if claim.role in {"case", "advantage", "application", "novelty"} else withdrawn
        contradicted = any(evidence[identifier].study_id in blocked for identifier in claim.evidence_ids)
        if contradicted and (claim.grounding_method or "").startswith("archived-author-novelty/"):
            continue
        if contradicted and claim.support == "supported":
            if (claim.grounding_method or "").startswith(("reviewed-novelty/", "reviewed-evidence/", "reviewed-application/")):
                raise EvidenceError("Сведения о статусе источника изменились; требуется новый экспертный пересмотр утверждения.")
            claim = claim.model_copy(update={"support": "unverified"})
        claims.append(claim)
    if tuple(claims) == passport.claims:
        return passport
    return passport.model_copy(update={"claims": tuple(claims), "limitations": tuple(dict.fromkeys((*passport.limitations,
        "Поддержка утверждений отозвана: сохранённые сведения указывают на отзыв работы или дополнительный материал вместо самостоятельного исследования.")))})


def assess_snapshot(candidate: Candidate, plan: QueryPlan, snapshot: CorpusSnapshot,
                    archive: RevisionSource, context: CancellationContext, *, passport: TrendCard,
                    verified_novelty: NoveltyAssessment | None = None,
                    verified_application: ApplicationAssessment | None = None,
                    antecedents: AntecedentBundle | None = None,
                    field_exposure: FieldExposure | None = None,
                    source_novelty: tuple[SourceNoveltyEvidence, ...] = (),
                    primary_observations: tuple[PrimaryStudyEvidence, ...] = (),
                    publication_status_revisions: tuple[DocumentRevisionRef, ...] = (),
                    methodology_version: MethodologyVersion = METHODOLOGY_VERSION) -> tuple[AssessmentArtifact, TrendCard, int]:
    """Pure local assessment after collection; useful for restore and sensitivity."""
    # Status, quotation and history checks revisit many of the same immutable
    # revisions. Share verified reads within this assessment only; a later
    # assessment must reread the archive so damaged files remain detectable.
    if isinstance(archive, DocumentArchive):
        archive = ArchiveReading(archive, limit=2048)
    _validated_rule(candidate)
    legacy = methodology_version == "3.0.0"
    if (snapshot.purpose != "history" or snapshot.plan_hash != plan.plan_hash
            or candidate.plan_hash != plan.plan_hash or passport.candidate != candidate):
        raise EvidenceError("Кандидат, паспорт, история и план не согласованы.")
    if field_exposure is not None:
        verify_field_exposure(field_exposure, plan)
    status_references: tuple[DocumentRevisionRef, ...] = ()
    withdrawn: frozenset[str] = frozenset()
    supporting: frozenset[str] = frozenset()
    if methodology_version == "3.4.0":
        from app.pilot.publication_status import collect_status_context

        statuses = collect_status_context((*publication_status_revisions, *snapshot.documents,
            *(antecedents.snapshot.documents if antecedents is not None else ())), archive, context,
            as_of=plan.as_of, evidence=passport.evidence)
        status_references, withdrawn, supporting = statuses.references, statuses.withdrawn, statuses.supporting
        blocked = withdrawn | supporting
        source_novelty = tuple(item for item in source_novelty if item.novelty.study_id not in blocked)
        primary_observations = tuple(item for item in primary_observations if item.result.study_id not in blocked)
        passport = apply_publication_status(passport, withdrawn, supporting)
    elif publication_status_revisions:
        raise EvidenceError("Контекст статусов публикаций требует методики 3.4.")
    if source_novelty:
        verify_signal_sources(source_novelty, candidate, archive, context, query_plan=plan,
                              method_version=novelty_method(methodology_version) if methodology_version == "3.4.0" else None)
    if primary_observations:
        if methodology_version not in {"3.3.0", "3.4.0"}:
            raise EvidenceError("Первичные наблюдения требуют методики 3.3.")
        verify_primary_sources(primary_observations, candidate, archive, context, query_plan=plan,
                               method_version=primary_method(methodology_version) if methodology_version == "3.4.0" else None)
    evidence_by_id = {evidence.evidence_id: evidence for evidence in passport.evidence}
    for primary in primary_observations:
        if evidence_by_id.get(primary.result.evidence_id) != primary.result:
            raise EvidenceError("Первичное наблюдение отсутствует в доказательствах паспорта.")
    for source_entry in source_novelty:
        for quotation in (source_entry.novelty, source_entry.experiment):
            if evidence_by_id.get(quotation.evidence_id) != quotation:
                raise EvidenceError("Авторское свидетельство отсутствует в архивных доказательствах паспорта.")
        source_document = archive.get(source_entry.novelty.revision_id)
        if (source_document.publication_year != source_entry.publication_year
                or source_entry.publication_year > plan.as_of.year
                or source_document.publication_date is not None and source_document.publication_date > plan.as_of
                or not candidate_title_matches(source_document.title, candidate)):
            raise EvidenceError("Дата или технология авторского свидетельства не соответствует исторической оценке.")
    for evidence in passport.evidence:
        archived = archive.get(evidence.revision_id)
        verify_evidence_text(evidence, archived_field(archived, evidence.text_field))
        if evidence.source_url != archived.url or evidence.source != archived.source:
            raise EvidenceError("Источник свидетельства не соответствует архивной ревизии.")
    for claim in passport.claims:
        if (methodology_version == "3.4.0" and claim.support == "supported"
                and not (claim.grounding_method or "").startswith(("reviewed-novelty/", "reviewed-application/", "reviewed-evidence/"))):
            from app.pilot.grounding import verify_supported_source_claim

            try:
                verify_supported_source_claim(claim, evidence_by_id,
                    {evidence_by_id[identifier].revision_id: archive.get(evidence_by_id[identifier].revision_id)
                     for identifier in claim.evidence_ids}, legacy=False,
                    candidate_studies=set(candidate.discovery_study_ids), methodology_version=methodology_version)
                if claim.role in {"advantage", "application"} and candidate.specificity != "specific_technology":
                    raise ValueError("Технологические границы преимущества не подтверждены.")
            except (KeyError, ValueError) as error:
                raise EvidenceError("Цитата не подтверждает роль поля паспорта: " + str(error)) from None
            continue
        if claim.support == "supported" and claim.grounding_method in {"exact-archived-quotation/1.0.0", "exact-contextual-quotation/2.0.0", "exact-contextual-quotation/3.0.0", "verified-application/research"}:
            if len(claim.evidence_ids) != 1 or claim.text != evidence_by_id[claim.evidence_ids[0]].quote:
                raise EvidenceError("Дословная цитата подменена неподтверждённым пересказом.")
            if claim.grounding_method == "exact-contextual-quotation/3.0.0":
                from app.pilot.evidence import QuoteSelection, _selection_supported
                quotation = evidence_by_id[claim.evidence_ids[0]]
                selection = QuoteSelection.model_validate(dict(role=claim.role, revision_id=quotation.revision_id,
                                           field=quotation.text_field, quote=quotation.quote))
                if not _selection_supported(selection, archive.get(quotation.revision_id), method_version=claim.grounding_method):
                    raise EvidenceError("Цитата не подтверждает роль первичного результата.")
        elif claim.support == "supported" and not (claim.grounding_method or "").startswith(("reviewed-novelty/", "reviewed-application/", "reviewed-evidence/")):
            raise EvidenceError("Для утверждения не указана поддерживаемая проверка доказательства.")
    start, end = date(plan.completed_years[0], 1, 1), date(plan.completed_years[-1], 12, 31)
    by_study: dict[str, list[tuple[DocumentRevisionRef, DocumentRecord]]] = {}
    for reference in snapshot.documents:
        context.check_cancelled()
        document = archive.get(reference.revision_id)
        if reference.source != "openalex" or document.source != "openalex":
            raise EvidenceError("Дополнительный каталог не может увеличивать reference-историю.")
        if (reference.source_id != document.source_id or reference.study_id != document.document_key
                or reference.publication_year != document.publication_year):
            raise EvidenceError("Историческая запись имеет неподтверждённый идентификатор исследования.")
        by_study.setdefault(document.document_key, []).append((reference, document))
    exclusion_cache: dict[tuple[str, bool], bool] = {}
    primary_only = methodology_version in {"3.3.0", "3.4.0"}
    historical_rules = publication_rules(methodology_version)

    def excluded(reference: DocumentRevisionRef, document: DocumentRecord, *, legacy_rule: bool) -> bool:
        # Both admission passes inspect the same immutable revision under the
        # same frozen rule. Keep the legacy flag in the key for old methods.
        key = reference.revision_id, legacy_rule
        if key not in exclusion_cache:
            exclusion_cache[key] = bool(exclusion_reason(document, start, end,
                legacy=legacy_rule,
                primary_only=primary_only, rules_version=historical_rules))
        return exclusion_cache[key]

    accepted: dict[str, tuple[DocumentRevisionRef, DocumentRecord]] = {}
    conflicts: list[str] = []
    rejected = 0
    for study, versions in by_study.items():
        context.check_cancelled()
        if study in withdrawn or study in supporting:
            rejected += 1
            continue
        years = {document.publication_year for _, document in versions}
        membership = {candidate_title_matches(document.title, candidate) for _, document in versions}
        if len(years) != 1 or None in years or len(membership) != 1:
            conflicts.append(study)
            continue
        # A retraction or supplement flag in ANY immutable version prevents a
        # clean later version hiding the exclusion during selection.
        if any(excluded(reference, document, legacy_rule=methodology_version in {"3.0.0", "3.1.0"})
               for reference, document in versions) or not next(iter(membership)):
            rejected += 1
            continue
        accepted[study] = max(versions, key=lambda item: (item[0].observed_at, item[0].revision_id))
    if not legacy and accepted:
        # A preprint and its journal version are one study, counted at the first
        # available publication year in this bounded historical corpus. Keep an
        # actual archived identity/year pair so provenance remains verifiable.
        from app.pilot.discovery import deduplicate
        pairs = [pair for versions in by_study.values() for pair in versions]
        families = deduplicate([document for _, document in pairs],
            version="study-families-v2-publication-units" if methodology_version == "3.4.0" else "study-families-v1")
        unified = {}
        for family in families:
            context.check_cancelled()
            members = [pairs[index] for index in family["input_indices"]]
            member_keys = {document.document_key for _, document in members}
            if not member_keys.intersection(accepted):
                continue
            if any(excluded(reference, document, legacy_rule=methodology_version == "3.1.0")
                   for reference, document in members):
                # A linked clean version cannot hide a retracted manifestation.
                continue
            if member_keys.intersection(conflicts) or not member_keys.issubset(accepted):
                conflicts.extend(sorted(member_keys - set(conflicts)))
                continue
            earliest = min(members, key=lambda item: (item[1].publication_year, item[1].document_key))
            unified[earliest[1].document_key] = earliest
        accepted = unified
    if antecedents is not None and not legacy:
        from app.pilot.antecedents import verify_antecedents
        verify_antecedents(antecedents, candidate, plan, archive, check=context.check_cancelled)
        conflicts.extend(_exclude_previously_published_families(accepted, by_study, antecedents, archive, context,
                                                               methodology_version=methodology_version,
                                                               excluded_study_ids=withdrawn | supporting))
        conflicts = sorted(set(conflicts))
    if len(accepted) > 3000:
        raise EvidenceError("Исторический кандидат превышает лимит 3000 исследований.")
    coverage = _aggregate_coverage(snapshot, plan, unresolved_studies=len(conflicts), admission_version=candidate.admission_rule_version)
    observations = tuple(YearStudies(year=year, study_ids=tuple(sorted(study for study, (_, document) in accepted.items()
                         if document.publication_year == year))) for year in plan.completed_years)
    first: tuple[int, str] | None = min(((document.publication_year, study) for study, (_, document) in accepted.items()
        if document.publication_year is not None), default=None)
    earlier_search_complete = False
    if antecedents is not None and not legacy:
        older = None
        if methodology_version == "3.4.0":
            usable_earlier = set(antecedents.matched_study_ids) - withdrawn - supporting
            older = min(((ref.publication_year, ref.study_id) for ref in antecedents.snapshot.documents
                if ref.study_id in usable_earlier and ref.publication_year is not None), default=None)
        elif antecedents.earliest_observed_year is not None and antecedents.earliest_observed_study_id is not None:
            older = (antecedents.earliest_observed_year, antecedents.earliest_observed_study_id)
        if older is not None and (first is None or older < first):
            first = older
        earlier_search_complete = antecedents.search_complete
    history = HistoricalSeries(candidate_id=candidate.candidate_id, snapshot_id=snapshot.snapshot_id,
        admission_rule_hash=candidate.admission_rule_hash, as_of=plan.as_of, observations=observations,
        coverage=coverage, first_observed_year=first[0] if first else None,
        first_observed_study_id=first[1] if first else None, earlier_search_complete=earlier_search_complete,
        date_conflicts=tuple(sorted(set(conflicts))))
    recent = {study for observation in observations[-3:] for study in observation.study_ids}
    independence, group_evidence, group_limitation = _independent_groups(accepted, recent, context, methodology_version=methodology_version)
    if verified_novelty is not None:
        claims = {claim.claim_id: claim for claim in passport.claims}
        source_claim = claims.get(verified_novelty.claim_id)
        if (source_claim is None or source_claim.role != "novelty" or source_claim.support != "supported"
                or source_claim.grounding_method is None
                or not source_claim.grounding_method.startswith("reviewed-novelty/")):
            raise EvidenceError("Новизна требует отдельной проверенной оценки с прямыми ссылками на свидетельства.")
    all_evidence = tuple({item.evidence_id: item for item in (*passport.evidence, *group_evidence)}.values())
    if len(all_evidence) > 200:
        independence, group_evidence = None, ()
        all_evidence = passport.evidence
        group_limitation = "Исчерпан бюджет доказательств независимых групп; этот компонент неизвестен."
    application = None
    if not legacy:
        application = _verified_application(passport, verified_application)
    inputs = AssessmentInput(candidate=candidate, history=history, claims=passport.claims,
        evidence_ids=tuple(item.evidence_id for item in all_evidence), novelty=verified_novelty,
        independence=independence, application=application,
        field_exposure=field_exposure if methodology_version in {"3.2.0", "3.3.0", "3.4.0"} else None,
        source_novelty=source_novelty if methodology_version in {"3.2.0", "3.3.0", "3.4.0"} else (),
        primary_observations=primary_observations if methodology_version in {"3.3.0", "3.4.0"} else (),
        publication_status_revisions=status_references,
        antecedents=antecedents if methodology_version in {"3.2.0", "3.3.0", "3.4.0"} else None)
    assessment = evaluate_candidate(inputs, version=methodology_version)
    artifact = AssessmentArtifact(inputs=inputs, assessment=assessment)
    limitations = [item for item in passport.limitations if item != "Кандидат ещё не проверен по полному историческому корпусу."]
    limitations.extend(assessment.limitations)
    limitations.append("Исторический отбор использует только названия и замороженные фразы; это не семантический отбор и не вся мировая литература.")
    limitations.append("Группы объединены по совпадающим OpenAlex ID авторов или организаций; это не доказательство независимого воспроизведения эксперимента." if legacy else
        "Библиометрические команды объединены по общим OpenAlex ID авторов; общая организация сама по себе не объединяет лаборатории. Эти метаданные не доказывают независимую репликацию.")
    if antecedents is not None and not legacy:
        limitations.append(f"Ранние упоминания проверены с {antecedents.first_year} года по замороженным фразам; это не доказательство даты изобретения или отсутствия более ранних работ.")
    if group_limitation:
        limitations.append(group_limitation)
    if verified_novelty is None:
        limitations.append("Рост может быть подтверждён, но новизна механизма и более ранние аналоги требуют отдельной проверки.")
    # Completeness describes the data, while each optional interpretation retains
    # its own unverified support flag. It cannot supply a confirmation gate.
    quality = assessment.quality
    category = assessment.category
    card = TrendCard(candidate=candidate, methodology_version=None if legacy else methodology_version,
        category=category, quality=quality, claims=passport.claims,
        evidence=all_evidence, historical_snapshot_id=snapshot.snapshot_id,
        assessment_hash=assessment.assessment_hash, limitations=tuple(dict.fromkeys(limitations)))
    return artifact, card, rejected


def verify_history_artifact(artifact: AssessmentArtifact, plan: QueryPlan, snapshot: CorpusSnapshot,
                            archive: RevisionSource, passport: TrendCard,
                            context: CancellationContext, *, antecedents: AntecedentBundle | None = None,
                            publication_status_revisions: tuple[DocumentRevisionRef, ...] | None = None,
                            methodology_version: MethodologyVersion | None = None) -> None:
    """Recompute admission, conflicts and groups from every archived revision.

    Already assessed cards are accepted: metadata evidence is deduplicated by
    its deterministic ID, preserving the original order and budget behavior.
    """
    replayed, _, _ = assess_snapshot(artifact.inputs.candidate, plan, snapshot, archive, context,
        passport=passport, verified_novelty=artifact.inputs.novelty, verified_application=artifact.inputs.application,
        antecedents=antecedents if antecedents is not None else artifact.inputs.antecedents,
        field_exposure=artifact.inputs.field_exposure, source_novelty=artifact.inputs.source_novelty,
        primary_observations=artifact.inputs.primary_observations,
        publication_status_revisions=(artifact.inputs.publication_status_revisions if publication_status_revisions is None
                                      else publication_status_revisions),
        methodology_version=methodology_version or artifact.assessment.methodology_version)
    if replayed != artifact:
        raise EvidenceError("Историческая оценка не воспроизводится по архивным источникам.")


def assess_history(candidate: Candidate, plan: QueryPlan, discovery_snapshot: CorpusSnapshot,
                   archive: DocumentArchive, credentials: CredentialStore, context: RunContext, *,
                   passport: TrendCard | None = None, remaining_new_documents: int = 20000,
                   provider_factory: Callable[[str], DocumentProvider] | None = None,
                   verified_novelty: NoveltyAssessment | None = None,
                   verified_application: ApplicationAssessment | None = None,
                   antecedents: AntecedentBundle | None = None,
                   field_exposure: FieldExposure | None = None,
                   source_novelty: tuple[SourceNoveltyEvidence, ...] = (),
                   primary_observations: tuple[PrimaryStudyEvidence, ...] = (),
                   methodology_version: MethodologyVersion = METHODOLOGY_VERSION) -> HistoryResult:
    _validated_rule(candidate)
    if type(remaining_new_documents) is not int or not 0 <= remaining_new_documents <= 20000:
        raise ValueError("Недопустимый остаток исторического бюджета.")
    stage = "history_" + content_hash({"id": candidate.candidate_id})[:20]
    checkpoint = context.load_checkpoint(stage)
    checkpoint_input = {"candidate": candidate.model_dump(mode="json"), "plan": plan.plan_hash}
    if methodology_version == "3.4.0":
        checkpoint_input.update(methodology_version=methodology_version,
                                publication_rules=publication_rules(methodology_version))
    input_hash = content_hash(checkpoint_input)
    if checkpoint is not None:
        if checkpoint.get("input_hash") != input_hash:
            raise EvidenceError("История относится к другому замороженному определению.")
        snapshot = CorpusSnapshot.model_validate(checkpoint["snapshot"])
        saved_count = checkpoint.get("new_document_count")
        if type(saved_count) is not int or not 0 <= saved_count <= 3000:
            raise EvidenceError("Некорректный сохранённый расход исторического бюджета.")
        new_document_count = saved_count
    else:
        if remaining_new_documents == 0 and methodology_version not in {"3.3.0", "3.4.0"}:
            raise TaskFailure("Исчерпан бюджет новых исторических документов; кандидат остаётся предварительным.")
        before = {reference.study_id for reference in discovery_snapshot.documents}
        queries = tuple(SearchQuery(source="openalex", purpose="history", text=phrase) for phrase in candidate.synonyms)
        if remaining_new_documents == 0:
            snapshot = CorpusSnapshot(snapshot_id="unavailable-" + content_hash({"candidate": candidate.candidate_id,
                "plan": plan.plan_hash}), plan_hash=plan.plan_hash, purpose="history", as_of=plan.as_of,
                created_at=discovery_snapshot.created_at, documents=(), coverage=(Coverage(source="openalex",
                purpose="history", query_hash=content_hash([query.model_dump(mode="json") for query in queries]), state="unavailable", requested_years=plan.completed_years,
                pagination_exhausted=False, comparable=False, reasons=("history_disabled_or_budget_unavailable",)),),
                normalizer_version="explicit-unavailable-history/1.0.0", deduplication_version="doi-source-id-v1")
        else:
            snapshot = collect_snapshot(plan, context, archive, credentials, purpose="history", queries=queries,
                max_documents=min(plan.limits.historical_documents_per_candidate, remaining_new_documents),
                provider_factory=provider_factory, rules_version=publication_rules(methodology_version))
        new_document_count = len({reference.study_id for reference in snapshot.documents} - before)
        context.checkpoint(stage, {"input_hash": input_hash, "snapshot": snapshot.model_dump(mode="json"),
                                   "new_document_count": new_document_count,
                                   "collection_limit": min(plan.limits.historical_documents_per_candidate, remaining_new_documents)})
    if passport is None:
        passport = build_passport(candidate, discovery_snapshot, archive, context, methodology_version=methodology_version)
    artifact, card, rejected = assess_snapshot(candidate, plan, snapshot, archive, context,
        passport=passport, verified_novelty=verified_novelty, verified_application=verified_application,
        antecedents=antecedents, field_exposure=field_exposure, source_novelty=source_novelty,
        primary_observations=primary_observations,
        publication_status_revisions=discovery_snapshot.documents if methodology_version == "3.4.0" else (),
        methodology_version=methodology_version)
    context.check_cancelled()
    return HistoryResult(artifact, card, snapshot, new_document_count, rejected)
