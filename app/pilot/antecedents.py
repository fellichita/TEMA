"""On-demand older-reference search under the same frozen title definition.

This is evidence of earlier *observed terminology*, never proof of invention,
maturity, worldwide absence or semantic equivalence. It runs after the primary
analysis, separately from its six-year growth counts.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, UTC
import hashlib
from typing import Final, Literal, Self

from pydantic import Field, model_validator

from app.backend.contracts import DocumentRecord, SearchRequest
from app.backend.errors import BackendError, CancelledError
from app.backend.providers.base import DocumentProvider
from app.pilot.archive import DocumentArchive, RevisionSource, document_text
from app.pilot.contracts import Candidate, Contract, CorpusSnapshot, Coverage, DocumentRevisionRef, Evidence, QueryPlan, content_hash, verify_evidence_text
from app.pilot.evidence import TITLE_ADMISSION_VERSION, EvidenceError, admission_hash, archived_field, quote_evidence, candidate_title_matches
from app.pilot.sources import PublicationProviderSession, exclusion_reason, make_provider, supporting_asset_keys
from app.pilot.retractions import LEGACY_STATUS_RULES, STATUS_RULES, retracted_family_keys
from app.runtime.credentials import CredentialStore, CredentialUnavailable
from app.runtime.jobs import RunContext, TaskCancelled

LEGACY_ANTECEDENT_VERSION: Final = "openalex-antecedent-title-review/1.0.0"
ANTECEDENT_VERSION: Final = "openalex-antecedent-title-review/1.2.0"


class AntecedentBundle(Contract):
    # An omitted version belongs to the original contract. New collection writes
    # its version explicitly so an old archive never gains new admission rules.
    version: Literal["openalex-antecedent-title-review/1.0.0",
                     "openalex-antecedent-title-review/1.1.0",
                     "openalex-antecedent-title-review/1.2.0"] = LEGACY_ANTECEDENT_VERSION
    candidate_id: str
    admission_rule_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    first_year: int = Field(ge=1800, le=9999, strict=True)
    query_plan: QueryPlan
    snapshot: CorpusSnapshot
    operational_status: Literal["earlier_matches_found", "none_found_within_queries", "incomplete_search"]
    earliest_observed_year: int | None = None
    earliest_observed_study_id: str | None = None
    matched_study_ids: tuple[str, ...] = Field(max_length=500)
    conflicting_study_ids: tuple[str, ...] = Field(default=(), max_length=500)
    evidence: tuple[Evidence, ...] = Field(max_length=60)
    limitations: tuple[str, ...]

    @property
    def bundle_hash(self) -> str:
        return content_hash(self)

    @property
    def search_complete(self) -> bool:
        return (bool(self.snapshot.coverage) and not self.conflicting_study_ids
                and all(item.state == "complete" for item in self.snapshot.coverage))

    @model_validator(mode="after")
    def consistent_bundle(self) -> Self:
        if self.snapshot.purpose != "enrichment" or any(item.source != "openalex" for item in self.snapshot.documents):
            raise ValueError("Antecedents require a separate OpenAlex enrichment snapshot")
        if len(set(self.matched_study_ids)) != len(self.matched_study_ids):
            raise ValueError("Duplicate antecedent studies")
        if not set(self.matched_study_ids).issubset({ref.study_id for ref in self.snapshot.documents}):
            raise ValueError("Antecedent study missing from snapshot")
        if (self.earliest_observed_year is None) != (self.earliest_observed_study_id is None):
            raise ValueError("First observation requires both year and study")
        if self.matched_study_ids:
            if self.operational_status != "earlier_matches_found" or self.earliest_observed_study_id not in self.matched_study_ids:
                raise ValueError("Known earlier matches require an observed source")
        elif self.operational_status != ("none_found_within_queries" if self.search_complete else "incomplete_search"):
            raise ValueError("No-result status must preserve incomplete coverage")
        return self


def _check_candidate(candidate: Candidate, plan: QueryPlan, first_year: int) -> None:
    if (candidate.plan_hash != plan.plan_hash or candidate.admission_rule_version not in {TITLE_ADMISSION_VERSION, "title-phrase-admission/1.0.0"}
            or candidate.admission_rule_hash != admission_hash(candidate) or not candidate.synonyms):
        raise EvidenceError("Обзор аналогов требует неизменного определения кандидата из текущего плана.")
    if type(first_year) is not int or not 1800 <= first_year < plan.completed_years[0]:
        raise ValueError("Начало обзора должно быть не ранее 1800 года и раньше основной истории.")


def _derive_matches(candidate: Candidate, snapshot: CorpusSnapshot, archive: RevisionSource,
                    first_year: int, last_year: int, check: Callable[[], None] | None = None,
                    *, version: str = ANTECEDENT_VERSION,
                    ) -> tuple[dict[str, tuple[DocumentRevisionRef, DocumentRecord]], tuple[str, ...]]:
    if version not in {LEGACY_ANTECEDENT_VERSION, "openalex-antecedent-title-review/1.1.0", "openalex-antecedent-title-review/1.2.0"}:
        raise ValueError("Unknown antecedent rules")
    by_study: dict[str, list[tuple[DocumentRevisionRef, DocumentRecord]]] = {}
    for reference in snapshot.documents:
        if check:
            check()
        document = archive.get(reference.revision_id)
        if (reference.source != "openalex" or document.source != "openalex"
                or reference.study_id != document.document_key or reference.source_id != document.source_id
                or reference.publication_year != document.publication_year
                or reference.observed_at != document.fetched_at
                or reference.publicly_available_at != document.publication_date
                or reference.text_hash != hashlib.sha256(document_text(document).encode("utf-8")).hexdigest()):
            raise EvidenceError("Ранний источник не соответствует архивной ревизии.")
        by_study.setdefault(reference.study_id, []).append((reference, document))
    accepted: dict[str, tuple[DocumentRevisionRef, DocumentRecord]] = {}
    conflicts = []
    withdrawn = retracted_family_keys([doc for versions in by_study.values() for _, doc in versions],
                                     check=check, rules_version=STATUS_RULES) if version == "openalex-antecedent-title-review/1.2.0" else set()
    supporting = supporting_asset_keys([doc for versions in by_study.values() for _, doc in versions],
        check=check, rules_version=STATUS_RULES) if version == "openalex-antecedent-title-review/1.2.0" else set()
    for study, versions in by_study.items():
        if check:
            check()
        years = {doc.publication_year for _, doc in versions}
        membership = {candidate_title_matches(doc.title, candidate) for _, doc in versions}
        if len(years) != 1 or None in years or len(membership) != 1:
            conflicts.append(study)
            continue
        if (study in withdrawn | supporting or not next(iter(membership))
                or any(exclusion_reason(doc, date(first_year, 1, 1), date(last_year, 12, 31),
                                              legacy=version == LEGACY_ANTECEDENT_VERSION,
                                              primary_only=version == "openalex-antecedent-title-review/1.2.0",
                                              rules_version=STATUS_RULES if version == "openalex-antecedent-title-review/1.2.0" else LEGACY_STATUS_RULES)
                                             for _, doc in versions)):
            continue
        accepted[study] = max(versions, key=lambda pair: (pair[0].observed_at, pair[0].revision_id))
    return accepted, tuple(sorted(conflicts))


def verify_antecedents(bundle: AntecedentBundle, candidate: Candidate, plan: QueryPlan,
                       archive: RevisionSource, *, check: Callable[[], None] | None = None) -> None:
    """Reproduce temporal membership and all quotations from immutable records."""
    bundle = AntecedentBundle.model_validate_json(bundle.model_dump_json())
    _check_candidate(candidate, plan, bundle.first_year)
    if (bundle.candidate_id != candidate.candidate_id or bundle.admission_rule_hash != candidate.admission_rule_hash
            or bundle.query_plan != plan or bundle.snapshot.plan_hash != plan.plan_hash or bundle.snapshot.as_of != plan.as_of):
        raise EvidenceError("Обзор ранних аналогов относится к другому определению или срезу.")
    years = tuple(range(bundle.first_year, plan.completed_years[0]))
    expected_queries = tuple(content_hash({"phrase": phrase, "start": years[0], "end": years[-1],
                                          "admission": candidate.admission_rule_hash}) for phrase in candidate.synonyms)
    if (tuple(item.query_hash for item in bundle.snapshot.coverage) != expected_queries
            or any(item.source != "openalex" or item.purpose != "enrichment" or item.requested_years != years
                   for item in bundle.snapshot.coverage)):
        raise EvidenceError("Покрытие обзора не соответствует всем замороженным поисковым фразам.")
    matched, conflicts = _derive_matches(candidate, bundle.snapshot, archive, years[0], years[-1], check,
                                         version=bundle.version)
    first = min(((doc.publication_year, study) for study, (_, doc) in matched.items()), default=None)
    if (tuple(sorted(matched)) != bundle.matched_study_ids or conflicts != bundle.conflicting_study_ids
            or bundle.earliest_observed_year != (first[0] if first else None)
            or bundle.earliest_observed_study_id != (first[1] if first else None)):
        raise EvidenceError("Наблюдение ранних аналогов не воспроизводится по исходным документам.")
    refs = {ref.revision_id: ref for ref in bundle.snapshot.documents}
    if len(refs) > 500 or len(refs) != len(bundle.snapshot.documents):
        raise EvidenceError("Обзор ранних источников превышает бюджет или содержит повторные ревизии.")
    for evidence in bundle.evidence:
        if check:
            check()
        reference = refs.get(evidence.revision_id)
        if reference is None or evidence.study_id not in matched or evidence.study_id != reference.study_id:
            raise EvidenceError("Цитата обзора ссылается на отсутствующий ранний источник.")
        document = archive.get(evidence.revision_id)
        verify_evidence_text(evidence, archived_field(document, evidence.text_field))
        if evidence.source != document.source or evidence.source_url != document.url:
            raise EvidenceError("Источник цитаты раннего аналога подменён.")


def _collect_antecedents(candidate: Candidate, plan: QueryPlan, archive: DocumentArchive,
                         credentials: CredentialStore, context: RunContext, *, first_year: int = 1900,
                         max_documents: int = 500,
                         provider_factory: Callable[[str], DocumentProvider] | None = None) -> AntecedentBundle:
    """Optional review-stage collection, capped at 500 scanned source records."""
    _check_candidate(candidate, plan, first_year)
    if type(max_documents) is not int or not 1 <= max_documents <= 500:
        raise ValueError("Обзор ранних аналогов ограничен 500 записями.")
    years = tuple(range(first_year, plan.completed_years[0]))
    identity = content_hash({"candidate": candidate.model_dump(mode="json"), "plan": plan.plan_hash, "first_year": first_year,
                             "max_documents": max_documents, "version": ANTECEDENT_VERSION})
    stage = "antecedents_" + identity[:20]
    saved = context.load_checkpoint(stage)
    if saved is not None:
        if saved.get("input_hash") != identity:
            raise EvidenceError("Сохранённый обзор ранних аналогов относится к другой задаче.")
        bundle = AntecedentBundle.model_validate(saved["bundle"])
        verify_antecedents(bundle, candidate, plan, archive, check=context.check_cancelled)
        return bundle
    references: dict[str, DocumentRevisionRef] = {}
    coverage = []
    consumed = 0
    for phrase in candidate.synonyms:
        context.check_cancelled()
        query_hash = content_hash({"phrase": phrase, "start": years[0], "end": years[-1],
                                   "admission": candidate.admission_rule_hash})
        scanned = accepted = rejected = unresolved = 0
        exhausted = limited = False
        reasons = []
        provider = None
        if consumed >= max_documents:
            limited = True
            reasons.append("antecedent_scan_budget")
        else:
            try:
                provider = provider_factory("openalex") if provider_factory else make_provider("openalex", credentials)
                request = SearchRequest(topic=phrase, source="openalex", from_date=date(years[0], 1, 1),
                    until_date=date(years[-1], 12, 31), max_results=max_documents - consumed)
                for page in provider.iter_pages(request, context.cancel_event):
                    context.check_cancelled()
                    scanned += page.scanned
                    consumed += page.scanned
                    unresolved += page.skipped
                    for document in page.documents:
                        if document.source != "openalex" or document.publication_year is None:
                            unresolved += 1
                        elif not years[0] <= document.publication_year <= years[-1]:
                            rejected += 1
                        else:
                            # Retain every version, including retraction flags,
                            # so later clean metadata cannot hide a conflict.
                            reference = archive.put(document)
                            if reference.revision_id in references:
                                rejected += 1
                            else:
                                references[reference.revision_id] = reference
                                accepted += 1
                    exhausted = page.exhausted
                    context.progress("antecedents", f"Обзор более ранних источников: просмотрено {consumed}",
                                     min(consumed, max_documents), max_documents)
                    if consumed >= max_documents:
                        limited = not exhausted
                        break
            except CancelledError:
                raise TaskCancelled() from None
            except CredentialUnavailable:
                reasons.append("credential_store_unavailable")
            except BackendError as error:
                reasons.append("source_" + error.code)
            finally:
                if provider is not None:
                    provider.close()
        if not exhausted:
            reasons.append("antecedent_pagination_incomplete")
        if unresolved:
            reasons.append("unresolved_antecedent_records")
        if limited:
            reasons.append("antecedent_scan_budget")
        complete = exhausted and not limited and not unresolved
        state: Literal["complete", "partial", "unavailable"] = "complete" if complete else "partial" if scanned else "unavailable"
        coverage.append(Coverage(source="openalex", purpose="enrichment", query_hash=query_hash,
            state=state, requested_years=years, completed_years=years if complete else (),
            pagination_exhausted=exhausted, comparable=False, scanned_records=scanned,
            accepted_records=accepted, rejected_records=rejected, unresolved_records=unresolved,
            limit_reached=limited, reasons=tuple(dict.fromkeys(reasons)) if not complete else ()))
    snapshot = CorpusSnapshot(snapshot_id=content_hash({"input_hash": identity,
        "revisions": sorted(references), "coverage": [item.model_dump(mode="json") for item in coverage]}), plan_hash=plan.plan_hash,
        purpose="enrichment", created_at=datetime.now(UTC), as_of=plan.as_of,
        documents=tuple(references[key] for key in sorted(references)), coverage=tuple(coverage),
        normalizer_version="backend-v2-antecedents-v1", deduplication_version="doi-source-id-v1")
    matched, conflicts = _derive_matches(candidate, snapshot, archive, years[0], years[-1], context.check_cancelled,
                                         version=ANTECEDENT_VERSION)
    first = min(((doc.publication_year, study) for study, (_, doc) in matched.items()), default=None)
    evidence = []
    # At most 30 earliest real sources / 60 exact extracts. The full matched ID
    # set and immutable archive remain available for broader expert inspection.
    ordered = sorted(matched.values(), key=lambda pair: (pair[1].publication_year or 9999, pair[0].study_id))
    for reference, document in ordered[:30]:
        evidence.append(quote_evidence(reference, document, text_field="title", quote=document.title[:6000]))
        if document.abstract and document.abstract.strip():
            evidence.append(quote_evidence(reference, document, text_field="abstract", quote=document.abstract[:6000].strip()))
    complete_search = all(item.state == "complete" for item in coverage) and not conflicts
    status = "earlier_matches_found" if matched else "none_found_within_queries" if complete_search else "incomplete_search"
    bundle = AntecedentBundle.model_validate(dict(version=ANTECEDENT_VERSION, candidate_id=candidate.candidate_id,
        admission_rule_hash=candidate.admission_rule_hash, first_year=first_year, query_plan=plan, snapshot=snapshot,
        operational_status=status, earliest_observed_year=first[0] if first else None,
        earliest_observed_study_id=first[1] if first else None, matched_study_ids=tuple(sorted(matched)),
        conflicting_study_ids=conflicts, evidence=evidence, limitations=(
            "Первая запись означает первое наблюдение среди найденных источников, а не дату изобретения.",
            "Поиск по замороженным фразам может пропустить более ранние названия и аналоги; отсутствие совпадений не доказывает новизну.",
            "Раннее совпадение названия не доказывает тождество механизмов или промышленную зрелость; требуется содержательная оценка.",
            "Источники до начала обзора и потерянные записи не исследованы; цитаты показаны для первых 30 найденных работ.",
        )))
    verify_antecedents(bundle, candidate, plan, archive, check=context.check_cancelled)
    context.checkpoint(stage, {"input_hash": identity, "bundle": bundle.model_dump(mode="json")})
    return bundle


def collect_antecedents(candidate: Candidate, plan: QueryPlan, archive: DocumentArchive,
                        credentials: CredentialStore, context: RunContext, *, first_year: int = 1900,
                        max_documents: int = 500,
                        provider_factory: Callable[[str], DocumentProvider] | None = None) -> AntecedentBundle:
    """Reuse one OpenAlex connection pool across all frozen synonym queries."""
    if provider_factory is not None:
        return _collect_antecedents(candidate, plan, archive, credentials, context,
                                    first_year=first_year, max_documents=max_documents,
                                    provider_factory=provider_factory)
    with PublicationProviderSession(credentials, provider_builder=make_provider) as session:
        return _collect_antecedents(candidate, plan, archive, credentials, context,
                                    first_year=first_year, max_documents=max_documents,
                                    provider_factory=session.provider)
