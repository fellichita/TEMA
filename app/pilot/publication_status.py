"""Small, immutable proof contexts for publication status across saved snapshots.

No archive mutation or external lookup occurs here. A missing status observation
is unknown; the helper only propagates explicit archived status/identity links.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
import hashlib
from typing import Protocol

from app.backend.contracts import DocumentRecord
from app.pilot.archive import RevisionSource, document_text
from app.pilot.contracts import DocumentRevisionRef, Evidence, content_hash, verify_evidence_text
from app.pilot.retractions import STATUS_RULES, retracted_family_keys, retraction_status_evidence
from app.pilot.sources import supporting_asset_status

MAX_STATUS_REVISIONS = 100000


class StatusCheck(Protocol):
    def check_cancelled(self) -> None: ...


@dataclass(frozen=True)
class StatusContext:
    references: tuple[DocumentRevisionRef, ...]
    withdrawn: frozenset[str]
    supporting: frozenset[str]


def _canonical_reference(document: DocumentRecord) -> DocumentRevisionRef:
    return DocumentRevisionRef.model_validate(dict(
        revision_id=content_hash(document), study_id=document.document_key,
        source=document.source, source_id=document.source_id,
        text_hash=hashlib.sha256(document_text(document).encode("utf-8")).hexdigest(),
        observed_at=document.fetched_at, publication_year=document.publication_year,
        publicly_available_at=document.publication_date))


def collect_status_context(references: Iterable[DocumentRevisionRef], archive: RevisionSource,
                           context: StatusCheck, *, as_of: date,
                           evidence: Iterable[Evidence] = ()) -> StatusContext:
    """Validate and retain the complete material dependency closure, in stable order.

    Every revision of a blocked identity/family is retained, including a clean
    alias bridge needed to replay withdrawal propagation. Parent records proving
    reverse supplement relations are retained even though the parent is research.
    Unrelated clean history is not copied into each assessment.
    """
    # Local import avoids a module cycle when extraction calls this helper.
    from app.pilot.evidence import EvidenceError, archived_field

    if type(as_of) is not date:
        raise EvidenceError("Publication status requires an explicit as_of date")
    documents: dict[str, DocumentRecord] = {}
    canonical: dict[str, DocumentRevisionRef] = {}

    def load(revision_id: str) -> tuple[DocumentRecord, DocumentRevisionRef]:
        context.check_cancelled()
        if revision_id not in documents:
            if len(documents) >= MAX_STATUS_REVISIONS:
                raise EvidenceError("Publication status exceeds 100000 unique revisions")
            document = archive.get(revision_id)
            expected = _canonical_reference(document)
            if expected.revision_id != revision_id:
                raise EvidenceError("Publication status revision hash differs from its archived document")
            if (document.publication_year is not None and document.publication_year > as_of.year
                    or document.publication_date is not None and document.publication_date > as_of
                    or document.publication_year == as_of.year and document.publication_month is not None
                    and document.publication_month > as_of.month):
                raise EvidenceError("Publication status source is published after as_of")
            documents[revision_id], canonical[revision_id] = document, expected
        return documents[revision_id], canonical[revision_id]

    for raw_reference in references:
        context.check_cancelled()
        reference = DocumentRevisionRef.model_validate(raw_reference.model_dump(mode="python")
            if isinstance(raw_reference, DocumentRevisionRef) else raw_reference)
        _, expected = load(reference.revision_id)
        if reference != expected:
            raise EvidenceError("Publication status reference differs from its immutable source")
    for raw_evidence in evidence:
        context.check_cancelled()
        item = Evidence.model_validate(raw_evidence.model_dump(mode="python")
            if isinstance(raw_evidence, Evidence) else raw_evidence)
        document, reference = load(item.revision_id)
        if (item.study_id != reference.study_id or item.source != reference.source
                or item.source_url != document.url or item.retrieved_at != document.fetched_at):
            raise EvidenceError("Publication status evidence refers to a different archived source")
        try:
            verify_evidence_text(item, archived_field(document, item.text_field))
        except ValueError:
            raise EvidenceError("Publication status evidence quotation differs from its archived field") from None
    ordered_ids = sorted(documents)
    ordered = [documents[key] for key in ordered_ids]
    withdrawn = frozenset(retracted_family_keys(ordered, rules_version=STATUS_RULES,
                                               check=context.check_cancelled))
    supporting_keys, supporting_material = supporting_asset_status(ordered, rules_version=STATUS_RULES,
                                                                   check=context.check_cancelled)
    supporting = frozenset(supporting_keys)
    blocked = withdrawn | supporting
    material = []
    for index, revision_id in enumerate(ordered_ids):
        document = documents[revision_id]
        context.check_cancelled()
        if (document.document_key in blocked
                or retraction_status_evidence(document, rules_version=STATUS_RULES)
                or index in supporting_material):
            material.append(canonical[revision_id])
    context.check_cancelled()
    return StatusContext(tuple(material), withdrawn, supporting)
