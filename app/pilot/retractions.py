"""Conservative publication-status screening shared by discovery and evidence.

Only explicit status markers and a publisher's opening self-declaration count.
A paper discussing retractions, citing a retracted article, or reporting tissue
retraction is not itself retracted. This is status observed in this revision;
it is never a claim that a withdrawal was known at publication time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from html import unescape
import re
from typing import Any
import unicodedata

from app.backend.contracts import DocumentRecord, normalize_doi

LEGACY_STATUS_RULES = "publication-status/1.0.0"
STATUS_RULES = "publication-status/2.0.0"


def validate_status_rules(rules_version: str) -> None:
    if rules_version not in {LEGACY_STATUS_RULES, STATUS_RULES}:
        raise ValueError("Unknown publication-status rules")

_STATUS = r"(?:retracted|withdrawn)"
_PREFIX = re.compile(
    rf"^(?:[\[(]\s*{_STATUS}\s*[\])]\s*|{_STATUS}(?:\s+(?:article|paper))?\s*[:\-–—]\s*"
    r"|retraction(?:\s+(?:notice|note))?\s*:\s*)", re.I)
_SUFFIX = re.compile(rf"\s*[\[(]\s*{_STATUS}\s*[\])]\s*$", re.I)
_DECLARATION = re.compile(
    rf"^(?:abstract\s*:\s*)?(?:this|the\s+present)\s+(?:article|paper|manuscript|preprint|study)\s+"
    rf"(?:has\s+been|was|is)\s+(?:(?:formally|officially)\s+)?{_STATUS}"
    r"(?=\s*(?:[.!;]|$|(?:by|due\s+to|at\s+the\s+request|because|from|on)\b))", re.I)
_STATUS_TYPES = frozenset({"retraction", "retraction-notice", "retracted-article", "withdrawal"})


def _plain(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", unescape(re.sub(r"<[^>]+>", " ", text))).split())


def _is_explicitly_retracted_v1(document: DocumentRecord) -> bool:
    """Reject explicit withdrawal evidence, not arbitrary keyword occurrences."""
    if document.raw_metadata.get("is_retracted") is True:
        return True
    kinds = (document.document_type, document.raw_metadata.get("type"))
    if any(isinstance(kind, str) and kind.casefold() in _STATUS_TYPES for kind in kinds):
        return True
    title = _plain(document.title)
    if _PREFIX.search(title) or _SUFFIX.search(title):
        return True
    # Crossref can preserve its JATS abstract even when a normalized record lacks
    # an abstract. Do not search arbitrary metadata, quotations, or references.
    abstracts = (document.abstract, document.raw_metadata.get("abstract"))
    return any(isinstance(value, str) and bool(_DECLARATION.search(_plain(value))) for value in abstracts)


def retraction_status_evidence(document: DocumentRecord, *,
                               rules_version: str = STATUS_RULES) -> tuple[dict[str, str], ...]:
    """Observed status links, separate from identity links and authored content.

    Crossref update-to describes the publication being updated by this notice.
    References, relation citations and free text about retraction are not status.
    """
    validate_status_rules(rules_version)
    evidence = []
    if _is_explicitly_retracted_v1(document):
        evidence.append({"kind": "explicit_publication_status", "source_key": document.document_key,
                         "target_key": document.document_key, "source": document.source,
                         "source_id": document.source_id})
    updates = document.raw_metadata.get("update-to")
    if rules_version == STATUS_RULES and isinstance(updates, list):
        for index, update in enumerate(updates):
            if not isinstance(update, dict):
                continue
            kind = update.get("type", update.get("label"))
            if not isinstance(kind, str) or kind.casefold() not in _STATUS_TYPES:
                continue
            identifier = update.get("DOI")
            if not isinstance(identifier, str):
                continue
            try:
                target = "doi:" + normalize_doi(identifier)
            except ValueError:
                continue
            evidence.append({"kind": "publisher_retraction_notice", "source_key": document.document_key,
                             "target_key": target, "source": document.source, "source_id": document.source_id,
                             "metadata_path": f"update-to[{index}]", "status": kind.casefold()})
    return tuple(sorted(evidence, key=lambda item: tuple(sorted(item.items()))))


def is_explicitly_retracted(document: DocumentRecord, *, rules_version: str = STATUS_RULES) -> bool:
    """Reject observed withdrawal evidence under explicitly replayable rules."""
    return bool(retraction_status_evidence(document, rules_version=rules_version))


def retracted_family_keys(documents: Sequence[DocumentRecord], *,
                          check: Callable[[], None] | None = None,
                          rules_version: str = STATUS_RULES) -> set[str]:
    """Propagate observed withdrawal only along the existing strong study links.

    All received versions must be included, even rejected records, otherwise a
    clean journal copy can survive when its withdrawn preprint was discarded
    before family resolution. Citations/reviews/supplements are not identity links.
    """
    from app.pilot.study_families import family_edges

    validate_status_rules(rules_version)
    groups: dict[str, list[int]] = {}
    withdrawn_keys = set()
    for index, document in enumerate(documents):
        if check:
            check()
        groups.setdefault(document.document_key, []).append(index)
        status = retraction_status_evidence(document, rules_version=rules_version)
        if status:
            withdrawn_keys.add(document.document_key)
            withdrawn_keys.update(item["target_key"] for item in status)
    if not withdrawn_keys:
        return set()
    identities: list[dict[str, Any]] = [{"study_id": key, "input_indices": indices} for key, indices in groups.items()]
    parents = list(range(len(identities)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    class CancellationCheck:
        def is_set(self) -> bool:
            if check:
                check()
            return False

    if rules_version == STATUS_RULES:
        # A DOI may appear only in a later revision of the same source record.
        # The stable provider ID still identifies that withdrawn manifestation.
        source_ids: dict[tuple[str, str], int] = {}
        for position, identity in enumerate(identities):
            for index in identity["input_indices"]:
                document = documents[index]
                source_id = document.source, document.source_id
                if source_id in source_ids:
                    first, second = find(position), find(source_ids[source_id])
                    parents[max(first, second)] = min(first, second)
                source_ids[source_id] = position
    for first, second, _ in family_edges(identities, list(documents), cancel=CancellationCheck(),
                                        rules_version=rules_version):
        first, second = find(first), find(second)
        parents[max(first, second)] = min(first, second)
    blocked = {find(index) for index, identity in enumerate(identities) if identity["study_id"] in withdrawn_keys}
    return withdrawn_keys | {identity["study_id"] for index, identity in enumerate(identities) if find(index) in blocked}
