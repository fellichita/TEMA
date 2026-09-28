"""Conservative, explainable links between manifestations of the same study.

This is identity resolution, not semantic duplicate detection. Similar titles,
common authors, citations and conference extensions alone are not identity links.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from app.backend.contracts import DocumentRecord, normalize_doi
from app.pilot.retractions import LEGACY_STATUS_RULES, STATUS_RULES, validate_status_rules

FAMILY_VERSION = "study-families-v1"
CURRENT_FAMILY_VERSION = "study-families-v2-publication-units"
_VERSION_RELATIONS = frozenset({"is-preprint-of", "has-preprint", "is-version-of", "has-version",
                                "is-identical-to", "is-manuscript-of", "has-manuscript"})
_PREPRINT_TYPES = frozenset({"preprint", "posted-content", "submitted-version", "accepted-version"})


def _text(value: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", value).casefold()))


def _authors(record: DocumentRecord) -> set[str]:
    return {normalized for author in record.authors if (normalized := _text(author))}


def _preprint(record: DocumentRecord) -> bool:
    metadata_type = record.raw_metadata.get("type")
    return (record.document_type in _PREPRINT_TYPES
            or isinstance(metadata_type, str) and metadata_type in _PREPRINT_TYPES)


def version_references(record: DocumentRecord) -> list[tuple[str, str]]:
    relation = record.raw_metadata.get("relation")
    if not isinstance(relation, dict):
        return []
    result = []
    for kind in sorted(_VERSION_RELATIONS):
        references = relation.get(kind, [])
        if not isinstance(references, list):
            continue
        for reference in references:
            if not isinstance(reference, dict) or reference.get("id-type") != "doi":
                continue
            identifier = reference.get("id")
            if not isinstance(identifier, str):
                continue
            try:
                result.append(("doi:" + normalize_doi(identifier), kind))
            except ValueError:
                continue
    return result


def family_edges(studies: list[dict[str, Any]], documents: list[DocumentRecord], *, cancel=None,
                  rules_version: str = LEGACY_STATUS_RULES,
                  revision_ids: Mapping[int, str] | None = None):
    """Yield strong version links with the evidence needed to audit each union."""
    from app.pilot.encoder import checkpoint
    from app.pilot.contracts import content_hash
    from app.pilot.sources import primary_research_exclusion

    validate_status_rules(rules_version)
    if rules_version == STATUS_RULES:
        revision_ids = (revision_ids if revision_ids is not None else
                        {id(record): content_hash(record.model_dump(mode="json")) for record in documents})
    else:
        revision_ids = {}
    # Dataset versions can link to each other, but do not turn into a paper
    # because a repository labels all deposited objects as posted-content.
    supporting = {position: any(primary_research_exclusion(documents[index], rules_version=rules_version)
                                 == "supporting_asset_not_research" for index in study["input_indices"])
                  for position, study in enumerate(studies)} if rules_version == STATUS_RULES else {}
    author_sets: dict[int, set[str]] = {}
    preprint_flags: dict[int, bool] = {}

    def authors(record: DocumentRecord) -> set[str]:
        key = id(record)
        if key not in author_sets:
            author_sets[key] = _authors(record)
        return author_sets[key]

    def is_preprint(record: DocumentRecord) -> bool:
        key = id(record)
        if key not in preprint_flags:
            preprint_flags[key] = _preprint(record)
        return preprint_flags[key]

    def compatible(first: int, second: int) -> bool:
        return rules_version == LEGACY_STATUS_RULES or supporting[first] == supporting[second]

    identifiers = {documents[index].document_key: position for position, study in enumerate(studies)
                   for index in study["input_indices"]}
    titles: dict[str, list[tuple[int, DocumentRecord]]] = {}
    seen_title_records = set()
    for position, study in enumerate(studies):
        checkpoint(cancel)
        indices = (sorted(study["input_indices"], key=lambda index: revision_ids[id(documents[index])])
                   if rules_version == STATUS_RULES else study["input_indices"])
        for index in indices:
            record = documents[index]
            for identifier, relation in version_references(record):
                if (identifier in identifiers and identifiers[identifier] != position
                        and compatible(position, identifiers[identifier])):
                    yield position, identifiers[identifier], {
                        "reason": "explicit_version_relation", "relation": relation,
                        "source_key": record.document_key, "target_key": identifier,
                        **({"source_revision_id": revision_ids[id(record)]}
                           if rules_version == STATUS_RULES else {})}
            title = _text(record.title)
            signature = (position, title, tuple(sorted(authors(record))), record.publication_year,
                         is_preprint(record))
            if len(title) >= 50 and len(title.split()) >= 8 and record.authors and signature not in seen_title_records:
                titles.setdefault(title, []).append((position, record))
                seen_title_records.add(signature)
    for members in titles.values():
        checkpoint(cancel)
        journal_identities = {record.document_key for _, record in members if record.doi and not is_preprint(record)}
        # At least one record must explicitly be a preprint/manuscript, with an
        # exact substantive title and >=2 matching authors (or identical sole author).
        # Two journal articles with different DOIs are not merged by this fallback.
        for offset, (first, record) in enumerate(members):
            for second, other in members[offset + 1:]:
                if first == second or not compatible(first, second) or not (is_preprint(record) or is_preprint(other)):
                    continue
                # An ambiguous preprint must not bridge two different known
                # journal DOIs using title/author similarity. Explicit links above
                # remain authoritative and reviewable.
                if len(journal_identities) > 1 and not (is_preprint(record) and is_preprint(other)):
                    continue
                if not record.publication_year or not other.publication_year:
                    continue
                if abs(record.publication_year - other.publication_year) > 3:
                    continue
                first_authors, second_authors = authors(record), authors(other)
                common = first_authors & second_authors
                if len(common) < 2 and not (first_authors == second_authors and len(first_authors) == 1):
                    continue
                source_record, target_record = (record, other)
                if rules_version == STATUS_RULES and record.document_key > other.document_key:
                    source_record, target_record = other, record
                yield first, second, {"reason": "exact_title_author_preprint_crosswalk",
                                      "source_key": source_record.document_key, "target_key": target_record.document_key,
                                      "matching_author_count": len(common),
                                      **({"source_revision_id": revision_ids[id(source_record)],
                                          "target_revision_id": revision_ids[id(target_record)]}
                                         if rules_version == STATUS_RULES else {})}
