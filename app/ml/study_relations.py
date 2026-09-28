"""Conservative, offline links between publications and supplementary records."""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from app.input_safety import MAX_TITLE_CHARACTERS
from app.ml.corpus import checkpoint, normalize_snapshot_doi
from app.ml.text import title_key

SOURCE_RELATIONS_PATH = Path(__file__).with_name("source_relations.json")


def source_relations_fingerprint():
    return hashlib.sha256(SOURCE_RELATIONS_PATH.read_bytes()).hexdigest()


def author_name_key(value):
    """Remove only a trailing numeric repository identifier, not name suffixes."""
    value = (value or "").rstrip()
    opening = value.rfind("(")
    if opening >= 0 and value.endswith(")"):
        identifier = value[opening + 1:-1].strip()
        if identifier and identifier.isascii() and identifier.isdecimal():
            value = value[:opening].rstrip()
    return title_key(value)


def first_author_surname(authors):
    return tuple(author_name_key(authors[0]).split()[-1:]) if authors else ()


def acs_supplement_parent_candidate(value):
    doi = _doi(value)
    match = re.fullmatch(r"(10\.1021/[^\s]+)\.s[0-9]{3}", doi or "")
    return match.group(1) if match else None


def _doi(value):
    if not isinstance(value, str):
        return None
    try:
        return normalize_snapshot_doi(value)
    except ValueError:
        return None


def _legacy_signature(document):
    title = document.get("title") or ""
    authors = document.get("authors") or []
    year = document.get("publication_year")
    if len(title) > MAX_TITLE_CHARACTERS or not authors or type(year) is not int:
        return None
    key, author = title_key(title), author_name_key(authors[0])
    if len(key.split()) < 6 or not any(char.isalpha() for char in author):
        return None
    return key, author, year


def supplement_relations(entries, end_year, cancel=None):
    """Return unambiguous DOI links supported by metadata or corroborated ACS records.

    Crossref relation names are is-supplement-to / is-supplemented-by, plus
    is-component-of only for explicitly typed component metadata.
    Legacy ACS .sNNN is only a candidate: a present parent must also match the
    exact long title, full normalized first author, and publication year.
    A suffix alone never creates a relation or an invented parent.
    """
    groups = defaultdict(list)
    for entry in entries:
        checkpoint(cancel)
        document = entry["document"]
        year = document.get("publication_year")
        doi = _doi(document.get("doi"))
        if doi and not (type(year) is int and year > end_year):
            groups[doi].append(document)

    reviews = json.loads(SOURCE_RELATIONS_PATH.read_text(encoding="utf-8"))["reviews"]
    claims = defaultdict(lambda: defaultdict(set))
    for doi, documents in sorted(groups.items()):
        checkpoint(cancel)
        records = [(document.get("raw_metadata") or {}, "source_metadata") for document in documents]
        review = reviews.get(doi)
        if review and review["registered_from_year"] <= end_year:
            records.append((review["metadata"], "verified_source_metadata"))
        for metadata, origin in records:
            relations = metadata.get("relation") if isinstance(metadata, dict) else None
            if not isinstance(relations, dict):
                continue
            kinds = ["is-supplement-to", "is-supplemented-by"]
            if metadata.get("type") == "component":
                kinds.append("is-component-of")
            for kind in kinds:
                references = relations.get(kind)
                if not isinstance(references, list):
                    continue
                for reference in references:
                    if not isinstance(reference, dict) or reference.get("id-type") != "doi":
                        continue
                    target = _doi(reference.get("id"))
                    if not target or target == doi:
                        continue
                    child, parent = (target, doi) if kind == "is-supplemented-by" else (doi, target)
                    if child in groups:
                        claims[child][parent].add(origin + ":" + kind)

    signatures_by_doi = {}

    def signatures(doi):
        if doi not in signatures_by_doi:
            signatures_by_doi[doi] = {_legacy_signature(document) for document in groups[doi]} - {None}
        return signatures_by_doi[doi]

    for doi, _documents in sorted(groups.items()):
        checkpoint(cancel)
        parent = acs_supplement_parent_candidate(doi)
        if parent is None or doi in claims:
            continue
        if parent not in groups:
            continue
        if signatures(doi) & signatures(parent):
            claims[doi][parent].add("acs_doi_exact_title_author_year")

    # Resolve each node once. Storing/replaying every ancestor chain per child
    # makes a valid long chain quadratic in time and output size.
    resolved = {}
    for child in sorted(claims):
        checkpoint(cancel)
        path, seen, current = [], set(), child
        while current in claims and current not in resolved:
            checkpoint(cancel)
            if current in seen:
                resolved[current] = (None, "cyclic_supplement_relation")
                break
            seen.add(current)
            path.append(current)
            parents = claims[current]
            if len(parents) != 1:
                resolved[current] = (None, "conflicting_supplement_parents")
                break
            current = next(iter(parents))
        outcome = resolved.get(current, (current, None))
        for node in path:
            resolved[node] = outcome

    links, conflicts = {}, {}
    for child, (root, problem) in sorted(resolved.items()):
        if problem:
            conflicts["doi:" + child] = problem
        else:
            direct_parent, evidence = next(iter(claims[child].items()))
            links["doi:" + child] = {"parent_id": "doi:" + root,
                                     "direct_parent_id": "doi:" + direct_parent,
                                     "evidence": sorted(evidence)}
    return links, conflicts


def supplement_descendants(relations, identities):
    """Follow parent→material only; total work is linear in the relation graph."""
    children = defaultdict(set)
    for child, link in relations.items():
        children[link["direct_parent_id"]].add(child)
    found, pending = set(identities), list(identities)
    while pending:
        for child in children.get(pending.pop(), ()):
            if child not in found:
                found.add(child)
                pending.append(child)
    return found - set(identities)
