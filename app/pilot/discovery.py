"""Universal local candidate discovery, separate from historical confirmation.

No direction profiles or topic dictionaries participate. Every reported count is
a count of deduplicated input studies. HDBSCAN finds semantic groups; NMF is a
deterministic, explicitly identified fallback, never evidence of trend emergence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any
import warnings

from app.backend.contracts import DocumentRecord, normalize_doi
from app.identity import validate_data_dir
from app.pilot.contracts import SCOPE_RULE_VERSION, Candidate, QueryPlan, content_hash
from app.pilot.encoder import (
    MAX_UNITS, Cancellation, EmbeddingCache, MultilingualEncoder,
    checkpoint, normalize_text, validate_vectors,
)
from app.pilot.retractions import LEGACY_STATUS_RULES, STATUS_RULES

DISCOVERY_VERSION = "semantic-discovery-v6-title-phrase-groups"
# One study can arrive as several source revisions, so the revision ceiling is
# twice the largest corpus a plan may request. Real revisions average about
# 11 KB, which keeps a full corpus inside the input limit below.
MAX_REVISIONS = 20_000
MAX_INPUT_BYTES = 300_000_000
MAX_OUTPUT_BYTES = 50_000_000
# The structural walk below bounds work on a file this application writes from
# its own archive. Its ceiling has to follow MAX_REVISIONS, because one archived
# revision is about 450 JSON nodes and at most about 1900: authors, affiliations
# and identifiers are each a node. A fixed 1 000 000 silently contradicted the
# revision ceiling — it refused any corpus past roughly 2 200 documents, so the
# standard and deep profiles (10 000 documents) always failed here and only the
# 400-document fast profile ever completed. Size and depth stay bounded by
# MAX_INPUT_BYTES and the separate nesting limit.
MAX_JSON_NODES = MAX_REVISIONS * 2048
MAX_JSON_DEPTH = 64


class DiscoveryError(ValueError):
    """Actionable failure with no source text, credentials or invented results."""


@dataclass(frozen=True)
class DiscoveryOptions:
    minimum_relevance: float = 0.80
    exclusion_margin: float = 0.03
    minimum_cluster_size: int = 8
    minimum_samples: int = 3
    maximum_candidates: int = 30
    maximum_chunks_per_document: int = 32
    algorithm: str = "auto"

    def __post_init__(self) -> None:
        for value in (self.minimum_relevance, self.exclusion_margin):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise DiscoveryError("Порог релевантности должен быть конечным числом.")
        if not 0 <= self.minimum_relevance <= 1 or not 0 <= self.exclusion_margin <= 0.3:
            raise DiscoveryError("Некорректный порог релевантности.")
        for value, minimum, maximum in ((self.minimum_cluster_size, 2, 100),
                                        (self.minimum_samples, 1, 100),
                                        (self.maximum_candidates, 1, 30),
                                        (self.maximum_chunks_per_document, 1, 64)):
            if type(value) is not int or not minimum <= value <= maximum:
                raise DiscoveryError("Некорректные ограничения обнаружения кандидатов.")
        if self.algorithm not in {"auto", "hdbscan", "nmf"}:
            raise DiscoveryError("Неизвестный алгоритм обнаружения кандидатов.")


def _study_text(record: DocumentRecord) -> str:
    return normalize_text(record.title + ("\n" + record.abstract if record.abstract else ""))


def _revision(record: DocumentRecord, *, revision_id: str | None = None) -> dict[str, Any]:
    return {"revision_id": revision_id if revision_id is not None else content_hash(record.model_dump(mode="json")),
            "source": record.source, "source_id": record.source_id, "url": record.url,
            "document_key": record.document_key,
            "text_hash": hashlib.sha256(_study_text(record).encode("utf-8")).hexdigest(),
            "observed_at": record.fetched_at.isoformat()}


def _deduplicate_identities(documents: list[DocumentRecord], *, cancel: Cancellation | None = None,
                            rules_version: str = LEGACY_STATUS_RULES,
                            revision_ids: dict[int, str] | None = None) -> list[dict[str, Any]]:
    """DOI/source identities first; conservative long-title+year+author crosswalk.

    Different known DOIs are never merged merely because their titles coincide.
    An identical source ID carrying conflicting known DOIs is a data error.
    Latest representatives are selected deterministically, retaining every version.
    """
    if len(documents) > MAX_REVISIONS or any(not isinstance(record, DocumentRecord) for record in documents):
        raise DiscoveryError("Обнаружение принимает не более 20 000 проверенных версий источников.")
    if revision_ids is None:
        revision_ids = {}

    def revision_id(index: int) -> str:
        record = documents[index]
        key = id(record)
        if key not in revision_ids:
            revision_ids[key] = content_hash(record.model_dump(mode="json"))
        return revision_ids[key]

    parents = list(range(len(documents)))
    known_dois = [{record.doi} if record.doi else set() for record in documents]

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def merge(first: int, second: int, *, identity: bool = False) -> None:
        first, second = find(first), find(second)
        if first != second:
            combined = known_dois[first] | known_dois[second]
            if len(combined) > 1:
                if identity:
                    raise DiscoveryError("Один идентификатор источника соответствует разным DOI.")
                return
            parents[max(first, second)] = min(first, second)
            known_dois[min(first, second)] = combined

    source_ids: dict[tuple[str, str], int] = {}
    doi_ids: dict[str, int] = {}
    titles: dict[tuple[str, int, str], list[int]] = {}
    from app.pilot.sources import primary_research_exclusion

    supporting = [primary_research_exclusion(record, rules_version=rules_version) == "supporting_asset_not_research"
                  for record in documents] if rules_version == STATUS_RULES else []
    for index, record in enumerate(documents):
        checkpoint(cancel)
        source_key = record.source, record.source_id
        if source_key in source_ids:
            other = documents[source_ids[source_key]]
            if record.doi and other.doi and record.doi != other.doi:
                raise DiscoveryError("Один идентификатор источника соответствует разным DOI.")
            merge(index, source_ids[source_key], identity=True)
        source_ids[source_key] = index
        if record.doi:
            if record.doi in doi_ids:
                merge(index, doi_ids[record.doi], identity=True)
            doi_ids[record.doi] = index
        title = normalize_text(record.title).casefold()
        if len(title) >= 50 and len(title.split()) >= 8 and record.publication_year and record.authors:
            author = " ".join(record.authors[0].casefold().split())
            titles.setdefault((title, record.publication_year, author), []).append(index)
    # Examine complete groups so a DOI-less record cannot bridge two distinct DOIs.
    for members in titles.values():
        known = {documents[index].doi for index in members if documents[index].doi}
        if len(known) <= 1:
            first_by_unit: dict[bool, int] = {}
            for index in members:
                unit = supporting[index] if rules_version == STATUS_RULES else False
                first = first_by_unit.setdefault(unit, index)
                merge(first, index)
    groups: dict[int, list[int]] = {}
    for index in range(len(documents)):
        groups.setdefault(find(index), []).append(index)
    result = []
    for members in groups.values():
        checkpoint(cancel)
        ordered = sorted(members, key=lambda index: (
            documents[index].fetched_at, len(documents[index].abstract or ""),
            revision_id(index)))
        representative = ordered[-1]
        keys = sorted({documents[index].document_key for index in members})
        canonical = next((key for key in keys if key.startswith("doi:")), keys[0])
        revisions = {}
        for index in members:
            revision = _revision(documents[index], revision_id=revision_id(index))
            revisions[revision["revision_id"]] = revision
        result.append({"study_id": canonical, "representative_index": representative,
                       "input_indices": sorted(members), "revisions": sorted(revisions.values(),
                                                                               key=lambda item: item["revision_id"])})
    return sorted(result, key=lambda item: item["study_id"])


def deduplicate(documents: list[DocumentRecord], *, cancel: Cancellation | None = None,
                version: str = "study-families-v1") -> list[dict[str, Any]]:
    """Versioned identity resolution; v1 discovery archives can retain their original replay."""
    from app.pilot.study_families import CURRENT_FAMILY_VERSION, FAMILY_VERSION, family_edges

    rules_version = STATUS_RULES if version in {DISCOVERY_VERSION, CURRENT_FAMILY_VERSION} else LEGACY_STATUS_RULES
    revision_ids: dict[int, str] = {}
    identities = _deduplicate_identities(documents, cancel=cancel, rules_version=rules_version,
                                        revision_ids=revision_ids)
    if version == "semantic-discovery-v1":
        return identities
    if version not in {FAMILY_VERSION, CURRENT_FAMILY_VERSION, "semantic-discovery-v2",
                       "semantic-discovery-v3", "semantic-discovery-v4-primary-units", DISCOVERY_VERSION}:
        raise DiscoveryError("Неизвестная версия объединения исследований.")
    parents = list(range(len(identities)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    links = []
    for first, second, evidence in family_edges(identities, documents, cancel=cancel,
                                                rules_version=rules_version, revision_ids=revision_ids):
        links.append((first, second, evidence))
        first, second = find(first), find(second)
        parents[max(first, second)] = min(first, second)
    groups: dict[int, list[int]] = {}
    for index in range(len(identities)):
        groups.setdefault(find(index), []).append(index)
    # Index links after all unions: their first endpoint can have changed root
    # since the edge was yielded. Scanning every edge for every family grows
    # quadratically on a corpus with many independently linked study pairs.
    links_by_family: dict[int, list[dict[str, Any]]] = {}
    for first, _, evidence in links:
        links_by_family.setdefault(find(first), []).append(evidence)
    result = []
    for root, members in groups.items():
        checkpoint(cancel)
        indices = sorted({index for member in members for index in identities[member]["input_indices"]})
        # Canonical IDs must name the actual revision used for semantic discovery
        # and evidence: never attach a rich preprint quotation to a journal ID.
        # Prefer journals on a text tie, then a stable key. Observation time only
        # resolves revisions of that same key, not the family's canonical ID.
        representative = min(indices, key=lambda index: (
            -len(documents[index].abstract or ""),
            -int(documents[index].raw_metadata.get("type") in ("article", "journal-article")),
            -int(bool(documents[index].doi)), documents[index].document_key,
            -documents[index].fetched_at.timestamp(),
            revision_ids[id(documents[index])]))
        keys = sorted({documents[index].document_key for index in indices})
        canonical = documents[representative].document_key
        revisions = {revision["revision_id"]: revision for member in members
                     for revision in identities[member]["revisions"]}
        years = [documents[index].publication_year for index in indices if documents[index].publication_year]
        family_evidence = links_by_family.get(root, [])
        if rules_version == STATUS_RULES:
            family_evidence = list({content_hash(evidence): evidence for evidence in family_evidence}.values())
        result.append({"study_id": canonical, "representative_index": representative,
                       "input_indices": indices, "revisions": sorted(revisions.values(), key=lambda item: item["revision_id"]),
                       "identity_keys": keys, "identity_version": (CURRENT_FAMILY_VERSION
                           if rules_version == STATUS_RULES else FAMILY_VERSION),
                       "first_publication_year": min(years) if years else None,
                       "family_links": sorted(family_evidence, key=content_hash)})
    return sorted(result, key=lambda item: item["study_id"])


def _eligible_studies(studies: list[dict[str, Any]], documents: list[DocumentRecord], plan: QueryPlan,
                      cancel: Cancellation | None, *, rules_version: str = STATUS_RULES,
                      ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Withdrawn work stays withdrawn; explicitly linked supplements add evidence, not studies."""
    from app.pilot.retractions import is_explicitly_retracted, retracted_family_keys
    from app.pilot.sources import primary_research_exclusion

    by_id = {study["study_id"]: study for study in studies}
    aliases = {key: study["study_id"] for study in studies for key in study.get("identity_keys", [study["study_id"]])}
    reasons: dict[str, str] = {}
    relations: dict[str, set[str]] = {}
    active_indices = sorted({index for study in studies for index in study["input_indices"]})
    withdrawn = (retracted_family_keys([documents[index] for index in active_indices],
                                      check=lambda: checkpoint(cancel), rules_version=rules_version)
                 if rules_version == STATUS_RULES else {documents[index].document_key for index in active_indices
                     if is_explicitly_retracted(documents[index], rules_version=LEGACY_STATUS_RULES)})
    for study in studies:
        checkpoint(cancel)
        for index in study["input_indices"]:
            record = documents[index]
            if record.document_key in withdrawn:
                reasons[study["study_id"]] = "explicitly_retracted"
            elif (record.publication_date and record.publication_date > plan.as_of
                  or record.publication_year and record.publication_year > plan.as_of.year
                  or record.publication_year == plan.as_of.year and record.publication_month
                  and record.publication_month > plan.as_of.month):
                reasons.setdefault(study["study_id"], "not_publicly_available_as_of")
            metadata = record.raw_metadata.get("relation")
            if not isinstance(metadata, dict) or not record.doi:
                continue
            for kind in ("is-supplement-to", "is-supplemented-by", "is-component-of"):
                if kind == "is-component-of" and record.raw_metadata.get("type") != "component":
                    continue
                references = metadata.get(kind, [])
                if not isinstance(references, list):
                    continue
                for reference in references:
                    if not isinstance(reference, dict) or reference.get("id-type") != "doi":
                        continue
                    reference_id = reference.get("id")
                    if not isinstance(reference_id, str):
                        continue
                    try:
                        target = "doi:" + normalize_doi(reference_id)
                        target = aliases.get(target, target)
                    except ValueError:
                        continue
                    child, parent = (target, study["study_id"]) if kind == "is-supplemented-by" else (
                        study["study_id"], target)
                    if child != parent:
                        relations.setdefault(child, set()).add(parent)
    roots: dict[str, str] = {}
    for study in studies:
        identifier, seen = study["study_id"], set()
        current = identifier
        while current in relations:
            if current in seen or len(relations[current]) != 1:
                reasons[identifier] = "ambiguous_supplement_relation"
                break
            seen.add(current)
            current = next(iter(relations[current]))
        roots[identifier] = current
        if current != identifier and identifier not in reasons:
            if current not in by_id:
                reasons[identifier] = "supplement_parent_not_in_corpus"
            elif current in reasons:
                reasons[identifier] = "supplement_parent_ineligible"
    for study in studies:
        identifier = study["study_id"]
        # An explicitly linked asset may remain inside its parent's evidence
        # family. A standalone dataset/software record never creates a study.
        supporting = [primary_research_exclusion(documents[index], rules_version=rules_version)
                      == "supporting_asset_not_research" for index in study["input_indices"]]
        if roots[identifier] == identifier and (any(supporting) if rules_version == STATUS_RULES else all(supporting)):
            reasons.setdefault(identifier, "supporting_asset_not_research")
        if rules_version == STATUS_RULES and any(primary_research_exclusion(documents[index], rules_version=rules_version)
                                                 == "not_independent_research" for index in study["input_indices"]):
            reasons.setdefault(identifier, "not_independent_research")
    merged: dict[str, dict[str, Any]] = {}
    for study in studies:
        identifier = study["study_id"]
        if identifier in reasons:
            continue
        root = roots[identifier]
        if root in reasons:
            reasons[identifier] = "supplement_parent_ineligible"
            continue
        if root not in merged:
            parent = by_id[root]
            merged[root] = {**parent, "input_indices": [], "revisions": [], "supplementary_study_ids": []}
        merged[root]["input_indices"].extend(study["input_indices"])
        merged[root]["revisions"].extend(study["revisions"])
        if root != identifier:
            merged[root]["supplementary_study_ids"].append(identifier)
    for study in merged.values():
        study["input_indices"].sort()
        study["revisions"].sort(key=lambda item: item["revision_id"])
        study["supplementary_study_ids"].sort()
    excluded = [{"study_id": identifier, "reason": reason} for identifier, reason in sorted(reasons.items())]
    return sorted(merged.values(), key=lambda item: item["study_id"]), excluded


def _matrix(texts: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(lowercase=True, strip_accents=None, stop_words="english",
                                 token_pattern=r"(?u)\b[^\W\d_][^\W\d_'-]{2,}\b", ngram_range=(1, 3),
                                 max_features=12000, min_df=1, sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return None, ()
    return matrix, tuple(vectorizer.get_feature_names_out())


def _nmf_labels(matrix, minimum_size: int, cancel: Cancellation | None):
    import numpy as np
    from sklearn.decomposition import NMF
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits

    if matrix is None or min(matrix.shape) < 2:
        return np.full(0 if matrix is None else matrix.shape[0], -1, dtype=int), {"state": "insufficient_terms"}
    topics = min(24, max(2, int(math.sqrt(matrix.shape[0]))), matrix.shape[0], matrix.shape[1])
    checkpoint(cancel)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        coefficients = NMF(n_components=topics, init="nndsvda", random_state=42,
                           max_iter=400, tol=1e-4).fit_transform(matrix)
    checkpoint(cancel)
    labels = coefficients.argmax(axis=1)
    mass = coefficients.sum(axis=1)
    concentration = coefficients.max(axis=1) / np.maximum(mass, 1e-12)
    labels[(mass <= 0) | (concentration < 0.45)] = -1
    for label in set(labels) - {-1}:
        if int((labels == label).sum()) < minimum_size:
            labels[labels == label] = -1
    return labels, {"state": "completed", "topics_requested": topics,
                    "converged": not any(issubclass(item.category, ConvergenceWarning) for item in caught)}


def _groups(labels) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        if int(label) >= 0:
            groups.setdefault(int(label), []).append(index)
    return groups


def _labels(vectors, matrix, options: DiscoveryOptions, cancel: Cancellation | None):
    import numpy as np
    from sklearn.cluster import HDBSCAN
    from threadpoolctl import threadpool_limits

    hdbscan_labels = np.full(len(vectors), -1, dtype=int)
    diagnostics: dict[str, Any] = {}
    if options.algorithm != "nmf" and len(vectors) >= options.minimum_cluster_size:
        checkpoint(cancel)
        with threadpool_limits(limits=1):
            hdbscan_labels = HDBSCAN(min_cluster_size=options.minimum_cluster_size,
                                     min_samples=min(options.minimum_samples, len(vectors)), metric="euclidean",
                                     n_jobs=1, allow_single_cluster=False, copy=True).fit_predict(vectors)
        checkpoint(cancel)
    diagnostics["hdbscan_clusters"] = len(_groups(hdbscan_labels))
    diagnostics["hdbscan_assigned"] = int((hdbscan_labels >= 0).sum())
    # Baseline is calculated on the same retained corpus for a reviewable comparison.
    nmf_labels, nmf = _nmf_labels(matrix, options.minimum_cluster_size, cancel)
    if len(nmf_labels) != len(vectors):
        nmf_labels = np.full(len(vectors), -1, dtype=int)
    diagnostics["nmf"] = {**nmf, "clusters": len(_groups(nmf_labels)),
                          "assigned": int((nmf_labels >= 0).sum())}
    if options.algorithm == "nmf":
        return nmf_labels, "nmf", diagnostics
    if options.algorithm == "auto" and not _groups(hdbscan_labels) and _groups(nmf_labels):
        return nmf_labels, "nmf_fallback", diagnostics
    return hdbscan_labels, "hdbscan", diagnostics


def _terms(matrix, vocabulary: tuple[str, ...], members: list[int], *, global_mean=None) -> tuple[str, ...]:
    import numpy as np

    if matrix is None or not vocabulary:
        return ()
    inside = np.asarray(matrix[members].mean(axis=0)).ravel()
    if global_mean is None:
        global_mean = np.asarray(matrix.mean(axis=0)).ravel()
    weights = inside / (global_mean + 0.02)
    weights[inside <= 0] = 0
    result: list[str] = []
    # Zero-weight vocabulary items can never be selected. Sorting only terms
    # present in this group preserves the exact weight/name ordering.
    positive = np.flatnonzero(weights > 0)
    for index in sorted(positive, key=lambda item: (-weights[item], vocabulary[item])):
        term = vocabulary[index]
        if any(term in chosen or chosen in term for chosen in result):
            continue
        result.append(term)
        if len(result) == 3:
            break
    return tuple(result)


def discover(documents: list[DocumentRecord], plan: QueryPlan, *, encoder, snapshot_id: str,
             cache_dir: Path | None = None, options: DiscoveryOptions | None = None,
             cancel: Cancellation | None = None, progress=None) -> dict[str, Any]:
    import numpy as np

    if not isinstance(plan, QueryPlan) or not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise DiscoveryError("Нужны проверенный поисковый план и идентификатор снимка.")
    options = options or DiscoveryOptions()
    checkpoint(cancel)
    if len(documents) > MAX_REVISIONS or any(not isinstance(record, DocumentRecord) for record in documents):
        raise DiscoveryError("Обнаружение принимает не более 20 000 проверенных версий источников.")
    # A later version must neither leak future text nor make an earlier public
    # preprint disappear from an as-of corpus. Preserve original input offsets.
    available_indices = [index for index, record in enumerate(documents) if not (
        record.publication_date and record.publication_date > plan.as_of
        or record.publication_year and record.publication_year > plan.as_of.year
        or record.publication_year == plan.as_of.year and record.publication_month
        and record.publication_month > plan.as_of.month)]
    available = [documents[index] for index in available_indices]
    deduplicated = deduplicate(available, cancel=cancel, version=DISCOVERY_VERSION)
    for study in deduplicated:
        study["representative_index"] = available_indices[study["representative_index"]]
        study["input_indices"] = [available_indices[index] for index in study["input_indices"]]
    studies, excluded = _eligible_studies(deduplicated, documents, plan, cancel)
    if len(studies) > plan.limits.discovery_documents:
        raise DiscoveryError("Превышен лимит уникальных исследований поискового плана.")
    from app.pilot.retractions import retraction_status_evidence

    status_evidence = [item | {"revision_id": _revision(record)["revision_id"]} for record in available
                       for item in retraction_status_evidence(record, rules_version=STATUS_RULES)]
    status_evidence = list({content_hash(item): item for item in status_evidence}.values())
    available_set = set(available_indices)
    excluded.extend({"study_id": record.document_key, "reason": "not_publicly_available_as_of"}
                    for index, record in enumerate(documents) if index not in available_set)
    excluded.sort(key=lambda item: (item["study_id"], item["reason"]))
    base: dict[str, Any] = {
        "schema_version": 3, "discovery_version": DISCOVERY_VERSION, "plan_hash": plan.plan_hash,
        "discovery_snapshot_id": snapshot_id, "encoder_fingerprint": encoder.fingerprint,
        "input_records": len(documents), "unique_studies": len(studies), "studies": studies,
        "deduplicated_input_studies": len(deduplicated), "excluded_studies": excluded,
        "publication_status_version": STATUS_RULES, "status_evidence": sorted(status_evidence, key=content_hash),
        "options": asdict(options), "candidates": [], "clusters": [], "relevance": [],
        "early_signal_study_ids": [], "unassigned_study_ids": [], "review_queue": [],
        "review_queue_metadata": [], "hierarchy": [], "retained_studies": 0, "text_units": 0,
        "quality": "insufficient_data", "methodology_calibrated": False,
        "scope_review_required": True,
        "limitations": ["Семантические пороги предварительные и требуют независимой оценки.",
                        "Группы документов ещё не подтверждены историей публикаций.",
                        "Лексические названия требуют проверки конкретности технологии."],
    }
    if not studies:
        return base
    texts: list[str] = []
    slices: list[tuple[int, int]] = []
    for position, study in enumerate(studies):
        checkpoint(cancel)
        if progress:
            # Text splitting and embedding count different things, so each phase
            # reports its own scale rather than inventing a shared percentage.
            progress(position, len(studies))
        record = documents[study["representative_index"]]
        chunked = encoder.chunk_text(_study_text(record), max_chunks=options.maximum_chunks_per_document,
                                      cancel=cancel)
        if not chunked.units:
            raise DiscoveryError("Один из документов не содержит пригодного текста.")
        start = len(texts)
        texts.extend(unit.text for unit in chunked.units)
        if len(texts) > MAX_UNITS:
            raise DiscoveryError("Корпус превышает лимит 100 000 текстовых фрагментов. Уточните область поиска.")
        slices.append((start, len(texts)))
        study.update({"text_units": len(chunked.units), "text_truncated": chunked.truncated,
                      "token_count": chunked.total_tokens})
    base["text_units"] = len(texts)
    if any(study["text_truncated"] for study in studies):
        base["limitations"].append("Часть длинных документов ограничена числом текстовых фрагментов.")
    if cache_dir is None:
        unit_vectors = encoder.encode(texts, kind="passage", cancel=cancel, progress=progress)
        base["embedding_cache_hit"] = False
    else:
        cache = EmbeddingCache(cache_dir, encoder.fingerprint)
        unit_vectors = cache.encode(encoder, texts, kind="passage", cancel=cancel, progress=progress)
        base["embedding_cache_hit"] = cache.last_hit
    if progress:
        progress(len(texts), len(texts))
    validate_vectors(unit_vectors, len(texts), cancel=cancel)
    queries = list(dict.fromkeys([plan.original_query, plan.english_query, *plan.subdirections, *plan.synonyms]))
    query_vectors = encoder.encode(queries, kind="query", cancel=cancel)
    validate_vectors(query_vectors, len(queries), cancel=cancel)
    anchor_count = len(set((plan.original_query, plan.english_query)))
    exclusion_vectors = encoder.encode(list(plan.exclusions), kind="query", cancel=cancel) if plan.exclusions else None
    if exclusion_vectors is not None:
        validate_vectors(exclusion_vectors, len(plan.exclusions), cancel=cancel)
    vectors = np.empty((len(studies), 384), dtype=np.float32)
    retained: list[int] = []
    scores: list[float] = []
    for index, (start, end) in enumerate(slices):
        checkpoint(cancel)
        chunks = unit_vectors[start:end]
        centroid = np.asarray(chunks.mean(axis=0), dtype=np.float32)
        norm = np.linalg.norm(centroid)
        if not np.isfinite(norm) or norm <= 0:
            raise DiscoveryError("Невозможно построить вектор документа.")
        vectors[index] = centroid / norm
        similarities = chunks @ query_vectors.T
        anchor = float(similarities[:, :anchor_count].max())
        direction = float(similarities.max())
        score = max(-1., min(1., 0.8 * anchor + 0.2 * direction))
        reason = "retained" if score >= options.minimum_relevance else "below_semantic_threshold"
        excluded_score = None
        if exclusion_vectors is not None:
            excluded_score = float((chunks @ exclusion_vectors.T).max())
            if excluded_score >= score + options.exclusion_margin:
                reason = "closer_to_excluded_scope"
        base["relevance"].append({"study_id": studies[index]["study_id"], "score": score,
                                  "exclusion_score": excluded_score, "decision": reason})
        scores.append(score)
        if reason == "retained":
            retained.append(index)
    base["retained_studies"] = len(retained)
    if not retained:
        base["limitations"].append("Нет документов, прошедших предварительный порог релевантности.")
        return base
    validate_vectors(vectors, len(studies), cancel=cancel)
    matrix, vocabulary = _matrix([_study_text(documents[studies[index]["representative_index"]])
                                  for index in retained])
    labels, algorithm, diagnostics = _labels(vectors[retained], matrix, options, cancel)
    base.update({"clustering_algorithm": algorithm, "clustering_diagnostics": diagnostics,
                 "quality": "partial"})
    from app.pilot.evidence import scope_is_anchored
    from app.pilot.hierarchy import (balanced_groups, refine_groups, study_hypotheses, study_review_priority,
                                     title_phrase_groups)

    groups = _groups(labels)
    refined, hierarchy = refine_groups(list(groups.values()), vectors[retained], cancel=cancel)
    retained_records = [documents[studies[index]["representative_index"]] for index in retained]
    phrase_groups = title_phrase_groups(
        [record.title for record in retained_records],
        # Scope anchoring reads only the document in each pair; this stage has
        # source records but has not built archive revision references yet.
        [scope_is_anchored(((None, record),), plan, rule_version=SCOPE_RULE_VERSION)
         for record in retained_records],
        scope_names=(plan.original_query, plan.english_query, *plan.subdirections, *plan.synonyms),
        existing=[leaf["members"] for leaf in refined], cancel=cancel)
    for group in phrase_groups:
        group.update(node_id=len(hierarchy), parent_node_id=None, depth=0, origin="title_phrase_group")
        hierarchy.append({**group, "split": False})
    base["hierarchy"] = [{**{key: value for key, value in node.items() if key != "members"},
                           "study_ids": [studies[retained[index]]["study_id"] for index in node["members"]]}
                          for node in hierarchy]
    ranked = balanced_groups([*refined, *phrase_groups],
                             score=lambda group: float(np.mean([scores[retained[index]] for index in group["members"]])),
                             key=lambda group: tuple(studies[retained[index]]["study_id"] for index in group["members"]))
    # A density assignment and a rejected binary split are not evidence that a
    # paper contains no distinct mechanism. Preserve the same individual route
    # for cluster members and noise, without granting either a scientific label.
    hypotheses = study_hypotheses(refined, len(retained))
    for group in hypotheses:
        record = documents[studies[retained[group["members"][0]]]["representative_index"]]
        group["literature_synthesis"] = bool(study_review_priority(record))
    hypotheses.sort(key=lambda group: (group["literature_synthesis"],
                                       -scores[retained[group["members"][0]]],
                                       studies[retained[group["members"][0]]]["study_id"]))
    included: set[int] = set()
    global_term_mean = None
    for ordinal, group in enumerate([*ranked, *hypotheses]):
        checkpoint(cancel)
        members = group["members"]
        global_members = [retained[index] for index in members]
        member_ids = tuple(sorted(studies[index]["study_id"] for index in global_members))
        centroid = vectors[global_members].mean(axis=0)
        cluster_norm = float(np.linalg.norm(centroid))
        if cluster_norm <= 1e-12:
            base["limitations"].append("Противоположные смысловые векторы не образуют пригодного кандидата.")
            continue
        centroid = np.asarray(centroid / cluster_norm, dtype=np.float32)
        terms: tuple[str, ...] = ()
        if len(members) > 1:
            # Every candidate compares with the same retained corpus; reduce
            # the sparse matrix only once, and only if a group needs terms.
            if global_term_mean is None and matrix is not None:
                global_term_mean = np.asarray(matrix.mean(axis=0)).ravel()
            terms = _terms(matrix, vocabulary, members, global_mean=global_term_mean)
        label = " / ".join(terms)[:200] if terms else "Группа технологических исследований"
        if group.get("phrase"):
            # The shared title phrase is what defines this group; it is also
            # the name a lexical fallback can admit for every member.
            label = group["phrase"][:200]
        if len(members) == 1:
            label = documents[studies[global_members[0]]["representative_index"]].title[:200]
        candidate_id = "candidate-" + content_hash({"plan_hash": plan.plan_hash, "members": member_ids})
        rule = {"version": DISCOVERY_VERSION, "encoder": encoder.fingerprint, "options": asdict(options),
                "algorithm": algorithm, "centroid": centroid.tolist(), "members": member_ids}
        candidate = Candidate(candidate_id=candidate_id, plan_hash=plan.plan_hash, label=label,
                              definition="Непроверенная исследовательская гипотеза. "
                                         "Технологический механизм, релевантность, новизна и стадия требуют проверки.",
                              admission_rule_version=DISCOVERY_VERSION, admission_rule_hash=content_hash(rule),
                              discovery_snapshot_id=snapshot_id, discovery_study_ids=member_ids,
                              specificity="uncertain", scope_rule_version=SCOPE_RULE_VERSION)
        origin = group.get("origin", "hierarchical_cluster" if group["depth"] else "semantic_cluster")
        metadata = {"candidate_id": candidate_id, "origin": origin,
                    "parent_node_id": group["parent_node_id"], "node_id": group["node_id"],
                    "mean_relevance": float(np.mean([scores[index] for index in global_members])),
                    "unique_study_count": len(member_ids), "review_required": True}
        if "literature_synthesis" in group:
            metadata["literature_synthesis"] = group["literature_synthesis"]
        if ordinal >= options.maximum_candidates or origin in {"unclustered_study", "cluster_member_study"}:
            base["review_queue"].append(candidate.model_dump(mode="json"))
            base["review_queue_metadata"].append(metadata)
            continue
        base["candidates"].append(candidate.model_dump(mode="json"))
        base["clusters"].append({**metadata, "study_ids": list(member_ids),
                                 "centroid": centroid.tolist(), "terms": list(terms), "algorithm": algorithm,
                                 "coherence": float(np.clip(np.mean(vectors[global_members] @ centroid), -1, 1)),
                                 "source_count": len({revision["source"] for index in global_members
                                                      for revision in studies[index]["revisions"]})})
        included.update(global_members)
    # Keep the old archive key for readers; the neutral name reflects its meaning.
    unassigned = [studies[index]["study_id"] for index in retained if index not in included]
    base["early_signal_study_ids"] = unassigned
    base["unassigned_study_ids"] = unassigned
    base["clustering_diagnostics"]["review_hypotheses"] = len(base["candidates"]) + len(base["review_queue"])
    if base["review_queue"]:
        base["limitations"].append("Непроверенные малые группы и отдельные работы сохранены в очереди проверки; это не подтверждённые тренды.")
    if not base["candidates"]:
        base["limitations"].append("Устойчивые группы не выделены; сохранена очередь непроверенных исследовательских гипотез.")
    if not diagnostics["nmf"].get("converged", True):
        base["limitations"].append("Сравнительный NMF достиг лимита итераций; его группы требуют проверки.")
    checkpoint(cancel)
    return base


def task(input_path: Path, output_path: Path, cancel_event) -> None:
    """Spawn worker entry: immutable JSON input, atomic JSON output, no DB or keys."""
    checkpoint(cancel_event)
    input_path, output_path = Path(input_path), Path(output_path)
    validate_data_dir(output_path.parent)
    if input_path.resolve() == output_path.resolve():
        raise DiscoveryError("Входной снимок нельзя заменять результатом расчёта.")
    def reject_constant(value):
        raise DiscoveryError("Вход содержит нечисловую JSON-константу.")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DiscoveryError("Вход содержит повторяющиеся ключи JSON.")
            result[key] = value
        return result

    try:
        with input_path.open("rb") as source:
            encoded_input = source.read(MAX_INPUT_BYTES + 1)
        if len(encoded_input) > MAX_INPUT_BYTES:
            raise DiscoveryError("Вход обнаружения превышает лимит 300 МБ.")
        payload = json.loads(encoded_input, parse_constant=reject_constant, object_pairs_hook=unique_object)
        if not isinstance(payload, dict) or set(payload) - {
                "query_plan", "documents", "discovery_snapshot_id", "model_dir", "cache_dir", "options"}:
            raise DiscoveryError("Неизвестный формат входа обнаружения.")
        if not isinstance(payload["documents"], list) or len(payload["documents"]) > MAX_REVISIONS:
            raise DiscoveryError("Неверный список документов обнаружения.")
        # Iterator stack is O(depth), rather than copying every member of a huge metadata list.
        pending = [iter([payload])]
        nodes = 0
        while pending:
            try:
                value = next(pending[-1])
            except StopIteration:
                pending.pop()
                continue
            nodes += 1
            if nodes > MAX_JSON_NODES or len(pending) > MAX_JSON_DEPTH:
                raise DiscoveryError("Превышен лимит размера или вложенности структуры JSON.")
            if nodes % 1024 == 0:
                checkpoint(cancel_event)
            if isinstance(value, dict):
                pending.append(iter(value.values()))
            elif isinstance(value, list):
                pending.append(iter(value))
            elif isinstance(value, float) and not math.isfinite(value):
                raise DiscoveryError("Вход содержит нечисловое JSON-значение.")
        checkpoint(cancel_event)
        plan = QueryPlan.model_validate(payload["query_plan"])
        documents = [DocumentRecord.model_validate(record) for record in payload["documents"]]
        options = DiscoveryOptions(**payload.get("options", {}))
        model_dir = Path(payload["model_dir"]) if payload.get("model_dir") else None
        cache_dir = Path(payload["cache_dir"]) if payload.get("cache_dir") else None
        encoder = MultilingualEncoder(model_dir, cancel=cancel_event)
        from app.runtime.worker import report_progress

        result = discover(documents, plan, encoder=encoder, snapshot_id=payload["discovery_snapshot_id"],
                          cache_dir=cache_dir, options=options, cancel=cancel_event,
                          progress=report_progress)
    except (KeyError, TypeError, RecursionError, UnicodeError) as error:
        raise DiscoveryError("Повреждён входной JSON-снимок обнаружения.") from error
    checkpoint(cancel_event)
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise DiscoveryError("Результат обнаружения превышает лимит 50 МБ.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".discovery-", dir=output_path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        checkpoint(cancel_event)
        os.replace(temporary, output_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
