"""Source-first coordinated arXiv imports retaining every admissible revision."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from threading import Event

from app.pilot.archive import DocumentArchive
from app.pilot.contracts import CorpusSnapshot, Coverage, QueryPlan, content_hash
from app.pilot.enrichment import (MAX_ARXIV_RECORDS, _arxiv_date, _arxiv_document,
                                  read_arxiv_feed)
from app.pilot.multisource.contracts import (ArxivImportReceipt, ArxivVersion, ExportRight, QueryProfile,
                                             SourceSnapshot, validate_import_export_right)
from app.pilot.multisource.store import SignalStore
from app.pilot.query import normalize_query
from app.runtime.jobs import TaskCancelled, TaskFailure

_ARXIV_COMMENT = "{http://arxiv.org/schemas/atom}comment"


def import_arxiv_discovery(store: SignalStore, archive: DocumentArchive, path: Path, profile_hash: str, *,
                           as_of: date, retention: str, observed_at: datetime | None = None,
                           cancel: Event | None = None, export_right: ExportRight = "local_only",
                           license_ref: str | None = None) -> str:
    """Archive all safe versions, then select one per family for later discovery."""
    if as_of > max(date.today(), datetime.now(timezone.utc).date()) or as_of.year < 1991:
        raise TaskFailure("Некорректная дата среза arXiv.")
    validate_import_export_right(export_right, license_ref)
    store.get_object(profile_hash, QueryProfile)
    timestamp = observed_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise TaskFailure("Дата импорта должна включать часовой пояс.")
    cancel_event = cancel if cancel is not None else Event()
    expected_hash, _ = read_arxiv_feed(path, cancel=cancel_event)
    raw_hash = store.put_raw(path, "xml", retention=retention, cancel=cancel_event)
    if raw_hash != expected_hash:
        raise TaskFailure("Экспорт arXiv изменился во время импорта.")
    verified = store.verify_raw(raw_hash, "xml", cancel=cancel_event)
    _, entries = read_arxiv_feed(verified, cancel=cancel_event)
    truncated = len(entries) > MAX_ARXIV_RECORDS
    entries = entries[:MAX_ARXIV_RECORDS]
    accepted: list[tuple[ArxivVersion, str]] = []
    rejected = 0
    for entry in entries:
        if cancel_event.is_set():
            raise TaskCancelled()
        try:
            document, version = _arxiv_document(entry, as_of, raw_hash)
            published = _arxiv_date(str(document.raw_metadata["published"]))
            updated = _arxiv_date(str(document.raw_metadata["updated"]))
            comment = entry.findtext(_ARXIV_COMMENT, "").strip()[:500]
            withdrawn = comment.casefold().startswith("withdrawn")
            stable_document = document.model_copy(update={"fetched_at": timestamp})
            reference = archive.put(stable_document)
            item = ArxivVersion(arxiv_id=document.source_id, version=version,
                                revision_id=reference.revision_id, raw_hash=raw_hash,
                                published_at=published, updated_at=updated,
                                status="withdrawn_reported" if withdrawn else "preprint_unreviewed")
            accepted.append((item, store.put_object(item, cancel=cancel_event)))
        except (ValueError, TypeError, OverflowError):
            rejected += 1
    groups: dict[str, list[ArxivVersion]] = defaultdict(list)
    for item, _ in accepted:
        groups[item.arxiv_id].append(item)
    selected = []
    for versions in groups.values():
        # Conflicting first-publication dates or two different texts under one
        # arXiv version make the whole family unsuitable for automatic discovery.
        by_number: dict[int, set[str]] = defaultdict(set)
        for item in versions:
            by_number[item.version].add(item.revision_id)
        if len({item.published_at for item in versions}) != 1 or any(len(values) != 1 for values in by_number.values()):
            continue
        latest = max(versions, key=lambda item: (item.version, item.updated_at))
        if latest.status != "withdrawn_reported":
            selected.append(latest.revision_id)
    snapshot = SourceSnapshot(source="arxiv", adapter_version="coordinated-arxiv-source-first/1",
                              request_hash=content_hash({"kind": "arxiv-atom", "query_profile_hash": profile_hash,
                                                         "as_of": as_of.isoformat(), "raw_hash": raw_hash}),
                              query_profile_hash=profile_hash, observed_at=timestamp, available_at=timestamp,
                              coverage="partial", comparable=False, raw_hash=raw_hash,
                              retention="local_allowed", export_right=export_right, license_ref=license_ref,
                              limitations=("Локальная Atom-выгрузка не подтверждает полноту поиска arXiv.",
                                           "Препринты не считаются рецензируемыми статьями или трендом."))
    snapshot_hash = store.put_object(snapshot, cancel=cancel_event)
    receipt = ArxivImportReceipt(query_profile_hash=profile_hash, snapshot_hash=snapshot_hash,
                                 raw_hash=raw_hash, as_of=as_of, scanned=len(entries), rejected=rejected,
                                 truncated=truncated, version_hashes=tuple(digest for _, digest in accepted),
                                 selected_revision_ids=tuple(sorted(selected)), completed_at=timestamp)
    store.verify_raw(raw_hash, "xml", cancel=cancel_event)
    return store.put_object(receipt, cancel=cancel_event)


def build_arxiv_corpus_snapshot(store: SignalStore, archive: DocumentArchive, receipt_hash: str,
                                plan: QueryPlan, *, cancel: Event | None = None) -> CorpusSnapshot:
    """Re-select actual frozen revisions at the scientific plan's cutoff."""
    receipt = store.get_object(receipt_hash, ArxivImportReceipt)
    profile = store.get_object(receipt.query_profile_hash, QueryProfile)
    snapshot = store.get_object(receipt.snapshot_hash, SourceSnapshot)
    if (snapshot.source != "arxiv" or snapshot.raw_hash != receipt.raw_hash
            or snapshot.query_profile_hash != receipt.query_profile_hash
            or normalize_query(plan.original_query).casefold() != profile.original_query.casefold()):
        raise TaskFailure("Экспорт arXiv относится к другой области анализа.")
    store.verify_raw(receipt.raw_hash, "xml", cancel=cancel)
    grouped: dict[str, list[ArxivVersion]] = defaultdict(list)
    for digest in receipt.version_hashes:
        if cancel is not None and cancel.is_set():
            raise TaskCancelled()
        item = store.get_object(digest, ArxivVersion)
        if item.raw_hash != receipt.raw_hash:
            raise TaskFailure("Версия arXiv относится к другому экспорту.")
        if item.updated_at.date() <= plan.as_of and item.published_at.date() <= plan.as_of:
            grouped[item.arxiv_id].append(item)
    references = []
    for versions in grouped.values():
        if len({item.published_at for item in versions}) != 1:
            continue
        by_number: dict[int, set[str]] = defaultdict(set)
        for item in versions:
            by_number[item.version].add(item.revision_id)
        if any(len(values) != 1 for values in by_number.values()):
            continue
        latest = max(versions, key=lambda item: (item.version, item.updated_at))
        if latest.status == "withdrawn_reported":
            continue
        document = archive.get(latest.revision_id)
        if document.source != "arxiv" or document.source_id != latest.arxiv_id or (
                document.raw_metadata.get("version") != latest.version):
            raise TaskFailure("Архив arXiv не соответствует версии источника.")
        references.append(archive.put(document))
    coverage = Coverage(source="arxiv", purpose="discovery", query_hash=snapshot.request_hash,
                        state="partial", requested_years=plan.completed_years, completed_years=(),
                        pagination_exhausted=False, comparable=False, scanned_records=receipt.scanned,
                        accepted_records=len(references), rejected_records=receipt.scanned - len(references),
                        reasons=("coordinated_atom_export_is_not_exhaustive",))
    return CorpusSnapshot(snapshot_id="arxiv/" + receipt_hash[:20] + "/" + plan.plan_hash[:20],
                          plan_hash=plan.plan_hash, purpose="discovery", created_at=receipt.completed_at,
                          as_of=plan.as_of, documents=tuple(sorted(references, key=lambda item: item.revision_id)),
                          coverage=(coverage,), normalizer_version="coordinated-arxiv-source-first/1",
                          deduplication_version="doi-source-id-v1")


def merge_arxiv_seed(base: CorpusSnapshot, seed: CorpusSnapshot) -> CorpusSnapshot:
    """Add archived preprints as discovery inputs without changing OA histories."""
    if (base.plan_hash != seed.plan_hash or base.as_of != seed.as_of or base.purpose != "discovery"
            or seed.purpose != "discovery" or len(seed.coverage) != 1 or seed.coverage[0].source != "arxiv"):
        raise TaskFailure("Препринты относятся к другой научной выборке.")
    references = {item.revision_id: item for item in base.documents}
    for item in seed.documents:
        previous = references.get(item.revision_id)
        if previous is not None and previous != item:
            raise TaskFailure("Конфликт арXiv-ревизии с научной выборкой.")
        references[item.revision_id] = item
    included = tuple(sorted(references.values(), key=lambda item: item.revision_id))
    coverage = (*base.coverage, seed.coverage[0])
    return CorpusSnapshot(snapshot_id=content_hash({"base": base.snapshot_hash, "seed": seed.snapshot_hash,
                                                     "revisions": [item.revision_id for item in included]}),
                          plan_hash=base.plan_hash, purpose="discovery", created_at=base.created_at,
                          as_of=base.as_of, documents=included, coverage=coverage,
                          normalizer_version=base.normalizer_version + "+arxiv-source-first/1",
                          deduplication_version=base.deduplication_version)
