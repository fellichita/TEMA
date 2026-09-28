"""Explicit local report/preprint imports and exact supplementary evidence.

These materials broaden evidence, but never alter publication growth denominators.
Text matches are candidates for review, not proof of relevance or performance.
"""

from datetime import date, datetime, UTC
from pathlib import Path
from threading import Event
from typing import Literal

from pydantic import Field

from app.pilot.archive import DocumentArchive
from app.pilot.contracts import Candidate, Contract, DocumentRevisionRef
from app.pilot.evidence import archived_field, normalize_title, quote_evidence, title_matches
from app.pilot.library import MAX_IMPORTS, catalogue_paths, publish_catalogue_artifact, read_artifact
from app.runtime.jobs import TaskCancelled, TaskFailure

MAX_MATCH_TEXT_BYTES = 20 * 1024 * 1024
MAX_MATCH_DOCUMENTS = 5000


class SupplementalImport(Contract):
    version: Literal[1] = 1
    kind: Literal["arxiv", "report"]
    created_at: datetime
    documents: tuple[DocumentRevisionRef, ...] = Field(max_length=1000)
    limitations: tuple[str, ...] = Field(default=(), max_length=100)


class SupplementalLibrary:
    def __init__(self, data_dir: Path, archive: DocumentArchive):
        self.directory = data_dir / "supplemental"
        self.archive = archive

    def _save(self, kind, documents, limitations=(), *, cancel):
        if len(catalogue_paths(self.directory)) >= MAX_IMPORTS:
            raise TaskFailure("Достигнут лимит 1000 импортов дополнительных материалов.")
        references = []
        for document in documents:
            if cancel.is_set():
                raise TaskCancelled()
            references.append(self.archive.put(document))
        if cancel.is_set():
            raise TaskCancelled()
        record = SupplementalImport(kind=kind, created_at=datetime.now(UTC),
                                    documents=tuple(references), limitations=limitations)
        digest = publish_catalogue_artifact(self.directory, record.model_dump(mode="json"), cancel=cancel)
        return {"id": digest, "kind": kind, "documents": len(record.documents), "limitations": list(record.limitations)}

    def import_report(self, path, metadata, *, credentials, cancel):
        from app.pilot.reports import ReportMetadata, import_report

        report = import_report(Path(path), ReportMetadata.model_validate(metadata), credentials=credentials, cancel=cancel)
        saved = self._save("report", (report,), report.limitations, cancel=cancel)
        return saved | {"status": report.extraction_status, "pages": report.pages_total, "pages_with_text": report.pages_with_text}

    def import_arxiv(self, path, *, cancel):
        from app.pilot.enrichment import import_arxiv_atom

        imported = import_arxiv_atom(Path(path), as_of=date.today(), cancel=cancel)
        return self._save("arxiv", imported.documents, imported.limitations, cancel=cancel)

    def entries(self, *, cancel=None):
        """Yield one entry at a time; cancel is a check_cancelled context."""
        if cancel is not None:
            cancel.check_cancelled()
        for path in sorted(catalogue_paths(self.directory)):
            if cancel is not None:
                cancel.check_cancelled()
            yield path.stem, SupplementalImport.model_validate(read_artifact(self.directory, path.stem))

    def list_rows(self):
        """Legacy complete listing; desktop callers use the bounded list_page."""
        rows = []
        for identifier, entry in self.entries():
            titles = [self.archive.get(ref.revision_id).title for ref in entry.documents[:3]]
            rows.append({"id": identifier, "kind": entry.kind, "created_at": entry.created_at.isoformat(),
                         "documents": len(entry.documents), "title": "; ".join(titles), "limitations": list(entry.limitations)})
        return rows

    def list_page(self, offset: int = 0, limit: int = 50, *, cancel: Event | None = None) -> dict:
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("Некорректная страница дополнительных материалов.")

        def check():
            if cancel is not None and cancel.is_set():
                raise TaskCancelled()

        check()
        paths = sorted(catalogue_paths(self.directory))
        rows = []
        for path in paths[offset:offset + limit]:
            check()
            entry = SupplementalImport.model_validate(read_artifact(self.directory, path.stem))
            check()
            titles = []
            for reference in entry.documents[:3]:
                check()
                titles.append(self.archive.get(reference.revision_id).title)
            check()
            rows.append({"id": path.stem, "kind": entry.kind, "created_at": entry.created_at.isoformat(),
                         "documents": len(entry.documents), "title": "; ".join(titles), "limitations": list(entry.limitations)})
        check()
        return {"items": rows, "total": len(paths), "offset": offset, "limit": limit}

    def match(self, candidate: Candidate, context, *, limit=30):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Допустимо от 1 до 100 совпадений дополнительных материалов.")
        context.check_cancelled()
        phrases = candidate.synonyms or (candidate.label,)
        matches: list[dict] = []
        seen = set()
        scanned = 0
        text_bytes = 0
        for _, entry in self.entries(cancel=context):
            for reference in entry.documents:
                context.check_cancelled()
                if reference.revision_id in seen:
                    continue
                seen.add(reference.revision_id)
                if scanned >= MAX_MATCH_DOCUMENTS:
                    return {"items": matches, "limited": True, "scanned": scanned,
                            "text_bytes": text_bytes, "limit_reason": "document_limit"}
                document = self.archive.get(reference.revision_id)
                scanned += 1
                document_bytes = sum(len(text.encode("utf-8")) for text in
                    (document.title, document.abstract or "", getattr(document, "full_text", "")))
                if text_bytes + document_bytes > MAX_MATCH_TEXT_BYTES:
                    return {"items": matches, "limited": True, "scanned": scanned,
                            "text_bytes": text_bytes, "limit_reason": "text_limit"}
                text_bytes += document_bytes
                fields: tuple[Literal["title", "abstract", "full_text"], ...] = (
                    ("title", "abstract", "full_text") if entry.kind == "report" else ("title", "abstract"))
                for field in fields:
                    text = archived_field(document, field)
                    if not title_matches(text, phrases, candidate.exclusions):
                        continue
                    # Bound each exact quote; preserve original bytes/Unicode offsets.
                    # A paragraph is used only if it contains a full frozen phrase.
                    paragraphs = text.splitlines() if field == "full_text" else [text]
                    quote = next((part for part in paragraphs if part.strip() and len(part) <= 12000
                                  and title_matches(part, phrases, candidate.exclusions)), None)
                    if quote is None:
                        # Long paragraph: use a matching sentence, never invent a snippet.
                        import re
                        quote = next((part for part in re.split(r"(?<=[.!?])\s+", text)
                                      if 0 < len(part) <= 12000 and title_matches(part, phrases, candidate.exclusions)), None)
                    if quote is None or not normalize_title(quote):
                        continue
                    evidence = quote_evidence(reference, document, text_field=field, quote=quote)
                    pages = [page.page for page in getattr(document, "page_spans", ())
                             if page.start < evidence.end and page.end > evidence.start] if field == "full_text" else []
                    matches.append({"title": document.title, "kind": entry.kind, "pages": pages,
                                    "evidence": evidence.model_dump(mode="json"),
                                    "notice": "Совпадение с формулировкой кандидата. Предметную релевантность проверяет аналитик."})
                    break
                if len(matches) >= limit:
                    return {"items": matches, "limited": True, "scanned": scanned,
                            "text_bytes": text_bytes, "limit_reason": "result_limit"}
        return {"items": matches, "limited": False, "scanned": scanned,
                "text_bytes": text_bytes, "limit_reason": None}
