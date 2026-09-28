"""Local, explicitly licensed PDF imports; no URL fetching and no implicit AI use.

Untrusted PDF parsing runs in the existing killable worker. pypdf 6.18.0
decompression limits apply before allocation, with an aggregate decoded-stream
budget as well as page, form-invocation, output and wall-clock limits. This is
resource isolation, not an operating-system filesystem sandbox or OCR engine.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from datetime import datetime, UTC
from pathlib import Path
from typing import Annotated, BinaryIO, Final, Literal, Self

from pydantic import Field, StringConstraints, field_validator, model_validator

from app.backend.contracts import Contract, DocumentRecord
from app.input_safety import is_safe_http_url
from app.runtime.credentials import CredentialStore
from app.runtime.worker import Cancellation, WorkerError, run_in_process

MAX_PDF_BYTES = 25_000_000
MAX_PAGES = 300
MAX_STREAM_BYTES = 8_000_000
MAX_DECOMPRESSED_BYTES = 32_000_000
MAX_TEXT_BYTES = 1_000_000
EXTRACTOR_VERSION: Final = "pypdf-6.18.0-bounded-text/1.0.0"


class ReportImportError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class ReportMetadata(Contract):
    title: str = Field(min_length=2, max_length=2000)
    source_url: str = Field(max_length=8000)
    publication_year: int = Field(ge=1000, le=9999, strict=True)
    public_license_allowed: bool = Field(strict=True)
    license_note: str = Field(min_length=3, max_length=2000)
    external_ai_allowed: bool = Field(default=False, strict=True)

    @field_validator("source_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        if not is_safe_http_url(value):
            raise ValueError("Нужна HTTP(S)-ссылка на публичный источник без пароля в адресе.")
        return value

    @field_validator("publication_year")
    @classmethod
    def validate_year(cls, value: int) -> int:
        if value > datetime.now(UTC).year:
            raise ValueError("Год публикации не может быть будущим.")
        return value


class ReportPage(Contract):
    page: int = Field(ge=1, le=MAX_PAGES, strict=True)
    start: int = Field(ge=0, strict=True)
    end: int = Field(ge=0, strict=True)


class ReportRecord(DocumentRecord):
    source: Literal["report"] = "report"
    document_type: Literal["report"] = "report"
    kind: Literal["report"] = "report"
    full_text: Annotated[str, StringConstraints(strip_whitespace=False)] = Field(max_length=MAX_TEXT_BYTES)
    page_spans: tuple[ReportPage, ...] = Field(max_length=MAX_PAGES)
    pages_total: int = Field(ge=1, le=MAX_PAGES, strict=True)
    pages_with_text: int = Field(ge=0, le=MAX_PAGES, strict=True)
    extraction_status: Literal["text", "partial", "unsupported_scanned"]
    pdf_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_version: Literal["pypdf-6.18.0-bounded-text/1.0.0"] = EXTRACTOR_VERSION
    public_license_allowed: bool = Field(strict=True)
    license_note: str = Field(min_length=3, max_length=2000)
    external_ai_allowed: bool = Field(default=False, strict=True)
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def report_consistency(self) -> Self:
        if not self.public_license_allowed or self.source_id != self.pdf_sha256:
            raise ValueError("Отчёт требует подтверждения прав и идентификатора исходного PDF.")
        if self.doi or self.patent_publication or self.patent_family_id:
            raise ValueError("Отчёты учитываются отдельно от научных работ и патентов.")
        if self.publication_year is None or self.date_precision != "year":
            raise ValueError("Необходимо указать год публикации отчёта.")
        if len(self.full_text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ValueError("Извлечённый текст превышает ограничение.")
        if len(self.page_spans) != self.pages_total or self.pages_with_text > self.pages_total:
            raise ValueError("Некорректное число страниц отчёта.")
        last_end = 0
        nonempty = 0
        for number, page in enumerate(self.page_spans, 1):
            expected_start = 0 if number == 1 else last_end + 2
            if (page.page != number or page.start != expected_start
                    or not page.start <= page.end <= len(self.full_text)
                    or (number > 1 and self.full_text[last_end:page.start] != "\n\n")):
                raise ValueError("Некорректные границы страниц отчёта.")
            nonempty += bool(self.full_text[page.start:page.end].strip())
            last_end = page.end
        if last_end != len(self.full_text):
            raise ValueError("Текст отчёта должен полностью соответствовать страницам.")
        expected = "text" if nonempty == self.pages_total else "partial" if nonempty else "unsupported_scanned"
        if nonempty != self.pages_with_text or self.extraction_status != expected:
            raise ValueError("Статус извлечения не соответствует тексту страниц.")
        return self


def _check_cancel(cancel: Cancellation) -> None:
    if cancel.is_set():
        from app.runtime.worker import WorkerCancelled
        raise WorkerCancelled()


def open_local_regular(path: Path) -> BinaryIO:
    """Avoid blocking on a pipe/device if a selected local path is replaced."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("A regular file is required")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _extract_pdf(input_path: Path, cancel: Cancellation) -> dict[str, object]:
    """Worker-only parser. Never call this synchronously from the application UI."""
    from pypdf import PdfReader, apply_configuration, filters
    from pypdf.errors import LimitReachedError
    from pypdf.generic import StreamObject

    used = 0
    original_decode = filters.decode_stream_data

    def bounded_decode(stream: StreamObject) -> bytes:
        nonlocal used
        _check_cancel(cancel)
        remaining = min(MAX_STREAM_BYTES, MAX_DECOMPRESSED_BYTES - used)
        if remaining <= 0:
            raise LimitReachedError("Total stream budget")
        with apply_configuration(zlib_maximum_output_length=remaining,
                lzw_maximum_output_length=remaining, run_length_maximum_output_length=remaining,
                jbig2_maximum_output_length=remaining, array_based_stream_maximum_output_length=remaining):
            decoded = original_decode(stream)
        used += len(decoded)
        if len(decoded) > remaining:
            raise LimitReachedError("Total stream budget")
        return decoded

    # The pinned library resolves this symbol at each encoded-stream read.
    # This worker-local adapter counts fonts/forms/object streams as well as
    # page contents; checking only page.get_contents() would miss those streams.
    filters.decode_stream_data = bounded_decode
    try:
        with apply_configuration(maximum_declared_stream_length=MAX_STREAM_BYTES,
                array_based_stream_maximum_output_length=MAX_STREAM_BYTES,
                zlib_maximum_output_length=MAX_STREAM_BYTES, lzw_maximum_output_length=MAX_STREAM_BYTES,
                run_length_maximum_output_length=MAX_STREAM_BYTES, jbig2_maximum_output_length=MAX_STREAM_BYTES,
                zlib_maximum_recovery_input_length=1_000_000, flate_maximum_columns=100_000,
                flate_maximum_row_length=1_000_000, image_maximum_buffer_size=MAX_STREAM_BYTES,
                xmp_maximum_input_length=100_000, xmp_maximum_element_count=1000,
                outline_maximum_entries=1000, outline_maximum_depth=30,
                page_tree_maximum_entries=1200, page_tree_maximum_depth=30,
                xform_maximum_invocations_per_extraction=100, jbig2dec_binary=None):
            reader = PdfReader(input_path, strict=True)
            if reader.is_encrypted:
                return {"error": "encrypted"}
            count = len(reader.pages)
            if not 1 <= count <= MAX_PAGES:
                return {"error": "page_limit"}
            parts: list[str] = []
            spans: list[dict[str, int]] = []
            characters = total_bytes = nonempty = 0
            for number, page in enumerate(reader.pages, 1):
                _check_cancel(cancel)
                text = page.extract_text(extraction_mode="plain")
                text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).strip()
                if number > 1:
                    parts.append("\n\n")
                    characters += 2
                    total_bytes += 2
                total_bytes += len(text.encode("utf-8"))
                if total_bytes > MAX_TEXT_BYTES:
                    return {"error": "text_limit"}
                spans.append({"page": number, "start": characters, "end": characters + len(text)})
                characters += len(text)
                parts.append(text)
                nonempty += bool(text)
            return {"text": "".join(parts), "pages": spans, "pages_total": count,
                    "pages_with_text": nonempty, "decoded_stream_bytes": used}
    finally:
        filters.decode_stream_data = original_decode


def extract_report_worker(input_path: Path, output_path: Path, cancel: Cancellation) -> None:
    """Trusted entry point for run_in_process; output is a bounded JSON object."""
    from pypdf.errors import LimitReachedError, PyPdfError

    try:
        result = _extract_pdf(input_path, cancel)
    except LimitReachedError:
        result = {"error": "decompression_limit"}
    except (PyPdfError, ValueError, TypeError, KeyError, IndexError, RecursionError, OverflowError):
        result = {"error": "invalid_pdf"}
    _check_cancel(cancel)
    output_path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")


_ERRORS = {
    "encrypted": "PDF защищён шифрованием. Импортируйте разрешённую незашифрованную копию.",
    "page_limit": "Допустим отчёт от 1 до 300 страниц.",
    "text_limit": "Извлечённый текст отчёта превышает 1 МБ. Импортируйте выбранный раздел отдельным PDF.",
    "decompression_limit": "PDF превышает безопасный бюджет распаковки или сложность документа.",
    "invalid_pdf": "Структура PDF повреждена или не поддерживается; текст не импортирован.",
}


def import_report(path: Path, metadata: ReportMetadata, *, credentials: CredentialStore,
                  cancel: Cancellation, timeout_seconds: float = 45.0) -> ReportRecord:
    """Import a local public report off the UI thread, with no remote transfer."""
    metadata = ReportMetadata.model_validate(metadata.model_dump())
    if not metadata.public_license_allowed:
        raise ReportImportError("rights_required", "Подтвердите публичность отчёта и право на локальное извлечение текста.")
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 120:
        raise ValueError("Ограничение времени должно быть от 0 до 120 секунд.")
    _check_cancel(cancel)
    with tempfile.TemporaryDirectory(prefix="trend-report-") as temporary:
        staged = Path(temporary) / "input.pdf"
        output = Path(temporary) / "extraction.json"
        digest = hashlib.sha256()
        size = 0
        try:
            # Read once into our private immutable worker input, avoiding a
            # second unbounded read if a selected file changes during import.
            with open_local_regular(path) as source, staged.open("xb") as target:
                header = source.read(5)
                if header != b"%PDF-":
                    raise ReportImportError("invalid_pdf", "Выбранный файл не является PDF.")
                digest.update(header)
                target.write(header)
                size = len(header)
                while chunk := source.read(256_000):
                    _check_cancel(cancel)
                    size += len(chunk)
                    if size > MAX_PDF_BYTES:
                        raise ReportImportError("file_limit", "Размер PDF не должен превышать 25 МБ.")
                    digest.update(chunk)
                    target.write(chunk)
        except OSError:
            raise ReportImportError("file_unavailable", "Не удалось прочитать выбранный локальный PDF.") from None
        try:
            run_in_process(extract_report_worker, staged, output, cancel, credentials=credentials,
                timeout_seconds=timeout_seconds, max_input_bytes=MAX_PDF_BYTES,
                max_output_bytes=2_200_000)
        except WorkerError:
            raise
        payload = json.loads(output.read_text(encoding="utf-8"))
        if "error" in payload:
            code = payload["error"]
            raise ReportImportError(code, _ERRORS.get(code, _ERRORS["invalid_pdf"]))
        nonempty, count = payload["pages_with_text"], payload["pages_total"]
        status = "text" if nonempty == count else "partial" if nonempty else "unsupported_scanned"
        limitations = ["Публичность, год и право на импорт указаны пользователем; приложение не подтверждает лицензию.",
                      "Текст PDF может терять порядок колонок и таблиц; числовые заявления требуют проверки страницы."]
        if status != "text":
            limitations.append("На части или всех страницах нет извлекаемого текста: возможны сканы, графики или пустые страницы. OCR не выполнялся.")
        return ReportRecord.model_validate(dict(source="report", source_id=digest.hexdigest(),
            title=metadata.title, url=metadata.source_url, publication_year=metadata.publication_year,
            date_precision="year", document_type="report", full_text=payload["text"],
            page_spans=payload["pages"], pages_total=count, pages_with_text=nonempty,
            extraction_status=status, pdf_sha256=digest.hexdigest(),
            public_license_allowed=True, license_note=metadata.license_note,
            external_ai_allowed=metadata.external_ai_allowed, limitations=limitations,
            raw_metadata={"decoded_stream_bytes": payload["decoded_stream_bytes"], "pdf_bytes": size}))
