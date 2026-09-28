"""Real PDF parsing, worker fencing, resource bounds and rights provenance."""

import json
import os
from threading import Event

import pytest
from pydantic import ValidationError
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.pilot import reports
from app.pilot.reports import ReportImportError, ReportMetadata, ReportRecord, import_report
from app.runtime.credentials import CredentialStore
from app.runtime.worker import WorkerCancelled, WorkerTimeout


def metadata(**changes):
    return ReportMetadata(**(dict(title="Public technology report", source_url="https://example.org/report.pdf",
        publication_year=2025, public_license_allowed=True, license_note="User confirms CC BY 4.0") | changes))


def make_pdf(path, texts=("Observed membrane performance improves selectivity.",), *, compressed=False, encrypted=False):
    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        if text is not None:
            stream = DecodedStreamObject()
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream.set_data(("BT /F1 12 Tf 10 760 Td (" + escaped + ") Tj ET").encode("latin-1"))
            page[NameObject("/Contents")] = writer._add_object(stream.flate_encode() if compressed else stream)
    if encrypted:
        writer.encrypt("test-password")
    writer.write(path)
    writer.close()
    return path


def test_worker_extracts_real_pdf_with_exact_page_offsets_and_no_external_ai(tmp_path):
    path = make_pdf(tmp_path / "report.pdf", ("A real PDF text.", None, "Second text page."))
    record = import_report(path, metadata(), credentials=CredentialStore(), cancel=Event())
    assert isinstance(record, ReportRecord) and record.source == "report" and record.kind == "report"
    assert record.pages_total == 3 and record.pages_with_text == 2 and record.extraction_status == "partial"
    assert not record.external_ai_allowed and record.public_license_allowed
    assert record.full_text[record.page_spans[0].start:record.page_spans[0].end] == "A real PDF text."
    assert record.full_text[record.page_spans[2].start:record.page_spans[2].end] == "Second text page."
    assert record.source_id == record.pdf_sha256 and record.abstract is None
    assert str(path) not in record.model_dump_json()
    assert ReportRecord.model_validate_json(record.model_dump_json()) == record


def test_first_blank_page_preserves_offsets_despite_contract_whitespace_stripping(tmp_path):
    record = import_report(make_pdf(tmp_path / "report.pdf", (None, "Second page.")), metadata(),
        credentials=CredentialStore(), cancel=Event())
    assert record.full_text.startswith("\n\n")
    assert record.full_text[record.page_spans[1].start:record.page_spans[1].end] == "Second page."


def test_image_or_empty_pdf_is_explicitly_unsupported_without_ocr(tmp_path):
    record = import_report(make_pdf(tmp_path / "blank.pdf", (None,)), metadata(),
        credentials=CredentialStore(), cancel=Event())
    assert record.extraction_status == "unsupported_scanned" and record.full_text == ""
    assert record.pages_with_text == 0 and any("OCR" in value for value in record.limitations)


@pytest.mark.parametrize("changes", [{"source_url": "file:///private/secret"},
    {"source_url": "https://user:secret@example.org/report"}, {"publication_year": 2999},
    {"public_license_allowed": "yes"}, {"license_note": ""}, {"title": ""}])
def test_metadata_rejects_missing_rights_metadata_and_unsafe_urls(changes):
    with pytest.raises(ValidationError):
        metadata(**changes)


def test_rights_are_checked_before_even_reading_missing_file(tmp_path):
    with pytest.raises(ReportImportError) as error:
        import_report(tmp_path / "absent.pdf", metadata(public_license_allowed=False),
            credentials=CredentialStore(), cancel=Event())
    assert error.value.code == "rights_required"


def test_input_size_and_regular_file_guards(tmp_path):
    path = tmp_path / "large.pdf"
    with path.open("wb") as stream:
        stream.write(b"%PDF-")
        stream.truncate(reports.MAX_PDF_BYTES + 1)
    with pytest.raises(ReportImportError) as error:
        import_report(path, metadata(), credentials=CredentialStore(), cancel=Event())
    assert error.value.code == "file_limit"
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "pipe.pdf"
        os.mkfifo(fifo)
        with pytest.raises(ReportImportError) as error:
            import_report(fifo, metadata(), credentials=CredentialStore(), cancel=Event())
        assert error.value.code == "file_unavailable"


@pytest.mark.parametrize("mode,code", [("encrypted", "encrypted"), ("pages", "page_limit"),
    ("invalid", "invalid_pdf"), ("compressed_bomb", "decompression_limit")])
def test_unsafe_pdf_forms_are_rejected_in_real_worker(tmp_path, mode, code):
    path = tmp_path / "test.pdf"
    if mode == "encrypted":
        make_pdf(path, encrypted=True)
    elif mode == "pages":
        make_pdf(path, (None,) * 301)
    elif mode == "invalid":
        path.write_bytes(b"%PDF-1.7\n broken document private contents")
    else:
        make_pdf(path, ("A" * (reports.MAX_STREAM_BYTES + 100),), compressed=True)
        assert path.stat().st_size < 20000
    with pytest.raises(ReportImportError) as error:
        import_report(path, metadata(), credentials=CredentialStore(), cancel=Event())
    assert error.value.code == code
    assert "private contents" not in str(error.value)


def test_total_decompression_budget_counts_multiple_streams(monkeypatch, tmp_path):
    path = make_pdf(tmp_path / "several.pdf", ("A" * 1000, "B" * 1000, "C" * 1000), compressed=True)
    monkeypatch.setattr(reports, "MAX_STREAM_BYTES", 2000)
    monkeypatch.setattr(reports, "MAX_DECOMPRESSED_BYTES", 2500)
    output = tmp_path / "result.json"
    reports.extract_report_worker(path, output, Event())
    assert json.loads(output.read_text(encoding="utf-8")) == {"error": "decompression_limit"}


def test_worker_timeout_and_cancellation_do_not_publish_record(tmp_path):
    path = make_pdf(tmp_path / "one.pdf")
    with pytest.raises(WorkerTimeout):
        import_report(path, metadata(), credentials=CredentialStore(), cancel=Event(), timeout_seconds=0.001)
    cancel = Event()
    cancel.set()
    with pytest.raises(WorkerCancelled):
        import_report(path, metadata(), credentials=CredentialStore(), cancel=cancel)


def test_imported_record_rejects_false_extraction_status(tmp_path):
    record = import_report(make_pdf(tmp_path / "report.pdf"), metadata(), credentials=CredentialStore(), cancel=Event())
    with pytest.raises(ValidationError):
        ReportRecord.model_validate(record.model_dump() | {"extraction_status": "unsupported_scanned"})
