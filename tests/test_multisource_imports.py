"""CSV import boundary: explicit mapping, encoding, limits and preview."""

from pathlib import Path
from threading import Event

import pytest

from app.pilot.multisource.imports import read_csv_document
from app.runtime.jobs import TaskCancelled, TaskFailure


def test_utf8_bom_semicolon_preview_preserves_original_cells(tmp_path: Path) -> None:
    path = tmp_path / "wordstat.csv"
    path.write_bytes("Месяц;Запросов;Доля\n08.2026;1 234;0,25%\n".encode("utf-8-sig"))
    document = read_csv_document(path, encoding="utf-8-sig", delimiter=";")
    assert document.preview() == {"headers": ("Месяц", "Запросов", "Доля"),
                                  "rows": (("08.2026", "1 234", "0,25%"),),
                                  "row_count": 1, "encoding": "utf-8-sig", "delimiter": ";"}
    assert document.rows[0].line == 2


def test_cp1251_is_explicit_and_wrong_encoding_fails(tmp_path: Path) -> None:
    path = tmp_path / "wordstat.csv"
    path.write_bytes("Месяц;Запросов\n08.2026;2\n".encode("cp1251"))
    assert read_csv_document(path, encoding="cp1251", delimiter=";").headers[0] == "Месяц"
    with pytest.raises(TaskFailure):
        read_csv_document(path, encoding="utf-8-sig", delimiter=";")


def test_unknown_mapping_and_duplicate_headers_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("month;month\n08.2026;1\n", encoding="utf-8")
    with pytest.raises(TaskFailure):
        read_csv_document(path, encoding="utf-8-sig", delimiter=";")
    path.write_text("month;count\n08.2026;1\n", encoding="utf-8")
    with pytest.raises(TaskFailure):
        read_csv_document(path, encoding="utf-8-sig", delimiter=",").column("count")
    with pytest.raises(TaskFailure):
        read_csv_document(path, encoding="utf-8-sig", delimiter=";").column("inferred count")


def test_cancel_before_open_and_ragged_rows_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "wordstat.csv"
    path.write_text("month;count\n08.2026;1;extra\n", encoding="utf-8")
    cancelled = Event()
    cancelled.set()
    with pytest.raises(TaskCancelled):
        read_csv_document(path, encoding="utf-8-sig", delimiter=";", cancel=cancelled)
    with pytest.raises(TaskFailure):
        read_csv_document(path, encoding="utf-8-sig", delimiter=";")
