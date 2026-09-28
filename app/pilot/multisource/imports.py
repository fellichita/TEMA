"""Explicit, bounded CSV mapping shared by the source adapters.

No column, locale, phrase or normalization unit is inferred from a filename.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Literal

from app.pilot.reports import open_local_regular
from app.runtime.jobs import TaskCancelled, TaskFailure

MAX_CSV_ROWS = 10_000
MAX_CSV_COLUMNS = 50
MAX_CSV_CELL_CHARS = 4_096
MAX_CSV_TOTAL_CHARS = 8_000_000
CsvEncoding = Literal["utf-8-sig", "cp1251"]
CsvDelimiter = Literal[";", ",", "\t"]


@dataclass(frozen=True)
class CsvRow:
    line: int
    cells: tuple[str, ...]


@dataclass(frozen=True)
class CsvDocument:
    headers: tuple[str, ...]
    rows: tuple[CsvRow, ...]
    encoding: CsvEncoding
    delimiter: CsvDelimiter

    def column(self, name: str) -> int:
        if name not in self.headers:
            raise TaskFailure(f"В CSV нет выбранной колонки: {name[:80]}")
        return self.headers.index(name)

    def preview(self, limit: int = 5) -> dict:
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("Invalid preview size")
        return {"headers": self.headers, "rows": tuple(row.cells for row in self.rows[:limit]),
                "row_count": len(self.rows), "encoding": self.encoding, "delimiter": self.delimiter}


def read_csv_document(path: Path, *, encoding: CsvEncoding, delimiter: CsvDelimiter,
                      cancel: Event | None = None) -> CsvDocument:
    if encoding not in {"utf-8-sig", "cp1251"} or delimiter not in {";", ",", "\t"}:
        raise TaskFailure("Выберите кодировку и разделитель CSV явно.")
    if cancel is not None and cancel.is_set():
        raise TaskCancelled()
    try:
        with open_local_regular(path) as raw:
            import io

            with io.TextIOWrapper(raw, encoding=encoding, errors="strict", newline="") as text:
                reader = csv.reader(text, delimiter=delimiter, strict=True)
                first = next(reader, None)
                if first is None or not 1 <= len(first) <= MAX_CSV_COLUMNS:
                    raise TaskFailure("CSV пустой или содержит слишком много колонок.")
                headers = tuple(item.strip() for item in first)
                if any(not item or len(item) > 200 for item in headers) or len(set(headers)) != len(headers):
                    raise TaskFailure("Названия колонок CSV пустые, повторяются или слишком длинные.")
                rows: list[CsvRow] = []
                total = sum(len(item) for item in headers)
                for cells in reader:
                    if cancel is not None and cancel.is_set():
                        raise TaskCancelled()
                    if len(rows) >= MAX_CSV_ROWS:
                        raise TaskFailure("CSV превышает предел строк; разбейте файл на части без потери строк.")
                    if len(cells) != len(headers) or any(len(item) > MAX_CSV_CELL_CHARS for item in cells):
                        raise TaskFailure(f"Строка CSV {reader.line_num} имеет неверное число или размер ячеек.")
                    total += sum(len(item) for item in cells)
                    if total > MAX_CSV_TOTAL_CHARS:
                        raise TaskFailure("CSV превышает предел текстовых данных; разбейте файл на части.")
                    rows.append(CsvRow(reader.line_num, tuple(item.strip() for item in cells)))
                if not rows:
                    raise TaskFailure("CSV не содержит строк данных.")
                return CsvDocument(headers, tuple(rows), encoding, delimiter)
    except (OSError, UnicodeError, csv.Error):
        raise TaskFailure("CSV не читается с выбранной кодировкой и разделителем.") from None
