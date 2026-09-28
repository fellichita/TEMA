"""Чтение датасета организаторов (xlsx) и отрицательных примеров команды.

Xlsx разбирается стандартной библиотекой и defusedxml, без openpyxl: файл
приходит извне, а лишняя зависимость ради одного листа не нужна.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import zipfile

from defusedxml import ElementTree

_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_RELS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
MAX_XLSX_BYTES = 20_000_000
MAX_ROWS = 50_000

# Заголовки колонок: у закрытого датасета названия могут немного отличаться,
# поэтому сравниваем по началу строки без учёта регистра.
HEADERS = {
    "title": ("технология", "название технологии", "название"),
    "area": ("область", "направление"),
    "companies": ("компании", "игроки"),
    "rationale": ("почему это слабый сигнал", "описание", "обоснование"),
    "stage": ("стадия развития", "стадия"),
    "trend": ("тренд упоминаний", "тренд", "динамика"),
    "score": ("балл",),
    "sources": ("источники", "источник"),
    "label": ("слабый сигнал", "метка", "класс", "label", "is_weak_signal"),
}
POSITIVE_LABELS = frozenset({"1", "да", "yes", "true", "слабый сигнал", "сигнал"})
NEGATIVE_LABELS = frozenset({"0", "нет", "no", "false", "не слабый сигнал", "не сигнал"})
_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")


@dataclass(frozen=True)
class Source:
    title: str
    url: str


@dataclass(frozen=True)
class Technology:
    """Одна строка датасета. `label` = 1 — слабый сигнал, 0 — нет, None — неизвестно."""
    title: str
    area: str = ""
    companies: str = ""
    rationale: str = ""
    stage: str = ""
    trend: str = ""
    score: int | None = None
    sources: tuple[Source, ...] = ()
    label: int | None = None
    # Для отрицательных примеров: «зрелая», «хайп» или «шум».
    kind: str | None = None


def parse_sources(text: str) -> tuple[Source, ...]:
    """Ссылки в markdown-формате `[название](url)`; голые адреса тоже принимаются."""
    found = [Source(title.strip() or url, url) for title, url in _LINK.findall(text)]
    if not found:
        found = [Source(url, url) for url in re.findall(r"https?://[^\s,;]+", text)]
    return tuple(found)


def _label(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip().casefold()
    if text in POSITIVE_LABELS:
        return 1
    if text in NEGATIVE_LABELS:
        return 0
    return None


def _column_of(header: str) -> str | None:
    text = " ".join(header.split()).casefold()
    # «Слабый сигнал» — метка, но «Технология (слабый сигнал)» — название.
    for field in ("rationale", "score", "stage", "trend", "sources", "companies", "area", "title", "label"):
        if any(text.startswith(prefix) for prefix in HEADERS[field]):
            return field
    return None


def _sheet_rows(path: Path) -> list[dict[str, str]]:
    if path.stat().st_size > MAX_XLSX_BYTES:
        raise ValueError("Файл датасета слишком большой.")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            for item in ElementTree.fromstring(archive.read("xl/sharedStrings.xml")).findall(f"{_MAIN}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{_MAIN}t")))
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relations = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.get("Id"): item.get("Target", "") for item in relations}
        first = workbook.find(f"{_MAIN}sheets")[0]
        target = targets[first.get(f"{_RELS}id")].lstrip("/")
        target = target if target.startswith("xl/") else "xl/" + target
        rows = []
        for row in ElementTree.fromstring(archive.read(target)).iter(f"{_MAIN}row"):
            cells: dict[str, str] = {}
            for cell in row.findall(f"{_MAIN}c"):
                match = re.match(r"[A-Z]+", cell.get("r", ""))
                if match is None:
                    raise ValueError("Некорректный адрес ячейки в датасете.")
                column = match.group()
                value, kind = cell.find(f"{_MAIN}v"), cell.get("t")
                if kind == "s" and value is not None:
                    cells[column] = shared[int(value.text)]
                elif kind == "inlineStr":
                    cells[column] = "".join(node.text or "" for node in cell.iter(f"{_MAIN}t"))
                elif value is not None and value.text is not None:
                    cells[column] = value.text
            rows.append(cells)
            if len(rows) > MAX_ROWS:
                raise ValueError("В датасете слишком много строк.")
    return rows


def read_xlsx(path: str | Path, *, default_label: int | None = 1) -> list[Technology]:
    """Первый лист; строка заголовков находится по колонке с названием технологии.

    У выданного датасета нет колонки метки: все строки — слабые сигналы, поэтому
    по умолчанию им ставится метка 1. Если колонка метки есть, берётся она.
    """
    rows = _sheet_rows(Path(path))
    for index, cells in enumerate(rows):
        columns = {letter: _column_of(text) for letter, text in cells.items() if text}
        if "title" in columns.values():
            header, mapping = index, {letter: field for letter, field in columns.items() if field}
            break
    else:
        raise ValueError("Не найдена строка заголовков с колонкой технологии.")
    technologies = []
    for cells in rows[header + 1:]:
        values = {field: (cells.get(letter) or "").strip() for letter, field in mapping.items()}
        if not values.get("title"):
            continue
        score = values.get("score")
        label = _label(values.get("label")) if "label" in values else default_label
        technologies.append(Technology(
            title=values["title"], area=values.get("area", ""), companies=values.get("companies", ""),
            rationale=values.get("rationale", ""), stage=values.get("stage", ""),
            trend=values.get("trend", ""),
            score=int(float(score)) if score and re.fullmatch(r"\d+(?:\.0+)?", score) else None,
            sources=parse_sources(values.get("sources", "")), label=label))
    return technologies


def read_negatives(path: str | Path) -> list[Technology]:
    """Отрицательные примеры команды: зрелые технологии, хайп и шум в схеме датасета."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Technology(title=item["Технология"], area=item["Область"], companies=item.get("Компании", ""),
                       rationale=item["Описание"], stage=item.get("Стадия развития", ""),
                       trend=item.get("Тренд упоминаний", ""), label=0, kind=item["Тип"])
            for item in items]
