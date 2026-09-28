"""Этап 1: чтение датасета, признаки, обучение и объяснение решения."""

from pathlib import Path
from xml.sax.saxutils import escape
import zipfile

import pytest

from app.signal_model.__main__ import main
from app.signal_model.dataset import Technology, parse_sources, read_negatives, read_xlsx
from app.signal_model.features import extract, stage_level, trend_level
from app.signal_model.model import SignalModel, fit

NEGATIVES = Path(__file__).resolve().parents[1] / "data/signal_model/negatives.json"
HEADER = ["№", "Технология (слабый сигнал)", "Область", "Компании", "Почему это слабый сигнал",
          "Стадия развития", "Тренд упоминаний", "Балл (стадия+тренд)", "Источники"]


def write_xlsx(path: Path, rows: list[list[str]]) -> Path:
    """Минимальный xlsx с inline-строками, как у выгрузок Excel без общей таблицы строк."""
    def cell(column: int, row: int, value: str) -> str:
        letter = "ABCDEFGHIJKLMNOP"[column]
        return f'<c r="{letter}{row}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'

    sheet_rows = "".join(f'<row r="{index}">' + "".join(cell(column, index, value)
                                                        for column, value in enumerate(values) if value) + "</row>"
                         for index, values in enumerate(rows, start=1))
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", f'<workbook xmlns="{main_ns}" xmlns:r="{rel_ns}"><sheets>'
                                            '<sheet name="Лист" sheetId="1" r:id="rId1"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels",
                         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                         '<Relationship Id="rId1" Target="worksheets/sheet1.xml" '
                         'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>'
                         '</Relationships>')
        archive.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{main_ns}"><sheetData>{sheet_rows}'
                                                     '</sheetData></worksheet>')
    return path


def signal(number: int, area: str) -> list[str]:
    return ["", str(number), f"Сигнал {number}: агентная технология {area}", area, "Стартап X, Стартап Y",
            f"Компании вышли из stealth в 2025–2026 с seed-раундами; тема обсуждается в нишевых медиа ({number})",
            "Прототип/PoC → Пилот", "Растёт быстро — препринты 2026", "5",
            "[arXiv](https://arxiv.org/abs/2601.00001), https://example.org/news"]


def dataset(tmp_path: Path) -> list[Technology]:
    areas = ("Edge", "Финтех", "Роботы", "Защита ИИ", "Индустриальный ИИ", "Инфраструктура ИИ")
    rows = [["", "Заголовок таблицы"], ["", *HEADER]] + [signal(index, areas[index % 6]) for index in range(60)]
    return read_xlsx(write_xlsx(tmp_path / "signals.xlsx", rows)) + read_negatives(NEGATIVES)


def test_reader_finds_the_header_row_and_parses_links(tmp_path):
    rows = read_xlsx(write_xlsx(tmp_path / "one.xlsx", [["", "Заголовок"], ["", *HEADER], signal(1, "Edge")]))
    assert len(rows) == 1 and rows[0].label == 1 and rows[0].score == 5
    assert rows[0].area == "Edge" and rows[0].stage == "Прототип/PoC → Пилот"
    assert [source.url for source in rows[0].sources] == ["https://arxiv.org/abs/2601.00001"]
    assert parse_sources("см. https://example.org/a, https://example.org/b")[1].url == "https://example.org/b"


def test_label_column_of_a_closed_dataset_is_respected(tmp_path):
    header = ["Технология", "Стадия развития", "Тренд упоминаний", "Слабый сигнал"]
    rows = read_xlsx(write_xlsx(tmp_path / "closed.xlsx", [header, ["A", "Пилот", "Растёт", "да"],
                                                           ["B", "Массовое внедрение", "Стабильный", "нет"]]))
    assert [row.label for row in rows] == [1, 0]


def test_stage_and_trend_scales():
    assert stage_level("Прототип/PoC → Пилот") == pytest.approx(7 / 3)
    assert stage_level("Раннее внедрение (10 банков и финтехов)") == 4
    assert stage_level("Массовое внедрение") == 6
    assert stage_level("Не технология (зонтичный термин)") is None
    assert trend_level("Растёт быстро — рост оценки") == 2
    assert trend_level("Стабильный/растёт") == 0.5
    assert trend_level("Пик хайпа пройден — снижается") == -1
    assert trend_level("") is None


def test_features_are_measured_from_text_not_from_links():
    base = Technology("Технология", rationale="Seed-раунд в 2026 году", stage="Пилот", trend="Растёт")
    with_links = Technology(**{**base.__dict__, "sources": parse_sources("https://example.org/a")})
    assert extract(base) == extract(with_links)
    assert extract(base)["early_market"] > 0 and extract(base)["first_mention_age"] == 0


def test_model_separates_signals_and_explains_in_russian(tmp_path):
    data = dataset(tmp_path)
    model = fit(data)
    path = tmp_path / "model.json"
    model.save(path)
    loaded = SignalModel.load(path)
    mature = next(item for item in data if item.kind == "зрелая")
    signal_prediction, mature_prediction = loaded.predict([data[0], mature])
    assert signal_prediction.is_signal and not mature_prediction.is_signal
    assert signal_prediction.probability > 0.75 > mature_prediction.probability
    labels = {part.label for part in mature_prediction.contributions}
    assert "Стадия развития" in labels and "Хайп без подтверждений" in labels


def test_cli_trains_writes_report_and_predicts(tmp_path, capsys):
    rows = [["", *HEADER]] + [signal(index, ("Edge", "Финтех", "Роботы", "Защита ИИ", "Индустриальный ИИ",
                                             "Инфраструктура ИИ")[index % 6]) for index in range(30)]
    source = write_xlsx(tmp_path / "train.xlsx", rows)
    model, report = tmp_path / "model.json", tmp_path / "report.md"
    assert main(["train", "--dataset", str(source), "--model", str(model), "--report", str(report)]) == 0
    text = report.read_text(encoding="utf-8")
    assert "Повторная стратифицированная 5-кратная кросс-валидация" in text and "| Все признаки |" in text
    labelled = write_xlsx(tmp_path / "closed.xlsx", [["Технология", "Стадия развития", "Тренд упоминаний",
                                                      "Почему это слабый сигнал", "Слабый сигнал"],
                                                     ["Агентный стартап", "Пилот", "Растёт быстро",
                                                      "Вышли из stealth с seed-раундом в 2026", "да"],
                                                     ["Облачные вычисления", "Массовое внедрение", "Стабильный",
                                                      "Сформированный рынок, выраженные лидеры", "нет"]])
    out = tmp_path / "predictions.csv"
    assert main(["predict", "--dataset", str(labelled), "--model", str(model), "--out", str(out)]) == 0
    assert "Точность на размеченных строках: 2/2" in capsys.readouterr().out
    assert "слабый сигнал" in out.read_text(encoding="utf-8-sig")
