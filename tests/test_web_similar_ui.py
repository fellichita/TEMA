"""График похожих материалов в карточках ТОПа: разбор ответа и отображение."""

from copy import deepcopy
import os
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("requests")
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from app.ui.api_client import ApiClient, ApiError, AnalysisStatus, parse_result
from app.ui.web import similar_markup

WEB = Path(__file__).resolve().parents[1] / "app/ui/web.py"
MONTHS = [f"2025-{month:02d}" for month in range(10, 13)] + [f"2026-{month:02d}" for month in range(1, 10)]


def similar(counts: list[int]) -> dict:
    return {"months": [{"month": month, "count": count, "collected": 10 if index else 0}
                       for index, (month, count) in enumerate(zip(MONTHS, counts, strict=True))],
            "total": sum(counts)}


def payload(counts: list[int]) -> dict:
    publication = {"publication_id": "a" * 64, "source_id": "arxiv", "kind": "preprint",
                   "title": "Sulfide electrolytes", "url": "https://arxiv.org/abs/2609.00001",
                   "published_at": "2026-09-20", "publication_year": 2026,
                   "date_basis": "published", "summary": None, "similar": similar(counts)}
    return {"signals": [], "publications": [publication], "top_publications": [publication],
            "publication_total": 1}


def test_top_card_shows_the_similar_chart_and_no_common_monthly_block():
    result = parse_result(payload([0, 1, 0, 0, 2, 0, 0, 3, 1, 0, 4, 1]))
    assert result.top_publications[0].similar.months[-1].collected == 10
    assert result.top_publications[0].similar.total == 12
    status = AnalysisStatus("succeeded", "run-similar", "Тема", "fast", result)
    with patch.dict(os.environ, {"TREND_WEB_REQUIRE_AUTH": "0"}), \
            patch.object(ApiClient, "current_analysis", return_value=status), patch.object(
            ApiClient, "mode_estimates", side_effect=ApiError("нет прогноза")):
        app = AppTest.from_file(WEB).run()
    assert not app.exception
    markup = "\n".join(item.value for item in app.markdown)
    assert "Похожие в выборке" in markup and "за 12 мес." in markup
    assert markup.count("<rect") == 12
    assert "окт 2025" in markup and "сен 2026" in markup and "пик 4" in markup
    assert "Активность найденных материалов" not in markup
    assert not any(item.label == "Период динамики" for item in app.segmented_control)


def test_empty_similar_chart_keeps_the_months_visible():
    result = parse_result(payload([0] * 12))
    markup = similar_markup(result.top_publications[0].similar)
    # График есть у каждой карточки: пустые месяцы — тонкие отметки.
    assert "Похожие в выборке" in markup and markup.count('class="ta-similar-empty"') == 12


def test_month_labels_and_tooltips_are_readable():
    markup = similar_markup(parse_result(payload([0] * 11 + [2])).top_publications[0].similar)
    assert "<title>сен 2026: 2 из 10 собранных</title>" in markup
    assert "<title>окт 2025: материалов за месяц не собрано</title>" in markup
    assert 'aria-label="Похожие материалы по месяцам: сен 2026 — 2"' in markup


@pytest.mark.parametrize("change", [
    lambda item: item.update(total=99),
    lambda item: item["months"][0].update(count=-1),
    lambda item: item["months"][1].update(count=11),
    lambda item: item["months"][1].pop("collected"),
    lambda item: item["months"][1].update(month="2025-10"),
    lambda item: item.update(months=[]),
])
def test_invalid_similar_payload_is_rejected(change):
    broken = deepcopy(payload([1] * 12))
    change(broken["publications"][0]["similar"])
    broken["top_publications"] = broken["publications"]
    with pytest.raises(ApiError, match="похожих материалов"):
        parse_result(broken)
