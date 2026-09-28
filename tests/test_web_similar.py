"""Похожие материалы по месяцам для карточек ТОПа."""

from datetime import date

from app.web_similar import WINDOW_MONTHS, similar_activity


def item(identifier: str, title: str, summary: str | None = None, month: str | None = "2026-08") -> dict:
    return {"publication_id": identifier, "title": title, "summary": summary,
            "publication_month": month, "published_at": None}


BATTERY = "Solid-state lithium battery electrolyte with sulfide interface stability"
POOL = [
    item("top", BATTERY),
    item("same-topic", "Sulfide electrolyte interface stability in solid-state lithium cells", month="2026-03"),
    item("same-topic-2", "Solid-state battery sulfide electrolyte degradation", month="2026-09"),
    item("other-topic", "Protein folding prediction with language models", month="2026-08"),
    item("too-old", "Sulfide solid-state electrolyte interface for lithium battery", month="2024-01"),
    item("undated", "Sulfide solid-state electrolyte interface for lithium battery", month=None),
    item("review", f'Review for "{BATTERY}"', month="2026-09"),
]


def test_counts_similar_materials_by_month_without_the_publication_itself():
    activity = similar_activity(POOL[:1], POOL, date(2026, 9, 26))["top"]
    months = {row["month"]: row for row in activity["months"]}
    # Окно — последние 12 месяцев, но начинается с первого месяца с собранными данными.
    assert activity["months"][0]["month"] == "2026-03" and activity["months"][-1]["month"] == "2026-09"
    assert months["2026-03"]["count"] == months["2026-09"]["count"] == 1
    # Другая тема, сама публикация, её рецензия, материал вне окна и без месяца не считаются.
    assert months["2026-08"]["count"] == 0
    assert activity["total"] == 2
    assert months["2026-08"]["collected"] == 2 and months["2026-09"]["collected"] == 2
    assert months["2026-05"]["collected"] == 0


def test_unrelated_publication_has_an_empty_chart():
    activity = similar_activity(POOL[3:4], POOL, date(2026, 9, 26))["other-topic"]
    assert activity["total"] == 0
    assert all(row["count"] == 0 for row in activity["months"])


def test_window_is_twelve_months_when_data_reaches_back():
    pool = [*POOL, item("old", "Protein folding benchmark", month="2025-10")]
    activity = similar_activity(pool[:1], pool, date(2026, 9, 26))["top"]
    assert len(activity["months"]) == WINDOW_MONTHS
    assert activity["months"][0]["month"] == "2025-10"


def test_publication_without_words_has_no_similar_materials():
    pool = [item("empty", "—"), *POOL]
    assert similar_activity(pool[:1], pool, date(2026, 9, 26))["empty"]["total"] == 0
