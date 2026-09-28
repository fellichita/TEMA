"""ТОП технологий в веб-интерфейсе: карточки, журнал исключений и страница-отчёт."""

import os
from pathlib import Path
import re
from unittest.mock import patch

import pytest

pytest.importorskip("requests")
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from app.ui.api_client import ApiClient, ApiError, parse_status
from app.ui.radar_view import curve_svg, find_report

WEB = Path(__file__).resolve().parents[1] / "app/ui/web.py"
MONTHS = [f"2025-{month:02d}" for month in range(10, 13)] + [f"2026-{month:02d}" for month in range(1, 10)]


def technology(title: str, probability: float, *, signal: bool = True, reasons=()) -> dict:
    return {
        "title": title, "probability": probability, "score": probability * 0.8, "is_signal": signal,
        "curve": {"confidence": 90 if signal else 0, "trend": "растёт быстро" if signal else "стабильный",
                  "materials": 40,
                  "months": [{"month": month, "materials": index, "weighted": float(index),
                              "smoothed": float(index), "fitted": float(index)} for index, month in enumerate(MONTHS)],
                  "checks": [{"name": "accelerating", "passed": True, "label": "Парабола ветвями вверх"}]},
        "passport": {"stage": "Прототип/PoC", "trend": "Растёт быстро", "rationale": "Тема нишевая: 30 работ.",
                     "companies": "", "first_year": 2023, "all_time": 30, "mature": not signal},
        "predictors": [{"label": "Динамика упоминаний", "value": "растёт быстро", "weight": 1.75},
                       {"label": "Стадия развития", "value": "прототип", "weight": -0.4}],
        "sources": [{"title": "Sulfide paper", "url": "https://doi.org/10.1234/x", "published": "2026-08-01",
                     "source": "Crossref", "source_type": "научная публикация", "language": "английский",
                     "trust": "высокий"},
                    {"title": "Bad link", "url": "javascript:alert(1)", "published": "", "source": "",
                     "source_type": "", "language": "", "trust": ""}],
        "description": "Sulfide electrolytes conduct lithium ions fast.", "advantage": None, "case": None,
        "reasons": list(reasons),
    }


def status(radar: dict):
    return parse_status({"state": "succeeded", "id": "run-radar", "query": "Батареи", "mode": "fast",
                         "result": {"signals": []}, "radar": radar})


def publication(index: int, title: str, url: str, score: int | None = None) -> dict:
    item = {"publication_id": f"{index:064x}", "source_id": "crossref", "kind": "journal-article",
            "title": title, "url": url, "published_at": "2026-08-01",
            "publication_year": 2026, "date_basis": "published", "summary": None}
    if score is not None:
        item["model_confidence"] = {"score": score, "reason": "Relevant title.",
                                    "evidence_quote": title, "basis": "title"}
    return item


def status_with_publications(radar: dict, publications: list[dict], *, run_id: str = "run-radar"):
    return parse_status({"state": "succeeded", "id": run_id, "query": "Батареи", "mode": "fast",
                         "result": {"signals": [], "publications": publications,
                                    "top_publications": publications, "publication_total": len(publications)},
                         "radar": radar})


READY = {"state": "ready", "completed": 2, "total": 2, "result": {
    "policy_version": "radar/1.3.0", "query": "Батареи", "as_of": "2026-09-26", "months": 12,
    "candidates_total": 2, "evaluated": 2,
    "high_confidence": 1, "sources_processed": 120,
    "technologies": [technology("sulfide solid electrolyte", 0.97)],
    "excluded": [{**technology("lithium-ion batteries", 0.88, signal=False,
                               reasons=["Зрелая тема: 8218 работ за всё время"]), "rule_excluded": True}],
    "translation": {"sulfide solid electrolyte": "сульфидный твёрдый электролит",
                    "Sulfide electrolytes conduct lithium ions fast.": "Сульфидные электролиты быстро проводят ионы."}}}


def run(state, query_params=None):
    with patch.dict(os.environ, {"TREND_WEB_REQUIRE_AUTH": "0"}), \
            patch.object(ApiClient, "current_analysis", return_value=state), patch.object(
            ApiClient, "mode_estimates", side_effect=ApiError("нет прогноза")):
        app = AppTest.from_file(WEB)
        for key, value in (query_params or {}).items():
            app.query_params[key] = value
        app.run()
    assert not app.exception
    return "\n".join(item.value for item in app.markdown)


def test_ready_radar_shows_cards_stats_and_exclusion_journal():
    markup = run(status(READY))
    assert "ТОП-15 зарождающихся технологий" in markup
    # Термин показывается в оригинале: перевод коротких названий искажает смысл.
    assert "sulfide solid electrolyte" in markup and "сульфидный твёрдый электролит" not in markup
    assert "Уверенность модели выше 75%" in markup and "Обработано материалов" in markup
    assert 'href="?report=t1"' in markup and "97%" in markup
    assert "Почему не вошли в ТОП · 1" in markup and "Зрелая тема: 8218 работ" in markup
    # У зрелой темы видна оценка модели и причина исключения: процент не меняет правило.
    assert "88% · исключено правилом" in markup
    assert "до 15 направлений" in markup
    assert "создан до обновления отбора направлений" not in markup


@pytest.mark.parametrize("version", ["radar/1.1.0", "radar/1.2.0"])
def test_saved_result_from_older_selection_policy_explains_how_to_refresh(version):
    from app.ui.radar_client import parse_radar
    from app.ui.radar_view import radar_markup

    old = {**READY, "result": {**READY["result"], "policy_version": version}}
    parsed = parse_radar(old)
    assert parsed.result.policy_version == version
    notice = ("Этот сохранённый анализ создан до обновления отбора направлений. "
              "Запустите новый анализ, чтобы увидеть результаты по новым правилам.")
    assert notice in radar_markup(parsed, "run-7")
    assert notice not in radar_markup(parsed)
    assert notice not in radar_markup(parse_radar(READY), "run-7")


def test_old_selection_policy_notice_appears_on_saved_result_page():
    from app.ui.api_client import AnalysisStatus

    old = {**READY, "result": {**READY["result"], "policy_version": "radar/1.1.0"}}
    saved = status_with_publications(old, [], run_id="run-7")
    with patch.dict(os.environ, {"TREND_WEB_REQUIRE_AUTH": "0"}), \
            patch.object(ApiClient, "current_analysis", return_value=AnalysisStatus("idle")), \
            patch.object(ApiClient, "saved_analysis", return_value=saved), \
            patch.object(ApiClient, "mode_estimates", side_effect=ApiError("нет прогноза")):
        app = AppTest.from_file(WEB)
        app.query_params["run"] = "run-7"
        app.run()
    assert not app.exception
    markup = "\n".join(item.value for item in app.markdown)
    assert "Этот сохранённый анализ создан до обновления отбора направлений." in markup


def test_ready_radar_merges_publications_into_their_technology_cards():
    first = technology("sulfide solid electrolyte", 0.97)
    second = technology("sodium cathode", 0.82)
    second["sources"] = [{"title": "Cathode study", "url": "https://doi.org/10.1234/cathode",
                          "published": "2026-07-01", "source": "Crossref",
                          "source_type": "научная публикация", "language": "английский",
                          "trust": "высокий"}]
    radar = {**READY, "result": {**READY["result"], "technologies": [first, second],
                                  "excluded": []}}
    publications = [
        {"publication_id": f"{index:064x}", "source_id": "crossref", "kind": "journal-article",
         "title": title, "url": url, "published_at": "2026-08-01",
         "publication_year": 2026, "date_basis": "published", "summary": None,
         "model_confidence": {"score": score, "reason": "Relevant title.",
                              "evidence_quote": title, "basis": "title"}}
        for index, title, url, score in (
            (1, "Sulfide paper", "https://doi.org/10.1234/X", 91),
            (2, "Cathode study", "https://doi.org/10.1234/cathode", 78))
    ]
    state = parse_status({"state": "succeeded", "id": "run-radar", "query": "Батареи", "mode": "fast",
                          "result": {"signals": [], "publications": publications,
                                     "top_publications": publications, "publication_total": 2},
                          "radar": radar})
    markup = run(state)
    assert '<div class="ta-grid ta-radar-grid">' in markup
    assert '<div class="ta-grid">' not in markup
    assert "Остальные публикации в списке" not in markup
    cards = re.findall(r'<article class="ta-card ta-radar-card">.*?</article>', markup, flags=re.S)
    assert len(cards) == 2
    assert "sulfide solid electrolyte" in cards[0] and "97%" in cards[0]
    assert "Sulfide paper" in cards[0] and "Соответствие запросу: <b>91/100</b>" in cards[0]
    assert "Cathode study" not in cards[0]
    assert "sodium cathode" in cards[1] and "82%" in cards[1]
    assert "Cathode study" in cards[1] and "Соответствие запросу: <b>78/100</b>" in cards[1]
    assert "Sulfide paper" not in cards[1]
    assert cards[0].count('<details class="ta-radar-details">') == 1
    assert "Источники · 1" in cards[0] and "Источники по технологии" in cards[0]
    assert 'href="?report=t1"' in cards[0] and 'href="?report=t2"' in cards[1]


@pytest.mark.parametrize("technologies", [[], [{**technology("empty evidence", 0.74), "sources": []}]])
def test_ready_radar_without_technology_sources_keeps_publication_fallback(technologies):
    radar = {**READY, "result": {**READY["result"], "technologies": technologies, "excluded": []}}
    state = status_with_publications(radar, [publication(1, "Fallback study",
                                                        "https://doi.org/10.1234/fallback", 83)])
    markup = run(state)
    assert '<div class="ta-grid">' in markup
    assert "Fallback study" in markup and "83<small>/100</small>" in markup
    assert "Публикации без связи с карточками технологий" not in markup


def test_ready_radar_shows_only_technology_sources_when_some_top_publications_do_not_match():
    papers = [publication(1, "Sulfide paper", "https://doi.org/10.1234/X", 91),
              publication(2, "Unmatched paper", "https://doi.org/10.1234/unmatched", 78)]
    state = status_with_publications(READY, papers)
    markup = run(state)
    assert '<div class="ta-grid ta-radar-grid">' in markup
    assert '<div class="ta-grid">' not in markup
    assert "Unmatched paper" not in markup
    assert "Публикации без связи с карточками технологий" not in markup
    assert "Соответствие запросу: <b>91/100</b>" in markup


def test_source_score_survives_the_full_report_link():
    state = status_with_publications(READY, [publication(1, "Sulfide paper",
                                                          "https://doi.org/10.1234/X", 91)])
    report = run(state, {"report": "t1"})
    assert "Источники" in report
    assert "Sulfide paper" in report
    assert "Соответствие запросу: <b>91/100</b>" in report
    assert "Публикации по технологии" not in report


def test_preprint_score_matches_the_journal_source_by_unique_work_title():
    item = technology("sulfide solid electrolyte", 0.97)
    item["sources"][0]["title"] = "Sulfide paper"
    item["sources"][0]["url"] = "https://doi.org/10.1234/journal-version"
    radar = {**READY, "result": {**READY["result"], "technologies": [item], "excluded": []}}
    state = status_with_publications(radar, [publication(1, "Sulfide paper",
                                                           "https://arxiv.org/abs/2609.12345", 91)])
    overview = run(state)
    assert "Соответствие запросу: <b>91/100</b>" in overview
    assert "Публикации без связи с карточками технологий" not in overview
    report = run(state, {"report": "t1"})
    assert "Соответствие запросу: <b>91/100</b>" in report


def test_title_matching_does_not_guess_between_two_top_publications():
    item = technology("sulfide solid electrolyte", 0.97)
    item["sources"][0]["title"] = "Sulfide paper"
    item["sources"][0]["url"] = "https://doi.org/10.1234/journal-version"
    radar = {**READY, "result": {**READY["result"], "technologies": [item], "excluded": []}}
    papers = [publication(1, "Sulfide paper", "https://arxiv.org/abs/2609.12345", 91),
              publication(2, "Sulfide-paper", "https://doi.org/10.1234/other", 72)]
    overview = run(status_with_publications(radar, papers))
    cards = re.findall(r'<article class="ta-card ta-radar-card">.*?</article>', overview, flags=re.S)
    assert len(cards) == 1
    assert "Соответствие запросу:" not in cards[0]
    assert "Публикации без связи с карточками технологий" not in overview


def test_direct_source_url_never_borrows_another_publications_title_score():
    item = technology("sulfide solid electrolyte", 0.97)
    item["sources"][0]["title"] = "Sulfide paper"
    radar = {**READY, "result": {**READY["result"], "technologies": [item], "excluded": []}}
    papers = [publication(1, "Different work", "https://doi.org/10.1234/x"),
              publication(2, "Sulfide paper", "https://doi.org/10.1234/another", 91)]
    state = status_with_publications(radar, papers)
    overview = run(state)
    cards = re.findall(r'<article class="ta-card ta-radar-card">.*?</article>', overview, flags=re.S)
    assert len(cards) == 1 and "Соответствие запросу:" not in cards[0]
    report = run(state, {"report": "t1"})
    assert "Соответствие запросу:" not in report


def test_lower_confidence_top_technology_is_labelled_candidate():
    candidate = technology("sodium cathode", 0.42, signal=False,
                           reasons=["Данных для уверенного сигнала пока мало"])
    radar = {**READY, "result": {**READY["result"], "technologies": [candidate],
                                  "excluded": []}}
    result = status(radar)
    overview = run(result)
    assert "Направлений в ТОПе" in overview
    assert '<span class="ta-radar-kind">Кандидат</span>' in overview
    assert "42%" in overview
    assert "Слабых сигналов в ТОПе" not in overview
    report = run(result, {"report": "t1"})
    assert "<dt>Статус</dt><dd>Кандидат</dd>" in report
    assert "Что ограничивает уверенность" in report
    assert "Почему не вошёл в ТОП" not in report


def test_unavailable_radar_keeps_the_publication_fallback():
    publication = {"publication_id": "a" * 64, "source_id": "crossref", "kind": "journal-article",
                   "title": "Fallback study", "url": "https://doi.org/10.1234/fallback",
                   "published_at": "2026-08-01", "publication_year": 2026,
                   "date_basis": "published", "summary": None}
    state = parse_status({"state": "succeeded", "id": "run-radar", "query": "Батареи", "mode": "fast",
                          "result": {"signals": [], "publications": [publication],
                                     "top_publications": [publication], "publication_total": 1},
                          "radar": {"state": "unavailable", "message": "Радар недоступен"}})
    markup = run(state)
    assert "Радар недоступен" in markup
    assert '<div class="ta-grid">' in markup and "Fallback study" in markup


def test_report_page_shows_everything_the_brief_requires():
    markup = run(status(READY), {"report": "t1"})
    assert "Отчёт по технологии" in markup and "← К результатам" in markup
    for part in ("Описание технологии", "Потенциальное преимущество", "Кейс-пример",
                 "Почему модель дала такую оценку", "Динамика: логика эксперта", "Источники"):
        assert part in markup
    assert "Сульфидные электролиты быстро проводят ионы." in markup and "машинный перевод" in markup
    assert "научная публикация" in markup and "английский" in markup and "высокий" in markup
    assert "javascript:" not in markup
    assert "ТОП-15 зарождающихся технологий" not in markup


def test_excluded_report_explains_the_exclusion():
    markup = run(status(READY), {"report": "x1"})
    assert "Почему не вошёл в ТОП" in markup and "Зрелая тема: 8218 работ" in markup


def test_running_radar_reports_progress_and_unknown_report_falls_back():
    markup = run(status({"state": "running", "completed": 3, "total": 20}), {"report": "t1"})
    assert "Считаем: 3 из 20" in markup and "помесячная история" not in markup


def test_reports_of_a_saved_analysis_stay_with_that_analysis():
    from app.ui.api_client import AnalysisStatus
    from app.ui.radar_client import parse_radar
    from app.ui.radar_view import radar_markup

    assert 'href="?run=run-7&amp;report=t1"' in radar_markup(parse_radar(READY), "run-7")
    assert 'href="?report=t1"' in radar_markup(parse_radar(READY))
    saved = status_with_publications(READY, [publication(1, "Sulfide paper",
                                                        "https://doi.org/10.1234/X", 91)], run_id="run-7")
    with patch.dict(os.environ, {"TREND_WEB_REQUIRE_AUTH": "0"}), \
            patch.object(ApiClient, "current_analysis", return_value=AnalysisStatus("idle")), \
            patch.object(ApiClient, "saved_analysis", return_value=saved), \
            patch.object(ApiClient, "mode_estimates", side_effect=ApiError("нет прогноза")):
        app = AppTest.from_file(WEB)
        app.query_params["run"] = "run-7"
        app.query_params["report"] = "t1"
        app.run()
    assert not app.exception
    markup = "\n".join(item.value for item in app.markdown)
    assert "Отчёт по технологии" in markup and 'href="?run=run-7"' in markup
    assert "Соответствие запросу: <b>91/100</b>" in markup


def test_top_saved_by_the_analysis_says_it_is_only_being_translated():
    markup = run(status({"state": "running", "message": "Переводим описания технологий на русский…"}))
    assert "Переводим описания технологий на русский…" in markup and "Считаем" not in markup


def test_report_keys_and_curve_are_safe():
    result = status(READY).radar.result
    assert find_report(result, "t1").title == "sulfide solid electrolyte"
    assert find_report(result, "t9") is None and find_report(result, "zz") is None
    svg = curve_svg(result.technologies[0].points, large=True)
    assert svg.count("<rect") == 12 and 'class="ta-curve-fit"' in svg
    # Координаты — числа с точкой, иначе браузер не рисует столбики.
    import re
    assert not re.search(r'(?:x|y|width|height)="[^"]*,', svg)
    assert "сумма коэффициентов 11,00" in svg
