"""Проверки веб-слоя на явно тестовых ответах, без моделей и сетевого анализа."""

import re
import os
from pathlib import Path
from threading import Event
from unittest.mock import patch

import pytest

requests = pytest.importorskip("requests")
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from app.ui.api_client import (ApiClient, ApiError, AnalysisStatus, parse_publication,
                               parse_result, parse_status)

WEB = Path(__file__).resolve().parents[1] / "app/ui/web.py"


def form_buttons(app):
    """Кнопки страницы без постоянной кнопки «История» в углу."""
    return [button for button in app.button if button.key != "history_toggle"]


def sample(**changes):
    return {"title": "Тестовая карточка", "summary": "Тест интерфейса, не результат исследования.",
            "category": "confirmed_trend", "source_urls": ["https://example.org/test"], **changes}


@pytest.fixture(autouse=True)
def idle_analysis():
    # Прогноз по умолчанию недоступен: тесты не ходят в настоящий API.
    with patch.dict(os.environ, {"TREND_WEB_REQUIRE_AUTH": "0"}), \
            patch.object(ApiClient, "current_analysis", return_value=AnalysisStatus("idle")), \
            patch.object(ApiClient, "mode_estimates", side_effect=ApiError("нет прогноза")),             patch.object(ApiClient, "report_presence", return_value={"access": None, "messages": []}),             patch.object(ApiClient, "check_access", return_value=None):
        yield


def test_owner_panel_learns_about_failed_and_successful_logins():
    environment = {"TREND_WEB_REQUIRE_AUTH": "1", "TREND_WEB_ACCESS_PASSWORD": "correct-horse-demo-password"}
    with patch.dict(os.environ, environment, clear=False),             patch.object(ApiClient, "report_presence", return_value={"access": None, "messages": []}) as report:
        app = AppTest.from_file(WEB).run()
        assert report.call_args.args[1] == "login" and report.call_args.kwargs["event"] is None
        session = report.call_args.args[0]
        app.text_input[0].set_value("wrong-password")
        form_buttons(app)[0].click().run()
        assert ("login", "login_failed") in [(call.args[1], call.kwargs["event"]) for call in report.call_args_list]
        app.text_input[0].set_value("correct-horse-demo-password")
        form_buttons(app)[0].click().run()
        assert not app.exception
        events = [(call.args[1], call.kwargs["event"]) for call in report.call_args_list]
        assert ("login", "login") in events and events[-1] == ("main", None)
        # Одна вкладка — один идентификатор, от экрана входа до главной.
        assert {call.args[0] for call in report.call_args_list} == {session}


def test_initial_and_blank_query_do_not_call_api():
    with patch.object(ApiClient, "start_analysis") as start:
        app = AppTest.from_file(WEB).run()
        assert not app.exception
        assert len(form_buttons(app)) == 1
        form_buttons(app)[0].click().run()
        assert "Введите тему" in app.error[0].value
        start.assert_not_called()


def test_public_demo_requires_the_server_password():
    environment = {"TREND_WEB_REQUIRE_AUTH": "1", "TREND_WEB_ACCESS_PASSWORD": "correct-horse-demo-password"}
    with patch.dict(os.environ, environment, clear=False), patch.object(ApiClient, "start_analysis") as start:
        app = AppTest.from_file(WEB).run()
        assert not app.exception
        assert len(app.text_input) == 1
        assert not app.text_area
        app.text_input[0].set_value("wrong-password")
        form_buttons(app)[0].click().run()
        assert "Неверный" in app.error[0].value
        start.assert_not_called()


@pytest.mark.parametrize("mode", [None, "", "false", "2"])
def test_web_denies_access_without_explicit_auth_mode(monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv("TREND_WEB_REQUIRE_AUTH", raising=False)
    else:
        monkeypatch.setenv("TREND_WEB_REQUIRE_AUTH", mode)
    with patch.object(ApiClient, "current_analysis") as current:
        app = AppTest.from_file(WEB).run()
    assert not app.exception
    assert not app.text_area
    assert any("Режим доступа" in error.value for error in app.error)
    current.assert_not_called()


def test_cards_escape_content_deduplicate_sources_and_survive_rerun():
    result = parse_result({"signals": [sample(title="<script>alert(1)</script>"), sample()]})
    status = AnalysisStatus("idle")

    def current(_self):
        return status

    def start(_self, query, mode):
        nonlocal status
        status = AnalysisStatus("succeeded", "run-1", query, mode, result)
        return status

    with patch.object(ApiClient, "current_analysis", current), patch.object(ApiClient, "start_analysis", start):
        app = AppTest.from_file(WEB).run()
        app.text_area[0].set_value("Тестовый запрос")
        form_buttons(app)[0].click().run()
        assert not app.exception
        markup = "\n".join(item.value for item in app.markdown)
        assert "&lt;script&gt;" in markup
        assert "<script>" not in markup
        assert "Ссылок в списке: 1" in markup
        assert "multilingual-e5-small" not in markup
        assert "Подтверждённый тренд" in markup
        app.run()
        assert not form_buttons(app)[0].disabled


def test_no_sources_produces_empty_state():
    result = parse_result({"incomplete_coverage": True,
                           "signals": [sample(source_urls=[]), sample(source_urls=["javascript:alert(1)"])]})
    assert result.omitted == 2
    status = AnalysisStatus("idle")

    def current(_self):
        return status

    def start(_self, query, mode):
        nonlocal status
        status = AnalysisStatus("succeeded", "run-1", query, mode, result)
        return status

    with patch.object(ApiClient, "current_analysis", current), patch.object(ApiClient, "start_analysis", start):
        app = AppTest.from_file(WEB).run()
        app.text_area[0].set_value("Тест")
        form_buttons(app)[0].click().run()
        assert not app.exception
        assert any("Нет результатов для показа" in item.value for item in app.markdown)
        assert any("не найдено публикаций или сигналов" in item.value
                   for item in app.markdown)
        assert any("Охват источников неполный" in item.value for item in app.markdown)


def test_publications_from_all_sources_render_together_without_signals():
    publications = [
        {"source_id": "openalex", "kind": "publication", "title": "Scientific study",
         "url": "https://doi.org/10.1234/study", "published_at": None,
         "publication_year": 2025, "date_basis": "published", "summary": "Abstract"},
        {"source_id": "arxiv", "kind": "preprint", "title": "<script>alert(1)</script>",
         "url": "https://arxiv.org/abs/1234.56789", "published_at": "2026-09-20",
         "publication_year": 2026, "date_basis": "published", "summary": "Early result"},
        {"source_id": "gdelt", "kind": "news_aggregate", "title": "News coverage",
         "url": "https://example.org/story", "published_at": "2026-09-21",
         "publication_year": 2026, "date_basis": "indexed", "summary": None},
        {"source_id": "openreview", "kind": "preprint", "title": "Public forum",
         "url": "https://openreview.net/forum?id=AbcDef12", "published_at": "2026-09-22",
         "publication_year": 2026, "date_basis": "created", "summary": None},
    ]
    for index, publication in enumerate(publications, start=1):
        publication["publication_id"] = f"{index:064x}"
    result = parse_result({
        "signals": [], "publications": publications,
        "top_publications": publications, "publication_total": 4,
        "source_coverage": [
            {"source_id": source, "state": "complete", "scanned": 1,
             "accepted": 1, "limit_reached": False, "reason_code": None}
            for source in ("openalex", "arxiv", "gdelt", "openreview")
        ],
    })
    status = AnalysisStatus("succeeded", "run-1", "quantum sensors", "fast", result)
    with patch.object(ApiClient, "current_analysis", return_value=status):
        app = AppTest.from_file(WEB).run()
    assert not app.exception
    markup = "\n".join(item.value for item in app.markdown)
    assert "ТОП-15 публикаций" not in markup
    assert "Scientific study" in markup
    assert "arXiv" in markup and "OpenAlex" in markup
    assert "&lt;script&gt;" in markup and "<script>" not in markup
    assert "https://arxiv.org/abs/1234.56789" in markup
    assert "Обнаружено 2026-09-21" in markup
    assert "Создано 2026-09-22" in markup
    assert "Нет результатов для показа" not in markup
    assert "Дополнительные источники" not in markup
    assert "не входят в научный TOP" not in markup


def test_publication_top_shows_fifteen_and_keeps_the_rest_in_the_same_list():
    publications = [
        {"publication_id": f"{index:064x}", "source_id": "arxiv", "kind": "preprint",
         "title": f"Paper {index}", "url": f"https://arxiv.org/abs/2609.{index:05d}",
         "published_at": "2026-09-20", "publication_year": 2026,
         "date_basis": "published", "summary": None}
        for index in range(1, 17)
    ]
    result = parse_result({
        "signals": [], "publications": publications,
        "top_publications": publications[:15], "publication_total": 16,
        "source_coverage": [{"source_id": "arxiv", "state": "complete", "scanned": 16,
                             "accepted": 16, "limit_reached": False, "reason_code": None}],
    })
    status = AnalysisStatus("succeeded", "run-1", "quantum sensors", "fast", result)
    with patch.object(ApiClient, "current_analysis", return_value=status):
        app = AppTest.from_file(WEB).run()
    assert not app.exception
    markup = "\n".join(item.value for item in app.markdown)
    assert "ТОП-15 публикаций" not in markup
    assert 'Остальные публикации в списке (1)' in markup
    assert 'Paper 15' in markup and 'Paper 16' in markup
    assert markup.count('class="ta-card ta-publication-card"') == 16
    assert "Дополнительные источники" not in markup


def test_publication_navigation_keeps_only_one_extra_page_and_resets_for_another_run():
    publications = [
        {"publication_id": f"{index:064x}", "source_id": "arxiv", "kind": "preprint",
         "title": f"Paper {index}", "url": f"https://arxiv.org/abs/2609.{index:05d}",
         "published_at": "2026-09-20", "publication_year": 2026,
         "date_basis": "published", "summary": None}
        for index in range(1, 406)
    ]
    result = parse_result({"signals": [], "publications": publications[:200],
                           "top_publications": publications[:15], "publication_total": 405})
    status = AnalysisStatus("succeeded", "run-1", "topic", "fast", result)
    calls = []

    def current(_self):
        return status

    def page(_self, run_id, *, offset, expected_total, known_ids):
        assert (run_id, expected_total) == ("run-1", 405)
        assert offset in {200, 300, 400}
        assert len(known_ids) <= 300
        calls.append(offset)
        return tuple(map(parse_publication, publications[offset:offset + 100]))

    with patch.object(ApiClient, "current_analysis", current), \
            patch.object(ApiClient, "publication_page", page):
        app = AppTest.from_file(WEB).run()
        assert not app.exception
        assert "Показано 200 из 405 найденных публикаций" in "\n".join(
            item.value for item in app.markdown)
        next(button for button in form_buttons(app) if button.label == "Следующая страница").click().run()
        assert not app.exception
        markup = "\n".join(item.value for item in app.markdown)
        assert "Показаны первые 200 и публикации 201–300 из 405 найденных" in markup
        assert ">Paper 201</a>" in markup
        assert markup.count('class="ta-card ta-publication-card"') == 300
        next(button for button in form_buttons(app) if button.label == "Следующая страница").click().run()
        assert not app.exception
        markup = "\n".join(item.value for item in app.markdown)
        assert "Показаны первые 200 и публикации 301–400 из 405 найденных" in markup
        assert ">Paper 301</a>" in markup and ">Paper 201</a>" not in markup
        assert '<span class="ta-index">301</span>' in markup
        assert markup.count('class="ta-card ta-publication-card"') == 300
        next(item for item in app.segmented_control if item.label == "Тема").set_value("Тёмная").run()
        assert "Показаны первые 200 и публикации 301–400 из 405 найденных" in "\n".join(
            item.value for item in app.markdown)
        next(button for button in form_buttons(app) if button.label == "Следующая страница").click().run()
        markup = "\n".join(item.value for item in app.markdown)
        assert "Показаны первые 200 и публикации 401–405 из 405 найденных" in markup
        assert markup.count('class="ta-card ta-publication-card"') == 205
        assert not any(button.label == "Следующая страница" for button in form_buttons(app))
        next(button for button in form_buttons(app) if button.label == "Предыдущая страница").click().run()
        markup = "\n".join(item.value for item in app.markdown)
        assert "Показаны первые 200 и публикации 301–400 из 405 найденных" in markup
        next(button for button in form_buttons(app) if button.label == "Предыдущая страница").click().run()
        markup = "\n".join(item.value for item in app.markdown)
        assert "Показаны первые 200 и публикации 201–300 из 405 найденных" in markup
        assert ">Paper 301</a>" not in markup
        next(button for button in form_buttons(app) if button.label == "Предыдущая страница").click().run()
        markup = "\n".join(item.value for item in app.markdown)
        assert "Показано 200 из 405 найденных публикаций" in markup
        assert markup.count('class="ta-card ta-publication-card"') == 200
        assert calls == [200, 300, 400, 300, 200]
        status = AnalysisStatus("succeeded", "run-2", "other topic", "fast", result)
        app.run()
        markup = "\n".join(item.value for item in app.markdown)
        assert "Показано 200 из 405 найденных публикаций" in markup
        assert ">Paper 301</a>" not in markup
        assert any(button.label == "Следующая страница" for button in form_buttons(app))


def test_result_parser_accepts_year_only_scientific_publication():
    publication = {"publication_id": "a" * 64, "source_id": "crossref",
                   "kind": "journal-article", "title": "Year-only study",
                   "url": "https://doi.org/10.1234/year-only", "published_at": None,
                   "publication_year": 2025, "date_basis": "published", "summary": None}
    result = parse_result({"signals": [], "publications": [publication],
                           "top_publications": [publication], "publication_total": 1})
    assert result.top_publications[0].kind == "journal-article"
    assert result.top_publications[0].published_at is None
    assert result.top_publications[0].publication_year == 2025


def test_result_parser_rejects_top_publications_out_of_common_order():
    publications = [
        {"publication_id": f"{index:064x}", "source_id": "arxiv", "kind": "preprint",
         "title": f"Paper {index}", "url": f"https://arxiv.org/abs/2609.{index:05d}",
         "published_at": "2026-09-20", "publication_year": 2026,
         "date_basis": "published", "summary": None}
        for index in (1, 2)
    ]
    with pytest.raises(ApiError, match="несовместимые"):
        parse_result({"signals": [], "publications": publications,
                      "top_publications": publications[1:], "publication_total": 2})


def test_source_count_and_status_include_sources_beyond_display_limit():
    publications = [
        {"publication_id": f"{index:064x}", "source_id": "openalex", "kind": "publication",
         "title": f"Study {index}", "url": f"https://openalex.org/W{index}",
         "published_at": "2026-09-20", "publication_year": 2026,
         "date_basis": "published", "summary": None}
        for index in range(1, 201)
    ]
    coverage = [
        {"source_id": "openalex", "state": "complete", "scanned": 200,
         "accepted": 200, "limit_reached": False, "reason_code": None},
        {"source_id": "arxiv", "state": "complete", "scanned": 1,
         "accepted": 1, "limit_reached": False, "reason_code": None},
        {"source_id": "biorxiv", "state": "complete", "scanned": 1,
         "accepted": 0, "limit_reached": False, "reason_code": None},
        {"source_id": "gdelt", "state": "unavailable", "scanned": 0,
         "accepted": 0, "limit_reached": False, "reason_code": "empty_response"},
    ]
    result = parse_result({"signals": [], "publications": publications,
                           "top_publications": publications[:15], "publication_total": 201,
                           "source_coverage": coverage})
    status = AnalysisStatus("succeeded", "run-1", "topic", "fast", result)
    with patch.object(ApiClient, "current_analysis", return_value=status):
        app = AppTest.from_file(WEB).run()
    assert not app.exception
    markup = "\n".join(item.value for item in app.markdown)
    assert "Источников</dt><dd>2" in markup
    # Источники свёрнуты в одну строку-сводку, раскрываются по нажатию.
    assert markup.count('<details class="ta-sources"><summary>Источники') == 1
    assert "ответили 3 из 4 площадок" in markup
    assert "OpenAlex" in markup and "arXiv" in markup
    assert "bioRxiv" in markup and "GDELT" in markup
    assert "bioRxiv</strong><span>Запрос обработан</span><span>Записей: 0" in markup
    assert "GDELT</strong><span>Источник недоступен</span><span>Записей: 0" in markup


def test_early_and_weak_results_keep_their_status_and_source_links():
    result = parse_result({"incomplete_coverage": True, "signals": [
        sample(title="Ранняя технология", category="early_signal"),
        sample(title="Кандидат на проверку", category="weak_signal_candidate",
               source_urls=["https://example.org/weak"]),
        sample(title="Зарождающийся кандидат", category="emerging_candidate",
               source_urls=["https://example.org/emerging"])]})
    status = AnalysisStatus("succeeded", "run-1", "Тест", "fast", result)
    with patch.object(ApiClient, "current_analysis", return_value=status):
        app = AppTest.from_file(WEB).run()
    assert not app.exception
    markup = "\n".join(item.value for item in app.markdown)
    assert "Ранний сигнал · гипотеза" in markup
    assert "Кандидат в слабые сигналы · автоматическая оценка" in markup
    assert "Зарождающийся кандидат · автоматическая оценка" in markup
    assert "Сигналов анализа</dt><dd>3" in markup
    assert "https://example.org/weak" in markup
    assert "https://example.org/emerging" in markup
    assert "Охват источников неполный" not in markup


def test_failure_does_not_expose_internal_exception_or_previous_result():
    result = parse_result({"signals": [sample()]})
    status = AnalysisStatus("idle")

    def current(_self):
        return status

    def start(_self, query, mode):
        nonlocal status
        if query == "Первый запрос":
            status = AnalysisStatus("succeeded", "run-1", query, mode, result)
        else:
            status = AnalysisStatus("failed", "run-2", query, mode, error="Не удалось обработать результат.")
        return status

    with patch.object(ApiClient, "current_analysis", current), patch.object(ApiClient, "start_analysis", start):
        app = AppTest.from_file(WEB).run()
        app.text_area[0].set_value("Первый запрос")
        form_buttons(app)[0].click().run()
        app.text_area[0].set_value("Второй запрос")
        form_buttons(app)[0].click().run()
        assert not app.exception
        assert "Не удалось обработать" in app.error[0].value
        assert "secret" not in app.error[0].value
        assert not any("Тестовая карточка" in item.value for item in app.markdown)


def test_running_analysis_survives_reload_and_can_be_cancelled():
    status = AnalysisStatus("idle")

    def current(_self):
        return status

    def start(_self, query, mode):
        nonlocal status
        status = AnalysisStatus("running", "run-1", query, mode)
        return status

    def cancel(_self, run_id):
        nonlocal status
        assert run_id == "run-1"
        status = AnalysisStatus("cancelled", "run-1", status.query, status.mode)
        return status

    with patch.object(ApiClient, "current_analysis", current), \
            patch.object(ApiClient, "start_analysis", start), \
            patch.object(ApiClient, "cancel_analysis", cancel):
        app = AppTest.from_file(WEB, default_timeout=10).run()
        app.text_area[0].set_value("Новый материал")
        form_buttons(app)[0].click().run()
        assert app.text_area[0].disabled
        assert len(form_buttons(app)) == 1
        assert form_buttons(app)[0].label == "Отменить анализ"
        reloaded = AppTest.from_file(WEB, default_timeout=10).run()
        assert reloaded.text_area[0].disabled
        assert reloaded.text_area[0].value == "Новый материал"
        assert len(form_buttons(reloaded)) == 1
        form_buttons(reloaded)[0].click().run()
        assert any("Анализ отменён" in item.value for item in reloaded.info)
        assert not reloaded.text_area[0].disabled


def test_stop_control_is_disabled_while_cancellation_finishes():
    status = AnalysisStatus("cancelling", "run-1", "Новый материал", "fast")
    with patch.object(ApiClient, "current_analysis", return_value=status):
        app = AppTest.from_file(WEB).run()
    assert len(form_buttons(app)) == 1
    assert form_buttons(app)[0].label == "Отменяем анализ"
    assert form_buttons(app)[0].disabled


@pytest.mark.parametrize("payload", [{}, {"signals": None},
                                     {"signals": [sample(summary="")]},
                                     {"signals": [sample(category="unknown")]},
                                     {"signals": [dict(sample(), category=None)]},
                                     {"signals": [sample()], "incomplete_coverage": "yes"}])
def test_malformed_response(payload):
    with pytest.raises(ApiError):
        parse_result(payload)


def test_status_parser_requires_a_result_for_completed_analysis():
    payload = {"id": "run-1", "query": "Тема", "mode": "fast", "state": "succeeded"}
    with pytest.raises(ApiError, match="не вернул результат"):
        parse_status(payload)
    assert parse_status({**payload, "result": {"signals": [sample()]}}).result is not None


def test_timeout_does_not_retry_or_expose_transport_details():
    client = ApiClient("http://127.0.0.1:8000")
    try:
        with patch.object(requests.Session, "post", side_effect=requests.Timeout("secret")) as post:
            with pytest.raises(ApiError, match="мог продолжиться"):
                client.analyze("Тест")
            assert post.call_count == 1
            assert post.call_args.kwargs["json"] == {"query": "Тест", "mode": "fast"}
    finally:
        client._executor.shutdown()


def test_second_request_rejected_while_first_runs():
    client = ApiClient("http://127.0.0.1:8000")
    release = Event()
    try:
        with patch.object(client, "analyze", side_effect=lambda *_: release.wait(5)):
            first = client.submit("Первый")
            with pytest.raises(ApiError, match="уже обрабатывает"):
                client.submit("Второй")
            release.set()
            first.result(timeout=5)
    finally:
        release.set()
        client._executor.shutdown()


CSS = WEB.with_name("web_styles.css").read_text(encoding="utf-8")

# Пары, которые обязаны держать контраст в обеих палитрах: текст на своей
# поверхности — 4.5:1, границы и кольцо фокуса — 3:1. Тот же бюджет, что и у
# настольной темы в tests/ui/test_theme.py.
TEXT_PAIRS = (("text", "surface"), ("text", "bg"), ("text", "raised"), ("text", "tonal"),
              ("text", "error-surface"), ("muted", "surface"), ("muted", "bg"),
              ("link", "surface"), ("link", "raised"), ("on-accent", "accent"))
NON_TEXT_PAIRS = (("accent", "bg"), ("accent", "surface"), ("border", "surface"),
                  ("border", "field"), ("error", "error-surface"))


def palette(dark: bool) -> dict:
    """Светлая палитра — первый блок :root, тёмная — блок в media-запросе."""
    block = CSS.split("@media (prefers-color-scheme: dark)")[1 if dark else 0]
    values = dict(re.findall(r"--ta-([a-z-]+):\s*(#[0-9A-Fa-f]{6})", block))
    return values if not dark else {**palette(False), **values}


def luminance(value: str) -> float:
    channels = [int(value[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in channels]
    return sum(v * w for v, w in zip(linear, (.2126, .7152, .0722), strict=True))


def ratio(foreground: str, background: str) -> float:
    dark, light = sorted((luminance(foreground), luminance(background)))
    return (light + .05) / (dark + .05)


@pytest.mark.parametrize("dark", [False, True])
def test_web_palette_keeps_the_contrast_budget(dark):
    colors = palette(dark)
    for pairs, floor in ((TEXT_PAIRS, 4.5), (NON_TEXT_PAIRS, 3.0)):
        for foreground, background in pairs:
            measured = ratio(colors[foreground], colors[background])
            assert measured >= floor, f"{foreground} на {background}: {measured:.2f} < {floor}"


def palette_block(marker: str) -> dict:
    """Палитра из блока, начинающегося с marker."""
    block = CSS.split(marker, 1)[1].split("}", 1)[0]
    return dict(re.findall(r"--ta-([a-z-]+):\s*([^;]+);", block))


def test_web_theme_overrides_match_the_base_palettes():
    """Копии для ручного выбора темы обязаны совпадать с исходными палитрами.

    Значения продублированы: системную тему задаёт media-запрос, а ручную —
    селектор :has(). Разойдясь, они дадут две разные тёмные темы.
    """
    assert palette_block(":root:has(.ta-theme-light) {") == palette_block(":root {")
    assert palette_block(":root:has(.ta-theme-dark) {") == palette_block(
        "@media (prefers-color-scheme: dark) {\n  :root {")


def test_mode_switch_offers_fast_and_deep_without_a_popup():
    app = AppTest.from_file(WEB).run()
    assert not app.exception
    control = next(item for item in app.segmented_control if item.label == "Режим анализа")
    # Два режима, по умолчанию выбран быстрый; подсказки со временем больше нет.
    assert len(control.options) == 2 and control.value == "fast"
    assert "ta-mode-note" not in " ".join(block.value for block in app.markdown)


@pytest.mark.parametrize(("pressed", "mode"), [(False, "fast"), (True, "deep")])
def test_the_pressed_button_starts_a_deep_analysis(pressed, mode):
    with patch.object(ApiClient, "start_analysis",
                      return_value=AnalysisStatus("queued", "run-1", "Тема", mode)) as start:
        app = AppTest.from_file(WEB).run()
        app.text_area[0].set_value("Тема")
        if pressed:
            next(item for item in app.segmented_control if item.label == "Режим анализа").set_value("deep")
        next(button for button in form_buttons(app) if button.label == "Анализировать").click().run()
    assert not app.exception
    start.assert_called_once_with("Тема", mode)


@pytest.mark.parametrize(("elapsed", "expected", "state", "clock", "left"), [
    (30, 600, "running", "00:30", "осталось около 10 мин"),
    (240, 600, "running", "04:00", "осталось около 6 мин"),
    (570, 600, "running", "09:30", "осталось меньше минуты"),
    (900, 600, "running", "15:00", "дольше прогноза (10 мин)"),
    (240, None, "running", "04:00", ""),
    (240, 600, "cancelling", "04:00", ""),
    (3725, None, "running", "1:02:05", ""),
])
def test_run_panel_clock_counts_elapsed_time_against_the_forecast(elapsed, expected, state, clock, left):
    from app.ui.web import run_markup

    status = AnalysisStatus(state, "run-1", "Тема", "deep", elapsed_seconds=elapsed,
                            expected_seconds=expected)
    markup = run_markup(status, 0)
    assert f'<b data-ta-clock="run:run-1" data-ta-elapsed="{elapsed}">{clock}</b>' in markup
    assert (f'<span class="ta-run-left">{left}</span>' in markup) if left else "ta-run-left" not in markup


def test_run_panel_marks_passed_current_and_upcoming_phases_without_going_back():
    from app.ui.web import RUN_FACTS, RUN_PHASES, run_markup, run_phase

    status = AnalysisStatus("running", "run-1", "<b>Тема</b>", "fast", stage="labels",
                            message="Проверяем названия и границы кандидатов: 3 из 16", completed=3, total=16)
    phase = run_phase(status)
    assert phase == 2
    markup = run_markup(status, phase)
    steps = re.findall(r'<li class="ta-step ta-step-(\w+)"', markup)
    assert steps == ["done", "done", "active", "next", "next"]
    assert '<li class="ta-step ta-step-active" aria-current="step">' in markup
    assert "Проверяем названия и границы кандидатов: 3 из 16" in markup and "3 из 16</span>" in markup
    assert f'<p class="ta-step-hint">{RUN_PHASES[3][1]}</p>' in markup
    assert "&lt;b&gt;Тема&lt;/b&gt;" in markup
    assert f'<p class="ta-fact">{RUN_FACTS[0]}</p>' in markup
    later = AnalysisStatus("running", "run-1", "Тема", "fast", stage="relevance", elapsed_seconds=8)
    later_markup = run_markup(later, run_phase(later), 0)
    # The next fact after eight seconds; a stage without a counter shows its own clock.
    assert f'<p class="ta-fact">{RUN_FACTS[1]}</p>' in later_markup
    assert 'на этапе <b data-ta-clock="stage:run-1:relevance" data-ta-elapsed="0">' in later_markup
    # A late stage that tops up an earlier collection does not move the ribbon back.
    assert run_phase(AnalysisStatus("running", "run-1", "Тема", "fast", stage="discovery"), 3) == 3
    assert run_phase(AnalysisStatus("queued", "run-1", "Тема", "fast")) == -1
    assert "Ставим анализ в очередь" in run_markup(AnalysisStatus("queued", "run-1", "Тема", "fast"), -1)


def test_elapsed_clock_survives_theme_switch_and_resets_for_new_analysis_topic():
    from app.ui.web import RUN_CLOCK

    assert "performance.now()" in RUN_CLOCK and "sleepOffset" in RUN_CLOCK
    status = AnalysisStatus("running", "run-1", "Первая тема", "fast", stage="labels",
                            elapsed_seconds=30)

    def current(_self):
        return status

    with patch.object(ApiClient, "current_analysis", current):
        app = AppTest.from_file(WEB).run()

        def panel() -> str:
            return next(item.value for item in app.markdown if '<section class="ta-run"' in item.value)

        assert not app.exception
        assert 'data-ta-clock="run:run-1" data-ta-elapsed="30">00:30' in panel()

        # A full rerender after a theme change must use the current API time,
        # even when the backend's elapsed value has been corrected meanwhile.
        status = AnalysisStatus("running", "run-1", "Первая тема", "fast", stage="labels",
                                elapsed_seconds=90)
        next(item for item in app.segmented_control if item.label == "Тема").set_value("Тёмная").run()
        assert not app.exception
        assert 'data-ta-clock="run:run-1" data-ta-elapsed="90">01:30' in panel()

        status = AnalysisStatus("running", "run-2", "Вторая тема", "deep", stage="history",
                                elapsed_seconds=3)
        app.run()
        assert not app.exception
        assert 'data-ta-clock="run:run-2" data-ta-elapsed="3">00:03' in panel()
        assert 'data-ta-clock="stage:run-2:history"' in panel()
        assert 'Первая тема' not in panel()


def test_openalex_budget_refusal_is_explained_instead_of_an_unexplained_empty_top():
    result = parse_result({"incomplete_coverage": True, "openalex_rate_limited": True, "signals": []})
    assert result.openalex_rate_limited
    status = AnalysisStatus("succeeded", "run-1", "Тема", "fast", result)
    with patch.object(ApiClient, "current_analysis", return_value=status):
        app = AppTest.from_file(WEB).run()
    assert not app.exception
    assert any("ключ OpenAlex" in item.value for item in app.markdown)
    with pytest.raises(ApiError):
        parse_result({"signals": [], "openalex_rate_limited": "yes"})


def test_signal_card_shows_the_confidence_in_the_trend():
    from app.ui.web import confidence_markup

    result = parse_result({"signals": [sample(confidence="low", growth_confirmed=False,
                                              checked_features=["novelty"],
                                              unchecked_features=["growth", "persistence",
                                                                  "independence", "application"])]})
    markup = confidence_markup(result.signals[0])
    assert '<span class="ta-meter-label">Уверенность в тренде</span>' in markup
    assert '<b class="ta-meter-value">низкая</b>' in markup
    assert markup.count('class="on"') == 1
    assert ">Проверено 1 из 5 признаков. Не проверены: рост, устойчивость, независимые группы, применение.<" in markup
    measured = parse_result({"signals": [sample(confidence="medium", growth_confirmed=False,
                                                checked_features=["growth", "persistence", "novelty"],
                                                unchecked_features=["independence", "application"])]})
    assert (">Проверено 3 из 5 признаков. Рост не подтверждён. Не проверены: независимые группы, применение.<"
            in confidence_markup(measured.signals[0]))
    assert confidence_markup(parse_result({"signals": [sample()]}).signals[0]) == ""
    status = AnalysisStatus("succeeded", "run-1", "Тема", "fast", result)
    with patch.object(ApiClient, "current_analysis", return_value=status):
        app = AppTest.from_file(WEB).run()
    assert not app.exception
    assert any("Уверенность в тренде" in item.value for item in app.markdown)


@pytest.mark.parametrize("changes", [
    {"confidence": "certain"}, {"confidence": "low", "growth_confirmed": "no"},
    {"confidence": "low", "checked_features": ["hype"]},
    {"confidence": "low", "checked_features": ["growth"], "unchecked_features": ["growth"]},
])
def test_invalid_trend_confidence_is_rejected(changes):
    with pytest.raises(ApiError, match="уверенность"):
        parse_result({"signals": [sample(**changes)]})


def test_signals_outside_top_do_not_displace_ranked_publications():
    publications = [
        {"publication_id": f"{index:064x}", "source_id": "crossref", "kind": "journal-article",
         "title": f"Paper {index}", "url": f"https://doi.org/10.1234/PAPER{index}",
         "published_at": "2026-09-20", "publication_year": 2026, "date_basis": "published",
         "summary": None}
        for index in range(1, 21)
    ]
    result = parse_result({
        "publications": publications, "top_publications": publications[:15], "publication_total": 20,
        "signals": [sample(title="Halide solid electrolytes", category="weak_signal_candidate",
                           source_urls=["https://doi.org/10.1234/paper18"], confidence="low",
                           checked_features=[], unchecked_features=["growth", "persistence", "novelty",
                                                                    "independence", "application"]),
                    sample(title="Signal without a listed paper", source_urls=["https://example.org/other"])]})
    status = AnalysisStatus("succeeded", "run-1", "Тема", "fast", result)
    with patch.object(ApiClient, "current_analysis", return_value=status):
        app = AppTest.from_file(WEB).run()
    assert not app.exception
    markup = "\n".join(item.value for item in app.markdown)
    assert "ТОП-15 публикаций" not in markup
    grids = [item.value for item in app.markdown if 'class="ta-grid"' in item.value]
    assert len(grids) == 1
    top = re.findall(r"<article .*?</article>", grids[0], flags=re.S)
    assert len(top) == 15 and "Paper 1" in top[0] and "Paper 15" in top[-1]
    assert all("Paper 18" not in card for card in top)
    assert "Другие сигналы анализа" in markup
    other = [item.value for item in app.markdown if 'class="ta-grid ta-signal-grid"' in item.value]
    assert len(other) == 1
    signal_cards = re.findall(r"<article .*?</article>", other[0], flags=re.S)
    assert len(signal_cards) == 2
    assert "Paper 18" in signal_cards[0] and "Halide solid electrolytes" in signal_cards[0]
    assert "Уверенность в тренде" in signal_cards[0]
    assert "Signal without a listed paper" in signal_cards[1]
    assert "Остальные публикации в списке (4)" in markup


def test_signal_on_ranked_publication_stays_on_that_card_without_changing_order():
    publications = [
        {"publication_id": f"{index:064x}", "source_id": "crossref", "kind": "journal-article",
         "title": f"Paper {index}", "url": f"https://doi.org/10.1234/paper{index}",
         "published_at": "2026-09-20", "publication_year": 2026, "date_basis": "published",
         "summary": None}
        for index in (1, 2)
    ]
    result = parse_result({"signals": [sample(source_urls=["https://doi.org/10.1234/paper2"])],
                           "publications": publications, "top_publications": publications,
                           "publication_total": 2})
    status = AnalysisStatus("succeeded", "run-1", "Тема", "fast", result)
    with patch.object(ApiClient, "current_analysis", return_value=status):
        app = AppTest.from_file(WEB).run()
    assert not app.exception
    markup = "\n".join(item.value for item in app.markdown)
    grid = next(item.value for item in app.markdown if 'class="ta-grid"' in item.value)
    cards = re.findall(r"<article .*?</article>", grid, flags=re.S)
    assert len(cards) == 2 and "Paper 1" in cards[0] and "Paper 2" in cards[1]
    assert "Сигнал анализа" in cards[1] and "Сигнал анализа" not in cards[0]
    assert "Другие сигналы анализа" not in markup


def test_every_top_card_shows_the_confidence_and_the_top_is_one_grid():
    trend = {"title": "Solid electrolytes", "confidence": "medium", "growth_confirmed": True,
             "checked_features": ["growth", "persistence"],
             "unchecked_features": ["novelty", "independence", "application"]}
    publications = [
        {"publication_id": f"{index:064x}", "source_id": "crossref", "kind": "journal-article",
         "title": f"Paper {index}", "url": f"https://doi.org/10.1234/paper{index}",
         "published_at": "2026-09-20", "publication_year": 2026, "date_basis": "published",
         "summary": None, **({"trend": trend} if index == 1 else {})}
        for index in range(1, 18)
    ]
    result = parse_result({"signals": [], "publications": publications,
                           "top_publications": publications[:15], "publication_total": 40})
    assert result.publications[0].trend.title == "Solid electrolytes"
    assert result.publications[1].trend is None
    status = AnalysisStatus("succeeded", "run-1", "Тема", "fast", result)
    with patch.object(ApiClient, "current_analysis", return_value=status):
        app = AppTest.from_file(WEB).run()
    assert not app.exception
    # Все плитки ТОПа — в одном блоке-сетке, остальные — в свёрнутом списке.
    grids = [item.value for item in app.markdown if 'class="ta-grid"' in item.value]
    assert len(grids) == 1
    top = re.findall(r"<article .*?</article>", grids[0], flags=re.S)
    assert len(top) == 15
    assert all("Соответствие запросу" in card for card in top)
    assert "Уверенность в тренде" in top[0]
    assert '<b class="ta-meter-value">средняя</b>' in top[0] and "Проверено 2 из 5 признаков." in top[0]
    assert "Тема: <b>Solid electrolytes</b>" in top[0]
    assert all("Уверенность в тренде" not in card for card in top[1:])
    assert all('<b class="ta-meter-value">не оценено</b>' in card for card in top)
    # Плитка целиком — ссылка на материал.
    assert '<h3><a href="https://doi.org/10.1234/paper1"' in top[0]
    markup = "\n".join(item.value for item in app.markdown)
    rest = markup[markup.index('<div class="ta-more">'):]
    assert "Paper 16" in rest and "не оценено" not in rest
    # Подпись о лимите лежит в одном блоке со списком и не налезает на него.
    assert "Показано 17 из 40 найденных публикаций" in rest


def test_publication_model_confidence_is_distinct_from_trend_and_keeps_server_order():
    publications = [
        {"publication_id": f"{index:064x}", "source_id": "crossref", "kind": "journal-article",
         "title": f"Paper {index}", "url": f"https://doi.org/10.1234/paper{index}",
         "published_at": "2026-09-20", "publication_year": 2026, "date_basis": "published",
         "summary": "Solid electrolyte storage works.",
         "model_confidence": {"score": score, "reason": "Relevant <research> title.",
                              "evidence_quote": "Solid electrolyte storage",
                              "basis": "title_and_summary"}}
        for index, score in ((1, 91), (2, 22))
    ]
    publications[0]["trend"] = {"title": "Electrolyte research", "confidence": "medium",
                                 "growth_confirmed": False, "checked_features": [],
                                 "unchecked_features": ["growth"]}
    result = parse_result({"signals": [], "publications": publications,
                           "top_publications": publications, "publication_total": 2})
    assert result.publications[0].model_confidence.score == 91
    status = AnalysisStatus("succeeded", "run-1", "Тема", "fast", result)
    with patch.object(ApiClient, "current_analysis", return_value=status):
        app = AppTest.from_file(WEB).run()
    assert not app.exception
    grids = [item.value for item in app.markdown if 'class="ta-grid"' in item.value]
    cards = re.findall(r"<article .*?</article>", grids[0], flags=re.S)
    assert len(cards) == 2 and "Paper 1" in cards[0] and "Paper 2" in cards[1]
    assert '<span class="ta-meter-label">Соответствие запросу</span>' in cards[0]
    assert '<b class="ta-meter-value">91<small>/100</small></b>' in cards[0]
    assert "Уверенность в тренде" in cards[0]
    assert "Основание оценки" in cards[0]
    assert 'class="ta-scale" style="--ta-value: 91%" aria-hidden="true"' in cards[0]
    assert cards[0].count("<li>") == 3
    assert "E5: лучший результат с исходным или английским запросом" in cards[0]
    assert "Текст: заголовок и начало описания" in cards[0]
    assert "0,75 → 0; 0,90 → 100" in cards[0]
    assert "Дата, тип источника и тренд не учитываются" in cards[0]
    assert "Relevant &lt;research&gt; title." not in cards[0]
    assert "Фрагмент оригинала:" not in cards[0]
    assert "<blockquote>" not in cards[0]
    assert '<b class="ta-meter-value">22<small>/100</small></b>' in cards[1]
    assert "Уверенность в тренде" not in cards[1]


@pytest.mark.parametrize("model_confidence", [
    {"score": True, "reason": "Relevant", "evidence_quote": "Study", "basis": "title"},
    {"score": -1, "reason": "Relevant", "evidence_quote": "Study", "basis": "title"},
    {"score": 101, "reason": "Relevant", "evidence_quote": "Study", "basis": "title"},
    {"score": 50, "reason": " ", "evidence_quote": "Study", "basis": "title"},
    {"score": 50, "reason": "Relevant", "evidence_quote": " ", "basis": "title"},
    {"score": 50, "reason": "Relevant", "evidence_quote": "Study", "basis": "abstract"},
    {"score": 50, "reason": "Relevant", "evidence_quote": "Study", "basis": []},
])
def test_invalid_publication_model_confidence_is_rejected(model_confidence):
    publication = {"publication_id": "a" * 64, "source_id": "crossref", "kind": "journal-article",
                   "title": "Study", "url": "https://doi.org/10.1234/study", "published_at": None,
                   "publication_year": 2025, "date_basis": "published", "summary": None,
                   "model_confidence": model_confidence}
    with pytest.raises(ApiError, match="оценку модели"):
        parse_result({"signals": [], "publications": [publication], "top_publications": [publication],
                      "publication_total": 1})


@pytest.mark.parametrize("trend", [
    {"title": "", "confidence": "low"}, {"title": "Topic", "confidence": None},
    {"title": "Topic", "confidence": "certain"}, "Topic",
])
def test_invalid_publication_trend_is_rejected(trend):
    publication = {"publication_id": "a" * 64, "source_id": "crossref", "kind": "journal-article",
                   "title": "Study", "url": "https://doi.org/10.1234/study", "published_at": None,
                   "publication_year": 2025, "date_basis": "published", "summary": None, "trend": trend}
    with pytest.raises(ApiError):
        parse_result({"signals": [], "publications": [publication], "top_publications": [publication],
                      "publication_total": 1})


def translated_status(translation):
    from app.ui.api_client import parse_translation

    publications = [
        {"publication_id": f"{index:064x}", "source_id": "crossref", "kind": "journal-article",
         "title": f"Paper {index}", "url": f"https://doi.org/10.1234/paper{index}",
         "published_at": "2026-09-20", "publication_year": 2026, "date_basis": "published",
         "summary": "Original abstract." if index == 1 else None}
        for index in range(1, 3)
    ]
    result = parse_result({"signals": [], "publications": publications,
                           "top_publications": publications, "publication_total": 2})
    return AnalysisStatus("succeeded", "run-1", "Тема", "fast", result,
                          translation=parse_translation(translation))


def test_top_cards_show_the_russian_draft_and_keep_the_original_at_hand():
    status = translated_status({"state": "ready", "completed": 2, "total": 2,
                                "texts": {"Paper 1": "Статья <1>", "Original abstract.": "Исходная аннотация."}})
    with patch.object(ApiClient, "current_analysis", return_value=status):
        app = AppTest.from_file(WEB).run()
        assert not app.exception
        markup = "\n".join(item.value for item in app.markdown)
        assert 'title="Paper 1"><span lang="ru">Статья &lt;1&gt;</span></a>' in markup
        assert '<span lang="ru">Исходная аннотация.</span>' in markup
        assert ">Paper 2</a>" in markup  # Untranslated text stays as it was.
        assert "Машинный перевод локальной моделью" not in markup  # Пояснение под переключателем убрано.
        app.segmented_control(key="top_language").set_value("Оригинал").run()
        assert not app.exception
        markup = "\n".join(item.value for item in app.markdown)
        assert ">Paper 1</a>" in markup and "Статья" not in markup and "Машинный перевод" not in markup


def test_visitor_switches_the_top_to_another_installed_language():
    languages = [{"code": "ru", "name": "Русский"}, {"code": "de", "name": "Немецкий"}]
    russian = translated_status({"state": "ready", "completed": 2, "total": 2, "language": "ru",
                                 "languages": languages, "texts": {"Paper 1": "Статья 1"}})
    german = translated_status({"state": "ready", "completed": 2, "total": 2, "language": "de",
                                "languages": languages, "texts": {"Paper 1": "Aufsatz 1"}})
    requested = []

    def current(_client, language=None):
        requested.append(language)
        return german if language == "de" else russian

    with patch.object(ApiClient, "current_analysis", current):
        app = AppTest.from_file(WEB).run()
        assert not app.exception
        control, = [control for control in app.segmented_control if control.key == "top_language"]
        assert list(control.options) == ["Русский", "Немецкий", "Оригинал"]
        control.set_value("Немецкий").run()
        assert not app.exception
        markup = "\n".join(item.value for item in app.markdown)
    assert '<span lang="de">Aufsatz 1</span>' in markup
    assert requested[0] is None and requested[-1] == "de"


@pytest.mark.parametrize("translation, text", [
    ({"state": "running", "completed": 1, "total": 4}, "Переводим ТОП на русский: 1 из 4"),
    ({"state": "unavailable", "message": "Модель перевода отсутствует."},
     "Перевод ТОПа недоступен: Модель перевода отсутствует."),
])
def test_top_stays_in_the_original_until_its_translation_is_ready(translation, text):
    status = translated_status(translation)
    with patch.object(ApiClient, "current_analysis", return_value=status):
        app = AppTest.from_file(WEB).run()
    assert not app.exception
    markup = "\n".join(item.value for item in app.markdown)
    assert text in markup
    assert ">Paper 1</a>" in markup and 'lang="ru"' not in markup
    assert not [control for control in app.segmented_control if control.key == "top_language"]


@pytest.mark.parametrize("translation", [
    {"state": "done"}, {"state": "ready", "texts": {"a": 1}},
    {"state": "running", "completed": 3, "total": 2}, {"state": "ready", "texts": ["a"]},
])
def test_invalid_top_translation_is_rejected(translation):
    from app.ui.api_client import parse_translation

    with pytest.raises(ApiError, match="перевод"):
        parse_translation(translation)


def test_titles_are_shown_in_russian_while_the_abstracts_are_still_translating():
    status = translated_status({"state": "running", "completed": 2, "total": 3,
                                "texts": {"Paper 1": "Статья 1", "Paper 2": "Статья 2"}})
    with patch.object(ApiClient, "current_analysis", return_value=status):
        app = AppTest.from_file(WEB).run()
    assert not app.exception
    markup = "\n".join(item.value for item in app.markdown)
    assert "Переводим аннотации: 2 из 3" in markup
    assert '<span lang="ru">Статья 1</span>' in markup
    assert ">Original abstract.<" in markup  # The abstract waits for its own step.
    assert [control for control in app.segmented_control if control.key == "top_language"]


def history_page(*, current_id=None, has_more=False):
    from datetime import UTC, datetime

    from app.ui.api_client import HistoryEntry, HistoryPage

    return HistoryPage((
        HistoryEntry("run-2", "<b>solid</b> state", "deep", "succeeded", datetime(2026, 9, 26, 12, tzinfo=UTC)),
        HistoryEntry("run-1", "Прерванный запрос", None, "interrupted", datetime(2026, 9, 25, 12, tzinfo=UTC)),
    ), 0, has_more, current_id)


def open_history(app):
    next(button for button in app.button if button.key == "history_toggle").click().run()
    return {button.key: button for button in app.button}


def saved_status(run_id="run-2", query="<b>solid</b> state"):
    return parse_status({"state": "succeeded", "id": run_id, "query": query, "mode": "deep",
                         "created_at": "2026-09-26T12:00:00+00:00", "result": {"signals": [sample()]}})


def test_history_opens_a_finished_analysis_as_its_own_page_and_leads_back():
    asked = []

    def saved(_self, run_id):
        asked.append(run_id)
        return saved_status(run_id)

    with patch.object(ApiClient, "history", return_value=history_page(has_more=True)), \
            patch.object(ApiClient, "saved_analysis", saved):
        app = AppTest.from_file(WEB).run()
        buttons = open_history(app)
        assert not app.exception
        markup = "\n".join(item.value for item in app.markdown)
        assert "&lt;b&gt;solid&lt;/b&gt; state" in markup and "<b>solid</b>" not in markup
        assert "Глубокий" in markup and "Прерван" in markup
        # Only a finished run can be opened; the older page is offered.
        assert "ta-hopen-run-2" in buttons and "ta-hopen-run-1" not in buttons
        assert buttons["history_newer"].disabled and not buttons["history_older"].disabled
        buttons["ta-hopen-run-2"].click().run()
        assert not app.exception
        assert app.query_params["run"] == ["run-2"] and asked and set(asked) == {"run-2"}
        assert not app.session_state["history_open"]
        markup = "\n".join(item.value for item in app.markdown)
        assert "Анализ из истории" in markup and "&lt;b&gt;solid&lt;/b&gt; state</h1>" in markup
        assert ">Тестовая карточка<" in markup or "Тестовая карточка" in markup
        assert [button.key for button in app.button] == ["history_toggle", "leave_saved"]
        # The saved page survives a reload: its address carries the run.
        reloaded = AppTest.from_file(WEB)
        reloaded.query_params["run"] = "run-2"
        reloaded.run()
        assert "Анализ из истории" in "\n".join(item.value for item in reloaded.markdown)
        next(button for button in app.button if button.key == "leave_saved").click().run()
        assert "run" not in app.query_params
        assert [button.key for button in app.button] == ["history_toggle", "start_analysis"]


def test_history_opens_saved_results_while_another_analysis_runs():
    from datetime import UTC, datetime

    from app.ui.api_client import HistoryEntry, HistoryPage

    running = HistoryEntry("run-3", "Новый материал", "fast", "running", datetime(2026, 9, 27, 9, tzinfo=UTC))
    page = history_page(current_id="run-3")
    page = HistoryPage((running, *page.entries), 0, False, "run-3")
    status = AnalysisStatus("running", "run-3", "Новый материал", "fast")
    with patch.object(ApiClient, "current_analysis", return_value=status), \
            patch.object(ApiClient, "history", return_value=page), \
            patch.object(ApiClient, "saved_analysis", lambda _self, run_id: saved_status(run_id)):
        app = AppTest.from_file(WEB).run()
        buttons = open_history(app)
        markup = "\n".join(item.value for item in app.markdown)
        # The running analysis is the one on the main page; finished ones open anyway.
        assert "Открыт сейчас" in markup and "ta-hopen-run-3" not in buttons
        buttons["ta-hopen-run-2"].click().run()
        markup = "\n".join(item.value for item in app.markdown)
        assert "Сейчас идёт анализ «Новый материал»" in markup
        back = next(button for button in app.button if button.key == "leave_saved")
        assert back.label == "К идущему анализу"
        open_history(app)
        markup = "\n".join(item.value for item in app.markdown)
        # From a saved page the running analysis leads back to its progress.
        assert "ta-hopen-run-3" in {button.key for button in app.button}


def test_saved_page_and_history_failures_are_shown_with_a_way_back():
    with patch.object(ApiClient, "saved_analysis",
                      side_effect=ApiError("У этого анализа нет готового результата.")):
        app = AppTest.from_file(WEB)
        app.query_params["run"] = "run-9"
        app.run()
    assert not app.exception
    assert "нет готового результата" in app.error[0].value
    assert "leave_saved" in {button.key for button in app.button}
    with patch.object(ApiClient, "history", side_effect=ApiError("Не удалось прочитать историю анализов.")):
        app = AppTest.from_file(WEB).run()
        open_history(app)
    assert "историю" in app.error[0].value


def test_history_dates_read_as_words_near_today():
    from datetime import date, datetime

    from app.ui.web import history_date

    today = date(2026, 9, 27)
    assert history_date(datetime(2026, 9, 27, 9, 5).astimezone(), today) == "сегодня, 09:05"
    assert history_date(datetime(2026, 9, 26, 23, 59).astimezone(), today) == "вчера, 23:59"
    assert history_date(datetime(2026, 5, 3, 7, 0).astimezone(), today) == "3 мая, 07:00"
    assert history_date(datetime(2025, 12, 31, 18, 30).astimezone(), today) == "31 дек 2025, 18:30"


@pytest.mark.parametrize("payload", [
    {"offset": 20, "has_more": False, "runs": []},
    {"offset": 0, "has_more": "no", "runs": []},
    {"offset": 0, "has_more": False, "runs": [{"id": "../x", "query": "q", "mode": None,
                                                "state": "succeeded", "created_at": "2026-09-26T12:00:00"}]},
    {"offset": 0, "has_more": False, "runs": [{"id": "a", "query": "q", "mode": "slow",
                                                "state": "succeeded", "created_at": "2026-09-26T12:00:00"}]},
    {"offset": 0, "has_more": False, "runs": [{"id": "a", "query": "q", "mode": None,
                                                "state": "succeeded", "created_at": "вчера"}]},
    {"offset": 0, "has_more": False, "runs": [{"id": "a", "query": "q", "mode": None, "state": "succeeded",
                                                "created_at": "2026-09-26T12:00:00"}] * 2},
])
def test_invalid_history_pages_are_rejected(payload):
    from app.ui.api_client import parse_history

    with pytest.raises(ApiError, match="историю"):
        parse_history(payload, offset=0, limit=20)


def test_a_visitor_pass_fits_only_this_launch_password():
    from app.ui.web import visitor_from_pass, visitor_pass

    password = "launch-password-0123456789"
    token = visitor_pass("visitor-abcdefghijklmnop", password)
    assert visitor_from_pass(token, password) == "visitor-abcdefghijklmnop"
    # A new launch has a new password; an edited or foreign value is no pass.
    assert visitor_from_pass(token, "next-launch-password-0123") is None
    forged = token[:-1] + ("0" if token[-1] != "0" else "1")
    for value in (forged, "visitor-abcdefghijklmnop", "../bad.abc", None, "x" * 300):
        assert visitor_from_pass(value, password) is None


def test_each_login_gets_its_own_visitor_for_every_api_call():
    from app.ui.api_client import VISITOR

    seen = []

    def current(self):
        seen.append(self._headers.get("X-Trend-Visitor"))
        return AnalysisStatus("idle")

    environment = {"TREND_WEB_REQUIRE_AUTH": "1", "TREND_WEB_ACCESS_PASSWORD": "correct-horse-demo-password"}
    visitors = []
    with patch.dict(os.environ, environment), patch.object(ApiClient, "current_analysis", current):
        for _ in range(2):
            app = AppTest.from_file(WEB).run()
            assert "видны только в этом браузере" in "\n".join(item.value for item in app.markdown)
            app.text_input[0].set_value("correct-horse-demo-password")
            form_buttons(app)[0].click().run()
            assert not app.exception
            visitors.append(app.session_state["visitor"])
    assert all(VISITOR.fullmatch(visitor) for visitor in visitors) and visitors[0] != visitors[1]
    assert set(seen) == set(visitors)


def test_a_busy_service_explains_itself_without_the_other_query():
    status = AnalysisStatus("idle", service_busy=True)
    with patch.object(ApiClient, "current_analysis", return_value=status):
        app = AppTest.from_file(WEB).run()
    assert not app.exception
    assert next(button for button in app.button if button.key == "start_analysis").disabled
    markup = "\n".join(item.value for item in app.markdown)
    assert "очередь заполнена" in markup and "страница обновится сама" in markup


def test_full_slots_explain_that_a_new_analysis_will_wait_its_turn():
    status = AnalysisStatus("idle", slots_full=True, queue_length=2)
    with patch.object(ApiClient, "current_analysis", return_value=status):
        app = AppTest.from_file(WEB).run()
    assert not app.exception
    assert not next(button for button in app.button if button.key == "start_analysis").disabled
    markup = "\n".join(item.value for item in app.markdown)
    assert "встанет в очередь — перед ним 2 анализа" in markup


def test_waiting_analysis_shows_its_place_in_the_queue_and_can_be_cancelled():
    status = AnalysisStatus("waiting", "wait-0123456789abcdef0123", "очередь", "fast", queue_position=3,
                            elapsed_seconds=12)
    with patch.object(ApiClient, "current_analysis", return_value=status):
        app = AppTest.from_file(WEB).run()
    assert not app.exception
    markup = "\n".join(item.value for item in app.markdown)
    assert "Ваш анализ в очереди" in markup and "перед вами 2 анализа" in markup
    assert any(button.key == "cancel_analysis" for button in app.button)


def test_address_behind_the_tunnel_is_the_last_external_one_in_the_chain():
    from app.ui.web import forwarded_address

    # Сервер туннеля пишет адрес человека, клиент туннеля дописывает свой 127.0.0.1;
    # подставной адрес, присланный самим посетителем, стоит левее.
    assert forwarded_address("8.8.8.8, 77.88.55.60, 127.0.0.1") == "77.88.55.60"
    assert forwarded_address("77.88.55.60", None) == "77.88.55.60"
    assert forwarded_address("192.168.1.5, 127.0.0.1") == "192.168.1.5"
    assert forwarded_address("garbage", "") is None


def test_blocked_browser_sees_only_the_closed_door():
    with patch.object(ApiClient, "check_access", return_value="blocked"), \
            patch.object(ApiClient, "current_analysis") as current:
        app = AppTest.from_file(WEB).run()
        assert not app.exception
        assert "Доступ закрыт" in "".join(block.value for block in app.markdown)
        assert not app.text_area and not form_buttons(app)
        current.assert_not_called()


def test_paused_service_explains_itself_and_keeps_the_start_button_off():
    with patch.object(ApiClient, "current_analysis", return_value=AnalysisStatus("idle", paused=True)):
        app = AppTest.from_file(WEB).run()
        assert not app.exception
        assert "приостановил новые анализы" in "".join(block.value for block in app.markdown)
        assert form_buttons(app)[0].disabled


def test_owner_message_reaches_the_visitor_page_until_dismissed():
    answer = {"access": None, "messages": ["Сервис перезапустится через 5 минут <b>"]}
    with patch.object(ApiClient, "report_presence", return_value=answer):
        app = AppTest.from_file(WEB).run()
    assert not app.exception
    markup = "".join(block.value for block in app.markdown)
    assert "Сообщение от владельца сервиса" in markup and "5 минут &lt;b&gt;" in markup
    app.button(key="owner_note_close").click().run()
    assert "Сообщение от владельца" not in "".join(block.value for block in app.markdown)


def test_signed_out_guest_returns_to_the_password_screen():
    environment = {"TREND_WEB_REQUIRE_AUTH": "1", "TREND_WEB_ACCESS_PASSWORD": "correct-horse-demo-password"}
    with patch.dict(os.environ, environment, clear=False), \
            patch.object(ApiClient, "check_access", return_value=None) as access:
        app = AppTest.from_file(WEB).run()
        app.text_input[0].set_value("correct-horse-demo-password")
        form_buttons(app)[0].click().run()
        assert app.text_area and not app.text_input
        access.return_value = "signed_out"
        app.run()
        assert not app.exception
        assert len(app.text_input) == 1 and not app.text_area
        assert app.session_state["cookie_revoked"] is True
