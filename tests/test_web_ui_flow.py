"""Пользовательские сценарии Streamlit без моделей, сети и работающего API."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from app.ui.api_client import ApiClient, ApiError, AnalysisStatus, parse_result


WEB = Path(__file__).resolve().parents[1] / "app/ui/web.py"
PASSWORD = "correct-horse-demo-password"


def form_buttons(app):
    """Кнопки страницы без постоянной кнопки «История» в углу."""
    return [button for button in app.button if button.key != "history_toggle"]


class FakeAnalysisService:
    """Состояние сервиса переживает создание новой сессии AppTest."""

    def __init__(self):
        self.status = AnalysisStatus("idle")
        self.starts = []
        self.cancels = 0
        self.cancel_requests = []
        self.current_error = None
        self.start_error = None
        self.cancel_error = None

    def current(self):
        if self.current_error:
            raise self.current_error
        return self.status

    def start(self, query, mode):
        self.starts.append((query, mode))
        if self.start_error:
            raise self.start_error
        self.status = AnalysisStatus("queued", "run-1", query, mode)
        return self.status

    def cancel(self, run_id):
        self.cancels += 1
        self.cancel_requests.append(run_id)
        if self.cancel_error:
            raise self.cancel_error
        if self.status.id != run_id:
            raise ApiError("Состояние анализа изменилось. Обновите страницу.")
        self.status = AnalysisStatus("cancelling", self.status.id, self.status.query, self.status.mode)
        return self.status


@pytest.fixture
def service():
    fake = FakeAnalysisService()
    with patch.dict(os.environ, {"TREND_WEB_REQUIRE_AUTH": "0"}), \
            patch.object(ApiClient, "current_analysis", fake.current), \
            patch.object(ApiClient, "start_analysis", fake.start), \
            patch.object(ApiClient, "cancel_analysis", fake.cancel),             patch.object(ApiClient, "mode_estimates", side_effect=ApiError("нет прогноза")):
        yield fake


def test_login_start_reload_cancel_and_return_to_input(service):
    """Пользователь может остановить анализ после обновления страницы."""
    with patch.dict(os.environ, {"TREND_WEB_REQUIRE_AUTH": "1", "TREND_WEB_ACCESS_PASSWORD": PASSWORD}):
        app = AppTest.from_file(WEB).run()
        assert not app.exception
        assert not app.text_area
        assert [button.label for button in form_buttons(app)] == ["Войти"]
        assert service.starts == []

        app.text_input[0].set_value(PASSWORD)
        form_buttons(app)[0].click().run()
        assert not app.exception
        assert [button.label for button in form_buttons(app)] == ["Анализировать"]
        assert len(app.text_area) == 1

        app.text_area[0].set_value("  Тестовое направление  ")
        form_buttons(app)[0].click().run()
        assert not app.exception
        assert service.starts == [("Тестовое направление", "fast")]
        assert app.text_area[0].disabled
        assert [button.label for button in form_buttons(app)] == ["Отменить анализ"]
        assert not form_buttons(app)[0].disabled

        service.status = AnalysisStatus("running", "run-1", "Тестовое направление", "fast")
        reloaded = AppTest.from_file(WEB).run()
        assert not reloaded.exception
        assert [button.label for button in form_buttons(reloaded)] == ["Войти"]
        reloaded.text_input[0].set_value(PASSWORD)
        form_buttons(reloaded)[0].click().run()
        assert not reloaded.exception
        assert reloaded.text_area[0].value == "Тестовое направление"
        assert reloaded.text_area[0].disabled
        assert [button.label for button in form_buttons(reloaded)] == ["Отменить анализ"]
        assert service.starts == [("Тестовое направление", "fast")]

        form_buttons(reloaded)[0].click().run()
        assert not reloaded.exception
        assert service.cancels == 1
        assert [button.label for button in form_buttons(reloaded)] == ["Отменяем анализ"]
        assert form_buttons(reloaded)[0].disabled

        service.status = AnalysisStatus("cancelled", "run-1", "Тестовое направление", "fast")
        reloaded.run()
        assert not reloaded.exception
        assert [button.label for button in form_buttons(reloaded)] == ["Анализировать"]
        assert not reloaded.text_area[0].disabled
        assert any("Анализ отменён" in item.value for item in reloaded.info)
        assert service.cancels == 1


def test_result_from_service_appears_in_new_session(service):
    """Результат не должен исчезать при обновлении страницы."""
    app = AppTest.from_file(WEB).run()
    app.text_area[0].set_value("Новые материалы")
    form_buttons(app)[0].click().run()
    assert not app.exception
    assert [button.label for button in form_buttons(app)] == ["Отменить анализ"]

    result = parse_result({"signals": [{"title": "Проверенный сигнал", "summary": "Тестовая сводка",
                                        "category": "confirmed_trend",
                                        "source_urls": ["https://example.org/evidence"]}]})
    service.status = AnalysisStatus("succeeded", "run-1", "Новые материалы", "fast", result)
    reloaded = AppTest.from_file(WEB).run()
    assert not reloaded.exception
    assert [button.label for button in form_buttons(reloaded)] == ["Анализировать"]
    assert not reloaded.text_area[0].disabled
    assert reloaded.text_area[0].value == "Новые материалы"
    markup = "\n".join(item.value for item in reloaded.markdown)
    assert "Проверенный сигнал" in markup
    assert "https://example.org/evidence" in markup
    assert service.starts == [("Новые материалы", "fast")]


def test_existing_session_shows_a_new_run_started_in_another_tab(service):
    """Черновик старого окна не должен подменять тему нового анализа."""
    service.status = AnalysisStatus("cancelled", "run-1", "Старая тема", "fast")
    app = AppTest.from_file(WEB).run()
    assert not app.exception
    app.text_area[0].set_value("Мой несохранённый черновик")
    app.segmented_control[1].set_value("fast").run()
    assert app.text_area[0].value == "Мой несохранённый черновик"

    # Другое окно начало анализ с новым идентификатором и другим режимом.
    service.status = AnalysisStatus("running", "run-2", "Новая тема", "deep")
    app.run()
    assert not app.exception
    assert app.text_area[0].value == "Новая тема"
    assert app.text_area[0].disabled
    assert app.segmented_control[1].value == "deep"
    assert app.segmented_control[1].disabled
    assert [button.label for button in form_buttons(app)] == ["Отменить анализ"]
    assert service.starts == []


def test_stale_stop_button_targets_only_the_run_it_displayed(service):
    service.status = AnalysisStatus("running", "run-1", "Старый анализ", "fast")
    app = AppTest.from_file(WEB).run()
    assert [button.label for button in form_buttons(app)] == ["Отменить анализ"]

    # Другая вкладка успела завершить старый анализ и запустить новый.
    service.status = AnalysisStatus("running", "run-2", "Новый анализ", "fast")
    form_buttons(app)[0].click().run()

    assert not app.exception
    assert service.cancel_requests == ["run-1"]
    assert service.status.id == "run-2"
    assert service.status.state == "running"
    assert app.text_area[0].value == "Новый анализ"
    assert [button.label for button in form_buttons(app)] == ["Отменить анализ"]


@pytest.mark.parametrize("state,error", [
    ("failed", "Не удалось обработать результат."),
    ("interrupted", "Анализ прерван при перезапуске сервиса."),
])
def test_terminal_failure_is_visible_after_reload_and_allows_retry(service, state, error):
    service.status = AnalysisStatus(state, "run-1", "Тема", "fast", error=error)
    app = AppTest.from_file(WEB).run()
    assert not app.exception
    assert [button.label for button in form_buttons(app)] == ["Анализировать"]
    assert not app.text_area[0].disabled
    assert app.text_area[0].value == "Тема"
    assert any(error in message.value for message in app.error)
    form_buttons(app)[0].click().run()
    assert not app.exception
    assert service.starts == [("Тема", "fast")]
    assert [button.label for button in form_buttons(app)] == ["Отменить анализ"]


def test_api_errors_keep_the_correct_control_visible(service):
    service.current_error = ApiError("Нет связи с сервисом анализа.")
    app = AppTest.from_file(WEB).run()
    assert not app.exception
    assert not form_buttons(app)
    assert any("Нет связи" in message.value for message in app.error)

    service.current_error = None
    service.start_error = ApiError("Сервис ещё не готов.")
    app.run()
    app.text_area[0].set_value("Тема")
    form_buttons(app)[0].click().run()
    assert not app.exception
    assert [button.label for button in form_buttons(app)] == ["Анализировать"]
    assert any("ещё не готов" in message.value for message in app.error)

    service.start_error = None
    form_buttons(app)[0].click().run()
    assert [button.label for button in form_buttons(app)] == ["Отменить анализ"]

    service.cancel_error = ApiError("Не удалось отменить анализ.")
    form_buttons(app)[0].click().run()
    assert not app.exception
    assert [button.label for button in form_buttons(app)] == ["Отменить анализ"]
    assert not form_buttons(app)[0].disabled
    assert any("Не удалось отменить" in message.value for message in app.error)
    assert service.starts == [("Тема", "fast"), ("Тема", "fast")]
    assert service.cancels == 1


def test_cancel_error_does_not_hide_a_later_result(service):
    service.status = AnalysisStatus("running", "run-1", "Тема", "fast")
    app = AppTest.from_file(WEB).run()
    service.cancel_error = ApiError("Временная ошибка отмены.")
    form_buttons(app)[0].click().run()
    assert any("Временная ошибка отмены" in message.value for message in app.error)

    result = parse_result({"signals": [{"title": "Сигнал", "summary": "Результат анализа",
                                        "category": "confirmed_trend",
                                        "source_urls": ["https://example.org/evidence"]}]})
    service.status = AnalysisStatus("succeeded", "run-1", "Тема", "fast", result)
    app.run()
    assert not app.exception
    assert not app.error
    assert any("Результат анализа" in item.value for item in app.markdown)


def test_run_panel_tracks_the_current_stage(service):
    service.status = AnalysisStatus("running", "run-1", "Тема", "fast",
                                    stage="labels", message="Проверяем кандидатов", completed=3, total=16)
    app = AppTest.from_file(WEB).run()
    assert not app.exception
    markup = "\n".join(item.value for item in app.markdown)
    assert 'class="ta-progress"' in markup
    assert 'aria-valuenow="19"' in markup
    assert 'style="width:19%"' in markup
    assert "3 из 16" in markup
    assert "Проверяем кандидатов" in markup
    assert '<li class="ta-step ta-step-active" aria-current="step">' in markup

    service.status = AnalysisStatus("running", "run-1", "Тема", "fast",
                                    stage="evidence", message="<script>alert(1)</script>")
    app.run()
    assert not app.exception
    markup = "\n".join(item.value for item in app.markdown)
    assert 'ta-progress-indeterminate' in markup
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in markup
    assert 'aria-valuenow=' not in markup

    service.status = AnalysisStatus("cancelled", "run-1", "Тема", "fast")
    app.run()
    assert not app.exception
    assert not any('class="ta-progress"' in item.value for item in app.markdown)
