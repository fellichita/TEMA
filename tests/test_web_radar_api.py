"""Фоновый радар технологий в веб-API: один расчёт на прогон, ход работы и отказ без падения."""

from datetime import date
from threading import Lock
import threading
from types import SimpleNamespace

from app import web_api
from app.web_api import WebAnalysisService, radar_input


def service(result_payload):
    fake = WebAnalysisService.__new__(WebAnalysisService)
    fake._radar_lock, fake._radars = Lock(), {}
    fake._reader_lock, fake._reader = Lock(), None
    fake.pilot = SimpleNamespace(result=lambda _run: result_payload, archive=None, data_dir=None)
    return fake


def wait(fake, run_id, query="тема"):
    fake._radar_for_run(run_id, query)
    for thread in threading.enumerate():
        if thread.name == "technology-radar":
            thread.join(timeout=10)
    return fake._radar_for_run(run_id, query)


def payload():
    from tests.test_web_api import approved_snapshot

    snapshot = approved_snapshot([f"https://arxiv.org/abs/2609.{index:05d}" for index in range(3)])
    return {"result": {"top_trend_ids": [], "cards": [],
                       "query_plan": {"as_of": "2026-09-10", "english_query": "solid-state batteries",
                                      "synonyms": ["solid electrolytes"]}},
            "approved_sources": snapshot}


def test_radar_input_returns_the_whole_pool_cutoff_and_english_terms():
    pool, as_of, terms = radar_input(payload())
    assert len(pool) == 3 and as_of == date(2026, 9, 10)
    assert terms == ["solid-state batteries", "solid electrolytes"]


def test_radar_input_ignores_malformed_synonyms():
    broken = payload()
    broken["result"]["query_plan"]["synonyms"] = 42
    _, _, terms = radar_input(broken)
    assert terms == ["solid-state batteries"]


def test_radar_runs_once_per_run_and_reports_progress(monkeypatch):
    calls = []

    def fake_radar(pool, *, query, query_terms, as_of, cancel, progress, openalex_key=None, cache=None):
        calls.append((len(pool), query, tuple(query_terms), as_of))
        progress(1, 1)
        return {"technologies": [{"title": "sulfide electrolyte", "description": "Fast ions.", "advantage": None,
                                  "case": None}], "excluded": []}

    monkeypatch.setattr("app.radar.pipeline.build_radar", fake_radar)
    fake = service(payload())
    fake._translate = lambda texts, *, cancel, done=None: [f"RU {text}" for text in texts]
    ready = wait(fake, "run-1", "Твердотельные батареи")
    assert ready["state"] == "ready" and ready["completed"] == ready["total"] == 1
    # Названия технологий остаются в оригинале, переводятся только выжимки.
    assert ready["result"]["translation"] == {"Fast ions.": "RU Fast ions."}
    assert calls == [(3, "Твердотельные батареи", ("solid-state batteries", "solid electrolytes"),
                      date(2026, 9, 10))]
    assert wait(fake, "run-1")["state"] == "ready" and len(calls) == 1


def test_radar_failure_is_reported_not_raised(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("app.radar.pipeline.build_radar", broken)
    state = wait(service(payload()), "run-1")
    assert state == {"state": "unavailable", "completed": 0, "total": 0,
                     "message": "Не удалось собрать ТОП технологий."}


def test_top_saved_by_the_analysis_is_shown_without_counting_again(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("ТОП уже посчитан анализом")

    monkeypatch.setattr("app.radar.pipeline.build_radar", forbidden)
    saved = {"technologies": [{"title": "sulfide electrolyte", "description": "Fast ions.",
                               "advantage": None, "case": None}], "excluded": []}
    fake = service(payload() | {"radar": {"state": "ready", "result": saved}})
    fake._translate = lambda texts, *, cancel, done=None: [f"RU {text}" for text in texts]
    ready = wait(fake, "run-1")
    assert ready["state"] == "ready"
    assert ready["result"]["technologies"] == saved["technologies"]
    assert ready["result"]["translation"] == {"Fast ions.": "RU Fast ions."}
    assert "translation" not in saved  # The stored result itself stays untouched.
    missing = service(payload() | {"radar": {"state": "unavailable", "message": "Нет материалов."}})
    assert wait(missing, "run-2") == {"state": "unavailable", "completed": 0, "total": 0,
                                      "message": "Нет материалов."}


def test_service_without_initializer_has_no_radar():
    assert WebAnalysisService.__new__(WebAnalysisService)._radar_for_run("run", "q") is None
    assert web_api.radar_input is radar_input


def test_a_top_handed_over_by_a_finished_analysis_is_awaited_not_recounted(monkeypatch):
    from concurrent.futures import Future
    from threading import Event

    def forbidden(*_args, **_kwargs):
        raise AssertionError("переданный ТОП не пересчитывается")

    monkeypatch.setattr("app.radar.pipeline.build_radar", forbidden)
    handed: Future = Future()
    steps = [7, 40]
    fake = service(payload() | {"radar": {"state": "pending"}})
    fake.pilot.take_radar = lambda run_id: (handed, steps, Event()) if run_id == "run-1" else None
    fake._radar_for_run("run-1", "тема")
    # Пока анализ досчитывает ТОП, сайт показывает его ход.
    for _ in range(100):
        state = fake._radar_for_run("run-1", "тема")
        if state["total"]:
            break
        Event().wait(0.05)
    assert (state["state"], state["completed"], state["total"]) == ("running", 7, 40)
    technologies = [{"title": "loot box", "description": "Boxes.", "advantage": None, "case": None}]
    handed.set_result({"state": "ready", "result": {"technologies": technologies, "excluded": [],
                                                    "translation": {"Boxes.": "Коробки."}}})
    ready = wait(fake, "run-1")
    # Перевод сделан ещё в анализе: сайт его не повторяет.
    assert ready["state"] == "ready" and ready["result"]["translation"] == {"Boxes.": "Коробки."}
