"""HTTP lifecycle checks for the web demo without loading models or calling providers."""

from http import HTTPStatus
from http.client import HTTPConnection
import json
from threading import Event, Lock, Thread

import pytest

from app.runtime.jobs import TaskFailure
from app.web_api import ApiHandler, ApiServer, WebAnalysisService


class FakePilot:
    def __init__(self):
        self.ready = True
        self.start_error = None
        self.before_start = None
        self.before_cancel = None
        self.runs = {}
        self.starts = []
        self.cancellations = []
        self.cancel_called = Event()
        self._lock = Lock()

    def status(self):
        return {"model_installed": self.ready, "model_state": "ready" if self.ready else "loading",
                "settings": {"provider": "deepseek"}, "keys": {"deepseek_api_key": True}}

    def start(self, query, *, collection_profile):
        if self.before_start is not None:
            self.before_start()
        if self.start_error is not None:
            raise self.start_error
        with self._lock:
            run_id = f"run-{len(self.starts) + 1}"
            self.starts.append((query, collection_profile))
            self.runs[run_id] = {"state": "running"}
            return run_id

    def get(self, run_id):
        with self._lock:
            return dict(self.runs[run_id])

    def set_state(self, run_id, state, *, error=None):
        with self._lock:
            self.runs[run_id] = {"state": state, "error": error}

    def set_progress(self, run_id, *, stage, message, completed, total):
        with self._lock:
            self.runs[run_id].update(stage=stage, message=message,
                                     completed=completed, total=total)

    def cancel(self, run_id):
        if self.before_cancel is not None:
            self.before_cancel()
        with self._lock:
            self.cancellations.append(run_id)
            self.cancel_called.set()
            return self.runs[run_id]["state"] in {"queued", "running"}

    def result(self, run_id):
        assert self.get(run_id)["state"] == "succeeded"
        return {"result": {"top_trend_ids": ["confirmed", "unconfirmed"], "cards": [
            {"candidate": {"candidate_id": "confirmed", "label": "Confirmed trend",
                           "definition": "Fallback summary"}, "category": "confirmed_trend",
             "claims": [{"role": "summary", "text": "Saved summary", "support": "supported"}],
             "evidence": [{"source_url": "https://example.org/study"}]},
            {"candidate": {"candidate_id": "unconfirmed", "label": "Early signal",
                           "definition": "An early hypothesis"},
             "category": "early_signal", "evidence": [{"source_url": "https://example.org/early"}]}]}}


@pytest.fixture
def api(monkeypatch):
    monkeypatch.delenv("TREND_API_TOKEN", raising=False)
    monkeypatch.setenv("TREND_API_ALLOW_UNAUTHENTICATED_LOCAL", "1")
    pilot = FakePilot()
    service = object.__new__(WebAnalysisService)
    service.pilot = pilot
    service._admission = Lock()
    service._state_lock = Lock()
    service._current = None
    service._cancel_requested_id = None
    service.timeout_seconds = 60
    server = ApiServer(("127.0.0.1", 0), ApiHandler)
    server.analysis = service
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, path, payload=None, *, raw=None, content_type="application/json"):
        body = raw if raw is not None else json.dumps(payload).encode() if payload is not None else None
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            connection.request(method, path, body=body, headers={"Content-Type": content_type})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    try:
        yield request, pilot, service
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_http_analysis_can_be_recovered_cancelled_and_started_again(api):
    request, pilot, _service = api
    assert request("GET", "/analyses/current") == (HTTPStatus.OK, {"state": "idle"})

    status, started = request("POST", "/analyses", {"query": "  topic  ", "mode": "fast"})
    assert status == HTTPStatus.ACCEPTED
    assert started == {"id": "run-1", "state": "queued", "query": "topic", "mode": "fast"}
    assert pilot.starts == [("topic", "fast")]
    assert request("GET", "/analyses/current") == (HTTPStatus.OK, {**started, "state": "running"})

    status, body = request("POST", "/analyses", {"query": "another topic"})
    assert status == HTTPStatus.TOO_MANY_REQUESTS
    assert "уже выполняет" in body["error"]
    assert pilot.starts == [("topic", "fast")]

    assert request("POST", "/analyses/cancel", {"id": "run-1"}) == (
        HTTPStatus.OK, {**started, "state": "cancelling"})
    assert pilot.cancellations == ["run-1"]
    assert request("GET", "/analyses/current")[1]["state"] == "cancelling"
    pilot.set_state("run-1", "cancelled")
    assert request("GET", "/analyses/current")[1]["state"] == "cancelled"

    status, second = request("POST", "/analyses", {"query": "next topic", "mode": "deep"})
    assert status == HTTPStatus.ACCEPTED
    assert second["id"] == "run-2"
    assert pilot.starts == [("topic", "fast"), ("next topic", "deep")]
    pilot.set_state("run-2", "succeeded")
    status, completed = request("GET", "/analyses/current")
    assert status == HTTPStatus.OK
    assert completed == {**second, "state": "succeeded", "result": {"signals": [
        {"title": "Confirmed trend", "summary": "Saved summary",
         "category": "confirmed_trend",
         "source_urls": ["https://example.org/study"]},
        {"title": "Early signal", "summary": "An early hypothesis",
         "category": "early_signal",
         "source_urls": ["https://example.org/early"]}], "incomplete_coverage": False,
        "publications": [], "top_publications": [], "publication_total": 0,
        "source_coverage": []}}
    assert request("GET", "/analyses/current") == (HTTPStatus.OK, completed)


def test_http_current_analysis_reports_saved_stage_progress(api):
    request, pilot, _service = api
    assert request("POST", "/analyses", {"query": "topic"})[0] == HTTPStatus.ACCEPTED

    pilot.set_progress("run-1", stage="labels", message="Проверяем кандидатов", completed=4, total=16)
    assert request("GET", "/analyses/current")[1] == {
        "id": "run-1", "state": "running", "query": "topic", "mode": "fast",
        "stage": "labels", "message": "Проверяем кандидатов", "completed": 4, "total": 16}

    # Each stage has its own counter; a smaller later count must be preserved.
    pilot.set_progress("run-1", stage="evidence", message="Проверяем доказательства",
                       completed=1, total=60)
    assert request("GET", "/analyses/current")[1]["completed"] == 1
    assert request("POST", "/analyses/cancel", {"id": "run-1"})[1]["stage"] == "evidence"


def test_http_current_analysis_drops_invalid_saved_progress(api):
    request, pilot, _service = api
    assert request("POST", "/analyses", {"query": "topic"})[0] == HTTPStatus.ACCEPTED
    pilot.set_progress("run-1", stage="labels", message="Internal", completed=True, total=16)
    status, body = request("GET", "/analyses/current")
    assert status == HTTPStatus.OK
    assert all(field not in body for field in ("stage", "message", "completed", "total"))


def test_stale_cancel_cannot_stop_a_new_run_started_in_another_tab(api):
    request, pilot, _service = api
    assert request("POST", "/analyses", {"query": "first"})[1]["id"] == "run-1"
    pilot.set_state("run-1", "cancelled")
    assert request("POST", "/analyses", {"query": "second"})[1]["id"] == "run-2"

    status, body = request("POST", "/analyses/cancel", {"id": "run-1"})
    assert status == HTTPStatus.CONFLICT
    assert "Обновите страницу" in body["error"]
    assert pilot.cancellations == []
    assert request("GET", "/analyses/current")[1]["state"] == "running"
    assert request("GET", "/analyses/current")[1]["id"] == "run-2"


def test_cancel_rechecks_run_if_another_tab_starts_during_request(api):
    request, pilot, service = api
    assert request("POST", "/analyses", {"query": "first"})[1]["id"] == "run-1"

    def start_next_while_cancel_is_in_flight():
        pilot.set_state("run-1", "cancelled")
        assert service.start_analysis("second")["id"] == "run-2"

    pilot.before_cancel = start_next_while_cancel_is_in_flight
    status, _body = request("POST", "/analyses/cancel", {"id": "run-1"})
    assert status == HTTPStatus.CONFLICT
    assert pilot.cancellations == ["run-1"]
    assert request("GET", "/analyses/current")[1]["id"] == "run-2"
    assert request("GET", "/analyses/current")[1]["state"] == "running"


def test_http_analysis_reports_input_readiness_and_pilot_failures(api):
    request, pilot, _service = api
    assert request("POST", "/analyses/cancel", {"id": "run-1"})[0] == HTTPStatus.CONFLICT
    assert request("POST", "/analyses", {"query": "  "})[0] == HTTPStatus.UNPROCESSABLE_ENTITY
    assert request("POST", "/analyses", {"query": "topic", "mode": "unknown"})[0] == \
        HTTPStatus.UNPROCESSABLE_ENTITY
    assert request("POST", "/analyses", {"query": "topic", "extra": 1}) == (
        HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "invalid_request"})
    assert request("POST", "/analyses", raw=b"{") == (
        HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "invalid_request"})
    assert request("POST", "/analyses", raw=b'{"query":"\\ud800"}') == (
        HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "Запрос содержит некорректные символы."})
    assert request("POST", "/analyses", raw=b"topic", content_type="text/plain") == (
        HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "invalid_content_type"})
    assert pilot.starts == []

    pilot.ready = False
    assert request("POST", "/analyses", {"query": "topic"})[0] == HTTPStatus.SERVICE_UNAVAILABLE
    pilot.ready = True
    pilot.start_error = TaskFailure("The provider rejected this topic")
    assert request("POST", "/analyses", {"query": "topic"}) == (
        HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "The provider rejected this topic"})
    pilot.start_error = None

    assert request("POST", "/analyses", {"query": "topic"})[0] == HTTPStatus.ACCEPTED
    pilot.set_state("run-1", "failed", error="Analysis failed in a worker")
    assert request("GET", "/analyses/current")[1]["error"] == "Analysis failed in a worker"


def test_http_status_stays_responsive_while_cancel_waits_for_pilot(api):
    request, pilot, _service = api
    assert request("POST", "/analyses", {"query": "topic"})[0] == HTTPStatus.ACCEPTED
    entered, release = Event(), Event()

    def wait_in_cancel():
        entered.set()
        assert release.wait(3)

    pilot.before_cancel = wait_in_cancel
    responses = []
    worker = Thread(target=lambda: responses.append(request("POST", "/analyses/cancel", {"id": "run-1"})),
                    daemon=True)
    try:
        worker.start()
        assert entered.wait(3)
        assert request("GET", "/analyses/current")[1]["state"] == "running"
        assert request("POST", "/analyses", {"query": "duplicate"})[0] == HTTPStatus.TOO_MANY_REQUESTS
    finally:
        release.set()
        worker.join(timeout=3)
    assert responses[0][0] == HTTPStatus.OK
    assert responses[0][1]["state"] == "cancelling"
    assert pilot.starts == [("topic", "fast")]


def test_concurrent_http_starts_admit_only_one_analysis(api):
    request, pilot, _service = api
    entered, release = Event(), Event()

    def wait_in_start():
        entered.set()
        assert release.wait(3)

    pilot.before_start = wait_in_start
    responses = []
    worker = Thread(target=lambda: responses.append(request("POST", "/analyses", {"query": "first"})),
                    daemon=True)
    try:
        worker.start()
        assert entered.wait(3)
        assert request("POST", "/analyses", {"query": "second"})[0] == HTTPStatus.TOO_MANY_REQUESTS
    finally:
        release.set()
        worker.join(timeout=3)
    assert responses[0][0] == HTTPStatus.ACCEPTED
    assert pilot.starts == [("first", "fast")]


def test_async_analysis_outlives_legacy_timeout_until_user_cancels(api):
    request, pilot, service = api
    service.timeout_seconds = 0.01
    assert request("POST", "/analyses", {"query": "slow local model"})[0] == HTTPStatus.ACCEPTED

    # The same limit still applies to /analyze, but an async run must remain
    # controllable after it elapses; a normal 10,000-study run may take longer.
    assert not pilot.cancel_called.wait(0.2)
    assert pilot.cancellations == []
    assert request("GET", "/analyses/current")[1]["state"] == "running"

    assert request("POST", "/analyses/cancel", {"id": "run-1"})[1]["state"] == "cancelling"
    assert pilot.cancellations == ["run-1"]


def test_legacy_blocking_analysis_still_times_out(api):
    request, pilot, service = api
    service.timeout_seconds = 0.01
    service.poll_seconds = 0.001
    status, body = request("POST", "/analyze", {"query": "slow local model"})
    assert status == HTTPStatus.GATEWAY_TIMEOUT
    assert "лимит времени" in body["error"]
    assert pilot.cancellations == ["run-1"]
