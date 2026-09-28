"""Public progress contract for the web UI's API client."""

import pytest

pytest.importorskip("requests")
from app.ui.api_client import (ApiClient, ApiError, parse_estimates, parse_publication_page,
                               parse_status)


def publication(index):
    return {"publication_id": f"{index:064x}", "source_id": "arxiv", "kind": "preprint",
            "title": f"Paper {index}", "url": f"https://arxiv.org/abs/2609.{index:05d}",
            "published_at": "2026-09-20", "publication_year": 2026,
            "date_basis": "published", "summary": None}


def test_publication_page_uses_run_bound_query_and_parses_bounded_records(monkeypatch):
    monkeypatch.delenv("TREND_API_TOKEN", raising=False)
    client = ApiClient("http://127.0.0.1:8000")
    paths = []

    def request(method, path):
        assert method == "GET"
        paths.append(path)
        return {"run_id": "run-1", "offset": 200, "total": 202,
                "publications": [publication(201), publication(202)]}

    monkeypatch.setattr(client, "_json_request", request)
    try:
        page = client.publication_page("run-1", offset=200, expected_total=202,
                                       known_ids=frozenset({publication(1)["publication_id"]}))
        assert [item.title for item in page] == ["Paper 201", "Paper 202"]
        assert paths == ["/analyses/current/publications?run_id=run-1&offset=200&limit=100"]
    finally:
        client._executor.shutdown()


def test_publication_page_round_trip_exposes_records_beyond_initial_web_limit(monkeypatch):
    from threading import Lock, Thread
    from app.web_api import ApiHandler, ApiServer, WebAnalysisService
    from tests.test_web_api import approved_snapshot

    token = "test-server-token-0123456789abcdef"
    monkeypatch.setenv("TREND_API_TOKEN", token)
    urls = [f"https://arxiv.org/abs/2609.{index:05d}" for index in range(205)]
    payload = {"result": {"top_trend_ids": [], "cards": []},
               "approved_sources": approved_snapshot(urls)}

    class Pilot:
        archive = None

        def get(self, run_id):
            assert run_id == "run-1"
            return {"state": "succeeded"}

        def result(self, run_id):
            assert run_id == "run-1"
            return payload

    service = WebAnalysisService.__new__(WebAnalysisService)
    service.pilot = Pilot()
    service._current = {"id": "run-1"}
    service._state_lock = Lock()
    service._rendered_lock = Lock()
    service._rendered = {}
    initial = service._web_result_for_run("run-1")
    server = ApiServer(("127.0.0.1", 0), ApiHandler)
    server.analysis = service
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = ApiClient(f"http://127.0.0.1:{server.server_port}")
    try:
        shown = initial["publications"]
        page = client.publication_page(
            "run-1", offset=len(shown), expected_total=initial["publication_total"],
            known_ids=frozenset(item["publication_id"] for item in shown))
        assert len(shown) == 200 and len(page) == 5
        assert {item.url for item in page} == set(urls[200:])
    finally:
        client._executor.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.parametrize("change", [
    {"run_id": "another-run"}, {"offset": 201}, {"total": 203},
    {"publications": []}, {"publications": [publication(201)]},
    {"publications": [publication(201), publication(201)]},
    {"publications": [publication(201), publication(202), publication(203)]},
    {"publications": [publication(201) | {"url": "file:///private/data"}]},
])
def test_publication_page_rejects_stale_duplicate_oversized_or_unsafe_data(change):
    payload = {"run_id": "run-1", "offset": 200, "total": 202,
               "publications": [publication(201), publication(202)]} | change
    with pytest.raises(ApiError):
        parse_publication_page(payload, run_id="run-1", total=202, offset=200, limit=100)


def test_publication_page_rejects_overlap_with_already_shown_records():
    with pytest.raises(ApiError, match="повторяющиеся"):
        parse_publication_page({"run_id": "run-1", "offset": 200, "total": 201,
                                "publications": [publication(201)]},
                               run_id="run-1", total=201, offset=200, limit=100,
                               known_ids=frozenset({publication(201)["publication_id"]}))


def test_publication_page_rejects_short_nonfinal_page():
    with pytest.raises(ApiError, match="неполную"):
        parse_publication_page({"run_id": "run-1", "offset": 200, "total": 400,
                                "publications": [publication(index) for index in range(201, 300)]},
                               run_id="run-1", total=400, offset=200, limit=100)


@pytest.mark.parametrize("run_id", ["run/1", "run 1", "ä", "x" * 129, ""])
def test_publication_page_rejects_invalid_run_identifier_without_request(run_id, monkeypatch):
    client = ApiClient("http://127.0.0.1:8000")
    monkeypatch.setattr(client, "_json_request", lambda *_args, **_kwargs: pytest.fail("unexpected request"))
    try:
        with pytest.raises(ApiError, match="страницу публикаций"):
            client.publication_page(run_id, offset=200, expected_total=202)
    finally:
        client._executor.shutdown()


def test_private_api_token_is_sent_only_to_loopback(monkeypatch):
    import requests
    from unittest.mock import patch

    token = "a" * 43
    monkeypatch.setenv("TREND_API_TOKEN", token)
    with pytest.raises(ApiError, match="Небезопасное соединение"):
        ApiClient("https://example.org")
    client = ApiClient("http://127.0.0.1:8000")
    try:
        def reject_request(session, *_args, **_kwargs):
            assert session.trust_env is False
            raise requests.ConnectionError

        with patch.object(requests.Session, "request", autospec=True,
                          side_effect=reject_request) as request:
            with pytest.raises(ApiError, match="Нет связи"):
                client.current_analysis()
            assert request.call_args.kwargs["headers"] == {"X-Trend-API-Token": token}
    finally:
        client._executor.shutdown()


def test_completed_result_is_parsed_once_and_reused_for_progress_polls(monkeypatch):
    monkeypatch.delenv("TREND_API_TOKEN", raising=False)
    client = ApiClient("http://127.0.0.1:8000")
    result = {"signals": []}
    common = {"id": "first-run", "query": "battery", "mode": "fast", "state": "succeeded",
              "result_version": "a" * 64}
    responses = iter([{**common, "result": result},
                      {**common, "translation": {"state": "running", "completed": 2,
                                                 "total": 10, "texts": {}}},
                      {"state": "idle"},
                      {**common, "result": result}])
    calls = []

    def request(_method, _path, _payload=None, *, extra_headers=None):
        calls.append(extra_headers)
        return next(responses)

    monkeypatch.setattr(client, "_json_request", request)
    try:
        first = client.current_analysis()
        second = client.current_analysis()
        assert first.result is second.result
        assert second.translation is not None and second.translation.completed == 2
        assert client.current_analysis().state == "idle"
        assert client.current_analysis().result is not None
        assert calls == [None, {"X-Trend-Result-Version": "a" * 64},
                         {"X-Trend-Result-Version": "a" * 64}, None]
    finally:
        client._executor.shutdown()


def test_completed_result_delta_must_match_cached_run_and_version(monkeypatch):
    monkeypatch.delenv("TREND_API_TOKEN", raising=False)
    client = ApiClient("http://127.0.0.1:8000")
    full = {"id": "first-run", "query": "battery", "mode": "fast", "state": "succeeded",
            "result_version": "a" * 64, "result": {"signals": []}}
    stale = {"id": "second-run", "query": "battery", "mode": "fast", "state": "succeeded",
             "result_version": "a" * 64}
    responses = iter([full, stale])
    monkeypatch.setattr(client, "_json_request", lambda *_args, **_kwargs: next(responses))
    try:
        client.current_analysis()
        with pytest.raises(ApiError, match="не вернул результат"):
            client.current_analysis()
    finally:
        client._executor.shutdown()


def test_status_parser_reads_current_stage_progress():
    status = parse_status({"id": "run-1", "query": "Тема", "mode": "fast", "state": "running",
                           "stage": "labels", "message": "Проверяем кандидатов",
                           "completed": 4, "total": 16})
    assert (status.stage, status.message, status.completed, status.total) == (
        "labels", "Проверяем кандидатов", 4, 16)
    assert parse_status({"id": "run-1", "query": "Тема", "mode": "fast", "state": "queued"}).stage is None


@pytest.mark.parametrize("changes", [
    {"stage": "labels"},
    {"stage": "labels", "message": "Проверяем", "completed": True, "total": 16},
    {"stage": "labels", "message": "Проверяем", "completed": 17, "total": 16},
    {"stage": "../../secrets", "message": "Проверяем", "completed": 4, "total": 16},
    {"stage": "labels", "message": "x" * 1_001, "completed": 4, "total": 16},
])
def test_status_parser_rejects_invalid_progress(changes):
    payload = {"id": "run-1", "query": "Тема", "mode": "fast", "state": "running", **changes}
    with pytest.raises(ApiError, match="прогресс"):
        parse_status(payload)


def test_status_parser_reads_elapsed_and_expected_time():
    status = parse_status({"id": "run-1", "query": "Тема", "mode": "deep", "state": "running",
                           "elapsed_seconds": 0, "expected_seconds": 1500})
    assert (status.elapsed_seconds, status.expected_seconds) == (0, 1500)
    plain = parse_status({"id": "run-1", "query": "Тема", "mode": "deep", "state": "running"})
    assert (plain.elapsed_seconds, plain.expected_seconds) == (None, None)


@pytest.mark.parametrize("changes", [
    {"elapsed_seconds": -1}, {"elapsed_seconds": 1.5}, {"elapsed_seconds": True},
    {"expected_seconds": 0}, {"expected_seconds": 10**9}, {"expected_seconds": "600"},
])
def test_status_parser_rejects_invalid_time(changes):
    payload = {"id": "run-1", "query": "Тема", "mode": "fast", "state": "running", **changes}
    with pytest.raises(ApiError, match="время"):
        parse_status(payload)


def estimate(**changes):
    return {"expected_seconds": 600, "basis": "history", "runs": 3, "documents": 3000,
            "top": 12, "patents": False, **changes}


def test_estimates_parser_reads_every_mode():
    estimates = parse_estimates({"modes": {"fast": estimate(expected_seconds=300, documents=1000, top=8),
                                           "deep": estimate(basis="default", runs=0, documents=10000,
                                                            top=15, patents=True)}})
    assert (estimates["fast"].expected_seconds, estimates["fast"].documents) == (300, 1000)
    assert (estimates["deep"].basis, estimates["deep"].top, estimates["deep"].patents) == ("default", 15, True)


@pytest.mark.parametrize("payload", [
    None, {"modes": []},
    {"modes": {"fast": estimate()}},
    {"modes": {"fast": estimate(), "deep": estimate(), "standard": estimate()}},
    {"modes": {"fast": estimate(basis="guess"), "deep": estimate()}},
    {"modes": {"fast": estimate(runs=-1), "deep": estimate()}},
    {"modes": {"fast": estimate(patents="no"), "deep": estimate()}},
    {"modes": {"fast": estimate(expected_seconds=None), "deep": estimate()}},
])
def test_estimates_parser_rejects_malformed_forecasts(payload):
    with pytest.raises(ApiError, match="прогноз|время"):
        parse_estimates(payload)
