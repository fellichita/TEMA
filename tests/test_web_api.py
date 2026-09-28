"""Contract checks for the loopback-only web bridge."""

from http import HTTPStatus
from http.client import HTTPConnection
from datetime import date
import json
import os
from threading import Event, Lock, Thread

import pytest

from app.web_api import ApiHandler, ApiServer, WebAnalysisService, WebApiError, web_result


@pytest.fixture(autouse=True)
def local_api_for_http_contract_tests(monkeypatch):
    monkeypatch.delenv("TREND_API_TOKEN", raising=False)
    monkeypatch.setenv("TREND_API_ALLOW_UNAUTHENTICATED_LOCAL", "1")


def card(identifier="one", *, category="confirmed_trend"):
    return {"candidate": {"candidate_id": identifier, "label": "Verified signal",
                          "definition": "Definition from the saved result."},
            "category": category,
            "claims": [{"role": "summary", "support": "unverified", "text": "Saved summary."}],
            "evidence": [{"source_url": "https://example.org/a"},
                         {"source_url": "javascript:alert(1)"},
                         {"source_url": "https://example.org/a"}]}


def test_web_result_exposes_ranked_signal_categories_with_safe_unique_sources():
    categories = ("confirmed_trend", "early_signal", "weak_signal_candidate", "emerging_candidate")
    identifiers = ("one", "two", "three", "four")
    payload = {"result": {"top_trend_ids": [*identifiers, "one"],
                          "cards": [card(identifier, category=category)
                                    for identifier, category in zip(identifiers, categories, strict=True)]}}
    assert web_result(payload) == {"signals": [{"title": "Verified signal",
        "summary": "Saved summary.", "category": category,
        "source_urls": ["https://example.org/a"]} for category in categories],
        "incomplete_coverage": False, "publications": [], "top_publications": [],
        "publication_total": 0, "source_coverage": []}


def test_signal_sources_keep_first_hundred_safe_unique_links():
    saved_card = card()
    links = [f"https://example.org/study/{index}" for index in range(101)]
    saved_card["evidence"] = ([{"source_url": links[0]},
                               {"source_url": "javascript:alert(1)"}]
                              + [{"source_url": link} for link in links]
                              + [{"source_url": links[1]}])
    result = web_result({"result": {"top_trend_ids": ["one"], "cards": [saved_card]}})
    assert result["signals"][0]["source_urls"] == links[:100]


def test_web_result_does_not_fill_an_empty_top_with_unselected_cards():
    assert web_result({"result": {"top_trend_ids": [], "cards": [card("early", category="early_signal")]}}) == {
        "signals": [], "incomplete_coverage": False, "publications": [], "top_publications": [],
        "publication_total": 0, "source_coverage": []}


def test_empty_collection_has_no_charts():
    from app.pilot.approved_sources import unavailable_snapshot

    snapshot = unavailable_snapshot("quantum sensor", date(2026, 9, 25))
    response = web_result({"result": {"top_trend_ids": [], "cards": []},
                           "approved_sources": snapshot.model_dump(mode="json")})
    assert response["publications"] == []
    assert "monthly_evidence" not in response


def test_web_result_never_promotes_unranked_or_ineligible_cards():
    payload = {"result": {"top_trend_ids": ["unassessed", "weak"],
                          "cards": [card("unassessed", category="unassessed_cluster"),
                                    card("weak", category="weak_signal_candidate"),
                                    card("unranked", category="confirmed_trend")]}}
    assert [item["category"] for item in web_result(payload)["signals"]] == ["weak_signal_candidate"]


def test_web_result_discloses_incomplete_source_coverage():
    payload = {"result": {"top_trend_ids": ["one"], "cards": [card()],
                          "snapshots": [{"purpose": "discovery", "coverage": [
                              {"state": "complete"}, {"state": "partial"}]}]},
               "discovery_summary": {"quality": "partial"}}
    assert web_result(payload)["incomplete_coverage"] is True
    payload["result"]["snapshots"][0]["coverage"][1]["state"] = "complete"
    assert web_result(payload)["incomplete_coverage"] is False


def test_web_result_prefers_supported_description_over_unverified_summary():
    saved_card = card()
    saved_card["claims"].append({"role": "advantage", "support": "supported", "text": "Archival finding."})
    payload = {"result": {"top_trend_ids": ["one"], "cards": [saved_card]}}
    assert web_result(payload)["signals"][0]["summary"] == "Archival finding."


def approved_snapshot(records):
    from app.pilot.approved_sources import unavailable_snapshot

    snapshot = unavailable_snapshot("quantum sensors", date(2026, 9, 10)).model_dump(mode="json")
    snapshot["observations"] = [
        {"source_id": "arxiv", "item_id": str(index), "kind": "preprint",
         "title": f"Paper {index}", "url": url,
         "published_at": "2026-09-09", "observed_at": "2026-09-10T12:00:00Z",
         "summary": "Short metadata", "rights": "local_only", "license_ref": None}
        for index, url in enumerate(records)
    ]
    snapshot["coverage"][0].update(state="partial", requested_limit=len(records),
                                    scanned=len(records), accepted=len(records),
                                    limit_reached=True, reason_code="source_limit")
    return snapshot


def test_web_result_places_approved_publications_in_top_even_without_signals():
    snapshot = approved_snapshot(["https://arxiv.org/abs/2609.12345"])
    response = web_result({"result": {"top_trend_ids": [], "cards": []}, "approved_sources": snapshot})
    assert response["signals"] == []
    from app.pilot.approved_sources.contracts import SOURCE_IDS

    assert len(response["source_coverage"]) == len(SOURCE_IDS)
    assert response["publication_total"] == 1
    assert response["top_publications"] == response["publications"]
    assert response["publications"][0]["source_id"] == "arxiv"
    assert response["publications"][0]["url"] == "https://arxiv.org/abs/2609.12345"
    assert "source_observations" not in response


def test_web_result_exposes_arxiv_fields_without_claiming_full_history():
    snapshot = approved_snapshot(["https://arxiv.org/abs/2609.12345",
                                  "https://arxiv.org/abs/2609.12346"])
    snapshot["observations"][0]["arxiv_primary_category"] = "cs.LG"
    snapshot["observations"][1]["arxiv_primary_category"] = "cs.LG"
    result = web_result({"result": {"top_trend_ids": [], "cards": []},
                         "approved_sources": snapshot})
    assert result["arxiv_domains"] == {
        "coverage_state": "partial", "scanned": 2,
        "months": [{"month": "2026-09-01", "domain": "cs",
                    "primary_category": "cs.LG", "article_count": 2}],
    }


def test_europe_pmc_and_biorxiv_doi_are_one_publication():
    from app.pilot.approved_sources import unavailable_snapshot

    snapshot = unavailable_snapshot("sensor", date(2026, 9, 10)).model_dump(mode="json")
    doi = "10.1101/2026.09.09.123456"
    snapshot["observations"] = [
        {"source_id": source, "item_id": doi, "kind": "preprint", "title": "Sensor preprint",
         "url": url, "published_at": "2026-09-09", "observed_at": "2026-09-10T12:00:00Z",
         "rights": "local_only"}
        for source, url in (("biorxiv", "https://www.biorxiv.org/content/" + doi),
                            ("europe_pmc", "https://doi.org/" + doi))
    ]
    for item in snapshot["coverage"]:
        if item["source_id"] in {"biorxiv", "europe_pmc"}:
            item.update(state="complete", requested_limit=1, scanned=1, accepted=1,
                        limit_reached=False, reason_code=None)
    result = web_result({"result": {"top_trend_ids": [], "cards": []},
                         "approved_sources": snapshot})
    assert result["publication_total"] == 1
    assert result["publications"][0]["source_ids"] == ["biorxiv", "europe_pmc"]
    # Копия той же работы из второго источника — не «похожий материал».
    assert result["publications"][0]["similar"]["total"] == 0


def test_nih_money_is_separate_from_publications_and_bad_links_are_rejected():
    funding = {"source_id": "nih_reporter", "topic": "quantum sensor",
               "from_date": "2024-10-01", "to_date": "2026-09-10",
               "date_basis": "award_notice_date",
               "amount_basis": "reported_fiscal_year_award_usd", "total_available": 10,
               "records_returned": 1, "rejected_records": 0, "partial_coverage": True,
               "coverage_reason": "page_cap", "awards": [{
                   "application_id": 123, "project_number": "R01-123", "title": "Quantum sensor",
                   "award_notice_date": "2026-09-09", "award_amount_usd": "100000.00",
                   "funding_mechanism": "RP", "funding_type": "grant_or_cooperative",
                   "detail_url": "https://reporter.nih.gov/project-details/123"}]}
    payload = {"result": {"top_trend_ids": [], "cards": []}, "funding_sources": funding}
    result = web_result(payload)
    assert result["publication_total"] == 0
    assert result["funding_evidence"]["awards"][0]["award_amount_usd"] == "100000.00"
    funding["awards"][0]["detail_url"] = "https://example.org/?token=secret"
    with pytest.raises(WebApiError, match="финансирования"):
        web_result(payload)


def test_web_result_builds_one_deduplicated_publication_pool_and_top_fifteen(tmp_path):
    from app.pilot.archive import DocumentArchive
    from tests.test_pilot_evidence import document, snapshot

    archive = DocumentArchive(tmp_path / "revisions")
    scientific = document(1, year=2026, publication_month=9,
                          publication_date=date(2026, 9, 10), date_precision="day")
    year_only = document(2, year=2025)
    discovery = snapshot((scientific, year_only), archive)
    approved = approved_snapshot([scientific.url] + [
        f"https://arxiv.org/abs/2609.{index:05d}" for index in range(15)
    ])
    response = web_result({"result": {"top_trend_ids": [], "cards": [],
                                      "snapshots": [discovery.model_dump(mode="json")]},
                           "approved_sources": approved}, archive=archive)
    assert response["publication_total"] == 17  # Same URL from two sources is one publication.
    assert len(response["publications"]) == 17
    shared = next(item for item in response["publications"] if item["title"] == scientific.title)
    assert shared["source_ids"] == ["arxiv", "openalex"]
    assert all(item["similar"]["months"][-1]["month"] == "2026-09" for item in response["top_publications"])
    assert all("similar" not in item for item in response["publications"][15:])
    assert len(response["top_publications"]) == 15
    assert response["top_publications"] == response["publications"][:15]
    assert {item["source_id"] for item in response["top_publications"]} == {"openalex", "arxiv"}
    undated = next(item for item in response["publications"] if item["title"] == year_only.title)
    assert undated["published_at"] is None
    assert undated["publication_year"] == 2025
    from app.pilot.approved_sources.contracts import SOURCE_IDS

    assert {item["source_id"] for item in response["source_coverage"]} == {"openalex", *SOURCE_IDS}
    assert "source_observations" not in response


def test_newer_approved_publication_precedes_older_assessed_scientific_publication(tmp_path):
    from app.pilot.archive import DocumentArchive
    from tests.test_pilot_evidence import document, snapshot

    archive = DocumentArchive(tmp_path / "revisions")
    older = document(1, year=2020)
    discovery = snapshot((older,), archive)
    study_id = discovery.documents[0].study_id
    assessed = card("topic") | {"candidate": {"candidate_id": "topic", "label": "Topic",
                                               "discovery_study_ids": [study_id]}}
    payload = {"result": {"top_trend_ids": ["topic"], "cards": [assessed],
                          "snapshots": [discovery.model_dump(mode="json")]},
               "assessments": [assessment("topic", "high", {"growth"}, growth=True)],
               "approved_sources": approved_snapshot(["https://arxiv.org/abs/2609.54321"])}
    top = web_result(payload, archive=archive)["top_publications"]
    assert [item["source_id"] for item in top] == ["arxiv", older.source]
    assert top[1]["trend"]["confidence"] == "high"


def test_web_result_bounds_common_publication_list_after_counting_all_matches():
    approved = approved_snapshot([
        f"https://arxiv.org/abs/2609.{index:05d}" for index in range(205)
    ])
    response = web_result({"result": {"top_trend_ids": [], "cards": []},
                           "approved_sources": approved})
    assert response["publication_total"] == 205
    assert len(response["publications"]) == 200
    assert response["top_publications"] == response["publications"][:15]
    # Похожие считаются по всей выборке, а не по 200 записям для браузера.
    # Одинаковые записи — копии одной работы, а не похожие материалы.
    assert response["top_publications"][0]["similar"]["months"] == [
        {"month": "2026-09", "count": 0, "collected": 205}]


def test_completed_web_run_pages_all_found_publications_without_recomputing_result():
    urls = [f"https://arxiv.org/abs/2609.{index:05d}" for index in range(1070)]
    payload = {"result": {"top_trend_ids": [], "cards": []},
               "approved_sources": approved_snapshot(urls)}

    class Pilot:
        archive = None
        state = "succeeded"
        result_calls = 0

        def get(self, run_id):
            from app.runtime.jobs import TaskFailure

            if run_id != "run-1":
                raise TaskFailure("Анализ не найден.")
            return {"state": self.state}

        def result(self, run_id):
            assert run_id == "run-1"
            self.result_calls += 1
            return payload

    service = WebAnalysisService.__new__(WebAnalysisService)
    service.pilot = Pilot()
    service._current = None  # Pages belong to any finished run, not only the shared current one.
    service._state_lock = Lock()
    service._rendered_lock = Lock()
    service._rendered = {}
    first = service._web_result_for_run("run-1")
    assert first["publication_total"] == 1070
    assert len(first["publications"]) == 200
    pages = [service.publications_page("run-1", offset, 100)
             for offset in range(200, 1070, 100)]
    found = first["publications"] + [item for page in pages for item in page["publications"]]
    assert len(found) == len({item["publication_id"] for item in found}) == 1070
    assert {item["url"] for item in found} == set(urls)
    assert [len(page["publications"]) for page in pages] == [100] * 8 + [70]
    assert all(page["total"] == 1070 and page["run_id"] == "run-1" for page in pages)
    assert service.pilot.result_calls == 1
    assert service.publications_page("run-1", 1070, 100)["publications"] == []
    with pytest.raises(WebApiError) as unknown:
        service.publications_page("other-run", 200, 100)
    assert unknown.value.status == HTTPStatus.NOT_FOUND
    with pytest.raises(WebApiError) as invalid:
        service.publications_page("run-1", 200, 101)
    assert invalid.value.status == HTTPStatus.UNPROCESSABLE_ENTITY
    service.pilot.state = "running"
    with pytest.raises(WebApiError) as unfinished:
        service.publications_page("run-1", 200, 100)
    assert unfinished.value.status == HTTPStatus.CONFLICT


def test_web_result_keeps_indexed_and_created_dates_distinct_from_publication_dates():
    from app.pilot.approved_sources import unavailable_snapshot

    snapshot = unavailable_snapshot("quantum sensors", date(2026, 9, 10)).model_dump(mode="json")
    snapshot["observations"] = [
        {"source_id": "gdelt", "item_id": "news-1", "kind": "news_aggregate",
         "title": "Indexed story", "url": "https://example.org/news-1",
         "published_at": "2026-09-09", "date_basis": "indexed",
         "observed_at": "2026-09-10T12:00:00Z", "summary": None,
         "rights": "local_only", "license_ref": None},
        {"source_id": "openreview", "item_id": "forum-1", "kind": "preprint",
         "title": "Created record", "url": "https://openreview.net/forum?id=forum-1",
         "published_at": "2026-09-08", "date_basis": "created",
         "observed_at": "2026-09-10T12:00:00Z", "summary": None,
         "rights": "local_only", "license_ref": None},
    ]
    for item in snapshot["coverage"]:
        if item["source_id"] in {"gdelt", "openreview"}:
            item.update(state="complete", requested_limit=1, scanned=1, accepted=1,
                        limit_reached=False, reason_code=None)
    response = web_result({"result": {"top_trend_ids": [], "cards": []},
                           "approved_sources": snapshot})
    assert [(item["source_id"], item["date_basis"]) for item in response["top_publications"]] == [
        ("gdelt", "indexed"), ("openreview", "created")]


def test_similar_materials_need_a_known_month(tmp_path):
    from app.pilot.archive import DocumentArchive
    from tests.test_pilot_evidence import document, snapshot

    archive = DocumentArchive(tmp_path / "revisions")
    # Близкие, но разные работы: одинаковые тексты считались бы копиями.
    month_only = document(1, year=2026, publication_month=8, date_precision="month",
                          title="Lithium selective membranes for brine extraction",
                          abstract="Selective membranes recover lithium from geothermal brine with low energy use.")
    conflict = document(2, year=2026, publication_month=7, date_precision="month",
                        title="Lithium extraction from brine with ion-sieve membranes",
                        abstract="Ion-sieve membranes improve lithium recovery from salt lake brine.")
    year_only = document(3, year=2026, title="Selective lithium membranes tested in pilot brine plant",
                         abstract="A pilot plant evaluates membrane selectivity for lithium in brine.")
    discovery = snapshot((month_only, conflict, year_only), archive)
    response = web_result({"result": {"top_trend_ids": [], "cards": [],
                                      "query_plan": {"as_of": "2026-09-10"},
                                      "snapshots": [discovery.model_dump(mode="json")]},
                           "approved_sources": approved_snapshot([conflict.url])}, archive=archive)
    assert response["publication_total"] == 3
    by_title = {item["title"]: item["similar"] for item in response["top_publications"]}
    # Для года без месяца и для противоречивых дат месяца нет: их не на что положить.
    assert by_title[month_only.title]["total"] == 0
    for other in (conflict, year_only):
        months = {row["month"]: row["count"] for row in by_title[other.title]["months"]}
        assert by_title[other.title]["total"] == months["2026-08"] == 1


def test_scientific_copies_preserve_both_source_ids_without_double_counting(tmp_path):
    from app.pilot.archive import DocumentArchive
    from tests.test_pilot_evidence import document, snapshot

    archive = DocumentArchive(tmp_path / "revisions")
    openalex = document(1, year=2026, publication_date=date(2026, 8, 10), date_precision="day")
    crossref = openalex.model_copy(update={"source": "crossref", "source_id": "C1"})
    discovery = snapshot((openalex, crossref), archive)
    response = web_result({"result": {"top_trend_ids": [], "cards": [],
                                      "snapshots": [discovery.model_dump(mode="json")]}}, archive=archive)
    assert response["publication_total"] == 1
    assert response["publications"][0]["source_ids"] == ["crossref", "openalex"]
    assert response["publications"][0]["similar"]["total"] == 0


def test_conflicting_day_precision_dates_do_not_pick_an_arbitrary_month(tmp_path):
    from app.pilot.archive import DocumentArchive
    from tests.test_pilot_evidence import document, snapshot

    archive = DocumentArchive(tmp_path / "revisions")
    july = document(1, year=2026, publication_date=date(2026, 7, 10), date_precision="day")
    august = july.model_copy(update={"source": "crossref", "source_id": "C1",
                                     "publication_date": date(2026, 8, 10)})
    discovery = snapshot((july, august), archive)
    response = web_result({"result": {"top_trend_ids": [], "cards": [],
                                      "snapshots": [discovery.model_dump(mode="json")]}}, archive=archive)
    assert response["publication_total"] == 1
    assert response["publications"][0]["published_at"] is None
    assert response["publications"][0]["date_basis"] == "published"


def test_analysis_rejects_parallel_request_without_queueing():
    service = object.__new__(WebAnalysisService)
    from threading import Lock
    service._admission = Lock()
    service._admission.acquire()
    with pytest.raises(WebApiError) as failure:
        service.analyze("query")
    assert failure.value.status == HTTPStatus.TOO_MANY_REQUESTS
    service._admission.release()


def test_analysis_polls_real_saved_result():
    done = Event()

    class Pilot:
        def status(self):
            return {"model_installed": True, "model_state": "ready", "settings": {"provider": "deepseek"},
                    "keys": {"deepseek_api_key": True}}

        def start(self, query, *, collection_profile=None):
            assert query == "query"
            assert collection_profile == "fast"
            return "run"

        def get(self, run_id):
            assert run_id == "run"
            done.set()
            return {"state": "succeeded"}

        def result(self, run_id):
            return {"result": {"top_trend_ids": ["one"], "cards": [card()]}}

    service = object.__new__(WebAnalysisService)
    from threading import Lock
    service.pilot = Pilot()
    service._admission = Lock()
    service.poll_seconds = 0
    service.timeout_seconds = 1
    assert service.analyze(" query ")["signals"][0]["title"] == "Verified signal"
    assert done.is_set()


def test_analysis_rejects_unknown_collection_mode_before_start():
    service = object.__new__(WebAnalysisService)
    from threading import Lock
    service._admission = Lock()
    with pytest.raises(WebApiError, match="режим"):
        service.analyze("query", "turbo")


def test_analysis_rejects_unpaired_unicode_surrogates_before_start():
    service = object.__new__(WebAnalysisService)
    for query in ("\ud800", "topic\udfff"):
        with pytest.raises(WebApiError, match="некорректные символы"):
            service.start_analysis(query)


def test_current_analysis_survives_ui_reload_and_cancel_stops_owned_run():
    class Pilot:
        state = "running"
        cancelled = False

        def status(self):
            return {"model_installed": True, "model_state": "ready", "settings": {"provider": "local"},
                    "keys": {}}

        def start(self, query, *, collection_profile):
            assert (query, collection_profile) == ("topic", "fast")
            return "run-id"

        def get(self, run_id):
            assert run_id == "run-id"
            return {"state": self.state}

        def cancel(self, run_id):
            assert run_id == "run-id"
            self.cancelled = True
            return True

        def result(self, run_id):
            assert run_id == "run-id"
            return {"result": {"top_trend_ids": ["one"], "cards": [card()]}}

    pilot = Pilot()
    service = object.__new__(WebAnalysisService)
    service.pilot = pilot
    service._admission = Lock()
    service._state_lock = Lock()
    service._current = None
    service._cancel_requested_id = None
    service._timed_out_id = None
    service._timer = None
    service.timeout_seconds = 60
    assert service.current_analysis() == {"state": "idle"}
    try:
        started = service.start_analysis(" topic ", "fast")
        assert started == {"id": "run-id", "state": "queued", "query": "topic", "mode": "fast"}
        assert service.current_analysis() == {"id": "run-id", "state": "running", "query": "topic", "mode": "fast"}
        with pytest.raises(WebApiError) as failure:
            service.start_analysis("topic", "fast")
        assert failure.value.status == HTTPStatus.TOO_MANY_REQUESTS
        assert service.cancel_analysis("run-id")["state"] == "cancelling"
        assert pilot.cancelled
        pilot.state = "cancelled"
        assert service.current_analysis()["state"] == "cancelled"
        pilot.state = "succeeded"
        assert service.current_analysis()["result"]["signals"][0]["title"] == "Verified signal"
    finally:
        if service._timer is not None:
            service._timer.cancel()


def test_http_status_and_cancel_are_available_while_old_analysis_request_waits():
    started, release = Event(), Event()

    class Analysis:
        cancel_calls = 0

        def start_analysis(self, query, mode):
            return {"id": "new-run", "state": "queued", "query": query, "mode": mode}

        def analyze(self, _query, _mode):
            started.set()
            assert release.wait(5)
            return {"signals": []}

        def current_analysis(self):
            return {"id": "run-id", "state": "running", "query": "topic", "mode": "fast"}

        def cancel_analysis(self, run_id):
            assert run_id == "run-id"
            self.cancel_calls += 1
            release.set()
            return {"id": "run-id", "state": "cancelling", "query": "topic", "mode": "fast"}

    service = Analysis()
    server = ApiServer(("127.0.0.1", 0), ApiHandler)
    server.analysis = service
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    port = server.server_port

    def request(method, path, payload=None, *, headers=None):
        connection = HTTPConnection("127.0.0.1", port, timeout=3)
        body = json.dumps(payload).encode() if payload is not None else None
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    legacy_response = []
    worker = Thread(target=lambda: legacy_response.append(request("POST", "/analyze", {"query": "topic"})),
                    daemon=True)
    try:
        worker.start()
        assert started.wait(3)
        assert request("GET", "/analyses/current")[1]["state"] == "running"
        # A remote site's Origin cannot use the loopback API to cancel the run.
        assert request("POST", "/analyses/cancel", {"id": "run-id"},
                       headers={"Origin": "https://other.example"})[0] == HTTPStatus.FORBIDDEN
        assert service.cancel_calls == 0
        status, body = request("POST", "/analyses/cancel", {"id": "run-id"})
        assert status == HTTPStatus.OK and body["state"] == "cancelling"
        worker.join(timeout=3)
        assert legacy_response == [(HTTPStatus.OK, {"signals": []})]
        status, body = request("POST", "/analyses", {"query": "next", "mode": "fast"})
        assert status == HTTPStatus.ACCEPTED and body == {
            "id": "new-run", "state": "queued", "query": "next", "mode": "fast"}
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=3)


def history_row(mode, minutes, *, provider="local", state="succeeded", attempt=1, operation=None,
                controls=None):
    from app.pilot.settings import COLLECTION_PROFILES

    settings = {"provider": provider, **(COLLECTION_PROFILES[mode] if controls is None else controls)}
    payload = {"collection_profile": mode, "settings": settings}
    if operation is not None:
        payload["operation"] = operation
    return {"state": state, "attempt": attempt,
            "input_json": json.dumps({"workflow_version": "test", "payload": payload}),
            "created_at": "2026-09-24T10:00:00+00:00",
            "updated_at": f"2026-09-24T10:{minutes:02d}:00+00:00"}


def test_mode_estimates_use_median_of_matching_runs_and_fall_back_to_defaults():
    from app.web_api import DEFAULT_MODE_SECONDS, mode_estimates

    rows = [history_row("fast", 4), history_row("fast", 9), history_row("fast", 5),
            # Не описывают нынешний быстрый режим: другой поставщик, прерванный
            # и продолженный прогон, неуспех, служебная операция, прежние лимиты.
            history_row("fast", 50, provider="deepseek"), history_row("fast", 50, attempt=2),
            history_row("fast", 50, state="failed"), history_row("fast", 50, operation="refine_candidate"),
            history_row("fast", 50, controls={"discovery_documents": 400, "candidate_limit": 8,
                                              "history_enabled": True, "patents_enabled": False}),
            {"state": "succeeded", "attempt": 1, "input_json": "not json",
             "created_at": "2026-09-24T10:00:00+00:00", "updated_at": "2026-09-24T10:30:00+00:00"}]
    estimates = mode_estimates(rows, "local")
    assert estimates["fast"] == {"expected_seconds": 300, "basis": "history", "runs": 3,
                                 "documents": 1000, "top": 8, "patents": False}
    assert estimates["deep"] == {"expected_seconds": DEFAULT_MODE_SECONDS["deep"], "basis": "default",
                                 "runs": 0, "documents": 10000, "top": 15, "patents": True}
    assert set(estimates) == {"fast", "deep"}


def test_mode_estimates_take_only_the_most_recent_runs():
    from app.web_api import ESTIMATE_SAMPLES, mode_estimates

    rows = [history_row("deep", 10)] * ESTIMATE_SAMPLES + [history_row("deep", 59)] * 20
    estimate = mode_estimates(rows, "local")["deep"]
    assert (estimate["expected_seconds"], estimate["runs"]) == (600, ESTIMATE_SAMPLES)


def test_started_run_carries_its_forecast_and_reports_elapsed_time():
    from datetime import UTC, datetime, timedelta

    started_at = (datetime.now(UTC) - timedelta(minutes=3)).isoformat()

    class Pilot:
        def status(self):
            return {"model_installed": True, "model_state": "ready", "settings": {"provider": "local"},
                    "keys": {}}

        def list_runs(self, *, offset, limit, source):
            assert (limit, source) == (50, "local")
            return [history_row("deep", 20)] if offset == 0 else []

        def start(self, query, *, collection_profile):
            return "run-id"

        def get(self, run_id):
            return {"state": "running", "created_at": started_at}

    service = object.__new__(WebAnalysisService)
    service.pilot = Pilot()
    service._admission = Lock()
    service._state_lock = Lock()
    service._current = None
    service._cancel_requested_id = None
    assert service.start_analysis("topic", "deep") == {
        "id": "run-id", "query": "topic", "mode": "deep", "expected_seconds": 1200, "state": "queued"}
    current = service.current_analysis()
    assert current["expected_seconds"] == 1200
    assert 175 <= current["elapsed_seconds"] <= 190


def test_forecast_failure_does_not_block_the_analysis():
    class Pilot:
        def status(self):
            return {"model_installed": True, "model_state": "ready", "settings": {"provider": "local"},
                    "keys": {}}

        def list_runs(self, **_):
            raise RuntimeError("history unavailable")

        def start(self, query, *, collection_profile):
            return "run-id"

    service = object.__new__(WebAnalysisService)
    service.pilot = Pilot()
    service._admission = Lock()
    service._state_lock = Lock()
    service._current = None
    service._cancel_requested_id = None
    assert service.start_analysis("topic", "fast") == {
        "id": "run-id", "query": "topic", "mode": "fast", "state": "queued"}


def test_estimates_endpoint_is_local_only_and_hides_internal_failures():
    class Service:
        fail = False

        def estimates(self):
            if self.fail:
                raise RuntimeError("C:/secret/path")
            return {"modes": {"fast": {"expected_seconds": 60}}}

    service = Service()
    server = ApiServer(("127.0.0.1", 0), ApiHandler)
    server.analysis = service
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(host=None):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            headers = {"Host": host} if host else {}
            connection.request("GET", "/analyses/estimates", headers=headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    try:
        assert request() == (HTTPStatus.OK, {"modes": {"fast": {"expected_seconds": 60}}})
        assert request("attacker.example")[0] == HTTPStatus.FORBIDDEN
        service.fail = True
        status, body = request()
        assert status == HTTPStatus.SERVICE_UNAVAILABLE and "secret" not in json.dumps(body)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_private_api_requires_launch_token_but_keeps_health_public(monkeypatch):
    token = "test-server-token-0123456789abcdef"
    monkeypatch.setenv("TREND_API_TOKEN", token)
    monkeypatch.setenv("TREND_API_INSTANCE_ID", "test-instance-0123456789abcdef")

    class Service:
        calls = 0
        health_calls = 0

        def status(self):
            self.health_calls += 1
            return {"ready": True, "model_state": "ready", "provider": "deepseek",
                    "provider_key_configured": True, "private_detail": "secret"}

        def current_analysis(self):
            self.calls += 1
            return {"state": "idle"}

        def start_analysis(self, query, mode):
            self.calls += 1
            return {"id": "new", "state": "queued", "query": query, "mode": mode}

    service = Service()
    server = ApiServer(("127.0.0.1", 0), ApiHandler)
    server.analysis = service
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, path, *, headers=None, payload=None):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        body = json.dumps(payload).encode() if payload is not None else None
        try:
            connection.request(method, path, body=body,
                               headers={"Content-Type": "application/json", **(headers or {})})
            response = connection.getresponse()
            return response.status, json.loads(response.read()), response.getheader("Connection")
        finally:
            connection.close()

    def request_with_duplicate_header(name, *, method="GET", path="/analyses/current", payload=None):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        body = json.dumps(payload).encode() if payload is not None else None
        try:
            connection.putrequest(method, path, skip_host=name == "Host")
            if name == "Host":
                connection.putheader("Host", f"127.0.0.1:{server.server_port}")
                connection.putheader("Host", f"127.0.0.1:{server.server_port}")
            else:
                connection.putheader("X-Trend-API-Token", token)
                if name == "X-Trend-API-Token":
                    connection.putheader(name, token)
                else:
                    connection.putheader("Content-Type", "application/json")
                    connection.putheader("Content-Length", str(len(body)))
                    connection.putheader("Content-Length", str(len(body)))
            connection.endheaders(body)
            response = connection.getresponse()
            response.read()
            return response.status
        finally:
            connection.close()

    def raw_post(body):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            connection.request("POST", "/analyses", body=body,
                               headers={"Content-Type": "application/json", "X-Trend-API-Token": token})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    try:
        assert request("GET", "/health")[:2] == (
            HTTPStatus.OK, {"ready": True, "model_state": "ready"})
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            assert response.getheader("X-Trend-Instance") == "test-instance-0123456789abcdef"
            assert "Python" not in response.getheader("Server", "")
            response.read()
        finally:
            connection.close()
        assert request("GET", "/analyses/current")[0] == HTTPStatus.FORBIDDEN
        assert request("POST", "/analyses", payload={"query": "topic"})[0] == HTTPStatus.FORBIDDEN
        assert service.calls == 0
        valid = {"X-Trend-API-Token": token}
        assert request("GET", "/analyses/current", headers=valid)[:2] == (
            HTTPStatus.OK, {"state": "idle"})
        assert request("POST", "/analyses", headers=valid, payload={"query": "topic"})[:2] == (
            HTTPStatus.ACCEPTED,
            {"id": "new", "state": "queued", "query": "topic", "mode": "fast"})
        assert service.calls == 2
        assert request("GET", "/analyses/current", headers={"X-Trend-API-Token": "wrong"})[0] == (
            HTTPStatus.FORBIDDEN)
        assert request("GET", "/analyses/current",
                       headers={"X-Trend-API-Token": "a" * 257})[0] == HTTPStatus.FORBIDDEN
        # The Streamlit client is server-side; even another local browser Origin
        # is not allowed to submit mutations directly to the private bridge.
        assert request("POST", "/analyses", headers=valid | {"Origin": "http://localhost:8501"},
                       payload={"query": "topic"})[0] == HTTPStatus.FORBIDDEN
        assert request("GET", "/health", headers={"Sec-Fetch-Site": "cross-site"})[0] == (
            HTTPStatus.FORBIDDEN)
        assert request("GET", "/analyses/current", headers=valid | {"Sec-Fetch-Site": "same-site"})[0] == (
            HTTPStatus.FORBIDDEN)
        assert request("GET", "/analyses/current", headers=valid | {"Host": "attacker.example"})[0] == (
            HTTPStatus.FORBIDDEN)
        assert request_with_duplicate_header("Host") == HTTPStatus.FORBIDDEN
        assert request_with_duplicate_header("X-Trend-API-Token") == HTTPStatus.FORBIDDEN
        assert request_with_duplicate_header("Content-Length", method="POST", path="/analyses",
                                             payload={"query": "topic"}) == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
        for raw in (b'{"query":"safe","query":"evil"}',
                    b'{"query":"topic","extra":{"same":1,"same":2}}',
                    b'{"query":"topic","mode":NaN}',
                    '{"query":"topic"}'.encode("utf-16"),
                    b"[" * 1_200 + b"]" * 1_200):
            assert raw_post(raw) == (HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "invalid_request"})
        assert request("GET", "/analyses/current?debug=1", headers=valid)[0] == HTTPStatus.NOT_FOUND
        assert request("GET", "/health")[2] == "close"
        assert service.health_calls == 1
        assert service.calls == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_current_analysis_omits_stable_result_only_for_matching_version(monkeypatch):
    monkeypatch.delenv("TREND_API_TOKEN", raising=False)

    class Service:
        def current_analysis(self):
            return {"id": "run-1", "query": "topic", "mode": "fast", "state": "succeeded",
                    "result_version": "a" * 64, "result": {"signals": [], "publications": [
                        {"title": "A study", "summary": "Scientific result" * 100 + "\ud800"}]}}

    server = ApiServer(("127.0.0.1", 0), ApiHandler)
    server.analysis = Service()
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(headers=None):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            connection.request("GET", "/analyses/current", headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    try:
        full_status, full_body = request()
        delta_status, delta_body = request({"X-Trend-Result-Version": "a" * 64})
        stale_status, stale_body = request({"X-Trend-Result-Version": "b" * 64})
        assert (full_status, delta_status, stale_status) == (HTTPStatus.OK,) * 3
        assert json.loads(full_body) == json.loads(stale_body)
        assert json.loads(full_body)["result"]["publications"][0]["summary"].endswith("\ud800")
        assert "result" not in json.loads(delta_body)
        assert json.loads(delta_body)["result_version"] == "a" * 64
        assert len(delta_body) < len(full_body) // 10
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_publication_pages_use_private_bounded_get_route(monkeypatch):
    token = "test-server-token-0123456789abcdef"
    monkeypatch.setenv("TREND_API_TOKEN", token)

    class Service:
        calls = 0

        def publications_page(self, run_id, offset, limit):
            self.calls += 1
            assert (run_id, offset, limit) == ("run-1", 200, 100)
            return {"run_id": run_id, "total": 1070, "offset": offset, "publications": []}

    service = Service()
    server = ApiServer(("127.0.0.1", 0), ApiHandler)
    server.analysis = service
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path, *, headers=None):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            connection.request("GET", path, headers=headers or {})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    path = "/analyses/current/publications?run_id=run-1&offset=200&limit=100"
    auth = {"X-Trend-API-Token": token}
    try:
        assert request(path)[0] == HTTPStatus.FORBIDDEN
        assert request(path, headers=auth | {"Origin": "http://localhost:8501"})[0] == HTTPStatus.FORBIDDEN
        status, page = request(path, headers=auth)
        assert status == HTTPStatus.OK and page == {
            "run_id": "run-1", "total": 1070, "offset": 200, "publications": []}
        assert service.calls == 1
        for invalid in ("&limit=99", "&extra=1", "&offset=-1", "&limit=101"):
            assert request(path + invalid, headers=auth)[0] == HTTPStatus.UNPROCESSABLE_ENTITY
        assert service.calls == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_completed_result_version_is_bound_to_run_and_cached():
    class Pilot:
        archive = None
        calls = 0

        def result(self, _run_id):
            self.calls += 1
            return {"result": {"top_trend_ids": [], "cards": []}}

    service = object.__new__(WebAnalysisService)
    service.pilot = Pilot()
    service._rendered_lock = Lock()
    service._rendered = {}
    first = service._web_result_for_run("first-run")
    version = service._rendered["first-run"][1]
    assert len(version) == 64
    assert service._web_result_for_run("first-run") is first
    assert service.pilot.calls == 1
    service._web_result_for_run("second-run")
    assert service._rendered["second-run"][1] != version
    # Several runs stay assembled: the history page and the current one alternate.
    assert service._web_result_for_run("first-run") is first
    assert service.pilot.calls == 2
    for index in range(5):
        service._web_result_for_run(f"run-{index}")
    assert "first-run" not in service._rendered and len(service._rendered) == 4


def test_api_rejects_invalid_token_configuration_and_caps_workers(monkeypatch):
    from app.web_api import MAX_API_CONNECTIONS

    with pytest.raises(ValueError, match="127.0.0.1"):
        ApiServer(("0.0.0.0", 0), ApiHandler)

    monkeypatch.delenv("TREND_API_TOKEN", raising=False)
    monkeypatch.delenv("TREND_API_ALLOW_UNAUTHENTICATED_LOCAL", raising=False)
    with pytest.raises(ValueError, match="TREND_API_TOKEN"):
        ApiServer(("127.0.0.1", 0), ApiHandler)

    monkeypatch.setenv("TREND_API_ALLOW_UNAUTHENTICATED_LOCAL", "true")
    with pytest.raises(ValueError, match="TREND_API_TOKEN"):
        ApiServer(("127.0.0.1", 0), ApiHandler)

    monkeypatch.setenv("TREND_API_ALLOW_UNAUTHENTICATED_LOCAL", "1")
    local_server = ApiServer(("127.0.0.1", 0), ApiHandler)
    local_server.server_close()

    monkeypatch.setenv("TREND_API_TOKEN", "short")
    with pytest.raises(ValueError, match="TREND_API_TOKEN"):
        ApiServer(("127.0.0.1", 0), ApiHandler)

    monkeypatch.setenv("TREND_API_TOKEN", "a" * 31)
    with pytest.raises(ValueError, match="TREND_API_TOKEN"):
        ApiServer(("127.0.0.1", 0), ApiHandler)

    monkeypatch.setenv("TREND_API_TOKEN", "a" * 32)
    server = ApiServer(("127.0.0.1", 0), ApiHandler)
    try:
        for _ in range(MAX_API_CONNECTIONS):
            assert server._client_slots.acquire(blocking=False)
        rejected = []
        monkeypatch.setattr(server, "shutdown_request", lambda request: rejected.append(request))
        marker = object()
        server.process_request(marker, ("127.0.0.1", 12345))
        assert rejected == [marker]
    finally:
        for _ in range(MAX_API_CONNECTIONS):
            server._client_slots.release()
        server.server_close()


def test_web_result_reports_openalex_budget_refusal():
    def result(reasons):
        return {"result": {"top_trend_ids": [], "cards": [], "snapshots": [
            {"purpose": "history", "coverage": [{"source": "openalex", "state": "unavailable",
                                                 "reasons": reasons}]}]}}

    assert web_result(result(["source_rate_limited"]))["openalex_rate_limited"] is True
    assert "openalex_rate_limited" not in web_result(result(["source_unavailable"]))


def assessment(identifier, confidence, measured, growth=False):
    return {"assessment": {"candidate_id": identifier, "confidence": confidence, "growth_confirmed": growth,
                           "components": [{"name": name, "value": 50.0 if name in measured else None}
                                          for name in ("growth", "persistence", "novelty",
                                                       "independence", "application")]}}


def test_publications_carry_trend_metadata_without_promoting_its_source(tmp_path):
    from app.pilot.archive import DocumentArchive
    from tests.test_pilot_evidence import document, snapshot

    archive = DocumentArchive(tmp_path / "revisions")
    older, newest, shared = document(1, year=2020), document(2, year=2026), document(3, year=2024)
    discovery = snapshot((older, newest, shared), archive)
    older_id, newest_id, shared_id = (reference.study_id for reference in discovery.documents)

    def topic(identifier, label, studies):
        return card(identifier) | {"category": "insufficient_evidence",
                                   "candidate": {"candidate_id": identifier, "label": label,
                                                 "discovery_study_ids": studies}}

    payload = {"result": {"top_trend_ids": [], "snapshots": [discovery.model_dump(mode="json")],
                          "cards": [topic("low", "Low topic", [older_id, shared_id]),
                                    topic("medium", " Medium topic ", [shared_id]),
                                    topic("unassessed", "Unassessed cluster", [newest_id])]},
               "assessments": [assessment("low", "low", set()),
                               assessment("medium", "medium", {"growth", "novelty"}, growth=True)]}
    response = web_result(payload, archive=archive)
    assert response["signals"] == []
    ranked = [(item["title"], item.get("trend", {}).get("title")) for item in response["top_publications"]]
    # A scientific trend verdict is descriptive metadata, not a source bonus.
    # All publications first compete by the same available-date rule.
    assert ranked == [(newest.title, None), (shared.title, "Medium topic"), (older.title, "Low topic")]
    assert response["top_publications"][1]["trend"] == {
        "title": "Medium topic", "confidence": "medium", "growth_confirmed": True,
        "checked_features": ["growth", "novelty"],
        "unchecked_features": ["persistence", "independence", "application"]}
    assert "trend" not in response["top_publications"][0]


def test_model_scores_sort_only_preselected_unified_top_and_keep_unscored_items_honest():
    approved = approved_snapshot([f"https://arxiv.org/abs/2609.{index:05d}" for index in range(16)])
    preliminary = web_result({"result": {"top_trend_ids": [], "cards": []},
                              "approved_sources": approved})
    chosen = preliminary["top_publications"]
    high, low = chosen[-1], chosen[0]
    scores = {
        high["publication_id"]: {"score": 93, "reason": "Direct topical match.",
                                  "evidence_quote": high["title"], "basis": "title_and_summary"},
        low["publication_id"]: {"score": 12, "reason": "Weak topical match.",
                                 "evidence_quote": low["title"], "basis": "title_and_summary"},
        chosen[1]["publication_id"]: {"score": 99, "reason": "Unsupported quote.",
                                       "evidence_quote": "never in metadata", "basis": "title_and_summary"},
    }
    result = web_result({"result": {"top_trend_ids": [], "cards": []},
                         "approved_sources": approved, "publication_confidences": scores})
    assert result["top_publications"][0]["publication_id"] == high["publication_id"]
    assert result["top_publications"][0]["model_confidence"]["score"] == 93
    assert result["top_publications"][1]["publication_id"] == low["publication_id"]
    assert result["top_publications"][1]["model_confidence"]["score"] == 12
    assert "model_confidence" not in result["top_publications"][2]
    assert result["top_publications"] == result["publications"][:15]
    assert {item["publication_id"] for item in result["top_publications"]} == {
        item["publication_id"] for item in chosen}
    assert result["publications"][15]["publication_id"] == preliminary["publications"][15]["publication_id"]


def test_top_signals_carry_the_methodology_confidence_in_the_trend():
    payload = {"result": {"top_trend_ids": ["one", "two", "three"],
                          "cards": [card("one"), card("two", category="early_signal"),
                                    card("three", category="weak_signal_candidate")]},
               "assessments": [assessment("one", "high", {"growth", "persistence", "novelty",
                                                          "independence", "application"}, growth=True),
                               assessment("two", "low", {"novelty"})]}
    first, second, third = web_result(payload)["signals"]
    assert (first["confidence"], first["growth_confirmed"], first["unchecked_features"]) == ("high", True, [])
    assert second["confidence"] == "low" and second["checked_features"] == ["novelty"]
    assert second["unchecked_features"] == ["growth", "persistence", "independence", "application"]
    # A result without an assessment shows no confidence rather than an invented one.
    assert "confidence" not in third


def test_top_texts_are_the_top_cards_titles_before_descriptions():
    from app.web_api import top_texts

    publication = {"title": "Paper", "summary": "Abstract.", "url": "https://doi.org/10.1234/A",
                   "trend": {"title": "Topic"}}
    signal_paper = {"title": "Signal paper", "summary": None, "url": "https://doi.org/10.1234/b"}
    beyond = {"title": "Beyond the TOP", "summary": "Not shown.", "url": "https://example.org/x"}
    result = {"signals": [{"title": "Signal", "summary": "Signal summary.",
                           "source_urls": ["https://doi.org/10.1234/B"]}],
              "publications": [publication, signal_paper, beyond], "top_publications": [publication]}
    # Signal cards outside the ranked TOP are in another section and do not
    # enter its translation job.
    assert top_texts(result) == ["Paper", "Topic", "Abstract."]
    result["top_publications"].append(signal_paper)
    assert top_texts(result) == ["Paper", "Topic", "Signal paper", "Signal", "Abstract.",
                                 "Signal summary."]


class BatchReader:
    """A fake translator: the batch call of the real one over a per-text rule."""

    def translate_many(self, texts, *, cancel=None, done=None):
        drafts = []
        for index, text in enumerate(texts):
            drafts.append(self.translate_text(text, cancel=cancel))
            if done is not None:
                done(index)
        return drafts


def translation_service(reader):
    from threading import Lock
    from types import SimpleNamespace

    service = WebAnalysisService.__new__(WebAnalysisService)
    service._translation_lock, service._translations, service._reader = Lock(), {}, reader
    service._reader_lock = Lock()
    service.pilot = SimpleNamespace(data_dir=None)
    return service


def finished(service, run_id, rendered):
    import threading

    service._translation_for_run(run_id, rendered)
    for thread in threading.enumerate():
        if thread.name == "top-translation":
            thread.join(timeout=10)
    return service._translation_for_run(run_id, rendered)


def test_top_translation_runs_once_in_the_background_and_arrives_whole():
    class Reader(BatchReader):
        calls = []

        def translate_text(self, text, *, cancel=None):
            self.calls.append(text)
            return "" if text == "Untranslatable" else f"RU {text}"

    reader = Reader()
    service = translation_service(reader)
    rendered = {"signals": [], "publications": [], "top_publications": [
        {"title": "Paper", "summary": "Untranslatable", "url": "https://example.org/a"}]}
    first = service._translation_for_run("run-1", rendered)
    assert first["state"] in {"running", "ready"} and first["total"] == 2
    assert first["state"] == "ready" or "texts" not in first  # never a half-translated TOP
    ready = finished(service, "run-1", rendered)
    assert ready == {"state": "ready", "completed": 2, "total": 2, "texts": {"Paper": "RU Paper"}}
    assert reader.calls == ["Paper", "Untranslatable"]  # polled again, translated once
    # Another run gets its own translation; the finished one is kept, not redone.
    assert finished(service, "run-2", rendered)["state"] == "ready"
    assert service._translation_for_run("run-1", rendered)["state"] == "ready"
    assert reader.calls == ["Paper", "Untranslatable"] * 2


def test_text_already_in_russian_is_not_sent_to_the_english_reader():
    class Reader(BatchReader):
        calls = []

        def translate_text(self, text, *, cancel=None):
            self.calls.append(text)
            return f"RU {text}"

    reader = Reader()
    service = translation_service(reader)
    rendered = {"signals": [], "publications": [], "top_publications": [
        {"title": "«В этом декабре уже 70% рекламных агентств используют нейронки»", "summary": None,
         "url": "https://example.org/ru"},
        {"title": "Neural networks in CAD", "summary": None, "url": "https://example.org/en"}]}
    ready = finished(service, "run-1", rendered)
    assert reader.calls == ["Neural networks in CAD"]
    assert ready == {"state": "ready", "completed": 2, "total": 2,
                     "texts": {"Neural networks in CAD": "RU Neural networks in CAD"}}


def test_missing_reading_model_leaves_the_top_in_the_original(monkeypatch):
    from app.pilot import translator

    def missing(*_args, **_kwargs):
        raise translator.TranslationError("Модель перевода отсутствует или повреждена.")

    monkeypatch.setattr(translator, "EnglishRussianTranslator", missing)
    service = translation_service(None)
    rendered = {"signals": [], "publications": [], "top_publications": [
        {"title": "Paper", "summary": None, "url": "https://example.org/a"}]}
    service._translation_for_run("run-1", rendered)
    state = finished(service, "run-1", rendered)
    assert state["state"] == "unavailable" and "texts" not in state
    assert "отсутствует" in state["message"]


def test_only_the_visible_part_of_an_abstract_is_translated():
    from app.web_api import EXCERPT_CHARACTERS, reading_excerpt

    short = "One sentence. Two sentences."
    assert reading_excerpt(short) == (short, False)
    long = " ".join(f"Sentence number {index} carries some words." for index in range(40))
    excerpt, clipped = reading_excerpt(long)
    assert clipped and len(excerpt) <= EXCERPT_CHARACTERS and excerpt.endswith("words.")
    assert long.startswith(excerpt)


def test_titles_reach_the_reader_before_the_descriptions():
    from threading import Event as Gate

    release = Gate()

    class Reader(BatchReader):
        def translate_text(self, text, *, cancel=None):
            if text.startswith("Abstract"):
                release.wait(timeout=10)
            return f"RU {text}"

    service = translation_service(Reader())
    rendered = {"signals": [], "publications": [], "top_publications": [
        {"title": f"Paper {index}", "summary": f"Abstract {index}.", "url": f"https://example.org/{index}"}
        for index in range(3)]}
    service._translation_for_run("run-1", rendered)
    for _ in range(200):
        partial = service._translation_for_run("run-1", rendered)
        if partial.get("texts"):
            break
        Event().wait(0.02)
    assert partial["state"] == "running"
    assert partial["texts"] == {f"Paper {index}": f"RU Paper {index}" for index in range(3)}
    release.set()
    ready = finished(service, "run-1", rendered)
    assert ready["state"] == "ready" and ready["texts"]["Abstract 1."] == "RU Abstract 1."


def saved_run(run_id, query, *, state="succeeded", mode="fast", operation=None,
              created_at="2026-09-26T12:35:09+00:00"):
    payload = {"query": query}
    if mode is not None:
        payload["collection_profile"] = mode
    if operation is not None:
        payload["operation"] = operation
    return {"id": run_id, "state": state, "attempt": 1, "created_at": created_at,
            "input_json": json.dumps({"workflow_version": "test", "payload": payload})}


class HistoryPilot:
    """Saved runs newest first, as the coordinator returns them."""

    def __init__(self, rows):
        self.rows = {row["id"]: row for row in rows}
        self.pages = []

    def list_runs(self, offset=0, limit=50, source="all"):
        assert source == "local" and 1 <= limit <= 50
        self.pages.append((offset, limit))
        return list(self.rows.values())[offset:offset + limit]

    def get(self, run_id):
        from app.runtime.jobs import TaskFailure

        if run_id not in self.rows:
            raise TaskFailure("Анализ не найден.")
        return dict(self.rows[run_id])


def history_service(pilot, current=None):
    service = object.__new__(WebAnalysisService)
    service.pilot = pilot
    service._admission = Lock()
    service._state_lock = Lock()
    service._current = current
    service._cancel_requested_id = "stale"
    return service


def test_history_lists_saved_analyses_without_service_operations_and_pages():
    from app.web_api import history_entry

    pilot = HistoryPilot([
        saved_run("run-3", "  solid-state batteries ", mode="deep"),
        saved_run("signals-1", "ignored", operation="signals"),
        saved_run("run-2", "desktop run", mode=None, state="failed",
                  created_at="2026-09-25T08:00:00"),
        saved_run("run-1", "oldest"),
    ])
    service = history_service(pilot, {"id": "run-3", "query": "solid-state batteries", "mode": "deep"})
    first = service.history(0, 3)
    assert pilot.pages == [(0, 4)]
    assert first["has_more"] and first["current_id"] == "run-3" and first["offset"] == 0
    assert first["runs"] == [
        {"id": "run-3", "query": "solid-state batteries", "mode": "deep", "state": "succeeded",
         "created_at": "2026-09-26T12:35:09+00:00"},
        # A run without a collection profile has no mode; a naive time is UTC.
        {"id": "run-2", "query": "desktop run", "mode": None, "state": "failed",
         "created_at": "2026-09-25T08:00:00+00:00"},
    ]
    last = service.history(3, 3)
    assert not last["has_more"] and [run["id"] for run in last["runs"]] == ["run-1"]
    for invalid in ((-1, 3), (0, 0), (0, 31), (0, True)):
        with pytest.raises(WebApiError) as failure:
            service.history(*invalid)
        assert failure.value.status == HTTPStatus.UNPROCESSABLE_ENTITY
    for broken in ({"input_json": "{"}, {"created_at": "yesterday"}, {"id": "../x"}, {"state": "lost"},
                   {"input_json": json.dumps({"payload": {"query": "  "}})}):
        assert history_entry({**saved_run("run-9", "topic"), **broken}) is None


def test_a_saved_analysis_is_read_while_another_runs_and_changes_nothing():
    pilot = HistoryPilot([saved_run("run-2", "running topic", state="running"),
                          saved_run("run-1", "old topic", mode=None),
                          saved_run("run-0", "broken", state="failed")])
    pilot.result = lambda run_id: {"result": {"top_trend_ids": ["one"], "cards": [card()]}}
    service = history_service(pilot, {"id": "run-2", "query": "running topic", "mode": "fast"})
    service._rendered_lock, service._rendered = Lock(), {}
    saved = service.saved_analysis("run-1")
    # A run started without a collection mode is shown in the default one.
    assert {key: saved[key] for key in ("id", "state", "query", "mode", "created_at")} == {
        "id": "run-1", "state": "succeeded", "query": "old topic", "mode": "fast",
        "created_at": "2026-09-26T12:35:09+00:00"}
    assert saved["result"]["signals"][0]["title"] == "Verified signal"
    assert len(saved["result_version"]) == 64
    # The shared current analysis keeps running and stays what the main page follows.
    assert service._current["id"] == "run-2" and service._cancel_requested_id == "stale"
    for run_id, status in (("run-2", HTTPStatus.CONFLICT), ("run-0", HTTPStatus.CONFLICT),
                           ("missing", HTTPStatus.NOT_FOUND), ("../run-1", HTTPStatus.UNPROCESSABLE_ENTITY)):
        with pytest.raises(WebApiError) as failure:
            service.saved_analysis(run_id)
        assert failure.value.status == status


def test_unwatched_background_work_of_an_old_page_gives_way(monkeypatch):
    from app import web_api

    clock = [100.0]
    monkeypatch.setattr(web_api.time, "monotonic", lambda: clock[0])

    def start(run_id):
        return lambda: {"run_id": run_id, "state": "running", "cancel": Event()}

    jobs = {}
    watched = web_api._job(jobs, "watched", start("watched"))
    left = web_api._job(jobs, "left", start("left"))
    clock[0] += web_api.JOB_IDLE_SECONDS - 1
    web_api._job(jobs, "watched", start("watched"))
    clock[0] += 2
    web_api._job(jobs, "new", start("new"))
    # The page nobody polls any more is cancelled; the watched one keeps working.
    assert left["cancel"].is_set() and "left" not in jobs
    assert not watched["cancel"].is_set() and list(jobs) == ["watched", "new"]
    for index in range(web_api.FINISHED_JOB_CACHE):
        web_api._job(jobs, f"extra-{index}", lambda: {"run_id": "x", "state": "ready", "cancel": Event()})
    assert len(jobs) == web_api.FINISHED_JOB_CACHE and watched["cancel"].is_set()


def test_history_and_saved_routes_are_private_and_bounded(monkeypatch):
    token = "test-server-token-0123456789abcdef"
    monkeypatch.setenv("TREND_API_TOKEN", token)
    version = "a" * 64

    class Service:
        def __init__(self):
            self.calls = []

        def history(self, offset, limit):
            self.calls.append(("history", offset, limit))
            return {"offset": offset, "has_more": False, "current_id": None, "runs": []}

        def saved_analysis(self, run_id):
            self.calls.append(("saved", run_id))
            if run_id == "missing":
                raise WebApiError(HTTPStatus.NOT_FOUND, "Анализ не найден.")
            return {"id": run_id, "state": "succeeded", "result": {"signals": []}, "result_version": version}

    service = Service()
    server = ApiServer(("127.0.0.1", 0), ApiHandler)
    server.analysis = service
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, path, body=None, *, headers=None):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            encoded = None if body is None else json.dumps(body).encode()
            connection.request(method, path, body=encoded, headers={
                **({"Content-Type": "application/json"} if body is not None else {}), **(headers or {})})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    auth = {"X-Trend-API-Token": token}
    try:
        assert request("GET", "/analyses/history?offset=0&limit=20")[0] == HTTPStatus.FORBIDDEN
        assert request("GET", "/analyses/saved?run_id=run-1")[0] == HTTPStatus.FORBIDDEN
        assert request("GET", "/analyses/history?offset=20&limit=20", headers=auth) == (
            HTTPStatus.OK, {"offset": 20, "has_more": False, "current_id": None, "runs": []})
        for invalid in ("?offset=0", "?offset=0&limit=20&x=1", "?offset=-1&limit=20",
                        "?offset=0&limit=200", "?offset=0&offset=1"):
            assert request("GET", "/analyses/history" + invalid, headers=auth)[0] == \
                HTTPStatus.UNPROCESSABLE_ENTITY
        status, saved = request("GET", "/analyses/saved?run_id=run-1", headers=auth)
        assert status == HTTPStatus.OK and saved["result"] == {"signals": []}
        # A page that already holds this result receives only the rest of the state.
        _, again = request("GET", "/analyses/saved?run_id=run-1",
                           headers=auth | {"X-Trend-Result-Version": version})
        assert "result" not in again and again["result_version"] == version
        assert request("GET", "/analyses/saved?run_id=missing", headers=auth)[0] == HTTPStatus.NOT_FOUND
        for invalid in ("", "?run_id=", "?run_id=../x", "?run_id=a&run_id=b", "?id=run-1"):
            assert request("GET", "/analyses/saved" + invalid, headers=auth)[0] == \
                HTTPStatus.UNPROCESSABLE_ENTITY
        # Viewing the history never switches the shared analysis.
        assert request("POST", "/analyses/open", {"id": "run-1"}, headers=auth)[0] == HTTPStatus.NOT_FOUND
        assert service.calls == [("history", 20, 20), ("saved", "run-1"), ("saved", "run-1"),
                                 ("saved", "missing")]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_finished_background_work_is_reused_when_a_saved_run_returns():
    import threading

    starts = []

    class Service(WebAnalysisService):
        def _build_radar(self, job, run_id, query):
            starts.append(run_id)
            with self._radar_lock:
                job.update(state="ready", result={"run": run_id})

    def settled():
        for thread in threading.enumerate():
            if thread.name == "technology-radar":
                thread.join(timeout=10)

    service = Service.__new__(Service)
    service._radar_lock, service._radars = Lock(), {}
    service._radar_for_run("run-1", "one")
    settled()
    assert service._radar_for_run("run-1", "one")["result"] == {"run": "run-1"}
    service._radar_for_run("run-2", "two")
    settled()
    reopened = service._radar_for_run("run-1", "one")
    assert reopened["state"] == "ready" and reopened["result"] == {"run": "run-1"}
    assert starts == ["run-1", "run-2"]


ALICE, BOB = "alice-visitor-0123456789", "bob-visitor-9876543210ab"


def visitor_service():
    class Pilot(HistoryPilot):
        def status(self):
            return {"model_installed": True, "model_state": "ready", "settings": {"provider": "local"},
                    "keys": {}}

        def start(self, query, *, collection_profile):
            run_id = f"run-{len(self.rows) + 1}"
            self.rows[run_id] = saved_run(run_id, query, state="running", mode=collection_profile)
            return run_id

        def result(self, _run_id):
            return {"result": {"top_trend_ids": ["one"], "cards": [card()]}}

        def cancel(self, _run_id):
            return True

        def list_runs(self, **_):
            raise AssertionError("A visitor never lists the whole profile")

    service = history_service(Pilot([]))
    service.separate_visitors = True
    service._runs, service._owners, service._latest = {}, {}, {}
    service._rendered_lock, service._rendered = Lock(), {}
    service._expected_seconds = lambda _mode: None
    return service


def test_visitors_see_only_their_own_analyses():
    service = visitor_service()
    assert service.current_analysis(visitor=ALICE) == {"state": "idle"}
    started = service.start_analysis("secret topic", "fast", visitor=ALICE)
    run_id = started["id"]
    assert service.current_analysis(visitor=ALICE)["query"] == "secret topic"
    # Bob learns only that the service is busy, never what Alice asked.
    assert service.current_analysis(visitor=BOB) == {"state": "idle", "service_busy": True}
    with pytest.raises(WebApiError) as foreign:
        service.cancel_analysis(run_id, visitor=BOB)
    assert foreign.value.status == HTTPStatus.CONFLICT
    assert service.history(0, 20, visitor=BOB) == {"offset": 0, "has_more": False, "current_id": None,
                                                    "runs": []}
    service.pilot.rows[run_id]["state"] = "succeeded"
    assert service.current_analysis(visitor=BOB) == {"state": "idle"}
    for read in (lambda: service.saved_analysis(run_id, visitor=BOB),
                 lambda: service.publications_page(run_id, 0, 10, visitor=BOB)):
        with pytest.raises(WebApiError) as hidden:
            read()
        assert hidden.value.status == HTTPStatus.NOT_FOUND
    mine = service.history(0, 20, visitor=ALICE)
    assert mine["current_id"] == run_id and [run["id"] for run in mine["runs"]] == [run_id]
    assert service.saved_analysis(run_id, visitor=ALICE)["query"] == "secret topic"
    # Bob's own analysis becomes his current one; Alice keeps hers.
    second = service.start_analysis("bob topic", "deep", visitor=BOB)["id"]
    assert service.current_analysis(visitor=BOB)["id"] == second
    assert service.current_analysis(visitor=ALICE)["id"] == run_id
    assert service.current_analysis(visitor=ALICE)["service_busy"] is True


def test_a_busy_service_queues_the_next_visitor_instead_of_refusing():
    service = visitor_service()
    alice = service.start_analysis("secret topic", "fast", visitor=ALICE)["id"]
    waiting = service.start_analysis("bob topic", "deep", visitor=BOB)
    # Очередь говорит Бобу его номер, но не чужой запрос.
    assert waiting["state"] == "waiting" and waiting["queue_position"] == 1 and "secret" not in str(waiting)
    status = service.current_analysis(visitor=BOB)
    assert (status["state"], status["query"], status["queue_position"]) == ("waiting", "bob topic", 1)
    with pytest.raises(WebApiError) as twice:
        service.start_analysis("another", "fast", visitor=BOB)
    assert twice.value.status == HTTPStatus.TOO_MANY_REQUESTS
    # Места нет, пока анализ Алисы идёт; освободилось — диспетчер запускает Боба.
    assert service._free_slots() == 0
    service.pilot.rows[alice]["state"] = "succeeded"
    assert service._free_slots() == 1 and service._dispatch_one() is True
    bob = service.current_analysis(visitor=BOB)
    assert bob["state"] == "running" and bob["query"] == "bob topic" and bob["id"] not in {alice, waiting["id"]}
    assert service.current_analysis(visitor=ALICE)["id"] == alice


def test_a_waiting_analysis_leaves_the_queue_when_cancelled():
    service = visitor_service()
    service.start_analysis("secret topic", "fast", visitor=ALICE)
    ticket = service.start_analysis("bob topic", "fast", visitor=BOB)["id"]
    with pytest.raises(WebApiError):
        service.cancel_analysis(ticket, visitor=ALICE)
    assert service.cancel_analysis(ticket, visitor=BOB)["state"] == "cancelled"
    assert service._queue == []
    # Отменённая заявка не мешает новой.
    assert service.start_analysis("bob again", "fast", visitor=BOB)["state"] == "waiting"


def test_separated_api_answers_only_a_named_visitor(monkeypatch):
    token = "test-server-token-0123456789abcdef"
    monkeypatch.setenv("TREND_API_TOKEN", token)

    class Service:
        separate_visitors = True

        def __init__(self):
            self.calls = []

        def current_analysis(self, *, visitor):
            self.calls.append(("current", visitor))
            return {"state": "idle"}

        def history(self, offset, limit, *, visitor):
            self.calls.append(("history", visitor))
            return {"offset": offset, "has_more": False, "current_id": None, "runs": []}

    service = Service()
    server = ApiServer(("127.0.0.1", 0), ApiHandler)
    server.analysis = service
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def get(path, headers):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            connection.request("GET", path, headers={"X-Trend-API-Token": token, **headers})
            return connection.getresponse().status
        finally:
            connection.close()

    try:
        for headers in ({}, {"X-Trend-Visitor": "short"}, {"X-Trend-Visitor": "bad visitor id value!"}):
            assert get("/analyses/current", headers) == HTTPStatus.FORBIDDEN
        assert get("/analyses/current", {"X-Trend-Visitor": ALICE}) == HTTPStatus.OK
        assert get("/analyses/history?offset=0&limit=20", {"X-Trend-Visitor": BOB}) == HTTPStatus.OK
        assert service.calls == [("current", ALICE), ("history", BOB)]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_owner_panel_routes_need_their_own_token_and_presence_needs_no_visitor(monkeypatch):
    token, admin = "test-server-token-0123456789abcdef", "test-admin-token-0123456789abcdef!"
    monkeypatch.setenv("TREND_API_TOKEN", token)
    monkeypatch.setenv("TREND_API_ADMIN_TOKEN", admin)

    class Service:
        separate_visitors = True

        def __init__(self):
            self.calls = []

        def admin_overview(self, after, person=None):
            self.calls.append(("overview", after))
            return {"events": []}

        def admin_cancel(self, run_id):
            self.calls.append(("cancel", run_id))
            return {"id": run_id, "state": "cancelling"}

        def presence(self, session, page, event, agent, ip, *, view, detail, visitor):
            self.calls.append(("presence", session, page, event, visitor))
            return {"access": None, "messages": []}

        def access(self, ip, *, visitor):
            self.calls.append(("access", ip, visitor))
            return {"access": "blocked" if ip == "198.51.100.66" else None}

        def admin_action(self, payload):
            self.calls.append(("action", payload["action"]))
            return {"message": "ok"}

        def visitor_refusal(self, visitor):
            return "signed_out" if visitor == BOB else None

        def current_analysis(self, *, visitor):
            return {"state": "idle"}

    service = Service()
    server = ApiServer(("127.0.0.1", 0), ApiHandler)
    server.analysis = service
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, path, headers, payload=None):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        body = json.dumps(payload).encode() if payload is not None else None
        try:
            connection.request(method, path, body=body, headers={"Content-Type": "application/json", **headers})
            return connection.getresponse().status
        finally:
            connection.close()

    site, owner = {"X-Trend-API-Token": token}, {"X-Trend-Admin-Token": admin}
    tab = {"session": "tab-0123456789", "page": "login", "event": "login_failed",
           "agent": "Mozilla/5.0", "ip": "203.0.113.7"}
    try:
        # Процесс сайта со своим токеном сводку обо всех посетителях не получит.
        assert request("GET", "/admin/overview?after=0", site) == HTTPStatus.FORBIDDEN
        assert request("POST", "/admin/cancel", site, {"id": "run-1"}) == HTTPStatus.FORBIDDEN
        assert request("GET", "/admin/overview?after=0", {"X-Trend-Admin-Token": token}) == HTTPStatus.FORBIDDEN
        assert request("GET", "/admin/overview?after=7", owner) == HTTPStatus.OK
        assert request("GET", "/admin/overview?after=x", owner) == HTTPStatus.UNPROCESSABLE_ENTITY
        assert request("POST", "/admin/cancel", owner, {"id": "run-1"}) == HTTPStatus.OK
        # Экран входа отмечается без посетителя; чужой токен и поддельный посетитель — нет.
        assert request("POST", "/presence", site, tab) == HTTPStatus.OK
        assert request("POST", "/presence", {**site, "X-Trend-Visitor": ALICE}, {**tab, "event": None}) == HTTPStatus.OK
        assert request("POST", "/presence", {**site, "X-Trend-Visitor": "bad"}, tab) == HTTPStatus.FORBIDDEN
        assert request("POST", "/presence", owner, tab) == HTTPStatus.FORBIDDEN
        assert request("POST", "/presence", site, {**tab, "extra": 1}) == HTTPStatus.UNPROCESSABLE_ENTITY
        assert request("POST", "/access", site, {"ip": "198.51.100.66"}) == HTTPStatus.OK
        assert request("POST", "/access", owner, {"ip": None}) == HTTPStatus.FORBIDDEN
        # Рычаги владельца — только с его токеном и в ожидаемом виде.
        person = "0123456789ab"
        assert request("POST", "/admin/action", site, {"action": "pause"}) == HTTPStatus.FORBIDDEN
        for bad in ({"action": "block"}, {"action": "block", "person": "../etc"},
                    {"action": "message", "person": person, "text": "   "},
                    {"action": "message", "person": person, "text": "x" * 501},
                    {"action": "unblock"}, {"action": "pause", "extra": 1}):
            assert request("POST", "/admin/action", owner, bad) == HTTPStatus.UNPROCESSABLE_ENTITY
        assert request("POST", "/admin/action", owner, {"action": "block", "person": person}) == HTTPStatus.OK
        assert request("GET", "/admin/overview?after=0&person=0123456789ab", owner) == HTTPStatus.OK
        assert request("GET", "/admin/overview?after=0&person=bad", owner) == HTTPStatus.UNPROCESSABLE_ENTITY
        # Выгнанный гость получает отказ на любой свой запрос.
        assert request("GET", "/analyses/current", {**site, "X-Trend-Visitor": BOB}) == HTTPStatus.FORBIDDEN
        assert request("GET", "/analyses/current", {**site, "X-Trend-Visitor": ALICE}) == HTTPStatus.OK
        assert service.calls == [("overview", 7), ("cancel", "run-1"),
                                 ("presence", "tab-0123456789", "login", "login_failed", None),
                                 ("presence", "tab-0123456789", "login", None, ALICE),
                                 ("access", "198.51.100.66", None), ("action", "block"), ("overview", 0)]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_owner_token_must_differ_from_the_site_token(monkeypatch):
    token = "test-server-token-0123456789abcdef"
    monkeypatch.setenv("TREND_API_TOKEN", token)
    monkeypatch.setenv("TREND_API_ADMIN_TOKEN", token)
    with pytest.raises(ValueError, match="TREND_API_ADMIN_TOKEN"):
        ApiServer(("127.0.0.1", 0), ApiHandler)


def test_owner_overview_names_visitors_and_follows_stages_of_the_current_run():
    from app.web_monitor import Monitor

    class Pilot:
        data_dir = "profile"
        row = {"state": "running", "stage": "discovery", "message": "Собираем публикации", "completed": 3,
               "total": 10, "created_at": "2026-09-27T10:00:00+00:00", "updated_at": "2026-09-27T10:00:30+00:00"}

        def status(self):
            return {"model_installed": True, "model_state": "ready", "settings": {"provider": "local"},
                    "keys": {"openalex_api_key": False}, "local_llm_installed": True}

        def start(self, query, *, collection_profile):
            return "run-now"

        def get(self, run_id):
            return dict(self.row)

        def cancel(self, run_id):
            return True

        def list_runs(self, *, offset, limit, source):
            saved = {"payload": {"query": "older topic", "collection_profile": "deep"}}
            return [{"id": "run-now", "state": "running", "stage": "discovery", "created_at": "2026-09-27T10:00:00+00:00",
                     "input_json": json.dumps({"payload": {"query": "solid-state batteries",
                                                           "collection_profile": "fast"}})},
                    {"id": "run-old", "state": "failed", "error": "Нет сети.", "created_at": "2026-09-26T09:00:00+00:00",
                     "updated_at": "2026-09-26T09:02:05+00:00", "input_json": json.dumps(saved)}]

    service = object.__new__(WebAnalysisService)
    service.pilot = Pilot()
    service.separate_visitors = True
    service._admission, service._state_lock = Lock(), Lock()
    service._current, service._cancel_requested_id = None, None
    service._runs, service._owners, service._latest = {}, {}, {}
    service.monitor = Monitor(anonymous="Не вошёл")
    # Наблюдатель этапов живёт до закрытия сервиса.
    service._closing = Event()
    service._expected_seconds = lambda mode: 240
    service.presence("tab-0123456789", "main", None, "Mozilla/5.0 Chrome/140.0 Windows", "203.0.113.7",
                     visitor=ALICE)
    service.start_analysis("solid-state batteries", "fast", visitor=ALICE)
    overview = service.admin_overview(0)
    current = overview["current"]
    assert (current["visitor"], current["stage_label"], current["completed"], current["expected_seconds"]) == (
        "Гость 1", "Сбор публикаций", 3, 240)
    assert [entry["label"] for entry in current["stages"]] == ["Сбор публикаций"]
    assert [(run["id"], run["visitor"], run.get("duration_seconds"), run.get("error")) for run in overview["runs"]] == [
        ("run-now", "Гость 1", None, None), ("run-old", None, 125, "Нет сети.")]
    assert overview["service"]["keys"]["openalex_api_key"] is False and overview["service"]["separate_visitors"]
    assert overview["visitors"][0]["runs"] == 1
    assert "Гость 1 запустил анализ «solid-state batteries» · быстрый" in [e["text"] for e in overview["events"]]
    assert service.admin_cancel("run-now") == {"id": "run-now", "state": "cancelling"}
    with pytest.raises(WebApiError):
        service.admin_cancel("run-old")
    service._closing.set()


def test_owner_resources_are_checked_applied_and_saved(tmp_path, monkeypatch):
    from app.pilot.local_llm import BATCH_VARIABLE
    from app.web_api import RESOURCE_DEFAULTS, valid_resources

    for bad in ({"slots": 0}, {"slots": 5}, {"llm_batch": "8"}, {"keep_llm_loaded": 1}, {"unknown": True}):
        with pytest.raises(ValueError):
            valid_resources(bad)

    class Coordinator:
        slots = 1

        def set_slots(self, slots):
            self.slots = slots

        def running(self):
            return []

    class Pilot:
        coordinator = Coordinator()
        keep_llm_loaded = False
        unloaded = False

        def unload_llm(self):
            self.unloaded = True
            return True

    monkeypatch.delenv(BATCH_VARIABLE, raising=False)
    service = object.__new__(WebAnalysisService)
    service.pilot, service._state_lock = Pilot(), Lock()
    service.resources, service._resources_path = dict(RESOURCE_DEFAULTS), tmp_path / "resources.json"
    service._apply_resources()
    assert service.pilot.coordinator.slots == 2 and service.pilot.keep_llm_loaded is True
    assert service.set_resources({"slots": 3, "llm_batch": 4, "keep_llm_loaded": False})["slots"] == 3
    assert (service.pilot.coordinator.slots, os.environ[BATCH_VARIABLE]) == (3, "4")
    # Выключенное «держать в памяти» при простое сразу освобождает видеопамять.
    assert service.pilot.unloaded is True
    assert json.loads((tmp_path / "resources.json").read_text(encoding="utf-8"))["llm_batch"] == 4


def test_analyst_summary_keeps_only_what_an_analyst_reads():
    from app.web_api import analyst_summary

    rendered = {"publication_total": 1236, "openalex_rate_limited": True, "incomplete_coverage": True,
                "signals": [{"title": "sensor networks", "summary": "A summary.", "category": "weak_signal_candidate",
                             "confidence": "low", "source_urls": ["https://doi.org/1", "https://doi.org/2"],
                             "checked_features": ["growth"], "growth_confirmed": False}],
                "top_publications": [{"title": "Paper", "published_at": "2026-09-01", "source_id": "arxiv",
                                      "kind": "preprint", "url": "https://arxiv.org/abs/1", "trend": {"title": "T"},
                                      "model_confidence": {"score": 82}}],
                "source_coverage": [{"source_id": "crossref", "state": "partial", "accepted": 900},
                                    {"source_id": "arxiv", "state": "complete", "accepted": 50},
                                    {"source_id": "github", "state": "complete", "accepted": 0}],
                "funding_evidence": {"awards": [{"award_amount_usd": "1000000"}, {"award_amount_usd": "500000"}]}}
    payload = {"radar": {"state": "ready", "result": {
        "technologies": [{"title": "loot box", "probability": 0.99, "is_signal": True, "pool_documents": 5,
                          "description": "Original.", "reasons": ["рост"], "sources": [
                              {"title": "Paper", "url": "https://arxiv.org/abs/1/", "published": "2026-09-01",
                               "source": "arXiv", "source_type": "препринт"},
                              {"title": "Duplicate", "url": "https://arxiv.org/abs/1/"},
                              {"title": "Unsafe", "url": "javascript:alert(1)"}]}],
        "excluded": [{"title": "gaming", "probability": 0.2, "is_signal": False, "reasons": ["зрелая тема"]}],
        "translation": {"Original.": "Перевод."}}}}
    summary = analyst_summary(rendered, payload, {"sensor networks": "сенсорные сети"})
    assert summary["numbers"] == {"publications": 1236, "signals": 1, "technologies": 1, "technology_candidates": 2,
                                  "sources": 2, "grants": 2, "grants_usd": 1_500_000}
    assert summary["signals"][0]["title_ru"] == "сенсорные сети" and summary["signals"][0]["sources"] == 2
    assert summary["technologies"][0]["description"] == "Перевод."
    assert summary["technologies"][0]["sources"] == [{
        "title": "Paper", "url": "https://arxiv.org/abs/1/", "published": "2026-09-01",
        "source": "arXiv", "type": "препринт", "model_confidence": 82}]
    assert summary["excluded"][0]["reasons"] == ["зрелая тема"]
    assert summary["top_publications"][0]["trend"] == "T"
    assert summary["top_publications"][0]["model_confidence"] == 82
    assert any("OpenAlex" in text for text in summary["warnings"])
    assert any("crossref" in text for text in summary["warnings"])


def test_analyst_summary_keeps_score_when_journal_url_replaces_preprint_url():
    from app.web_api import analyst_summary

    publication = {"title": "Narrow cathode mechanism", "url": "https://arxiv.org/abs/2401.1",
                   "model_confidence": {"score": 91}}
    source = {"title": "Narrow cathode mechanism", "url": "https://doi.org/10.1234/cathode",
              "source": "Crossref"}
    radar = {"state": "ready", "result": {"technologies": [{"title": "Cathode", "sources": [source]}]}}
    summary = analyst_summary({"top_publications": [publication]}, {"radar": radar}, {})
    assert summary["technologies"][0]["sources"][0]["model_confidence"] == 91

    # A generic title used by two publications must not transfer either score.
    duplicate = {"title": publication["title"], "url": "https://arxiv.org/abs/2401.2",
                 "model_confidence": {"score": 12}}
    ambiguous = analyst_summary({"top_publications": [publication, duplicate]}, {"radar": radar}, {})
    assert "model_confidence" not in ambiguous["technologies"][0]["sources"][0]


def test_analyst_summary_does_not_borrow_title_score_from_an_unscored_direct_url():
    from app.web_api import analyst_summary

    rendered = {"top_publications": [
        {"title": "Directly linked paper", "url": "https://doi.org/10.1234/direct"},
        {"title": "Scored paper", "url": "https://arxiv.org/abs/2401.1",
         "model_confidence": {"score": 91}},
    ]}
    radar = {"state": "ready", "result": {"technologies": [{"title": "Cathode", "sources": [
        {"title": "Scored paper", "url": "https://doi.org/10.1234/direct"}]}]}}

    summary = analyst_summary(rendered, {"radar": radar}, {})
    assert "model_confidence" not in summary["technologies"][0]["sources"][0]


def test_analyst_summary_does_not_choose_a_score_for_a_duplicate_normalized_url():
    from app.web_api import analyst_summary

    rendered = {"top_publications": [
        {"title": "First version", "url": "https://doi.org/10.1234/X",
         "model_confidence": {"score": 71}},
        {"title": "Second version", "url": "https://doi.org/10.1234/x/",
         "model_confidence": {"score": 88}},
    ]}
    radar = {"state": "ready", "result": {"technologies": [{"title": "Cathode", "sources": [
        {"title": "First version", "url": "https://doi.org/10.1234/x"}]}]}}

    summary = analyst_summary(rendered, {"radar": radar}, {})
    assert "model_confidence" not in summary["technologies"][0]["sources"][0]


@pytest.mark.parametrize(("version", "expected"), [
    ("radar/1.1.0", "radar/1.1.0"),
    ("radar/1.2.0", "radar/1.2.0"),
    ("radar/1.3.0", "radar/1.3.0"),
    ("radar/2.0.0", "radar/2.0.0"),
    ("radar/1.2.0<script>", None),
    ("radar/1234567.2.0", None),
    (123, None),
    (None, None),
])
def test_analyst_summary_exposes_only_safe_radar_policy_versions(version, expected):
    from app.web_api import analyst_summary

    radar = {"state": "ready", "result": {"policy_version": version, "technologies": []}}
    summary = analyst_summary({}, {"radar": radar}, {})
    assert summary["radar_policy_version"] == expected
