"""Audit A06: source exhaustion does not prove consistent, complete coverage."""

import httpx
import pytest

from app.backend.config import BackendSettings
from app.backend.contracts import JobRecord, SearchRequest, utc_now
from app.backend.history import HistoryRequest
from app.backend.providers.crossref import CrossrefProvider
from app.backend.providers.openalex import OpenAlexProvider
from app.backend.service import Backend
from app.ui.presentation import empty_collection, job_state


def crossref_response(total, count=1):
    message = {"items": [{"DOI": "10.1234/audit", "title": ["Coverage audit"]}] * count,
               "next-cursor": "opaque-next"}
    if total is not None:
        message["total-results"] = total
    return {"status": "ok", "message": message}


def settings(path):
    return BackendSettings(data_dir=path, history_period_delay_seconds=0)


def crossref(client, page_size=100):
    return CrossrefProvider(client, page_size=page_size, page_delay=0, max_retries=0)


@pytest.mark.parametrize("total,count,complete", [
    (100, 1, False), (100, 0, False), (1, 1, True), (0, 0, True),
    (None, 1, True), (None, 0, True),
])
def test_real_crossref_short_page_retains_honest_coverage_after_reopen(tmp_path, total, count, complete):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=crossref_response(total, count))

    with (httpx.Client(transport=httpx.MockTransport(handler)) as client,
          Backend(settings(tmp_path), lambda: crossref(client)) as backend):
        job = backend.collect(SearchRequest(topic="coverage audit", max_results=100))
        assert job.state == "succeeded"
        assert job.scanned == job.stored == count
        assert job.total_available == total and job.source_exhausted
        assert job.coverage_complete is complete
        assert backend.list_documents(job_id=job.id).total == count
    # Crossref documents short pages as terminal, even with an opaque next cursor.
    assert len(calls) == 1
    with Backend(settings(tmp_path)) as backend:
        reopened = backend.get_job(job.id)
        assert reopened == job and reopened.coverage_complete is complete
        assert empty_collection(reopened) is (count == 0 and complete)
        if not complete:
            assert job_state(reopened) == "Завершено — неполная выборка"


@pytest.mark.parametrize("later_total", [None, 0, 1])
def test_later_page_cannot_erase_a_known_larger_source_total(tmp_path, later_total):
    calls = []

    def handler(request):
        calls.append(request)
        body = crossref_response(100) if len(calls) == 1 else crossref_response(later_total, 0)
        return httpx.Response(200, json=body)

    with (httpx.Client(transport=httpx.MockTransport(handler)) as client,
          Backend(settings(tmp_path), lambda: crossref(client, page_size=1)) as backend):
        job = backend.collect(SearchRequest(topic="coverage audit", max_results=100))
        assert job.state == "succeeded" and job.source_exhausted
        assert job.scanned == job.stored == 1
        assert job.total_available == 100 and not job.coverage_complete
    assert len(calls) == 2
    with Backend(settings(tmp_path)) as backend:
        assert backend.get_job(job.id) == job


def test_increasing_source_total_is_recorded(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=crossref_response(2 if len(calls) == 1 else 100))

    with (httpx.Client(transport=httpx.MockTransport(handler)) as client,
          Backend(settings(tmp_path), lambda: crossref(client, page_size=1)) as backend):
        job = backend.collect(SearchRequest(topic="coverage audit", max_results=2))
        assert job.total_available == 100 and not job.coverage_complete
        assert job.scanned == 2 and not job.source_exhausted
    assert len(calls) == 2


def test_raw_count_is_not_confused_with_deduplicated_storage(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=crossref_response(2))

    with (httpx.Client(transport=httpx.MockTransport(handler)) as client,
          Backend(settings(tmp_path), lambda: crossref(client, page_size=1)) as backend):
        job = backend.collect(SearchRequest(topic="coverage audit", max_results=2))
        assert job.scanned == job.total_available == 2 and job.stored == 1
        assert job.source_exhausted and job.coverage_complete
    assert len(calls) == 2


def test_openalex_empty_page_with_known_matches_is_incomplete(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"meta": {"count": 100, "next_cursor": None}, "results": []})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        def factory():
            return OpenAlexProvider(client=client, page_delay=0, max_retries=0)

        with Backend(settings(tmp_path), factory) as backend:
            job = backend.collect(SearchRequest(topic="coverage audit", source="openalex"))
            assert job.state == "succeeded" and job.source_exhausted
            assert job.scanned == job.stored == 0 and job.total_available == 100
            assert not job.coverage_complete and not empty_collection(job)
    assert len(calls) == 1


def test_inconsistent_history_can_be_retried_without_rewriting_its_old_job(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=crossref_response(100 if len(calls) == 1 else 1))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        def factory():
            return crossref(client)

        with Backend(settings(tmp_path), factory) as backend:
            report = backend.collect_history(HistoryRequest(
                topic="coverage audit", from_date="2024-01-01", until_date="2024-12-31",
                period="year", sources=("crossref",),
            ))
            period = report.periods[0]
            assert report.state == "partial" and not report.coverage_complete
            assert period.incomplete_reason == "inconsistent_total"
            assert period.job is not None and not period.job.coverage_complete
        with Backend(settings(tmp_path), factory) as backend:
            assert backend.get_history(report.id) == report
            assert backend.get_history_progress(report.id).periods[0].coverage == "incomplete"
            backend.resume_history(report.id, retry_incomplete=True)
            retried = backend.wait_history(report.id)
            assert retried.coverage_complete and retried.periods[0].job.coverage_complete
            assert len(retried.periods[0].attempts) == 2
            assert backend.get_job(period.job.id) == period.job
            assert not backend.get_job(period.job.id).coverage_complete
    assert len(calls) == 2


@pytest.mark.parametrize("state,scanned,stored,skipped,total,exhausted,expected", [
    ("succeeded", 0, 0, 0, 0, True, True),
    ("succeeded", 0, 0, 0, 1, True, False),
    ("succeeded", 1, 1, 0, 100, True, False),
    ("succeeded", 1, 1, 0, None, True, True),
    ("succeeded", 2, 1, 0, 2, True, True),
    ("succeeded", 2, 1, 1, 2, True, False),
    ("succeeded", 1, 1, 0, 1, False, False),
    *[(state, 1, 1, 0, 1, True, False)
      for state in ("queued", "running", "failed", "cancelled", "interrupted")],
])
def test_existing_job_json_uses_the_same_coverage_invariant(
    state, scanned, stored, skipped, total, exhausted, expected,
):
    job = JobRecord(
        id="saved-job", request=SearchRequest(topic="coverage audit"), state=state,
        created_at=utc_now(), updated_at=utc_now(), scanned=scanned, stored=stored,
        skipped=skipped, total_available=total, source_exhausted=exhausted,
    )
    assert JobRecord.model_validate_json(job.model_dump_json()).coverage_complete is expected
