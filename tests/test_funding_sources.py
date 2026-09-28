"""The NIH funding adapter stays bounded and never promotes partial data to full coverage."""

from __future__ import annotations

from datetime import date
import json
from threading import Event

import httpx
import pytest

from app.pilot.funding_sources import (
    FundingSourceError, MAX_NIH_PAGE_SIZE, MAX_NIH_RESPONSE_BYTES, fetch_nih_grants,
)
from app.runtime.jobs import TaskCancelled


FROM = date(2025, 1, 1)
TO = date(2025, 12, 31)


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _row(appl_id: int = 101, **overrides):
    value = {
        "appl_id": appl_id,
        "project_num": "5R01AI123456-02",
        "project_title": "Novel vaccine delivery",
        "award_notice_date": "2025-04-10T04:00:00Z",
        "award_amount": 334377,
        "funding_mechanism": "Non-SBIR/STTR",
        "activity_code": "R01",
        "subproject_id": None,
    }
    value.update(overrides)
    return value


def test_public_project_search_one_page_has_safe_fixed_request_and_partial_coverage():
    calls = []

    def handler(request):
        calls.append(request)
        assert request.method == "POST"
        assert str(request.url) == "https://api.reporter.nih.gov/v2/projects/search"
        assert request.headers.get("authorization") is None
        payload = json.loads(request.content)
        assert payload["offset"] == 0 and payload["limit"] == MAX_NIH_PAGE_SIZE
        assert payload["criteria"] == {
            "advanced_text_search": {
                "operator": "and", "search_field": "projecttitle,terms,abstracttext",
                "search_text": "vaccine delivery",
            },
            "award_notice_date": {"from_date": "2025-01-01", "to_date": "2025-12-31"},
            "exclude_subprojects": True,
        }
        assert "funding_mechanism" not in payload["criteria"]
        assert payload["include_fields"] == [
            "ApplId", "ProjectNum", "ProjectTitle", "AwardNoticeDate", "AwardAmount",
            "FundingMechanism", "ActivityCode", "SubprojectId",
        ]
        return httpx.Response(200, json={"meta": {"total": 80}, "results": [_row()]})

    snapshot = fetch_nih_grants("vaccine / delivery", FROM, TO, client=_client(handler))
    assert len(calls) == 1
    assert snapshot.source_id == "nih_reporter"
    assert snapshot.partial_coverage and snapshot.coverage_reason == "page_cap"
    assert snapshot.total_available == 80 and snapshot.records_returned == 1
    award, = snapshot.awards
    assert award.application_id == 101 and award.funding_type == "grant_or_cooperative"
    assert award.detail_url == "https://reporter.nih.gov/project-details/101"
    assert award.award_amount_usd == 334377
    assert snapshot.amount_basis == "reported_fiscal_year_award_usd"
    assert json.loads(json.dumps(snapshot.to_dict()))["awards"][0]["award_amount_usd"] == "334377"


def test_complete_only_with_known_total_and_all_records_valid():
    snapshot = fetch_nih_grants("sensor", FROM, TO, limit=4, client=_client(
        lambda _: httpx.Response(200, json={"meta": {"total": 4}, "results": [
            _row(1), _row(2, activity_code="N01", funding_mechanism="R&D Contracts"),
            _row(3, activity_code="Z01", funding_mechanism="Intramural Research"),
            _row(4, activity_code="X01", funding_mechanism="Unknown"),
        ]})))
    assert snapshot.partial_coverage is False and snapshot.coverage_reason == "complete"
    assert [award.funding_type for award in snapshot.awards] == [
        "grant_or_cooperative", "contract", "intramural", "other_or_unknown",
    ]


def test_bad_and_duplicate_rows_are_rejected_and_coverage_remains_partial():
    rows = [
        _row(1), _row(1), _row(2, award_amount="NaN"),
        _row(3, award_notice_date="2026-01-01T00:00:00Z"),
        _row(4, subproject_id=10),
    ]
    snapshot = fetch_nih_grants("sensor", FROM, TO, client=_client(
        lambda _: httpx.Response(200, json={"meta": {"total": 5}, "results": rows})))
    assert [item.application_id for item in snapshot.awards] == [1]
    assert snapshot.rejected_records == 4
    assert snapshot.partial_coverage and snapshot.coverage_reason == "invalid_records"


def test_unknown_total_is_partial_even_for_empty_results():
    snapshot = fetch_nih_grants("sensor", FROM, TO, client=_client(
        lambda _: httpx.Response(200, json={"results": []})))
    assert snapshot.awards == ()
    assert snapshot.partial_coverage and snapshot.coverage_reason == "unknown_total"


@pytest.mark.parametrize("status,code", [
    (302, "unexpected_redirect"), (429, "rate_limited"), (500, "source_unavailable"),
])
def test_http_statuses_fail_without_following_or_exposing_response(status, code):
    client = _client(lambda _: httpx.Response(status, headers={"location": "https://evil.example/"},
                                            content=b"private-secret"))
    with pytest.raises(FundingSourceError, match=code) as error:
        fetch_nih_grants("sensor", FROM, TO, client=client)
    assert "private-secret" not in str(error.value)


def test_body_limit_and_malformed_shape_fail_closed():
    big = _client(lambda _: httpx.Response(200, content=b"x" * (MAX_NIH_RESPONSE_BYTES + 1)))
    with pytest.raises(FundingSourceError, match="response_too_large"):
        fetch_nih_grants("sensor", FROM, TO, client=big)
    bad = _client(lambda _: httpx.Response(200, json={"meta": {"total": 1}, "results": [_row(), _row(2)]}))
    with pytest.raises(FundingSourceError, match="invalid_response"):
        fetch_nih_grants("sensor", FROM, TO, client=bad)


def test_cancelled_before_network_and_during_body():
    cancel = Event()
    cancel.set()
    calls = []
    client = _client(lambda request: calls.append(request) or httpx.Response(200, json={"results": []}))
    with pytest.raises(TaskCancelled):
        fetch_nih_grants("sensor", FROM, TO, client=client, cancel=cancel)
    assert calls == []


@pytest.mark.parametrize("topic,from_date,to_date,limit,timeout,code", [
    ("", FROM, TO, 50, 10.0, "invalid_topic"),
    ("sensor", TO, FROM, 50, 10.0, "invalid_period"),
    ("sensor", FROM, TO, 51, 10.0, "invalid_limit"),
    ("sensor", FROM, TO, 50, 16.0, "invalid_timeout"),
])
def test_invalid_budgets_are_rejected_before_network(topic, from_date, to_date, limit, timeout, code):
    with pytest.raises(FundingSourceError, match=code):
        fetch_nih_grants(topic, from_date, to_date, limit=limit, timeout_seconds=timeout,
                         client=_client(lambda _: pytest.fail("unexpected network request")))
