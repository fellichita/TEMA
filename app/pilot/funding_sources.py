"""One-page NIH RePORTER funding snapshot, separate from publications and VC deals.

The public Project Search API includes grants, cooperative agreements, contracts,
and intramural projects. We keep its funding mechanism and classify each record;
callers must not treat every project as a grant or as money already disbursed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import json
import math
import re
from threading import Event
from time import monotonic
from typing import Literal

import httpx

from app.runtime.jobs import TaskCancelled


NIH_PROJECT_SEARCH_URL = "https://api.reporter.nih.gov/v2/projects/search"
MAX_NIH_PAGE_SIZE = 50
MAX_NIH_RESPONSE_BYTES = 1_000_000
MAX_NIH_TIMEOUT_SECONDS = 15.0

FundingType = Literal["grant_or_cooperative", "contract", "intramural", "other_or_unknown"]
CoverageReason = Literal["complete", "page_cap", "invalid_records", "unknown_total"]


class FundingSourceError(RuntimeError):
    """Safe failure code; remote response bodies are never included."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class FundingAward:
    application_id: int
    project_number: str | None
    title: str
    award_notice_date: date
    award_amount_usd: Decimal
    funding_mechanism: str | None
    funding_type: FundingType
    detail_url: str

    def to_dict(self) -> dict[str, object]:
        return {
            "application_id": self.application_id,
            "project_number": self.project_number,
            "title": self.title,
            "award_notice_date": self.award_notice_date.isoformat(),
            "award_amount_usd": format(self.award_amount_usd, "f"),
            "funding_mechanism": self.funding_mechanism,
            "funding_type": self.funding_type,
            "detail_url": self.detail_url,
        }


@dataclass(frozen=True)
class FundingSnapshot:
    topic: str
    from_date: date
    to_date: date
    awards: tuple[FundingAward, ...]
    total_available: int | None
    records_returned: int
    rejected_records: int
    partial_coverage: bool
    coverage_reason: CoverageReason
    source_id: Literal["nih_reporter"] = "nih_reporter"
    date_basis: Literal["award_notice_date"] = "award_notice_date"
    amount_basis: Literal["reported_fiscal_year_award_usd"] = "reported_fiscal_year_award_usd"

    def to_dict(self) -> dict[str, object]:
        """JSON-safe data for a service boundary; Decimal amounts stay exact strings."""
        return {
            "source_id": self.source_id,
            "topic": self.topic,
            "from_date": self.from_date.isoformat(),
            "to_date": self.to_date.isoformat(),
            "date_basis": self.date_basis,
            "amount_basis": self.amount_basis,
            "awards": [award.to_dict() for award in self.awards],
            "total_available": self.total_available,
            "records_returned": self.records_returned,
            "rejected_records": self.rejected_records,
            "partial_coverage": self.partial_coverage,
            "coverage_reason": self.coverage_reason,
        }


def _check_cancel(cancel: Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise TaskCancelled()


def _topic(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 180:
        raise FundingSourceError("invalid_topic")
    # The advanced search uses its own query language. Pass only plain terms.
    terms = re.findall(r"[^\W_]+", value, flags=re.UNICODE)
    if not terms or len(terms) > 12:
        raise FundingSourceError("invalid_topic")
    return " ".join(terms)


def _validate(from_date: date, to_date: date, limit: int, timeout_seconds: float) -> None:
    if type(from_date) is not date or type(to_date) is not date or from_date > to_date:
        raise FundingSourceError("invalid_period")
    if type(limit) is not int or not 1 <= limit <= MAX_NIH_PAGE_SIZE:
        raise FundingSourceError("invalid_limit")
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= MAX_NIH_TIMEOUT_SECONDS):
        raise FundingSourceError("invalid_timeout")


def _funding_type(activity_code: object, mechanism: str | None) -> FundingType:
    activity = activity_code.upper() if isinstance(activity_code, str) else ""
    code = mechanism.upper() if mechanism else ""
    if activity.startswith("N") or code in {"NSRDC", "SRDC"} or "CONTRACT" in code:
        return "contract"
    if activity.startswith("Z") or code == "IM" or "INTRAMURAL" in code:
        return "intramural"
    if code in {"RP", "SB", "RC", "OR", "TR", "TI", "CO"} or code in {
            "NON-SBIR/STTR", "SBIR/STTR", "RESEARCH CENTERS", "OTHER RESEARCH-RELATED",
            "TRAINING, INDIVIDUAL", "TRAINING, INSTITUTIONAL", "CONSTRUCTION GRANTS"}:
        return "grant_or_cooperative"
    return "other_or_unknown"


def _date(value: object) -> date | None:
    if not isinstance(value, str) or not re.match(r"^\d{4}-\d{2}-\d{2}(?:$|T)", value):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _amount(value: object) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return None
    return amount if amount.is_finite() and 0 <= amount <= 1_000_000_000_000 else None


def _award(value: object, from_date: date, to_date: date) -> FundingAward | None:
    if not isinstance(value, dict):
        return None
    application_id = value.get("appl_id")
    title = value.get("project_title")
    notice_date = _date(value.get("award_notice_date"))
    amount = _amount(value.get("award_amount"))
    if (type(application_id) is not int or application_id <= 0 or not isinstance(title, str)
            or not 1 <= len(title.strip()) <= 500 or notice_date is None
            or not from_date <= notice_date <= to_date or amount is None
            or value.get("subproject_id") is not None):
        return None
    project_number = value.get("project_num")
    if not isinstance(project_number, str) or not 1 <= len(project_number) <= 100:
        project_number = None
    mechanism = value.get("funding_mechanism")
    if not isinstance(mechanism, str) or not 1 <= len(mechanism) <= 100:
        mechanism = None
    return FundingAward(
        application_id=application_id, project_number=project_number,
        title=" ".join(title.split()), award_notice_date=notice_date,
        award_amount_usd=amount, funding_mechanism=mechanism,
        funding_type=_funding_type(value.get("activity_code"), mechanism),
        detail_url=f"https://reporter.nih.gov/project-details/{application_id}",
    )


def _read_response(client: httpx.Client, payload: dict[str, object], timeout_seconds: float,
                   cancel: Event | None) -> dict[str, object]:
    _check_cancel(cancel)
    deadline = monotonic() + timeout_seconds
    try:
        with client.stream("POST", NIH_PROJECT_SEARCH_URL, json=payload,
                           headers={"Accept": "application/json"}, timeout=timeout_seconds,
                           follow_redirects=False) as response:
            if response.status_code == 429:
                raise FundingSourceError("rate_limited")
            if 300 <= response.status_code < 400:
                raise FundingSourceError("unexpected_redirect")
            if response.status_code >= 500:
                raise FundingSourceError("source_unavailable")
            if response.status_code != 200:
                raise FundingSourceError("source_http_error")
            body = bytearray()
            for chunk in response.iter_bytes(chunk_size=65536):
                _check_cancel(cancel)
                if monotonic() > deadline:
                    raise FundingSourceError("timeout")
                body.extend(chunk)
                if len(body) > MAX_NIH_RESPONSE_BYTES:
                    raise FundingSourceError("response_too_large")
    except httpx.TimeoutException as exc:
        _check_cancel(cancel)
        raise FundingSourceError("timeout") from exc
    except (httpx.HTTPError, OSError) as exc:
        _check_cancel(cancel)
        raise FundingSourceError("source_unavailable") from exc
    _check_cancel(cancel)
    try:
        data = json.loads(body)
    except (UnicodeError, ValueError) as exc:
        raise FundingSourceError("invalid_response") from exc
    if not isinstance(data, dict):
        raise FundingSourceError("invalid_response")
    return data


def fetch_nih_grants(topic: str, from_date: date, to_date: date, *,
                     client: httpx.Client | None = None, limit: int = MAX_NIH_PAGE_SIZE,
                     timeout_seconds: float = 12.0, cancel: Event | None = None) -> FundingSnapshot:
    """Fetch at most one page of public NIH funded projects for a bounded period.

    A complete flag means only that the NIH API reported no further matches for
    this exact query. It never claims complete historical coverage of a topic.
    The NIH API recommends no more than one request per second across callers.
    """
    clean_topic = _topic(topic)
    _validate(from_date, to_date, limit, timeout_seconds)
    _check_cancel(cancel)
    payload: dict[str, object] = {
        "criteria": {
            "advanced_text_search": {
                "operator": "and", "search_field": "projecttitle,terms,abstracttext",
                "search_text": clean_topic,
            },
            "award_notice_date": {
                "from_date": from_date.isoformat(), "to_date": to_date.isoformat(),
            },
            "exclude_subprojects": True,
        },
        "include_fields": [
            "ApplId", "ProjectNum", "ProjectTitle", "AwardNoticeDate", "AwardAmount",
            "FundingMechanism", "ActivityCode", "SubprojectId",
        ],
        "offset": 0,
        "limit": limit,
        "sort_field": "award_notice_date",
        "sort_order": "desc",
    }
    own_client = client is None
    http = client if client is not None else httpx.Client(
        trust_env=False, follow_redirects=False,
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
    )
    try:
        data = _read_response(http, payload, timeout_seconds, cancel)
    finally:
        if own_client:
            http.close()
    results = data.get("results")
    if not isinstance(results, list) or len(results) > limit:
        raise FundingSourceError("invalid_response")
    meta = data.get("meta")
    raw_total = meta.get("total") if isinstance(meta, dict) else None
    if raw_total is None:
        total = None
    elif type(raw_total) is int and raw_total >= len(results):
        total = raw_total
    else:
        raise FundingSourceError("invalid_response")
    awards: list[FundingAward] = []
    seen: set[int] = set()
    rejected = 0
    for raw in results:
        _check_cancel(cancel)
        award = _award(raw, from_date, to_date)
        if award is None or award.application_id in seen:
            rejected += 1
            continue
        seen.add(award.application_id)
        awards.append(award)
    reason: CoverageReason = (
        "unknown_total" if total is None else
        "page_cap" if total > len(results) else
        "invalid_records" if rejected else "complete"
    )
    return FundingSnapshot(
        topic=clean_topic, from_date=from_date, to_date=to_date, awards=tuple(awards),
        total_available=total, records_returned=len(results), rejected_records=rejected,
        partial_coverage=reason != "complete", coverage_reason=reason,
    )
