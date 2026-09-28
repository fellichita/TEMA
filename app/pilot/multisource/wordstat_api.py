"""Opt-in Yandex Search API v2 monthly Wordstat import with a durable spend fence."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

import httpx

from app.pilot.contracts import content_hash
from app.pilot.multisource.contracts import (QueryProfile, SearchObservation, SourceSnapshot,
                                             WordstatImportReceipt, load_policy)
from app.pilot.multisource.queries import compile_wordstat_monthly
from app.pilot.multisource.store import SignalStore
from app.runtime.budget import BudgetError, BudgetLimits, BudgetService, RequestAllowance
from app.runtime.jobs import TaskFailure

ENDPOINT = "https://searchapi.api.cloud.yandex.net/v2/wordstat/dynamics"
PRICE_MICRO_RUB = 20_000  # Official RUB price on 2026-09-19: 20 RUB / 1000 GetDynamics calls.
MAX_RESPONSE_BYTES = 3_000_000


def reserve_wordstat_request(connection, request_id: str, hour_cap: int, daily_cap: int) -> None:
    """Coordinator maintenance function: reserve once before any network dispatch."""
    budget = BudgetService(connection)
    now = datetime.now(UTC)
    scopes = (("wordstat/hour/" + now.strftime("%Y%m%dT%H"), hour_cap),
              ("wordstat/day/" + now.strftime("%Y%m%d"), daily_cap))
    for identifier, cap in scopes:
        desired = BudgetLimits(cap, 0, 0, cap * PRICE_MICRO_RUB)
        existing = connection.execute("SELECT 1 FROM pilot_budget_scopes WHERE scope_id=?", (identifier,)).fetchone()
        if existing is None:
            budget.create_scope(identifier, desired, currency="RUB")
        elif budget.snapshot(identifier).limits != desired:
            budget.update_limits(identifier, desired)
    budget.reserve(request_id, tuple(identifier for identifier, _ in scopes),
                   RequestAllowance(0, 0, PRICE_MICRO_RUB))
    budget.mark_sent(request_id)


def settle_wordstat_request(connection, request_id: str, charged: bool) -> None:
    BudgetService(connection).settle(request_id, RequestAllowance(0, 0, PRICE_MICRO_RUB if charged else 0))


def uncertain_wordstat_request(connection, request_id: str) -> None:
    BudgetService(connection).mark_unknown(request_id)


def _monthly_periods(first: date, last: date) -> tuple[date, ...]:
    periods = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        periods.append(date(year, month, 1))
        year, month = year + (month == 12), month % 12 + 1
    return tuple(periods)


def _duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate API response key")
        value[key] = item
    return value


def _rows(raw: bytes, expected: tuple[date, ...]) -> tuple[tuple[date, int, str], ...]:
    try:
        payload = json.loads(raw.decode("utf-8"), parse_float=Decimal, object_pairs_hook=_duplicates,
                             parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Invalid number")))
        if not isinstance(payload, dict) or set(payload) != {"results"} or not isinstance(payload["results"], list):
            raise ValueError("Unexpected API response")
        if len(payload["results"]) > 120 or len(payload["results"]) != len(expected):
            raise ValueError("Incomplete or excessive monthly response")
        by_period = {}
        for row in payload["results"]:
            if not isinstance(row, dict) or set(row) != {"date", "count", "share"}:
                raise ValueError("Invalid monthly row")
            timestamp = datetime.fromisoformat(row["date"].replace("Z", "+00:00"))
            if timestamp.utcoffset() != timedelta(0) or timestamp.day != 1 or timestamp.time() != datetime.min.time():
                raise ValueError("API month is not UTC month start")
            period = timestamp.date()
            count_text = row["count"]
            if not isinstance(count_text, str) or not re.fullmatch(r"[0-9]{1,19}", count_text):
                raise ValueError("Invalid API count")
            count = int(count_text)
            if count > 2**63 - 1:
                raise ValueError("API count exceeds int64")
            if not isinstance(row["share"], (str, int, Decimal)) or isinstance(row["share"], bool):
                raise ValueError("Invalid API share type")
            share = Decimal(str(row["share"]))
            share_text = format(share, "f")
            if not share.is_finite() or share < 0 or not re.fullmatch(r"[0-9]{1,20}(?:\.[0-9]{1,20})?", share_text):
                raise ValueError("Invalid API share")
            if period in by_period:
                raise ValueError("Duplicate API month")
            by_period[period] = (period, count, share_text)
        if set(by_period) != set(expected):
            raise ValueError("Missing or extra API month")
        return tuple(by_period[period] for period in expected)
    except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError, AttributeError):
        raise TaskFailure("Ответ Wordstat API не соответствует ожидаемой месячной схеме.") from None


def fetch_wordstat_dynamics(store: SignalStore, profile_hash: str, *, folder_id: str,
                            api_key: str, from_date: date, to_date: date,
                            hour_cap: int, daily_cap: int, coordinator,
                            transport: httpx.BaseTransport | None = None) -> str:
    """One explicit paid call; never retry a sent request automatically."""
    profile = store.get_object(profile_hash, QueryProfile)
    policy, _ = load_policy()
    if to_date >= date.today().replace(day=1):
        raise TaskFailure("Для API запрашивайте только завершённые месяцы.")
    request = compile_wordstat_monthly(profile, policy, from_date=from_date, to_date=to_date)[0]
    body = request.api_body(folder_id)
    if not isinstance(api_key, str) or not api_key or any(char.isspace() for char in api_key):
        raise TaskFailure("Ключ Wordstat API не настроен.")
    if not 1 <= hour_cap <= policy.wordstat_attempt_limit_per_hour or not 1 <= daily_cap <= 200:
        raise TaskFailure("Лимит обращений к Wordstat некорректен.")
    if len(_monthly_periods(from_date, to_date)) > 120:
        raise TaskFailure("Запрос Wordstat превышает десять лет помесячных данных.")
    request_id = "wordstat-" + uuid4().hex
    try:
        coordinator.maintenance(reserve_wordstat_request, request_id, hour_cap, daily_cap)
    except BudgetError:
        raise TaskFailure("Лимит Wordstat исчерпан или требует сверки перед новым запросом.") from None
    settled = False
    try:
        with httpx.Client(timeout=httpx.Timeout(20.0, connect=5.0), follow_redirects=False,
                          trust_env=False, transport=transport) as client:
            with client.stream("POST", ENDPOINT, json=body,
                               headers={"Authorization": "Api-Key " + api_key,
                                        "Content-Type": "application/json"}) as response:
                if response.status_code in {401, 403}:
                    coordinator.maintenance(settle_wordstat_request, request_id, False)
                    settled = True
                    raise TaskFailure("Wordstat API отклонил ключ или права доступа.")
                if response.status_code != 200:
                    raise TaskFailure("Wordstat API недоступен или ограничил частоту запросов.")
                chunks = bytearray()
                for block in response.iter_bytes():
                    chunks.extend(block)
                    if len(chunks) > MAX_RESPONSE_BYTES:
                        raise TaskFailure("Ответ Wordstat API превышает допустимый размер.")
                raw = bytes(chunks)
        coordinator.maintenance(settle_wordstat_request, request_id, True)
        settled = True
    except TaskFailure:
        if not settled:
            coordinator.maintenance(uncertain_wordstat_request, request_id)
        raise
    except (httpx.HTTPError, OSError):
        if not settled:
            coordinator.maintenance(uncertain_wordstat_request, request_id)
        raise TaskFailure("Нет ответа Wordstat API; расход требует сверки перед повтором.") from None
    expected = _monthly_periods(from_date, to_date)
    rows = _rows(raw, expected)
    observed = datetime.now(UTC)
    api_request_hash = content_hash({"compiled": request.request_hash, "folder_id": folder_id})
    temporary: Path | None = None
    try:
        store.root.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="wb", suffix=".json", dir=store.root.parent, delete=False) as handle:
            handle.write(raw)
            temporary = Path(handle.name)
        raw_hash = store.put_raw(temporary, "json", retention="local_allowed")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    snapshot = SourceSnapshot(source="wordstat", adapter_version="wordstat-api-v2/1",
                              request_hash=api_request_hash, query_profile_hash=profile_hash,
                              observed_at=observed, available_at=observed, coverage="complete", comparable=True,
                              raw_hash=raw_hash, source_url=ENDPOINT, retention="local_allowed",
                              export_right="local_only")
    snapshot_hash = store.put_object(snapshot)
    observations = []
    for period, count, share in rows:
        next_month = date(period.year + (period.month == 12), period.month % 12 + 1, 1)
        item = SearchObservation(series_id=api_request_hash, query_profile_hash=profile_hash,
                                 snapshot_hash=snapshot_hash, phrase=request.phrase,
                                 phrase_role="technology", matching_mode=request.matching_mode,
                                 region_ids=request.region_ids, devices=request.devices,
                                 period_start=period, period_end=next_month - timedelta(days=1),
                                 is_complete_period=True, count=count, value_status="observed",
                                 share_raw=share, share_unit="unknown", share_fraction=None,
                                 normalization_status="unknown_unit", observed_at=observed,
                                 available_at=observed, row_locator="api:" + period.isoformat())
        observations.append(store.put_object(item))
    receipt = WordstatImportReceipt(kind="dynamics", query_profile_hash=profile_hash,
                                    snapshot_hash=snapshot_hash, raw_hash=raw_hash,
                                    mapping_hash=content_hash({"adapter": "wordstat-api-v2/1", "request": api_request_hash}),
                                    row_count=len(rows), accepted_count=len(rows), rejected_count=0,
                                    unselected_count=0,
                                    observation_hashes=tuple(observations), completed_at=observed)
    return store.put_object(receipt)
