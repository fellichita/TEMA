"""Paid Wordstat adapter: request contract, durable quota, and fail-closed data."""

from datetime import date, datetime, timezone
import json
from pathlib import Path

import httpx
import pytest

from app.pilot.multisource.contracts import QueryProfile, SearchObservation, SourceSnapshot, WordstatImportReceipt
from app.pilot.multisource.queries import build_manual_profile
from app.pilot.multisource.store import SignalStore
from app.pilot.multisource.wordstat_api import fetch_wordstat_dynamics
from app.pilot.service import PilotService
from app.runtime.credentials import CredentialStore
from app.runtime.jobs import Coordinator, TaskFailure
from app.sqlite_runtime import sqlite3


def _profile(store: SignalStore) -> str:
    value = build_manual_profile("ДНК память", "Хранение цифровых данных в молекулах",
                                 seed_terms=("ДНК память",), primary_phrase="ДНК память",
                                 confirmed_at=datetime(2025, 9, 1, tzinfo=timezone.utc))
    return store.put_object(value)


def _response(share: object = 0.01) -> bytes:
    return json.dumps({"results": [
        {"date": "2025-07-01T00:00:00Z", "count": "100", "share": share},
        {"date": "2025-08-01T00:00:00Z", "count": "120", "share": 0.02},
    ]}).encode()


def _fetch(store: SignalStore, profile_hash: str, coordinator: Coordinator,
           transport: httpx.BaseTransport, *, hour_cap: int = 3) -> str:
    return fetch_wordstat_dynamics(store, profile_hash, folder_id="b1g-folder", api_key="secret-test-key",
                                   from_date=date(2025, 7, 1), to_date=date(2025, 8, 31),
                                   hour_cap=hour_cap, daily_cap=20, coordinator=coordinator, transport=transport)


def _ledger(path: Path) -> tuple[list[tuple[str, int]], int]:
    with sqlite3.connect(path / "pilot.sqlite3") as connection:
        requests = connection.execute("SELECT state,charged_cost FROM pilot_budget_requests").fetchall()
        return [(row[0], row[1]) for row in requests], connection.execute(
            "SELECT COUNT(*) FROM pilot_budget_scopes WHERE scope_id LIKE 'wordstat/%'").fetchone()[0]


def test_get_dynamics_request_and_verified_json_receipt(tmp_path: Path) -> None:
    store = SignalStore(tmp_path)
    profile_hash = _profile(store)
    seen = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url == "https://searchapi.api.cloud.yandex.net/v2/wordstat/dynamics"
        assert request.headers["Authorization"] == "Api-Key secret-test-key"
        assert json.loads(request.content) == {
            "phrase": "ДНК память", "period": "PERIOD_MONTHLY",
            "fromDate": "2025-07-01T00:00:00Z", "toDate": "2025-08-31T00:00:00Z",
            "folderId": "b1g-folder",
        }
        return httpx.Response(200, content=_response())

    coordinator = Coordinator(tmp_path, lambda *_: {})
    try:
        digest = _fetch(store, profile_hash, coordinator, httpx.MockTransport(respond))
        receipt = store.get_object(digest, WordstatImportReceipt)
        snapshot = store.get_object(receipt.snapshot_hash, SourceSnapshot)
        observations = [store.get_object(item, SearchObservation) for item in receipt.observation_hashes]
        assert snapshot.adapter_version == "wordstat-api-v2/1" and snapshot.export_right == "local_only"
        assert store.verify_raw(receipt.raw_hash, "json").read_bytes() == _response()
        assert [item.count for item in observations] == [100, 120]
        assert all(item.share_unit == "unknown" and item.share_fraction is None
                   and item.normalization_status == "unknown_unit" for item in observations)
        assert len(seen) == 1 and _ledger(tmp_path)[0] == [("settled", 20_000)]
    finally:
        coordinator.close()


def test_documented_numeric_share_may_use_json_exponent(tmp_path: Path) -> None:
    store = SignalStore(tmp_path)
    profile_hash = _profile(store)
    coordinator = Coordinator(tmp_path, lambda *_: {})
    try:
        digest = _fetch(store, profile_hash, coordinator,
                        httpx.MockTransport(lambda _: httpx.Response(200, content=_response(1e-8))))
        receipt = store.get_object(digest, WordstatImportReceipt)
        first = store.get_object(receipt.observation_hashes[0], SearchObservation)
        assert first.share_raw == "0.00000001" and first.share_fraction is None
    finally:
        coordinator.close()


def test_api_json_receipt_replays_through_signal_workflow(tmp_path: Path) -> None:
    from uuid import uuid4
    from app.pilot.multisource.contracts import TechnologyConcept

    store = SignalStore(tmp_path)
    query_hash = _profile(store)
    query = store.get_object(query_hash, QueryProfile)
    concept_hash = store.put_object(TechnologyConcept(concept_id=uuid4(), label="ДНК память",
        definition=query.definition, identity_status="confirmed", confirmed_at=datetime.now(timezone.utc),
        provenance_hashes=(query_hash,)))
    coordinator = Coordinator(tmp_path, lambda *_: {})
    try:
        receipt_hash = _fetch(store, query_hash, coordinator,
                              httpx.MockTransport(lambda _: httpx.Response(200, content=_response())))
    finally:
        coordinator.close()
    service = PilotService(tmp_path, CredentialStore())
    try:
        run_id = service.start_signals(query_hash, (concept_hash,), wordstat_receipt_hash=receipt_hash)
        service.coordinator.wait()
        assert service.get(run_id)["state"] == "succeeded"
        assert service.signal_result(run_id)["profile"]["import_receipt_hashes"] == [receipt_hash]
    finally:
        service.close()


@pytest.mark.parametrize(("status", "expected_state", "cost"), [
    (401, "settled", 0), (403, "settled", 0), (429, "unknown", 20_000), (500, "unknown", 20_000),
])
def test_auth_and_provider_errors_have_explicit_budget_states(tmp_path: Path, status: int,
                                                              expected_state: str, cost: int) -> None:
    store = SignalStore(tmp_path)
    profile_hash = _profile(store)
    coordinator = Coordinator(tmp_path, lambda *_: {})
    try:
        with pytest.raises(TaskFailure):
            _fetch(store, profile_hash, coordinator,
                   httpx.MockTransport(lambda _: httpx.Response(status, content=b"secret-test-key")))
        assert _ledger(tmp_path)[0] == [(expected_state, cost)]
        assert not list(store.raw.glob("*.json"))
    finally:
        coordinator.close()


def test_timeout_after_dispatch_reserves_cost_and_cap_survives_restart(tmp_path: Path) -> None:
    store = SignalStore(tmp_path)
    profile_hash = _profile(store)
    def timeout(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("secret-test-key")

    coordinator = Coordinator(tmp_path, lambda *_: {})
    try:
        with pytest.raises(TaskFailure) as error:
            _fetch(store, profile_hash, coordinator, httpx.MockTransport(timeout), hour_cap=1)
        assert "secret-test-key" not in str(error.value)
        assert _ledger(tmp_path)[0] == [("unknown", 20_000)]
    finally:
        coordinator.close()
    reopened = Coordinator(tmp_path, lambda *_: {})
    try:
        with pytest.raises(TaskFailure, match="Лимит Wordstat"):
            _fetch(store, profile_hash, reopened,
                   httpx.MockTransport(lambda _: httpx.Response(200, content=_response())), hour_cap=1)
        assert len(_ledger(tmp_path)[0]) == 1
    finally:
        reopened.close()


@pytest.mark.parametrize("response", [
    b'{"results":[{"date":"2025-07-01T00:00:00Z","count":"1","share":0.1}]}',
    b'{"results":[{"date":"2025-07-01T00:00:00Z","count":"1","share":0.1},'
    b'{"date":"2025-08-01T00:00:00Z","count":"2","share":0.2}],"extra":1}',
    _response("nan"),
])
def test_malformed_or_incomplete_200_is_charged_but_never_published(tmp_path: Path, response: bytes) -> None:
    store = SignalStore(tmp_path)
    profile_hash = _profile(store)
    coordinator = Coordinator(tmp_path, lambda *_: {})
    try:
        with pytest.raises(TaskFailure, match="схеме"):
            _fetch(store, profile_hash, coordinator,
                   httpx.MockTransport(lambda _: httpx.Response(200, content=response)))
        assert _ledger(tmp_path)[0] == [("settled", 20_000)]
        assert len(list(store.objects.glob("*.json"))) == 1  # query only
    finally:
        coordinator.close()
