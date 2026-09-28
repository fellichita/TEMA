"""Explicit watch choices survive a restart and cannot mutate signal findings."""

from pathlib import Path

import pytest

from app.pilot.multisource.store import SignalStore
from app.pilot.service import PilotService
from app.runtime.credentials import CredentialStore
from app.runtime.jobs import TaskFailure
from tests.test_multisource_service import _profile, _wordstat


def test_watch_and_unwatch_are_immutable_and_survive_restart(tmp_path: Path) -> None:
    store = SignalStore(tmp_path)
    query_hash, concept_hash, concept = _profile(store)
    receipt = _wordstat(tmp_path, store, query_hash)
    service = PilotService(tmp_path, CredentialStore())
    try:
        run = service.start_signals(query_hash, (concept_hash,), wordstat_receipt_hash=receipt)
        service.coordinator.wait()
        baseline = service.signal_result(run)["profile"]
        identifier = str(concept.concept_id)
        assert service.signal_watch_state(run, identifier)["watched"] is False
        first = service.set_signal_watch(run, identifier, True, "Проверить независимые работы")
        assert first["watched"] is True and first["note"] == "Проверить независимые работы"
        second = service.set_signal_watch(run, identifier, False)
        assert second["watched"] is False and second["record_hash"] != first["record_hash"]
        assert service.signal_result(run)["profile"] == baseline
        with pytest.raises(TaskFailure):
            service.set_signal_watch(run, identifier, True, " ")
        with pytest.raises(TaskFailure):
            service.signal_watch_state(run, "00000000-0000-4000-8000-000000000000")
    finally:
        service.close()
    reopened = PilotService(tmp_path, CredentialStore())
    try:
        assert reopened.signal_watch_state(run, str(concept.concept_id)) == second
        assert len(store.watch_records()) == 2
    finally:
        reopened.close()


def test_corrupt_watch_alias_is_rejected(tmp_path: Path) -> None:
    store = SignalStore(tmp_path)
    query_hash, concept_hash, concept = _profile(store)
    receipt = _wordstat(tmp_path, store, query_hash)
    service = PilotService(tmp_path, CredentialStore())
    try:
        run = service.start_signals(query_hash, (concept_hash,), wordstat_receipt_hash=receipt)
        service.coordinator.wait()
        saved = service.set_signal_watch(run, str(concept.concept_id), True)
        path = store.watch / (saved["record_hash"] + ".json")
        path.write_bytes(b"corrupt")
        with pytest.raises(TaskFailure):
            service.signal_watch_state(run, str(concept.concept_id))
    finally:
        service.close()
