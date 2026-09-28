"""Saved reads retain integrity without monopolizing cancellation or shutdown."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
import json
from threading import Event, get_ident

import pytest

from app.pilot.archive import DocumentArchive
from app.pilot.contracts import content_hash
from app.pilot.service import PilotService
from app.pilot.supplemental import SupplementalImport, SupplementalLibrary
from app.runtime.jobs import Coordinator, TaskCancelled, TaskFailure
from app.runtime.credentials import CredentialStore
from tests.test_pilot_evidence import document


@pytest.fixture
def runtime(tmp_path):
    coordinator = Coordinator(tmp_path, lambda *_: {"text": "x" * 200_000})
    run_id = coordinator.submit({})
    coordinator.wait(timeout=2)
    try:
        yield coordinator, run_id
    finally:
        coordinator.close()


@pytest.mark.parametrize("method", ["result", "checkpoint_value"])
def test_saved_file_read_does_not_hold_cancel_gate(runtime, monkeypatch, method):
    coordinator, saved_id = runtime
    active_started, read_started, release = Event(), Event(), Event()

    def process(context, _payload):
        active_started.set()
        assert context.cancel_event.wait(5)
        context.check_cancelled()

    monkeypatch.setattr(coordinator, "_processor", process)
    active_id = coordinator.submit({})
    assert active_started.wait(2)
    original = coordinator._read_checkpoint

    def paused(*args, **kwargs):
        read_started.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(coordinator, "_read_checkpoint", paused)
    args = (saved_id,) if method == "result" else (saved_id, "result")
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = executor.submit(getattr(coordinator, method), *args)
        assert read_started.wait(2)
        try:
            assert executor.submit(coordinator.cancel, active_id).result(timeout=0.5)
        finally:
            release.set()
            assert pending.result(timeout=2)["text"] == "x" * 200_000
            coordinator.cancel(active_id)
            coordinator.wait(timeout=2)


@pytest.mark.parametrize("method", ["result", "checkpoint_value"])
def test_checkpoint_cancel_stops_after_one_bounded_block(runtime, monkeypatch, method):
    from app.pilot import reports

    coordinator, run_id = runtime
    requested, sizes = Event(), []
    original = reports.open_local_regular

    @contextmanager
    def cancel_after_read(path):
        with original(path) as source:
            class Reader:
                def read(self, size):
                    sizes.append(size)
                    data = source.read(size)
                    requested.set()
                    return data

            yield Reader()

    monkeypatch.setattr(reports, "open_local_regular", cancel_after_read)
    args = (run_id,) if method == "result" else (run_id, "result")
    with pytest.raises(TaskCancelled):
        getattr(coordinator, method)(*args, cancel=requested)
    assert sizes == [64 * 1024]


def test_close_keeps_profile_owned_until_reader_exits_and_discards_late_payload(runtime, monkeypatch):
    coordinator, run_id = runtime
    entered, release = Event(), Event()
    original = coordinator._read_checkpoint

    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(coordinator, "_read_checkpoint", paused)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(coordinator.result, run_id)
        assert entered.wait(2)
        try:
            with pytest.raises(TimeoutError):
                coordinator.close(timeout=0.01)
            with pytest.raises(Exception, match="используется|уже"):
                Coordinator(coordinator.data_dir, lambda *_: {})
        finally:
            release.set()
            with pytest.raises(TaskCancelled):
                pending.result(timeout=2)
    coordinator.close()
    with pytest.raises(TaskCancelled):
        coordinator.result(run_id)


def test_cancel_after_json_parse_cannot_publish_a_payload(runtime, monkeypatch):
    coordinator, run_id = runtime
    requested = Event()
    loads = json.loads

    def cancel_after_parse(data, *args, **kwargs):
        value = loads(data, *args, **kwargs)
        requested.set()
        return value

    monkeypatch.setattr("app.runtime.jobs.json.loads", cancel_after_parse)
    with pytest.raises(TaskCancelled):
        coordinator.result(run_id, cancel=requested)


@pytest.mark.parametrize("method", ["result", "checkpoint_value"])
def test_checkpoint_integrity_and_closed_state_remain_strict(runtime, method):
    coordinator, run_id = runtime
    args = (run_id,) if method == "result" else (run_id, "result")
    path = next((coordinator.data_dir / "checkpoints").glob("*.json"))
    data = path.read_bytes()
    path.write_bytes(data.replace(b"x", b"y", 1))
    with pytest.raises(TaskFailure, match="повреждён"):
        getattr(coordinator, method)(*args)
    path.write_bytes(data)
    assert getattr(coordinator, method)(*args)["text"] == "x" * 200_000
    coordinator.close()
    with pytest.raises(TaskCancelled):
        getattr(coordinator, method)(*args)


@pytest.mark.parametrize("method", ["get", "result", "documents", "supplemental_list"])
def test_service_saved_reads_respect_view_cancellation(tmp_path, monkeypatch, method):
    credentials = CredentialStore()
    monkeypatch.setattr(credentials, "_load_backend", lambda: None)
    service = PilotService(tmp_path, credentials)
    monkeypatch.setattr(service.coordinator, "_processor", lambda *_: {})
    run_id = service.coordinator.submit({})
    service.coordinator.wait(timeout=2)
    service.view_cancel.set()
    try:
        args = () if method == "supplemental_list" else (run_id,)
        with pytest.raises(TaskCancelled):
            getattr(service, method)(*args)
    finally:
        service.close()
        credentials.close()


def test_service_propagates_mid_read_cancellation_to_local_result(tmp_path, monkeypatch):
    credentials = CredentialStore()
    monkeypatch.setattr(credentials, "_load_backend", lambda: None)
    service = PilotService(tmp_path, credentials)
    monkeypatch.setattr(service.coordinator, "_processor", lambda *_: {})
    run_id = service.coordinator.submit({})
    service.coordinator.wait(timeout=2)
    original = service.coordinator._read_checkpoint

    def cancel_after_read(*args, **kwargs):
        assert kwargs["cancel"] is service.view_cancel
        value = original(*args, **kwargs)
        service.view_cancel.set()
        return value

    monkeypatch.setattr(service.coordinator, "_read_checkpoint", cancel_after_read)
    try:
        with pytest.raises(TaskCancelled):
            service.result(run_id)
    finally:
        service.close()
        credentials.close()


def test_close_shares_one_deadline_between_reserved_io_worker_and_sqlite(runtime, monkeypatch):
    coordinator, _run_id = runtime
    clock, observed = [100.0], []
    monkeypatch.setattr("app.runtime.jobs.monotonic", lambda: clock[0], raising=False)

    class Preparation:
        def wait(self, timeout):
            observed.append(("preparation", timeout))
            clock[0] += 4
            return True

    coordinator._idle_operation = (get_ident() + 1, Preparation())
    wait, submit = coordinator.wait, coordinator._executor.submit

    def worker_wait(timeout):
        observed.append(("worker", timeout))
        clock[0] += 3
        return wait(timeout)

    def closing_submit(*args, **kwargs):
        future = submit(*args, **kwargs)

        class Closing:
            def result(self, timeout):
                observed.append(("sqlite", timeout))
                return future.result(timeout)

        return Closing()

    monkeypatch.setattr(coordinator, "wait", worker_wait)
    monkeypatch.setattr(coordinator._executor, "submit", closing_submit)
    coordinator.close(timeout=10)
    assert observed == [("preparation", 10), ("worker", 6), ("sqlite", 3)]


@pytest.fixture
def supplemental(tmp_path):
    archive = DocumentArchive(tmp_path / "revisions")
    reference = archive.put(document(0))
    library = SupplementalLibrary(tmp_path, archive)
    library.directory.mkdir()
    for index in range(1000):
        value = SupplementalImport(kind="arxiv", created_at="2026-09-12T00:00:00Z",
                                   documents=(reference,), limitations=(f"fixture {index}",)).model_dump(mode="json")
        (library.directory / (content_hash(value) + ".json")).write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return library


def test_supplemental_page_reads_only_selected_metadata_and_revisions(supplemental, monkeypatch):
    from app.pilot import supplemental as module

    loaded, documents = [], []
    original, get = module.read_artifact, supplemental.archive.get

    def read(directory, identifier):
        loaded.append(identifier)
        return original(directory, identifier)

    def read_document(identifier):
        documents.append(identifier)
        return get(identifier)

    monkeypatch.setattr(module, "read_artifact", read)
    monkeypatch.setattr(supplemental.archive, "get", read_document)
    expected = sorted(path.stem for path in supplemental.directory.glob("*.json"))
    page = supplemental.list_page(offset=50, limit=50)
    assert page["total"] == 1000 and page["offset"] == page["limit"] == 50
    assert [row["id"] for row in page["items"]] == expected[50:100]
    assert loaded == expected[50:100] and len(documents) == 50
    assert supplemental.list_page(offset=1000)["items"] == []
    assert len(loaded) == 50


def test_supplemental_cancel_during_page_stops_before_next_document(supplemental, monkeypatch):
    requested, seen = Event(), []
    get = supplemental.archive.get

    def read_document(identifier):
        seen.append(identifier)
        requested.set()
        return get(identifier)

    monkeypatch.setattr(supplemental.archive, "get", read_document)
    with pytest.raises(TaskCancelled):
        supplemental.list_page(cancel=requested)
    assert len(seen) == 1


def test_supplemental_page_does_not_trust_tampered_selected_metadata(supplemental):
    paths = sorted(supplemental.directory.glob("*.json"))
    paths[0].write_text("{}")
    assert supplemental.list_page(offset=1, limit=1)["total"] == 1000
    with pytest.raises(TaskFailure, match="повреждён"):
        supplemental.list_page(limit=1)


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": 51}, {"offset": -1}, {"offset": True}])
def test_supplemental_rejects_invalid_page_before_io(tmp_path, kwargs):
    library = SupplementalLibrary(tmp_path, DocumentArchive(tmp_path / "revisions"))
    with pytest.raises(ValueError):
        library.list_page(**kwargs)
