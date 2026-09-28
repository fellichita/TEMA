"""Bounded imported-result reads retain verification and filesystem invalidation."""

import json
import os
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from time import perf_counter
from types import SimpleNamespace

import pytest

from app.pilot import library as module
from app.pilot.archive import DocumentArchive
from app.pilot.contracts import AnalysisResult
from app.pilot.library import ResultLibrary
from app.runtime.backup import ArchiveError
from app.runtime.jobs import TaskCancelled, TaskFailure
from tests.test_pilot_evidence import NOW, document, query_plan, snapshot
from tests.test_pilot_export import make_result
from tests.platform_support import require_symlinks


@pytest.fixture
def saved(tmp_path):
    result, archive, artifacts = make_result(tmp_path, historical=True)
    library = ResultLibrary(tmp_path, archive)
    run_id = library.save_result(result, artifacts)["id"]
    return library, run_id, result


def test_list_summary_reuses_verified_metadata_and_invalidates_same_size_timestamp_restore(saved, monkeypatch):
    library, run_id, _ = saved
    original_metadata = module._identity_metadata
    monkeypatch.setattr(module, "_identity_metadata", lambda info: (*original_metadata(info)[:5], 12345))
    original_read = module.read_artifact
    reads = []

    def tracked(directory, digest):
        reads.append(digest)
        return original_read(directory, digest)

    monkeypatch.setattr(module, "read_artifact", tracked)
    first = library.list_rows()
    assert library.list_rows() == first and len(reads) == 1
    first[0]["input_json"] = '{"query":"caller mutation"}'
    assert library.list_rows()[0]["input_json"] != first[0]["input_json"]
    path = library.directory / (run_id.removeprefix("import-") + ".json")
    previous = path.stat()
    metadata_before = module._identity_metadata(previous)
    content = path.read_bytes()
    path.write_bytes(content.replace(b"result-one", b"result-bad"))
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert path.stat().st_size == previous.st_size
    assert module._identity_metadata(path.stat()) == metadata_before
    assert library.list_rows()[0]["state"] == "failed"
    assert len(reads) == 2


@pytest.mark.parametrize("target", ["artifact", "unselected_revision"])
def test_document_cache_rejects_tampering_outside_requested_page(saved, target):
    library, run_id, result = saved
    assert len(library.documents(run_id, limit=1)["items"]) == 1
    if target == "artifact":
        path = library.directory / (run_id.removeprefix("import-") + ".json")
    else:
        history = next(item for item in result.snapshots if item.purpose == "history")
        path = library.archive.path(history.documents[0].revision_id)
    previous = path.stat()
    content = path.read_bytes()
    path.write_bytes(content.replace(b"2025", b"2024", 1) if b"2025" in content else content[:-1] + b" ")
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert path.stat().st_size == previous.st_size
    with pytest.raises((TaskFailure, ArchiveError)):
        library.documents(run_id, offset=1, limit=1)


@pytest.mark.parametrize("target", ["artifact", "unselected_revision"])
def test_cached_view_rejects_changed_bytes_when_all_file_metadata_collides(saved, monkeypatch, target):
    library, run_id, result = saved
    original_metadata = module._identity_metadata
    monkeypatch.setattr(module, "_identity_metadata", lambda info: (*original_metadata(info)[:5], 12345))
    library.documents(run_id, limit=1)
    if target == "artifact":
        path = library.directory / (run_id.removeprefix("import-") + ".json")
    else:
        history = next(item for item in result.snapshots if item.purpose == "history")
        path = library.archive.path(history.documents[0].revision_id)
    before = module._file_identity(path)
    previous = path.stat()
    metadata_before = module._identity_metadata(previous)
    data = path.read_bytes()
    path.write_bytes(data[:-1] + b" ")
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    after = module._file_identity(path)
    assert module._identity_metadata(path.stat()) == metadata_before
    assert before[:-1] == after[:-1]
    assert before[-1] != after[-1]
    with pytest.raises((TaskFailure, ArchiveError)):
        library.documents(run_id, offset=1, limit=1)


def test_content_hash_is_chunk_bounded_and_observes_mid_read_cancel(tmp_path, monkeypatch):
    path = tmp_path / "sample.json"
    path.write_bytes(b"x" * 100_000)
    cancel = Event()
    original_open = module.open_local_regular
    reads = []

    class Reader:
        def __init__(self, handle):
            self.handle = handle

        def fileno(self):
            return self.handle.fileno()

        def read(self, size):
            reads.append(size)
            cancel.set()
            return self.handle.read(size)

    @contextmanager
    def opened(target):
        with original_open(target) as handle:
            yield Reader(handle)

    monkeypatch.setattr(module, "open_local_regular", opened)
    with pytest.raises(TaskCancelled):
        module._file_identity(path, cancel)
    assert reads == [64 * 1024]


def test_content_identity_detects_same_size_write_with_restored_modification_time(tmp_path):
    path = tmp_path / "native-change-time.json"
    path.write_bytes(b'{"value":1}')
    original = path.stat()
    before = module._file_identity(path)
    path.write_bytes(b'{"value":2}')
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    after = module._file_identity(path)
    assert before[:-1] == after[:-1]
    assert before[-1] != after[-1]


def _with_change_time(info, change_time):
    return SimpleNamespace(**{
        name: change_time if name == "st_ctime_ns" else getattr(info, name)
        for name in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    })


def test_cached_result_accepts_distinct_path_creation_time_and_descriptor_change_time(saved, monkeypatch):
    library, run_id, result = saved
    original_fstat = module.os.fstat

    def descriptor_stat(descriptor):
        info = original_fstat(descriptor)
        # CPython 3.13 Windows path stat exposes creation time as ctime, while
        # fstat exposes FileBasicInfo.ChangeTime. Both describe the same file.
        return _with_change_time(info, info.st_ctime_ns + 1_000_000)

    monkeypatch.setattr(module.os, "fstat", descriptor_stat)
    assert library.read(run_id)["result"] == result.model_dump(mode="json")
    assert library.documents(run_id, limit=1)["items"]


def test_content_identity_rejects_change_time_mutation_within_descriptor_api(tmp_path, monkeypatch):
    path = tmp_path / "descriptor-change.json"
    path.write_bytes(b'{"value":1}')
    original_fstat = module.os.fstat
    calls = 0

    def descriptor_stat(descriptor):
        nonlocal calls
        calls += 1
        info = original_fstat(descriptor)
        # open_local_regular checks the descriptor once before identity captures
        # its baseline. A later change must fail even when path ctime is stable.
        return _with_change_time(info, info.st_ctime_ns + (1_000_000 if calls >= 3 else 0))

    monkeypatch.setattr(module.os, "fstat", descriptor_stat)
    with pytest.raises(TaskFailure):
        module._file_identity(path)
    assert calls == 3


def test_content_identity_rejects_change_time_mutation_within_path_api(tmp_path, monkeypatch):
    path = tmp_path / "path-change.json"
    path.write_bytes(b'{"value":1}')
    original_stat = Path.stat
    calls = 0

    def path_stat(target, *args, **kwargs):
        nonlocal calls
        info = original_stat(target, *args, **kwargs)
        if target == path:
            calls += 1
            return _with_change_time(info, info.st_ctime_ns + (1_000_000 if calls >= 2 else 0))
        return info

    monkeypatch.setattr(Path, "stat", path_stat)
    with pytest.raises(TaskFailure):
        module._file_identity(path)
    assert calls == 2


def test_content_identity_rejects_oversized_file_before_reading(tmp_path, monkeypatch):
    path = tmp_path / "oversized.json"
    with path.open("wb") as handle:
        handle.truncate(module._MAX_ARTIFACT_BYTES + 1)
    monkeypatch.setattr(module, "open_local_regular", lambda _: pytest.fail("Oversized file must not be read"))
    with pytest.raises(TaskFailure):
        module._file_identity(path)


def test_content_identity_enforces_size_limit_during_growth(tmp_path, monkeypatch):
    path = tmp_path / "growing.json"
    path.write_bytes(b"x" * 100_000)
    monkeypatch.setattr(module, "_MAX_ARTIFACT_BYTES", 100_000)
    original_open = module.open_local_regular
    reads = []

    class Reader:
        def __init__(self, handle):
            self.handle = handle

        def fileno(self):
            return self.handle.fileno()

        def read(self, size):
            reads.append(size)
            data = self.handle.read(size)
            if len(reads) == 1:
                with path.open("ab") as writer:
                    writer.write(b"y" * 100_000)
            return data

    @contextmanager
    def opened(target):
        with original_open(target) as handle:
            yield Reader(handle)

    monkeypatch.setattr(module, "open_local_regular", opened)
    with pytest.raises(TaskFailure):
        module._file_identity(path)
    assert reads == [64 * 1024, 64 * 1024]


def test_content_identity_rejects_nonregular_file_before_opening(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "open_local_regular", lambda _: pytest.fail("Directory must not be read"))
    with pytest.raises(TaskFailure):
        module._file_identity(tmp_path)


def test_content_identity_rejects_path_replacement_during_same_descriptor_read(tmp_path, monkeypatch):
    path = tmp_path / "sample.json"
    path.write_bytes(b'{"value":1}')
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b'{"value":2}')
    original = path.stat()
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    original_open = module.open_local_regular

    @contextmanager
    def replaced(target):
        with original_open(target) as handle:
            yield handle
        replacement.replace(path)

    monkeypatch.setattr(module, "open_local_regular", replaced)
    with pytest.raises(TaskFailure):
        module._file_identity(path)


def test_content_identity_rejects_symlink_swap_before_open(tmp_path, monkeypatch):
    require_symlinks()
    path = tmp_path / "sample.json"
    path.write_bytes(b'{"value":1}')
    external = tmp_path / "external.json"
    external.write_bytes(path.read_bytes())
    original_open = module.open_local_regular

    @contextmanager
    def replaced(target):
        path.unlink()
        path.symlink_to(external)
        with original_open(target) as handle:
            yield handle

    monkeypatch.setattr(module, "open_local_regular", replaced)
    with pytest.raises(TaskFailure):
        module._file_identity(path)
    assert external.read_bytes() == b'{"value":1}'


def test_summary_hashes_only_selected_catalogue_page(saved, monkeypatch):
    library, run_id, _ = saved
    artifact = library.directory / (run_id.removeprefix("import-") + ".json")
    content = artifact.read_bytes()
    # Incomplete entries remain visible as failed rows; unselected entries must
    # not be opened or hashed just to order a catalogue page.
    paths = [library.directory / (f"{index:064x}" + ".json") for index in range(1000)]
    for index, path in enumerate(paths):
        path.write_bytes(content)
        os.utime(path, ns=(1_000_000_000 + index, 1_000_000_000 + index))
    artifact.unlink()
    selected = set(paths[900:950])
    opened = []
    original_open = module.open_local_regular

    @contextmanager
    def tracked(path):
        opened.append(Path(path))
        with original_open(path) as handle:
            yield handle

    monkeypatch.setattr(module, "open_local_regular", tracked)
    rows = library.list_rows(offset=50, limit=50)
    assert len(rows) == 50
    assert [row["id"] for row in rows] == ["import-" + path.stem for path in reversed(paths[900:950])]
    assert all(row["state"] == "failed" for row in rows)
    assert set(opened) == selected
    assert library.list_rows(offset=1000) == []


def test_result_callers_cannot_mutate_cached_view(saved):
    library, run_id, result = saved
    returned = library.read(run_id)
    returned["result"]["query_plan"]["original_query"] = "caller mutation"
    returned["result"]["snapshots"][0]["documents"].clear()
    assert library.read(run_id)["result"] == result.model_dump(mode="json")
    assert library.documents(run_id)["total"] == len(result.snapshots[0].documents)


def test_changed_dependency_during_initial_verification_is_not_cached(saved, monkeypatch):
    library, run_id, result = saved
    verify = module.verify_result
    path = library.archive.path(result.snapshots[0].documents[0].revision_id)

    def mutate_after_verification(*args, **kwargs):
        answer = verify(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b"\n")
        return answer

    monkeypatch.setattr(module, "verify_result", mutate_after_verification)
    with pytest.raises(TaskFailure, match="изменились"):
        library.read(run_id)
    with pytest.raises((TaskFailure, ArchiveError)):
        library.documents(run_id)


def test_cached_result_still_observes_cancellation(saved, monkeypatch):
    library, run_id, _ = saved
    library.read(run_id)
    cancel = Event()
    cancel.set()
    monkeypatch.setattr(module, "_file_identity", lambda _: pytest.fail("Cancelled read must not visit files"))
    with pytest.raises(TaskCancelled):
        library.documents(run_id, cancel=cancel)


def test_only_one_full_result_view_is_retained_and_clear_releases_it(saved, monkeypatch):
    library, first, result = saved
    # Existing assessments are kept intact; changing a result ID is not scientific recomputation.
    payload = library.read(first)
    payload["result"]["result_id"] = "other-result"
    from app.pilot.library import publish_catalogue_artifact

    second = "import-" + publish_catalogue_artifact(library.directory, payload)
    verify = module.verify_result
    replays = []

    def tracked(*args, **kwargs):
        replays.append(args[0].result_id)
        return verify(*args, **kwargs)

    monkeypatch.setattr(module, "verify_result", tracked)
    library.read(first)
    library.read(second)
    library.read(first)
    assert replays == ["other-result", result.result_id]
    library.clear_cache()
    library.read(first)
    assert replays == ["other-result", result.result_id, result.result_id]


def test_warm_document_paging_reads_only_page_revisions_without_replay(tmp_path, monkeypatch):
    archive = DocumentArchive(tmp_path / "revisions")
    corpus = snapshot(tuple(document(100_000 + index, abstract="Synthetic public abstract. " * 200)
                            for index in range(128)), archive)
    result = AnalysisResult(result_id="bounded-fixture", run_id="run-fixture", query_plan=query_plan(),
        created_at=NOW, quality="insufficient_data", snapshots=(corpus,), cards=(),
        limitations=("Synthetic workload without scientific claims",))
    library = ResultLibrary(tmp_path, archive)
    run_id = library.save_result(result, ())["id"]
    original_read, original_verify, original_get = module.read_artifact, module.verify_result, archive.get
    counts = dict(artifacts=0, replays=0, revisions=0)

    def read(*args):
        counts["artifacts"] += 1
        return original_read(*args)

    def verify(*args, **kwargs):
        counts["replays"] += 1
        return original_verify(*args, **kwargs)

    def get(*args):
        counts["revisions"] += 1
        return original_get(*args)

    monkeypatch.setattr(module, "read_artifact", read)
    monkeypatch.setattr(module, "verify_result", verify)
    monkeypatch.setattr(archive, "get", get)
    start = perf_counter()
    first = library.documents(run_id, limit=10)
    cold_seconds = perf_counter() - start
    cold = dict(counts)
    start = perf_counter()
    second = library.documents(run_id, offset=10, limit=10)
    warm_seconds = perf_counter() - start
    warm = {key: counts[key] - cold[key] for key in counts}
    assert first["total"] == second["total"] == 128
    assert [row["source_id"] for row in second["items"]] == [f"W{100_010 + index}" for index in range(10)]
    assert cold["artifacts"] == cold["replays"] == 1
    assert cold["revisions"] >= 128
    assert warm == dict(artifacts=0, replays=0, revisions=10)
    assert library.documents(run_id, offset=200)["items"] == []
    print(json.dumps(dict(cold=cold, warm=warm, cold_seconds=cold_seconds, warm_seconds=warm_seconds)))
