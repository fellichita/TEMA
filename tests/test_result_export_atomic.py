"""No-clobber publication is atomic across real CLI processes."""

import errno
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.ml import service
from app.ml.contracts import AnalysisInputError
from tests.mvp_fixture import snapshot
from tests.platform_support import require_symlinks

ROOT = Path(__file__).resolve().parents[1]


def empty_snapshot(identifier):
    data = snapshot()
    data["history"]["id"] = identifier
    for period, batch in zip(data["history"]["periods"], data["batches"], strict=True):
        period["job"].update(stored=0, scanned=0, total_available=0)
        batch.update(total=0, documents=[])
    return data


def test_two_real_cli_exports_publish_exactly_one_complete_result(tmp_path):
    # Only scheduling is controlled. Each process executes the actual CLI,
    # snapshot validation, analysis and export. Both pass the initial exists()
    # check before either creates its temporary output file.
    script = """
from pathlib import Path
import sys, time
from app.ml import service
from app.ml.__main__ import main
folder, marker = Path(sys.argv[1]), sys.argv[2]
create_temporary = service.tempfile.NamedTemporaryFile
def simultaneous_temporary(*args, **kwargs):
    (folder / ('ready-' + marker)).touch()
    deadline = time.monotonic() + 10
    while not all((folder / ('ready-' + name)).exists() for name in ('A', 'B')):
        if time.monotonic() > deadline:
            raise TimeoutError('Second exporter did not reach publication')
        time.sleep(0.01)
    return create_temporary(*args, **kwargs)
service.tempfile.NamedTemporaryFile = simultaneous_temporary
raise SystemExit(main(sys.argv[3:]))
"""
    target = tmp_path / "result.json"
    processes, originals = {}, {}
    for marker in ("A", "B"):
        source = tmp_path / (marker + ".json")
        originals[source] = json.dumps(empty_snapshot("history-" + marker)).encode()
        source.write_bytes(originals[source])
        processes[marker] = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), marker,
             "--snapshot", str(source), "--output", str(target)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    completed = {}
    try:
        for marker, process in processes.items():
            stdout, stderr = process.communicate(timeout=20)
            completed[marker] = (process.returncode, stdout, stderr)
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
    assert sorted(row[0] for row in completed.values()) == [0, 2], completed
    winner = next(marker for marker, row in completed.items() if row[0] == 0)
    loser = completed["B" if winner == "A" else "A"]
    assert loser[1] == "" and "Файл уже существует" in loser[2]
    assert all("Traceback" not in row[2] for row in completed.values())
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["provenance"]["history_id"] == "history-" + winner
    assert result["status"] == "insufficient_data"
    assert all(source.read_bytes() == original for source, original in originals.items())
    assert {path.name for path in tmp_path.iterdir()} == {"A.json", "B.json", "ready-A", "ready-B", "result.json"}


@pytest.mark.parametrize("target_kind", ["file", "symlink", "dangling_symlink"])
def test_no_clobber_preserves_existing_names_including_dangling_symlinks(tmp_path, target_kind):
    require_symlinks()
    target = tmp_path / "result.json"
    original = tmp_path / "original.json"
    if target_kind != "dangling_symlink":
        original.write_bytes(b"original bytes")
    if target_kind == "file":
        target.write_bytes(b"existing result")
    else:
        target.symlink_to(original)
    original_link = target.readlink() if target.is_symlink() else None
    before = {path.name for path in tmp_path.iterdir()}
    with pytest.raises(AnalysisInputError, match="уже существует"):
        service.export_result({"candidates": []}, target)
    if target_kind == "file":
        assert target.read_bytes() == b"existing result"
    else:
        # Windows may expose the target with an extended-path prefix. Preserve
        # the exact original link representation, including for dangling links.
        assert target.is_symlink() and target.readlink() == original_link
    if target_kind == "dangling_symlink":
        assert not original.exists()
    else:
        assert original.read_bytes() == b"original bytes"
    assert {path.name for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("alias", [False, True])
def test_protected_corpus_cannot_be_replaced_even_with_overwrite_or_symlink_alias(tmp_path, alias):
    require_symlinks()
    source = tmp_path / "corpus.json"
    source.write_bytes(b"saved input corpus")
    target = source
    if alias:
        target = tmp_path / "alias.json"
        target.symlink_to(source)
    with pytest.raises(AnalysisInputError, match="исходный корпус"):
        service.export_result({"candidates": []}, target, protected_paths=[source], overwrite=True)
    assert source.read_bytes() == b"saved input corpus"
    if alias:
        assert target.is_symlink()


def test_explicit_overwrite_still_atomically_replaces_the_result(tmp_path):
    target = tmp_path / "result.json"
    target.write_bytes(b"previous")
    result = {"candidates": [], "marker": "new result"}
    assert service.export_result(result, target, overwrite=True) == str(target)
    assert json.loads(target.read_text(encoding="utf-8")) == result
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("failure", ["fsync", "link", "replace", "encoding"])
def test_failed_write_or_publication_cleans_temporary_file_and_preserves_old_data(tmp_path, monkeypatch, failure):
    target = tmp_path / "result.json"
    overwrite = failure == "replace"
    if overwrite:
        target.write_bytes(b"previous result")

    def fail(*args, **kwargs):
        raise OSError(errno.EIO, "Simulated storage failure")

    result = {"candidates": []}
    if failure == "encoding":
        result["marker"] = "\ud800"
    else:
        monkeypatch.setattr(service.os, failure, fail)
    with pytest.raises((OSError, UnicodeError)):
        service.export_result(result, target, overwrite=overwrite)
    if overwrite:
        assert target.read_bytes() == b"previous result"
        assert list(tmp_path.iterdir()) == [target]
    else:
        assert list(tmp_path.iterdir()) == []


def test_lost_publication_race_has_a_readable_error_and_no_temporary_leak(tmp_path, monkeypatch):
    target = tmp_path / "result.json"

    def winner_arrives_before_link(source, destination):
        target.write_bytes(b"another successful writer")
        raise FileExistsError(errno.EEXIST, "File exists", str(destination))

    monkeypatch.setattr(service.os, "link", winner_arrives_before_link)
    with pytest.raises(AnalysisInputError, match="уже существует"):
        service.export_result({"candidates": []}, target)
    assert target.read_bytes() == b"another successful writer"
    assert list(tmp_path.iterdir()) == [target]
