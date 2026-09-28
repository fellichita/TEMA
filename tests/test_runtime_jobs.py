from concurrent.futures import TimeoutError
import json
import os
import stat
from threading import Event, Thread

import pytest

from app.runtime.jobs import Coordinator, TaskFailure
from app.sqlite_runtime import sqlite3
from tests.platform_support import require_symlinks


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_analysis_database_alias_cannot_modify_an_external_file(tmp_path, alias):
    data_dir = tmp_path / "profile"
    data_dir.mkdir()
    outside = tmp_path / "outside.db"
    outside.write_bytes(b"")
    database = data_dir / "pilot.sqlite3"
    if alias == "symlink":
        require_symlinks()
        database.symlink_to(outside)
    else:
        try:
            os.link(outside, database)
        except OSError:
            pytest.skip("Hard links are unavailable on this filesystem")

    with pytest.raises(TaskFailure, match="обычным файлом"):
        Coordinator(data_dir, lambda *_: {})
    assert outside.read_bytes() == b""


def test_checkpoint_directory_symlink_cannot_redirect_write(tmp_path):
    require_symlinks()
    data_dir = tmp_path / "profile"
    data_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (data_dir / "checkpoints").symlink_to(outside, target_is_directory=True)
    runtime = Coordinator(data_dir, lambda _context, payload: payload)
    try:
        run_id = runtime.submit({"answer": 1})
        runtime.wait()
        assert runtime.get(run_id)["state"] == "failed"
        assert list(outside.iterdir()) == []
    finally:
        runtime.close()


@pytest.mark.skipif(os.name == "nt", reason="Windows uses profile ACLs rather than POSIX file modes")
def test_analysis_database_and_checkpoints_are_private_on_disk(tmp_path):
    data_dir = tmp_path / "profile"
    data_dir.mkdir()
    data_dir.chmod(0o755)

    def processor(context, payload):
        context.checkpoint("plan", payload)
        return payload

    runtime = Coordinator(data_dir, processor)
    try:
        run_id = runtime.submit({"value": 1})
        runtime.wait()
        assert runtime.get(run_id)["state"] == "succeeded"
        checkpoint = next((data_dir / "checkpoints").glob("*.json"))
        assert stat.S_IMODE(data_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(runtime.path.stat().st_mode) == 0o600
        assert stat.S_IMODE(checkpoint.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(checkpoint.stat().st_mode) == 0o600
    finally:
        runtime.close()


def test_review_lookup_uses_exact_pair_latest_order_and_index_beyond_recent_history(tmp_path):
    runtime = Coordinator(tmp_path, lambda _context, payload: payload)

    def seed(connection):
        rows = [(f"recent-{index:04}", {"operation": "antecedents", "source_run_id": "different", "candidate_id": "candidate"},
                 "2099-01-01") for index in range(240)]
        rows.extend((identifier, payload, date) for identifier, payload, date in (
            ("review-a", {"operation": "antecedents", "source_run_id": "source", "candidate_id": "candidate"}, "2025-01-01"),
            ("review-z", {"operation": "antecedents", "source_run_id": "source", "candidate_id": "candidate"}, "2025-01-01"),
            ("other-candidate", {"operation": "antecedents", "source_run_id": "source", "candidate_id": "different"}, "2099-01-01"),
            ("not-review", {"source_run_id": "source", "candidate_id": "candidate"}, "2099-01-01"),
        ))
        connection.executemany("INSERT INTO analysis_runs(id,attempt,state,input_json,created_at,updated_at) VALUES (?,1,'succeeded',?,?,?)",
                               [(identifier, json.dumps({"payload": payload}), date, date) for identifier, payload, date in rows])

    try:
        runtime._executor.submit(seed, runtime._connection).result(timeout=5)
        assert runtime.find_antecedents_run("source", "candidate")["id"] == "review-z"
        assert runtime.find_antecedents_run("source", "missing") is None
        assert "review-z" not in {row["id"] for row in runtime.list_runs()}
        connection = sqlite3.connect(tmp_path / "pilot.sqlite3")
        try:
            plan = connection.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM analysis_runs WHERE json_extract(input_json, '$.payload.operation')='antecedents' "
                "AND json_extract(input_json, '$.payload.source_run_id')=? "
                "AND json_extract(input_json, '$.payload.candidate_id')=? ORDER BY created_at DESC,id DESC LIMIT 1",
                ("source", "candidate")).fetchall()
            assert any("USING INDEX ix_analysis_antecedents_lookup" in row[3] for row in plan)
            assert not any("TEMP B-TREE" in row[3] or "SCAN " in row[3] for row in plan)
        finally:
            connection.close()
    finally:
        runtime.close()


def test_analyses_pagination_filters_review_jobs_before_limit(tmp_path):
    runtime = Coordinator(tmp_path, lambda _context, payload: payload)
    try:
        analyses = []
        for index in range(6):
            analyses.append(runtime.submit({"query": str(index)}))
            runtime.wait()
            runtime.submit({"operation": "antecedents", "source_run_id": analyses[-1]})
            runtime.wait()
        assert [row["id"] for row in runtime.list_runs(limit=2, analyses_only=True)] == analyses[-1:-3:-1]
        assert [row["id"] for row in runtime.list_runs(limit=2, offset=2, analyses_only=True)] == analyses[-3:-5:-1]
        assert len(runtime.list_runs()) == 12
    finally:
        runtime.close()


def test_saved_result_opens_offline_after_restart(tmp_path):
    def processor(context, payload):
        context.progress("plan", "Понимаем запрос", 1, 1)
        context.checkpoint("plan", payload)
        return {"query": payload["query"], "items": []}

    runtime = Coordinator(tmp_path, processor)
    payload = {"query": "Исследование материалов"}
    run = runtime.submit(payload)
    payload["query"] = "caller mutation"
    runtime.wait()
    assert runtime.get(run)["state"] == "succeeded"
    runtime.close()
    reopened = Coordinator(tmp_path, lambda *_: pytest.fail("Offline reading must not compute"))
    try:
        assert reopened.result(run) == {"query": "Исследование материалов", "items": []}
        assert len(reopened.list_runs()) == 1
    finally:
        reopened.close()


def test_cancel_fences_late_success_and_resumes_verified_checkpoint(tmp_path):
    entered, proceed = Event(), Event()

    def processor(context, payload):
        prior = context.load_checkpoint("discovery")
        if prior:
            return {"resumed": prior}
        context.checkpoint("discovery", {"documents": ["doi:10.1/example"]})
        entered.set()
        assert proceed.wait(5)
        return {"late": True}

    runtime = Coordinator(tmp_path, processor)
    try:
        run = runtime.submit({"query": "test"})
        assert entered.wait(5)
        with pytest.raises(TaskFailure, match="Дождитесь"):
            runtime.submit({})
        assert runtime.cancel(run)
        proceed.set()
        runtime.wait()
        assert runtime.get(run)["state"] == "cancelled"
        with pytest.raises(TaskFailure, match="отсутствует"):
            runtime.result(run)
        runtime.resume(run)
        runtime.wait()
        assert runtime.get(run)["attempt"] == 2
        assert runtime.result(run) == {"resumed": {"documents": ["doi:10.1/example"]}}
    finally:
        proceed.set()
        runtime.close()


def test_interrupted_runs_require_explicit_resume(tmp_path):
    runtime = Coordinator(tmp_path, lambda *_: {})
    runtime.close()
    connection = sqlite3.connect(tmp_path / "pilot.sqlite3")
    try:
        connection.execute("INSERT INTO analysis_runs (id,attempt,state,input_json,created_at,updated_at) "
                           "VALUES ('interrupted',1,'running','{}','2026-01-01','2026-01-01')")
        connection.commit()
    finally:
        connection.close()
    calls = []
    runtime = Coordinator(tmp_path, lambda *_: calls.append(True) or {})
    try:
        assert runtime.get("interrupted")["state"] == "interrupted"
        assert calls == []
        runtime.resume("interrupted")
        runtime.wait()
        assert calls == [True]
    finally:
        runtime.close()


def test_corrupt_checkpoint_never_returns_or_recomputes_silently(tmp_path):
    runtime = Coordinator(tmp_path, lambda *_: {"answer": "real saved result"})
    try:
        run = runtime.submit({})
        runtime.wait()
        next((tmp_path / "checkpoints").glob("*.json")).write_text('{"answer":"tampered"}')
        with pytest.raises(TaskFailure, match="повреждён"):
            runtime.result(run)
    finally:
        runtime.close()


def test_unexpected_error_is_safe_and_another_run_still_works(tmp_path):
    def processor(context, payload):
        if payload.get("fail"):
            raise ValueError("Bearer PRIVATE-KEY IN PROVIDER BODY")
        return {"ok": True}

    runtime = Coordinator(tmp_path, processor)
    try:
        run = runtime.submit({"fail": True})
        runtime.wait()
        assert runtime.get(run)["state"] == "failed"
        assert "PRIVATE" not in runtime.get(run)["error"]
        second = runtime.submit({})
        runtime.wait()
        assert runtime.result(second) == {"ok": True}
    finally:
        runtime.close()


def test_close_timeout_retains_storage_lock_until_worker_exits(tmp_path):
    entered, proceed = Event(), Event()

    def processor(*_):
        entered.set()
        assert proceed.wait(5)
        return {}

    runtime = Coordinator(tmp_path, processor)
    runtime.submit({})
    assert entered.wait(5)
    try:
        with pytest.raises(TimeoutError):
            runtime.close(timeout=0.01)
        with pytest.raises(Exception, match="используется|уже"):
            Coordinator(tmp_path, lambda *_: {})
    finally:
        proceed.set()
        runtime.close()
    reopened = Coordinator(tmp_path, lambda *_: {})
    reopened.close()


def test_single_writer_and_readers_can_poll_during_compute(tmp_path):
    entered, proceed = Event(), Event()

    def processor(context, _):
        context.progress("discovery", "Получаем документы", 10, 100)
        entered.set()
        assert proceed.wait(5)
        return {}

    runtime = Coordinator(tmp_path, processor)
    try:
        run = runtime.submit({})
        assert entered.wait(5)
        assert runtime.get(run)["completed"] == 10
        assert runtime.list_runs()[0]["total"] == 100
        with pytest.raises(sqlite3.ProgrammingError):
            runtime._connection.execute("SELECT 1")
    finally:
        proceed.set()
        runtime.close()


def test_resume_after_workflow_change_requires_new_run(tmp_path):
    def fail(context, _):
        context.checkpoint("plan", {"version": "old"})
        raise TaskFailure("source unavailable")

    runtime = Coordinator(tmp_path, fail, workflow_version="old")
    run = runtime.submit({})
    runtime.wait()
    runtime.close()
    runtime = Coordinator(tmp_path, lambda *_: {}, workflow_version="new")
    try:
        with pytest.raises(TaskFailure, match="Метод анализа изменился"):
            runtime.resume(run)
    finally:
        runtime.close()


@pytest.mark.parametrize("committed", [False, True])
def test_transient_terminal_error_retries_only_idempotent_state_write(tmp_path, monkeypatch, committed):
    calls = []
    runtime = Coordinator(tmp_path, lambda *_: calls.append("processor") or {"ok": True})
    terminal = runtime._terminal
    writes = []

    def fail_once(context, state, error):
        writes.append(state)
        if len(writes) == 1:
            if committed:
                terminal(context, state, error)
            raise sqlite3.OperationalError("Secret provider payload must never appear")
        terminal(context, state, error)

    monkeypatch.setattr(runtime, "_terminal", fail_once)
    try:
        run = runtime.submit({})
        runtime.wait()
        assert calls == ["processor"] and writes == ["succeeded", "succeeded"]
        assert runtime.get(run)["state"] == "succeeded"
        assert runtime.result(run) == {"ok": True}
    finally:
        runtime.close()


def test_permanent_terminal_failure_is_visible_blocks_admission_and_recovers_on_restart(tmp_path, monkeypatch):
    runtime = Coordinator(tmp_path, lambda *_: {"ok": True})

    def broken_terminal(*_):
        raise sqlite3.OperationalError("disk failure with SECRET KEY")

    monkeypatch.setattr(runtime, "_terminal", broken_terminal)
    run = runtime.submit({"query": "robotics"})
    runtime.wait()
    try:
        row = runtime.get(run)
        assert row["state"] == "failed" and row["persistence_error"] is True
        assert "SECRET" not in row["error"]
        assert runtime.list_runs()[0] == row
        connection = sqlite3.connect(tmp_path / "pilot.sqlite3")
        try:
            assert connection.execute("SELECT state FROM analysis_runs WHERE id=?", (run,)).fetchone()[0] == "running"
        finally:
            connection.close()
        with pytest.raises(TaskFailure, match="сохранить"):
            runtime.result(run)
        with pytest.raises(TaskFailure, match="сохранить"):
            runtime.submit({})
        with pytest.raises(TaskFailure, match="сохранить"):
            runtime.resume(run)
        with pytest.raises(Exception, match="используется|уже"):
            Coordinator(tmp_path, lambda *_: {})
    finally:
        runtime.close()
    reopened = Coordinator(tmp_path, lambda context, _: context.load_checkpoint("result"))
    try:
        assert reopened.get(run)["state"] == "interrupted"
        reopened.resume(run)
        reopened.wait()
        assert reopened.result(run) == {"ok": True}
    finally:
        reopened.close()


def test_real_sqlite_write_refusal_finishes_polling_and_survives_unreadable_db(tmp_path, monkeypatch):
    def processor(context, _):
        context.checkpoint("retrieval", {"documents": []})
        context.connection.execute("PRAGMA query_only=ON")
        return {"ok": True}

    runtime = Coordinator(tmp_path, processor)
    try:
        run = runtime.submit({})
        runtime.wait()
        assert runtime.get(run)["state"] == "failed"
        assert runtime.get(run)["persistence_error"]

        def broken_reader():
            raise sqlite3.OperationalError("unreadable DB SECRET")

        monkeypatch.setattr(runtime, "_reader", broken_reader)
        assert runtime.get(run)["state"] == runtime.list_runs()[0]["state"] == "failed"
        with pytest.raises(TaskFailure, match="сохранить"):
            runtime.result(run)
        with pytest.raises(TaskFailure, match="прочитать"):
            runtime.get("missing")
    finally:
        runtime.close()


def test_initial_running_transition_failure_does_not_execute_processor(tmp_path, monkeypatch):
    calls = []
    runtime = Coordinator(tmp_path, lambda *_: calls.append(True) or {})
    connect = runtime._run_connection

    def connection_refusing_writes():
        # Каждый запуск пишет своим соединением: отказ записи — в нём.
        connection = connect()
        connection.execute("PRAGMA query_only=ON")
        return connection

    monkeypatch.setattr(runtime, "_run_connection", connection_refusing_writes)
    try:
        run = runtime.submit({})
        runtime.wait()
        assert calls == [] and runtime.get(run)["state"] == "failed"
        assert runtime.get(run)["persistence_error"]
    finally:
        runtime.close()


def test_cancelled_task_with_unwritable_terminal_row_does_not_remain_running(tmp_path, monkeypatch):
    entered, proceed = Event(), Event()

    def processor(*_):
        entered.set()
        assert proceed.wait(5)
        return {}

    runtime = Coordinator(tmp_path, processor)

    def broken_terminal(*_):
        raise sqlite3.OperationalError("read-only")

    monkeypatch.setattr(runtime, "_terminal", broken_terminal)
    try:
        run = runtime.submit({})
        assert entered.wait(5)
        assert runtime.cancel(run)
        proceed.set()
        runtime.wait()
        assert runtime.get(run)["state"] == "failed" and runtime.get(run)["persistence_error"]
        with pytest.raises(TaskFailure):
            runtime.result(run)
    finally:
        proceed.set()
        runtime.close()


def test_unconfirmed_commit_cannot_publish_success_from_underlying_row(tmp_path, monkeypatch):
    runtime = Coordinator(tmp_path, lambda *_: {"ok": True})
    terminal = runtime._terminal

    def commit_then_fail(context, state, error):
        terminal(context, state, error)
        raise sqlite3.OperationalError("unconfirmed commit")

    monkeypatch.setattr(runtime, "_terminal", commit_then_fail)
    try:
        run = runtime.submit({})
        runtime.wait()
        assert runtime.get(run)["state"] == "failed"
        with pytest.raises(TaskFailure):
            runtime.result(run)
    finally:
        runtime.close()


def test_executor_level_exit_is_read_visible_and_close_still_releases_after_exit(tmp_path, monkeypatch):
    runtime = Coordinator(tmp_path, lambda *_: {})

    def escaped(*_):
        raise SystemExit("must not escape to UI")

    monkeypatch.setattr(runtime, "_run", escaped)
    run = runtime.submit({})
    with pytest.raises(SystemExit):
        runtime.wait()
    assert runtime.get(run)["state"] == "failed"
    assert runtime.list_runs()[0]["persistence_error"]
    runtime.close()
    reopened = Coordinator(tmp_path, lambda *_: {})
    try:
        assert reopened.get(run)["state"] == "interrupted"
    finally:
        reopened.close()


def test_uncommitted_processor_transaction_never_publishes_false_success(tmp_path):
    def processor(context, _):
        context.connection.execute("BEGIN IMMEDIATE")
        return {"unsaved": True}

    runtime = Coordinator(tmp_path, processor)
    try:
        run = runtime.submit({})
        runtime.wait()
        assert runtime.get(run)["state"] == "failed"
        with pytest.raises(TaskFailure, match="отсутствует"):
            runtime.result(run)
    finally:
        runtime.close()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Native Unix pipe boundary")
def test_checkpoint_replaced_by_fifo_cannot_block_result_or_shutdown(tmp_path):
    runtime = Coordinator(tmp_path, lambda *_: {"ok": True})
    run = runtime.submit({})
    runtime.wait()
    path = next((tmp_path / "checkpoints").glob("*.json"))
    path.unlink()
    os.mkfifo(path)
    errors = []

    def read():
        try:
            runtime.result(run)
        except TaskFailure as error:
            errors.append(error)

    thread = Thread(target=read, daemon=True)
    thread.start()
    thread.join(0.25)
    blocked = thread.is_alive()
    if blocked:
        descriptor = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
        os.write(descriptor, b"{}")
        os.close(descriptor)
        thread.join(1)
    try:
        assert not blocked and len(errors) == 1
    finally:
        runtime.close()


def test_history_offset_pages_do_not_repeat_or_skip_completed_runs(tmp_path):
    runtime = Coordinator(tmp_path, lambda *_: {})
    try:
        identifiers = []
        for _ in range(5):
            identifiers.append(runtime.submit({}))
            runtime.wait()
        assert [row["id"] for row in runtime.list_runs(limit=2, offset=1)] == identifiers[-2:-4:-1]
        assert runtime.list_runs(offset=50) == []
        for offset in (-1, True, "0"):
            with pytest.raises(ValueError):
                runtime.list_runs(offset=offset)
    finally:
        runtime.close()


def test_two_slots_run_two_analyses_at_once_and_refuse_a_third(tmp_path):
    started, release = [], Event()

    def processor(context, payload):
        started.append(payload["n"])
        assert release.wait(10)
        # Каждый запуск пишет ход своим соединением, одновременно с соседом.
        context.progress("plan", f"run {payload['n']}")
        context.checkpoint("plan", {"n": payload["n"]})
        return {"n": payload["n"]}

    runtime = Coordinator(tmp_path, processor, slots=2)
    try:
        first, second = runtime.submit({"n": 1}), runtime.submit({"n": 2})
        for _ in range(200):
            if len(started) == 2:
                break
            Event().wait(0.02)
        assert sorted(started) == [1, 2]
        assert sorted(runtime.running()) == sorted([first, second])
        with pytest.raises(TaskFailure, match="места"):
            runtime.submit({"n": 3})
        release.set()
        runtime.wait()
        assert runtime.get(first)["state"] == runtime.get(second)["state"] == "succeeded"
        assert runtime.result(second) == {"n": 2}
        third = runtime.submit({"n": 3})
        runtime.wait()
        assert runtime.get(third)["state"] == "succeeded" and runtime.running() == []
    finally:
        release.set()
        runtime.close()


def test_one_slot_keeps_the_desktop_rule_of_one_analysis_at_a_time(tmp_path):
    release = Event()
    runtime = Coordinator(tmp_path, lambda *_: release.wait(10) and {})
    try:
        runtime.submit({})
        with pytest.raises(TaskFailure, match="Дождитесь текущего анализа"):
            runtime.submit({})
        runtime.set_slots(2)
        runtime.submit({})
        with pytest.raises(ValueError):
            runtime.set_slots(9)
    finally:
        release.set()
        runtime.close()
