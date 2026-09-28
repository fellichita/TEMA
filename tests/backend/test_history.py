import json
import sqlite3
import subprocess
import sys
from datetime import date, timedelta
from threading import Event

import pytest
from pydantic import ValidationError

from app.backend import __main__ as cli
from app.backend.config import BackendSettings
from app.backend.contracts import DocumentRecord, SourcePage
from app.backend.errors import BackendError
from app.backend.history import HistoryRequest, HistoryStore
from app.backend.repository import Repository, SCHEMA_VERSION, _SCHEMA
from app.backend.service import Backend


def settings(path, **kwargs):
    return BackendSettings(data_dir=path, history_period_delay_seconds=0, **kwargs)


def request(**kwargs):
    return HistoryRequest.model_validate(dict(topic="test topic", from_date="2023-01-01",
                                              until_date="2024-12-31", period="year", sources=["crossref"]) | kwargs)


def page(req, *, exhausted=True, skipped=0, title=None):
    return SourcePage(documents=(DocumentRecord(
        source=req.source, source_id=str(req.from_date), doi=f"10.1234/{req.from_date}",
        title=title or f"Publication {req.from_date}", url=f"https://doi.org/10.1234/{req.from_date}",
    ),), scanned=1 + skipped, skipped=skipped, total_available=1 + skipped if exhausted else 100,
                      exhausted=exhausted)


class Provider:
    def __init__(self, calls=None, fail_year=None, exhausted=True, skipped=0):
        self.calls = calls if calls is not None else []
        self.fail_year, self.exhausted, self.skipped = fail_year, exhausted, skipped

    def iter_pages(self, req, cancel):
        self.calls.append(req)
        if req.from_date.year == self.fail_year:
            raise BackendError("source_unavailable", "Temporary failure")
        yield page(req, exhausted=self.exhausted, skipped=self.skipped)

    def close(self):
        pass


@pytest.mark.parametrize("period", ["month", "year"])
def test_periods_are_contiguous_inclusive_and_clipped(period):
    req = request(from_date="2023-02-15", until_date="2024-03-10", period=period)
    intervals = list(req.intervals())
    assert intervals[0][0] == date(2023, 2, 15)
    assert intervals[-1][1] == date(2024, 3, 10)
    assert not intervals[0][2] and not intervals[-1][2]
    assert all(a[1] + timedelta(days=1) == b[0] for a, b in zip(intervals, intervals[1:], strict=False))
    if period == "month":
        assert (date(2024, 2, 1), date(2024, 2, 29), True) in intervals


@pytest.mark.parametrize("changes", [
    {"from_date": "2025-01-01"}, {"until_date": "2999-01-01"}, {"from_date": "0900-01-01"},
    {"from_date": "1900-01-01"}, {"sources": []}, {"sources": ["crossref", "crossref"]},
    {"sources": ["unknown"]}, {"period": "week"}, {"max_results_per_period": True},
    {"max_results_per_period": 0}, {"topic": "x"}, {"topic": "hello\nworld"},
    {"from_date": "1980-01-01", "sources": ["crossref", "openalex", "epo"], "period": "month"},
])
def test_bad_plans_are_rejected(changes):
    with pytest.raises(ValidationError):
        request(**changes)


@pytest.mark.parametrize("changes", [
    {"sources": ["crossref"]}, {"sources": ["openalex", "crossref"]},
    {"primary_topic_ids": ["T1", "https://openalex.org/T1"]},
    {"primary_topic_ids": ["T1|T2"]},
])
def test_primary_topic_plan_rejects_unsupported_or_ambiguous_scope(changes):
    with pytest.raises(ValidationError):
        request(**(dict(sources=["openalex"], primary_topic_ids=["T1"]) | changes))


def test_multi_year_multi_source_and_stable_resume(tmp_path):
    calls = []
    with Backend(settings(tmp_path), lambda: Provider(calls)) as backend:
        report = backend.collect_history(request(sources=["crossref", "openalex"]))
        assert report.state == "succeeded" and report.coverage_complete
        assert report.completed_periods == report.total_periods == report.processed_periods == 4
        assert len(calls) == 4
        assert len(backend.list_jobs()) == 4
        assert backend.list_documents(history_id=report.id).total == 2  # DOI dedup across sources
        assert backend.list_history()[0]["id"] == report.id
        assert backend.resume_history(report.id) == report.id
        assert backend.wait_history(report.id) == report
        assert len(calls) == 4
        assert backend.cancel_history(report.id) is False


def test_failed_period_is_retried_but_complete_one_is_unchanged(tmp_path):
    with Backend(settings(tmp_path), lambda: Provider(fail_year=2023)) as backend:
        before = backend.collect_history(request())
        assert before.state == "partial" and before.failed_periods == 1
        assert before.periods[1].state == "complete"
    calls = []
    with Backend(settings(tmp_path), lambda: Provider(calls)) as backend:
        backend.resume_history(before.id)
        after = backend.wait_history(before.id)
        assert after.coverage_complete and len(calls) == 1
        assert calls[0].from_date.year == 2023
        assert len(after.periods[0].attempts) == 2
        assert after.periods[1].job.id == before.periods[1].job.id
        assert backend.get_job(before.periods[0].job.id).state == "failed"


@pytest.mark.parametrize("exhausted,skipped,reason", [
    (False, 0, "source_not_exhausted"), (True, 1, "invalid_records_skipped"),
])
def test_incomplete_is_explicit_and_retried_only_on_request(tmp_path, exhausted, skipped, reason):
    calls = []
    with Backend(settings(tmp_path), lambda: Provider(calls, exhausted=exhausted, skipped=skipped)) as backend:
        report = backend.collect_history(request())
        assert report.state == "partial" and not report.coverage_complete
        assert report.partial_periods == 2 and report.processed_periods == 2
        assert report.periods[0].incomplete_reason == reason
        backend.resume_history(report.id)
        assert backend.wait_history(report.id).partial_periods == 2
        assert len(calls) == 2
        backend.resume_history(report.id, retry_incomplete=True)
        retried = backend.wait_history(report.id)
        assert len(calls) == 4 and len(retried.periods[0].attempts) == 2


def test_cancel_and_resume_keeps_saved_pages_and_old_snapshot(tmp_path):
    entered = Event()
    class Blocking(Provider):
        def iter_pages(self, req, cancel):
            yield page(req, exhausted=False, title="Old incomplete snapshot")
            entered.set()
            assert cancel.wait(5)
    with Backend(settings(tmp_path), Blocking) as backend:
        run_id = backend.submit_history(request())
        assert entered.wait(3)
        progress = backend.get_history(run_id)
        assert progress.periods[0].job.stored == 1 and progress.periods[1].state == "pending"
        assert backend.cancel_history(run_id)
        report = backend.wait_history(run_id, timeout=5)
        assert report.state == "cancelled"
        assert report.periods[0].state == "cancelled"
        old_job = report.periods[0].job.id
    with Backend(settings(tmp_path), Provider) as backend:
        backend.resume_history(run_id)
        report = backend.wait_history(run_id)
        assert report.coverage_complete
        assert backend.list_documents(history_id=run_id).total == 2
        assert backend.list_documents(job_id=old_job).items[0].document.title == "Old incomplete snapshot"
        assert backend.list_documents(history_id=run_id, query="Old incomplete").total == 0


def test_duplicate_resume_and_shared_queue_limit(tmp_path):
    entered = Event()
    class Blocking(Provider):
        def iter_pages(self, req, cancel):
            entered.set()
            assert cancel.wait(5)
            yield page(req)
    with Backend(settings(tmp_path, max_pending_jobs=1), Blocking) as backend:
        run_id = backend.submit_history(request())
        assert entered.wait(3)
        with pytest.raises(BackendError, match="уже запущен"):
            backend.resume_history(run_id)
        with pytest.raises(BackendError) as error:
            backend.submit_history(request())
        assert error.value.code == "queue_full"
        from app.backend.contracts import SearchRequest
        with pytest.raises(BackendError) as error:
            backend.submit_collection(SearchRequest(topic="test"))
        assert error.value.code == "queue_full"
    with Backend(settings(tmp_path), Provider) as backend:
        assert backend.get_history(run_id).state == "cancelled"


def test_hard_process_exit_and_restart(tmp_path):
    code = '''
import os, sys
from pathlib import Path
from app.backend.service import Backend
from app.backend.config import BackendSettings
from app.backend.history import HistoryRequest
from app.backend.contracts import DocumentRecord, SourcePage
class Provider:
    def iter_pages(self, req, cancel):
        doc = DocumentRecord(source=req.source, source_id=str(req.from_date), doi=f"10.1234/{req.from_date}", title="Saved before crash", url="https://example.org")
        yield SourcePage(documents=(doc,), scanned=1, total_available=1 if req.from_date.year==2023 else 10, exhausted=req.from_date.year==2023)
        if req.from_date.year == 2024:
            os._exit(27)
    def close(self): pass
backend=Backend(BackendSettings(data_dir=Path(sys.argv[1]), history_period_delay_seconds=0), Provider)
run_id=backend.submit_history(HistoryRequest(topic="test topic", from_date="2023-01-01", until_date="2024-12-31", period="year", sources=("crossref",)))
print(run_id, flush=True)
backend.wait_history(run_id)
'''
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=15)
    assert child.returncode == 27, child.stderr
    run_id = child.stdout.strip()
    calls = []
    with Backend(settings(tmp_path), lambda: Provider(calls)) as backend:
        before = backend.get_history(run_id)
        assert before.state == "interrupted"
        assert before.periods[0].state == "complete"
        assert before.periods[1].state == "interrupted" and before.periods[1].job.stored == 1
        backend.resume_history(run_id)
        after = backend.wait_history(run_id)
        assert after.coverage_complete and len(calls) == 1
        assert after.periods[0].attempts == before.periods[0].attempts
        assert len(after.periods[1].attempts) == 2
        assert backend.list_documents(history_id=run_id).total == 2


def test_atomic_job_and_attempt_link(tmp_path):
    repo = Repository(tmp_path / "test.sqlite3")
    store = HistoryStore(repo)
    run_id = store.create(request())
    store.state(run_id, "running")
    period = store.report(run_id).periods[0]
    with sqlite3.connect(repo.path) as connection:
        connection.execute("CREATE TRIGGER fail_attempt BEFORE INSERT ON history_attempts BEGIN SELECT RAISE(ABORT, 'test'); END")
    with pytest.raises(BackendError):
        store.begin_attempt(run_id, period.id)
    assert repo.list_jobs() == ()
    assert store.report(run_id).periods[0].attempts == ()


def test_v2_migration_preserves_database_and_backup(tmp_path):
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as connection:
        for statement in _SCHEMA:
            connection.execute(statement)
        connection.execute("ALTER TABLE jobs ADD COLUMN contract_version INTEGER NOT NULL DEFAULT 1")
        connection.execute("PRAGMA user_version=2")
    repo = Repository(path)
    assert HistoryStore(repo).list_runs() == ()
    backups = list(tmp_path.glob(f"old.sqlite3.before-v{SCHEMA_VERSION}-*.bak"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_latest_attempt_documents_pagination_and_missing_run(tmp_path):
    with Backend(settings(tmp_path), Provider) as backend:
        report = backend.collect_history(request())
        first = backend.list_documents(history_id=report.id, limit=1)
        second = backend.list_documents(history_id=report.id, limit=1, offset=1)
        assert first.total == second.total == 2
        assert first.items[0].document_key != second.items[0].document_key
        with pytest.raises(BackendError):
            backend.list_documents(history_id=report.id, job_id=report.periods[0].job.id)
        for operation in (backend.get_history, backend.resume_history, backend.wait_history, backend.cancel_history):
            with pytest.raises(BackendError) as error:
                operation("missing")
            assert error.value.code == "history_not_found"


def test_cli_history_resume_and_documents(tmp_path, capsys, monkeypatch):
    class TestBackend(Backend):
        def __init__(self, config):
            super().__init__(settings(config.data_dir), Provider)
    monkeypatch.setattr(cli, "Backend", TestBackend)
    base = ["--data-dir", str(tmp_path)]
    assert cli.main(base + ["collect-history", "test", "--from-date", "2023-01-01", "--until-date", "2024-12-31", "--period", "year", "--sources", "crossref"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["coverage_complete"] and report["completed_periods"] == 2
    run_id = report["id"]
    assert cli.main(base + ["history", run_id]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == run_id
    assert cli.main(base + ["histories"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["id"] == run_id
    assert cli.main(base + ["resume-history", run_id]) == 0
    capsys.readouterr()
    assert cli.main(base + ["documents", "--history-id", run_id]) == 0
    assert json.loads(capsys.readouterr().out)["total"] == 2


def test_cli_invalid_plan_creates_no_storage(tmp_path):
    path = tmp_path / "missing"
    assert cli.main(["--data-dir", str(path), "collect-history", "test", "--from-date", "2999-01-01"]) == 2
    assert not path.exists()


def test_failed_orchestrator_can_resume_orphan_attempt_without_reopen(tmp_path, monkeypatch):
    with Backend(settings(tmp_path), Provider) as backend:
        original = backend._collect
        def broken(*args):
            raise RuntimeError("private internal details")
        monkeypatch.setattr(backend, "_collect", broken)
        before = backend.collect_history(request())
        assert before.state == "failed" and before.periods[0].job.state == "queued"
        assert "private internal details" not in before.model_dump_json()
        monkeypatch.setattr(backend, "_collect", original)
        backend.resume_history(before.id)
        after = backend.wait_history(before.id)
        assert after.coverage_complete and len(after.periods[0].attempts) == 2
        assert backend.get_job(before.periods[0].job.id).state == "interrupted"


def test_inconsistent_source_count_never_claims_complete_history(tmp_path):
    class Inconsistent(Provider):
        def iter_pages(self, req, cancel):
            yield page(req).model_copy(update={"total_available": 100})
    with Backend(settings(tmp_path), Inconsistent) as backend:
        report = backend.collect_history(request())
        assert report.partial_periods == 2 and not report.coverage_complete
        assert report.periods[0].incomplete_reason == "inconsistent_total"
        backend.resume_history(report.id, retry_incomplete=True)
        assert len(backend.wait_history(report.id).periods[0].attempts) == 2


def test_complete_months_do_not_imply_complete_calendar_year(tmp_path):
    with Backend(settings(tmp_path), Provider) as backend:
        report = backend.collect_history(request(from_date="2024-01-01", until_date="2024-02-29", period="month"))
        assert report.coverage_complete
        assert all(p.full_calendar_period for p in report.periods)
        assert report.partial_calendar_years == (2024,)
