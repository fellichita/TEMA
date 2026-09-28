import json
import sqlite3
from datetime import date, timedelta
from threading import Event

import pytest
from pydantic import ValidationError

from app.backend.config import BackendSettings
from app.backend.contracts import DocumentRecord, SourcePage
from app.backend.errors import BackendError
from app.backend.history import HISTORY_SCHEMA, HistoryRequest, HistoryStore
from app.backend.repository import Repository, _SCHEMA
from app.backend.service import Backend
from app.backend import __main__ as cli


def config(path):
    return BackendSettings(data_dir=path, history_period_delay_seconds=0)


def plan(**changes):
    return HistoryRequest.model_validate(dict(topic="adaptive test", from_date="2024-01-30",
        until_date="2024-02-02", period="year", sources=["crossref"], max_results_per_period=1) | changes)


def result(req, exhausted):
    key = f"{req.from_date}-{req.until_date}"
    doc = DocumentRecord(source=req.source, source_id=key, doi=f"10.1234/{key}",
                         title=key, url=f"https://doi.org/10.1234/{key}")
    return SourcePage(documents=(doc,), scanned=1, total_available=1 if exhausted else 10, exhausted=exhausted)


class Provider:
    def __init__(self, calls=None, daily_overflow=False):
        self.calls = calls if calls is not None else []
        self.daily_overflow = daily_overflow

    def iter_pages(self, req, cancel):
        self.calls.append((req.source, req.from_date, req.until_date))
        yield result(req, req.from_date == req.until_date and not self.daily_overflow)

    def close(self):
        pass


def test_year_month_day_tree_and_leaf_only_documents(tmp_path):
    calls = []
    with Backend(config(tmp_path), lambda: Provider(calls)) as backend:
        report = backend.collect_history(plan())
        assert report.coverage_complete and report.total_periods == report.completed_periods == 4
        assert report.split_periods == 3 and report.total_plan_periods == 7 and len(calls) == 7
        assert report.contract_version == 2
        parent = report.periods[0]
        assert parent.state == "split" and parent.granularity == "year" and parent.parent_id is None
        assert {p.granularity for p in report.periods} == {"year", "month", "day"}
        for node in report.periods:
            children = [p for p in report.periods if p.parent_id == node.id]
            if children:
                assert children[0].from_date == node.from_date and children[-1].until_date == node.until_date
                assert all(a.until_date + timedelta(days=1) == b.from_date for a, b in zip(children, children[1:], strict=False))
                assert {p.source for p in children} == {node.source}
        assert backend.list_documents(history_id=report.id).total == 4
        assert backend.list_documents(job_id=parent.job.id).total == 1
        assert backend.list_documents().total == 7
        backend.resume_history(report.id)
        assert backend.wait_history(report.id).coverage_complete and len(calls) == 7


def test_primary_topic_selection_survives_adaptive_split_and_resume(tmp_path):
    calls = []

    class TopicProvider(Provider):
        def iter_pages(self, req, cancel):
            calls.append(req)
            yield result(req, req.from_date == req.until_date)

    with Backend(config(tmp_path), TopicProvider) as backend:
        report = backend.collect_history(plan(sources=["openalex"], primary_topic_ids=["T2", "T1"]))
        expected = ("https://openalex.org/T1", "https://openalex.org/T2")
        assert report.coverage_complete and len(calls) == 7
        assert report.request.primary_topic_ids == expected
        assert all(request.primary_topic_ids == expected for request in calls)
        assert all(period.job.request.primary_topic_ids == expected for period in report.periods)
        backend.resume_history(report.id)
        assert backend.wait_history(report.id) == report and len(calls) == 7


def test_leap_month_has_29_daily_children(tmp_path):
    with Backend(config(tmp_path), Provider) as backend:
        report = backend.collect_history(plan(from_date="2024-02-01", until_date="2024-02-29", period="month"))
        assert report.coverage_complete and report.total_periods == 29
        assert report.periods[-1].until_date == date(2024, 2, 29)
        assert all(p.full_calendar_period for p in report.periods)


def test_single_day_overflow_is_terminal_partial(tmp_path):
    with Backend(config(tmp_path), lambda: Provider(daily_overflow=True)) as backend:
        report = backend.collect_history(plan())
        assert report.state == "partial" and report.partial_periods == 4
        assert all(p.incomplete_reason == "daily_limit_reached" for p in report.periods if p.state != "split")
        before = len(backend.list_jobs())
        backend.resume_history(report.id)
        after = backend.wait_history(report.id)
        assert after.partial_periods == 4 and len(backend.list_jobs()) == before


def test_budget_blocks_whole_split_without_missing_dates(tmp_path):
    with Backend(config(tmp_path), Provider) as backend:
        report = backend.collect_history(plan(max_periods=2))
        assert report.total_plan_periods == report.total_periods == report.partial_periods == 1
        assert report.periods[0].incomplete_reason == "period_budget_exceeded"
        assert backend.list_documents(history_id=report.id).total == 1


def test_exact_budget_and_source_isolation(tmp_path):
    with Backend(config(tmp_path), Provider) as backend:
        report = backend.collect_history(plan(sources=["crossref", "openalex"], max_periods=14))
        assert report.coverage_complete and report.total_plan_periods == 14 and report.total_periods == 8
        assert backend.list_documents(history_id=report.id).total == 4


def test_disabled_split_keeps_old_behavior(tmp_path):
    with Backend(config(tmp_path), Provider) as backend:
        report = backend.collect_history(plan(auto_split=False))
        assert report.total_plan_periods == 1 and report.partial_periods == 1
        assert report.periods[0].incomplete_reason == "source_not_exhausted"


@pytest.mark.parametrize("changes", [{"max_periods": 0}, {"max_periods": True}, {"max_periods": 10001},
    {"auto_split": "yes"}, {"max_periods": 1, "sources": ["crossref", "openalex"]}])
def test_invalid_budget(changes):
    with pytest.raises(ValidationError):
        plan(**changes)


@pytest.mark.parametrize("mode", ["failure", "skipped", "early_end", "inconsistent"])
def test_non_overflow_problems_do_not_spawn_children(tmp_path, mode):
    class Problem(Provider):
        def iter_pages(self, req, cancel):
            if mode == "failure":
                raise BackendError("rate_limited", "Rate limited")
            if mode == "skipped":
                yield SourcePage(scanned=1, skipped=1, exhausted=True, total_available=1)
            elif mode == "early_end":
                yield SourcePage(scanned=0, exhausted=False, total_available=10)
            else:
                yield result(req, True).model_copy(update={"total_available": 10})
    with Backend(config(tmp_path), Problem) as backend:
        report = backend.collect_history(plan())
        assert not report.coverage_complete and report.total_plan_periods == 1


def test_atomic_split_rollback_and_idempotence(tmp_path):
    repo = Repository(tmp_path / "test.sqlite3")
    store = HistoryStore(repo)
    run_id = store.create(plan())
    store.state(run_id, "running")
    parent = store.report(run_id).periods[0]
    job = store.begin_attempt(run_id, parent.id)
    repo.start_job(job.id)
    repo.ingest_page(job.id, result(job.request, False))
    repo.finish_job(job.id, "succeeded")
    with sqlite3.connect(repo.path) as con:
        con.execute("CREATE TRIGGER reject_second BEFORE INSERT ON history_periods WHEN NEW.ordinal=2 BEGIN SELECT RAISE(ABORT,'test'); END")
    with pytest.raises(BackendError):
        store.split_overflow(run_id, parent.id)
    assert store.report(run_id).total_plan_periods == 1
    assert store.report(run_id).periods[0].state == "partial"
    with sqlite3.connect(repo.path) as con:
        con.execute("DROP TRIGGER reject_second")
    assert len(store.split_overflow(run_id, parent.id)) == 2
    assert store.split_overflow(run_id, parent.id) == ()
    with pytest.raises(BackendError):
        store.begin_attempt(run_id, parent.id)


def test_resume_after_split_commit_does_not_fetch_parent_again(tmp_path, monkeypatch):
    with Backend(config(tmp_path), Provider) as backend:
        original = backend.history.split_overflow
        def crash_after_commit(*args):
            ids = original(*args)
            if ids:
                raise RuntimeError("Injected crash after durable split")
            return ids
        monkeypatch.setattr(backend.history, "split_overflow", crash_after_commit)
        before = backend.collect_history(plan())
        assert before.state == "failed" and before.split_periods == 1
        assert all(p.state == "pending" for p in before.periods[1:])
    calls = []
    with Backend(config(tmp_path), lambda: Provider(calls)) as backend:
        backend.resume_history(before.id)
        after = backend.wait_history(before.id)
        assert after.coverage_complete and len(calls) == 6
        assert after.periods[0].attempts == before.periods[0].attempts


def test_resume_after_parent_job_commit_splits_without_refetch(tmp_path, monkeypatch):
    with Backend(config(tmp_path), Provider) as backend:
        original = backend.history.split_overflow
        def crash_before_split(run_id, period_id):
            report = backend.history.report(run_id)
            if any(p.job for p in report.periods):
                raise RuntimeError("Injected interruption")
            return original(run_id, period_id)
        monkeypatch.setattr(backend.history, "split_overflow", crash_before_split)
        before = backend.collect_history(plan())
        assert before.state == "failed" and before.periods[0].state == "partial"
    calls = []
    with Backend(config(tmp_path), lambda: Provider(calls)) as backend:
        backend.resume_history(before.id)
        assert backend.wait_history(before.id).coverage_complete and len(calls) == 6


def test_cancel_child_then_resume(tmp_path):
    entered = Event()
    class Blocking(Provider):
        def iter_pages(self, req, cancel):
            yield result(req, req.from_date == req.until_date)
            if req.from_date == req.until_date:
                entered.set()
                assert cancel.wait(5)
    with Backend(config(tmp_path), Blocking) as backend:
        run_id = backend.submit_history(plan())
        assert entered.wait(3)
        assert backend.cancel_history(run_id)
        before = backend.wait_history(run_id)
        assert before.state == "cancelled" and before.split_periods == 3
    with Backend(config(tmp_path), Provider) as backend:
        backend.resume_history(run_id)
        assert backend.wait_history(run_id).coverage_complete


def test_epo_uses_actual_2000_cap(tmp_path):
    class Epo(Provider):
        def iter_pages(self, req, cancel):
            if req.from_date != req.until_date:
                doc = result(req, False).documents[0]
                yield SourcePage(documents=(doc,) * 2000, scanned=2000, total_available=5000, exhausted=False)
            else:
                yield result(req, True)
    with Backend(config(tmp_path), Epo) as backend:
        report = backend.collect_history(plan(from_date="2024-02-01", until_date="2024-02-02", period="month",
                                              sources=["epo"], max_results_per_period=3000))
        assert report.coverage_complete and report.split_periods == 1 and report.total_periods == 2


def test_v3_migration_preserves_old_plan_mode(tmp_path):
    path = tmp_path / "old.sqlite3"
    raw = plan().model_dump(mode="json", exclude={"auto_split", "max_periods"})
    with sqlite3.connect(path) as con:
        for sql in _SCHEMA:
            con.execute(sql)
        con.execute("ALTER TABLE jobs ADD COLUMN contract_version INTEGER NOT NULL DEFAULT 1")
        for sql in HISTORY_SCHEMA:
            con.execute(sql)
        con.execute("INSERT INTO history_runs VALUES('legacy',?,'partial','2024-01-01','2024-01-01',NULL,NULL)", (json.dumps(raw),))
        con.execute("INSERT INTO history_periods VALUES('parent','legacy',0,'crossref','2024-01-30','2024-02-02',0,NULL)")
        con.execute("PRAGMA user_version=3")
    repo = Repository(path)
    report = HistoryStore(repo).report("legacy")
    assert report.request.auto_split is False and report.periods[0].granularity == "year"
    assert report.periods[0].parent_id is None
    assert len(list(tmp_path.glob("old.sqlite3.before-v4-*.bak"))) == 1
    with sqlite3.connect(path) as con:
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("disabled", [False, True])
def test_cli_split_options(tmp_path, monkeypatch, capsys, disabled):
    class TestBackend(Backend):
        def __init__(self, settings):
            super().__init__(config(settings.data_dir), Provider)
    monkeypatch.setattr(cli, "Backend", TestBackend)
    args = ["--data-dir", str(tmp_path), "collect-history", "adaptive test", "--from-date", "2024-01-30",
            "--until-date", "2024-02-02", "--period", "year", "--sources", "crossref",
            "--limit-per-period", "1", "--max-periods", "7"]
    if disabled:
        args.append("--no-auto-split")
    assert cli.main(args) == (1 if disabled else 0)
    report = json.loads(capsys.readouterr().out)
    assert report["request"]["auto_split"] == (not disabled)
    assert report["split_periods"] == (0 if disabled else 3)


def test_cli_budget_rejected_before_storage(tmp_path):
    path = tmp_path / "not-created"
    assert cli.main(["--data-dir", str(path), "collect-history", "adaptive test", "--from-date", "2024-01-01",
                     "--until-date", "2024-01-02", "--max-periods", "0"]) == 2
    assert not path.exists()
