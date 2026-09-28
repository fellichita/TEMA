from io import StringIO
import json
from threading import Event

import pytest

from app.backend import __main__ as cli
from app.backend.config import BackendSettings
from app.backend.contracts import DocumentRecord, SourcePage
from app.backend.errors import BackendError
from app.backend.history import HistoryRequest
from app.backend.history_progress import ProgressPrinter, history_progress, render_history_progress
from app.backend.service import Backend


def request(**changes):
    return HistoryRequest.model_validate(dict(topic="progress test", from_date="2024-01-01",
        until_date="2024-02-29", period="month", sources=["crossref"], auto_split=False) | changes)


def config(path):
    return BackendSettings(data_dir=path, history_period_delay_seconds=0)


class Provider:
    def __init__(self, mode="complete"):
        self.mode = mode

    def iter_pages(self, req, cancel):
        if self.mode == "failed":
            raise BackendError("source_unavailable", "Source unavailable")
        if self.mode == "empty":
            yield SourcePage(scanned=0, total_available=0, exhausted=True)
            return
        doc = DocumentRecord(source=req.source, source_id=str(req.from_date), title="test",
                             url="https://example.org", raw_metadata={"hidden": "PRIVATE-RAW"})
        skipped = self.mode == "skipped"
        yield SourcePage(documents=(doc,), scanned=2 if skipped else 1, skipped=int(skipped),
                         total_available=None if self.mode == "unknown" else 2 if skipped else 1 if self.mode == "complete" else 100,
                         exhausted=self.mode in {"complete", "skipped"})

    def close(self):
        pass


def test_pending_is_not_empty_or_complete(tmp_path):
    with Backend(config(tmp_path), Provider) as backend:
        run_id = backend.history.create(request())
        progress = backend.get_history_progress(run_id)
        assert progress.execution_percent == 0 and progress.pending_periods == 2
        assert not progress.coverage_complete
        for period in progress.periods:
            assert period.coverage == "unknown"
            assert period.scanned is None and period.total_available is None and period.job_id is None
            assert period.attempt_count == 0
        text = render_history_progress(progress)
        assert "?/?/?" in text and "пока неизвестна" in text


@pytest.mark.parametrize("mode,coverage,reason", [
    ("complete", True, None), ("partial", False, "source_not_exhausted"),
    ("skipped", False, "invalid_records_skipped"), ("failed", False, "source_unavailable"),
])
def test_execution_and_coverage_are_separate(tmp_path, mode, coverage, reason):
    with Backend(config(tmp_path), lambda: Provider(mode)) as backend:
        report = backend.collect_history(request())
        progress = backend.get_history_progress(report.id)
        assert progress.execution_percent == 100
        assert progress.coverage_complete is coverage
        assert progress.processed_periods == progress.total_periods == 2
        assert progress.periods[0].incomplete_reason == reason
        assert progress.periods[0].coverage == ("complete" if coverage else "incomplete")
        assert "PRIVATE-RAW" not in progress.model_dump_json()
        text = render_history_progress(progress)
        assert "Процент выполнения не означает полноту данных" in text
        assert "Годы, охваченные не целиком: 2024" in text


def test_empty_success_and_unknown_total_are_different(tmp_path):
    with Backend(config(tmp_path / "empty"), lambda: Provider("empty")) as backend:
        progress = history_progress(backend.collect_history(request()))
        assert progress.periods[0].scanned == progress.periods[0].total_available == 0
        assert progress.periods[0].coverage == "complete"
    with Backend(config(tmp_path / "unknown"), lambda: Provider("unknown")) as backend:
        progress = history_progress(backend.collect_history(request()))
        assert progress.periods[0].scanned == 1 and progress.periods[0].total_available is None
        assert "всего у источника: ?" in render_history_progress(progress)


def test_progress_updates_from_saved_pages_and_cancel(tmp_path):
    entered = Event()
    class Blocking(Provider):
        def iter_pages(self, req, cancel):
            yield from Provider("unknown").iter_pages(req, cancel)
            entered.set()
            assert cancel.wait(5)
    with Backend(config(tmp_path), Blocking) as backend:
        run_id = backend.submit_history(request())
        assert entered.wait(3)
        progress = backend.get_history_progress(run_id)
        assert progress.active_periods == 1 and progress.pending_periods == 1
        assert progress.periods[0].scanned == 1 and progress.periods[0].stored == 1
        assert progress.periods[0].coverage == "unknown"
        backend.cancel_history(run_id)
        backend.wait_history(run_id)
        progress = backend.get_history_progress(run_id)
        assert progress.cancelled_periods == 1 and progress.execution_percent == 0
        assert progress.periods[0].coverage == "incomplete"


def test_split_parents_excluded_from_progress_denominator(tmp_path):
    class Adaptive(Provider):
        def iter_pages(self, req, cancel):
            yield from Provider("complete" if req.from_date == req.until_date else "partial").iter_pages(req, cancel)
    with Backend(config(tmp_path), Adaptive) as backend:
        report = backend.collect_history(request(until_date="2024-01-02", max_results_per_period=1, auto_split=True))
        progress = history_progress(report)
        assert progress.total_plan_periods == 3 and progress.total_periods == 2
        assert progress.split_periods == 1 and progress.completed_periods == 2
        assert progress.execution_percent == 100 and progress.coverage_complete
        assert progress.periods[0].coverage == "replaced"
        assert progress.periods[0].job_id is not None
        assert "по дочерним периодам" in render_history_progress(progress)


def test_epo_effective_limit_and_incomplete_calendar_period(tmp_path):
    with Backend(config(tmp_path), lambda: Provider("complete")) as backend:
        progress = history_progress(backend.collect_history(request(sources=["epo"], max_results_per_period=5000,
                                                                     from_date="2024-01-02")))
        assert progress.periods[0].effective_limit == 2000
        assert not progress.periods[0].full_calendar_period
        assert "неполный календарный период" in render_history_progress(progress)


def test_printer_deduplicates_unchanged_reports_and_includes_fast_finished_jobs(tmp_path):
    with Backend(config(tmp_path), Provider) as backend:
        report = backend.collect_history(request())
        stream = StringIO()
        printer = ProgressPrinter(stream)
        printer.update(report)
        first = stream.getvalue()
        assert "2024-01-01..2024-01-31" in first and "2024-02-01..2024-02-29" in first
        printer.update(report)
        assert stream.getvalue() == first


@pytest.mark.parametrize("format", ["json", "text", "progress-json"])
def test_cli_formats_read_saved_data_offline(tmp_path, capsys, monkeypatch, format):
    with Backend(config(tmp_path), Provider) as backend:
        report = backend.collect_history(request())
    class Offline(Backend):
        def __init__(self, settings):
            super().__init__(settings, lambda: (_ for _ in ()).throw(AssertionError("Network is forbidden")))
    monkeypatch.setattr(cli, "Backend", Offline)
    assert cli.main(["--data-dir", str(tmp_path), "history", report.id, "--format", format]) == 0
    output = capsys.readouterr().out
    if format == "text":
        assert "Полная выдача" in output and "2024-02-01..2024-02-29" in output
    else:
        data = json.loads(output)
        assert data["id"] == report.id
        assert ("execution_percent" in data) == (format == "progress-json")


def test_cli_final_progress_is_stderr_and_stdout_remains_json(tmp_path, monkeypatch, capsys):
    class Local(Backend):
        def __init__(self, settings):
            super().__init__(config(settings.data_dir), Provider)
    monkeypatch.setattr(cli, "Backend", Local)
    assert cli.main(["--data-dir", str(tmp_path), "collect-history", "test", "--from-date", "2024-01-01",
                     "--until-date", "2024-01-31", "--sources", "crossref"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["coverage_complete"]
    assert "2024-01-01..2024-01-31" in output.err and "Полная выдача" in output.err


def test_progress_missing_history_and_closed_backend(tmp_path):
    backend = Backend(config(tmp_path), Provider)
    with pytest.raises(BackendError) as error:
        backend.get_history_progress("missing")
    assert error.value.code == "history_not_found"
    backend.close()
    with pytest.raises(BackendError) as error:
        backend.get_history_progress("missing")
    assert error.value.code == "backend_closed"
