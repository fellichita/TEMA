"""Foreground ownership under real Controller concurrency and Tk callbacks."""

import json
import tempfile
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

from app.ui.window import Application
from tests.ui.test_desktop import TkCase
from tests.ui.test_trend_result_sections import result_sections


class OperationStateTests(TkCase):
    def open_panel(self):
        self.app = Application(self.root, lambda: self.backend)
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready and not self.app.loading
                  and "ml_histories" not in self.controller.pending)
        panel = self.app.trends_panel
        panel.snapshot_path = "local-corpus.json"
        return panel

    def test_history_error_does_not_unlock_or_replace_active_analysis(self):
        panel = self.open_panel()
        history_started, history_release = Event(), Event()
        analysis_started, analysis_release = Event(), Event()

        def histories(**kwargs):
            history_started.set()
            history_release.wait(5)
            raise OSError("private history failure")

        def analyze(*args, **kwargs):
            analysis_started.set()
            analysis_release.wait(5)
            return result_sections()

        with patch.object(self.backend, "list_history", side_effect=histories), patch(
                "app.ml.service.run_analysis", side_effect=analyze) as worker:
            try:
                panel.refresh_histories()
                self.pump(history_started.is_set)
                panel.analyze()
                self.pump(analysis_started.is_set)
                history_release.set()
                self.pump(lambda: "ml_histories" not in self.controller.pending)
                self.assertTrue(panel.busy)
                self.assertIn("ml_analysis", self.controller.pending)
                self.assertTrue(panel.topic_entry.instate(["disabled"]))
                self.assertIn("список историй", panel.notice.get())
                panel.analyze()
                self.assertEqual(worker.call_count, 1)
            finally:
                history_release.set()
                analysis_release.set()
            self.pump(lambda: not panel.busy)
        self.assertIsNotNone(panel.result)
        self.assertEqual(worker.call_args.args[1]["topic"], panel.topic.get())

    def test_rejected_foreground_requests_preserve_result_and_idle_controls(self):
        panel = self.open_panel()
        result = result_sections()
        panel.render_result(result)
        old_event = panel.cancel_event
        actions = (panel.analyze, lambda: panel.load_snapshot("different.json"), panel.collect, panel.export)
        with patch.object(self.controller, "call", return_value=False), patch(
                "app.ui.trends_panel.filedialog.asksaveasfilename", return_value="trends.json"):
            for action in actions:
                with self.subTest(action=action):
                    action()
                    self.assertFalse(panel.busy)
                    self.assertIs(panel.result, result)
                    self.assertIs(panel.cancel_event, old_event)
                    self.assertIn("не запущена", panel.status.get())
                    self.assertTrue(panel.run_button.instate(["!disabled"]))
                    self.assertTrue(panel.export_button.instate(["!disabled"]))

    def test_changed_inputs_discard_inflight_result_and_invalidate_ready_result(self):
        panel = self.open_panel()
        started, release = Event(), Event()

        def analyze(*args, **kwargs):
            started.set()
            release.wait(5)
            return result_sections()

        with patch("app.ml.service.run_analysis", side_effect=analyze):
            try:
                panel.analyze()
                self.pump(started.is_set)
                # UI fields are disabled; this also guards an external variable update.
                panel.topic.set("dna data storage")
            finally:
                release.set()
            self.pump(lambda: not panel.busy)
        self.assertIsNone(panel.result)
        self.assertIn("Параметры изменились", panel.status.get())
        panel.render_result(result_sections())
        panel.start_year.set("2019")
        self.assertIsNone(panel.result)
        self.assertFalse(panel.current_sources)
        self.assertTrue(panel.export_button.instate(["disabled"]))
        self.assertTrue(panel.link_button.instate(["disabled"]))

    def test_cancel_and_old_progress_cannot_affect_next_analysis(self):
        panel = self.open_panel()
        started, release = Event(), Event()
        old_progress = []

        def slow(*args, **kwargs):
            old_progress.append(kwargs["progress"])
            started.set()
            release.wait(5)
            kwargs["progress"](99, "stale progress")
            return result_sections()

        with patch("app.ml.service.run_analysis", side_effect=slow):
            try:
                panel.analyze()
                self.pump(started.is_set)
                first_event = panel.cancel_event
                panel.cancel()
                self.assertTrue(first_event.is_set())
                self.assertTrue(panel.cancel_button.instate(["disabled"]))
                panel.poll()
            finally:
                release.set()
            self.pump(lambda: not panel.busy)
        self.assertIsNone(panel.result)
        self.assertIn("отменена", panel.status.get())
        started.clear()
        release.clear()
        with patch("app.ml.service.run_analysis", side_effect=slow):
            try:
                panel.analyze()
                self.pump(started.is_set)
                self.assertIsNot(panel.cancel_event, first_event)
                self.assertFalse(panel.cancel_event.is_set())
                old_progress[0](98, "old request must stay invisible")
                panel.poll()
                self.assertNotIn("old request", panel.status.get())
                self.assertNotEqual(panel.progress["value"], 98)
            finally:
                release.set()
            self.pump(lambda: not panel.busy)
        self.assertIsNotNone(panel.result)

    def test_inspection_receives_cancel_and_suppresses_auto_analysis(self):
        panel = self.open_panel()
        started, release = Event(), Event()
        events = []

        def inspect(path, *, cancel=None):
            events.append(cancel)
            started.set()
            release.wait(5)
            return {"topic": "different corpus", "source": "crossref", "start_year": 2020,
                    "end_year": 2025, "occurrences": 1}

        with patch("app.ml.service.inspect_snapshot", side_effect=inspect), patch(
                "app.ml.service.run_analysis") as analyze:
            try:
                panel.load_snapshot("different.json", auto_run=True)
                self.pump(started.is_set)
                panel.cancel()
                self.assertIs(events[0], panel.cancel_event)
                self.assertTrue(events[0].is_set())
            finally:
                release.set()
            self.pump(lambda: not panel.busy)
            analyze.assert_not_called()
        self.assertEqual(panel.snapshot_path, "local-corpus.json")
        self.assertIsNone(panel.result)
        self.assertTrue(panel.file_button.instate(["!disabled"]))

    def test_export_is_explicitly_not_cancellable_and_writes_complete_result(self):
        from app.ml.service import export_result

        panel = self.open_panel()
        result = result_sections()
        panel.render_result(result)
        started, release = Event(), Event()

        def delayed(*args, **kwargs):
            started.set()
            release.wait(5)
            return export_result(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.json"
            with patch("app.ml.service.export_result", side_effect=delayed), patch(
                    "app.ui.trends_panel.filedialog.asksaveasfilename", return_value=str(target)):
                try:
                    panel.export()
                    self.pump(started.is_set)
                    self.assertTrue(panel.busy)
                    self.assertTrue(panel.cancel_button.instate(["disabled"]))
                    before = panel.status.get()
                    panel.cancel()
                    self.assertEqual(panel.status.get(), before)
                    self.assertFalse(panel.cancel_event.is_set())
                finally:
                    release.set()
                self.pump(lambda: not panel.busy)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), result)
        self.assertIn("Результат сохранён", panel.status.get())
        self.assertTrue(panel.export_button.instate(["!disabled"]))

    def test_source_open_error_does_not_unlock_export(self):
        panel = self.open_panel()
        panel.render_result(result_sections())
        browser_started, browser_release = Event(), Event()
        export_started, export_release = Event(), Event()

        def browser(url):
            browser_started.set()
            browser_release.wait(5)
            raise OSError("private URL detail")

        def export(*args, **kwargs):
            export_started.set()
            export_release.wait(5)
            return "trends.json"

        with patch("webbrowser.open", side_effect=browser), patch(
                "app.ml.service.export_result", side_effect=export), patch(
                "app.ui.trends_panel.filedialog.asksaveasfilename", return_value="trends.json"):
            try:
                panel.open_source()
                self.pump(browser_started.is_set)
                panel.export()
                self.pump(export_started.is_set)
                browser_release.set()
                self.pump(lambda: "ml_open_source" not in self.controller.pending)
                self.assertTrue(panel.busy)
                self.assertTrue(panel.run_button.instate(["disabled"]))
                self.assertTrue(panel.cancel_button.instate(["disabled"]))
                self.assertIn("открыть источник", panel.notice.get())
            finally:
                browser_release.set()
                export_release.set()
            self.pump(lambda: not panel.busy)

    def test_failed_history_poll_retains_collection_and_retries_cancellation(self):
        panel = self.open_panel()
        panel.collecting_id = panel.history_id = "active-history"
        panel.controls()
        with patch.object(self.backend, "get_history", side_effect=OSError("private DB detail"),
                          create=True), patch.object(self.backend, "cancel_history",
                          side_effect=OSError("private cancellation detail"), create=True):
            panel.poll()
            self.pump(lambda: "ml_history_progress" not in self.controller.pending)
            self.assertEqual(panel.collecting_id, "active-history")
            self.assertTrue(panel.run_button.instate(["disabled"]))
            self.assertTrue(panel.cancel_button.instate(["!disabled"]))
            panel.cancel()
            self.pump(lambda: "ml_cancel_history" not in self.controller.pending)
            self.assertEqual(panel.collecting_id, "active-history")
            self.assertTrue(panel.cancel_button.instate(["!disabled"]))
            self.assertIn("Повторите отмену", panel.notice.get())
        # A delayed terminal reply for a different collection cannot release this one.
        panel.history_progress(SimpleNamespace(state="succeeded", processed_periods=1, total_periods=1),
                               "older-history")
        self.assertEqual(panel.collecting_id, "active-history")
        panel.collecting_id = None

    def test_close_sets_worker_event_and_never_renders_late_result(self):
        panel = self.open_panel()
        started, release = Event(), Event()

        def analyze(*args, **kwargs):
            started.set()
            release.wait(5)
            kwargs["progress"](100, "late close progress")
            return result_sections()

        with patch("app.ml.service.run_analysis", side_effect=analyze):
            try:
                panel.analyze()
                self.pump(started.is_set)
                self.app.close()
                self.assertTrue(panel.cancel_event.is_set())
                self.assertFalse(self.controller.stopped)
                panel.analyze()
            finally:
                release.set()
            self.pump(lambda: self.controller.stopped)
        self.assertIsNone(panel.result)

    def test_cancel_during_collection_start_is_forwarded_after_id_arrives(self):
        panel = self.open_panel()
        started, release = Event(), Event()

        def collect(request):
            started.set()
            release.wait(5)
            return "new-history"

        history = SimpleNamespace(state="cancelled", processed_periods=0, total_periods=6)
        with patch.object(self.backend, "submit_history", side_effect=collect, create=True), patch.object(
                self.backend, "cancel_history", return_value=True, create=True) as cancel, patch.object(
                self.backend, "get_history", return_value=history, create=True), patch(
                "app.ml.service.run_analysis") as analyze:
            try:
                panel.collect()
                self.pump(started.is_set)
                panel.cancel()
                self.assertTrue(panel.busy)
                self.assertTrue(panel.cancel_event.is_set())
                self.assertIsNone(panel.collecting_id)
            finally:
                release.set()
            self.pump(lambda: cancel.called)
            cancel.assert_called_once_with("new-history")
            panel.poll()
            self.pump(lambda: panel.collecting_id is None)
            analyze.assert_not_called()
        self.assertFalse(panel.busy)
        self.assertEqual(panel.history_id, "new-history")
        self.assertIsNone(panel.snapshot_path)
        self.assertTrue(panel.run_button.instate(["!disabled"]))
