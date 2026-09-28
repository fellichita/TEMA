"""Actual Tk event loop and real backend with a synthetic, offline provider."""

from pathlib import Path
import tempfile
from threading import Event
from unittest.mock import patch

from app.backend.config import BackendSettings
from app.backend.service import Backend
from app.ui.window import Application
from app.ml.contracts import AnalysisOptions
from app.ml.corpus import unpack_snapshot
from app.ml.engine import analyze
from tests.mvp_fixture import Provider, snapshot
from tests.ui.test_desktop import TkCase


class LocalMVPWindowTests(TkCase):
    def test_minimum_window_keeps_export_and_source_actions_visible_and_usable(self):
        self.app = Application(self.root, lambda: self.backend)
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready)
        panel = self.app.trends_panel
        self.app.tabs.select(self.app.trends_tab)
        panel.render_result(analyze(unpack_snapshot(snapshot()),
                                    AnalysisOptions(topic="photonic neuromorphic computing")))
        panel.input_label.set("Корпус: corpus-portable.json · 7962 исходных записей")
        self.root.deiconify()
        for size in ("1180x840", "940x740", "1180x840"):
            self.root.geometry(size)
            settled = []
            self.root.after(200, lambda settled=settled: settled.append(True))
            self.pump(lambda settled=settled: bool(settled))
            for button in (panel.export_button, panel.link_button):
                self.assertTrue(button.winfo_ismapped(), (size, button["text"]))
                self.assertTrue(button.instate(["!disabled"]))
                self.assertGreaterEqual(button.winfo_rooty(), panel.parent.winfo_rooty())
                self.assertLessEqual(button.winfo_rooty() + button.winfo_height(),
                                     panel.parent.winfo_rooty() + panel.parent.winfo_height())
            if size == "940x740":
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "trends.json"
                    with patch("app.ui.trends_panel.filedialog.asksaveasfilename", return_value=str(output)):
                        panel.export_button.invoke()
                        self.pump(lambda output=output: output.exists() and not panel.busy)
                with patch("webbrowser.open", return_value=True) as browser:
                    panel.link_button.invoke()
                    self.pump(lambda: browser.called)

    def test_collect_from_form_analyze_show_cards_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = BackendSettings(data_dir=Path(directory))
            self.app = Application(self.root, lambda: Backend(settings, provider_factory=Provider))
            self.controller = self.app.controller
            self.pump(lambda: self.app.ready, timeout=5)
            panel = self.app.trends_panel
            panel.start_year.set("2020")
            panel.end_year.set("2025")
            panel.collect()
            self.pump(lambda: panel.result is not None, timeout=25)
            # Large/background groups are separate from the main emerging TOP.
            self.assertGreaterEqual(sum(len(panel.result.get(key, [])) for key in
                                        ("candidates", "preliminary_signals", "established")), 2)
            self.assertIn("фрагмент источника", panel.detail.get("1.0", "end"))
            output = Path(directory) / "trends.json"
            with patch("app.ui.trends_panel.filedialog.asksaveasfilename", return_value=str(output)):
                panel.export()
                self.pump(lambda: output.exists() and not panel.busy)
            self.assertTrue(panel.export_button.instate(["!disabled"]))
            self.pump(lambda: bool(panel.histories))
            panel.history_box.set(next(iter(panel.histories)))
            panel.select_history()
            self.assertTrue(panel.history_id)
            self.assertIsNone(panel.snapshot_path)
            panel.analyze()
            self.pump(lambda: panel.result is not None, timeout=25)
            self.app.close()
            self.pump(lambda: self.controller.stopped, timeout=5)

    def test_corrupt_json_restores_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text("broken")
            self.app = Application(self.root, lambda: self.backend)
            self.controller = self.app.controller
            self.pump(lambda: self.app.ready)
            panel = self.app.trends_panel
            panel.load_snapshot(path)
            self.pump(lambda: not panel.busy)
            self.assertIn("JSON", panel.status.get())
            self.assertTrue(panel.file_button.instate(["!disabled"]))

    def test_ml_does_not_block_document_reads_and_close_waits_for_worker(self):
        self.app = Application(self.root, lambda: self.backend)
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready)
        started, release = Event(), Event()

        def long_analysis(*args, **kwargs):
            started.set()
            release.wait(5)
            return {}

        with patch("app.ml.service.run_analysis", side_effect=long_analysis):
            try:
                self.controller.call("test_ml", "ml_analyze", self.fail, self.fail, {})
                self.pump(started.is_set)
                documents = []
                self.controller.call("test_read", "list_documents", documents.append, self.fail)
                self.pump(lambda: bool(documents))
                self.app.close()
                self.assertFalse(self.controller.stopped)
                ticks = []
                self.root.after(10, lambda: ticks.append(True))
                self.pump(lambda: bool(ticks))
            finally:
                release.set()
            self.pump(lambda: self.controller.stopped)
