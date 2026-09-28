"""Exercise actual text cleaning with a Tk cancellation callback, off the UI thread."""

import json
import tempfile
import time
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event
from unittest.mock import patch

from app.ml import engine
from app.ml.contracts import AnalysisInputError
from app.ui.window import Application
from tests.mvp_fixture import snapshot
from tests.ui.test_desktop import TkCase


class InputResponsivenessTests(TkCase):
    def start(self):
        self.app = Application(self.root, lambda: self.backend)
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready and not self.app.loading and self.app.total == 65)

    def test_oversized_title_is_rejected_before_cleaning_in_worker(self):
        self.start()
        data = snapshot()
        oversized = '<' * 80_000
        data['batches'][0]['documents'][0]['document']['title'] = oversized
        original = engine.clean

        def guarded_clean(value):
            self.assertNotEqual(value, oversized, 'Oversized input reached text preparation')
            return original(value)

        errors, results = [], []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'oversized.json'
            path.write_text(json.dumps(data))
            with patch.object(engine, 'clean', side_effect=guarded_clean):
                self.controller.call('input-probe', 'ml_analyze', results.append, errors.append,
                    {'topic': 'photonic neuromorphic computing'}, snapshot_path=path)
                self.pump(lambda: bool(errors or results), timeout=5)
        self.assertEqual(results, [])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], AnalysisInputError)
        self.assertEqual(self.callback_errors, [])

    def test_valid_size_malformed_abstract_does_not_delay_cancel_callback(self):
        self.start()
        data = snapshot()
        malformed = '<' * 200_000
        data['batches'][0]['documents'][0]['document']['abstract'] = malformed
        entered, release, cancel = Event(), Event(), Event()
        original = engine.clean
        calls, errors, results, callback_times, deadlines = [], [], [], [], []

        def observed_clean(value):
            if value != malformed:
                return original(value)
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Test did not release worker')
            started = time.monotonic()
            result = original(value)  # The production sanitizer does all the work.
            calls.append(time.monotonic() - started)
            # Keep the operation alive until its scheduled cancellation. This
            # wait releases the GIL, unlike the original pathological regex.
            cancel.wait(5)
            return result

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'malformed.json'
            path.write_text(json.dumps(data))
            with patch.object(engine, 'clean', side_effect=observed_clean):
                try:
                    self.controller.call('input-probe', 'ml_analyze', results.append, errors.append,
                        {'topic': 'photonic neuromorphic computing'}, snapshot_path=path, cancel=cancel)
                    self.pump(entered.is_set, timeout=5)
                    self.assertTrue(entered.is_set())

                    def request_cancel():
                        callback_times.append(time.monotonic())
                        cancel.set()

                    def start_probe():
                        # Begin both the worker and deadline inside the active
                        # event loop, after pending startup geometry. Measuring
                        # a stopped/restarted mainloop includes display-server
                        # setup rather than cancellation responsiveness.
                        deadlines.append(time.monotonic() + .1)
                        self.root.after(100, request_cancel)
                        release.set()

                    self.root.after_idle(start_probe)
                    self.pump(lambda: bool(errors or results), timeout=5)
                finally:
                    release.set()
                    cancel.set()
        self.assertTrue(callback_times)
        self.assertEqual(len(deadlines), 1)
        self.assertLess(callback_times[0] - deadlines[0], .5)
        self.assertTrue(calls)
        self.assertLess(max(calls), .5)
        self.assertEqual(results, [])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], CancelledError)
        self.assertEqual(self.callback_errors, [])
