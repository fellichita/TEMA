"""Offline UI checks: python -m unittest discover -s tests/ui -v."""

import threading
import time
import tkinter as tk
import unittest
import gc
import importlib.util
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace as Record

import pytest

from app.ui.controller import Controller
from app.ui.presentation import collection_values, error_message, job_state, publication_date
from app.ui.window import Application


def document(number):
    doc = Record(title=f"Исследование {number}", source="crossref", source_id=str(number),
                 abstract=None, authors=(), publication_year=2024, publication_month=None,
                 publication_date=None, date_precision="year", citation_count=None, language=None,
                 document_type="publication", doi=f"10.1234/{number}", patent_publication=None,
                 fetched_at=datetime(2026, 9, 1, tzinfo=timezone.utc), url=f"https://doi.org/10.1234/{number}")
    return Record(document_key=f"doi:10.1234/{number}", document=doc, sources=("crossref",), revision_id=str(number))


class FakeBackend:
    def __init__(self):
        self.settings = Record(data_dir=Path("test-storage"))
        self.records = [document(i) for i in range(65)]
        self.jobs = []
        self.closed = False
        self.read_gate = None
        self.close_gate = None
        self.read_started = threading.Event()
        self.calls = []
        self.read_error = False
        self.fail_close = False

    def sources(self):
        return ({"id": "crossref"}, {"id": "openalex"}, {"id": "epo", "credentials_configured": False})

    def list_documents(self, query=None, limit=50, offset=0, job_id=None, sort_by="default", descending=False, history_id=None):
        self.calls.append((query, offset, job_id, threading.get_ident()))
        self.read_started.set()
        gate, self.read_gate = self.read_gate, None
        if gate is not None:
            gate.wait(4)
        if self.read_error:
            raise RuntimeError("private details must not reach the UI")
        items = [r for r in self.records if not query or query.casefold() in r.document.title.casefold()]
        if sort_by == "title":
            items.sort(key=lambda r: r.document.title.casefold(), reverse=descending)
        return Record(items=items[offset:offset + limit], total=len(items))

    def list_jobs(self, limit=100):
        return self.jobs[:limit]

    def list_history(self, limit=50):
        return ()

    def cancel(self, job_id):
        for job in self.jobs:
            if job.id == job_id:
                job.state = "cancelled"
                job.updated_at += 1
                return True
        return False

    def close(self):
        if self.close_gate:
            self.close_gate.wait(4)
        if self.fail_close:
            self.fail_close = False
            raise OSError("private storage path")
        self.closed = True


@pytest.mark.gui
class TkCase(unittest.TestCase):
    def setUp(self):
        self._ui_started_at = time.monotonic()
        # Finalize destroyed Tk roots on their owner thread before workers start.
        # Otherwise a later ML allocation may collect an old Tcl interpreter.
        gc.collect()
        self.root = tk.Tk()
        self.root.withdraw()
        self.callback_errors = []
        self.root.report_callback_exception = lambda *error: self.callback_errors.append(error)
        self.backend = FakeBackend()
        self.controller = None

    def drop_topmost(self):
        """Release a window raised for a hit test, if it still exists."""
        try:
            self.root.attributes("-topmost", False)
        except tk.TclError:
            pass  # Teardown may have destroyed the interpreter already.

    def pump(self, predicate, timeout=10):
        # This is a functional-state budget, not the product startup SLO.
        # Slow X11 dispatch can take >3s before a completed worker is observed;
        # record both first success and loop return rather than hiding latency.
        self.assertFalse(getattr(self, "_pump_active", False), "Nested TkCase.pump calls are not supported")
        started = time.monotonic()
        cpu_started = time.process_time()
        deadline = started + timeout
        self._pump_active = active = True
        matched = timed_out = False
        predicate_error = None
        timer = None
        deadline_state = None
        matched_at = None
        code = getattr(predicate, "__code__", None)
        observation = {"predicate": (f"{Path(code.co_filename).name}:{code.co_firstlineno}"
                                     if code is not None else type(predicate).__name__),
                       "timeout": timeout,
                       "since_setup": started - getattr(self, "_ui_started_at", started),
                       "ready_before": getattr(getattr(self, "app", None), "ready", None),
                       "loading_before": getattr(getattr(self, "app", None), "loading", None)}

        def snapshot():
            app = getattr(self, "app", None)
            pending = getattr(getattr(self, "controller", None), "pending", {})
            state = {name: getattr(app, name, None) for name in ("ready", "loading", "closing")}
            window = {}
            for name in ("state", "winfo_ismapped", "winfo_viewable", "winfo_geometry",
                         "winfo_reqwidth", "winfo_reqheight", "winfo_screenwidth", "winfo_screenheight"):
                method = getattr(self.root, name, None)
                if method is not None:
                    try:
                        window[name] = method()
                    except tk.TclError:
                        window[name] = "destroyed"
            for name in ("focus_get", "focus_displayof", "focus_lastfor"):
                method = getattr(self.root, name, None)
                if method is not None:
                    try:
                        focused = method()
                        window[name] = str(focused) if focused is not None else None
                    except tk.TclError:
                        window[name] = "destroyed"
            return {"app": state, "pending": sorted(str(key) for key in pending), "window": window}

        # Use the app's real event loop. update() can drain geometry events
        # indefinitely on macOS and prevent the Python timeout from running.
        def tick():
            nonlocal timer, matched, timed_out, predicate_error, deadline_state, matched_at
            timer = None
            if not active:
                return
            try:
                matched = bool(predicate())
                if matched:
                    matched_at = time.monotonic() - started
            except BaseException as error:
                predicate_error = error
                self.root.quit()
                return
            timed_out = not matched and time.monotonic() >= deadline
            if matched or timed_out or self.callback_errors:
                if timed_out:
                    deadline_state = snapshot()
                self.root.quit()
            else:
                timer = self.root.after(5, tick)

        try:
            timer = self.root.after(0, tick)
            self.root.mainloop()
        finally:
            active = self._pump_active = False
            if timer is not None:
                try:
                    self.root.after_cancel(timer)
                except tk.TclError:
                    pass  # Shutdown may have destroyed the interpreter/window.
            observation.update(elapsed=time.monotonic() - started,
                               process_cpu=time.process_time() - cpu_started,
                               matched_at=matched_at, timed_out=timed_out,
                               ready_after=getattr(getattr(self, "app", None), "ready", None),
                               loading_after=getattr(getattr(self, "app", None), "loading", None))
            if not hasattr(self, "_ui_waits"):
                self._ui_waits = []
            # Diagnostics stay bounded even if a failing test loops repeatedly.
            if len(self._ui_waits) < 256:
                self._ui_waits.append(observation)
        if predicate_error is not None:
            raise predicate_error
        self.assertEqual(self.callback_errors, [])
        # A shutdown callback can destroy the root before the next tick runs.
        # Otherwise, preserve success observed inside Tk: other callbacks may
        # enqueue work while quit unwinds, making an already-met predicate false.
        if not matched and not timed_out:
            matched = bool(predicate())
            if matched:
                observation["matched_at"] = time.monotonic() - started
        observation["matched"] = matched
        if not matched:
            final_state = snapshot()
            reason = "Timed out waiting for UI state" if timed_out else "Tk event loop exited before UI state was reached"
            self.fail(f"{reason}; elapsed={time.monotonic() - started:.3f}s, limit={timeout}s, "
                      f"app={final_state['app']}, pending={final_state['pending']}, "
                      f"deadline_state={deadline_state}, final_state={final_state}")

    def wait_mapped(self, widgets, timeout=10):
        """Wait for explicit visible targets and their ancestors, not hidden tabs."""
        nodes = []
        seen = set()
        for widget in widgets:
            while widget is not None:
                if id(widget) not in seen:
                    seen.add(id(widget))
                    nodes.append(widget)
                if widget is self.root:
                    break
                widget = widget.master

        def geometry():
            rows = []
            for widget in nodes:
                try:
                    rows.append({"widget": str(widget), "viewable": bool(widget.winfo_viewable()),
                                 "width": widget.winfo_width(), "height": widget.winfo_height()})
                except tk.TclError:
                    rows.append({"widget": str(widget), "viewable": False, "width": 0, "height": 0})
            return rows

        observed = []

        def mapped():
            nonlocal observed
            observed = geometry()
            return all(row["viewable"] and row["width"] > 1 and row["height"] > 1 for row in observed)

        try:
            self.pump(mapped, timeout=timeout)
        except AssertionError as error:
            raise self.failureException(f"{error}; observed widget/ancestor geometry={observed}; "
                                        f"final widget/ancestor geometry={geometry()}") from error

    def assert_arrow_hit_regions(self, control, arrows):
        observed = {}

        def native_layout_ready():
            width, height = control.winfo_width(), control.winfo_height()
            observed.update(widget=str(control), viewable=bool(control.winfo_viewable()),
                            size=(width, height), requested=(control.winfo_reqwidth(), control.winfo_reqheight()),
                            state=control.state(), style=str(control.cget('style')),
                            root_geometry=self.root.winfo_geometry(),
                            center=control.identify(width // 2, height // 2))
            return observed['viewable'] and width > 1 and height > 1 and bool(observed['center'])

        # Tk's mapped window can precede its idle-time element placement. Wait
        # for a placed native layout, then independently require every arrow.
        try:
            self.pump(native_layout_ready)
        except AssertionError as error:
            raise self.failureException(f'{error}; native hit layout={observed}') from error
        regions = {control.identify(x, y) for x in range(control.winfo_width())
                   for y in range(control.winfo_height())}
        for arrow in arrows:
            self.assertTrue(any(region.endswith(arrow) for region in regions), (arrow, regions, observed))

    def tearDown(self):
        if self.backend.close_gate:
            self.backend.close_gate.set()
        if self.backend.read_gate:
            self.backend.read_gate.set()
        if self.controller and not self.controller.stopped:
            if hasattr(self, "app"):
                self.app.close()
            else:
                self.controller.close(lambda: None, lambda error: None)
            self.pump(lambda: self.controller.stopped, timeout=5)
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        gc.collect()


class ControllerTests(TkCase):
    def open_controller(self):
        self.controller = Controller(self.root, lambda: self.backend)
        opened = []
        self.controller.call("open", "open", opened.append, self.fail)
        self.pump(lambda: bool(opened))

    def test_calls_are_off_thread_callbacks_are_on_tk_and_duplicates_are_bounded(self):
        self.open_controller()
        main = threading.get_ident()
        gate = threading.Event()
        self.backend.read_gate = gate
        callbacks = []
        self.assertTrue(self.controller.call("read", "list_documents", lambda result: callbacks.append(
            threading.get_ident()), self.fail))
        self.assertFalse(self.controller.call("read", "list_documents", self.fail, self.fail))
        self.pump(self.backend.read_started.is_set)
        self.assertEqual(callbacks, [])
        gate.set()
        self.pump(lambda: bool(callbacks))
        self.assertEqual(callbacks, [main])
        self.assertNotEqual(self.backend.calls[0][3], main)

    def test_close_during_startup_waits_and_suppresses_late_callbacks(self):
        gate = threading.Event()
        def factory():
            gate.wait(3)
            return self.backend
        self.controller = Controller(self.root, factory)
        callbacks = []
        self.controller.call("open", "open", callbacks.append, callbacks.append)
        self.controller.close(lambda: callbacks.append("closed"), self.fail)
        self.assertFalse(self.controller.call("read", "list_documents", self.fail, self.fail))
        gate.set()
        self.pump(lambda: self.controller.stopped)
        self.assertEqual(callbacks, ["closed"])
        self.assertTrue(self.backend.closed)

    def test_failed_close_can_retry_without_accepting_new_work(self):
        self.open_controller()
        self.backend.fail_close = True
        errors = []
        self.controller.close(self.fail, errors.append)
        self.pump(lambda: bool(errors))
        self.assertFalse(self.controller.stopped)
        self.assertFalse(self.controller.call("read", "list_documents", self.fail, self.fail))
        self.controller.close(lambda: None, self.fail)
        self.pump(lambda: self.controller.stopped)


class WindowTests(TkCase):
    def open_window(self, factory=None):
        self.app = Application(self.root, factory or (lambda: self.backend))
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready and not self.app.loading and self.app.total == 65)

    def test_sort_resets_page_and_applies_to_whole_result(self):
        self.open_window()
        self.app.change_page(1)
        self.pump(lambda: not self.app.loading)
        self.app.sort_choice.set("Название: Я → А")
        self.app.change_sort()
        self.pump(lambda: not self.app.loading)
        self.assertEqual(self.app.offset, 0)
        expected = sorted(self.backend.records, key=lambda r: r.document.title.casefold(), reverse=True)
        self.assertEqual(list(self.app.documents), [r.document_key for r in expected[:50]])
        self.app.change_page(1)
        self.pump(lambda: not self.app.loading)
        self.assertEqual(list(self.app.documents), [r.document_key for r in expected[50:]])
        self.assertIn("Я → А", self.app.scope.get())

    def test_repeat_prefills_form_without_submitting(self):
        self.open_window()
        job = Record(request=Record(topic="robotics", from_date=date(2023, 1, 1),
                                    until_date=date(2024, 1, 1), max_results=75, source="openalex"))
        self.app.jobs = {"repeat": job}
        self.app.job_tree.insert("", "end", iid="repeat")
        self.app.job_tree.selection_set("repeat")
        self.app.repeat_job()
        self.assertEqual(self.app.fields["topic"].get(), "robotics")
        self.assertEqual(self.app.fields["start"].get(), "2023-01-01")
        self.assertEqual(self.app.fields["end"].get(), "2024-01-01")
        self.assertEqual(self.app.fields["limit"].get(), "75")
        self.assertTrue(self.app.source_vars["openalex"].get())
        self.assertFalse(self.app.source_vars["crossref"].get())
        self.assertEqual(self.app.tabs.select(), str(self.app.collection_tab))
        self.assertNotIn("start", self.controller.pending)
        self.app.jobs = {}

    def test_offline_page_card_search_and_empty_result(self):
        self.open_window()
        self.assertEqual(len(self.app.documents), 50)
        self.app.change_page(1)
        self.pump(lambda: not self.app.loading)
        self.assertEqual(len(self.app.documents), 15)
        key = next(iter(self.app.documents))
        self.app.document_tree.selection_set(key)
        self.pump(lambda: "Аннотация отсутствует" in self.app.detail.get("1.0", "end"))
        self.assertIn("Аннотация отсутствует", self.app.detail.get("1.0", "end"))
        self.assertIn("2024 (год)", self.app.detail.get("1.0", "end"))
        self.app.search.insert(0, "Исследование 64")
        self.app.search_documents()
        self.pump(lambda: not self.app.loading)
        self.assertEqual(self.app.total, 1)
        self.assertEqual(self.app.offset, 0)
        self.app.search.delete(0, "end")
        self.app.search.insert(0, "не существующий документ")
        self.app.search_documents()
        self.pump(lambda: not self.app.loading)
        self.assertEqual(self.app.total, 0)
        self.assertIn("Ничего не найдено", self.app.detail.get("1.0", "end"))
        self.assertTrue(self.app.open_link.instate(["disabled"]))

    def test_slow_old_search_cannot_replace_newer_search(self):
        self.open_window()
        gate = threading.Event()
        self.backend.read_gate = gate
        self.backend.read_started.clear()
        self.app.search.insert(0, "Исследование 1")
        self.app.search_documents()
        self.pump(self.backend.read_started.is_set)
        self.app.search.delete(0, "end")
        self.app.search.insert(0, "Исследование 64")
        self.app.search_documents()
        gate.set()
        self.pump(lambda: not self.app.loading)
        self.assertEqual(list(self.app.documents), ["doi:10.1234/64"])

    def test_read_failure_keeps_saved_cards_and_hides_private_exception(self):
        self.open_window()
        self.backend.read_error = True
        self.app.refresh_documents()
        self.pump(lambda: not self.app.loading)
        self.assertEqual(len(self.app.documents), 50)
        self.assertNotIn("private", self.app.status.get())
        self.backend.read_error = False
        self.app.refresh_documents()
        self.pump(lambda: not self.app.loading)
        self.assertEqual(self.app.total, 65)

    def test_form_errors_and_unconfigured_epo(self):
        self.open_window()
        self.app.start_collection()
        self.assertTrue(self.app.field_errors["topic"].get())
        self.assertTrue(self.app.source_checks["epo"].instate(["disabled"]))
        self.assertNotIn("start", self.controller.pending)

    def test_partial_failure_and_cancel_keep_documents_available(self):
        self.backend.jobs = [Record(id="job1", request=Record(topic="test", source="crossref"),
                                   state="running", updated_at=1, scanned=5, stored=5, skipped=0,
                                   total_available=20, coverage_complete=False,
                                   error_message=None, error_code=None)]
        self.open_window()
        self.pump(lambda: "job1" in self.app.jobs)
        self.app.cancel_job()
        self.pump(lambda: self.backend.jobs[0].state == "cancelled" and "cancel" not in self.controller.pending)
        self.assertEqual(len(self.app.documents), 50)
        self.app._show_job()
        self.pump(lambda: not self.app.loading)
        self.assertEqual(self.backend.calls[-1][2], "job1")
        self.backend.jobs[0].state = "failed"
        self.backend.jobs[0].updated_at += 1
        self.backend.jobs[0].error_message = "Источник недоступен"
        self.backend.jobs[0].error_code = "source_unavailable"
        self.app._request_jobs()
        self.pump(lambda: "source_unavailable" in self.app.job_detail.get())
        self.assertEqual(len(self.app.documents), 50)

    def test_close_keeps_event_loop_responsive_until_backend_stops(self):
        self.open_window()
        gate = self.backend.close_gate = threading.Event()
        self.app.close()
        ticks = []
        self.root.after(20, lambda: ticks.append(True))
        self.pump(lambda: bool(ticks))
        self.assertFalse(self.controller.stopped)
        gate.set()
        self.pump(lambda: self.controller.stopped)

    def test_startup_failure_can_retry(self):
        attempts = []
        def factory():
            attempts.append(1)
            if len(attempts) == 1:
                raise ImportError("missing dependency")
            return self.backend
        self.app = Application(self.root, factory)
        self.controller = self.app.controller
        self.pump(lambda: "Не установлены" in self.app.status.get())
        self.app._retry_startup()
        self.pump(lambda: self.app.ready and not self.app.loading)
        self.assertEqual(self.app.total, 65)


class PresentationTests(unittest.TestCase):
    def test_invalid_dates_limits_sources_and_epo_query(self):
        _, errors = collection_values('topic "quoted"', "2026-02-30", "2027-01-01", "1.5", ("epo",),
                                      today=date(2026, 9, 6))
        self.assertEqual(set(errors), {"topic", "start", "end", "limit"})
        _, errors = collection_values("valid topic", "2025-01-01", "2024-01-01", "200", ())
        self.assertEqual(set(errors), {"start", "sources"})

    def test_optional_dates_and_partial_precision_are_not_invented(self):
        values, errors = collection_values(" valid topic ", "", "", "200", ("crossref",),
                                           today=date(2026, 9, 6))
        self.assertFalse(errors)
        self.assertIsNone(values["from_date"])
        self.assertEqual(values["until_date"], date(2026, 9, 6))
        self.assertEqual(publication_date(document(1).document), "2024 (год)")
        self.assertIn("неполная", job_state(Record(state="succeeded", coverage_complete=False)))
        self.assertNotIn("secret", error_message(RuntimeError("secret")))


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("httpx", "pydantic", "defusedxml")),
                     "Production backend dependencies are not installed")
class BackendIntegrationTests(TkCase):
    def test_form_real_backend_save_close_and_offline_reopen(self):
        from app.backend.config import BackendSettings
        from app.backend.contracts import DocumentRecord, SourcePage
        from app.backend.service import Backend

        class Provider:
            def iter_pages(self, request, cancel):
                doc = DocumentRecord(source=request.source, source_id="test1", doi="10.1234/ui-test",
                                     title="Документ из формы", url="https://doi.org/10.1234/ui-test")
                yield SourcePage(documents=(doc,), scanned=1, total_available=1, exhausted=True)

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            settings = BackendSettings(data_dir=Path(directory))
            self.app = Application(self.root, lambda: Backend(settings, provider_factory=Provider))
            self.controller = self.app.controller
            self.pump(lambda: self.app.ready and not self.app.loading)
            self.app.fields["topic"].insert(0, "Тестовый сбор")
            self.app.source_vars["openalex"].set(True)
            self.app.start_collection()
            self.pump(lambda: len(self.app.jobs) == 2 and all(j.state == "succeeded"
                                                            for j in self.app.jobs.values()), timeout=5)
            self.pump(lambda: not self.app.loading and self.app.total == 1)
            self.app.close()
            self.pump(lambda: self.controller.stopped)

            def no_network():
                raise AssertionError("Offline reading must never construct a provider")

            with Backend(settings, provider_factory=no_network) as backend:
                page = backend.list_documents(query="формы")
                self.assertEqual(page.total, 1)
                self.assertEqual(set(page.items[0].sources), {"crossref", "openalex"})


if __name__ == "__main__":
    unittest.main()
