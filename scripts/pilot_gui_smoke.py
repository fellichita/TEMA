"""Real desktop replay of a saved pilot run, using an isolated data copy.

No test controller, fabricated result, network collection or paid model call is
used. SQLite backup reads a coherent source snapshot without modifying it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from app.identity import validate_data_dir
from app.sqlite_runtime import sqlite3
from app.ui.controller import create_backend
from app.ui.display import enable_high_dpi
from app.ui.viewport import ScrollViewport
from app.ui.window import Application


def descendants(widget: tk.Misc):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def wait_for_gui(root, predicate: Callable[[], bool], label: str, errors: list, timeout: float = 20) -> None:
    """Latch success inside Tk and clean up the one scheduled polling callback."""
    deadline = time.monotonic() + timeout
    matched = timed_out = False
    active = True
    timer = None
    predicate_error = None

    def tick():
        nonlocal matched, timed_out, timer, predicate_error
        timer = None
        if not active:
            return
        try:
            matched = bool(predicate())
        except Exception as error:
            predicate_error = error
        timed_out = time.monotonic() >= deadline
        if matched or predicate_error or errors or timed_out:
            root.quit()
        else:
            timer = root.after(20, tick)

    try:
        timer = root.after(0, tick)
        root.mainloop()
    finally:
        active = False
        if timer is not None:
            try:
                root.after_cancel(timer)
            except tk.TclError:
                pass
    if predicate_error is not None:
        raise predicate_error
    # Successful shutdown destroys Tk before another polling tick can run.
    if not matched and not timed_out and not errors:
        try:
            destroyed = not root.winfo_exists()
        except tk.TclError:
            destroyed = True
        if destroyed:
            matched = bool(predicate())
    if errors or not matched:
        reason = "deadline" if timed_out else "callback error" if errors else "event loop exited"
        raise RuntimeError(f"GUI smoke failed at {label}: {reason}")


def run(source: Path, output: Path, *, hold_open: float = 0) -> dict[str, Any]:
    source = validate_data_dir(source)
    output = validate_data_dir(output)
    if not (source / "pilot.sqlite3").is_file():
        raise ValueError("A previously collected pilot database is required")
    source_hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source.rglob("*.json") if "revisions" in path.parts or "checkpoints" in path.parts}
    started = time.monotonic()
    report: dict[str, Any] = {"created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "status": "failed", "release_acceptance": False,
        "scope": "real Application with saved public documents; no new collection or model call"}
    with tempfile.TemporaryDirectory(prefix="main2-проверка-интерфейса-") as temporary:
        data = Path(temporary) / "данные"
        data.mkdir()
        for name in ("revisions", "checkpoints"):
            shutil.copytree(source / name, data / name)
        if (source / "settings.json").is_file():
            shutil.copy2(source / "settings.json", data / "settings.json")
        with sqlite3.connect((source / "pilot.sqlite3").as_uri() + "?mode=ro", uri=True) as original:
            with sqlite3.connect(data / "pilot.sqlite3") as copied:
                original.backup(copied)
        enable_high_dpi()
        root = tk.Tk()
        errors = []
        root.report_callback_exception = lambda error_type, _value, _traceback: errors.append(error_type.__name__)
        app = None

        def wait_for(predicate: Callable[[], bool], label: str, timeout: float = 20) -> None:
            wait_for_gui(root, predicate, label, errors, timeout)

        def settle(targets=()) -> None:
            def ready():
                windows = [getattr(owner, "window", owner) for owner in app.child_windows]
                viewports = [app.pilot_tab]
                viewports += [widget for window in windows for widget in descendants(window)
                              if isinstance(widget, ScrollViewport)]
                # Tk may unmap controls entirely outside a scrolled canvas.
                # Require the viewport and the explicit revealed targets;
                # offscreen content is checked only after scrolling to it.
                widgets = [*windows, *targets]
                widgets += [widget for viewport in viewports for widget in
                            (viewport, viewport.canvas, viewport.content)]
                hidden = [{"widget": str(widget), "geometry": widget.winfo_geometry(),
                           "requested": [widget.winfo_reqwidth(), widget.winfo_reqheight()],
                           "viewable": widget.winfo_viewable()} for widget in widgets
                          if not (widget.winfo_viewable() and widget.winfo_width() > 1 and widget.winfo_height() > 1)]
                if hidden:
                    report["geometry_wait"] = {"hidden": hidden}
                    return False
                for viewport in viewports:
                    canvas, content = viewport.canvas, viewport.content
                    if (viewport.timer is not None
                            or content.winfo_width() != max(canvas.winfo_width(), content.winfo_reqwidth())
                            or content.winfo_height() != max(canvas.winfo_height(), content.winfo_reqheight())
                            or content.winfo_rootx() != canvas.winfo_rootx() - round(canvas.canvasx(0))
                            or content.winfo_rooty() != canvas.winfo_rooty() - round(canvas.canvasy(0))):
                        report["geometry_wait"] = {"viewport": str(viewport), "timer": viewport.timer,
                            "content": content.winfo_geometry(), "canvas": canvas.winfo_geometry(),
                            "actual_position": [content.winfo_rootx(), content.winfo_rooty()],
                            "expected_position": [canvas.winfo_rootx() - round(canvas.canvasx(0)),
                                                  canvas.winfo_rooty() - round(canvas.canvasy(0))]}
                        return False
                report.pop("geometry_wait", None)
                return True
            wait_for(ready, "geometry")

        try:
            app = Application(root, factory=lambda: create_backend(data))
            wait_for(lambda: app.ready, "application_ready")
            root.minsize(1, 1)  # Simulate a display smaller than the normal recommended minimum.
            root.geometry("900x650")
            app.tabs.select(app.pilot_tab)
            wait_for(lambda: app.pilot_panel.loaded, "pilot_status")
            panel = app.pilot_panel
            panel.history()
            wait_for(lambda: app.analysis_history.loaded, "history")
            history = app.analysis_history
            tree = history.tree
            saved = next((identifier for identifier, row in history.rows.items()
                          if row["state"] == "succeeded"), None)
            if saved is None:
                raise RuntimeError("Saved source contains no completed analysis")
            tree.selection_set(saved)
            history.open_button.invoke()
            wait_for(lambda: panel.payload is not None, "saved_result")
            settle()
            payload = panel.payload
            if payload is None:
                raise RuntimeError("Result did not remain visible")
            controls = []
            for name, widget in (("query", panel.query), ("start", panel.start_button),
                                 ("documents", panel.documents_button), ("export", panel.export_button)):
                app.pilot_tab.reveal(widget)
                settle([widget])
                canvas = app.pilot_tab.canvas
                x = widget.winfo_rootx() - canvas.winfo_rootx()
                y = widget.winfo_rooty() - canvas.winfo_rooty()
                report.setdefault("control_geometry", []).append({"name": name, "x": x, "y": y,
                    "size": [widget.winfo_width(), widget.winfo_height()],
                    "viewport": [canvas.winfo_width(), canvas.winfo_height()],
                    "scrollregion": str(canvas.cget("scrollregion")), "view": canvas.yview()})
                if not (x < canvas.winfo_width() and y < canvas.winfo_height()
                        and x + widget.winfo_width() > 0 and y + widget.winfo_height() > 0):
                    raise RuntimeError("A control is unreachable in a small viewport")
                controls.append(name)
            child_count = len(app.child_windows)
            panel.documents()
            wait_for(lambda: len(app.child_windows) > child_count, "documents")
            documents_window = app.child_windows[-1]
            box = next(item for item in descendants(documents_window) if isinstance(item, tk.Text))
            visible_documents = box.get("1.0", "end-1c")
            report["document_page_has_sources"] = "https://" in visible_documents
            report["document_window_title"] = documents_window.title()
            documents_window.destroy()
            if panel.cards:
                tree = panel.tree if panel.tree.get_children() else panel.other_tree
                panel.result_tabs.select(0 if tree is panel.tree else 1)
                first = tree.get_children()[0]
                tree.selection_set(first)
                panel.passport()
                settle()
                passport_window = app.child_windows[-1]
                passport_window.geometry("900x650")
                settle()
                labels = [str(item.cget("text")) for item in descendants(passport_window) if isinstance(item, ttk.Label)]
                evidence = panel.cards[first]["evidence"]
                report["passport"] = {"opened": True, "evidence": len(evidence),
                    "exact_quotes_visible": all(any(item["quote"] in label for label in labels) for item in evidence),
                    "source_urls_visible": all(item["source_url"] in labels for item in evidence),
                    "history_chart_canvases": sum(isinstance(item, tk.Canvas) for item in descendants(passport_window)) - 1}
                if not report["passport"]["exact_quotes_visible"] or not report["passport"]["source_urls_visible"]:
                    raise RuntimeError("Passport failed to render exact evidence")
            else:
                report["passport"] = {"opened": False, "reason": "honest_empty_result"}
            after_hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in source.rglob("*.json") if "revisions" in path.parts or "checkpoints" in path.parts}
            if source_hashes != after_hashes:
                raise RuntimeError("The original saved documents changed")
            report.update(status="passed", run_id=saved, cards=len(panel.cards),
                assessments=len(payload.get("assessments", [])), quality=payload["result"]["quality"],
                callback_errors=errors, original_archive_unchanged=True,
                main_window=[root.winfo_width(), root.winfo_height()], reachable_controls=controls,
                viewport=[app.pilot_tab.canvas.winfo_width(), app.pilot_tab.canvas.winfo_height()],
                tabs=len(app.tabs.tabs()), elapsed_seconds=round(time.monotonic() - started, 3))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"status": "passed", "report": str(output), "holding_seconds": hold_open}), flush=True)
            if hold_open:
                done = []
                root.after(round(hold_open * 1000), lambda: done.append(True))
                wait_for(lambda: bool(done), "visual_inspection", timeout=hold_open + 5)
        finally:
            try:
                if app is not None:
                    app.close()
                    wait_for(lambda: app.controller.stopped, "clean_shutdown", timeout=15)
                    report["clean_shutdown"] = True
            except Exception as error:
                report["shutdown_error_type"] = type(error).__name__
                report["status"] = "failed"
                raise
            finally:
                try:
                    root.destroy()
                except tk.TclError:
                    pass
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hold-open", type=float, default=0)
    args = parser.parse_args()
    if not 0 <= args.hold_open <= 60:
        parser.error("Visual inspection hold must be between 0 and 60 seconds")
    run(args.source, args.output, hold_open=args.hold_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
