"""Offline runtime checks for the application, using isolated temporary profiles.

All mutable smoke data live in a fresh temporary Unicode directory. The optional
snapshot/model are read-only inputs. Reports contain versions and exception types,
never environment values, document contents or credential-bearing tracebacks.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib
import importlib.metadata
from io import BytesIO
import json
import multiprocessing
from multiprocessing.connection import Connection
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
from threading import Event
import time
from typing import Any, Callable, Iterator
from uuid import uuid4

REQUIRED_RUNTIME_CHECKS: tuple[str, ...] = ("sqlite", "resources", "native_dependencies", "spawn", "backup_restore")


def _install_network_denial():
    """Block Python IPv4/IPv6 connects and DNS in this process."""
    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def deny_connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("Network access during offline model smoke")
        return original_connect(self, address)

    def deny_connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("Network access during offline model smoke")
        return original_connect_ex(self, address)

    def deny_dns(*args, **kwargs):
        raise RuntimeError("DNS access during offline model smoke")

    socket.socket.connect = deny_connect
    socket.socket.connect_ex = deny_connect_ex
    socket.getaddrinfo = deny_dns
    return original_connect, original_connect_ex, original_getaddrinfo


def _network_denial_verified() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.connect(("127.0.0.1", 9))
    except RuntimeError as error:
        return str(error) == "Network access during offline model smoke"
    except OSError:
        return False
    return False


@contextmanager
def _deny_external_network() -> Iterator[None]:
    previous = os.environ.get("TRENDANALYZER_SMOKE_OFFLINE")
    original_connect, original_connect_ex, original_getaddrinfo = _install_network_denial()
    os.environ["TRENDANALYZER_SMOKE_OFFLINE"] = "1"
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        socket.getaddrinfo = original_getaddrinfo
        if previous is None:
            os.environ.pop("TRENDANALYZER_SMOKE_OFFLINE", None)
        else:
            os.environ["TRENDANALYZER_SMOKE_OFFLINE"] = previous


def sqlite_is_patched(version: str) -> bool:
    """Require the current security floor, including the FTS5 and WAL fixes."""
    try:
        major, minor, patch = (int(part) for part in version.split("."))
    except (ValueError, TypeError):
        return False
    from app.sqlite_runtime import is_patched

    return is_patched((major, minor, patch))


def _sqlite_check(directory: Path) -> dict[str, Any]:
    from app.sqlite_runtime import MIN_SQLITE_VERSION, sqlite3

    database = directory / "проверка базы.sqlite3"
    with sqlite3.connect(database) as connection:
        journal = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        connection.execute("PRAGMA synchronous=FULL")
        synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
        connection.execute("CREATE VIRTUAL TABLE documents USING fts5(title)")
        connection.execute("INSERT INTO documents VALUES (?)", ("квантовая память",))
        found = connection.execute(
            "SELECT count(*) FROM documents WHERE documents MATCH ?", ("квантовая",)
        ).fetchone()[0]
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if journal != "wal" or synchronous != 2 or found != 1 or integrity != "ok":
        raise RuntimeError("SQLite WAL/FULL/FTS5 integrity smoke failed")
    security_fixes = sqlite_is_patched(sqlite3.sqlite_version)
    return {"version": sqlite3.sqlite_version, "wal": True, "synchronous": "FULL", "fts5": True,
            "integrity": integrity, "wal_reset_fix": security_fixes,
            "minimum_secure_version": ".".join(map(str, MIN_SQLITE_VERSION)),
            "security_fixes": security_fixes}


def _backup_restore_check(directory: Path, pilot_result: Path | None = None) -> dict[str, Any]:
    """Exercise durable APIs in an owned profile; never open the user's library."""
    from datetime import date

    from app.backend.contracts import DocumentRecord, SearchRequest, SourcePage
    from app.backend.history import HistoryRequest, HistoryStore
    from app.backend.repository import Repository
    from app.identity import APP_ID
    from app.pilot.archive import DocumentArchive
    from app.pilot.contracts import content_hash
    from app.pilot.library import ResultLibrary
    from app.pilot.service import PilotService
    from app.pilot.settings import PilotSettings, load_settings, save_settings
    from app.profiles import activate_profile, resolve_profile
    from app.runtime.backup import BackupSession, create_backup, restore_backup, unpack_package
    from app.runtime.jobs import Coordinator
    from app.runtime.credentials import CredentialStore

    def package_digest() -> str | None:
        if pilot_result is None:
            return None
        with pilot_result.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    original_digest = package_digest()
    with tempfile.TemporaryDirectory(prefix="backup-restore-", dir=directory) as temporary:
        scratch = Path(temporary)
        source = scratch / "исходный профиль"
        repository = Repository(source / "documents.sqlite3")
        history = HistoryStore(repository)
        history_id = history.create(HistoryRequest(topic="robot control", from_date=date(2025, 1, 1),
            until_date=date(2025, 1, 31), sources=("openalex",), auto_split=False))
        history.state(history_id, "running")
        period = history.report(history_id).periods[0]
        archive = DocumentArchive(source / "revisions")
        documents = tuple(DocumentRecord(source="openalex", source_id="W1", title="Synthetic robot control record",
            abstract=f"Synthetic robot control measurement, revision {index}; проверка точного текста.",
            publication_year=2025, date_precision="year", url="https://example.org/robot-control")
            for index in (1, 2))
        references = []
        for index, document in enumerate(documents):
            job = (history.begin_attempt(history_id, period.id) if index == 0 else
                   repository.create_job(SearchRequest(topic="robot control", source="openalex")))
            repository.start_job(job.id)
            repository.ingest_page(job.id, SourcePage(documents=(document,), scanned=1, exhausted=True))
            repository.finish_job(job.id, "succeeded")
            references.append(archive.put(document))
        history.state(history_id, "succeeded")
        settings = PilotSettings(external_ai_allowed=False, run_cost_micro=0, day_cost_micro=0, discovery_documents=100)
        save_settings(source, settings)
        checkpoint = {"references": [reference.model_dump(mode="json") for reference in references]}
        result = {"retained": checkpoint["references"], "label": "Synthetic offline smoke; not a scientific assessment"}

        def process(context, _payload):
            context.checkpoint("retrieval", checkpoint)
            return result

        coordinator = Coordinator(source, process)
        try:
            run_id = coordinator.submit({"query": "Synthetic robot control smoke", "external_ai_allowed": False})
            coordinator.wait(timeout=5)
            expected_runs = coordinator.list_runs()
            if len(expected_runs) != 1 or expected_runs[0]["state"] != "succeeded" or coordinator.result(run_id) != result:
                raise RuntimeError("Synthetic durable run did not finish")
        finally:
            coordinator.close(timeout=5)

        # Exercise the same no-key signal import and immutable result that a
        # colleague can open after restoring a private profile backup.
        signal_source = scratch / "wordstat-smoke.csv"
        rows = ["Месяц;Запросов;Доля"]
        for year in (2025, 2026):
            for month in range(1, 13):
                if year == 2026 and month > 8:
                    break
                recent = year == 2026 and month in (6, 7, 8)
                rows.append(f"{month:02d}.{year};{60 if recent else 30};"
                            f"{'0,20%' if recent else '0,10%'}")
        signal_source.write_bytes(("\n".join(rows) + "\n").encode("utf-8-sig"))
        signal_service = PilotService(source, CredentialStore())
        try:
            created = signal_service.create_signal_query(
                "Молекулярная память", "Хранение данных в молекулах", "ДНК память")
            mapping = dict(date_column="Месяц", count_column="Запросов", share_column="Доля",
                           date_format="MM.YYYY", share_unit="percent", phrase="ДНК память",
                           expected_from="2025-01", expected_to="2026-08")
            receipt = signal_service.import_signal_csv(created["query_profile_hash"], str(signal_source),
                                                        "wordstat", mapping, "utf-8-sig", ";",
                                                        retention_confirmed=True)
            signal_run = signal_service.start_signals(
                created["query_profile_hash"], (created["concept_hash"],),
                wordstat_receipt_hash=receipt["receipt_hash"])
            signal_service.coordinator.wait(timeout=20)
            signal_profile = signal_service.signal_result(signal_run)["profile"]
            if (signal_profile["findings"][0]["search_state"] != "sustained_growth"
                    or len(signal_profile["attention_ids"]) != 1):
                raise RuntimeError("Offline signal import did not produce a verified review card")
            if not signal_service.set_signal_watch(signal_run, created["concept_id"], True)["watched"]:
                raise RuntimeError("Explicit signal watch choice was not stored")
        finally:
            signal_service.close()

        expected_page = repository.list_documents()
        document_key = expected_page.items[0].document_key
        expected_versions = repository.list_document_versions(document_key)
        expected_jobs = repository.list_jobs()
        expected_job_pages = {job.id: repository.list_documents(job_id=job.id) for job in expected_jobs}
        expected_history = history.report(history_id)
        expected_history_page = repository.list_documents(history_id=history_id)
        if (expected_page.total != 1 or expected_versions.total != 2 or len(expected_jobs) != 2
                or expected_history.total_periods != 1 or not expected_history.coverage_complete
                or expected_page.items[0].document != documents[1]
                or expected_history_page.total != 1 or expected_history_page.items[0].document != documents[0]):
            raise RuntimeError("Synthetic source profile did not retain its two document versions and history")
        imported = (ResultLibrary(source, archive).import_file(pilot_result) if pilot_result is not None else None)
        with BackupSession(source) as session:
            saved = create_backup(session, scratch / "backups")
        with unpack_package(saved.path, expected_kind="trendanalizer-backup") as (_, manifest):
            if manifest.application_id != APP_ID:
                raise RuntimeError("Backup application identity changed")
        restored = restore_backup(saved.path, scratch / "восстановленный профиль")
        anchor = scratch / "выбранная библиотека"
        if activate_profile(anchor, restored) is not None:
            raise RuntimeError("Restored profile selection could not be durably saved")

        def no_recompute(_context, _payload):
            raise RuntimeError("Reopening a saved profile must not start work")

        for _ in range(2):
            # Fresh API instances after all previous coordinator handles close.
            if resolve_profile(anchor) != restored:
                raise RuntimeError("Restored profile selection changed")
            reopened = Repository(restored / "documents.sqlite3")
            if (reopened.list_documents() != expected_page
                    or reopened.list_document_versions(document_key) != expected_versions
                    or reopened.list_jobs() != expected_jobs
                    or any(reopened.list_documents(job_id=job_id) != page for job_id, page in expected_job_pages.items())
                    or HistoryStore(reopened).report(history_id) != expected_history
                    or reopened.list_documents(history_id=history_id) != expected_history_page
                    or load_settings(restored) != settings):
                raise RuntimeError("Restored documents, versions, collection history or settings changed")
            restored_archive = DocumentArchive(restored / "revisions")
            if any(restored_archive.get(reference.revision_id) != document
                   for reference, document in zip(references, documents, strict=True)):
                raise RuntimeError("Restored immutable document identity changed")
            reader = Coordinator(restored, no_recompute)
            try:
                if (reader.list_runs(analyses_only=True) != expected_runs or reader.result(run_id) != result
                        or reader.checkpoint_value(run_id, "retrieval") != checkpoint):
                    raise RuntimeError("Restored analysis history or checkpoint changed")
            finally:
                reader.close(timeout=5)
            restored_service = PilotService(restored, CredentialStore())
            try:
                restored_profile = restored_service.signal_result(signal_run)["profile"]
                if (restored_profile != signal_profile or
                        restored_service.signal_scenario(signal_run, "exclude_wordstat")["attention_after"] != 0 or
                        not restored_service.signal_watch_state(signal_run, created["concept_id"])["watched"]):
                    raise RuntimeError("Restored signal profile or offline scenario changed")
            finally:
                restored_service.close()
            if imported is not None:
                library = ResultLibrary(restored, restored_archive)
                if library.count() != 1 or content_hash(library.read(imported["id"])) != content_hash(imported["payload"]):
                    raise RuntimeError("Restored imported result identity or assessed data changed")
                library.clear_cache()
        # Reacquiring both service locks also detects leaked coordinator ownership.
        with BackupSession(restored):
            pass
        if package_digest() != original_digest:
            raise RuntimeError("Original result package changed during backup verification")
        details = {"application_id": manifest.application_id, "backup_sha256": saved.sha256,
                   "documents": expected_page.total, "document_revisions": expected_versions.total,
                   "collection_jobs": len(expected_jobs), "history_periods": expected_history.total_periods,
                   "analysis_runs": len(expected_runs), "reopen_count": 2,
                   "profile_selection_reopened": True, "imported_result_verified": imported is not None,
                   "signal_profile_restored": True, "signal_scenario_offline": True,
                   "signal_watch_restored": True,
                   "input_package_sha256": original_digest, "no_network_or_credentials_required": True}
    if scratch.exists():
        raise RuntimeError("Backup smoke temporary profile was not removed")
    return details | {"temporary_data_removed": True}


def _resources_check() -> dict[str, Any]:
    from app.ml.local_encoder import load_spec
    from app.ml.directions import direction_profile
    from app.ui import fonts
    from app.pilot.encoder import load_spec as load_pilot_spec

    directory = Path(fonts.__file__).parent / "assets" / "fonts"
    names = ("OpenSans-Regular.ttf", "OpenSans-Medium.ttf", "OpenSans-SemiBold.ttf", "OpenSans-Bold.ttf")
    if any(not (directory / name).is_file() for name in (*names, "OFL.txt")):
        raise FileNotFoundError("Application font resources are incomplete")
    if direction_profile("photonic neuromorphic computing") is None:
        raise ValueError("Legacy direction resources are missing")
    spec = load_spec()
    pilot_spec = load_pilot_spec()
    return {"fonts": len(names), "font_license": True, "encoder_manifest": spec["model_id"],
            "pilot_encoder_manifest": pilot_spec["model_id"], "pilot_encoder_revision": pilot_spec["revision"],
            "legacy_direction_resources": True}


# The graphics build of the runtime carries the same module and version under a
# different distribution name; the report must name whichever is installed.
_EQUIVALENT_DISTRIBUTIONS = {"onnxruntime": ("onnxruntime-gpu",)}


def _runtime_version(name: str) -> str:
    for candidate in (name, *_EQUIVALENT_DISTRIBUTIONS.get(name, ())):
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    raise RuntimeError(f"Runtime package {name} is not installed")


def _native_check() -> dict[str, Any]:
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    import numpy as np
    import onnxruntime as ort  # type: ignore[import-untyped]  # Vendor ships no type stubs.
    from scipy import special  # type: ignore[import-untyped]  # SciPy stubs are a separate package.
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from tokenizers import Tokenizer
    from pypdf import PdfReader, PdfWriter

    ort.disable_telemetry_events()
    matrix = TfidfVectorizer().fit_transform(["quantum memory", "optical memory"])
    if matrix.shape != (2, 3) or not np.isfinite(matrix.data).all():
        raise RuntimeError("Native numeric runtime calculation failed")
    factors = NMF(n_components=2, init="nndsvda", random_state=42, max_iter=500).fit_transform(
        np.array([[1., 2., 0.], [2., 1., 0.], [0., 0., 2.]])
    )
    if factors.shape != (3, 2) or not np.isfinite(factors).all() or special.ndtr(0) != 0.5:
        raise RuntimeError("Native clustering or SciPy special runtime failed")
    if "CPUExecutionProvider" not in ort.get_available_providers() or not callable(Tokenizer.from_file):
        raise RuntimeError("CPU inference or tokenizer runtime is unavailable")
    pdf_bytes = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(pdf_bytes)
    pdf_bytes.seek(0)
    if len(PdfReader(pdf_bytes).pages) != 1:
        raise RuntimeError("PDF dependency cannot reopen a page")
    keyring = {"darwin": ("keyring.backends.macOS", "Keyring"),
               "win32": ("keyring.backends.Windows", "WinVaultKeyring")}.get(sys.platform)
    if keyring is not None:
        if not callable(getattr(importlib.import_module(keyring[0]), keyring[1], None)):
            raise RuntimeError("Native keyring backend is unavailable")
    return {"packages": {name: _runtime_version(name)
                         for name in ("numpy", "scipy", "scikit-learn", "onnxruntime", "tokenizers", "pypdf", "keyring")},
            "numeric_calculation": True, "nmf_factorization": True, "scipy_special": True,
            "onnx_cpu_provider": True, "pdf_page_reopened": True,
            "native_keyring_backend_imported": keyring[0] if keyring else None,
            "keyring_read_or_write_performed": False}


def _spawn_worker(connection: Connection) -> None:
    try:
        details = _native_check()
        details.update({"pid": os.getpid(), "parent_pid": os.getppid(),
                        "tk_loaded": "tkinter" in sys.modules})
        connection.send({"status": "passed", "details": details})
    except Exception as error:
        connection.send({"status": "failed", "error_type": type(error).__name__})
    finally:
        connection.close()


def _spawn_check(timeout: float = 45) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_spawn_worker, args=(writer,), name="trendanalyser-runtime-smoke")
    started = False
    try:
        process.start()
        started = True
        writer.close()
        if not reader.poll(timeout):
            raise TimeoutError("Spawn worker did not finish within the runtime smoke limit")
        report = reader.recv()
        process.join(timeout=5)
        if process.exitcode != 0 or report["status"] != "passed":
            raise RuntimeError("Spawn worker did not exit successfully")
        details = report["details"]
        if details["pid"] == os.getpid() or details["parent_pid"] != os.getpid() or details["tk_loaded"]:
            raise RuntimeError("Spawn isolation failed or the worker imported Tk")
        return {"start_method": "spawn", "separate_process": True, "tk_imported_in_worker": False,
                "native_dependencies": details["packages"], "exitcode": process.exitcode}
    finally:
        writer.close()
        reader.close()
        if started:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            process.close()


class GuiSmokeFailure(RuntimeError):
    """Fixed stage identifiers make failures actionable without recording UI data."""

    def __init__(self, stage: str, callback_type: str | None = None):
        if stage not in {"application_ready", "tab_navigation", "pilot_status", "result_import",
                         "passport", "shutdown", "credential_isolation"}:
            raise ValueError("Unknown GUI smoke stage")
        self.stage = stage
        self.callback_type = (callback_type if callback_type and len(callback_type) <= 100
                              and callback_type.isascii() and callback_type.isidentifier() else None)
        super().__init__("Native GUI smoke did not complete its requested stage")


@contextmanager
def _isolated_gui_credentials() -> Iterator[None]:
    """Use the real OS backend against an empty smoke-only service name.

    The CLI runs in its own process. Refuse to replace an existing application
    session, and never query its native keychain entries or change persisted
    configuration. Diagnostics must not prompt for the owner's API key.
    """
    from app.runtime import credentials, session

    if session._credentials is not None:
        raise GuiSmokeFailure("credential_isolation")
    original_namespace = credentials.KEYRING_NAMESPACE
    credentials.KEYRING_NAMESPACE = original_namespace + ".runtime-smoke." + uuid4().hex
    try:
        yield
    finally:
        if session._credentials is not None:
            session._credentials.close()
            session._credentials = None
        credentials.KEYRING_NAMESPACE = original_namespace


def _gui_check(directory: Path, pilot_result: Path | None = None) -> dict[str, Any]:
    with _isolated_gui_credentials():
        details = _gui_check_in_process(directory, pilot_result)
    return details | {"credential_namespace": "isolated_temporary", "real_user_keys_read": False}


def _gui_check_in_process(directory: Path, pilot_result: Path | None = None) -> dict[str, Any]:
    import tkinter as tk
    from app.ui.display import enable_high_dpi
    from app.ui.controller import create_backend
    from app.ui.window import Application
    from scripts.pilot_gui_smoke import wait_for_gui

    enable_high_dpi()
    root = tk.Tk()
    errors: list[str] = []
    root.report_callback_exception = lambda error_type, value, traceback: errors.append(error_type.__name__)
    application = None

    def wait_for(predicate: Callable[[], bool], stage: str, timeout: float = 20) -> None:
        try:
            wait_for_gui(root, predicate, stage, errors, timeout)
        except Exception:
            raise GuiSmokeFailure(stage, errors[0] if errors else None) from None

    def mapped(page: tk.Misc) -> bool:
        return bool(page.winfo_viewable() and page.winfo_width() > 1 and page.winfo_height() > 1)

    try:
        application = Application(root, factory=lambda: create_backend(directory / "данные приложения"))
        wait_for(lambda: application.ready, "application_ready")
        tabs = len(application.tabs.tabs())
        for identifier in application.tabs.tabs():
            application.tabs.select(identifier)
            page = root.nametowidget(identifier)
            wait_for(partial(mapped, page), "tab_navigation")
        if errors:
            raise GuiSmokeFailure("tab_navigation", errors[0])
        panel = application.pilot_panel
        wait_for(lambda: panel.loaded, "pilot_status")
        details = {"tk_version": str(root.tk.call("package", "provide", "Tk")),
                   "tcl_version": str(root.tk.call("info", "patchlevel")), "application_ready": True,
                   "tabs_opened": tabs, "callback_errors": 0, "pilot_status_loaded": True}
        if pilot_result is not None:
            application.controller.call("native-smoke-pilot-import", "pilot_import_result", panel._imported,
                lambda error: errors.append(type(error).__name__), str(pilot_result.resolve()))
            wait_for(lambda: panel.payload is not None, "result_import", timeout=120)
            payload = panel.payload
            identifier = panel.run_id
            if payload is None or identifier is None or not identifier.startswith("import-"):
                raise RuntimeError("Native result did not open through the real import flow")
            expected_top = payload["result"].get("top_trend_ids")
            if expected_top is None:
                expected_top = [card["candidate"]["candidate_id"] for card in payload["result"]["cards"]
                                if card["category"] == "confirmed_trend"][:15]
            shown_top, shown_other = panel.tree.get_children(), panel.other_tree.get_children()
            if (shown_top != tuple(expected_top) or set(shown_top) & set(shown_other)
                    or set((*shown_top, *shown_other)) != set(panel.cards)):
                raise RuntimeError("Native TOP/order or separate candidate table differs from the imported result")
            details.update(pilot_imported_cards=len(panel.cards),
                pilot_top_ids=list(shown_top), pilot_other_candidates=len(shown_other),
                pilot_imported_assessments=len(payload.get("assessments", [])),
                pilot_imported_quality=payload["result"]["quality"], pilot_passport_opened=False)
            if panel.cards:
                tree = panel.tree if shown_top else panel.other_tree
                panel.result_tabs.select(0 if shown_top else 1)
                tree.selection_set(tree.get_children()[0])
                panel.passport()
                wait_for(lambda: bool(application.child_windows) and application.child_windows[-1].winfo_viewable(), "passport")
                details["pilot_passport_opened"] = True
        return details
    finally:
        if application is not None:
            application.close()
            if not application.controller.stopped:
                wait_for(lambda: application.controller.stopped, "shutdown", timeout=15)
            if not application.controller.stopped:
                raise TimeoutError("Application did not release its backend on close")
        try:
            root.destroy()
        except tk.TclError:
            pass


def _snapshot_check(path: Path) -> dict[str, Any]:
    from app.ml.corpus import read_snapshot

    with path.open("rb") as stream:
        before = hashlib.file_digest(stream, "sha256").hexdigest()
    corpus = read_snapshot(path)
    with path.open("rb") as stream:
        after = hashlib.file_digest(stream, "sha256").hexdigest()
    if before != after:
        raise RuntimeError("Snapshot input changed during read-only smoke")
    return {"sha256": after, "documents": len(corpus["entries"]), "read_only": True}


def _model_check(directory: Path) -> dict[str, Any]:
    import numpy as np
    from app.ml.local_encoder import LocalEncoder

    encoder = LocalEncoder(directory)
    vectors = encoder.encode(["quantum memory", "оптическая память"], kind="query")
    if vectors.shape != (2, 384) or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise RuntimeError("Pinned encoder returned invalid vectors")
    return {"model_id": encoder.manifest()["model_id"], "rows": 2, "dimensions": 384,
            "finite_normalized_vectors": True, "diagnostic_vectors": vectors.tolist(),
            "python_network_guard_active": (_network_denial_verified() if
                                            os.environ.get("TRENDANALYZER_SMOKE_OFFLINE") == "1" else False)}


def _pilot_encoder_task(input_path: Path, output_path: Path, cancel) -> None:
    """Trusted spawn task; tiny hand-authored diagnostics are not a quality benchmark."""
    import numpy as np
    from app.pilot.encoder import MultilingualEncoder

    offline_guard = os.environ.get("TRENDANALYZER_SMOKE_OFFLINE") == "1"
    if offline_guard:
        _install_network_denial()
        if not _network_denial_verified():
            raise RuntimeError("Spawn worker network guard was not installed")

    value = json.loads(input_path.read_text(encoding="utf-8"))
    encoder = MultilingualEncoder(Path(value["model_dir"]), cancel=cancel)
    queries = encoder.encode(["квантовая память для фотонных кубитов",
                              "lithium-selective membranes for extraction from brines"], kind="query", cancel=cancel)
    passages = encoder.encode([
        "Long-lived quantum memory stores photonic qubits in rare earth doped crystals.",
        "Lithium selective membranes separate lithium ions from saline brines.",
        "Medieval ceramic bowls were catalogued by museum curators.",
    ], kind="passage", cancel=cancel)
    if (queries.shape != (2, 384) or passages.shape != (3, 384)
            or not np.isfinite(queries).all() or not np.isfinite(passages).all()
            or not np.allclose(np.linalg.norm(queries, axis=1), 1, atol=1e-5)
            or not np.allclose(np.linalg.norm(passages, axis=1), 1, atol=1e-5)):
        raise RuntimeError("Multilingual encoder produced invalid vectors")
    scores = queries @ passages.T
    if scores.argmax(axis=1).tolist() != [0, 1]:
        raise RuntimeError("Basic RU/EN diagnostic did not retrieve its matching sentence")
    chunked = encoder.chunk_text("квантовая память " * 900, cancel=cancel)
    if chunked.truncated or len(chunked.units) < 2 or any(item.token_count > 512 for item in chunked.units):
        raise RuntimeError("Token-limited multilingual chunking failed")
    write_report(output_path, {"model_id": encoder.spec["model_id"], "revision": encoder.spec["revision"],
        "encoder_fingerprint": encoder.fingerprint, "dimensions": 384,
        "finite_normalized_vectors": True, "basic_ru_en_matching": True,
        "diagnostic_similarity": scores.tolist(), "query_vectors": queries.tolist(),
        "passage_vectors": passages.tolist(), "chunk_tokens": [item.token_count for item in chunked.units],
        "pid": os.getpid(), "parent_pid": os.getppid(), "tk_imported_in_worker": "tkinter" in sys.modules,
        "python_network_guard_active": offline_guard,
        "scientific_quality_benchmark": False, "reference_pytorch_parity_tested": False})


def _pilot_model_check(model_dir: Path, scratch: Path) -> dict[str, Any]:
    from app.pilot.encoder import spec_fingerprint
    from app.runtime.credentials import CredentialStore
    from app.runtime.worker import run_in_process
    # When invoked with -m, obtain a stable importable descriptor instead of the
    # __main__ function that production worker validation deliberately rejects.
    from tools.runtime_smoke import _pilot_encoder_task as task

    input_path, output_path = scratch / "pilot-model-input.json", scratch / "pilot-model-output.json"
    write_report(input_path, {"model_dir": str(model_dir.resolve())})
    credentials = CredentialStore()
    try:
        result = run_in_process(task, input_path, output_path, Event(), credentials=credentials,
            timeout_seconds=90, max_input_bytes=10000, max_output_bytes=100000)
    finally:
        credentials.close()
    details = json.loads(output_path.read_text(encoding="utf-8"))
    if (details["pid"] == os.getpid() or details["parent_pid"] != os.getpid()
            or details["tk_imported_in_worker"] or details["encoder_fingerprint"] != spec_fingerprint()):
        raise RuntimeError("Multilingual spawn isolation or pinned model identity failed")
    details.update(start_method="spawn", production_worker=True, output_bytes=result.output_bytes,
                   output_sha256=result.sha256, implicit_downloads=False)
    return details


def _pilot_result_check(path: Path, scratch: Path) -> dict[str, Any]:
    from app.pilot.archive import DocumentArchive
    from app.pilot.contracts import content_hash
    from app.pilot.export import read_result_package
    from app.pilot.library import ResultLibrary
    from app.pilot.methodology import AssessmentArtifact

    with path.open("rb") as stream:
        before = hashlib.file_digest(stream, "sha256").hexdigest()
    with read_result_package(path) as package:
        result_hash = content_hash(package.result)
        assessment_hashes = [item.assessment.assessment_hash for item in package.assessments]
        details = {"schema_version": package.result.schema_version, "result_hash": result_hash,
            "cards": len(package.result.cards), "quality": package.result.quality,
            "assessments_replayed": len(assessment_hashes),
            "exact_quotations_verified": sum(len(card.evidence) for card in package.result.cards),
            "archived_revisions": len({item.revision_id for snapshot in package.result.snapshots for item in snapshot.documents})}
    directory = scratch / "повторное открытие результата"
    library = ResultLibrary(directory, DocumentArchive(directory / "revisions"))
    imported = library.import_file(path)
    reopened = library.read(imported["id"])
    if (content_hash(reopened["result"]) != result_hash
            or [AssessmentArtifact.model_validate(item).assessment.assessment_hash for item in reopened["assessments"]] != assessment_hashes
            or library.import_file(path)["id"] != imported["id"]):
        raise RuntimeError("Imported result did not reopen with identical assessed semantics")
    with path.open("rb") as stream:
        after = hashlib.file_digest(stream, "sha256").hexdigest()
    if before != after:
        raise RuntimeError("Original result package changed during verification")
    details.update(package_sha256=before, read_only_source=True, local_library_reopened=True,
                   duplicate_import_idempotent=True, no_model_or_network_required=True)
    return details


def _pilot_discovery_check(model_dir: Path, result_path: Path, scratch: Path) -> dict[str, Any]:
    from app.pilot.discovery import task
    from app.pilot.encoder import spec_fingerprint
    from app.pilot.export import read_result_package
    from app.runtime.credentials import CredentialStore
    from app.runtime.worker import run_in_process

    with read_result_package(result_path) as package:
        snapshot = next((item for item in package.result.snapshots if item.purpose == "discovery"), None)
        if snapshot is None or not snapshot.documents:
            raise ValueError("A real discovery snapshot with documents is required for the optional worker smoke")
        references = snapshot.documents[:32]
        plan = package.result.query_plan
        payload = {"query_plan": plan.model_dump(mode="json"), "discovery_snapshot_id": snapshot.snapshot_id,
            "model_dir": str(model_dir.resolve()), "cache_dir": str(scratch / "проверка кеша"),
            "documents": [package.archive.get(item.revision_id).model_dump(mode="json") for item in references]}
    input_path, output_path = scratch / "pilot-discovery-input.json", scratch / "pilot-discovery-output.json"
    write_report(input_path, payload)
    credentials = CredentialStore()
    try:
        worker = run_in_process(task, input_path, output_path, Event(), credentials=credentials,
            timeout_seconds=120, max_input_bytes=100_000_000, max_output_bytes=25_000_000)
    finally:
        credentials.close()
    result = json.loads(output_path.read_text(encoding="utf-8"))
    if (result.get("schema_version") != 3 or result.get("plan_hash") != plan.plan_hash
            or result.get("input_records") != len(references)
            or result.get("encoder_fingerprint") != spec_fingerprint()
            or result.get("methodology_calibrated") is not False
            or result.get("scope_review_required") is not True):
        raise RuntimeError("Pilot discovery worker returned inconsistent provenance or overstated quality")
    return {"input_records": result["input_records"], "unique_studies": result["unique_studies"],
        "text_units": result["text_units"], "candidates": len(result["candidates"]),
        "quality": result["quality"], "production_worker": True, "start_method": "spawn",
        "output_sha256": worker.sha256, "output_bytes": worker.output_bytes,
        "selection": "First at most 32 archived discovery revisions; bounded runtime proof, not full reanalysis",
        "scientific_quality_benchmark": False, "scope_review_required": result["scope_review_required"]}


def run_checks(*, gui: bool = False, model_dir: Path | None = None,
               snapshot: Path | None = None, pilot_model_dir: Path | None = None,
               pilot_result: Path | None = None) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    def check(name: str, function: Callable[[], dict[str, Any]]) -> None:
        started = time.monotonic()
        try:
            checks[name] = {"status": "passed", "details": function()}
        except Exception as error:
            checks[name] = {"status": "failed", "error_type": type(error).__name__}
            if isinstance(error, GuiSmokeFailure):
                checks[name]["stage"] = error.stage
                if error.callback_type:
                    checks[name]["callback_error_type"] = error.callback_type
        checks[name]["elapsed_seconds"] = round(time.monotonic() - started, 3)

    with tempfile.TemporaryDirectory(prefix="Trendanalyser проверка ") as temporary:
        directory = Path(temporary)
        check("sqlite", lambda: _sqlite_check(directory))
        check("resources", _resources_check)
        check("native_dependencies", _native_check)
        check("spawn", _spawn_check)
        check("backup_restore", lambda: _backup_restore_check(directory, pilot_result))
        if gui:
            check("gui", lambda: _gui_check(directory, pilot_result))
        else:
            checks["gui"] = {"status": "not_run", "reason": "GUI smoke was not requested"}
        if model_dir is not None:
            check("model", lambda: _model_check(model_dir))
        else:
            checks["model"] = {"status": "not_run", "reason": "No external model package supplied"}
        if snapshot is not None:
            check("snapshot", lambda: _snapshot_check(snapshot))
        else:
            checks["snapshot"] = {"status": "not_run", "reason": "No real read-only snapshot supplied"}
        if pilot_model_dir is not None:
            check("pilot_model_spawn", lambda: _pilot_model_check(pilot_model_dir, directory))
        else:
            checks["pilot_model_spawn"] = {"status": "not_run", "reason": "No external multilingual model supplied"}
        if pilot_result is not None:
            check("pilot_result", lambda: _pilot_result_check(pilot_result, directory))
        else:
            checks["pilot_result"] = {"status": "not_run", "reason": "No real v3 result package supplied"}
        if pilot_model_dir is not None and pilot_result is not None:
            check("pilot_discovery_spawn", lambda: _pilot_discovery_check(pilot_model_dir, pilot_result, directory))
        else:
            checks["pilot_discovery_spawn"] = {"status": "not_run", "reason": "Both a multilingual model and real v3 result are needed"}
    required = REQUIRED_RUNTIME_CHECKS
    runtime_passed = (all(checks.get(name, {}).get("status") == "passed" for name in required)
                      and all(item["status"] != "failed" for item in checks.values()))
    sqlite_fixed = checks.get("sqlite", {}).get("details", {}).get("security_fixes") is True
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "platform": sys.platform, "machine": platform.machine(), "python": platform.python_version(),
            "frozen": bool(getattr(sys, "frozen", False)), "checks": checks,
            "runtime_passed": runtime_passed, "sqlite_security_gate_passed": sqlite_fixed,
            "all_requested_checks_passed": runtime_passed and sqlite_fixed,
            "release_acceptance": False,
            "limitations": ["This local smoke does not prove clean-machine installation, code signing, "
                            "Windows acceptance, live APIs or field-pilot quality."]}


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline application runtime checks")
    parser.add_argument("--output", type=Path, required=True, metavar="REPORT_JSON")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--pilot-model-dir", type=Path, help="Separate pinned multilingual-e5-small model package")
    parser.add_argument("--pilot-result", type=Path, help="Real v3 .trendresult package for offline replay/import")
    arguments = parser.parse_args(argv)
    report = run_checks(gui=arguments.gui, model_dir=arguments.model_dir, snapshot=arguments.snapshot,
                        pilot_model_dir=arguments.pilot_model_dir, pilot_result=arguments.pilot_result)
    write_report(arguments.output.resolve(), report)
    return 0 if report["all_requested_checks_passed"] else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
