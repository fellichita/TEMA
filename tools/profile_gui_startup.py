"""Diagnose synthetic Tk startup in mapped and withdrawn fresh processes.

This is an instrumented diagnostic, not a startup performance acceptance gate.
It uses FakeBackend's 65 documents without profiles, credentials or network.
cProfile attributes elapsed time on the Tk thread; process CPU is reported
separately and includes all threads. Native Tk internals need native sampling.
"""

import argparse
import cProfile
from collections.abc import Callable
import json
import math
from pathlib import Path
import sys
import tempfile
from threading import Lock, get_ident
import time
from types import CodeType

from tools.run_checks import REPO, run_process, source_environment, source_fingerprint


def enable_current_thread_profile(profile: cProfile.Profile) -> None:
    # CPython 3.13 cProfile uses interpreter-wide monitoring callbacks but one
    # stack. Exclude worker events before they can corrupt the GUI call tree.
    # Use public monitoring registration, retaining cProfile's own callbacks.
    owner = get_ident()
    profile.enable()
    if sys.version_info < (3, 13):
        # Older cProfile versions already limit events to the enabling thread.
        return
    monitoring = sys.monitoring
    events = monitoring.get_events(monitoring.PROFILER_ID)
    if not events:
        return
    # Replacing callbacks while events run would split call/return pairs.
    monitoring.set_events(monitoring.PROFILER_ID, 0)
    callbacks = events
    if events & monitoring.events.CALL:
        # These are implicitly activated by CALL, but absent from get_events().
        callbacks |= monitoring.events.C_RETURN | monitoring.events.C_RAISE

    def on_owner(callback: Callable[..., object]) -> Callable[..., object]:
        def observed(*args: object) -> object:
            if get_ident() == owner:
                return callback(*args)
            return None
        return observed

    while callbacks:
        event = callbacks & -callbacks
        callbacks ^= event
        callback = monitoring.register_callback(monitoring.PROFILER_ID, event, None)
        if callback is not None:
            monitoring.register_callback(monitoring.PROFILER_ID, event, on_owner(callback))
    profile.clear()
    monitoring.set_events(monitoring.PROFILER_ID, events)


def source_location(code: CodeType | str) -> tuple[str, int, str]:
    if isinstance(code, str):
        return '<native>', 0, ' '.join(code.split())[:160]
    path = Path(code.co_filename)
    for prefix in (REPO, Path(sys.prefix), Path(sys.base_prefix)):
        try:
            source = path.relative_to(prefix).as_posix()
            break
        except ValueError:
            source = path.name
    return source[:200], code.co_firstlineno, code.co_name[:160]


def summarize_profile(profile: cProfile.Profile) -> dict[str, list[dict[str, object]]]:
    entries = []
    for entry in profile.getstats():
        source, line, function = source_location(entry.code)
        row = {'source': source, 'line': line, 'function': function,
               'calls': entry.callcount, 'recursive_calls': entry.reccallcount,
               'self_seconds': entry.inlinetime, 'cumulative_seconds': entry.totaltime}
        entries.append((entry.totaltime, entry.inlinetime, row))
    cumulative = sorted(entries, key=lambda entry: entry[0], reverse=True)
    own = sorted(entries, key=lambda entry: entry[1], reverse=True)
    invalid = [row for total, own, row in entries
               if not math.isfinite(total) or not math.isfinite(own) or own < 0 or total + 1e-9 < own]
    return {'top_cumulative': [entry[2] for entry in cumulative[:30]],
            'top_self': [entry[2] for entry in own[:30]],
            'tk_calls': [entry[2] for entry in cumulative if '_tkinter' in str(entry[2]['function'])][:30],
            'invalid_entries': invalid[:30]}


def profile_one(mode: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix='gui-startup-profile-') as directory:
        return _profile_one(mode, Path(directory))


def _profile_one(mode: str, directory: Path) -> dict[str, object]:
    from app.ui.window import Application
    from tests.ui.test_desktop import TkCase

    class StartupCase(TkCase):
        app: Application

    case = StartupCase('runTest')
    case.setUp()
    case.backend.settings.data_dir = directory
    profile = cProfile.Profile()
    calls: list[dict[str, object]] = []
    lock = Lock()
    started, cpu_started = time.monotonic(), time.process_time()

    def instrument(name: str) -> None:
        method = getattr(case.backend, name)

        def observed(*args: object, **kwargs: object) -> object:
            entered, cpu = time.monotonic(), time.thread_time()
            try:
                return method(*args, **kwargs)
            finally:
                with lock:
                    if len(calls) < 256:
                        calls.append({'method': name, 'started_seconds': entered - started,
                                      'returned_seconds': time.monotonic() - started,
                                      'wall_seconds': time.monotonic() - entered,
                                      'thread_cpu_seconds': time.thread_time() - cpu})

        setattr(case.backend, name, observed)

    for name in ('sources', 'list_documents', 'list_jobs', 'list_history'):
        instrument(name)
    result: dict[str, object] = {'mode': mode, 'fixture_documents': len(case.backend.records),
                                'functional_timeout_seconds': 10, 'completed': False,
                                'phase_started_since_setup_seconds': started - case._ui_started_at}
    stage = 'construction'
    enable_current_thread_profile(profile)
    try:
        app = Application(case.root, lambda: case.backend, ui_scale=1)
        case.app = app
        case.controller = app.controller
        result['construction_wall_seconds'] = time.monotonic() - started
        result['construction_cpu_seconds'] = time.process_time() - cpu_started
        if mode == 'mapped':
            case.root.deiconify()
        stage = 'initial_data'
        case.pump(lambda: app.ready and not app.loading and len(app.document_tree.get_children()) == 50)
        result['initial_data_ready_at_seconds'] = time.monotonic() - started
        if mode == 'mapped':
            stage = 'mapped_geometry'
            case.wait_mapped([case.root, app.document_tree])
        result['completed'] = True
    except Exception as error:
        result.update(failed_stage=stage, error_type=type(error).__name__)
    finally:
        profile.disable()
        statistics = summarize_profile(profile)
        if statistics['invalid_entries']:
            result.update(completed=False, profile_warning='Inconsistent cProfile attribution')
        result.update(wall_seconds=time.monotonic() - started,
                      cpu_seconds=time.process_time() - cpu_started,
                      callback_errors=len(case.callback_errors), profile=statistics,
                      ui_waits=getattr(case, '_ui_waits', []))
        with lock:
            result['backend_calls'] = list(calls)
        try:
            result['window'] = {'state': case.root.state(), 'mapped': bool(case.root.winfo_ismapped()),
                                'geometry': case.root.winfo_geometry(),
                                'requested': [case.root.winfo_reqwidth(), case.root.winfo_reqheight()],
                                'screen': [case.root.winfo_screenwidth(), case.root.winfo_screenheight()],
                                'tk': case.root.tk.call('package', 'provide', 'Tk'),
                                'tcl': case.root.tk.call('info', 'patchlevel'),
                                'windowing_system': case.root.tk.call('tk', 'windowingsystem')}
        finally:
            try:
                case.tearDown()
            except Exception as error:
                result.update(completed=False, cleanup_error_type=type(error).__name__)
    return result


def diagnostic_complete(samples: list[dict], records: list[dict], unchanged: bool) -> bool:
    return (unchanged and len(samples) == len(records) == 2
            and {sample.get('mode') for sample in samples} == {'mapped', 'withdrawn'}
            and all(sample.get('completed') is True and sample.get('callback_errors') == 0 for sample in samples)
            and all(record['returncode'] == 0 and not record['timed_out'] for record in records))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('build/checks/startup-profile.json'))
    parser.add_argument('--worker', choices=('mapped', 'withdrawn'), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        sample = profile_one(args.worker)
        destination.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return 0 if sample['completed'] else 1
    from tools.runtime_environment import diagnostic_environment

    environment = source_environment()
    directory = destination.parent / (destination.stem + '-workers')
    directory.mkdir(parents=True, exist_ok=True)
    samples, records = [], []
    for mode in ('withdrawn', 'mapped'):
        output, log = directory / f'{mode}.json', directory / f'{mode}.log'
        output.unlink(missing_ok=True)
        record = run_process([sys.executable, '-B', '-m', 'tools.profile_gui_startup',
                              '--worker', mode, '--output', str(output)], log, 60, env=diagnostic_environment())
        records.append(record)
        if output.is_file():
            try:
                sample = json.loads(output.read_text(encoding='utf-8'))
                if isinstance(sample, dict):
                    samples.append(sample)
            except ValueError:
                pass
        print(f'Startup profile {mode}: process {record["elapsed_seconds"]:.3f}s, '
              f'exit {record["returncode"]}, timeout {record["timed_out"]}', flush=True)
    unchanged = environment['source_tree_sha256'] == source_fingerprint()
    completed = diagnostic_complete(samples, records, unchanged)
    report = {'environment': environment, 'scope': __doc__, 'samples': samples, 'records': records,
              'source_unchanged': unchanged, 'completed': completed, 'release_acceptance': False,
              'limitations': ['cProfile overhead is included; timings are diagnostic, not acceptance measurements.',
                              'The main-thread profile cannot resolve work inside native Tk/Windows frames.',
                              'Backend calls use only a synthetic in-memory corpus of 65 documents.',
                              'Each worker has a 60-second process deadline; functional waits retain 10 seconds.']}
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return 0 if completed else 1


if __name__ == '__main__':
    raise SystemExit(main())
