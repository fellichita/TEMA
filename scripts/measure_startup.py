"""Measure real Tk startup in fresh processes and isolated empty libraries.

These are source-runtime measurements on the current host with a warm OS cache,
not clean-machine installation, a cold reboot, or a populated-library benchmark.
"""

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time


def _fd_count() -> int | None:
    for path in ('/dev/fd', '/proc/self/fd'):
        try:
            with os.scandir(path) as entries:
                # Exclude the temporary descriptor used by this inspection.
                return max(0, sum(1 for _ in entries) - 1)
        except OSError:
            continue
    return None


@contextmanager
def _isolated_profile(parent: Path | None, report: dict):
    directory = None
    try:
        with tempfile.TemporaryDirectory(prefix='trendanalyser-startup-', dir=parent) as name:
            directory = Path(name)
            report['profile_directory'] = str(directory.resolve())
            try:
                yield directory
            finally:
                report['open_fds_after_shutdown'] = _fd_count()
    finally:
        report['profile_cleanup_complete'] = directory is None or not directory.exists()
        report['open_fds_after_cleanup'] = _fd_count()


def measure_one(profile_parent: Path | None = None) -> dict:
    started, cpu_started = time.perf_counter(), time.process_time()
    stages = []

    def memory():
        try:
            import resource
            maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return maximum / (1024**2 if sys.platform == 'darwin' else 1024)
        except ImportError:
            return None

    def stage(name):
        value = {'name': name, 'seconds': time.perf_counter() - started,
                 'peak_rss_mib': memory(), 'open_fds': _fd_count()}
        stages.append(value)
        print('STARTUP_STAGE ' + json.dumps(value), flush=True)

    stage('python')
    import tkinter as tk

    from app.ui.controller import create_backend
    from app.ui.display import enable_high_dpi
    from app.ui.window import Application

    stage('ui_imports')

    from tools.runtime_smoke import _isolated_gui_credentials

    callback_errors = []
    report = {'credential_namespace': 'isolated_temporary', 'real_user_keys_read': False}
    with _isolated_profile(profile_parent, report) as directory, _isolated_gui_credentials():
        enable_high_dpi()
        root = tk.Tk()
        stage('tk_root')
        root.report_callback_exception = lambda kind, _value, _trace: callback_errors.append(kind.__name__)
        app = None
        deadline = time.perf_counter() + 20
        ready = False
        delays = []
        previous = time.perf_counter()

        def tick():
            nonlocal ready, previous
            now = time.perf_counter()
            delays.append(max(0, now - previous - .02))
            previous = now
            ready = bool(app is not None and app.ready and not app.loading and root.winfo_viewable())
            if ready or callback_errors or now >= deadline:
                root.quit()
            else:
                root.after(20, tick)

        def open_backend():
            opening = time.perf_counter()
            backend = create_backend(directory)
            report['backend_construct_seconds'] = time.perf_counter() - opening
            stage('backend_opened')
            return backend

        try:
            root.after(20, tick)
            app = Application(root, factory=open_backend)
            stage('widgets_created')
            root.mainloop()
            if not ready or callback_errors:
                raise RuntimeError('Application startup did not become ready without callback errors')
            elapsed, cpu = time.perf_counter() - started, time.process_time() - cpu_started
            stage('ready')
            report.update({'ready_seconds': elapsed, 'cpu_seconds': cpu, 'peak_rss_mib': memory(),
                    'stages': stages,
                    'loaded_scientific_modules': [name for name in ('numpy', 'scipy', 'sklearn', 'onnxruntime')
                                                  if name in sys.modules],
                    'maximum_event_loop_delay_ms': max(delays, default=0) * 1000,
                    'callback_errors': callback_errors, 'tk': root.tk.call('package', 'provide', 'Tk'),
                    'tcl': root.tk.call('info', 'patchlevel')})
            return report
        finally:
            if app is not None:
                app.close()
                shutdown_deadline = time.perf_counter() + 15

                def shutdown_tick():
                    if app.controller.stopped or time.perf_counter() >= shutdown_deadline:
                        root.quit()
                    else:
                        root.after(20, shutdown_tick)

                root.after(20, shutdown_tick)
                root.mainloop()
                if not app.controller.stopped:
                    raise RuntimeError('Application did not finish shutdown')
            else:
                root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--output', type=Path, default=Path('build/startup-measurements.json'))
    parser.add_argument('--profile-parent', type=Path,
                        help='Create disposable empty profiles below this directory instead of OS temporary storage')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        import faulthandler

        faulthandler.enable()
        faulthandler.dump_traceback_later(10, repeat=True)
        try:
            print(json.dumps(measure_one(args.profile_parent)))
        finally:
            faulthandler.cancel_dump_traceback_later()
        return 0
    if not 5 <= args.samples <= 100:
        parser.error('Use between 5 and 100 fresh processes')
    from tools.run_checks import run_process, source_environment, source_fingerprint
    from tools.runtime_environment import diagnostic_environment

    environment = source_environment()
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    profile_parent = args.profile_parent.resolve() if args.profile_parent is not None else None
    if profile_parent is not None:
        profile_parent.mkdir(parents=True, exist_ok=True)
    samples = []
    with (tempfile.TemporaryDirectory(prefix='startup-logs-', dir=destination.parent) as logs,
          tempfile.TemporaryDirectory(prefix='startup-profiles-', dir=profile_parent) as profiles):
        for number in range(args.samples):
            path = Path(logs) / f'{number}.log'
            result = run_process([sys.executable, '-m', 'scripts.measure_startup', '--worker',
                                  '--profile-parent', profiles], path, 45, env=diagnostic_environment())
            if result['returncode'] != 0 or result['timed_out']:
                failed = destination.with_suffix('.failed.log')
                failed.write_bytes(path.read_bytes())
                destination.with_suffix('.failed.json').write_text(json.dumps(
                    {**result, 'log': str(failed), 'profile_parent': profiles}, indent=2) + '\n')
                raise RuntimeError(f'Startup measurement failed; see {failed}')
            samples.append(json.loads(path.read_text(encoding="utf-8").splitlines()[-1]))
            print(f'Startup sample {number + 1}/{args.samples}: {samples[-1]["ready_seconds"]:.3f}s', flush=True)
    values = sorted(sample['ready_seconds'] for sample in samples)
    report = {'environment': environment, 'samples': samples, 'sample_count': len(samples),
              'profile_parent': str(profile_parent) if profile_parent is not None else 'OS temporary directory',
              'all_profiles_removed': not Path(profiles).exists() and all(sample['profile_cleanup_complete'] for sample in samples),
              'p50_ready_seconds': statistics.median(values),
              'p95_ready_seconds': values[math.ceil(.95 * len(values)) - 1],
              'source_unchanged': environment['source_tree_sha256'] == source_fingerprint(),
              'scope': __doc__, 'release_acceptance': False,
              'limitations': ['Fresh source processes and empty profiles; OS/filesystem cache is warm.',
                              'Credential reads use a random empty native keyring namespace; no real user keys are read.',
                              'FD counts exclude the inspection descriptor; Tk/OS resources can remain until process exit.',
                              'This does not reproduce the old populated Documents library or prove its delay fixed.']}
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
