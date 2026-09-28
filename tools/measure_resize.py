"""Gate mapped Documents resize in fresh Tk processes with 50 synthetic rows.

Run in a quiet native GUI session (or Xvfb plus a ready window manager):
python -m tools.measure_resize --samples 5 --output build/resize-measurements.json
This measures source rendering at 100% UI scale; it does not measure startup,
backend speed, real-library content, other DPI settings, or frozen packaging.
"""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import time


def heartbeat_delay(observations: list[tuple[float, float]], started: float, ended: float) -> float:
    """Include a callback still waiting when the measured phase ends."""
    return max((max(0.0, min(observed, ended) - max(due, started))
                for due, observed in observations if due <= ended and observed >= started), default=0.0)


def violations(sample: dict, *, wall: float, cpu: float, lag: float) -> list[str]:
    """Evaluate actual elapsed time even when Tk finally returned success."""
    failures = []
    for name, limit in (('wall_seconds', wall), ('cpu_seconds', cpu), ('max_event_loop_lag_seconds', lag)):
        value = sample.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value > limit:
            failures.append(f'{name} exceeds {limit:g}s or is invalid')
    if sample.get('rows') != 50:
        failures.append('resize workload must contain 50 rows')
    if type(sample.get('callback_errors')) is not int or sample['callback_errors'] != 0:
        failures.append('Tk callback errors must be zero')
    if sample.get('initial_geometry') == sample.get('geometry'):
        failures.append('host window geometry did not allow an actual resize')
    return failures


def measure_one() -> dict:
    # Reuse the strict mapped-window scenario; this intentionally depends on
    # the offline test fixture, never a real profile, credential store or API.
    from tests.ui.test_visible_pages import VisiblePagesTests

    case = VisiblePagesTests('test_minimum_window_at_100_percent')
    case.setUp()
    timer = None
    due = time.monotonic() + .02
    observations: list[tuple[float, float]] = []

    def heartbeat():
        nonlocal timer, due
        now = time.monotonic()
        observations.append((due, now))
        # The subprocess deadline bounds the run; cap retained diagnostics too.
        del observations[:-4096]
        due = now + .02
        timer = case.root.after(20, heartbeat)

    try:
        timer = case.root.after(20, heartbeat)
        case.start(1)
        phase = case.resize_observation
        observations.append((due, time.monotonic()))
        result = {key: value for key, value in phase.items() if key not in {'started', 'ended'}}
        result.update(wall_seconds=phase['ended'] - phase['started'],
                      max_event_loop_lag_seconds=heartbeat_delay(observations, phase['started'], phase['ended']),
                      tk=case.root.tk.call('package', 'provide', 'Tk'),
                      tcl=case.root.tk.call('info', 'patchlevel'),
                      windowing_system=case.root.tk.call('tk', 'windowingsystem'),
                      scale=1, callback_errors=len(case.callback_errors), peak_rss_mib=None)
        try:
            import resource
            maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result['peak_rss_mib'] = maximum / (1024**2 if sys.platform == 'darwin' else 1024)
        except ImportError:
            pass  # Windows does not expose POSIX getrusage; report unavailable.
        return result
    finally:
        if timer is not None:
            case.root.after_cancel(timer)
        case.tearDown()


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('budget must be finite and positive')
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--wall-budget', type=positive, default=2.0)
    parser.add_argument('--cpu-budget', type=positive, default=1.0)
    parser.add_argument('--lag-budget', type=positive, default=1.0)
    parser.add_argument('--output', type=Path, default=Path('build/resize-measurements.json'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(measure_one()), flush=True)
        return 0
    if not 5 <= args.samples <= 100:
        parser.error('Use between 5 and 100 fresh processes')
    from tools.runtime_environment import diagnostic_environment
    from tools.run_checks import run_process, source_environment, source_fingerprint

    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    environment = source_environment()
    samples = []
    errors = []
    with tempfile.TemporaryDirectory(prefix='resize-logs-', dir=destination.parent) as logs:
        for number in range(args.samples):
            log = Path(logs) / f'{number}.log'
            record = run_process([sys.executable, '-B', '-m', 'tools.measure_resize', '--worker'],
                                 log, 45, env=diagnostic_environment())
            if record['returncode'] != 0 or record['timed_out']:
                failed = destination.with_suffix('.failed.log')
                failed.write_bytes(log.read_bytes())
                errors.append({**record, 'log': str(failed)})
                break
            try:
                sample = json.loads(log.read_text().splitlines()[-1])
            except (IndexError, ValueError):
                errors.append({'reason': 'worker returned no valid measurement', 'sample': number})
                break
            sample['violations'] = violations(sample, wall=args.wall_budget, cpu=args.cpu_budget, lag=args.lag_budget)
            samples.append(sample)
            print(f'Resize {number + 1}/{args.samples}: wall {sample["wall_seconds"]:.3f}s, '
                  f'CPU {sample["cpu_seconds"]:.3f}s, loop lag {sample["max_event_loop_lag_seconds"]:.3f}s', flush=True)
    unchanged = environment['source_tree_sha256'] == source_fingerprint()
    passed = (len(samples) == args.samples and not errors and unchanged
              and all(not sample['violations'] for sample in samples))
    report = {'environment': environment, 'scope': __doc__, 'samples': samples,
              'sample_count': len(samples), 'requested_samples': args.samples, 'errors': errors,
              'budgets_seconds': {'wall': args.wall_budget, 'cpu': args.cpu_budget, 'event_loop_lag': args.lag_budget},
              'source_unchanged': unchanged, 'passed': passed, 'release_acceptance': False,
              'limitations': ['Synthetic 50-row source-runtime Documents page at 100% UI scale.',
                              'Requires a quiet CPU and native display or ready Xvfb/window manager.',
                              'Peak RSS is for the process, including imports/startup, and unavailable on Windows.',
                              'All samples must meet each budget; late functional success cannot override it.']}
    if samples:
        report['p50_wall_seconds'] = statistics.median(sample['wall_seconds'] for sample in samples)
        report['max_wall_seconds'] = max(sample['wall_seconds'] for sample in samples)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'Resize gate {"PASS" if passed else "FAIL"}: {destination}', flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
