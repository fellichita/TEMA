"""Startup diagnostics are bounded evidence, not a replacement acceptance gate."""

import cProfile
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest

from tools.profile_gui_startup import diagnostic_complete, source_location, summarize_profile
from tools import profile_gui_startup


@pytest.mark.parametrize('version', [(3, 11), (3, 12)])
def test_older_cprofile_keeps_its_native_thread_local_profiling(monkeypatch, version):
    enabled = []
    monkeypatch.setattr(profile_gui_startup, 'sys', SimpleNamespace(version_info=version))
    profile_gui_startup.enable_current_thread_profile(SimpleNamespace(enable=lambda: enabled.append(True)))
    assert enabled == [True]


def test_profile_does_not_merge_worker_thread_into_gui_call_stack():
    completed = []

    def worker_only():
        completed.append(True)
        try:
            {}.pop('missing')
        except KeyError:
            pass

    def main_only():
        return len(completed)

    profile = cProfile.Profile()
    profile_gui_startup.enable_current_thread_profile(profile)
    try:
        worker = Thread(target=worker_only)
        worker.start()
        worker.join(2)
        for _ in range(5):
            main_only()
    finally:
        profile.disable()
    assert completed == [True]
    assert all(entry.code != worker_only.__code__ for entry in profile.getstats())
    assert summarize_profile(profile)['invalid_entries'] == []
    entry = next(entry for entry in profile.getstats() if entry.code == main_only.__code__)
    assert entry.callcount == 5 and entry.reccallcount == 0


def test_profile_source_locations_do_not_include_external_absolute_paths():
    code = compile('pass', '/private/user-home/application/module.py', 'exec')
    assert source_location(code) == ('module.py', 1, '<module>')


def test_profile_summary_has_bounded_tables_and_retains_cost_attribution():
    profile = cProfile.Profile()
    profile.runcall(sum, range(100))
    summary = summarize_profile(profile)
    assert all(len(rows) <= 30 for rows in summary.values())
    assert any('sum' in row['function'] for row in summary['top_cumulative'])
    assert all(not Path(row['source']).is_absolute() for row in summary['top_cumulative'])


def test_inconsistent_native_profile_is_explicit_in_diagnostic():
    entry = SimpleNamespace(code='<native fixture>', callcount=1, reccallcount=0,
                            totaltime=.1, inlinetime=.6)
    profile = SimpleNamespace(getstats=lambda: [entry])
    summary = summarize_profile(profile)
    assert summary['invalid_entries'] == summary['top_self']


@pytest.mark.parametrize('fail', [False, True])
def test_diagnostic_owns_and_removes_its_temporary_profile(monkeypatch, fail):
    directories = []

    def observed(mode, directory):
        assert directory.is_dir()
        (directory / 'fixture-only').write_text('synthetic')
        directories.append(directory)
        if fail:
            raise RuntimeError('synthetic setup failure')
        return {'mode': mode}

    monkeypatch.setattr(profile_gui_startup, '_profile_one', observed)
    if fail:
        with pytest.raises(RuntimeError, match='synthetic setup'):
            profile_gui_startup.profile_one('mapped')
    else:
        assert profile_gui_startup.profile_one('mapped') == {'mode': 'mapped'}
    assert len(directories) == 1 and not directories[0].exists()


@pytest.mark.parametrize('failure', ['timeout', 'exit', 'missing', 'callback', 'source-change'])
def test_diagnostic_does_not_claim_completion_after_worker_failure(failure):
    samples = [{'mode': mode, 'completed': True, 'callback_errors': 0} for mode in ('mapped', 'withdrawn')]
    records = [{'returncode': 0, 'timed_out': False} for _ in range(2)]
    if failure == 'timeout':
        records[0]['timed_out'] = True
    elif failure == 'exit':
        records[0]['returncode'] = 1
    elif failure == 'missing':
        samples.pop()
    elif failure == 'callback':
        samples[0]['callback_errors'] = 1
    assert not diagnostic_complete(samples, records, failure != 'source-change')


def test_both_modes_must_complete_without_callback_errors():
    samples = [{'mode': mode, 'completed': True, 'callback_errors': 0} for mode in ('mapped', 'withdrawn')]
    records = [{'returncode': 0, 'timed_out': False} for _ in range(2)]
    assert diagnostic_complete(samples, records, True)
