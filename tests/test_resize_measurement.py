"""The performance gate must reject late success and stalled callbacks."""

import pytest

from tools.measure_resize import heartbeat_delay, violations


def sample(**updates):
    return {'wall_seconds': .05, 'cpu_seconds': .02, 'max_event_loop_lag_seconds': .01,
            'rows': 50, 'callback_errors': 0,
            'initial_geometry': [1240, 850], 'geometry': [1000, 680], **updates}


def test_completed_resize_after_budget_is_rejected():
    failed = violations(sample(wall_seconds=14.8, cpu_seconds=14.2), wall=2, cpu=1, lag=1)
    assert len(failed) == 2
    assert any('wall_seconds' in message for message in failed)
    assert any('cpu_seconds' in message for message in failed)


def test_stalled_heartbeat_spanning_the_phase_is_measured():
    lag = heartbeat_delay([(1.01, 1.02), (1.04, 15.8), (15.82, 15.83)], 1.03, 15.80)
    assert lag == pytest.approx(14.76)
    assert violations(sample(max_event_loop_lag_seconds=lag), wall=2, cpu=1, lag=1)


def test_waiting_callback_at_phase_end_cannot_disappear_from_lag():
    assert heartbeat_delay([(1, 1.1), (1.12, 4)], 1.2, 3) == pytest.approx(1.8)
    assert heartbeat_delay([(1, 2), (5, 6)], 3, 4) == 0


@pytest.mark.parametrize('field,value', [('wall_seconds', float('nan')), ('cpu_seconds', -1),
                                        ('max_event_loop_lag_seconds', float('inf')), ('rows', 0),
                                        ('callback_errors', 1), ('callback_errors', None)])
def test_invalid_or_empty_measurements_fail(field, value):
    assert violations(sample(**{field: value}), wall=2, cpu=1, lag=1)


def test_identical_geometry_is_not_a_resize():
    assert violations(sample(initial_geometry=[1000, 680]), wall=2, cpu=1, lag=1)


def test_valid_sample_meets_explicit_budgets():
    assert violations(sample(), wall=2, cpu=1, lag=1) == []
