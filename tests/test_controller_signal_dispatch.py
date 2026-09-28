"""The actual desktop controller admits every signal-window service call."""

from types import SimpleNamespace

import pytest

from app.pilot.service import PilotService
from app.ui.controller import AUXILIARY_METHODS, Controller
from tests.test_controller_lanes import Scheduler


_SIGNAL_METHODS = (
    "create_signal_query", "preview_signal_csv", "import_signal_csv", "import_signal_atom",
    "fetch_signal_wordstat", "start_signals", "list_signal_runs", "signal_result", "export_signal", "import_signal",
    "signal_associations", "confirm_signal_grant", "signal_arxiv_candidates", "link_signal_arxiv",
    "signal_scientific_runs", "signal_scientific_cards", "signal_finding_evidence",
    "signal_scenario", "signal_compare", "signal_largest_event_options",
    "signal_watch_state", "set_signal_watch",
)


@pytest.mark.parametrize("method", _SIGNAL_METHODS)
def test_signal_window_operation_is_routed_to_real_service_method(method: str) -> None:
    controller = Controller(Scheduler())
    controller.backend = SimpleNamespace()
    calls = []

    class RecordingPilot:
        def __getattr__(self, name):
            assert hasattr(PilotService, name)
            def record(*args, **kwargs):
                calls.append((name, args, kwargs))
                return name
            return record

    controller.pilot = RecordingPilot()
    try:
        assert "pilot_" + method in AUXILIARY_METHODS
        assert controller._dispatch("pilot_" + method, ("value",), {}) == method
        assert calls == [(method, ("value",), {})]
    finally:
        controller.executor.shutdown(wait=True, cancel_futures=True)
        controller.ml_executor.shutdown(wait=True, cancel_futures=True)
        controller.read_executor.shutdown(wait=True, cancel_futures=True)
