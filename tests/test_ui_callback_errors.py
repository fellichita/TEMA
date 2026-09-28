"""Unexpected presentation errors restore operation state and do not stop polling."""

from types import SimpleNamespace

from app.ui.controller import Controller


def test_render_exception_reaches_owner_failure_and_tk_reporter():
    observed, failures, reports = [], [], []
    scheduler = SimpleNamespace(after=lambda *_: None,
                                report_callback_exception=lambda *args: reports.append(args))
    controller = Controller(scheduler)
    controller._dispatch = lambda *_: "result"
    error = RuntimeError("test-only unexpected renderer failure")
    def render(_):
        raise error
    try:
        controller.call("broken", "read", render, failures.append)
        controller.pending["broken"][0].result(1)
        controller.call("healthy", "read", observed.append, failures.append)
        controller.pending["healthy"][0].result(1)
        controller._poll()
        assert failures == [error]
        assert observed == ["result"]
        assert len(reports) == 1 and reports[0][1] is error
        assert not controller.pending
    finally:
        controller.executor.shutdown(wait=True, cancel_futures=True)
        controller.ml_executor.shutdown(wait=True, cancel_futures=True)
        controller.read_executor.shutdown(wait=True, cancel_futures=True)


def test_broken_failure_callback_is_reported_without_stopping_other_deliveries():
    reports, observed = [], []
    scheduler = SimpleNamespace(after=lambda *_: None,
                                report_callback_exception=lambda *args: reports.append(args))
    controller = Controller(scheduler)
    def dispatch(method, *_):
        if method == "broken":
            raise ValueError("worker failure")
        return "healthy"
    def failed(_):
        raise RuntimeError("broken error renderer")
    controller._dispatch = dispatch
    try:
        controller.call("broken", "broken", observed.append, failed)
        controller.call("healthy", "healthy", observed.append, failed)
        controller.pending["healthy"][0].result(1)
        controller._poll()
        assert observed == ["healthy"]
        assert len(reports) == 1 and isinstance(reports[0][1], RuntimeError)
    finally:
        controller.executor.shutdown(wait=True, cancel_futures=True)
        controller.ml_executor.shutdown(wait=True, cancel_futures=True)
        controller.read_executor.shutdown(wait=True, cancel_futures=True)
