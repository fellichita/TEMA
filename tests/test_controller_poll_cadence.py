"""Idle UI work must not consume the same timer budget as active operations."""

from app.ui.controller import Controller


class Clock:
    def __init__(self):
        self.callbacks = {}
        self.next_id = 0

    def after(self, delay, callback):
        self.next_id += 1
        self.callbacks[self.next_id] = delay, callback
        return self.next_id

    def after_cancel(self, timer):
        del self.callbacks[timer]

    def run(self):
        timer = next(iter(self.callbacks))
        _, callback = self.callbacks.pop(timer)
        callback()

    def delay(self):
        assert len(self.callbacks) == 1
        return next(iter(self.callbacks.values()))[0]


def test_idle_poll_wakes_immediately_for_new_work_and_close():
    clock = Clock()
    controller = Controller(clock)
    controller._dispatch = lambda method, args, kwargs: method
    delivered = []
    try:
        assert clock.delay() == 500
        assert controller.call("work", "operation", delivered.append, delivered.append)
        assert clock.delay() == 50
        controller.pending["work"][0].result(timeout=1)
        clock.run()
        assert delivered == ["operation"]
        assert clock.delay() == 500
        controller.close(lambda: delivered.append("closed"), delivered.append)
        assert clock.delay() == 50
        controller.close_future.result(timeout=1)
        clock.run()
        assert delivered == ["operation", "closed"]
        assert clock.callbacks == {}
    finally:
        controller.executor.shutdown(wait=True, cancel_futures=True)
        controller.ml_executor.shutdown(wait=True, cancel_futures=True)
        controller.read_executor.shutdown(wait=True, cancel_futures=True)
