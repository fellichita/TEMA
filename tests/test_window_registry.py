"""Native destruction releases dialog owners without retaining them in callbacks."""

import gc
from types import SimpleNamespace
import weakref

from app.ui.windows import WindowRegistry


class Window:
    def __init__(self):
        self.callbacks = []

    def bind(self, sequence, callback, add):
        assert sequence == "<Destroy>"
        assert add == "+"
        self.callbacks.append(callback)

    def event(self, widget):
        for callback in self.callbacks:
            callback(SimpleNamespace(widget=widget))


class Owner:
    def __init__(self, window):
        self.window = window


def test_raw_and_wrapped_windows_unregister_only_on_their_own_destruction():
    registry = WindowRegistry()
    raw, wrapped = Window(), Owner(Window())
    registry.append(raw)
    registry.append(wrapped)
    registry.append(raw)
    assert registry == [raw, wrapped]
    raw.event(Window())
    wrapped.window.event(Window())
    assert registry == [raw, wrapped]
    raw.event(raw)
    assert registry == [wrapped]
    wrapped.window.event(wrapped.window)
    assert registry == []
    # Repeated or late native notifications remain harmless.
    raw.event(raw)


def test_removed_owner_and_registry_are_not_retained_by_native_callbacks():
    window = Window()
    registry = WindowRegistry()
    owner = Owner(window)
    owner_ref, registry_ref = weakref.ref(owner), weakref.ref(registry)
    registry.append(owner)
    registry.clear()
    del owner, registry
    gc.collect()
    assert owner_ref() is None
    assert registry_ref() is None
    window.event(window)


def test_destroyed_wrapper_releases_its_payload_without_closing_the_application():
    registry = WindowRegistry()
    owner = Owner(Window())
    owner.payload = Owner(None)
    references = [weakref.ref(owner), weakref.ref(owner.payload)]
    registry.append(owner)
    owner.window.event(owner.window)
    del owner
    gc.collect()
    assert registry == []
    assert all(reference() is None for reference in references)
