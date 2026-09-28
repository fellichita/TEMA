"""Keep only live dialog owners in the application's window registry."""

from weakref import ref


class WindowRegistry(list):
    """List-compatible ownership for raw Toplevels and dialog wrappers.

    Dialog factories register with append. Removing an owner or clearing the
    registry does not keep it alive through a callback on its native window.
    """

    def append(self, owner):
        if owner in self:
            return
        window = getattr(owner, "window", owner)
        registry_ref, owner_ref, window_ref = ref(self), ref(owner), ref(window)

        def destroyed(event):
            # A toplevel's bind tag also receives its descendants' events.
            if event.widget is not window_ref():
                return
            registry, registered = registry_ref(), owner_ref()
            if registry is not None and registered is not None and registered in registry:
                registry.remove(registered)

        window.bind("<Destroy>", destroyed, add="+")
        super().append(owner)
