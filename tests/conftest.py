"""Keep unrelated service tests deterministic and free of live source requests."""

from __future__ import annotations

from importlib.util import find_spec

import pytest


@pytest.fixture(autouse=True)
def isolate_approved_source_network(monkeypatch):
    # The lightweight web-only test environment intentionally lacks Pydantic.
    if find_spec("pydantic") is None:
        return

    import app.pilot.approved_sources as approved_sources

    def offline_snapshot(query, *, as_of, cancel, **_kwargs):
        return approved_sources.unavailable_snapshot(query, as_of, "test_offline")

    monkeypatch.setattr(approved_sources, "collect_approved_sources", offline_snapshot)
