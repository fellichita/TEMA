"""Independent observation channel for the ten user-approved sources."""

from app.pilot.approved_sources.collector import collect_approved_sources, unavailable_snapshot
from app.pilot.approved_sources.contracts import (
    ExternalObservation, ObservationPage, SourceCoverage, SourceFetchError, SourceSnapshot,
)

__all__ = (
    "ExternalObservation", "ObservationPage", "SourceCoverage", "SourceFetchError", "SourceSnapshot",
    "collect_approved_sources", "unavailable_snapshot",
)
