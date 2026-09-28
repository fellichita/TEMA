"""Fair, bounded parallel collection for independent external observations."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, date, datetime
import math
import re
from threading import Event
from typing import Protocol, cast
import unicodedata

from pydantic import ValidationError

from app.backend.errors import BackendError, CancelledError
from app.pilot.approved_sources.catalog import SOURCE_INFO, SourcePolicy
from app.pilot.approved_sources.contracts import (
    SOURCE_IDS, CoverageState, ExternalObservation, ObservationPage, SourceCoverage, SourceFetchError,
    SourceId, SourceSnapshot,
)
from app.runtime.jobs import TaskCancelled

_MAX_WORKERS = 12


class ObservationAdapter(Protocol):
    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]: ...

    def close(self) -> None: ...


def default_adapters() -> dict[SourceId, ObservationAdapter]:
    """Build fresh, operation-owned clients for the approved source catalogue."""
    from app.pilot.approved_sources.adapters_news import make_news_adapters
    from app.pilot.approved_sources.adapters_open import make_open_adapters
    from app.pilot.approved_sources.adapters_science import make_science_adapters

    adapters: dict[str, object] = {str(source): adapter for source, adapter in make_science_adapters().items()}
    for factory in (make_news_adapters, make_open_adapters):
        for source, adapter in factory().items():
            adapters[str(source)] = adapter
    if set(adapters) != set(SOURCE_IDS):
        raise ValueError("The approved source registry is incomplete")
    return {cast(SourceId, source): cast(ObservationAdapter, adapter) for source, adapter in adapters.items()}


@dataclass(frozen=True)
class _SourceResult:
    observations: tuple[ExternalObservation, ...]
    coverage: SourceCoverage


def _safe_code(value: object) -> str:
    return value if isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value) else "source_error"


def _check_cancel(cancel: Event) -> None:
    if cancel.is_set():
        raise TaskCancelled()


def _normal_query(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("Query must be text")
    result = unicodedata.normalize("NFKC", query).strip()
    if (not 1 <= len(result) <= 500 or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"}
                                          for char in result)):
        raise ValueError("Query is empty, too long, or contains controls")
    return result


def _caps(per_source_cap: int, max_observations: int, active: tuple[SourceId, ...] = SOURCE_IDS,
          weights: Mapping[str, float] | None = None) -> dict[SourceId, int]:
    """Reserve source shares before tasks start; completion timing changes nothing.

    Only active sources share the budget. Without learned weights the shares are
    equal; with them a source that proved useful in earlier analyses receives
    proportionally more of the same total, never above the per-source cap.
    """
    caps: dict[SourceId, int] = dict.fromkeys(SOURCE_IDS, 0)
    if not active:
        return caps
    total = min(max_observations, per_source_cap * len(active))
    if not weights:
        equal_share, remainder = divmod(total, len(active))
        for index, source in enumerate(active):
            caps[source] = equal_share + (index < remainder)
        return caps
    shares = {source: max(0.0, float(weights.get(source, 1.0))) for source in active}
    scale = sum(shares.values()) or 1.0
    for source in active:
        caps[source] = min(per_source_cap, int(total * shares[source] / scale))
    # Integer rounding leftovers go to the strongest sources first.
    leftover = total - sum(caps.values())
    for source in sorted(active, key=lambda item: (-shares[item], SOURCE_IDS.index(item))):
        if leftover <= 0:
            break
        room = per_source_cap - caps[source]
        if room > 0:
            step = min(room, leftover)
            caps[source] += step
            leftover -= step
    return caps


def _not_collected(source: SourceId, limit: int, reason: str) -> _SourceResult:
    return _SourceResult((), SourceCoverage(
        source_id=source, state="unavailable", requested_limit=limit, scanned=0,
        accepted=0, rejected=0, duplicates=0, limit_reached=limit == 0,
        reason_code=reason,
    ))


def unavailable_snapshot(query: str, as_of: date, reason_code: str = "collector_error") -> SourceSnapshot:
    """A safe sidecar when the optional collector itself cannot be started."""
    text = _normal_query(query)
    if type(as_of) is not date:
        raise ValueError("A collection date is required")
    reason = _safe_code(reason_code)
    return SourceSnapshot(query=text, as_of=as_of, collected_at=datetime.now(UTC),
                          coverage=tuple(_not_collected(source, 0, reason).coverage for source in SOURCE_IDS))


def _fetch_source(source: SourceId, adapter: ObservationAdapter, query: str, as_of: date,
                  limit: int, timeout_seconds: float, cancel: Event) -> _SourceResult:
    observations: list[ExternalObservation] = []
    seen_ids: set[str] = set()
    scanned = rejected = duplicates = 0
    total_available: int | None = None
    exhausted = False
    failure: str | None = None
    iterator: Iterator[ObservationPage] | None = None
    try:
        _check_cancel(cancel)
        iterator = adapter.iter_pages(query, as_of=as_of, limit=limit,
                                      timeout_seconds=timeout_seconds, cancel=cancel)
        for raw_page in iterator:
            _check_cancel(cancel)
            try:
                # Revalidate even frozen objects: model_copy(update=...) skips Pydantic checks.
                payload = raw_page.model_dump(mode="json") if isinstance(raw_page, ObservationPage) else raw_page
                page = ObservationPage.model_validate(payload)
            except (TypeError, ValueError, ValidationError):
                failure = "invalid_response"
                break
            if (page.scanned > limit - scanned or not page.exhausted and page.scanned == 0
                    or any(item.source_id != source for item in page.observations)):
                failure = "invalid_response"
                break
            scanned += page.scanned
            rejected += page.scanned - len(page.observations)
            if page.total_available is not None:
                total_available = max(total_available or 0, page.total_available)
            for item in page.observations:
                if item.published_at > as_of:
                    rejected += 1
                elif item.item_id in seen_ids:
                    duplicates += 1
                else:
                    seen_ids.add(item.item_id)
                    observations.append(item)
            exhausted = page.exhausted
            if exhausted or scanned >= limit:
                break
        if failure is None and not exhausted and scanned < limit:
            failure = "missing_end_marker"
    except (TaskCancelled, CancelledError):
        if cancel.is_set():
            raise TaskCancelled() from None
        failure = "cancelled_by_source"
    except SourceFetchError as error:
        failure = error.code
    except BackendError as error:
        failure = _safe_code(error.code)
    except TimeoutError:
        failure = "timeout"
    except Exception:
        # A malformed response or transport may put private URLs in its message.
        # Only a fixed code enters the result and the browser.
        failure = "source_error"
    finally:
        if iterator is not None:
            close_iterator = getattr(iterator, "close", None)
            if callable(close_iterator):
                try:
                    close_iterator()
                except Exception:
                    failure = failure or "source_error"
    limited = not exhausted and scanned >= limit
    reason = failure or ("source_limit" if limited else None)
    state: CoverageState = "complete" if reason is None else "partial" if scanned else "unavailable"
    return _SourceResult(tuple(observations), SourceCoverage(
        source_id=source, state=state, requested_limit=limit, scanned=scanned,
        accepted=len(observations), rejected=rejected, duplicates=duplicates,
        limit_reached=limited, reason_code=reason, total_available=total_available,
    ))


def collect_approved_sources(query: str, *, as_of: date, cancel: Event,
                             progress: Callable[[int, int], None] | None = None,
                             adapters: Mapping[SourceId, ObservationAdapter] | None = None,
                             per_source_cap: int = 50, max_observations: int = 900,
                             max_workers: int = _MAX_WORKERS,
                             timeout_seconds: float = 20.0,
                             policy: SourcePolicy | None = None,
                             localized: Mapping[str, str] | None = None) -> SourceSnapshot:
    """Search independent sources concurrently with a bounded pending queue.

    The cap is on scanned source items, not accepted matches. Adapters must honor
    the supplied network timeout and cancellation event; Python threads cannot
    forcibly interrupt an uncooperative blocking HTTP call.

    `policy` holds the owner's country filter and disabled sources: a skipped
    source keeps its place in the coverage with a reason and no budget.
    `localized` maps a language code to the query in that language (for
    example the user's Russian wording for Russian-language sources).
    """
    text = _normal_query(query)
    if type(as_of) is not date or not isinstance(cancel, Event):
        raise ValueError("A date and cancellation event are required")
    if (type(per_source_cap) is not int or not 1 <= per_source_cap <= 1000
            or type(max_observations) is not int or not 10 <= max_observations <= 10000
            or type(max_workers) is not int or not 1 <= max_workers <= _MAX_WORKERS
            or type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds) or not 1 <= timeout_seconds <= 180):
        raise ValueError("Invalid observation collection limits")
    _check_cancel(cancel)
    registry = default_adapters() if adapters is None else dict(adapters)
    if any(source not in SOURCE_IDS for source in registry):
        raise ValueError("An unapproved source was supplied")
    policy = policy or SourcePolicy()
    skipped = {source: reason for source in SOURCE_IDS
               if (reason := policy.skip_reason(source)) is not None}
    active = tuple(source for source in SOURCE_IDS if source in registry and source not in skipped)
    caps = _caps(per_source_cap, max_observations, active, policy.weights or None)
    languages = {key: value for key, value in (localized or {}).items()
                 if isinstance(key, str) and isinstance(value, str) and value.strip()}
    for source, configured in registry.items():
        configure = getattr(configured, "configure", None)
        if callable(configure) and source not in skipped:
            info = SOURCE_INFO[source]
            configure(countries=policy.country_filter() if info.multi_country else (),
                      editions=policy.editions(getattr(configured, "EDITIONS", {})) if info.multi_country else (),
                      localized=languages)
    results: dict[SourceId, _SourceResult] = {}
    total = len(SOURCE_IDS)
    completed = 0
    try:
        if progress is not None:
            progress(completed, total)
        worker_count = min(max_workers, len(registry) or 1)
        with ThreadPoolExecutor(max_workers=worker_count,
                                thread_name_prefix="approved-source") as pool:
            pending: dict[Future[_SourceResult], SourceId] = {}
            ready: deque[tuple[SourceId, ObservationAdapter]] = deque()
            for source in SOURCE_IDS:
                adapter = registry.get(source)
                if source in skipped:
                    results[source] = _not_collected(source, 0, skipped[source])
                    completed += 1
                    if progress is not None:
                        progress(completed, total)
                elif adapter is None or caps[source] == 0:
                    results[source] = _not_collected(source, caps[source],
                                                     "not_configured" if adapter is None else "source_limit")
                    completed += 1
                    if progress is not None:
                        progress(completed, total)
                else:
                    ready.append((source, adapter))

            def dispatch() -> None:
                # Submit only work that can run now. Queued futures can otherwise
                # begin new network requests after cancellation was requested.
                while ready and len(pending) < worker_count and not cancel.is_set():
                    source, adapter = ready.popleft()
                    # Русскоязычный источник ищет по русской формулировке, если она есть.
                    language = getattr(adapter, "language", "en")
                    source_query = _normal_query(languages[language]) if language != "en" and language in languages \
                        else text
                    pending[pool.submit(_fetch_source, source, adapter, source_query, as_of,
                                        caps[source], float(timeout_seconds), cancel)] = source

            dispatch()
            while pending or ready:
                _check_cancel(cancel)
                finished, _ = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in sorted(finished, key=lambda item: SOURCE_IDS.index(pending[item])):
                    source = pending.pop(future)
                    results[source] = future.result()
                    completed += 1
                    if progress is not None:
                        progress(completed, total)
                dispatch()
    finally:
        # Also close instances whose task was never submitted because progress
        # or cancellation interrupted setup. All workers have exited here.
        for adapter in registry.values():
            try:
                adapter.close()
            except Exception:
                pass
    _check_cancel(cancel)
    # Страна наблюдения: своя у записи (издание, журнал) или страна самого источника.
    observations = tuple(item if item.country is not None else item.model_copy(
        update={"country": SOURCE_INFO[source].country})
        for source in SOURCE_IDS for item in results[source].observations)
    coverage = tuple(results[source].coverage for source in SOURCE_IDS)
    return SourceSnapshot(query=text, as_of=as_of, collected_at=datetime.now(UTC),
                          observations=observations, coverage=coverage)
