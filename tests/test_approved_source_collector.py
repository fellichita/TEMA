from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from threading import Barrier, Event

import pytest

from app.pilot.approved_sources import collect_approved_sources, unavailable_snapshot
from app.pilot.approved_sources.contracts import (
    SOURCE_IDS, ExternalObservation, ObservationPage, SourceFetchError, SourceSnapshot,
)
from app.runtime.jobs import TaskCancelled


AS_OF = date(2026, 9, 23)


def observation(source="arxiv", item="1", **changes):
    values = dict(source_id=source, item_id=item, kind="preprint",
                  title="New technology", url=f"https://example.org/{source}/{item}",
                  published_at=date(2026, 9, 20), observed_at=datetime(2026, 9, 23, tzinfo=UTC))
    return ExternalObservation(**(values | changes))


class Adapter:
    def __init__(self, pages, *, barrier=None, started=None):
        self.pages = pages
        self.barrier = barrier
        self.started = started
        self.calls = []
        self.closed = False

    def iter_pages(self, query, *, as_of, limit, timeout_seconds, cancel):
        self.calls.append((query, as_of, limit, timeout_seconds))
        if self.started is not None:
            self.started.set()
        if self.barrier is not None:
            self.barrier.wait(3)
        for page in self.pages:
            if isinstance(page, Exception):
                raise page
            yield page

    def close(self):
        self.closed = True


def test_parallel_sources_have_stable_order_and_every_source_has_coverage():
    barrier = Barrier(2)
    first = Adapter([ObservationPage(observations=(observation("arxiv"),), scanned=1, exhausted=True)],
                    barrier=barrier)
    second = Adapter([ObservationPage(observations=(observation("biorxiv"),), scanned=1, exhausted=True)],
                     barrier=barrier)
    progress = []
    snapshot = collect_approved_sources("  Quantum sensors  ", as_of=AS_OF, cancel=Event(),
        progress=lambda done, total: progress.append((done, total)),
        adapters={"biorxiv": second, "arxiv": first}, max_workers=2)
    assert snapshot.query == "Quantum sensors"
    assert [item.source_id for item in snapshot.observations] == ["arxiv", "biorxiv"]
    assert [item.source_id for item in snapshot.coverage] == list(SOURCE_IDS)
    assert [item.state for item in snapshot.coverage[:2]] == ["complete", "complete"]
    assert all(item.state == "unavailable" and item.reason_code == "not_configured"
               for item in snapshot.coverage[2:])
    assert first.closed and second.closed
    assert first.calls[0][2] == second.calls[0][2] == 50
    assert progress[0] == (0, len(SOURCE_IDS)) and progress[-1] == (len(SOURCE_IDS), len(SOURCE_IDS))
    assert SourceSnapshot.model_validate(snapshot.model_dump(mode="json")) == snapshot


def test_cap_and_partial_failure_keep_received_observations():
    cap = Adapter([ObservationPage(observations=(observation(item="1"), observation(item="2")),
                                   scanned=2, exhausted=False)])
    fail = Adapter([ObservationPage(observations=(observation("biorxiv"),), scanned=1, exhausted=False),
                    SourceFetchError("rate_limited")])
    snapshot = collect_approved_sources("technology", as_of=AS_OF, cancel=Event(),
        adapters={"arxiv": cap, "biorxiv": fail}, per_source_cap=2, max_observations=20)
    assert [item.item_id for item in snapshot.observations] == ["1", "2", "1"]
    assert snapshot.coverage[0].state == "partial" and snapshot.coverage[0].limit_reached
    assert snapshot.coverage[0].reason_code == "source_limit"
    assert snapshot.coverage[1].state == "partial" and snapshot.coverage[1].reason_code == "rate_limited"
    assert cap.calls[0][2] == fail.calls[0][2] == 2
    assert cap.closed and fail.closed


def test_future_and_duplicate_items_are_counted_without_publishing():
    future = observation(item="future").model_copy(update={"published_at": date(2026, 9, 24)})
    provider = Adapter([ObservationPage(observations=(observation(), future, observation()),
                                        scanned=4, exhausted=True)])
    snapshot = collect_approved_sources("technology", as_of=AS_OF, cancel=Event(),
                                        adapters={"arxiv": provider})
    assert len(snapshot.observations) == 1
    assert (snapshot.coverage[0].scanned, snapshot.coverage[0].accepted,
            snapshot.coverage[0].rejected, snapshot.coverage[0].duplicates) == (4, 1, 2, 1)


def test_invalid_source_page_is_isolated_and_does_not_publish_it():
    wrong = Adapter([ObservationPage(observations=(observation("arxiv", item="wrong"),),
                                     scanned=1, exhausted=True)])
    good = Adapter([ObservationPage(observations=(observation("arxiv"),),
                                    scanned=1, exhausted=True)])
    snapshot = collect_approved_sources("technology", as_of=AS_OF, cancel=Event(),
                                        adapters={"arxiv": good, "biorxiv": wrong})
    assert [item.source_id for item in snapshot.observations] == ["arxiv"]
    assert snapshot.coverage[1].state == "unavailable"
    assert snapshot.coverage[1].reason_code == "invalid_response"
    assert wrong.closed and good.closed


def test_mutated_page_revalidates_nested_observation_before_publication():
    unsafe = observation().model_copy(update={"url": "http://user:pw@example.org/story"})
    page = ObservationPage(observations=(observation(),), scanned=1, exhausted=True)
    provider = Adapter([page.model_copy(update={"observations": (unsafe,)})])
    snapshot = collect_approved_sources("technology", as_of=AS_OF, cancel=Event(),
                                        adapters={"arxiv": provider})
    assert snapshot.observations == ()
    assert snapshot.coverage[0].state == "unavailable"
    assert snapshot.coverage[0].reason_code == "invalid_response"


def test_cancellation_closes_source_and_does_not_publish_snapshot():
    cancel, started = Event(), Event()

    class WaitingAdapter(Adapter):
        def iter_pages(self, query, *, as_of, limit, timeout_seconds, cancel):
            started.set()
            cancel.wait(3)
            raise TaskCancelled()
            yield  # pragma: no cover

    adapter = WaitingAdapter([])
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(collect_approved_sources, "technology", as_of=AS_OF, cancel=cancel,
                             adapters={"arxiv": adapter})
        assert started.wait(3)
        cancel.set()
        with pytest.raises(TaskCancelled):
            future.result(timeout=3)
    assert adapter.closed


def test_cancellation_does_not_start_sources_waiting_for_a_worker():
    cancel = Event()
    first_started, second_started = Event(), Event()

    class WaitingAdapter(Adapter):
        def iter_pages(self, query, *, as_of, limit, timeout_seconds, cancel):
            self.calls.append((query, as_of, limit, timeout_seconds))
            self.started.set()
            cancel.wait(3)
            raise TaskCancelled()
            yield  # pragma: no cover

    first = WaitingAdapter([], started=first_started)
    second = WaitingAdapter([], started=second_started)
    waiting = Adapter([ObservationPage(scanned=0, exhausted=True)])
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(collect_approved_sources, "technology", as_of=AS_OF, cancel=cancel,
                             adapters={"arxiv": first, "biorxiv": second, "openreview": waiting},
                             max_workers=2)
        assert first_started.wait(3) and second_started.wait(3)
        cancel.set()
        with pytest.raises(TaskCancelled):
            future.result(timeout=3)
    assert not waiting.calls
    assert first.closed and second.closed and waiting.closed


def test_progress_failure_before_submission_still_closes_adapters():
    adapter = Adapter([])

    def stop(_done, _total):
        raise TaskCancelled()

    with pytest.raises(TaskCancelled):
        collect_approved_sources("technology", as_of=AS_OF, cancel=Event(),
                                 adapters={"arxiv": adapter}, progress=stop)
    assert adapter.closed


@pytest.mark.parametrize("url", ["https://example.org/story?token=secret", "http://user:pw@example.org/story",
                                      "http://127.0.0.1\\@example.org/story"])
def test_observation_rejects_credential_or_unsafe_urls(url):
    with pytest.raises(ValueError):
        observation(url=url)


def test_unavailable_snapshot_is_safe_and_keeps_all_source_statuses():
    snapshot = unavailable_snapshot("technology", AS_OF, "collector_error")
    assert snapshot.observations == ()
    assert [item.source_id for item in snapshot.coverage] == list(SOURCE_IDS)
    assert all(item.state == "unavailable" and item.reason_code == "collector_error"
               for item in snapshot.coverage)
    assert SourceSnapshot.model_validate(snapshot.model_dump(mode="json")) == snapshot


def test_saved_ten_source_snapshot_remains_readable_after_catalogue_expansion():
    saved = unavailable_snapshot("technology", AS_OF).model_dump(mode="json")
    saved["coverage"] = saved["coverage"][:10]
    restored = SourceSnapshot.model_validate(saved)
    assert len(restored.coverage) == 10
    assert tuple(item.source_id for item in restored.coverage) == SOURCE_IDS[:10]


def test_legacy_snapshot_cannot_contain_observations_without_coverage():
    saved = unavailable_snapshot("technology", AS_OF).model_dump(mode="json")
    saved["coverage"] = saved["coverage"][:10]
    saved["observations"] = [observation("habr", kind="community").model_dump(mode="json")]
    with pytest.raises(ValueError, match="no source coverage"):
        SourceSnapshot.model_validate(saved)


def test_gdelt_index_date_cannot_be_described_as_publisher_date():
    with pytest.raises(ValueError):
        observation("gdelt", kind="news_aggregate")
    item = observation("gdelt", kind="news_aggregate", date_basis="indexed")
    assert item.date_basis == "indexed"


def test_adapter_exception_text_never_enters_coverage():
    provider = Adapter([RuntimeError("https://example.org/?token=secret")])
    snapshot = collect_approved_sources("technology", as_of=AS_OF, cancel=Event(),
                                        adapters={"arxiv": provider})
    assert snapshot.coverage[0].reason_code == "source_error"
    assert "secret" not in snapshot.model_dump_json()
