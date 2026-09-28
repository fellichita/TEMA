from datetime import UTC, date, datetime
from concurrent.futures import ThreadPoolExecutor
import os
import stat
from threading import Event, Lock
from time import monotonic

import httpx
import pytest

from app.backend.contracts import DocumentRecord, SourcePage
from app.backend.errors import BackendError
from app.backend.providers.crossref import CrossrefProvider
from app.backend.providers.openalex import OpenAlexProvider
from app.pilot.archive import DocumentArchive
from app.pilot.contracts import CorpusSnapshot, QueryLimits, QueryPlan, SearchQuery
from app.pilot.sources import (
    PublicationProviderSession, _FetchedQuery, _schedule_source_queries, collect_snapshot,
)
from app.runtime.credentials import CredentialStore
from app.runtime.jobs import TaskCancelled, TaskFailure


def plan(purpose="discovery", sources=("openalex",)):
    return QueryPlan(original_query="Новые материалы", language="ru", definition="materials",
                     english_query="materials", subdirections=("materials",),
                     queries=tuple(SearchQuery(source=source, text="materials", purpose=purpose) for source in sources),
                     completed_years=(2020, 2021, 2022, 2023, 2024, 2025), as_of=date(2026, 9, 10),
                     planner_version="test")


def document(number=1, **changes):
    values = dict(source="openalex", source_id=f"W{number}", doi=f"10.1000/{number}", title="Material study",
                  publication_year=2025, date_precision="year", url=f"https://openalex.org/W{number}")
    return DocumentRecord.model_validate(values | changes)


class Context:
    cancel_event = Event()

    def __init__(self):
        self.progress_events = []

    def check_cancelled(self):
        assert not self.cancel_event.is_set()

    def progress(self, *args):
        self.progress_events.append(args)


class Provider:
    def __init__(self, pages):
        self.pages, self.closed = pages, False

    def iter_pages(self, request, cancel):
        for page in self.pages:
            if isinstance(page, Exception):
                raise page
            yield page

    def close(self):
        self.closed = True


def test_publication_provider_session_reuses_clients_per_source_and_owns_lifetime():
    credentials = CredentialStore()
    credentials.set("openalex_api_key", "TEST_KEY")
    session = PublicationProviderSession(credentials)
    with session:
        with ThreadPoolExecutor(max_workers=3) as pool:
            openalex = tuple(pool.map(lambda _: session.provider("openalex"), range(3)))
        crossref = session.provider("crossref")
        openalex_client = openalex[0]._client
        crossref_client = crossref._client
        assert all(provider._client is openalex_client for provider in openalex)
        assert crossref_client is not openalex_client
        assert openalex[0].page_size == 100
        assert crossref.page_size == 400
        for provider in (*openalex, crossref):
            provider.close()
        assert not openalex_client.is_closed and not crossref_client.is_closed
    assert openalex_client.is_closed and crossref_client.is_closed


def test_provider_created_before_cancellation_is_closed(tmp_path):
    class CancellingContext(Context):
        def __init__(self):
            super().__init__()
            self.cancel_event = Event()

        def check_cancelled(self):
            if self.cancel_event.is_set():
                raise TaskCancelled()

    context = CancellingContext()
    built = []

    def factory(source):
        item = Provider([])
        built.append(item)
        context.cancel_event.set()
        return item

    with pytest.raises(TaskCancelled):
        collect_snapshot(plan(sources=("openalex", "crossref")), context,
                         DocumentArchive(tmp_path), CredentialStore(), provider_factory=factory)
    assert len(built) == 1 and built[0].closed


def test_real_record_revisions_survive_offline_and_detect_tampering(tmp_path):
    archive = DocumentArchive(tmp_path)
    original = document(abstract="Immutable scientific evidence.")
    reference = archive.put(original)
    assert archive.get(reference.revision_id) == original
    assert archive.put(original) == reference
    archive.path(reference.revision_id).write_text("{}")
    with pytest.raises(TaskFailure, match="повреждена"):
        archive.get(reference.revision_id)
    with pytest.raises(TaskFailure):
        archive.get("../../escape")


@pytest.mark.skipif(os.name == "nt", reason="Windows uses profile ACLs rather than POSIX file modes")
def test_archive_revisions_are_private_even_in_an_existing_permissive_directory(tmp_path):
    directory = tmp_path / "revisions"
    directory.mkdir()
    directory.chmod(0o755)
    archive = DocumentArchive(directory)
    reference = archive.put(document(abstract="Private report evidence."))
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(archive.path(reference.revision_id).parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(archive.path(reference.revision_id).stat().st_mode) == 0o600


def test_one_reading_verifies_each_revision_once_but_the_archive_never_caches(tmp_path):
    """A single operation reads a revision once; the archive itself stays stateless."""
    from app.pilot.archive import ArchiveReading

    archive = DocumentArchive(tmp_path)
    first = archive.put(document(1, abstract="First archived record."))
    second = archive.put(document(2, abstract="Second archived record."))
    reads = []
    original = DocumentArchive.get
    DocumentArchive.get = lambda self, revision_id: (reads.append(revision_id), original(self, revision_id))[1]
    try:
        reading = ArchiveReading(archive)
        assert reading.get(first.revision_id) == reading.get(first.revision_id)
        assert len(reads) == 1
        assert reading.get(second.revision_id).source_id != reading.get(first.revision_id).source_id
        assert len(reads) == 2
        assert reading.directory == archive.directory and reading.path(first.revision_id) == archive.path(first.revision_id)
        # A separate reading of the same archive starts from the files again.
        ArchiveReading(archive).get(first.revision_id)
        assert len(reads) == 3
    finally:
        DocumentArchive.get = original
    # The archive keeps no state, so a replaced file is still refused afterwards.
    stored = archive.path(first.revision_id)
    stored.write_bytes(stored.read_bytes()[:-1] + b" ")
    with pytest.raises(TaskFailure, match="повреждена"):
        archive.get(first.revision_id)


def test_reading_beyond_its_bound_keeps_answering_from_the_archive(tmp_path):
    from app.pilot.archive import ArchiveReading

    archive = DocumentArchive(tmp_path)
    references = [archive.put(document(number, abstract=f"Record {number}.")) for number in range(3)]
    reading = ArchiveReading(archive, limit=1)
    assert [reading.get(item.revision_id).source_id for item in references] == [f"W{n}" for n in range(3)]
    assert reading.get(references[2].revision_id).source_id == "W2"
    with pytest.raises(TaskFailure):
        ArchiveReading(archive, limit=0)


def test_reading_byte_budget_does_not_retain_large_revisions(tmp_path, monkeypatch):
    from app.pilot import archive as archive_module
    from app.pilot.archive import ArchiveReading

    archive = DocumentArchive(tmp_path)
    first = archive.put(document(1, abstract="First archived record."))
    second = archive.put(document(2, abstract="Second archived record."))
    monkeypatch.setattr(archive_module, "MAX_READING_CACHE_BYTES", archive.path(first.revision_id).stat().st_size)
    reads = []
    original = archive.get

    def counted(revision_id):
        reads.append(revision_id)
        return original(revision_id)

    monkeypatch.setattr(archive, "get", counted)
    reading = ArchiveReading(archive)
    assert reading.get(first.revision_id) == reading.get(first.revision_id)
    assert reading.get(second.revision_id) == reading.get(second.revision_id)
    assert reads == [first.revision_id, second.revision_id, second.revision_id]


def test_rearchiving_the_same_document_still_refuses_a_replaced_file(tmp_path):
    """Re-archiving is cheap but not blind: stored bytes must equal what is offered."""
    archive = DocumentArchive(tmp_path)
    original = document(abstract="Immutable scientific evidence.")
    reference = archive.put(original)
    assert archive.put(original) == reference
    stored = archive.path(reference.revision_id)
    stored.write_bytes(stored.read_bytes()[:-1] + b" ")
    with pytest.raises(TaskFailure, match="повреждена"):
        archive.put(original)


def test_source_failure_keeps_received_evidence_and_never_claims_full_history(tmp_path):
    provider = Provider([SourcePage(documents=(document(),), scanned=1, exhausted=False),
                         BackendError("rate_limit", "Слишком много запросов")])
    snapshot = collect_snapshot(plan("history"), Context(), DocumentArchive(tmp_path), CredentialStore(),
                                purpose="history", provider_factory=lambda _: provider)
    assert provider.closed
    assert len(snapshot.documents) == 1
    assert snapshot.coverage[0].state == "partial"
    assert not snapshot.coverage[0].complete_history
    assert "source_rate_limit" in snapshot.coverage[0].reasons


def test_unexpected_source_parser_failure_keeps_other_parallel_source(tmp_path):
    def factory(source):
        if source == "openalex":
            return Provider([SourcePage(documents=(document(1),), scanned=1, exhausted=False),
                             RuntimeError("https://example.org/?token=secret")])
        return Provider([SourcePage(documents=(document(2, source="crossref", source_id="10.1000/2"),),
                                    scanned=1, exhausted=True)])

    snapshot = collect_snapshot(plan(sources=("openalex", "crossref")), Context(),
                                DocumentArchive(tmp_path), CredentialStore(), provider_factory=factory)
    assert {reference.source for reference in snapshot.documents} == {"openalex", "crossref"}
    assert [item.state for item in snapshot.coverage] == ["partial", "complete"]
    assert "source_error" in snapshot.coverage[0].reasons
    assert "secret" not in snapshot.model_dump_json()


def test_skipped_bad_records_prevent_complete_history(tmp_path):
    provider = Provider([SourcePage(documents=(document(),), scanned=2, skipped=1, exhausted=True)])
    snapshot = collect_snapshot(plan("history"), Context(), DocumentArchive(tmp_path), CredentialStore(),
                                purpose="history", provider_factory=lambda _: provider)
    assert snapshot.coverage[0].unresolved_records == 1
    assert not snapshot.coverage[0].complete_history


@pytest.mark.parametrize("rules_version", ["publication-status/1.0.0", "publication-status/2.0.0"])
def test_retractions_and_supplements_are_excluded_not_counted_as_studies(tmp_path, rules_version):
    records = (document(1), document(2, raw_metadata={"is_retracted": True}),
               document(3, raw_metadata={"relation": {"is-supplement-to": [{"id-type": "doi", "id": "10.1000/1"}]}}))
    provider = Provider([SourcePage(documents=records, scanned=3, exhausted=True)])
    snapshot = collect_snapshot(plan("history"), Context(), DocumentArchive(tmp_path), CredentialStore(),
                                purpose="history", provider_factory=lambda _: provider, rules_version=rules_version)
    assert len(snapshot.documents) == (3 if rules_version.endswith("/2.0.0") else 1)
    assert snapshot.coverage[0].accepted_records == 1
    assert snapshot.coverage[0].rejected_records == 2
    assert snapshot.coverage[0].complete_history


def test_document_limit_does_not_turn_top_page_into_a_complete_series(tmp_path):
    provider = Provider([SourcePage(documents=(document(),), scanned=1, total_available=100, exhausted=False)])
    snapshot = collect_snapshot(plan("history"), Context(), DocumentArchive(tmp_path), CredentialStore(),
                                purpose="history", max_documents=1, provider_factory=lambda _: provider)
    assert snapshot.coverage[0].limit_reached
    assert not snapshot.coverage[0].complete_history


def test_cross_source_doi_identity_retains_both_provenances(tmp_path):
    def factory(source):
        record = document(source=source, source_id="10.1000/1" if source == "crossref" else "W1")
        return Provider([SourcePage(documents=(record,), scanned=1, exhausted=True)])

    snapshot = collect_snapshot(plan(sources=("openalex", "crossref")), Context(),
                                DocumentArchive(tmp_path), CredentialStore(), provider_factory=factory)
    assert len(snapshot.documents) == 2
    assert len({reference.study_id for reference in snapshot.documents}) == 1
    assert {reference.source for reference in snapshot.documents} == {"openalex", "crossref"}
    assert all(not item.complete_history for item in snapshot.coverage)


@pytest.mark.parametrize("total", [None, 0])
def test_empty_exhausted_query_is_a_known_empty_snapshot(tmp_path, total):
    snapshot = collect_snapshot(plan("history"), Context(), DocumentArchive(tmp_path), CredentialStore(),
                                purpose="history", provider_factory=lambda _: Provider([
                                    SourcePage(scanned=0, total_available=total, exhausted=True)]))
    assert not snapshot.documents
    assert snapshot.coverage[0].state == "complete"
    assert snapshot.coverage[0].complete_history
    assert not snapshot.coverage[0].reasons


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("purpose", ["discovery", "history"])
def test_exhausted_page_with_known_missing_matches_is_incomplete(tmp_path, count, purpose):
    provider = Provider([SourcePage(documents=tuple(document(index) for index in range(count)),
                                    scanned=count, total_available=100, exhausted=True)])
    snapshot = collect_snapshot(plan(purpose), Context(), DocumentArchive(tmp_path), CredentialStore(),
                                purpose=purpose, provider_factory=lambda _: provider)
    coverage = snapshot.coverage[0]
    assert provider.closed
    assert coverage.state == ("partial" if count else "unavailable")
    assert coverage.pagination_exhausted  # Preserve the marker, without trusting it as proof of completeness.
    assert coverage.scanned_records == coverage.accepted_records == count
    assert not coverage.completed_years and not coverage.comparable
    assert not coverage.complete_history
    assert "inconsistent_total" in coverage.reasons
    assert CorpusSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot


@pytest.mark.parametrize("source,payload,count", [
    ("openalex", {"meta": {"count": 100, "next_cursor": None}, "results": []}, 0),
    ("crossref", {"status": "ok", "message": {"total-results": 100, "items": [
        {"DOI": "10.1000/1", "title": ["Material study"], "published": {"date-parts": [[2025]]}},
    ]}}, 1),
])
def test_real_adapter_end_marker_does_not_override_advertised_matches(tmp_path, source, payload, count):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    with httpx.Client(transport=transport) as client:
        provider_type = OpenAlexProvider if source == "openalex" else CrossrefProvider
        provider = provider_type(client=client, page_size=100, page_delay=0)
        snapshot = collect_snapshot(plan(sources=(source,)), Context(), DocumentArchive(tmp_path),
                                    CredentialStore(), max_documents=100, provider_factory=lambda _: provider)
    coverage = snapshot.coverage[0]
    assert coverage.pagination_exhausted
    assert coverage.scanned_records == count
    assert coverage.state != "complete"
    assert "inconsistent_total" in coverage.reasons


@pytest.mark.parametrize("later_total", [None, 0, 2])
def test_smaller_or_missing_later_total_does_not_erase_known_matches(tmp_path, later_total):
    provider = Provider([
        SourcePage(documents=(document(1),), scanned=1, total_available=100, exhausted=False),
        SourcePage(documents=(document(2),), scanned=1, total_available=later_total, exhausted=True),
    ])
    snapshot = collect_snapshot(plan("history"), Context(), DocumentArchive(tmp_path), CredentialStore(),
                                purpose="history", provider_factory=lambda _: provider)
    coverage = snapshot.coverage[0]
    assert len(snapshot.documents) == coverage.scanned_records == 2
    assert coverage.state == "partial" and not coverage.complete_history
    assert "inconsistent_total" in coverage.reasons


def test_exact_raw_total_remains_complete_after_identity_deduplication(tmp_path):
    record = document()
    provider = Provider([
        SourcePage(documents=(record,), scanned=1, total_available=2, exhausted=False),
        SourcePage(documents=(record,), scanned=1, total_available=2, exhausted=True),
    ])
    snapshot = collect_snapshot(plan("history"), Context(), DocumentArchive(tmp_path), CredentialStore(),
                                purpose="history", provider_factory=lambda _: provider)
    coverage = snapshot.coverage[0]
    assert coverage.scanned_records == 2
    assert coverage.accepted_records == coverage.rejected_records == len(snapshot.documents) == 1
    assert coverage.complete_history and not coverage.reasons


@pytest.mark.parametrize("total", [None, 1])
def test_exhausted_exact_limit_is_complete_when_total_is_consistent(tmp_path, total):
    provider = Provider([SourcePage(documents=(document(),), scanned=1, total_available=total, exhausted=True)])
    snapshot = collect_snapshot(plan("history"), Context(), DocumentArchive(tmp_path), CredentialStore(),
                                purpose="history", max_documents=1, provider_factory=lambda _: provider)
    assert snapshot.coverage[0].complete_history
    assert not snapshot.coverage[0].limit_reached


@pytest.mark.parametrize("pages", [
    [],
    [None],
    [{"scanned": 1, "documents": [], "exhausted": True}],
    [SourcePage(scanned=0, exhausted=True).model_copy(update={"total_available": -1})],
    [SourcePage(scanned=0, exhausted=False)],
    [SourcePage(documents=(document(source="crossref"),), scanned=1, exhausted=True)],
    [SourcePage(documents=(document(1), document(2)), scanned=2, exhausted=True)],
], ids=["no-pages", "not-a-page", "inconsistent-counts", "invalid-frozen-model", "no-progress",
        "wrong-source", "over-limit"])
def test_invalid_source_page_is_reported_without_archiving_it(tmp_path, pages):
    archive = DocumentArchive(tmp_path)
    provider = Provider(pages)
    snapshot = collect_snapshot(plan("history"), Context(), archive, CredentialStore(),
                                purpose="history", max_documents=1, provider_factory=lambda _: provider)
    assert provider.closed
    assert not snapshot.documents
    assert not tuple(tmp_path.rglob("*.json"))
    coverage = snapshot.coverage[0]
    assert coverage.state == "unavailable" and not coverage.complete_history
    assert "source_invalid_response" in coverage.reasons


@pytest.mark.parametrize("last_page", [
    SourcePage(scanned=0, exhausted=True),
    BackendError("source_unavailable", "Источник недоступен"),
], ids=["page-after-exhaustion", "failure-after-exhaustion"])
def test_failure_after_terminal_page_does_not_publish_complete_history(tmp_path, last_page):
    provider = Provider([SourcePage(documents=(document(),), scanned=1, total_available=1, exhausted=True),
                         last_page])
    snapshot = collect_snapshot(plan("history"), Context(), DocumentArchive(tmp_path), CredentialStore(),
                                purpose="history", provider_factory=lambda _: provider)
    assert provider.closed
    assert len(snapshot.documents) == snapshot.coverage[0].accepted_records == 1
    assert snapshot.coverage[0].state == "partial"
    assert not snapshot.coverage[0].complete_history
    assert any(reason.startswith("source_") for reason in snapshot.coverage[0].reasons)


@pytest.mark.parametrize("sources", [("openalex", "crossref"), ("crossref", "openalex")])
@pytest.mark.parametrize("rules_version", ["publication-status/1.0.0", "publication-status/2.0.0"])
def test_retraction_in_one_source_excludes_other_source_doi_copy(tmp_path, sources, rules_version):
    def factory(source):
        record = document(source=source, raw_metadata={"is_retracted": True} if source == "openalex" else {})
        return Provider([SourcePage(documents=(record,), scanned=1, exhausted=True)])

    snapshot = collect_snapshot(plan(sources=sources), Context(), DocumentArchive(tmp_path), CredentialStore(),
                                provider_factory=factory, rules_version=rules_version)
    assert len(snapshot.documents) == (2 if rules_version.endswith("/2.0.0") else 0)
    assert sum(item.accepted_records for item in snapshot.coverage) == 0
    assert sum(item.rejected_records for item in snapshot.coverage) == 2


def test_the_counter_grows_with_every_received_page_not_only_when_a_source_ends(tmp_path):
    def factory(source):
        record = document(source=source, source_id="10.1000/1" if source == "crossref" else "W1")
        return Provider([SourcePage(documents=(record,), scanned=1, exhausted=False),
                         SourcePage(documents=(), scanned=0, exhausted=True)])

    context = Context()
    collect_snapshot(plan(sources=("openalex", "crossref")), context, DocumentArchive(tmp_path),
                     CredentialStore(), provider_factory=factory)
    messages = [message for _, message, *_ in context.progress_events]
    assert messages[0] == "Запрашиваем документы у источников"
    received = [completed for _, message, completed, _ in context.progress_events if "получено" in message]
    # Paging a source takes minutes, so the number rises while it is still running.
    assert received == sorted(received) and received[-1] == 2
    # Admission counters follow the whole wave, never interleave with it.
    admitted = [index for index, message in enumerate(messages) if "документов в выборке" in message]
    requested = [index for index, message in enumerate(messages) if "получено" in message]
    assert admitted and requested and min(admitted) > max(requested)
    assert all(0 <= completed <= total for *_, completed, total in context.progress_events)


def wide_plan(texts=("materials", "membranes"), sources=("openalex", "crossref")):
    return QueryPlan(original_query="Новые материалы", language="ru", definition="materials",
                     english_query="materials", subdirections=("materials",),
                     queries=tuple(SearchQuery(source=source, text=text, purpose="discovery")
                                   for text in texts for source in sources),
                     completed_years=(2020, 2021, 2022, 2023, 2024, 2025), as_of=date(2026, 9, 10),
                     planner_version="test")


class SlowProvider:
    """Yields one page, and records how many of our queries a source holds at once."""

    def __init__(self, number, traffic=None, both_sources=None, release=None, done=None, failure=None):
        self.number, self.closed = number, False
        self.traffic, self.both_sources = traffic, both_sources
        self.release, self.done, self.failure = release, done, failure

    def iter_pages(self, request, cancel):
        if self.traffic is not None:
            self.traffic.enter(request.source)
            assert self.both_sources.wait(3), "оба источника должны опрашиваться одновременно"
        try:
            if self.release is not None:
                assert self.release.wait(3), "ordering event was never released"
            if self.failure is not None:
                raise self.failure
            # A fixed observation time keeps two runs byte-identical, so the test
            # compares ordering rather than the moment each request happened to finish.
            yield SourcePage(documents=(document(self.number, source=request.source, source_id=f"S{self.number}",
                                                 fetched_at=datetime(2026, 9, 10, tzinfo=UTC)),),
                             scanned=1, exhausted=True)
            if self.done is not None:
                self.done.set()
        finally:
            if self.traffic is not None:
                self.traffic.leave(request.source)

    def close(self):
        self.closed = True


class Traffic:
    """How many of our queries each source is answering, and the peak of that."""

    def __init__(self, sources, ready):
        self.lock = Lock()
        self.active = {source: 0 for source in sources}
        self.peak = dict(self.active)
        self.peak_total = 0
        self.ready = ready

    def enter(self, source):
        with self.lock:
            self.active[source] += 1
            self.peak[source] = max(self.peak[source], self.active[source])
            self.peak_total = max(self.peak_total, sum(self.active.values()))
            if all(count >= 1 for count in self.active.values()):
                self.ready.set()

    def leave(self, source):
        with self.lock:
            self.active[source] -= 1


def test_sources_are_asked_together_and_one_source_answers_one_query_at_a_time(tmp_path):
    """A source rate-limits a burst of our queries; different sources do not see each other."""
    both_sources = Event()
    traffic = Traffic(("openalex", "crossref"), both_sources)
    built = []

    def factory(source):
        provider = SlowProvider(len(built) + 1, traffic=traffic, both_sources=both_sources)
        built.append(provider)
        return provider

    snapshot = collect_snapshot(wide_plan(), Context(), DocumentArchive(tmp_path), CredentialStore(),
                               provider_factory=factory)
    assert len(snapshot.documents) == 4
    assert len(built) == 4 and all(provider.closed for provider in built)
    assert traffic.peak_total == 2, "источники опрашиваются параллельно"
    assert max(traffic.peak.values()) == 1, "внутри одного источника запросы идут по очереди"


def test_result_order_follows_the_query_plan_not_the_order_answers_arrive(tmp_path):
    def run(reverse):
        first_done, built = Event(), []

        def factory(source):
            number = len(built) + 1
            # The first query is deliberately the last to answer.
            provider = SlowProvider(number, release=first_done if reverse and number == 1 else None,
                                    done=first_done if reverse and number == 2 else None)
            built.append(provider)
            return provider

        return collect_snapshot(wide_plan(texts=("materials",)), Context(), DocumentArchive(tmp_path),
                                CredentialStore(), provider_factory=factory)

    ordered, reversed_answers = run(False), run(True)
    assert [item.source for item in ordered.coverage] == ["openalex", "crossref"]
    assert [item.source for item in reversed_answers.coverage] == ["openalex", "crossref"]
    assert reversed_answers.snapshot_id == ordered.snapshot_id
    assert [ref.revision_id for ref in reversed_answers.documents] == [ref.revision_id for ref in ordered.documents]


def test_one_unavailable_source_never_cancels_the_other_requests(tmp_path):
    built = []

    def factory(source):
        number = len(built) + 1
        provider = SlowProvider(number, failure=BackendError("network_error", "unavailable") if number == 1 else None)
        built.append(provider)
        return provider

    snapshot = collect_snapshot(wide_plan(texts=("materials",)), Context(), DocumentArchive(tmp_path),
                               CredentialStore(), provider_factory=factory)
    assert [item.state for item in snapshot.coverage] == ["unavailable", "complete"]
    assert snapshot.coverage[0].reasons == ("source_network_error",)
    assert len(snapshot.documents) == 1


def test_planned_shares_are_equal_and_never_exceed_the_received_revision_budget():
    from app.pilot.sources import _planned_caps

    assert _planned_caps(4, 3000, purpose="discovery", budget=10000) == (750, 750, 750, 750)
    assert sum(_planned_caps(3, 10, purpose="discovery", budget=None)) == 10
    assert sum(_planned_caps(4, 20000, purpose="discovery", budget=10000)) == 10000
    fifty_caps = _planned_caps(50, 10, purpose="discovery", budget=8)
    assert sum(fifty_caps) <= 8
    assert 0 in fifty_caps
    assert _planned_caps(2, 500, purpose="history", budget=None) == (500, 500)


def test_source_scheduler_bounds_fifty_independent_sources_to_twelve_workers():
    release, first_wave, cancel = Event(), Event(), Event()
    lock = Lock()
    active = peak = 0
    started = []
    groups = {f"source-{number}": (number,) for number in range(50)}

    def fetch(source, number):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            started.append((source, number))
            if len(started) == 12:
                first_wave.set()
        try:
            assert release.wait(10), "the first wave was not released"
            return _FetchedQuery(scanned=1, exhausted=True)
        finally:
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=1) as runner:
        result = runner.submit(_schedule_source_queries, groups, fetch, cancel, lambda: None)
        try:
            assert first_wave.wait(10), "twelve sources did not start together"
            with lock:
                assert peak == len(started) == 12
        finally:
            release.set()
        answers = result.result(timeout=10)

    assert len(answers) == len(started) == 50
    assert set(answers) == set(range(50))
    assert all(answer.scanned == 1 for answer in answers.values())
    assert peak == 12


def test_source_scheduler_serves_waiting_sources_before_a_second_query():
    cancel, release_first, release_others, first_started, next_source_started = (
        Event(), Event(), Event(), Event(), Event()
    )
    lock = Lock()
    started = []
    active_by_source = {}
    peak_by_source = {}
    groups = {"source-0": (0, 50)} | {f"source-{number}": (number,) for number in range(1, 50)}

    def fetch(source, number):
        with lock:
            started.append((source, number))
            active_by_source[source] = active_by_source.get(source, 0) + 1
            peak_by_source[source] = max(peak_by_source.get(source, 0), active_by_source[source])
        try:
            if number == 0:
                first_started.set()
                assert release_first.wait(10)
            elif 1 <= number <= 11 or number == 12:
                if number == 12:
                    next_source_started.set()
                assert release_others.wait(10)
            return _FetchedQuery(exhausted=True)
        finally:
            with lock:
                active_by_source[source] -= 1

    with ThreadPoolExecutor(max_workers=1) as runner:
        result = runner.submit(_schedule_source_queries, groups, fetch, cancel, lambda: None)
        try:
            assert first_started.wait(10)
            release_first.set()
            assert next_source_started.wait(10), "the next source was starved by a repeat query"
            with lock:
                assert ("source-0", 50) not in started
        finally:
            release_first.set()
            release_others.set()
        answers = result.result(timeout=10)

    assert set(answers) == set(range(51))
    assert all(peak == 1 for peak in peak_by_source.values())


def test_source_scheduler_cancellation_does_not_start_waiting_sources():
    cancel, release, first_wave = Event(), Event(), Event()
    lock = Lock()
    started = []
    groups = {f"source-{number}": (number,) for number in range(50)}

    def fetch(source, number):
        with lock:
            started.append((source, number))
            if len(started) == 12:
                first_wave.set()
        assert release.wait(10)
        return _FetchedQuery(cancelled=cancel.is_set())

    with ThreadPoolExecutor(max_workers=1) as runner:
        result = runner.submit(_schedule_source_queries, groups, fetch, cancel, lambda: None)
        try:
            assert first_wave.wait(10)
            cancel.set()
        finally:
            release.set()
        with pytest.raises(TaskCancelled):
            result.result(timeout=10)

    assert len(started) == 12
    assert {number for _, number in started} == set(range(12))


def test_source_scheduler_preserves_partial_failure_and_completes_other_sources():
    cancel, failed = Event(), Event()
    groups = {f"source-{number}": (number,) for number in range(50)}

    def fetch(source, number):
        if number == 0:
            failed.set()
            return _FetchedQuery(scanned=1, failure="source_rate_limited")
        assert failed.wait(10)
        return _FetchedQuery(scanned=1, exhausted=True)

    answers = _schedule_source_queries(groups, fetch, cancel, lambda: None)
    assert len(answers) == 50
    assert answers[0].scanned == 1 and answers[0].failure == "source_rate_limited"
    assert all(answers[number].failure is None and answers[number].exhausted for number in range(1, 50))
    assert not cancel.is_set()


def test_source_scheduler_paces_crossref_queries_and_uses_other_source_in_gap():
    started = []

    def fetch(source, number):
        started.append((source, number, monotonic()))
        return _FetchedQuery(exhausted=True)

    answers = _schedule_source_queries(
        {"crossref": (0, 2), "other": (1,)}, fetch, Event(), lambda: None,
        max_workers=1, source_cooldowns={"crossref": 0.05},
    )
    assert set(answers) == {0, 1, 2}
    assert [(source, number) for source, number, _ in started] == [
        ("crossref", 0), ("other", 1), ("crossref", 2)
    ]
    assert started[2][2] - started[0][2] >= 0.045


def test_source_scheduler_cancels_while_waiting_for_source_cooldown():
    cancel, first_done = Event(), Event()
    started = []

    def fetch(source, number):
        started.append(number)
        return _FetchedQuery(exhausted=True)

    with ThreadPoolExecutor(max_workers=1) as runner:
        result = runner.submit(_schedule_source_queries, {"crossref": (0, 1)}, fetch,
                               cancel, first_done.set, source_cooldowns={"crossref": 1.0})
        assert first_done.wait(10)
        cancel.set()
        with pytest.raises(TaskCancelled):
            result.result(timeout=2)
    assert started == [0]


class CappedProvider:
    def __init__(self, records, calls, *, reorder_on_refill=False):
        self.records, self.calls = records, calls
        self.reorder_on_refill = reorder_on_refill
        self.closed = False

    def iter_pages(self, request, cancel):
        self.calls.append((request.source, request.max_results))
        selected = self.records[:request.max_results]
        if self.reorder_on_refill and request.max_results > 2:
            selected = (selected[1], selected[0], *selected[2:])
        yield SourcePage(documents=selected, scanned=len(selected), total_available=len(self.records),
                         exhausted=len(selected) == len(self.records))

    def close(self):
        self.closed = True


def test_discovery_refills_unused_unique_slots_and_recounts_final_coverage(tmp_path):
    calls, built = [], []
    records = {
        "openalex": tuple(document(index) for index in (1, 2, 3)),
        "crossref": (document(4, source="crossref", source_id="C4"),),
    }

    def factory(source):
        provider = CappedProvider(records[source], calls)
        built.append(provider)
        return provider

    query_plan = plan(sources=("openalex", "crossref")).model_copy(
        update={"limits": QueryLimits(discovery_documents=4)})
    snapshot = collect_snapshot(query_plan, Context(), DocumentArchive(tmp_path), CredentialStore(),
                                provider_factory=factory)
    assert sorted(calls) == [("crossref", 2), ("openalex", 2), ("openalex", 3)]
    assert len({ref.study_id for ref in snapshot.documents}) == 4
    assert [(item.scanned_records, item.accepted_records, item.state) for item in snapshot.coverage] == [
        (3, 3, "complete"), (1, 1, "complete")]
    assert all(provider.closed for provider in built)


def test_duplicate_refill_uses_unique_study_count_and_keeps_loss_accounting(tmp_path):
    calls = []
    records = {
        "openalex": tuple(document(index) for index in (1, 2, 3)),
        "crossref": tuple(document(index, source="crossref", source_id=f"C{index}")
                          for index in (1, 4, 5)),
    }
    query_plan = plan(sources=("openalex", "crossref")).model_copy(
        update={"limits": QueryLimits(discovery_documents=4)})
    snapshot = collect_snapshot(query_plan, Context(), DocumentArchive(tmp_path), CredentialStore(),
                                provider_factory=lambda source: CappedProvider(records[source], calls))
    assert sorted(calls) == [("crossref", 2), ("openalex", 2), ("openalex", 3)]
    assert len({ref.study_id for ref in snapshot.documents if ref.source == "openalex"}) == 3
    assert len({ref.study_id for ref in snapshot.documents}) == 4
    assert sum(item.accepted_records for item in snapshot.coverage) == 5  # two source revisions of one study
    assert snapshot.coverage[1].scanned_records == 2
    assert snapshot.coverage[1].accepted_records == 2
    assert snapshot.coverage[1].limit_reached


def test_refill_with_changed_source_order_keeps_initial_pages_and_marks_partial(tmp_path):
    calls = []
    records = {
        "openalex": tuple(document(index) for index in (1, 2, 3)),
        "crossref": (document(4, source="crossref", source_id="C4"),),
    }
    query_plan = plan(sources=("openalex", "crossref")).model_copy(
        update={"limits": QueryLimits(discovery_documents=4)})
    snapshot = collect_snapshot(query_plan, Context(), DocumentArchive(tmp_path), CredentialStore(),
        provider_factory=lambda source: CappedProvider(records[source], calls,
                                                       reorder_on_refill=source == "openalex"))
    assert ("openalex", 3) in calls
    assert len({ref.study_id for ref in snapshot.documents}) == 3
    assert snapshot.coverage[0].scanned_records == 2
    assert "source_refill_not_reproducible" in snapshot.coverage[0].reasons


def test_refill_respects_total_received_revision_budget(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pilot.sources._RECEIVED_REVISION_BUDGET", 4)
    calls = []
    records = {
        "openalex": tuple(document(index) for index in (1, 2, 3)),
        "crossref": (document(4, source="crossref", source_id="C4"),),
    }
    query_plan = plan(sources=("openalex", "crossref")).model_copy(
        update={"limits": QueryLimits(discovery_documents=4)})
    snapshot = collect_snapshot(query_plan, Context(), DocumentArchive(tmp_path), CredentialStore(),
        provider_factory=lambda source: CappedProvider(records[source], calls))
    assert sorted(calls) == [("crossref", 2), ("openalex", 2)]
    assert len({ref.study_id for ref in snapshot.documents}) == 3


def test_refill_does_not_request_more_than_source_contract_allows(tmp_path):
    calls = []

    class MostlySkipped:
        def iter_pages(self, request, cancel):
            calls.append(request.max_results)
            yield SourcePage(documents=(document(),), scanned=10_000, skipped=9_999,
                             total_available=20_000, exhausted=False)

        def close(self):
            pass

    query_plan = plan().model_copy(update={"limits": QueryLimits(discovery_documents=10_000)})
    snapshot = collect_snapshot(query_plan, Context(), DocumentArchive(tmp_path), CredentialStore(),
                                provider_factory=lambda _: MostlySkipped())
    assert calls == [10_000]
    assert snapshot.coverage[0].scanned_records == 10_000
    assert snapshot.coverage[0].unresolved_records == 9_999
