"""Real bounded source retrieval, with explicit loss and coverage accounting."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime, UTC
import math
import re
from threading import Event, Lock
from time import monotonic
from typing import Literal

import httpx

from app.backend.contracts import DocumentRecord, SearchRequest, SourcePage, normalize_doi
from app.backend.errors import BackendError, CancelledError
from app.backend.providers.base import DocumentProvider
from app.backend.providers.crossref import CrossrefProvider
from app.backend.providers.openalex import OpenAlexProvider
from app.pilot.archive import DocumentArchive
from app.pilot.contracts import CorpusSnapshot, Coverage, DocumentRevisionRef, QueryPlan, SearchQuery, content_hash
from app.pilot.retractions import (
    LEGACY_STATUS_RULES, STATUS_RULES, is_explicitly_retracted, retracted_family_keys, validate_status_rules,
)
from app.runtime.credentials import CredentialStore, CredentialUnavailable
from app.runtime.jobs import RunContext, TaskCancelled


def make_provider(source: str, credentials: CredentialStore,
                  client: httpx.Client | None = None) -> DocumentProvider:
    if source == "openalex":
        # Anonymous search at OpenAlex is throttled under load: it answers 429
        # and names how long to wait — measured at 36 seconds, just past the
        # default ceiling, which turned a pause into a lost source. One wait
        # and retry keeps such a query alive. More did not: on 24.09.2026 the
        # "elevated load" refusal (Retry-After 39 s) outlasted three attempts,
        # costing about two minutes per operation for nothing. A free API key
        # removes the throttling altogether.
        return OpenAlexProvider(api_key=credentials.get("openalex_api_key"), client=client, page_size=100,
                                timeout_seconds=15, max_retries=1, max_response_bytes=5_000_000,
                                page_deadline_seconds=180, max_retry_delay=60)
    if source == "crossref":
        # Discovery shares commonly exceed 200 records. A 400-row page avoids
        # another one-second public-pool pause; oversized replies still trigger
        # the provider's existing bounded 100-row retry at the same cursor.
        return CrossrefProvider(client=client, page_size=400, timeout_seconds=15,
                                max_retries=1, max_response_bytes=5_000_000)
    raise ValueError("This collector accepts publication sources only")


class PublicationProviderSession:
    """Own one thread-safe keep-alive client per publication source operation."""

    def __init__(self, credentials: CredentialStore,
                 provider_builder: Callable[..., DocumentProvider] | None = None):
        self.credentials = credentials
        self._provider_builder = provider_builder or make_provider
        self._clients: dict[str, httpx.Client] = {}
        self._lock = Lock()
        self._closed = False

    def provider(self, source: str) -> DocumentProvider:
        if source not in {"openalex", "crossref"}:
            raise ValueError("This collector accepts publication sources only")
        with self._lock:
            if self._closed:
                raise RuntimeError("Publication provider session is closed")
            client = self._clients.get(source)
            if client is None:
                client = httpx.Client(follow_redirects=False, timeout=15,
                    limits=httpx.Limits(max_connections=4, max_keepalive_connections=2))
                self._clients[source] = client
        return self._provider_builder(source, self.credentials, client)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            clients = tuple(self._clients.values())
            self._clients.clear()
        for client in clients:
            client.close()

    def __enter__(self) -> PublicationProviderSession:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


_SUPPORTING_TYPES = frozenset({"dataset", "data-set", "data", "software", "component", "supplementary-material",
                              "supplement", "image", "audio", "video", "peer-review", "reference-entry"})
_SYNTHESIS_TYPES = frozenset({"review", "systematic-review", "meta-analysis", "perspective", "editorial"})
_SUPPORTING_TITLE = re.compile(
    r"^(?:supporting\s+(?:information|data)|supplement(?:ary|al)?\s+(?:material|information|data)"
    r"|(?:data\s*set|source\s+data|code|software)\s+(?:for|associated\s+with|supporting)\b)", re.I)
_SYNTHESIS_TITLE = re.compile(r"\b(?:review|survey|roadmap|meta.analysis)\b|обзор", re.I)
_SYNTHESIS_ABSTRACT = re.compile(
    r"\b(?:we|this\s+(?:paper|article|work|study|review|perspective))\s+(?:(?:systematically|comprehensively)\s+)?"
    r"(?:reviews?|surveys?|summari[sz]es?|provides?\s+(?:an?\s+)?(?:review|survey|overview)|is\s+(?:an?\s+)?review)\b"
    r"|\b(?:this|our)\s+(?:review|survey|roadmap)\b|(?:в\s+(?:этой|данной)\s+(?:работе|статье)|мы)\s+(?:обобщаем|рассматриваем\s+литературу)"
    r"|\b(?:in\s+this\s+(?:article|paper|work|study)|the\s+present\s+(?:article|paper))[^.!?]{0,220}\b(?:is|are)\s+reviewed\b"
    r"|(?:представлен|приведён|приведен)\s+обзор", re.I)


def _doi_relations(document: DocumentRecord, kind: str) -> set[str]:
    relation = document.raw_metadata.get("relation")
    references = relation.get(kind, []) if isinstance(relation, dict) else []
    result = set()
    if isinstance(references, list):
        for reference in references:
            if (isinstance(reference, dict) and reference.get("id-type") == "doi"
                    and isinstance(reference.get("id"), str)):
                try:
                    result.add("doi:" + normalize_doi(reference["id"]))
                except ValueError:
                    continue
    return result


def primary_research_exclusion(document: DocumentRecord, *, rules_version: str = STATUS_RULES) -> str | None:
    """Classify evidence units without treating datasets or literature surveys as studies.

    These records can still be kept as supporting material by their importer.
    Merely mentioning a previous review in a primary abstract is not an exclusion.
    """
    validate_status_rules(rules_version)
    kinds = {document.document_type.casefold()}
    raw_kind = document.raw_metadata.get("type")
    if isinstance(raw_kind, str):
        kinds.add(raw_kind.casefold())
    if rules_version == STATUS_RULES:
        for field in ("subtype", "resource_type", "resource-type"):
            value = document.raw_metadata.get(field)
            if isinstance(value, str):
                kinds.add(value.casefold())
        if (_SUPPORTING_TITLE.search(document.title)
                or _doi_relations(document, "is-supplement-to")):
            return "supporting_asset_not_research"
    if kinds & _SUPPORTING_TYPES:
        return "supporting_asset_not_research"
    if (kinds & _SYNTHESIS_TYPES or _SYNTHESIS_TITLE.search(document.title)
            or _SYNTHESIS_ABSTRACT.search(document.abstract or "")):
        return "literature_synthesis_not_primary_research"
    if kinds & {"patent", "report", "paratext", "correction", "retraction", "erratum"}:
        return "not_independent_research"
    return None


def supporting_asset_status(documents: Sequence[DocumentRecord], *, rules_version: str = STATUS_RULES,
                            check: Callable[[], None] | None = None) -> tuple[set[str], frozenset[int]]:
    """Asset identities and positions of revisions proving an asset relation.

    A generic OpenAlex manifestation must not become a primary case when the
    Crossref copy or its parent's metadata explicitly identifies a supplement.
    Version relations cannot override a conflicting publication-unit type.
    """
    validate_status_rules(rules_version)
    supporting: set[str] = set()
    material: set[int] = set()
    source_keys: dict[tuple[str, str], set[str]] = {}
    for index, document in enumerate(documents):
        if check:
            check()
        source_keys.setdefault((document.source, document.source_id), set()).add(document.document_key)
        self_supporting = primary_research_exclusion(document, rules_version=rules_version) == "supporting_asset_not_research"
        if self_supporting:
            supporting.add(document.document_key)
        related = _doi_relations(document, "is-supplemented-by") if rules_version == STATUS_RULES else set()
        if related:
            supporting.update(related)
        if self_supporting or related:
            material.add(index)
    if rules_version == STATUS_RULES:
        neighbors: dict[str, set[str]] = {}
        for keys in source_keys.values():
            anchor = min(keys)
            for key in keys:
                neighbors.setdefault(key, set()).add(anchor)
                neighbors.setdefault(anchor, set()).add(key)
        pending = list(supporting)
        while pending:
            key = pending.pop()
            new_keys = neighbors.get(key, set()) - supporting
            supporting.update(new_keys)
            pending.extend(new_keys)
    return supporting, frozenset(material)


def supporting_asset_keys(documents: Sequence[DocumentRecord], *, rules_version: str = STATUS_RULES,
                           check: Callable[[], None] | None = None) -> set[str]:
    """Asset identities observed in any source revision, never their parent papers."""
    return supporting_asset_status(documents, rules_version=rules_version, check=check)[0]


def exclusion_reason(document: DocumentRecord, start: date, end: date, *, legacy: bool = False,
                     primary_only: bool = False, rules_version: str = STATUS_RULES) -> str | None:
    validate_status_rules(rules_version)
    if document.publication_year is None:
        return "unknown_publication_year"
    if not start.year <= document.publication_year <= end.year:
        return "outside_requested_years"
    if document.publication_date and not start <= document.publication_date <= end:
        return "outside_requested_dates"
    if (document.raw_metadata.get("is_retracted") is True if legacy else
            is_explicitly_retracted(document, rules_version=rules_version)):
        return "retracted"
    kind = document.document_type.casefold()
    if kind in {"paratext", "editorial", "correction", "retraction", "erratum", "peer-review"}:
        return "not_independent_research"
    if primary_only and (reason := primary_research_exclusion(
            document, rules_version=LEGACY_STATUS_RULES if legacy else rules_version)):
        return reason
    relations = document.raw_metadata.get("relation", {})
    if (isinstance(relations, dict) and relations.get("is-supplement-to")
            and (legacy or rules_version == LEGACY_STATUS_RULES or _doi_relations(document, "is-supplement-to"))):
        return "supplement"
    # Do not collapse uncertain cross-source near matches by title alone.
    return None


_FETCH_WORKERS = 12
# Crossref's public multi-record pool allows one request per second. Provider
# page_delay covers pages, but not the boundary between separate queries.
_SOURCE_QUERY_COOLDOWNS: Mapping[str, float] = {"crossref": 1.0}
# Kept in step with discovery's revision ceiling: a full 10 000-study corpus
# arrives as more revisions than studies, and every one of them is archived.
_RECEIVED_REVISION_BUDGET = 20000


def _planned_caps(count: int, limit: int, *, purpose: str, budget: int | None) -> tuple[int, ...]:
    """Decide every query's page budget before the first request is sent.

    The sequential collector narrowed each share by what earlier queries had
    already accepted. Concurrent requests cannot observe each other, so the
    shares are fixed here by the same equal-share rule, which produces the very
    same numbers whenever a query fills its share. A query whose share is
    exhausted by an earlier one is reported as limited, exactly as before.
    """
    if purpose != "discovery":
        return tuple([limit] * count)
    caps: list[int] = []
    remaining, left = limit, budget
    for number in range(count):
        if remaining == 0 or left == 0:
            caps.append(0)
            continue
        share = max(1, math.ceil(remaining / (count - number)))
        if left is not None:
            share = min(share, math.ceil(left / (count - number)))
        caps.append(share)
        remaining = max(0, remaining - share)
        if left is not None:
            left = max(0, left - share)
    return tuple(caps)


@dataclass
class _FetchedQuery:
    """One query's raw source response: no admission decision is taken here."""
    pages: tuple[SourcePage, ...] = ()
    scanned: int = 0
    total_available: int | None = None
    exhausted: bool = False
    failure: str | None = None
    cancelled: bool = False


def _fetch_query(provider: DocumentProvider, request: SearchRequest, cap: int, source: str,
                 cancel: Event, on_page: Callable[[int], None] | None = None) -> _FetchedQuery:
    """Page one source in a worker thread, validating only its own response.

    Nothing here writes to the archive, reads another query's results or decides
    admission; those stay sequential so a run remains reproducible. Failures are
    returned, never raised, so one unavailable source cannot cancel the others.
    """
    pages: list[SourcePage] = []
    scanned, total_available, exhausted, pages_seen = 0, None, False, False
    try:
        for page in provider.iter_pages(request, cancel):
            if cancel.is_set():
                raise CancelledError()
            # Revalidate even frozen models: a provider's model_copy can
            # bypass its constructor. Reject a bad page before archiving it.
            try:
                page = SourcePage.model_validate(page.model_dump() if isinstance(page, SourcePage) else page)
            except (ValueError, TypeError):
                raise BackendError("invalid_response", "Источник вернул некорректную страницу документов.") from None
            if (exhausted or scanned + page.scanned > cap
                    or (not page.scanned and not page.exhausted)
                    or any(document.source != source for document in page.documents)):
                raise BackendError("invalid_response", "Источник нарушил формат или лимит выдачи.")
            pages_seen = True
            # Live sources may omit or reduce a later total. Neither an
            # empty page nor that reduction erases earlier known matches.
            if page.total_available is not None:
                total_available = max(total_available or 0, page.total_available)
            scanned += page.scanned
            exhausted = page.exhausted
            pages.append(page)
            if on_page is not None:
                # Paging a whole source takes minutes; a counter that only moves
                # when the source is finished tells the user nothing meanwhile.
                on_page(page.scanned)
        if not pages_seen:
            raise BackendError("invalid_response", "Источник не сообщил результат загрузки.")
    except CancelledError:
        return _FetchedQuery(cancelled=True)
    except BackendError as error:
        return _FetchedQuery(tuple(pages), scanned, total_available, exhausted, failure="source_" + error.code)
    except CredentialUnavailable:
        return _FetchedQuery(tuple(pages), scanned, total_available, exhausted,
                             failure="credential_storage_unavailable")
    except Exception:
        # A provider or its parser can fail outside the expected backend error
        # types. Keep already received pages and let other sources finish;
        # exception text may contain a URL or credentials, so expose only a
        # fixed reason code in coverage.
        if cancel.is_set():
            return _FetchedQuery(cancelled=True)
        return _FetchedQuery(tuple(pages), scanned, total_available, exhausted,
                             failure="source_error")
    return _FetchedQuery(tuple(pages), scanned, total_available, exhausted)


def _schedule_source_queries(groups: Mapping[str, Sequence[int]],
                             fetch_one: Callable[[str, int], _FetchedQuery], cancel: Event,
                             on_poll: Callable[[], None], *,
                             max_workers: int = _FETCH_WORKERS,
                             source_cooldowns: Mapping[str, float] | None = None) -> dict[int, _FetchedQuery]:
    """Fetch with a bounded, fair source queue and no overlap within a source.

    Each completed source goes behind sources still waiting for their first
    query. The caller owns providers and result admission; neither this queue
    nor its workers write to the archive or report progress to the coordinator.
    """
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("Source worker count must be positive")
    cooldowns = _SOURCE_QUERY_COOLDOWNS if source_cooldowns is None else source_cooldowns
    if any(isinstance(delay, bool) or not isinstance(delay, (int, float))
           or not math.isfinite(delay) or delay < 0 for delay in cooldowns.values()):
        raise ValueError("Source query cooldowns must be finite non-negative seconds")
    remaining = {source: deque(numbers) for source, numbers in groups.items() if numbers}
    if not remaining:
        return {}
    ready = deque(remaining)
    source_order = {source: number for number, source in enumerate(remaining)}
    next_ready_at: dict[str, float] = {}
    worker_count = min(max_workers, len(remaining))
    active: dict[Future[_FetchedQuery], tuple[str, int]] = {}
    results: dict[int, _FetchedQuery] = {}

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="source-fetch") as pool:
        def dispatch() -> None:
            # Scan at most one full round: cooling sources go to the back and
            # never hold a worker slot while unrelated sources can run.
            for _ in range(len(ready)):
                if len(active) >= worker_count or cancel.is_set():
                    break
                source = ready.popleft()
                if next_ready_at.get(source, 0.0) > monotonic():
                    ready.append(source)
                    continue
                number = remaining[source].popleft()
                active[pool.submit(fetch_one, source, number)] = source, number

        dispatch()
        while active or ready:
            if cancel.is_set():
                raise TaskCancelled()
            if active:
                timeout = 0.5
                if ready and len(active) < worker_count:
                    timeout = min(timeout, max(0.0, min(next_ready_at.get(source, 0.0)
                                                        for source in ready) - monotonic()))
                finished, _ = wait(active, timeout=timeout, return_when=FIRST_COMPLETED)
            else:
                delay = max(0.0, min(next_ready_at.get(source, 0.0)
                                     for source in ready) - monotonic())
                if cancel.wait(min(0.5, delay)):
                    raise TaskCancelled()
                finished = set()
            # Several completions in one poll should not make the next source
            # depend on Python's arbitrary set iteration order.
            for future in sorted(finished, key=lambda item: source_order[active[item][0]]):
                source, number = active.pop(future)
                answer = future.result()
                results[number] = answer
                if answer.cancelled:
                    cancel.set()
                elif remaining[source] and not cancel.is_set():
                    next_ready_at[source] = monotonic() + cooldowns.get(source, 0.0)
                    ready.append(source)
            on_poll()
            if cancel.is_set():
                raise TaskCancelled()
            dispatch()
    return results


def _fetch_queries(selected: Sequence[SearchQuery], caps: Sequence[int], context: RunContext,
                   credentials: CredentialStore,
                   provider_factory: Callable[[str], DocumentProvider] | None, *,
                   purpose: str, start: date, end: date) -> tuple[_FetchedQuery, ...]:
    """Ask every source at once, and each source one query at a time.

    Sources are independent of each other, so they are asked in parallel. A
    single source is not: several of our queries hitting OpenAlex together look
    like one client exceeding its rate, and it answers with a refusal that costs
    the run whole pages of documents. Within one source the queries therefore
    run in order, which is also how they were collected before.

    Providers are built on this thread, so credential access and any injected
    factory stay single-threaded; only the paging itself runs concurrently.
    """
    results: dict[int, _FetchedQuery] = {}
    providers: dict[int, DocumentProvider] = {}
    try:
        pending = [number for number, cap in enumerate(caps) if cap]
        for number in pending:
            context.check_cancelled()
            query = selected[number]
            try:
                providers[number] = (provider_factory(query.source) if provider_factory
                                     else make_provider(query.source, credentials))
            except CredentialUnavailable:
                results[number] = _FetchedQuery(failure="credential_storage_unavailable")
        ready = [number for number in pending if number in providers]
        groups: dict[str, list[int]] = {}
        for number in ready:
            groups.setdefault(selected[number].source, []).append(number)
        expected = sum(caps[number] for number in ready) or 1
        context.progress(purpose, "Запрашиваем документы у источников", 0, expected)

        received = [0]
        counter = Lock()

        def note(scanned: int) -> None:
            with counter:
                received[0] += scanned

        def fetch_one(source: str, number: int) -> _FetchedQuery:
            query = selected[number]
            request = SearchRequest.model_validate(dict(
                topic=query.text, source=source, from_date=start, until_date=end,
                max_results=caps[number]))
            return _fetch_query(providers[number], request, caps[number], source,
                                context.cancel_event, note)

        reported = -1

        def on_poll() -> None:
            nonlocal reported
            # Progress is reported from this thread only: the workers touch no
            # database, they just count what they received.
            with counter:
                scanned = received[0]
            if scanned != reported:
                reported = scanned
                context.progress(purpose, f"Запрашиваем документы: получено {scanned} из {expected}",
                                 min(scanned, expected), expected)

        results.update(_schedule_source_queries(groups, fetch_one, context.cancel_event, on_poll))
    finally:
        for provider in providers.values():
            provider.close()
    return tuple(results.get(number, _FetchedQuery()) for number in range(len(selected)))


def _refill_discovery_queries(
    selected: Sequence[SearchQuery], caps: Sequence[int], fetched: tuple[_FetchedQuery, ...],
    limit: int, start: date, end: date, context: RunContext, credentials: CredentialStore,
    provider_factory: Callable[[str], DocumentProvider] | None, rules_version: str,
) -> tuple[_FetchedQuery, ...]:
    """Spend unused discovery slots once, without exceeding the raw revision ceiling.

    A larger request starts at page one. Its records (apart from retrieval time)
    must begin with the original response before it can replace that response;
    only the replacement is admitted and counted in coverage. The first response
    remains usable if the source changes records or a retry fails.
    """
    records = [document for response in fetched for page in response.pages for document in page.documents]
    blocked = retracted_family_keys(records, check=context.check_cancelled, rules_version=rules_version)
    supporting = (supporting_asset_keys(records, rules_version=rules_version,
                                         check=context.check_cancelled)
                  if rules_version == STATUS_RULES else set())
    excluded_keys = blocked | supporting
    distinct: set[str] = set()
    source_ids: set[tuple[str, str]] = set()
    for response in fetched:
        for page in response.pages:
            for document in page.documents:
                context.check_cancelled()
                identity = (document.source, document.source_id)
                if (document.document_key in excluded_keys or identity in source_ids
                        or exclusion_reason(document, start, end, rules_version=rules_version)):
                    continue
                distinct.add(document.document_key)
                source_ids.add(identity)
    missing = max(0, limit - len(distinct))
    available_raw = _RECEIVED_REVISION_BUDGET - sum(response.scanned for response in fetched)
    eligible = [number for number, response in enumerate(fetched)
                if response.failure is None and not response.exhausted and response.scanned == caps[number]
                and 0 < caps[number] < 10_000]
    if not missing or not eligible or available_raw <= min(caps[number] for number in eligible):
        return fetched
    # Repeated requests include their original pages. If all can be repeated,
    # spread additional slots over sources; otherwise refill one query whose
    # first pass contributed the most distinct publication identities.
    if sum(caps[number] for number in eligible) + missing > available_raw:
        eligible = [max((number for number in eligible if caps[number] < available_raw), key=lambda number: (
            len({document.document_key for page in fetched[number].pages for document in page.documents}),
            -number))]
    new_caps = [0] * len(selected)
    for position, number in enumerate(eligible):
        raw_left = available_raw - sum(new_caps)
        extra = min(math.ceil(missing / (len(eligible) - position)), raw_left - caps[number],
                    10_000 - caps[number])
        if extra <= 0:
            continue
        new_caps[number] = caps[number] + extra
        missing -= extra
    if not any(new_caps):
        return fetched
    context.progress("discovery", "Добираем публикации из неполных поисковых запросов")
    retried = _fetch_queries(selected, new_caps, context, credentials, provider_factory,
                             purpose="discovery", start=start, end=end)
    result = list(fetched)
    for number, cap in enumerate(new_caps):
        if not cap:
            continue
        original, replacement = fetched[number], retried[number]
        old_records = tuple(document.model_dump(mode="json", exclude={"fetched_at"})
                            for page in original.pages for document in page.documents)
        new_records = tuple(document.model_dump(mode="json", exclude={"fetched_at"})
                            for page in replacement.pages for document in page.documents)
        if replacement.scanned >= original.scanned and new_records[:len(old_records)] == old_records:
            result[number] = _FetchedQuery(replacement.pages, replacement.scanned,
                max(original.total_available or 0, replacement.total_available or 0) or None,
                replacement.exhausted, replacement.failure)
        else:
            result[number] = _FetchedQuery(original.pages, original.scanned,
                original.total_available, original.exhausted,
                replacement.failure or "source_refill_not_reproducible")
    return tuple(result)


def _collect_snapshot(
    plan: QueryPlan, context: RunContext, archive: DocumentArchive,
    credentials: CredentialStore, *, purpose: Literal["discovery", "history"] = "discovery",
    queries: tuple[SearchQuery, ...] | None = None,
    max_documents: int | None = None,
    provider_factory: Callable[[str], DocumentProvider] | None = None,
    rules_version: str = STATUS_RULES,
) -> CorpusSnapshot:
    validate_status_rules(rules_version)
    preserve_received = rules_version == STATUS_RULES
    selected = queries if queries is not None else tuple(
        query for query in plan.queries if query.purpose == purpose and query.source in {"openalex", "crossref"})
    if not selected or any(query.source not in {"openalex", "crossref"} or query.purpose != purpose
                           for query in selected):
        raise ValueError("Publication queries with the correct purpose are required")
    limit = max_documents if max_documents is not None else plan.limits.discovery_documents
    if type(limit) is not int or not 1 <= limit <= 20000:
        raise ValueError("Invalid document limit")
    if purpose == "history" and any(query.source != "openalex" for query in selected):
        raise ValueError("Historical time series use the fixed OpenAlex reference source")
    start = date(plan.completed_years[-3] if purpose == "discovery" else plan.completed_years[0], 1, 1)
    end = plan.as_of if purpose == "discovery" else date(plan.completed_years[-1], 12, 31)
    years = tuple(range(start.year, end.year + 1))
    coverage: list[Coverage] = []
    revisions: dict[str, DocumentRevisionRef] = {}
    unique_studies: set[str] = set()
    seen_source_ids: set[tuple[str, str]] = set()
    blocked_studies: set[str] = set()
    received_documents: list[DocumentRecord] = []
    owners: dict[str, int] = {}
    # Equal discovery shares preserve subdirection diversity; history never
    # presents this truncated, relevance-sorted page budget as a full series.
    caps = _planned_caps(len(selected), limit, purpose=purpose,
                         budget=_RECEIVED_REVISION_BUDGET if preserve_received and purpose == "discovery" else None)
    fetched = _fetch_queries(selected, caps, context, credentials, provider_factory,
                             purpose=purpose, start=start, end=end)
    if purpose == "discovery" and preserve_received:
        fetched = _refill_discovery_queries(selected, caps, fetched, limit, start, end, context,
                                            credentials, provider_factory, rules_version)
    # Every request is already answered; admission below stays sequential and in
    # query order, so the archive, the counters and the snapshot are unchanged.
    for number, query in enumerate(selected):
        context.check_cancelled()
        cap = caps[number]
        received = fetched[number]
        scanned = accepted = rejected = unresolved = 0
        exhausted = limited = False
        total_available: int | None = None
        source_failed = False
        reasons: list[str] = []
        if len(unique_studies) >= limit or not cap:
            limited = True
            reasons.append("run_document_limit" if cap else "received_revision_budget")
        else:
            total_available = received.total_available
            exhausted = received.exhausted
            for page in received.pages:
                context.check_cancelled()
                scanned += page.scanned
                unresolved += page.skipped
                for document in page.documents:
                    context.check_cancelled()
                    received_documents.append(document)
                    if is_explicitly_retracted(document, rules_version=rules_version):
                        blocked_studies.add(document.document_key)
                    # Discovery retains contextual/supporting records for archiving;
                    # its eligibility pass separates them from primary mechanisms.
                    reason = exclusion_reason(document, start, end, primary_only=purpose == "history",
                                              rules_version=rules_version)
                    if (preserve_received and primary_research_exclusion(document, rules_version=rules_version)
                            == "supporting_asset_not_research" and reason is None):
                        reason = "supporting_asset_not_research"
                    # Keep received metadata independently of research admission.
                    # A later provider may be the only source of a supplement
                    # link or withdrawal notice; never copy its fields into the parent.
                    archived = None
                    if (preserve_received and reason not in {"unknown_publication_year", "outside_requested_years",
                                                              "outside_requested_dates"}
                            and (not document.publication_date or document.publication_date <= plan.as_of)):
                        archived = archive.put(document)
                        revisions[archived.revision_id] = archived
                        if reason == "retracted":
                            reasons.append("archived_status_revisions")
                        elif reason in {"supplement", "supporting_asset_not_research"}:
                            reasons.append("archived_supporting_revisions")
                    identity = document.source, document.source_id
                    if reason == "unknown_publication_year":
                        unresolved += 1
                    elif reason or identity in seen_source_ids or document.document_key in blocked_studies:
                        rejected += 1
                    elif document.document_key not in unique_studies and len(unique_studies) >= limit:
                        unresolved += 1
                        limited = True
                    else:
                        # Admission never changes the bytes, so a document already
                        # archived above yields the very same revision; archiving it
                        # twice only reread and recompared the file just written.
                        reference = archived if archived is not None else archive.put(document)
                        revisions[reference.revision_id] = reference
                        owners[reference.revision_id] = number
                        unique_studies.add(reference.study_id)
                        seen_source_ids.add(identity)
                        accepted += 1
                context.progress(purpose, f"{query.source}: документов в выборке {len(unique_studies)}",
                                 min(len(unique_studies), limit), limit)
            if received.failure is not None:
                source_failed = True
                reasons.append(received.failure)
            elif not exhausted:
                limited = scanned >= cap or limited
                reasons.append("pagination_not_exhausted")
        if unresolved:
            reasons.append("unresolved_records")
        if limited:
            reasons.append("document_limit")
        total_reconciled = total_available is None or scanned >= total_available
        if not total_reconciled:
            reasons.append("inconsistent_total")
        complete = exhausted and not limited and not unresolved and not source_failed and total_reconciled
        state: Literal["complete", "partial", "unavailable"] = (
            "complete" if complete else "partial" if scanned else "unavailable")
        coverage.append(Coverage(
            source=query.source, purpose=purpose, query_hash=content_hash(query), state=state,
            requested_years=years, completed_years=years if complete else (),
            pagination_exhausted=exhausted, comparable=complete and purpose == "history",
            scanned_records=scanned, accepted_records=accepted, rejected_records=rejected,
            unresolved_records=unresolved, limit_reached=limited,
            reasons=tuple(dict.fromkeys(reasons)) if not complete or preserve_received else (),
        ))
    # A later source may reveal a retraction that an earlier source omitted.
    # Retain rejected manifests in the identity check: otherwise a clean linked
    # journal version can survive a retracted preprint discarded during paging.
    blocked_studies.update(retracted_family_keys(received_documents, check=context.check_cancelled,
                                                  rules_version=rules_version))
    supporting_studies = supporting_asset_keys(received_documents, rules_version=rules_version,
        check=context.check_cancelled) if preserve_received else set()
    # Remove the whole known study family, adjusting its original query accounting.
    for revision_id, reference in list(revisions.items()):
        if reference.study_id in blocked_studies | supporting_studies and revision_id in owners:
            index = owners[revision_id]
            previous = coverage[index]
            coverage[index] = Coverage.model_validate(previous.model_dump() | {
                "accepted_records": previous.accepted_records - 1,
                "rejected_records": previous.rejected_records + 1,
                **({"reasons": tuple(dict.fromkeys((*previous.reasons,
                    "archived_status_revisions" if reference.study_id in blocked_studies else "archived_supporting_revisions")))}
                   if preserve_received else {}),
            })
            if not preserve_received:
                del revisions[revision_id]
    refs = tuple(sorted(revisions.values(), key=lambda ref: ref.revision_id))
    return CorpusSnapshot(
        snapshot_id=content_hash({"plan": plan.plan_hash, "purpose": purpose,
                                  "revisions": [ref.revision_id for ref in refs],
                                  "coverage": [item.model_dump(mode="json") for item in coverage]}),
        plan_hash=plan.plan_hash, purpose=purpose, created_at=datetime.now(UTC), as_of=plan.as_of,
        documents=refs, coverage=tuple(coverage), normalizer_version=("backend-v2-pilot-v3-publication-status"
            if preserve_received else "backend-v2-pilot-v2-primary-research"),
        deduplication_version="doi-source-id-v1",
    )


def collect_snapshot(
    plan: QueryPlan, context: RunContext, archive: DocumentArchive,
    credentials: CredentialStore, *, purpose: Literal["discovery", "history"] = "discovery",
    queries: tuple[SearchQuery, ...] | None = None,
    max_documents: int | None = None,
    provider_factory: Callable[[str], DocumentProvider] | None = None,
    rules_version: str = STATUS_RULES,
) -> CorpusSnapshot:
    """Collect with source-scoped keep-alive clients unless a test/provider is injected."""
    if provider_factory is not None:
        return _collect_snapshot(plan, context, archive, credentials, purpose=purpose, queries=queries,
                                 max_documents=max_documents, provider_factory=provider_factory,
                                 rules_version=rules_version)
    with PublicationProviderSession(credentials) as session:
        return _collect_snapshot(plan, context, archive, credentials, purpose=purpose, queries=queries,
                                 max_documents=max_documents, provider_factory=session.provider,
                                 rules_version=rules_version)
