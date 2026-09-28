"""Source-first Atom imports preserve version availability and real revision refs."""

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from threading import Event

import pytest

from app.pilot.archive import DocumentArchive
from app.pilot.contracts import CorpusSnapshot, Coverage, QueryPlan, SearchQuery, content_hash
from app.pilot.multisource.arxiv import build_arxiv_corpus_snapshot, import_arxiv_discovery
from app.pilot.enrichment import import_arxiv_atom
from app.pilot.multisource.contracts import ArxivImportReceipt, ArxivVersion, SourceSnapshot
from app.pilot.multisource.queries import build_manual_profile
from app.pilot.multisource.store import SignalStore
from app.pilot.service import PilotService
from app.runtime.credentials import CredentialStore
from app.runtime.jobs import TaskCancelled, TaskFailure


NOW = datetime(2026, 9, 19, tzinfo=timezone.utc)
ATOM = 'http://www.w3.org/2005/Atom'
ARXIV = 'http://arxiv.org/schemas/atom'
SCOPE = "advanced membrane separation"


def _entry(version: int, *, published: str = "2025-01-15T10:00:00Z",
           updated: str = "2025-01-15T10:00:00Z", comment: str = "") -> str:
    note = f"<arxiv:comment>{comment}</arxiv:comment>" if comment else ""
    return (f"<entry><id>https://arxiv.org/abs/2501.12345v{version}</id>"
            f"<published>{published}</published><updated>{updated}</updated>"
            "<title>Advanced membrane separation process</title>"
            "<summary>Experimental selective transport using a new membrane.</summary>"
            f"<author><name>A Researcher</name></author>{note}</entry>")


def _feed(path: Path, *entries: str) -> Path:
    path.write_text(f'<feed xmlns="{ATOM}" xmlns:arxiv="{ARXIV}">' + "".join(entries) + "</feed>",
                    encoding="utf-8")
    return path


def _plan(as_of: date) -> QueryPlan:
    years = tuple(range(as_of.year - 6, as_of.year))
    return QueryPlan(original_query=SCOPE, language="en", english_query=SCOPE, definition=SCOPE,
                     subdirections=(SCOPE,), queries=(SearchQuery(source="openalex", text=SCOPE),),
                     completed_years=years, as_of=as_of, planner_version="test")


def _setup(tmp_path: Path):
    data = tmp_path / "app"
    store = SignalStore(data)
    archive = DocumentArchive(data / "revisions")
    profile = build_manual_profile(SCOPE, SCOPE, seed_terms=(SCOPE,), primary_phrase=SCOPE, confirmed_at=NOW)
    return store, archive, store.put_object(profile)


def test_all_admissible_versions_are_archived_before_family_selection(tmp_path: Path) -> None:
    store, archive, profile_hash = _setup(tmp_path)
    path = _feed(tmp_path / "export.atom", _entry(1), _entry(2, updated="2025-02-01T10:00:00Z"))
    receipt_hash = import_arxiv_discovery(store, archive, path, profile_hash, as_of=date(2026, 9, 19),
                                          retention="local_allowed", observed_at=NOW)
    receipt = store.get_object(receipt_hash, ArxivImportReceipt)
    assert receipt.scanned == 2 and len(receipt.version_hashes) == 2 and len(receipt.selected_revision_ids) == 1
    assert store.get_object(receipt.snapshot_hash, SourceSnapshot).comparable is False
    versions = [store.get_object(item, ArxivVersion) for item in receipt.version_hashes]
    assert [item.version for item in versions] == [1, 2]
    early = build_arxiv_corpus_snapshot(store, archive, receipt_hash, _plan(date(2025, 1, 20)))
    late = build_arxiv_corpus_snapshot(store, archive, receipt_hash, _plan(date(2026, 9, 19)))
    assert len(early.documents) == len(late.documents) == 1
    assert early.documents[0].revision_id == versions[0].revision_id
    assert late.documents[0].revision_id == versions[1].revision_id
    assert archive.get(early.documents[0].revision_id).url.endswith("v1")
    assert early.coverage[0].state == "partial" and early.coverage[0].comparable is False


def test_late_version_and_withdrawal_do_not_leak_into_past_discovery(tmp_path: Path) -> None:
    store, archive, profile_hash = _setup(tmp_path)
    path = _feed(tmp_path / "export.atom", _entry(1),
                 _entry(2, updated="2025-02-01T10:00:00Z", comment="Withdrawn by authors"))
    receipt_hash = import_arxiv_discovery(store, archive, path, profile_hash, as_of=date(2026, 9, 19),
                                          retention="local_allowed", observed_at=NOW)
    receipt = store.get_object(receipt_hash, ArxivImportReceipt)
    assert receipt.selected_revision_ids == ()
    assert len(build_arxiv_corpus_snapshot(store, archive, receipt_hash, _plan(date(2025, 1, 20))).documents) == 1
    assert build_arxiv_corpus_snapshot(store, archive, receipt_hash, _plan(date(2026, 9, 19))).documents == ()


def test_conflicting_publication_dates_exclude_family_and_wrong_scope_fails(tmp_path: Path) -> None:
    store, archive, profile_hash = _setup(tmp_path)
    path = _feed(tmp_path / "export.atom", _entry(1),
                 _entry(2, published="2025-01-16T10:00:00Z", updated="2025-02-01T10:00:00Z"))
    receipt_hash = import_arxiv_discovery(store, archive, path, profile_hash, as_of=date(2026, 9, 19),
                                          retention="local_allowed", observed_at=NOW)
    assert store.get_object(receipt_hash, ArxivImportReceipt).selected_revision_ids == ()
    wrong = _plan(date(2026, 9, 19)).model_copy(update={"original_query": "unrelated technology"})
    with pytest.raises(TaskFailure):
        build_arxiv_corpus_snapshot(store, archive, receipt_hash, wrong)


def test_invalid_atom_rights_and_cancellation_fail_before_publication(tmp_path: Path) -> None:
    store, archive, profile_hash = _setup(tmp_path)
    path = _feed(tmp_path / "export.atom", _entry(1))
    with pytest.raises(TaskFailure):
        import_arxiv_discovery(store, archive, path, profile_hash, as_of=date(2026, 9, 19),
                               retention="unknown", observed_at=NOW)
    cancel = Event()
    cancel.set()
    with pytest.raises(TaskCancelled):
        import_arxiv_discovery(store, archive, path, profile_hash, as_of=date(2026, 9, 19),
                               retention="local_allowed", observed_at=NOW, cancel=cancel)
    path.write_text('<!DOCTYPE x [<!ENTITY x SYSTEM "file:///etc/passwd">]><feed>&x;</feed>')
    with pytest.raises(ValueError):
        import_arxiv_discovery(store, archive, path, profile_hash, as_of=date(2026, 9, 19),
                               retention="local_allowed", observed_at=NOW)
    assert not (store.root / "catalogue.json").exists()


def test_local_calendar_day_is_valid_during_utc_date_rollover(tmp_path: Path, monkeypatch) -> None:
    class PreviousUtcDay(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime.combine(date.today() - timedelta(days=1), time(23, 30), tzinfo=timezone.utc)
            return value.astimezone(tz) if tz is not None else value.replace(tzinfo=None)

    store, archive, profile_hash = _setup(tmp_path)
    path = _feed(tmp_path / "export.atom", _entry(1))
    monkeypatch.setattr("app.pilot.multisource.arxiv.datetime", PreviousUtcDay)
    monkeypatch.setattr("app.pilot.enrichment.datetime", PreviousUtcDay)
    receipt = import_arxiv_discovery(store, archive, path, profile_hash, as_of=date.today(),
                                     retention="local_allowed", observed_at=NOW)
    assert store.get_object(receipt, ArxivImportReceipt).scanned == 1
    assert len(import_arxiv_atom(path, as_of=date.today(), cancel=Event()).documents) == 1


def test_seed_revisions_enter_actual_scientific_discovery_input(tmp_path: Path, monkeypatch) -> None:
    credentials = CredentialStore()
    monkeypatch.setattr(credentials, "get", lambda _: None)
    monkeypatch.setattr("app.pilot.encoder.verify_artifacts", lambda _, **kwargs: None)
    service = PilotService(tmp_path / "app", credentials)
    try:
        store = SignalStore(service.data_dir)
        profile_hash = store.put_object(build_manual_profile(SCOPE, SCOPE, seed_terms=(SCOPE,),
                                                              primary_phrase=SCOPE, confirmed_at=NOW))
        path = _feed(tmp_path / "export.atom", _entry(1))
        receipt_hash = import_arxiv_discovery(store, service.archive, path, profile_hash,
                                              as_of=date(2026, 9, 19), retention="local_allowed", observed_at=NOW)
        receipt = store.get_object(receipt_hash, ArxivImportReceipt)

        def empty_base(plan, *_):
            return CorpusSnapshot(snapshot_id="empty-base", plan_hash=plan.plan_hash, purpose="discovery",
                                  as_of=plan.as_of, created_at=NOW, documents=(),
                                  coverage=(Coverage(source="openalex", purpose="discovery",
                                      query_hash=content_hash(plan.queries[0]), state="unavailable",
                                      requested_years=plan.completed_years, pagination_exhausted=False,
                                      comparable=False, reasons=("test_source_unavailable",)),),
                                  normalizer_version="fixture", deduplication_version="doi-source-id-v1")

        def inspect_discovery(_, snapshot, plan):
            assert snapshot.plan_hash == plan.plan_hash
            assert receipt.selected_revision_ids[0] in {item.revision_id for item in snapshot.documents}
            assert snapshot.coverage[-1].source == "arxiv"
            raise TaskFailure("seed reached discovery")

        monkeypatch.setattr("app.pilot.service.collect_snapshot", empty_base)
        monkeypatch.setattr(service, "_discover", inspect_discovery)
        run_id = service.start(SCOPE, SCOPE, seed_arxiv_receipt_hash=receipt_hash)
        service.coordinator.wait()
        assert "seed reached discovery" in service.get(run_id)["error"]
    finally:
        service.close()
