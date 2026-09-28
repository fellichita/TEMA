"""Integration boundaries: source isolation, identity, historical schema migration."""

import hashlib
import json
import sqlite3
from threading import Event, Lock

import pytest
from pydantic import ValidationError

from app.backend.config import BackendSettings
from app.backend.contracts import DocumentRecord, SearchRequest, SourcePage
from app.backend.errors import BackendError
from app.backend.repository import Repository, SCHEMA_VERSION, _SCHEMA
from app.backend.service import Backend


def record(source="crossref", **changes):
    return DocumentRecord.model_validate(dict(
        source=source, source_id="10.1234/test" if source == "crossref" else "W123",
        doi="10.1234/test", title="Original title", url="https://doi.org/10.1234/test",
    ) | changes)


class Provider:
    def __init__(self, source, fail=False):
        self.source, self.fail, self.closed = source, fail, False

    def iter_pages(self, request, cancel):
        assert request.source == self.source
        if self.fail:
            raise BackendError("credentials_required", "Credentials required")
        yield SourcePage(documents=(record(self.source),), scanned=1, total_available=1, exhausted=True)

    def close(self):
        self.closed = True


def ingest(repo, document):
    job = repo.create_job(SearchRequest(topic="test", source=document.source))
    repo.start_job(job.id)
    repo.ingest_page(job.id, SourcePage(documents=(document,), scanned=1, exhausted=True))
    repo.finish_job(job.id, "succeeded")
    return job.id


def test_multisource_deduplicates_and_keeps_source_specific_snapshots(tmp_path):
    providers = {source: Provider(source) for source in ("crossref", "openalex")}
    with Backend(BackendSettings(data_dir=tmp_path), provider_factories={
        source: lambda provider=provider: provider for source, provider in providers.items()
    }) as backend:
        jobs = backend.collect_many(SearchRequest(topic="test"))
        assert all(job.state == "succeeded" and job.stored == 1 for job in jobs)
        results = backend.list_documents()
        assert results.total == 1
        assert results.items[0].sources == ("crossref", "openalex")
        assert results.items[0].document.source == "openalex"
        versions = backend.list_document_versions(results.items[0].document_key)
        assert versions.total == 2
        assert {item.document.source for item in versions.items} == {"crossref", "openalex"}
        for job in jobs:
            snapshot = backend.list_documents(job_id=job.id).items[0]
            assert snapshot.sources == (job.request.source,)
            assert snapshot.document.source == job.request.source
    assert all(provider.closed for provider in providers.values())


def test_failed_source_does_not_prevent_next_source(tmp_path):
    providers = {"epo": Provider("epo", fail=True), "openalex": Provider("openalex")}
    with Backend(BackendSettings(data_dir=tmp_path), provider_factories={
        source: lambda provider=provider: provider for source, provider in providers.items()
    }) as backend:
        failed, succeeded = backend.collect_many(SearchRequest(topic="test"), ("epo", "openalex"))
        assert failed.state == "failed" and failed.error_code == "credentials_required"
        assert succeeded.state == "succeeded" and succeeded.stored == 1
        assert backend.list_documents().total == 1
    assert all(provider.closed for provider in providers.values())


def test_independent_sources_collect_concurrently_and_keep_requested_order(tmp_path):
    entered = set()
    entered_lock = Lock()
    both_entered = Event()
    release = Event()

    class BlockingProvider(Provider):
        def iter_pages(self, request, cancel):
            with entered_lock:
                entered.add(self.source)
                if entered == {"crossref", "openalex"}:
                    both_entered.set()
            assert release.wait(3)
            yield SourcePage(documents=(record(self.source),), scanned=1,
                             total_available=1, exhausted=True)

    providers = {source: BlockingProvider(source) for source in ("crossref", "openalex")}
    with Backend(BackendSettings(data_dir=tmp_path), provider_factories={
        source: lambda provider=provider: provider for source, provider in providers.items()
    }) as backend:
        job_ids = backend.submit_collections(SearchRequest(topic="test"))
        try:
            assert both_entered.wait(2), "independent source requests did not overlap"
        finally:
            release.set()
        jobs = tuple(backend.wait(job_id, timeout=3) for job_id in job_ids)
        assert tuple(job.request.source for job in jobs) == ("crossref", "openalex")
        assert all(job.state == "succeeded" for job in jobs)


@pytest.mark.parametrize("sources", [(), "crossref", ("crossref", "crossref"), ("crossref", "unknown")])
def test_invalid_source_group_has_no_partial_jobs(tmp_path, sources):
    with Backend(BackendSettings(data_dir=tmp_path)) as backend:
        with pytest.raises((BackendError, ValidationError)):
            backend.submit_collections(SearchRequest(topic="test"), sources)
        assert backend.list_jobs() == ()


def test_group_queue_capacity_is_checked_before_any_job(tmp_path):
    with Backend(BackendSettings(data_dir=tmp_path, max_pending_jobs=1)) as backend:
        with pytest.raises(BackendError) as error:
            backend.submit_collections(SearchRequest(topic="test"))
        assert error.value.code == "queue_full"
        assert backend.list_jobs() == ()


def test_patent_publication_identity_keeps_kinds_and_family_members_separate(tmp_path):
    repo = Repository(tmp_path / "test.sqlite3")
    def patent(number, source_id):
        return record("epo", source_id=source_id, doi=None, document_type="patent",
                      patent_publication=number, patent_family_id="12345")
    ingest(repo, patent("EP 1234567 A1", "a"))
    ingest(repo, patent("ep-1234567-a1", "another-source-id"))
    ingest(repo, patent("EP1234567B1", "b"))
    ingest(repo, patent("US20240000001A1", "us"))
    assert repo.list_documents().total == 3
    assert repo.list_document_versions("patent:EP1234567A1").total == 2


@pytest.mark.parametrize("changes", [
    {"patent_publication": "EP1234567"},
    {"patent_publication": "EP1234567A1"},
    {"patent_family_id": "family"},
])
def test_invalid_patent_identity_is_rejected(changes):
    with pytest.raises(ValidationError):
        record(**changes)


def test_version_pagination_and_missing_document(tmp_path):
    repo = Repository(tmp_path / "test.sqlite3")
    ingest(repo, record())
    ingest(repo, record("openalex"))
    first = repo.list_document_versions("doi:10.1234/test", limit=1)
    second = repo.list_document_versions("doi:10.1234/test", limit=1, offset=1)
    assert first.total == second.total == 2
    assert first.items[0].revision_id != second.items[0].revision_id
    with pytest.raises(BackendError) as error:
        repo.list_document_versions("missing")
    assert error.value.code == "document_not_found"
    with pytest.raises(BackendError):
        repo.list_document_versions("doi:10.1234/test", limit=0)


def test_v1_migration_backs_up_and_preserves_payload_hash_and_job(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    doc = record()
    legacy = doc.model_dump(mode="json", exclude={"patent_publication", "patent_family_id"})
    payload = json.dumps(legacy, ensure_ascii=False)
    stable = {key: value for key, value in legacy.items() if key != "fetched_at"}
    serialized = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    revision = hashlib.sha256((doc.document_key + "\n" + serialized).encode()).hexdigest()
    now = doc.fetched_at.isoformat()
    with sqlite3.connect(path) as connection:
        for statement in _SCHEMA:
            connection.execute(statement)
        connection.execute("PRAGMA user_version=1")
        connection.execute("INSERT INTO documents VALUES(?,?,?,?)", (doc.document_key, revision, "original title", now))
        connection.execute("INSERT INTO revisions VALUES(?,?,?,?)", (revision, doc.document_key, payload, now))
        connection.executemany("INSERT INTO aliases VALUES(?,?)", [
            (doc.document_key, doc.document_key), ("source:crossref:10.1234/test", doc.document_key),
        ])
        connection.execute("INSERT INTO jobs(id,request_json,state,created_at,updated_at,scanned,stored) VALUES(?,?,'succeeded',?,?,1,1)",
                           ("old-job", SearchRequest(topic="test").model_dump_json(), now, now))
        connection.execute("INSERT INTO job_documents VALUES(?,?,?,?,?)",
                           ("old-job", doc.document_key, revision, now, "original title"))
    repo = Repository(path)
    assert repo.get_job("old-job").contract_version == 1
    assert repo.list_documents(job_id="old-job").items[0].revision_id == revision
    backups = list(tmp_path.glob(f"legacy.sqlite3.before-v{SCHEMA_VERSION}-*.bak"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute("PRAGMA user_version").fetchone()[0] == 1
        assert backup.execute("SELECT payload FROM revisions").fetchone()[0] == payload
        assert backup.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    job_id = ingest(repo, doc)
    assert repo.get_job(job_id).contract_version == 2
    assert repo.list_document_versions(doc.document_key).total == 1
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT payload FROM revisions").fetchone()[0] == payload
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    Repository(path)
    assert len(list(tmp_path.glob(f"legacy.sqlite3.before-v{SCHEMA_VERSION}-*.bak"))) == 1


def test_source_listing_does_not_expose_keys(monkeypatch):
    from app.runtime.credentials import CredentialStore

    for name in ("OPENALEX_API_KEY", "EPO_OPS_KEY", "EPO_OPS_SECRET"):
        monkeypatch.setenv(name, "PRIVATE-TEST-KEY")
    store = CredentialStore()
    store.import_legacy_environment()
    sources = Backend.sources(store)
    assert sources[1]["key_configured"] and sources[2]["credentials_configured"]
    assert "PRIVATE-TEST-KEY" not in json.dumps(sources)
