import os
import sqlite3
import stat
from datetime import datetime, timezone

import pytest

from app.backend.contracts import DocumentRecord, SearchRequest, SourcePage
from app.backend.errors import BackendError
from app.backend.repository import Repository


@pytest.fixture
def repository(tmp_path):
    return Repository(tmp_path / "test.sqlite3")


def document(**changes):
    return DocumentRecord.model_validate({
        "source": "crossref", "source_id": "10.1234/test", "doi": "10.1234/test",
        "title": "Нейроморфные вычисления", "url": "https://doi.org/10.1234/test",
        **changes,
    })


def running(repository, max_results=100):
    job = repository.create_job(SearchRequest(topic="ИИ", max_results=max_results))
    repository.start_job(job.id)
    return job.id


def ingest(repository, job_id, *documents):
    return repository.ingest_page(job_id, SourcePage(
        documents=documents, scanned=len(documents), total_available=len(documents), exhausted=True,
    ))


def test_duplicates_across_runs_share_document_and_unchanged_revision(repository):
    first, second = running(repository), running(repository)
    ingest(repository, first, document(fetched_at=datetime(2024, 1, 1, tzinfo=timezone.utc)))
    ingest(repository, second, document(fetched_at=datetime(2024, 2, 1, tzinfo=timezone.utc)))
    old = repository.list_documents(job_id=first).items[0]
    new = repository.list_documents(job_id=second).items[0]
    assert old.revision_id == new.revision_id
    assert repository.list_documents().total == 1


def test_updated_record_does_not_change_old_run(repository):
    first, second = running(repository), running(repository)
    ingest(repository, first, document(title="Старое название"))
    ingest(repository, second, document(title="Новое название"))
    assert repository.list_documents(job_id=first).items[0].document.title == "Старое название"
    assert repository.list_documents().items[0].document.title == "Новое название"
    assert repository.list_documents(job_id=first, query="старое").total == 1
    assert repository.list_documents(job_id=first, query="новое").total == 0


def test_unversioned_document_is_hidden_from_library_count(repository):
    with repository._transaction() as connection:
        connection.execute(
            "INSERT INTO documents(document_key,updated_at) VALUES(?,?)",
            ("unversioned", "2024-01-01T00:00:00+00:00"),
        )
    page = repository.list_documents()
    assert page.total == 0 and page.items == ()


def test_first_version_within_a_run_is_kept(repository):
    job_id = running(repository)
    ingest(repository, job_id, document(title="Первая версия"))
    job = ingest(repository, job_id, document(title="Вторая версия"))
    assert job.scanned == 2 and job.stored == 1
    assert repository.list_documents(job_id=job_id).items[0].document.title == "Первая версия"


def test_doi_deduplicates_across_sources(repository):
    first, second = running(repository), running(repository)
    ingest(repository, first, document())
    ingest(repository, second, document(source="another", source_id="abc", doi="DOI:10.1234/TEST"))
    assert repository.list_documents().total == 1


def test_adding_doi_preserves_initial_key_and_references(repository):
    first, second = running(repository), running(repository)
    ingest(repository, first, document(source_id="local-1", doi=None))
    old = repository.list_documents(job_id=first).items[0]
    ingest(repository, second, document(source_id="local-1"))
    new = repository.list_documents(job_id=second).items[0]
    assert old.document_key == new.document_key
    assert old.document.doi is None and new.document.doi == "10.1234/test"
    assert repository.list_documents().total == 1


def test_identity_conflict_rolls_back_whole_page(repository):
    initial = running(repository)
    ingest(repository, initial, document(source_id="a", doi=None))
    ingest(repository, initial, document(source_id="b", doi="10.1234/b"))
    job_id = running(repository)
    with pytest.raises(BackendError) as error:
        ingest(repository, job_id,
               document(source_id="new", doi="10.1234/new"),
               document(source_id="a", doi="10.1234/b"))
    assert error.value.code == "identity_conflict"
    assert repository.list_documents().total == 2
    assert repository.list_documents(job_id=job_id).total == 0
    assert repository.get_job(job_id).scanned == 0


def test_pagination_unicode_and_literal_sql_metacharacters(repository):
    job_id = running(repository)
    ingest(repository, job_id, document(title="Скидка 100%_test"),
           document(doi="10.1234/second", source_id="second", title="ВТОРОЙ ДОКУМЕНТ"))
    assert repository.list_documents(query="второй").total == 1
    assert repository.list_documents(query="%_").total == 1
    assert repository.list_documents(query="%' OR 1=1 --").total == 0
    first = repository.list_documents(job_id=job_id, limit=1, offset=0)
    second = repository.list_documents(job_id=job_id, limit=1, offset=1)
    assert first.total == second.total == 2
    assert first.items[0].document_key != second.items[0].document_key


@pytest.mark.parametrize("limit,offset", [(0, 0), (1001, 0), (1, -1), (True, 0), (1, True)])
def test_invalid_pagination(repository, limit, offset):
    with pytest.raises(BackendError):
        repository.list_documents(limit=limit, offset=offset)


def test_only_valid_state_transitions_are_allowed(repository):
    job = repository.create_job(SearchRequest(topic="ИИ"))
    with pytest.raises(BackendError):
        ingest(repository, job.id, document())
    with pytest.raises(BackendError):
        repository.finish_job(job.id, "succeeded")
    repository.start_job(job.id)
    repository.finish_job(job.id, "cancelled")
    with pytest.raises(BackendError):
        repository.finish_job(job.id, "succeeded")
    with pytest.raises(BackendError):
        repository.start_job(job.id)


def test_list_jobs_preserves_order_and_limit(repository):
    jobs = [repository.create_job(SearchRequest(topic=f"Тема {index}")) for index in range(3)]
    with repository._transaction() as connection:
        for index, job in enumerate(jobs):
            connection.execute(
                "UPDATE jobs SET created_at=? WHERE id=?",
                (f"2024-01-0{index + 1}T00:00:00+00:00", job.id),
            )
    listed = repository.list_jobs(limit=2)
    assert [job.id for job in listed] == [jobs[2].id, jobs[1].id]
    assert [job.request.topic for job in listed] == ["Тема 2", "Тема 1"]


def test_missing_job_is_not_confused_with_empty_result(repository):
    with pytest.raises(BackendError) as error:
        repository.list_documents(job_id="missing")
    assert error.value.code == "job_not_found"


def test_limit_is_enforced_before_any_write(repository):
    job_id = running(repository, max_results=1)
    with pytest.raises(BackendError):
        ingest(repository, job_id, document(), document(doi="10.1234/other"))
    assert repository.list_documents().total == 0


def test_recovery_only_changes_unfinished_jobs(repository):
    queued = repository.create_job(SearchRequest(topic="ИИ"))
    active = running(repository)
    done = running(repository)
    repository.finish_job(done, "succeeded")
    assert repository.recover_interrupted() == 2
    assert repository.get_job(queued.id).state == "interrupted"
    assert repository.get_job(active).state == "interrupted"
    assert repository.get_job(done).state == "succeeded"
    assert repository.recover_interrupted() == 0


def test_future_schema_rejected_without_modification(tmp_path):
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(BackendError) as error:
        Repository(path)
    assert error.value.code == "unsupported_schema"
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 99


def test_unrelated_database_is_not_adopted(tmp_path):
    path = tmp_path / "unrelated.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated(value TEXT)")
    with pytest.raises(BackendError) as error:
        Repository(path)
    assert error.value.code == "unsupported_schema"


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_database_alias_cannot_open_or_modify_an_external_file(tmp_path, alias):
    outside = tmp_path / "outside.sqlite3"
    with sqlite3.connect(outside) as connection:
        connection.execute("CREATE TABLE sentinel(value TEXT)")
    database = tmp_path / "data" / "documents.sqlite3"
    database.parent.mkdir()
    if alias == "symlink":
        try:
            database.symlink_to(outside)
        except OSError:
            pytest.skip("Symlinks are unavailable on this filesystem")
    else:
        try:
            os.link(outside, database)
        except OSError:
            pytest.skip("Hard links are unavailable on this filesystem")
    with pytest.raises(BackendError) as error:
        Repository(database)
    assert error.value.code == "invalid_storage"
    with sqlite3.connect(outside) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='sentinel'").fetchone() is not None


@pytest.mark.skipif(os.name == "nt", reason="Windows uses profile ACLs rather than POSIX file modes")
def test_new_database_is_private(tmp_path):
    path = tmp_path / "data" / "documents.sqlite3"
    Repository(path)
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("scope", ["all", "job"])
def test_sorting_precedes_pagination_and_keeps_unknown_values_last(repository, scope):
    job_id = running(repository)
    records = [document(source_id=str(i), doi=f"10.1234/{i}", title=title,
                        citation_count=count, publication_year=year,
                        date_precision="year" if year else "unknown")
               for i, (title, count, year) in enumerate([
                   ("яблоко", None, None), ("Бета", 2, 2023), ("альфа", 10, 2024)])]
    ingest(repository, job_id, *records)
    options = {"job_id": job_id} if scope == "job" else {}
    for field, descending, expected in [
        ("title", False, [2, 1, 0]), ("title", True, [0, 1, 2]),
        ("citations", False, [1, 2, 0]), ("citations", True, [2, 1, 0]),
        ("date", False, [1, 2, 0]), ("date", True, [2, 1, 0]),
    ]:
        pages = [repository.list_documents(**options, sort_by=field, descending=descending,
                                           limit=1, offset=i) for i in range(3)]
        assert [p.items[0].document.source_id for p in pages] == list(map(str, expected))
        assert all(p.total == 3 for p in pages)
    result = repository.list_documents(**options, query="АЛЬФА", sort_by="citations", descending=True)
    assert result.total == 1 and result.items[0].document.title == "альфа"


def test_sorting_rejects_untrusted_parameters(repository):
    for options in ({"sort_by": "title; DROP TABLE documents"}, {"descending": "DESC"}):
        with pytest.raises(BackendError) as error:
            repository.list_documents(**options)
        assert error.value.code == "invalid_query"


def test_date_sort_preserves_partial_dates(repository):
    job_id = running(repository)
    ingest(repository, job_id,
           document(source_id="year", doi="10.1234/year", publication_year=2024, date_precision="year"),
           document(source_id="month", doi="10.1234/month", publication_year=2024,
                    publication_month=5, date_precision="month"),
           document(source_id="day", doi="10.1234/day", publication_year=2024,
                    publication_date="2024-05-02", date_precision="day"))
    for descending in (True, False):
        result = repository.list_documents(sort_by="date", descending=descending)
        assert [item.document.source_id for item in result.items] == ["day", "month", "year"]
        assert result.items[-1].document.publication_date is None
