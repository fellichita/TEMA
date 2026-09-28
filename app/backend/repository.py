"""SQLite-хранилище: короткие транзакции, неизменяемые ревизии, страницы результатов."""

import hashlib
import json
import os
import stat
from app.sqlite_runtime import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from app.backend.contracts import (
    CONTRACT_VERSION, TERMINAL_STATES, DocumentPage, DocumentRecord, DocumentSnapshot, JobRecord,
    SearchRequest, SourcePage, utc_now,
)
from app.backend.errors import BackendError
from app.backend.validation import validate_pagination, validate_search_query

SCHEMA_VERSION = 4
_SCHEMA = (
    """CREATE TABLE documents (
        document_key TEXT PRIMARY KEY, latest_revision TEXT REFERENCES revisions(revision_id),
        search_text TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL)""",
    """CREATE TABLE revisions (
        revision_id TEXT PRIMARY KEY, document_key TEXT NOT NULL REFERENCES documents(document_key),
        payload TEXT NOT NULL, created_at TEXT NOT NULL,
        UNIQUE(revision_id, document_key))""",
    """CREATE TABLE aliases (
        alias TEXT PRIMARY KEY, document_key TEXT NOT NULL REFERENCES documents(document_key))""",
    """CREATE TABLE jobs (
        id TEXT PRIMARY KEY, request_json TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('queued','running','succeeded','failed','cancelled','interrupted')),
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        scanned INTEGER NOT NULL DEFAULT 0 CHECK(scanned >= 0),
        stored INTEGER NOT NULL DEFAULT 0 CHECK(stored >= 0),
        skipped INTEGER NOT NULL DEFAULT 0 CHECK(skipped >= 0),
        total_available INTEGER, source_exhausted INTEGER NOT NULL DEFAULT 0,
        error_code TEXT, error_message TEXT)""",
    """CREATE TABLE job_documents (
        job_id TEXT NOT NULL REFERENCES jobs(id), document_key TEXT NOT NULL REFERENCES documents(document_key),
        revision_id TEXT NOT NULL, observed_at TEXT NOT NULL, search_text TEXT NOT NULL,
        PRIMARY KEY(job_id, document_key),
        FOREIGN KEY(revision_id, document_key) REFERENCES revisions(revision_id, document_key))""",
    "CREATE INDEX ix_revisions_document ON revisions(document_key)",
    "CREATE INDEX ix_jobs_created ON jobs(created_at DESC, id)",
    "CREATE INDEX ix_jobs_state ON jobs(state)",
    "CREATE INDEX ix_documents_updated ON documents(updated_at DESC, document_key)",
)


def _prepare_database_file(path: Path) -> None:
    """Create SQLite files with private modes and reject aliased existing files."""
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except FileExistsError:
        pass
    except OSError:
        raise BackendError("invalid_storage", "Файл базы данных недоступен или имеет небезопасный тип.") from None
    else:
        os.close(descriptor)
    try:
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OSError("Unsafe database file")
        if os.name != "nt":
            os.chmod(path, 0o600)
    except OSError:
        raise BackendError("invalid_storage", "Файл базы данных недоступен или имеет небезопасный тип.") from None


class Repository:
    def __init__(self, path: Path):
        if str(path) == ":memory:":
            raise BackendError("invalid_storage", "Требуется путь к постоянной базе SQLite.")
        selected = Path(path)
        if selected.is_symlink():
            raise BackendError("invalid_storage", "Файл базы данных не должен быть символической ссылкой.")
        self.path = selected.resolve()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _prepare_database_file(self.path)
        self.initialize()

    @contextmanager
    def _connection(self):
        connection = None
        try:
            connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.create_collation("UNICODE_NOCASE", lambda a, b: (a.casefold() > b.casefold()) - (a.casefold() < b.casefold()))
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("PRAGMA synchronous=FULL")
            yield connection
        except sqlite3.Error:
            raise BackendError("storage_error", "Ошибка локальной базы. Проверьте доступ и свободное место.") from None
        finally:
            if connection is not None:
                connection.close()

    @contextmanager
    def _transaction(self):
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def initialize(self) -> None:
        with self._connection() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise BackendError("unsupported_schema", "База создана более новой версией приложения.")
            if version == 0:
                tables = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
                if tables:
                    raise BackendError("unsupported_schema", "Обнаружена неизвестная схема базы данных.")
            # WAL относится к локальному файлу, а не сетевой папке.
            connection.execute("PRAGMA journal_mode=WAL")
            if 0 < version < SCHEMA_VERSION:
                backup_path = self.path.with_name(self.path.name + f".before-v{SCHEMA_VERSION}-{uuid4().hex[:8]}.bak")
                _prepare_database_file(backup_path)
                backup = sqlite3.connect(backup_path)
                try:
                    connection.backup(backup)
                finally:
                    backup.close()
        with self._transaction() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                for statement in _SCHEMA:
                    connection.execute(statement)
                version = 1
            if version == 1:
                connection.execute("ALTER TABLE jobs ADD COLUMN contract_version INTEGER NOT NULL DEFAULT 1")
                version = 2
            if version == 2:
                from app.backend.history import HISTORY_SCHEMA
                for statement in HISTORY_SCHEMA:
                    connection.execute(statement)
                version = 3
            if version == 3:
                connection.execute("ALTER TABLE history_periods ADD COLUMN parent_id TEXT REFERENCES history_periods(id)")
                connection.execute("ALTER TABLE history_periods ADD COLUMN granularity TEXT NOT NULL DEFAULT 'month'")
                connection.execute("ALTER TABLE history_periods ADD COLUMN is_split INTEGER NOT NULL DEFAULT 0 CHECK(is_split IN (0,1))")
                connection.execute("ALTER TABLE history_periods ADD COLUMN split_block_reason TEXT")
                connection.execute("CREATE INDEX ix_history_parent ON history_periods(parent_id)")
                # Existing plans must not suddenly expand their network workload on resume.
                for row in connection.execute("SELECT id,request_json FROM history_runs").fetchall():
                    request = json.loads(row["request_json"])
                    request["auto_split"] = False
                    connection.execute("UPDATE history_runs SET request_json=? WHERE id=?",
                                       (json.dumps(request, ensure_ascii=False), row["id"]))
                    connection.execute("UPDATE history_periods SET granularity=? WHERE run_id=?",
                                       (request["period"], row["id"]))
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @staticmethod
    def _job(connection, job_id: str) -> JobRecord:
        row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return Repository._job_record(row)

    @staticmethod
    def _job_record(row) -> JobRecord:
        if row is None:
            raise BackendError("job_not_found", "Задание не найдено.")
        values = dict(row)
        values["request"] = SearchRequest.model_validate_json(values.pop("request_json"))
        values["source_exhausted"] = bool(values["source_exhausted"])
        return JobRecord.model_validate(values)

    @staticmethod
    def _active(job: JobRecord) -> None:
        if job.state != "running":
            raise BackendError("invalid_state", "Операция разрешена только для выполняющегося задания.")

    def create_job(self, request: SearchRequest) -> JobRecord:
        request = SearchRequest.model_validate(request)
        with self._transaction() as connection:
            return self._create_job(connection, request)

    def _create_job(self, connection, request):
        job_id, now = str(uuid4()), utc_now().isoformat()
        connection.execute(
            "INSERT INTO jobs(id,request_json,state,created_at,updated_at,contract_version) VALUES(?,?,'queued',?,?,?)",
            (job_id, request.model_dump_json(), now, now, CONTRACT_VERSION),
        )
        return self._job(connection, job_id)

    def start_job(self, job_id: str) -> JobRecord:
        with self._transaction() as connection:
            if self._job(connection, job_id).state != "queued":
                raise BackendError("invalid_state", "Запустить можно только задание в очереди.")
            connection.execute("UPDATE jobs SET state='running',updated_at=? WHERE id=?",
                               (utc_now().isoformat(), job_id))
            return self._job(connection, job_id)

    @staticmethod
    def _store_document(connection, document: DocumentRecord, job_id: str) -> None:
        now = utc_now().isoformat()
        aliases = [f"source:{document.source}:{document.source_id}"]
        if document.doi:
            aliases.append(f"doi:{document.doi}")
        if document.patent_publication:
            aliases.append(f"patent:{document.patent_publication}")
        placeholders = ",".join("?" for _ in aliases)
        keys = {row[0] for row in connection.execute(
            f"SELECT document_key FROM aliases WHERE alias IN ({placeholders})", aliases
        )}
        if len(keys) > 1:
            raise BackendError("identity_conflict", "Источники связали разные документы одним идентификатором.")
        key = next(iter(keys)) if keys else document.document_key
        connection.execute(
            "INSERT INTO documents(document_key,updated_at) VALUES(?,?) ON CONFLICT DO NOTHING", (key, now)
        )
        for alias in aliases:
            connection.execute("INSERT INTO aliases(alias,document_key) VALUES(?,?) ON CONFLICT DO NOTHING",
                               (alias, key))
        # Preserve v1 hashes for unchanged scholarly records after adding patent fields.
        stable = document.model_dump(mode="json", exclude={"fetched_at", "patent_publication", "patent_family_id"})
        if document.patent_publication is not None:
            stable["patent_publication"] = document.patent_publication
        if document.patent_family_id is not None:
            stable["patent_family_id"] = document.patent_family_id
        serialized = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        revision = hashlib.sha256((key + "\n" + serialized).encode("utf-8")).hexdigest()
        connection.execute(
            "INSERT INTO revisions(revision_id,document_key,payload,created_at) VALUES(?,?,?,?) ON CONFLICT DO NOTHING",
            (revision, key, document.model_dump_json(), now),
        )
        search_text = (document.title + "\n" + (document.abstract or "")).casefold()
        connection.execute(
            "UPDATE documents SET latest_revision=?,search_text=?,updated_at=? WHERE document_key=?",
            (revision, search_text, now, key),
        )
        connection.execute(
            """INSERT INTO job_documents(job_id,document_key,revision_id,observed_at,search_text)
               VALUES(?,?,?,?,?) ON CONFLICT(job_id,document_key) DO NOTHING""",
            (job_id, key, revision, now, search_text),
        )

    def ingest_page(self, job_id: str, page: SourcePage) -> JobRecord:
        page = SourcePage.model_validate(page)
        with self._transaction() as connection:
            job = self._job(connection, job_id)
            self._active(job)
            if job.scanned + page.scanned > job.request.max_results:
                raise BackendError("result_limit", "Источник превысил установленный лимит записей.")
            for document in page.documents:
                self._store_document(connection, document, job_id)
            stored = connection.execute("SELECT COUNT(*) FROM job_documents WHERE job_id=?", (job_id,)).fetchone()[0]
            # Keep the largest advertised total for this attempt. A later page
            # may omit or reduce it as the live source changes, which must not
            # erase earlier evidence that the saved collection may be partial.
            totals = [value for value in (job.total_available, page.total_available) if value is not None]
            total_available = max(totals, default=None)
            connection.execute(
                """UPDATE jobs SET scanned=scanned+?, skipped=skipped+?, stored=?, total_available=?,
                   source_exhausted=?,updated_at=? WHERE id=?""",
                (page.scanned, page.skipped, stored, total_available, int(page.exhausted),
                 utc_now().isoformat(), job_id),
            )
            return self._job(connection, job_id)

    def finish_job(self, job_id: str, state: str, error_code=None, error_message=None) -> JobRecord:
        if state not in TERMINAL_STATES:
            raise BackendError("invalid_state", "Требуется конечное состояние задания.")
        with self._transaction() as connection:
            job = self._job(connection, job_id)
            if job.state in TERMINAL_STATES or (state == "succeeded" and job.state != "running"):
                raise BackendError("invalid_state", "Недопустимый переход состояния задания.")
            connection.execute(
                "UPDATE jobs SET state=?,error_code=?,error_message=?,updated_at=? WHERE id=?",
                (state, error_code, error_message, utc_now().isoformat(), job_id),
            )
            return self._job(connection, job_id)

    def get_job(self, job_id: str) -> JobRecord:
        with self._connection() as connection:
            return self._job(connection, job_id)

    def list_jobs(self, limit: int = 50) -> tuple[JobRecord, ...]:
        validate_pagination(limit)
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM jobs ORDER BY created_at DESC,id LIMIT ?", (limit,)).fetchall()
            return tuple(self._job_record(row) for row in rows)

    def list_documents(self, job_id: str | None = None, query: str | None = None,
                       limit: int = 100, offset: int = 0, *, history_id: str | None = None,
                       sort_by: str = "default", descending: bool = False) -> DocumentPage:
        validate_pagination(limit, offset)
        if sort_by not in ("default", "title", "date", "citations") or not isinstance(descending, bool):
            raise BackendError("invalid_query", "Неизвестный порядок сортировки документов.")
        if job_id is not None and history_id is not None:
            raise BackendError("invalid_query", "Укажите либо задание, либо исторический сбор.")
        validate_search_query(query)
        with self._connection() as connection:
            # Один read snapshot для COUNT и страницы, даже если worker дописывает данные.
            connection.execute("BEGIN")
            values = []
            conditions = []
            if history_id is not None:
                from app.backend.history import HistoryStore
                HistoryStore._run(connection, history_id)
                relation = """(SELECT document_key,revision_id,search_text,
                    ROW_NUMBER() OVER (PARTITION BY document_key ORDER BY observed_at DESC,job_id DESC) AS rn
                    FROM job_documents WHERE job_id IN
                    (SELECT latest_job_id FROM history_periods WHERE run_id=? AND is_split=0)) d
                    JOIN revisions r ON r.revision_id=d.revision_id"""
                values.append(history_id)
                conditions.append("d.rn=1")
                order = "d.document_key"
            elif job_id is not None:
                self._job(connection, job_id)
                relation = "job_documents d JOIN revisions r ON r.revision_id=d.revision_id"
                conditions.append("d.job_id=?")
                values.append(job_id)
                order = "d.document_key"
            else:
                relation = "documents d JOIN revisions r ON r.revision_id=d.latest_revision"
                order = "d.updated_at DESC,d.document_key"
            direction = "DESC" if descending else "ASC"
            if sort_by == "title":
                order = f"json_extract(r.payload, '$.title') COLLATE UNICODE_NOCASE {direction},d.document_key"
            elif sort_by == "citations":
                value = "json_extract(r.payload, '$.citation_count')"
                order = f"({value} IS NULL),{value} {direction},d.document_key"
            elif sort_by == "date":
                # Partial dates keep their precision; unknown components follow known ones.
                components = ["json_extract(r.payload, '$.publication_year')",
                              "COALESCE(json_extract(r.payload, '$.publication_month'), CAST(strftime('%m', json_extract(r.payload, '$.publication_date')) AS INTEGER))",
                              "CAST(strftime('%d', json_extract(r.payload, '$.publication_date')) AS INTEGER)"]
                order = ",".join(f"({value} IS NULL),{value} {direction}" for value in components) + ",d.document_key"
            if query:
                literal = query.casefold().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                conditions.append("d.search_text LIKE ? ESCAPE '\\'")
                values.append(f"%{literal}%")
            where = " WHERE " + " AND ".join(conditions) if conditions else ""
            if not query and history_id is None:
                # Committed job documents always reference a revision. A library
                # document is visible only after its latest revision is set.
                if job_id is None:
                    total = connection.execute(
                        "SELECT COUNT(*) FROM documents WHERE latest_revision IS NOT NULL"
                    ).fetchone()[0]
                else:
                    total = connection.execute(
                        "SELECT COUNT(*) FROM job_documents WHERE job_id=?", (job_id,)
                    ).fetchone()[0]
            else:
                total = connection.execute(f"SELECT COUNT(*) FROM {relation}{where}", values).fetchone()[0]
            rows = connection.execute(
                f"SELECT r.revision_id,d.document_key,r.payload FROM {relation}{where} ORDER BY {order} LIMIT ? OFFSET ?",
                (*values, limit, offset),
            ).fetchall()
            sources = self._sources(connection, [row["document_key"] for row in rows]) if job_id is None and history_id is None else {}
            items = []
            for row in rows:
                document = DocumentRecord.model_validate_json(row["payload"])
                items.append(DocumentSnapshot(
                    revision_id=row["revision_id"], document_key=row["document_key"],
                    document=document, sources=sources.get(row["document_key"], (document.source,)),
                ))
            return DocumentPage(items=tuple(items), total=total, limit=limit, offset=offset)

    @staticmethod
    def _sources(connection, keys):
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        values: dict[str, set[str]] = {}
        for row in connection.execute(
            f"SELECT document_key,alias FROM aliases WHERE document_key IN ({placeholders}) AND alias LIKE 'source:%'",
            keys,
        ):
            values.setdefault(row[0], set()).add(row[1].split(":", 2)[1])
        return {key: tuple(sorted(value)) for key, value in values.items()}

    def list_document_versions(self, document_key: str, limit=100, offset=0) -> DocumentPage:
        validate_pagination(limit, offset)
        with self._connection() as connection:
            connection.execute("BEGIN")
            if connection.execute("SELECT 1 FROM documents WHERE document_key=?", (document_key,)).fetchone() is None:
                raise BackendError("document_not_found", "Документ не найден.")
            total = connection.execute("SELECT COUNT(*) FROM revisions WHERE document_key=?", (document_key,)).fetchone()[0]
            rows = connection.execute(
                "SELECT revision_id,payload FROM revisions WHERE document_key=? ORDER BY created_at DESC,revision_id LIMIT ? OFFSET ?",
                (document_key, limit, offset),
            ).fetchall()
            items = []
            for row in rows:
                document = DocumentRecord.model_validate_json(row[1])
                items.append(DocumentSnapshot(
                    revision_id=row[0], document_key=document_key,
                    document=document, sources=(document.source,),
                ))
            return DocumentPage(items=tuple(items), total=total, limit=limit, offset=offset)

    def recover_interrupted(self) -> int:
        """Вызывать только после получения блокировки владельца backend."""
        with self._transaction() as connection:
            result = connection.execute(
                """UPDATE jobs SET state='interrupted',error_code='interrupted',
                   error_message='Предыдущий процесс завершился до окончания сбора.',updated_at=?
                   WHERE state IN ('queued','running')""", (utc_now().isoformat(),),
            )
            return result.rowcount
