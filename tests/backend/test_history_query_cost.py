"""Progress reads scale in data size, not in separate SQL round trips."""

from contextlib import contextmanager
from datetime import date

from app.backend.contracts import SearchRequest
from app.backend.history import HistoryRequest, HistoryStore
from app.backend.repository import Repository


def test_large_history_uses_four_selects_and_preserves_attempts(tmp_path, monkeypatch):
    repository = Repository(tmp_path / 'history.sqlite3')
    history = HistoryStore(repository)
    run_id = history.create(HistoryRequest(topic='robotics', from_date=date(1925, 1, 1),
        until_date=date(2024, 12, 31), period='month', sources=('crossref',)))
    expected = {}
    # Fixture construction is batched; the production read is real SQLite.
    with repository._transaction() as connection:
        periods = connection.execute('SELECT * FROM history_periods WHERE run_id=? ORDER BY ordinal', (run_id,)).fetchall()
        for period in periods:
            job = repository._create_job(connection, SearchRequest(topic='robotics', source=period['source'],
                from_date=period['from_date'], until_date=period['until_date']))
            connection.execute('INSERT INTO history_attempts VALUES(?,?,1)', (period['id'], job.id))
            connection.execute('UPDATE history_periods SET latest_job_id=? WHERE id=?', (job.id, period['id']))
            expected[period['id']] = job
    statements = []
    original_connection = repository._connection

    @contextmanager
    def traced_connection():
        with original_connection() as connection:
            connection.set_trace_callback(statements.append)
            yield connection

    monkeypatch.setattr(repository, '_connection', traced_connection)
    report = history.report(run_id)
    assert len(report.periods) == 1200
    assert all(period.job == expected[period.id] and period.attempts == (expected[period.id].id,)
               for period in report.periods)
    selects = [statement for statement in statements if statement.lstrip().upper().startswith('SELECT')]
    assert len(selects) == 4
    assert not report.coverage_complete
