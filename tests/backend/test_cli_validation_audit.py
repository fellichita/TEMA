"""Audit A13: invalid CLI arguments are diagnosed before opening storage."""

import subprocess
import sys

import pytest

from app.backend import __main__ as cli
from app.backend.errors import BackendError

INVALID_ARGUMENTS = [
    (["collect", "valid topic", "--from-date", "2025-12-01", "--until-date", "2024-01-01"],
     "Начало периода позже окончания"),
    (["collect", "valid topic", "--until-date", "2999-01-01"], "будущем"),
    (["collect", "valid topic", "--from-date", "PRIVATE_INPUT"], "from_date"),
    (["collect", "valid topic", "--limit", "0"], "max_results"),
    (["collect", "valid topic", "--limit", "10001"], "max_results"),
    (["collect", "valid topic", "--sources", "crossref", "crossref"], "разных источников"),
    (["documents", "--limit", "0"], "Размер страницы"),
    (["documents", "--limit", "1001"], "Размер страницы"),
    (["documents", "--offset", "-1"], "смещение"),
    (["documents", "--query", "PRIVATE_INPUT" * 50], "Поисковая строка"),
    (["versions", "doi:10.1234/test", "--limit", "0"], "Размер страницы"),
    (["versions", "doi:10.1234/test", "--offset", "-1"], "смещение"),
    (["collect-history", "valid topic", "--from-date", "2025-12-01", "--until-date", "2024-01-01"],
     "Начало периода позже окончания"),
    (["collect-history", "valid topic", "--from-date", "2024-01-01", "--sources", "crossref", "crossref"],
     "разных источников"),
    (["collect-history", "valid topic", "--from-date", "2024-01-01", "--max-periods", "0"], "max_periods"),
]


@pytest.mark.parametrize("arguments,reason", INVALID_ARGUMENTS)
def test_invalid_arguments_return_two_and_create_no_storage(tmp_path, capsys, arguments, reason):
    path = tmp_path / "not-created"
    assert cli.main(["--data-dir", str(path), *arguments]) == 2
    captured = capsys.readouterr()
    assert reason in captured.err
    assert "PRIVATE_INPUT" not in captured.err and "Traceback" not in captured.err
    assert not captured.out and not path.exists()


@pytest.mark.parametrize("arguments,reason", INVALID_ARGUMENTS)
def test_real_process_exit_codes_and_safe_diagnostics(tmp_path, arguments, reason):
    path = tmp_path / "not-created"
    completed = subprocess.run(
        [sys.executable, "-m", "app.backend", "--data-dir", str(path), *arguments],
        capture_output=True, text=True, encoding="utf-8", timeout=15, check=False,
    )
    assert completed.returncode == 2
    assert reason in completed.stderr
    assert "PRIVATE_INPUT" not in completed.stderr and "Traceback" not in completed.stderr
    assert not completed.stdout and not path.exists()


def test_nul_query_api_to_cli_is_rejected_before_storage(tmp_path, capsys):
    path = tmp_path / "not-created"
    assert cli.main(["--data-dir", str(path), "documents", "--query", "\x00PRIVATE_INPUT"]) == 2
    captured = capsys.readouterr()
    assert "invalid_query" in captured.err and "PRIVATE_INPUT" not in captured.err
    assert not path.exists()


@pytest.mark.parametrize("code", [
    "invalid_response", "invalid_credentials", "rate_limited", "storage_error", "history_not_found",
])
def test_backend_failures_are_not_misclassified_as_invalid_arguments(tmp_path, capsys, monkeypatch, code):
    def failure(settings):
        raise BackendError(code, "Безопасная ошибка backend.")

    monkeypatch.setattr(cli, "Backend", failure)
    assert cli.main(["--data-dir", str(tmp_path), "documents"]) == 1
    assert code in capsys.readouterr().err
