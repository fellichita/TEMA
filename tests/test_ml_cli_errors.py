"""CLI failures stay readable and never create a result file."""

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from app.backend.config import BackendSettings
from app.backend.service import Backend
from app.ml import __main__ as cli
from app.ml import service
from app.ml.contracts import AnalysisInputError
from tests.mvp_fixture import snapshot


ROOT = Path(__file__).resolve().parents[1]
TOPIC = "photonic neuromorphic computing"


def run_cli(tmp_path, *args):
    environment = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run(
        [sys.executable, "-m", "app.ml", *map(str, args)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )


def assert_input_failure(completed, output, message):
    assert completed.returncode == 2, completed.stderr
    assert completed.stdout == ""
    assert message in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize("topic_args", [[], ["--topic", TOPIC]])
def test_unknown_history_is_a_readable_cli_error(tmp_path, topic_args):
    output = tmp_path / "result.json"
    completed = run_cli(
        tmp_path, "--history-id", "unknown-history", "--data-dir", tmp_path / "backend",
        *topic_args, "--output", output,
    )
    assert_input_failure(completed, output, "Исторический сбор не найден.")


def test_busy_backend_is_a_readable_cli_error(tmp_path):
    data_dir = tmp_path / "backend"
    output = tmp_path / "result.json"
    with Backend(BackendSettings(data_dir=data_dir)):
        completed = run_cli(
            tmp_path, "--history-id", "unknown-history", "--data-dir", data_dir, "--output", output,
        )
    assert_input_failure(completed, output, "Этот каталог данных уже используется другим backend.")


def test_valid_snapshot_still_exports_analysis(tmp_path):
    source = tmp_path / "snapshot.json"
    source.write_text(json.dumps(snapshot()), encoding="utf-8")
    original = source.read_bytes()
    output = tmp_path / "result.json"

    completed = run_cli(tmp_path, "--snapshot", source, "--output", output)

    assert completed.returncode == 0, completed.stderr
    assert "Результат:" in completed.stdout
    assert "Traceback" not in completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert all(isinstance(result[key], list) for key in (
        "candidates", "preliminary_signals", "established", "excluded_off_direction",
    ))
    assert any(result[key] for key in ("candidates", "preliminary_signals", "established"))
    assert source.read_bytes() == original


def test_missing_snapshot_stays_a_readable_cli_error(tmp_path):
    source = tmp_path / "missing.json"
    output = tmp_path / "result.json"
    completed = run_cli(tmp_path, "--snapshot", source, "--output", output)
    assert_input_failure(completed, output, "missing.json")


def test_programming_errors_are_not_hidden(tmp_path, monkeypatch):
    def fail_analysis(*args, **kwargs):
        raise RuntimeError("unexpected implementation error")

    monkeypatch.setattr(cli, "run_analysis", fail_analysis)
    output = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="unexpected implementation error"):
        cli.main(["--snapshot", str(tmp_path / "snapshot.json"), "--topic", TOPIC, "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize(("malformation", "message"), [
    ("missing_revision_id", "Ожидается JSON исторического корпуса"),
    ("numeric_abstract", "document.abstract"),
])
def test_malformed_snapshot_with_explicit_topic_fails_before_analysis(tmp_path, monkeypatch, malformation, message):
    data = snapshot()
    entry = data["batches"][0]["documents"][0]
    if malformation == "missing_revision_id":
        del entry["revision_id"]
    else:
        entry["document"]["abstract"] = 123
    source = tmp_path / "malformed.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    original = source.read_bytes()
    output = tmp_path / "result.json"

    completed = run_cli(tmp_path, "--snapshot", source, "--topic", TOPIC, "--output", output)
    assert_input_failure(completed, output, message)

    analyze = Mock(side_effect=AssertionError("Malformed snapshot reached analysis"))
    monkeypatch.setattr(service, "analyze", analyze)
    with pytest.raises(AnalysisInputError, match=message):
        service.run_analysis(None, {"topic": TOPIC}, snapshot_path=source)
    analyze.assert_not_called()
    assert source.read_bytes() == original
