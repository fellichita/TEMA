"""Semantic mode is explicit, local, and never installs a model during analysis."""

from unittest.mock import Mock

import pytest

from app.ml import __main__ as cli
from app.ml.contracts import AnalysisInputError


def arguments(tmp_path, *extra):
    return ["--snapshot", str(tmp_path / "corpus.json"), "--topic", "solid state batteries",
            "--output", str(tmp_path / "result.json"), *extra]


@pytest.fixture
def analysis(monkeypatch):
    result = {key: [] for key in ("candidates", "preliminary_signals", "established", "excluded_off_direction")}
    analyze = Mock(return_value=result)
    export = Mock()
    monkeypatch.setattr(cli, "run_analysis", analyze)
    monkeypatch.setattr(cli, "export_result", export)
    return analyze, export


def test_default_cli_mode_stays_lexical_without_model_path(tmp_path, analysis):
    analyze, export = analysis
    assert cli.main(arguments(tmp_path)) == 0
    assert analyze.call_args.args[1]["relevance_mode"] == "lexical"
    assert "model_dir" not in analyze.call_args.kwargs
    export.assert_called_once()


@pytest.mark.parametrize("path", [None, "weights with spaces"])
def test_semantic_cli_forwards_only_explicit_local_directory(tmp_path, analysis, path):
    analyze, export = analysis
    args = ["--relevance-mode", "semantic"]
    if path is not None:
        args += ["--model-dir", str(tmp_path / path)]
    assert cli.main(arguments(tmp_path, *args)) == 0
    assert analyze.call_args.args[1]["relevance_mode"] == "semantic"
    assert analyze.call_args.kwargs.get("model_dir") == (tmp_path / path if path is not None else None)
    export.assert_called_once()


@pytest.mark.parametrize("mode", [[], ["--relevance-mode", "lexical"]])
def test_model_directory_requires_semantic_mode_before_analysis(tmp_path, analysis, capsys, mode):
    analyze, export = analysis
    with pytest.raises(SystemExit) as error:
        cli.main(arguments(tmp_path, *mode, "--model-dir", str(tmp_path / "weights")))
    assert error.value.code == 2
    assert "--relevance-mode semantic" in capsys.readouterr().err
    analyze.assert_not_called()
    export.assert_not_called()


def test_missing_weights_are_not_silently_replaced_with_lexical_analysis(tmp_path, analysis, capsys):
    analyze, export = analysis
    analyze.side_effect = AnalysisInputError("Локальная модель не установлена. python -m scripts.install_ml_model")
    assert cli.main(arguments(tmp_path, "--relevance-mode", "semantic")) == 2
    assert "scripts.install_ml_model" in capsys.readouterr().err
    assert analyze.call_count == 1
    export.assert_not_called()
    assert not (tmp_path / "result.json").exists()


def test_missing_semantic_dependencies_explain_explicit_installation(tmp_path, analysis, capsys):
    analyze, export = analysis
    analyze.side_effect = ImportError("onnxruntime")
    assert cli.main(arguments(tmp_path, "--relevance-mode", "semantic")) == 2
    assert "requirements/semantic.lock" in capsys.readouterr().err
    export.assert_not_called()
