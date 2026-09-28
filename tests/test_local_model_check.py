"""The offline smoke reports diagnostics without turning examples into calibration."""

import json
from copy import deepcopy

import numpy as np
import pytest

from app.ml.contracts import AnalysisInputError
from scripts import check_local_model as check


class FakeEncoder:
    def __init__(self, *, inverted=False, changing=False):
        self.calls = []
        self.inverted, self.changing = inverted, changing
        cases = json.loads(check.CASES_PATH.read_text(encoding="utf-8"))["queries"]
        self.vectors = {}
        for index, case in enumerate(cases):
            vector = np.zeros(384, dtype=np.float32)
            vector[index] = 1
            other = np.zeros(384, dtype=np.float32)
            other[10 + index] = 1
            self.vectors[("query", case["query"])] = vector
            for document in case["documents"]:
                text = check.passage_text(document)
                higher = document["id"] != case["smoke_expected"]["lower"]
                self.vectors[("passage", text)] = vector if higher != inverted else other

    def manifest(self):
        return {"model_id": "fake", "revision": "test", "dimensions": 384}

    def encode(self, texts, kind="passage"):
        self.calls.append(kind)
        rows = np.asarray([self.vectors[(kind, text)] for text in texts])
        if self.changing and len(self.calls) > 2:
            rows = np.roll(rows, 1, axis=1)
        return rows


def test_smoke_keeps_unvalidated_adversarial_scores_outside_quality_gate():
    encoder = FakeEncoder()
    result = check.check_local_model(encoder=encoder)
    assert result["ok"]
    assert result["calibrated"] is False
    assert result["scientific_quality_evaluation"] is False
    assert result["checks"]["smoke_pairs_passed"] == 5
    assert result["checks"]["smoke_pairs_total"] == 5
    assert encoder.calls == ["query", "passage", "query", "passage"]
    adversarial = [d for q in result["queries"] for d in q["documents"]
                   if d["kind"] == "adversarial_unvalidated"]
    assert len(adversarial) == 2
    assert all(d["similarity"] == 1 for d in adversarial)
    assert all("lexical_decision" in d for d in adversarial)
    assert "precision" not in result and "threshold" not in result


def test_failed_smoke_pair_is_reported_without_changing_thresholds_or_hiding_scores():
    result = check.check_local_model(encoder=FakeEncoder(inverted=True))
    assert result["ok"] is False
    assert result["checks"]["smoke_pairs_passed"] == 0
    assert len(result["queries"]) == 5
    assert all(q["smoke_expected"]["passed"] is False for q in result["queries"])


def test_nondeterministic_vectors_fail_repeatability_even_when_similarities_match():
    result = check.check_local_model(encoder=FakeEncoder(changing=True))
    assert result["checks"]["repeatable_embeddings"] is False
    assert result["ok"] is False


@pytest.mark.parametrize("bad", [np.zeros((5, 383)), np.full((5, 384), np.nan),
                                  np.zeros((5, 384)), np.ones((5, 384))])
def test_bad_encoder_shape_finiteness_or_normalization_is_a_readable_error(bad):
    encoder = FakeEncoder()
    encoder.encode = lambda texts, kind: bad
    with pytest.raises(AnalysisInputError, match="эмбеддинг"):
        check.check_local_model(encoder=encoder)


def test_stable_report_fingerprint_excludes_resource_measurements():
    first = check.check_local_model(encoder=FakeEncoder())
    second = check.check_local_model(encoder=FakeEncoder())
    assert first["report_fingerprint"] == second["report_fingerprint"]
    assert first["fixtures_sha256"] == second["fixtures_sha256"]
    assert first["measurements"]["process_peak_rss_bytes"] is None or first["measurements"]["process_peak_rss_bytes"] > 0


def test_cli_writes_atomic_json_without_overwriting_existing_result(tmp_path, monkeypatch, capsys):
    report = check.check_local_model(encoder=FakeEncoder())
    monkeypatch.setattr(check, "check_local_model", lambda **kwargs: deepcopy(report))
    output = tmp_path / "smoke.json"
    assert check.main(["--output", str(output), "--model-dir", str(tmp_path / "model")]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == report
    before = output.read_bytes()
    monkeypatch.setattr(check, "check_local_model", lambda **kwargs: pytest.fail("Existing output should fail before inference"))
    assert check.main(["--output", str(output)]) == 2
    assert output.read_bytes() == before
    assert "существует" in capsys.readouterr().err


def test_cli_missing_model_does_not_write_report_or_install(tmp_path, monkeypatch, capsys):
    def missing(**kwargs):
        raise AnalysisInputError("Модель отсутствует; python -m scripts.install_ml_model")
    monkeypatch.setattr(check, "check_local_model", missing)
    output = tmp_path / "missing.json"
    assert check.main(["--output", str(output)]) == 2
    assert not output.exists()
    assert "install_ml_model" in capsys.readouterr().err


def test_cli_returns_failure_status_with_visible_negative_report(monkeypatch, capsys):
    report = check.check_local_model(encoder=FakeEncoder(inverted=True))
    monkeypatch.setattr(check, "check_local_model", lambda **kwargs: report)
    assert check.main([]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
