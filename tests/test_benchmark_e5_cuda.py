"""Offline batch benchmark tests; no E5 weights, network or NVIDIA card needed."""

import os

import numpy as np
import pytest

from app.runtime.inference import BATCH_VARIABLE, PROVIDER_VARIABLE
from scripts import benchmark_e5_cuda


def test_benchmark_compares_three_cuda_batches_and_restores_environment(monkeypatch):
    monkeypatch.setenv(PROVIDER_VARIABLE, "cpu")
    monkeypatch.setenv(BATCH_VARIABLE, "7")
    instances = []

    class FakeEncoder:
        provider = "CUDAExecutionProvider"

        def __init__(self):
            assert os.environ[PROVIDER_VARIABLE] == "cuda"
            self.size = int(os.environ[BATCH_VARIABLE])
            self.fingerprint = f"cuda-batch-{self.size}"
            self.calls = 0
            instances.append(self)

        def encode(self, texts, *, kind):
            assert kind == "passage"
            self.calls += 1
            result = np.zeros((len(texts), 384), dtype=np.float32)
            result[:, 0] = 1
            result[:, 1] = self.size * 1e-6
            result /= np.linalg.norm(result, axis=1, keepdims=True)
            return result

    report = benchmark_e5_cuda.benchmark(warmups=1, repeats=2, model_factory=FakeEncoder)
    assert report["benchmark"] == "e5-cuda-batch-v1"
    assert report["texts"] == 128
    assert len(report["corpus_sha256"]) == 64
    assert [row["batch_size"] for row in report["measurements"]] == [8, 16, 32]
    assert [row["fingerprint"] for row in report["measurements"]] == [
        "cuda-batch-8", "cuda-batch-16", "cuda-batch-32"]
    assert report["measurements"][0]["max_abs_diff_vs_batch_8"] == 0
    assert report["measurements"][1]["max_abs_diff_vs_batch_8"] > 0
    assert report["measurements"][2]["max_abs_diff_vs_batch_8"] > report["measurements"][1][
        "max_abs_diff_vs_batch_8"]
    assert [instance.calls for instance in instances] == [3, 3, 3]
    assert os.environ[PROVIDER_VARIABLE] == "cpu"
    assert os.environ[BATCH_VARIABLE] == "7"


def test_benchmark_rejects_cpu_fallback_and_restores_environment(monkeypatch):
    monkeypatch.delenv(PROVIDER_VARIABLE, raising=False)
    monkeypatch.delenv(BATCH_VARIABLE, raising=False)

    class FellBack:
        provider = "CPUExecutionProvider"

    with pytest.raises(RuntimeError, match="требуется CUDAExecutionProvider"):
        benchmark_e5_cuda.benchmark(warmups=0, repeats=1, model_factory=FellBack)
    assert PROVIDER_VARIABLE not in os.environ
    assert BATCH_VARIABLE not in os.environ


@pytest.mark.parametrize("warmups,repeats", [(-1, 1), (0, 0), (1.5, 1), (0, True)])
def test_benchmark_rejects_invalid_repeat_counts_before_loading_model(warmups, repeats):
    with pytest.raises(ValueError):
        benchmark_e5_cuda.benchmark(warmups=warmups, repeats=repeats,
                                    model_factory=lambda: pytest.fail("model loaded"))


def test_cli_prints_json_with_fake_cuda_encoder(monkeypatch, capsys):
    class FakeEncoder:
        provider = "CUDAExecutionProvider"
        fingerprint = "fake-cuda"

        def encode(self, texts, *, kind):
            result = np.zeros((len(texts), 384), dtype=np.float32)
            result[:, 0] = 1
            return result

    monkeypatch.setattr(benchmark_e5_cuda, "MultilingualEncoder", FakeEncoder)
    assert benchmark_e5_cuda.main(["--warmups", "0", "--repeats", "1"]) == 0
    output = capsys.readouterr()
    assert '"benchmark": "e5-cuda-batch-v1"' in output.out
    assert output.err == ""
