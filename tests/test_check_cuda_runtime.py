"""Offline checks for the CUDA server preflight."""

from types import SimpleNamespace

import pytest

from app.pilot import encoder, local_llm
from scripts import check_cuda_runtime


@pytest.fixture
def fake_cuda(monkeypatch):
    events = []
    state = {"qwen_provider": "CUDAExecutionProvider", "e5_provider": "CUDAExecutionProvider",
             "e5_error": None}

    monkeypatch.setattr(check_cuda_runtime.sys, "platform", "linux")
    monkeypatch.setattr(
        check_cuda_runtime, "_installed_version",
        lambda name: "1.29.0" if name == "onnxruntime-gpu" else None,
    )
    monkeypatch.setattr(check_cuda_runtime, "_nvidia_smi", lambda: None)
    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])

    class FakeQwen:
        def __init__(self):
            self.provider = state["qwen_provider"]
            events.append(("qwen_loaded", check_cuda_runtime.os.environ.get(
                "TRENDANALIZER_INFERENCE_PROVIDER")))

        def generate(self, **kwargs):
            events.append(("qwen_generated", kwargs["max_new_tokens"]))
            return SimpleNamespace(completion_tokens=2)

        def close(self):
            events.append(("qwen_closed", None))

    class FakeE5:
        def __init__(self):
            self.provider = state["e5_provider"]
            events.append(("e5_loaded", check_cuda_runtime.os.environ.get(
                "TRENDANALIZER_INFERENCE_PROVIDER")))

        def encode(self, texts, *, kind):
            events.append(("e5_encoded", (texts, kind)))
            if state["e5_error"] is not None:
                raise state["e5_error"]
            return SimpleNamespace(shape=(1, 384))

    monkeypatch.setattr(local_llm, "LocalInstructModel", FakeQwen)
    monkeypatch.setattr(encoder, "MultilingualEncoder", FakeE5)
    return state, events


def test_preflight_runs_both_models_on_cuda_and_restores_provider(fake_cuda, monkeypatch, capsys):
    _, events = fake_cuda
    monkeypatch.setenv("TRENDANALIZER_INFERENCE_PROVIDER", "cpu")

    assert check_cuda_runtime.main() == 0

    assert events == [
        ("qwen_loaded", "cuda"), ("qwen_generated", 2), ("qwen_closed", None),
        ("e5_loaded", "cuda"),
        ("e5_encoded", (["Scientific research on optical computing."], "query")),
    ]
    assert check_cuda_runtime.os.environ["TRENDANALIZER_INFERENCE_PROVIDER"] == "cpu"
    assert "Qwen и E5 запущены через CUDAExecutionProvider" in capsys.readouterr().out


def test_preflight_rejects_e5_cpu_without_inference(fake_cuda, monkeypatch, capsys):
    state, events = fake_cuda
    state["e5_provider"] = "CPUExecutionProvider"
    monkeypatch.delenv("TRENDANALIZER_INFERENCE_PROVIDER", raising=False)

    assert check_cuda_runtime.main() == 1

    assert ("qwen_closed", None) in events
    assert not any(name == "e5_encoded" for name, _ in events)
    assert "TRENDANALIZER_INFERENCE_PROVIDER" not in check_cuda_runtime.os.environ
    assert "E5 запущена через CPUExecutionProvider" in capsys.readouterr().err


def test_preflight_rejects_e5_execution_failure(fake_cuda, capsys):
    state, events = fake_cuda
    state["e5_error"] = encoder.EncoderError("GPU execution failed")

    assert check_cuda_runtime.main() == 1

    assert any(name == "e5_encoded" for name, _ in events)
    output = capsys.readouterr()
    assert "Проверка E5 на CUDA не прошла" in output.err
    assert "OK:" not in output.out


def test_preflight_stops_before_e5_when_qwen_falls_back(fake_cuda, capsys):
    state, events = fake_cuda
    state["qwen_provider"] = "CPUExecutionProvider"

    assert check_cuda_runtime.main() == 1

    assert ("qwen_closed", None) in events
    assert not any(name == "e5_loaded" for name, _ in events)
    assert "Qwen запущена через CPUExecutionProvider" in capsys.readouterr().err
