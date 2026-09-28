"""Offline encoder integrity, pooling, batching and explicit installer boundary."""

import os
import subprocess
import sys
from concurrent.futures import CancelledError
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from app.ml import local_encoder
from app.ml.contracts import AnalysisInputError
from scripts import install_ml_model
from tests.platform_support import require_symlinks


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    spec = local_encoder.load_spec()
    for item in spec["files"]:
        payload = item["name"].encode()
        (tmp_path / item["name"]).write_bytes(payload)
        item.update(bytes=len(payload), sha256=sha256(payload).hexdigest())
    monkeypatch.setattr(local_encoder, "load_spec", lambda: spec)
    return tmp_path, spec


class Tokenizer:
    texts: ClassVar[list[str]] = []

    @classmethod
    def from_file(cls, path):
        return cls()

    def enable_truncation(self, **kwargs):
        assert kwargs == {"max_length": 512}

    def enable_padding(self, **kwargs):
        assert kwargs == {"pad_id": 0, "pad_token": "[PAD]"}

    def encode_batch(self, texts):
        self.texts.extend(texts)
        return [SimpleNamespace(ids=[1, 2, 0], attention_mask=[1, 1, 0], type_ids=[0, 0, 0])
                for _ in texts]


class Session:
    batches: ClassVar[list[int]] = []

    def __init__(self, path, sess_options, providers, enable_fallback):
        assert providers == ["CPUExecutionProvider"]
        assert enable_fallback is False

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in ("input_ids", "attention_mask", "token_type_ids")]

    def get_outputs(self):
        return [SimpleNamespace(name="last_hidden_state")]

    def run(self, names, feeds):
        assert names == ["last_hidden_state"]
        self.batches.append(len(feeds["input_ids"]))
        assert all(value.dtype == np.int64 for value in feeds.values())
        hidden = np.zeros((self.batches[-1], 3, 384), dtype=np.float32)
        hidden[:, 0, 0], hidden[:, 1, 1], hidden[:, 2, 2] = 3, 4, 999
        return [hidden]


class NativeFail(Exception):
    pass


class NativeCancelled(Exception):
    pass


@pytest.fixture
def runtime(monkeypatch):
    Tokenizer.texts, Session.batches = [], []
    native = SimpleNamespace(**{name: NativeFail for name in (
        "Fail", "InvalidArgument", "NoSuchFile", "NoModel", "EngineError", "RuntimeException", "InvalidProtobuf",
        "ModelLoaded", "NotImplemented", "InvalidGraph", "EPFail", "ModelRequiresCompilation", "NotFound", "DeviceReset")},
        ModelLoadCanceled=NativeCancelled)
    ort = SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=Session,
                          capi=SimpleNamespace(onnxruntime_pybind11_state=native))
    monkeypatch.setattr(local_encoder, "_load_runtime", lambda: (np, ort, Tokenizer))
    return ort


def test_integrity_precedes_loading_runtime(artifacts, monkeypatch):
    path, _ = artifacts
    (path / "model.onnx").write_bytes(b"wrong")
    monkeypatch.setattr(local_encoder, "_load_runtime", lambda: pytest.fail("Unverified weights loaded"))
    with pytest.raises(AnalysisInputError, match="model.onnx"):
        local_encoder.LocalEncoder(path)


def test_missing_artifact_is_explicit_and_never_downloaded(artifacts, monkeypatch):
    path, _ = artifacts
    (path / "tokenizer.json").unlink()
    monkeypatch.setattr(local_encoder, "_load_runtime", lambda: pytest.fail("Missing artifact accepted"))
    with pytest.raises(AnalysisInputError, match="install_ml_model"):
        local_encoder.LocalEncoder(path)


def test_legacy_encoder_rejects_symlinked_weight_even_with_matching_bytes(artifacts):
    require_symlinks()
    path, spec = artifacts
    model = path / "model.onnx"
    target = path / "original-weights"
    model.rename(target)
    model.symlink_to(target)
    with pytest.raises(AnalysisInputError, match="model.onnx"):
        local_encoder.verify_artifacts(path, spec)


def test_mean_pooling_masks_padding_normalizes_and_bounds_batches(artifacts, runtime, monkeypatch):
    # Размер батча подбирается по машине; здесь он закреплён, чтобы проверять
    # саму разбивку на батчи, а не производительность конкретного процессора.
    monkeypatch.setenv("TRENDANALIZER_INFERENCE_BATCH", "8")
    encoder = local_encoder.LocalEncoder(artifacts[0])
    progress = []
    result = encoder.encode(["a document"] * 10, progress=lambda n, total: progress.append((n, total)))
    assert result.shape == (10, 384) and result.dtype == np.float32
    np.testing.assert_allclose(result[:, :3], [[.6, .8, 0]] * 10)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), np.ones(10))
    assert Session.batches == [8, 2]
    assert Tokenizer.texts == ["passage: a document"] * 10
    assert progress == [(8, 10), (10, 10)]
    encoder.encode(["a direction"], kind="query")
    assert Tokenizer.texts[-1] == "query: a direction"


def test_empty_batch_shape_and_invalid_input(artifacts, runtime):
    encoder = local_encoder.LocalEncoder(artifacts[0])
    assert encoder.encode([]).shape == (0, 384)
    for texts in ([""], ["  "], [None], "a string"):
        with pytest.raises(AnalysisInputError):
            encoder.encode(texts)
    for kind in ("invalid", None, []):
        with pytest.raises(AnalysisInputError):
            encoder.encode(["text"], kind=kind)


def test_cancellation_between_batches_stops_remaining_inference(artifacts, runtime, monkeypatch):
    monkeypatch.setenv("TRENDANALIZER_INFERENCE_BATCH", "8")
    encoder = local_encoder.LocalEncoder(artifacts[0])
    event = Event()
    with pytest.raises(CancelledError):
        encoder.encode(["doc"] * 10, cancel=event, progress=lambda *_: event.set())
    assert Session.batches == [8]


@pytest.mark.parametrize("value", [0, np.nan, np.inf])
def test_invalid_embedding_is_an_error_not_a_fallback(artifacts, runtime, monkeypatch, value):
    monkeypatch.setattr(Session, "run", lambda *args: [np.full((1, 3, 384), value, dtype=np.float32)])
    encoder = local_encoder.LocalEncoder(artifacts[0])
    with pytest.raises(AnalysisInputError, match="эмбеддинг"):
        encoder.encode(["doc"])


def test_manifest_is_stable_json_and_has_no_local_paths(artifacts, runtime):
    import json
    encoder = local_encoder.LocalEncoder(artifacts[0])
    before = encoder.manifest()
    assert encoder.manifest() == before
    assert before["revision"] == "ffb93f3bd4047442299a41ebb6fa998a38507c52"
    assert before["pooling"] == "attention_mask_mean_l2"
    assert str(artifacts[0]) not in json.dumps(before)
    before["files"][0]["sha256"] = "changed"
    assert encoder.manifest()["files"][0]["sha256"] != "changed"


def test_installer_refuses_nonempty_corrupt_target_before_network(artifacts, monkeypatch):
    path, _ = artifacts
    (path / "model.onnx").write_bytes(b"local data")
    import httpx
    monkeypatch.setattr(httpx, "Client", lambda **_: pytest.fail("Network before local safety check"))
    with pytest.raises(AnalysisInputError, match="не перезаписывается"):
        install_ml_model.install(path)
    assert (path / "model.onnx").read_bytes() == b"local data"


def fake_downloads(monkeypatch, payloads, *, header=None):
    import httpx
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        @contextmanager
        def stream(self, method, url):
            calls.append((method, url))
            assert "/resolve/ffb93f3bd4047442299a41ebb6fa998a38507c52/" in url
            name = url.rsplit("/", 1)[-1]
            yield SimpleNamespace(raise_for_status=lambda: None,
                                  headers={"content-length": header} if header else {},
                                  iter_bytes=lambda chunk_size: iter(payloads[name]))

    monkeypatch.setattr(httpx, "Client", Client)
    return calls


def test_installer_commits_only_verified_files_and_repeat_is_offline(artifacts, monkeypatch, tmp_path):
    _, spec = artifacts
    destination = tmp_path / "installed"
    destination.mkdir()
    payloads = {item["name"]: [item["name"].encode()] for item in spec["files"]}
    calls = fake_downloads(monkeypatch, payloads)
    result = install_ml_model.install(destination)
    assert result["state"] == "installed" and len(calls) == 2
    assert {path.name for path in destination.iterdir()} == {"model.onnx", "tokenizer.json"}
    assert install_ml_model.install(destination)["state"] == "already_present"
    assert len(calls) == 2
    assert not list(tmp_path.glob(".e5-install-*"))


@pytest.mark.parametrize("payload", [b"short", b"wrong-data", b"oversized-model-bytes"])
def test_failed_download_never_publishes_partial_model(artifacts, monkeypatch, tmp_path, payload):
    destination = tmp_path / "new-install"
    fake_downloads(monkeypatch, {"model.onnx": [payload]})
    with pytest.raises(AnalysisInputError):
        install_ml_model.install(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".e5-install-*"))


def test_invalid_content_length_is_a_readable_download_error(artifacts, monkeypatch, tmp_path):
    destination = tmp_path / "bad-header"
    fake_downloads(monkeypatch, {"model.onnx": []}, header="invalid")
    with pytest.raises(AnalysisInputError):
        install_ml_model.install(destination)
    assert not destination.exists()


@pytest.mark.parametrize("stage", ["constructor", "run"])
@pytest.mark.parametrize("frozen_bundle", [False, True])
def test_native_failure_is_readable_and_keeps_its_cause(artifacts, runtime, monkeypatch, stage, frozen_bundle):
    native_error = NativeFail("native allocation failure")
    monkeypatch.setattr(local_encoder, "frozen", lambda: frozen_bundle)

    def fail(*args, **kwargs):
        raise native_error

    if stage == "constructor":
        monkeypatch.setattr(runtime, "InferenceSession", fail)
    else:
        monkeypatch.setattr(Session, "run", fail)
    with pytest.raises(AnalysisInputError, match="ONNX") as caught:
        local_encoder.LocalEncoder(artifacts[0]).encode(["text"])
    assert caught.value.__cause__ is native_error
    assert "allocation failure" not in str(caught.value)
    assert ("переустановите приложение" in str(caught.value)) is frozen_bundle
    assert ("requirements/semantic.lock" in str(caught.value)) is not frozen_bundle


@pytest.mark.parametrize("stage", ["constructor", "run"])
@pytest.mark.parametrize("error", [RuntimeError("programmer bug"), TypeError("wrong API call"),
                                  CancelledError("cancelled")])
def test_programming_errors_and_cancellation_are_not_wrapped(artifacts, runtime, monkeypatch, stage, error):
    def fail(*args, **kwargs):
        raise error

    if stage == "constructor":
        monkeypatch.setattr(runtime, "InferenceSession", fail)
    else:
        monkeypatch.setattr(Session, "run", fail)
    with pytest.raises(type(error)) as caught:
        local_encoder.LocalEncoder(artifacts[0]).encode(["text"])
    assert caught.value is error


@pytest.mark.parametrize("stage", ["constructor", "run"])
def test_native_cancellation_uses_application_cancellation_type(artifacts, runtime, monkeypatch, stage):
    native_error = NativeCancelled("model loading cancelled")

    def fail(*args, **kwargs):
        raise native_error

    if stage == "constructor":
        monkeypatch.setattr(runtime, "InferenceSession", fail)
    else:
        monkeypatch.setattr(Session, "run", fail)
    with pytest.raises(CancelledError) as caught:
        local_encoder.LocalEncoder(artifacts[0]).encode(["text"])
    assert caught.value.__cause__ is native_error


@pytest.mark.parametrize("stage", ["constructor", "run"])
@pytest.mark.parametrize("name", ["Fail", "RuntimeException", "InvalidArgument", "EPFail"])
def test_actual_onnx_status_exceptions_follow_the_error_contract(artifacts, runtime, monkeypatch, stage, name):
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "1")
    native = pytest.importorskip("onnxruntime.capi.onnxruntime_pybind11_state")
    runtime.capi.onnxruntime_pybind11_state = native
    error = getattr(native, name)("native status failure")

    def fail(*args, **kwargs):
        raise error

    if stage == "constructor":
        monkeypatch.setattr(runtime, "InferenceSession", fail)
    else:
        monkeypatch.setattr(Session, "run", fail)
    with pytest.raises(AnalysisInputError, match=name) as caught:
        local_encoder.LocalEncoder(artifacts[0]).encode(["text"])
    assert caught.value.__cause__ is error


def test_native_runtime_import_disables_telemetry_before_initialization(tmp_path):
    # Optional native dependency: the other tests exercise the interface without it.
    import importlib.util
    if importlib.util.find_spec("onnxruntime") is None or importlib.util.find_spec("tokenizers") is None:
        pytest.skip("Optional semantic runtime is not installed")
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                   "ORT_DISABLE_TELEMETRY": "0"}
    probe = ("from app.ml.local_encoder import _load_runtime; "
             "import os; _load_runtime(); assert os.environ['ORT_DISABLE_TELEMETRY'] == '1'")
    completed = subprocess.run([sys.executable, "-B", "-c", probe], cwd=tmp_path, env=environment,
                               capture_output=True, text=True, timeout=30, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "telemetry" not in completed.stderr.casefold()
    assert list(tmp_path.iterdir()) == []
