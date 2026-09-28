"""Настройки скорости локального инференса не должны менять результат.

Потоки, размер батча и порядок текстов подобраны замером и живут отдельно от
спецификации модели: спецификация участвует в ключе кеша эмбеддингов, а
manifest — в отпечатке анализа. Попади туда число ядер, один и тот же корпус
давал бы разные отпечатки на разных машинах.
"""

import os

import pytest

from app.runtime.inference import (
    BATCH_VARIABLE, MAX_BATCH_SIZE, MAX_INTRA_OP_THREADS, PROVIDER_VARIABLE, THREADS_VARIABLE,
    RequestedCudaUnavailable, batch_size, execution_providers, length_order, require_requested_provider,
    session_threads,
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (THREADS_VARIABLE, BATCH_VARIABLE, PROVIDER_VARIABLE):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("cores,expected", [(1, 1), (2, 2), (4, 2), (8, 4), (16, 8), (64, 8), (None, 1)])
def test_threads_follow_cores_and_stay_bounded(monkeypatch, cores, expected):
    monkeypatch.setattr(os, "cpu_count", lambda: cores)
    monkeypatch.setattr(os, "process_cpu_count", lambda: cores, raising=False)
    intra, inter = session_threads()
    assert (intra, inter) == (expected, 1)
    assert 1 <= intra <= MAX_INTRA_OP_THREADS


def test_threads_respect_a_single_available_core_under_affinity(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 1, raising=False)
    assert session_threads() == (1, 1)


def test_threads_fall_back_to_host_core_count_on_older_python(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    assert session_threads() == (4, 1)


@pytest.mark.parametrize("raw,expected", [("4", 4), ("1", 1), ("99", MAX_INTRA_OP_THREADS)])
def test_thread_override_is_honoured_and_capped(monkeypatch, raw, expected):
    monkeypatch.setenv(THREADS_VARIABLE, raw)
    assert session_threads() == (expected, 1)


@pytest.mark.parametrize("raw", ["0", "-2", "", "два", "4.5"])
def test_invalid_thread_override_falls_back_to_cores(monkeypatch, raw):
    monkeypatch.setenv(THREADS_VARIABLE, raw)
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 16, raising=False)
    assert session_threads() == (8, 1)


def test_batch_size_never_shrinks_below_the_specification(monkeypatch):
    assert batch_size(8) == MAX_BATCH_SIZE
    assert batch_size(64) == 64
    monkeypatch.setenv(BATCH_VARIABLE, "4")
    assert batch_size(8) == 4


def test_batch_override_is_capped_and_specification_is_validated(monkeypatch):
    monkeypatch.setenv(BATCH_VARIABLE, "4096")
    assert batch_size(8) == MAX_BATCH_SIZE
    monkeypatch.delenv(BATCH_VARIABLE)
    for broken in (0, -1, 1.5, "8", None):
        with pytest.raises(ValueError):
            batch_size(broken)


def test_length_order_is_a_stable_permutation():
    texts = ["ccc", "a", "bbbb", "dd", "e"]
    order = length_order(texts)
    assert sorted(order) == list(range(len(texts)))
    assert [texts[index] for index in order] == ["a", "e", "dd", "ccc", "bbbb"]
    # Тексты равной длины сохраняют исходный порядок: батчи воспроизводимы.
    assert length_order(["aa", "bb", "cc"]) == [0, 1, 2]


def test_manifests_carry_no_machine_specific_settings():
    """Отпечаток результата не должен зависеть от числа ядер машины."""
    from pathlib import Path
    import json

    forbidden = {"intra_op_threads", "inter_op_threads", "batch_size"}
    for module in ("app/pilot/encoder.py", "app/ml/local_encoder.py"):
        source = Path(module).read_text(encoding="utf-8")
        manifest = source.split("def manifest(")[1].split("def ")[0]
        assert not (forbidden & set(json.dumps(manifest).split('"')))


def _offers(monkeypatch, *providers):
    import onnxruntime

    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: list(providers))


@pytest.mark.parametrize("requested", [None, "", "  ", "gpu", "CUDA0", "tensorrt", "директx"])
def test_an_available_card_is_taken_by_default_and_by_an_unclear_request(monkeypatch, requested):
    """The fastest available provider runs; the encoder records which one it was."""
    if requested is not None:
        monkeypatch.setenv(PROVIDER_VARIABLE, requested)
    _offers(monkeypatch, "CUDAExecutionProvider", "CPUExecutionProvider")
    assert execution_providers() == ("CUDAExecutionProvider", "CPUExecutionProvider")


@pytest.mark.parametrize("requested", [None, "", "мусор"])
def test_a_machine_without_a_card_keeps_working_on_the_processor(monkeypatch, requested):
    """A missing card, package or library is never an error, only slower."""
    if requested is not None:
        monkeypatch.setenv(PROVIDER_VARIABLE, requested)
    _offers(monkeypatch, "CPUExecutionProvider")
    assert execution_providers() == ("CPUExecutionProvider",)


def test_explicit_cuda_refuses_a_silent_cpu_fallback(monkeypatch):
    monkeypatch.setenv(PROVIDER_VARIABLE, "cuda")
    _offers(monkeypatch, "CPUExecutionProvider")
    with pytest.raises(RequestedCudaUnavailable, match="CUDA недоступна"):
        execution_providers()


def test_cuda_preloads_libraries_from_python_site_packages(monkeypatch):
    import onnxruntime

    monkeypatch.setenv(PROVIDER_VARIABLE, "cuda")
    _offers(monkeypatch, "CUDAExecutionProvider", "CPUExecutionProvider")
    directories = []
    monkeypatch.setattr(onnxruntime, "preload_dlls", lambda **kwargs: directories.append(kwargs["directory"]))

    assert execution_providers()[0] == "CUDAExecutionProvider"
    assert directories == [""]


def test_cuda_library_load_failure_is_visible_when_explicit_and_safe_when_automatic(monkeypatch):
    import onnxruntime

    _offers(monkeypatch, "CUDAExecutionProvider", "CPUExecutionProvider")

    def broken(**_kwargs):
        raise OSError("CUDA runtime unavailable")

    monkeypatch.setattr(onnxruntime, "preload_dlls", broken)
    assert execution_providers() == ("CPUExecutionProvider",)
    monkeypatch.setenv(PROVIDER_VARIABLE, "cuda")
    with pytest.raises(RequestedCudaUnavailable, match="CUDA недоступна"):
        execution_providers()


def test_explicit_cuda_refuses_a_missing_runtime(monkeypatch):
    import builtins

    monkeypatch.setenv(PROVIDER_VARIABLE, "cuda")
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("no runtime")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(RequestedCudaUnavailable, match="CUDA недоступна"):
        execution_providers()


def test_explicit_cuda_refuses_a_session_that_fell_back_to_cpu(monkeypatch):
    monkeypatch.setenv(PROVIDER_VARIABLE, "cuda")
    with pytest.raises(RequestedCudaUnavailable, match="CPU вместо запрошенной CUDA"):
        require_requested_provider("CPUExecutionProvider")
    require_requested_provider("CUDAExecutionProvider")
    monkeypatch.setenv(PROVIDER_VARIABLE, "")
    require_requested_provider("CPUExecutionProvider")


def test_the_processor_can_be_demanded_even_where_a_card_exists(monkeypatch):
    """Bit-for-bit continuity with earlier results stays reachable on any machine."""
    monkeypatch.setenv(PROVIDER_VARIABLE, "cpu")
    _offers(monkeypatch, "CUDAExecutionProvider", "CPUExecutionProvider")
    assert execution_providers() == ("CPUExecutionProvider",)


def test_a_runtime_without_onnxruntime_still_answers(monkeypatch):
    import builtins

    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("no runtime")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    assert execution_providers() == ("CPUExecutionProvider",)


def test_the_provider_belongs_to_the_manifest_because_it_changes_values():
    """Threads do not reach the fingerprint; the provider must, and does."""
    from pathlib import Path

    source = Path("app/pilot/encoder.py").read_text(encoding="utf-8")
    manifest = source.split("def manifest(")[1].split("def ")[0]
    assert "execution_provider" in manifest and "self.provider" in manifest


def test_the_processor_keeps_its_historical_fingerprint(monkeypatch):
    """Caches written by earlier versions stay valid while the card gets its own."""
    from app.pilot.encoder import encoder_fingerprint, load_spec, spec_fingerprint

    spec = load_spec()
    assert encoder_fingerprint(spec, "CPUExecutionProvider") == spec_fingerprint(spec)
    assert len(spec_fingerprint(spec)) == 64


def test_cuda_bucketing_gets_a_new_cache_fingerprint():
    """Old CUDA vectors cannot be mixed with results from reordered batches."""
    import hashlib
    import json

    from app.pilot.encoder import encoder_fingerprint, load_spec

    spec = load_spec()
    old_identity = {"spec": spec, "provider": "CUDAExecutionProvider"}
    old = hashlib.sha256(json.dumps(old_identity, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    current = encoder_fingerprint(spec, "CUDAExecutionProvider")
    assert current != old
    assert current == encoder_fingerprint(spec, "CUDAExecutionProvider")
    assert current != encoder_fingerprint(spec, "CPUExecutionProvider")


def test_cuda_cache_fingerprint_includes_effective_batch_size(monkeypatch):
    from app.pilot.encoder import encoder_fingerprint, load_spec

    spec = load_spec()
    gpu = []
    cpu = []
    for size in (8, 16, 32):
        monkeypatch.setenv(BATCH_VARIABLE, str(size))
        gpu.append(encoder_fingerprint(spec, "CUDAExecutionProvider"))
        cpu.append(encoder_fingerprint(spec, "CPUExecutionProvider"))
        assert gpu[-1] == encoder_fingerprint(spec, "CUDAExecutionProvider", batch_size=size)
    assert len(set(gpu)) == 3
    assert len(set(cpu)) == 1
