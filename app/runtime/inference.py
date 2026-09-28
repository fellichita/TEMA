"""Настройки локального инференса: провайдер, потоки, батч и порядок текстов.

Раньше оба энкодера были жёстко зафиксированы на двух потоках ONNX Runtime.
На машине с 16 ядрами это оставляло процессор почти простаивающим: замер на
multilingual-e5-small дал 6.2 документа в секунду против 14.3 при потоках по
числу ядер, большем батче и группировке текстов по длине.

Ни одна настройка отсюда не меняет результат. Проверено прямым сравнением:
эмбеддинги при intra=2/batch=8 и при intra=8/batch=32 с сортировкой совпадают
бит в бит, поэтому отпечатки анализа и ключи кеша остаются прежними. Регрессия
на это есть в tests/test_inference_runtime.py; если будущая версия ONNX Runtime
перестанет быть детерминированной, тест это покажет.

Значения не входят ни в спецификацию модели, ни в её manifest: спецификация
участвует в ключе кеша эмбеддингов, а manifest — в отпечатке результата, и
число ядер конкретной машины не должно на них влиять.
"""

from __future__ import annotations

import os
from pathlib import Path

# Дальше восьми потоков прирост почти исчезает (замер: ×1.88 на восьми против
# ×2.04 на шестнадцати), а интерфейс начинает голодать без ядер.
MAX_INTRA_OP_THREADS = 8
# Батч крупнее спецификации ускоряет счёт, но держит в памяти больше токенов.
MAX_BATCH_SIZE = 32
THREADS_VARIABLE = "TRENDANALIZER_INFERENCE_THREADS"
BATCH_VARIABLE = "TRENDANALIZER_INFERENCE_BATCH"


def _positive_override(name: str, maximum: int) -> int | None:
    """Ручное значение из окружения; мусор молча игнорируется."""
    raw = os.environ.get(name, "").strip()
    if not raw.isdigit():
        return None
    value = int(raw)
    return min(value, maximum) if value >= 1 else None


def session_threads() -> tuple[int, int]:
    """Потоки для одной сессии ONNX Runtime: (внутриоперационные, межоперационные).

    Оставляем часть ядер приложению и другим задачам: берём половину, но не
    больше MAX_INTRA_OP_THREADS. На однопроцессорной машине не создаём второй
    вычислительный поток. Межоперационный параллелизм для одной последовательной
    модели выигрыша не даёт и остаётся равным одному.
    """
    override = _positive_override(THREADS_VARIABLE, MAX_INTRA_OP_THREADS)
    if override is not None:
        return override, 1
    # process_cpu_count accounts for the process affinity on Python 3.13+;
    # older supported runtimes retain the cpu_count fallback.
    available = getattr(os, "process_cpu_count", None)
    cores = (available() if available is not None else None) or os.cpu_count() or 1
    return (1 if cores == 1 else max(2, min(MAX_INTRA_OP_THREADS, cores // 2))), 1


def batch_size(spec_batch_size: int) -> int:
    """Размер батча времени выполнения; спецификация модели не меняется."""
    override = _positive_override(BATCH_VARIABLE, MAX_BATCH_SIZE)
    if override is not None:
        return max(override, 1)
    if type(spec_batch_size) is not int or spec_batch_size < 1:
        raise ValueError("Размер батча в спецификации модели должен быть натуральным числом.")
    return max(spec_batch_size, MAX_BATCH_SIZE)


def length_order(texts) -> list[int]:
    """Порядок обхода, при котором в батч попадают тексты близкой длины.

    Токенизатор дополняет батч до самой длинной последовательности в нём, и
    короткий абстракт рядом с длинным считается как длинный. Сортировка снижает
    долю паддинга; исходный порядок восстанавливает вызывающая сторона.
    """
    # Python's stable sort preserves the original index order for equal lengths.
    return sorted(range(len(texts)), key=lambda index: len(texts[index]))


PROVIDER_VARIABLE = "TRENDANALIZER_INFERENCE_PROVIDER"
_GPU_PROVIDERS = {"cuda": "CUDAExecutionProvider"}
_DLL_DIRECTORY_HANDLES: list[object] = []
_LOADED_DLL_DIRECTORIES: set[str] = set()


class RequestedCudaUnavailable(RuntimeError):
    """An explicitly requested CUDA run must not silently execute on the CPU."""


def _cuda_unavailable() -> RequestedCudaUnavailable:
    return RequestedCudaUnavailable(
        "CUDA недоступна. Проверьте драйвер NVIDIA, CUDA-библиотеки и установку onnxruntime-gpu.")


def require_requested_provider(actual: str) -> None:
    """Reject a CUDA session that ONNX Runtime quietly moved onto the CPU."""
    if os.environ.get(PROVIDER_VARIABLE, "").strip().lower() == "cuda" and actual != "CUDAExecutionProvider":
        raise RequestedCudaUnavailable(
            "ONNX Runtime запустил модель на CPU вместо запрошенной CUDA. "
            "Проверьте драйвер NVIDIA, CUDA-библиотеки и установку onnxruntime-gpu.")


def _gpu_library_directories() -> list[str]:
    """Where recent NVIDIA wheels place their runtime libraries on Windows."""
    import site

    directories = []
    roots = [*site.getsitepackages(), site.getusersitepackages()]
    for root in roots:
        for relative in ("nvidia/cu13/bin/x86_64", "nvidia/cu13/bin", "nvidia/cudnn/bin",
                         "nvidia/cudnn/bin/x86_64", "nvidia/cudnn/lib"):
            candidate = Path(root) / relative
            if candidate.is_dir():
                directories.append(str(candidate))
    return directories


def prepare_gpu_libraries() -> None:
    """Make CUDA libraries findable before onnxruntime asks for its provider.

    The CUDA 13 wheels install into a directory onnxruntime does not search, and
    the provider library resolves its own dependencies through the ordinary DLL
    search path, so the directory is added there and the libraries are loaded in
    advance. A machine without those wheels simply finds nothing and keeps CPU.
    """
    if os.name != "nt":
        return
    import ctypes

    add_dll_directory = getattr(os, "add_dll_directory", None)
    load_dll = getattr(ctypes, "WinDLL", None)
    for directory in _gpu_library_directories():
        if directory in _LOADED_DLL_DIRECTORIES:
            continue
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
        if add_dll_directory is not None:
            try:
                # Keep the handle: closing it removes the directory again.
                _DLL_DIRECTORY_HANDLES.append(add_dll_directory(directory))
            except OSError:
                continue
        if load_dll is not None:
            for library in sorted(Path(directory).glob("*.dll")):
                try:
                    load_dll(str(library))
                except OSError:
                    continue  # A library this build does not need must not stop the rest.
        _LOADED_DLL_DIRECTORIES.add(directory)


def execution_providers() -> tuple[str, ...]:
    """Providers for one session, in order of preference.

    A graphics card computes the same embedding with slightly different last
    decimals, so which provider ran is part of what produced a result: the
    encoder records the provider it actually got and gives it its own cache.
    Within that accounting the fastest available provider is used, because on a
    real corpus it is the difference between one minute and four.

    ``TRENDANALIZER_INFERENCE_PROVIDER`` decides: ``cpu`` keeps the processor even
    where a card exists, ``cuda`` demands that card, and anything else — including
    an unset variable — takes the card when this runtime really offers one and the
    processor otherwise. Automatic mode can fall back to CPU; an explicit CUDA
    request fails before analysis instead of silently running for minutes on CPU.
    """
    requested = os.environ.get(PROVIDER_VARIABLE, "").strip().lower()
    if requested == "cpu":
        return ("CPUExecutionProvider",)
    provider = _GPU_PROVIDERS.get(requested, _GPU_PROVIDERS["cuda"])
    prepare_gpu_libraries()
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError:
        if requested == "cuda":
            raise _cuda_unavailable() from None
        return ("CPUExecutionProvider",)
    if provider not in ort.get_available_providers():
        if requested == "cuda":
            raise _cuda_unavailable()
        return ("CPUExecutionProvider",)
    # The CUDA wheel can be present while its NVIDIA libraries live only in
    # Python site-packages. Load them before constructing any ONNX session, on
    # both Windows and Linux. The session itself still checks whether CUDA can
    # actually execute this model.
    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        try:
            preload(directory="")
        except (OSError, RuntimeError) as error:
            if requested == "cuda":
                raise _cuda_unavailable() from error
            return ("CPUExecutionProvider",)
    return (provider, "CPUExecutionProvider")
