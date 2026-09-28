"""Check that both pinned analysis models actually run with CUDA.

Run after installing the GPU wheel and both models. This check never downloads
a model or modifies the user profile. It generates two Qwen tokens and computes
one E5 embedding from fixed text without printing either result.
"""

from __future__ import annotations

from importlib import metadata
import os
import shutil
import subprocess
import sys
from time import perf_counter


EXPECTED_ORT_VERSION = "1.29.0"


def _installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _nvidia_smi() -> None:
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi не найден: проверьте установку драйвера NVIDIA.", file=sys.stderr)
        return
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"nvidia-smi не ответил: {error}", file=sys.stderr)
        return
    print("NVIDIA GPU / драйвер / память:")
    print(result.stdout.strip())


def main() -> int:
    if sys.platform not in {"win32", "linux"}:
        print("CUDA-сборка ONNX Runtime поддерживается здесь только на Windows или Linux.", file=sys.stderr)
        return 1

    gpu_version = _installed_version("onnxruntime-gpu")
    cpu_version = _installed_version("onnxruntime")
    if gpu_version != EXPECTED_ORT_VERSION or cpu_version is not None:
        print(
            f"Нужен только onnxruntime-gpu=={EXPECTED_ORT_VERSION}; сейчас GPU={gpu_version}, "
            f"CPU={cpu_version}. См. docs/guides/local-ai.md.", file=sys.stderr,
        )
        return 1

    _nvidia_smi()
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError as error:
        print(f"ONNX Runtime не импортируется: {error}", file=sys.stderr)
        return 1
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("ONNX Runtime не предлагает CUDAExecutionProvider.", file=sys.stderr)
        return 1

    # Prefer CUDA even when the application would otherwise choose a provider
    # automatically. Checking the session protects against silent CPU fallback.
    from app.pilot.local_llm import LocalInstructModel, LocalModelError
    from app.pilot.encoder import EncoderError, MultilingualEncoder

    provider_variable = "TRENDANALIZER_INFERENCE_PROVIDER"
    previous_provider = os.environ.get(provider_variable)
    os.environ[provider_variable] = "cuda"

    try:
        model = None
        try:
            qwen_started = perf_counter()
            model = LocalInstructModel()
            if model.provider != "CUDAExecutionProvider":
                raise LocalModelError(f"Qwen запущена через {model.provider}, а не CUDAExecutionProvider.")
            completion = model.generate(
                system="Return a JSON object.", user="Return {\"ok\":true}.", max_new_tokens=2,
            )
            qwen_seconds = perf_counter() - qwen_started
        except (LocalModelError, OSError, RuntimeError) as error:
            print(f"Проверка Qwen на CUDA не прошла: {error}", file=sys.stderr)
            return 1
        finally:
            if model is not None:
                model.close()

        try:
            e5_started = perf_counter()
            encoder = MultilingualEncoder()
            if encoder.provider != "CUDAExecutionProvider":
                raise EncoderError(f"E5 запущена через {encoder.provider}, а не CUDAExecutionProvider.")
            encoder.encode(["Scientific research on optical computing."], kind="query")
            e5_seconds = perf_counter() - e5_started
        except (EncoderError, OSError, RuntimeError) as error:
            print(f"Проверка E5 на CUDA не прошла: {error}", file=sys.stderr)
            return 1
    finally:
        if previous_provider is None:
            os.environ.pop(provider_variable, None)
        else:
            os.environ[provider_variable] = previous_provider

    print(
        f"OK: Qwen и E5 запущены через CUDAExecutionProvider; Qwen {qwen_seconds:.2f} с "
        f"({completion.completion_tokens} токен(ов)), E5 {e5_seconds:.2f} с (1 эмбеддинг)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
