"""Pinned local E5 inference. This module never downloads files or accesses the network."""

import hashlib
import json
import os
from concurrent.futures import CancelledError
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from app.ml.contracts import AnalysisInputError
from app.ml.corpus import checkpoint
from app.runtime.inference import batch_size as runtime_batch_size, length_order, session_threads
from app.runtime.model_resources import frozen, resolve_model

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "storage/models/e5-small-v2"
INSTALL_COMMAND = "python -m scripts.install_ml_model"


def _repair_hint():
    return "Переустановите приложение." if frozen() else f"Выполните {INSTALL_COMMAND}."


def load_spec():
    """Read the specification bundled with application code, never from a model directory."""
    spec = json.loads(Path(__file__).with_name("encoder_spec.json").read_text(encoding="utf-8"))
    if (spec["schema_version"] != 1 or spec["model_id"] != "intfloat/e5-small-v2"
            or {item["name"] for item in spec["files"]} != {"model.onnx", "tokenizer.json"}
            or len(spec["files"]) != 2):
        raise RuntimeError("Некорректная встроенная спецификация локальной модели.")
    return spec


def verify_artifacts(model_dir, spec=None):
    """Verify exact pinned bytes before loading native inference code."""
    spec = spec if spec is not None else load_spec()
    directory = Path(model_dir)
    for item in spec["files"]:
        path = directory / item["name"]
        try:
            if path.is_symlink() or not path.is_file():
                raise AnalysisInputError(f"Отсутствует локальный файл {item['name']}. {_repair_hint()}")
            if path.stat().st_size != item["bytes"]:
                raise AnalysisInputError(f"Неверный размер локального файла {item['name']}. "
                                         + _repair_hint())
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != item["sha256"]:
                raise AnalysisInputError(f"Нарушена целостность локального файла {item['name']}. "
                                         + _repair_hint())
        except OSError as error:
            raise AnalysisInputError(f"Не удалось прочитать локальный файл {item['name']}.") from error
    return directory


def _load_runtime():
    # ORT 1.29's POSIX telemetry initializes at import, before the Python disable
    # API is callable. The environment opt-out prevents uploader/cache creation.
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    try:
        import numpy as np
        import onnxruntime as ort  # type: ignore[import-untyped]  # Vendor ships no stubs.
        from tokenizers import Tokenizer
    except (ImportError, OSError) as error:
        message = ("Не загружается встроенный ML runtime. Переустановите приложение." if frozen() else
                   "Установите зависимости локальной модели: "
                   "python -m pip install -r requirements/semantic.lock")
        raise AnalysisInputError(message) from error
    # ONNX Runtime is used only as a CPU executor; model/hub clients are never imported.
    ort.disable_telemetry_events()
    return np, ort, Tokenizer


def _runtime_versions():
    result = {}
    for package in ("numpy", "onnxruntime", "tokenizers"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


class LocalEncoder:
    """CPU encoder with bounded batches; create one instance per analysis worker.

    encode accepts a list/tuple of nonempty strings. Its optional progress callback
    receives (completed_documents, total_documents) after each batch. Cancellation
    is checked before and after each synchronous ONNX call; a call is not preempted.
    Empty input returns a float32 array of shape (0, 384).
    """

    def __init__(self, model_dir=None):
        self._spec = load_spec()
        location = resolve_model("e5-small-v2", self._spec["revision"], explicit_dir=model_dir,
                                 development_default=DEFAULT_MODEL_DIR)
        directory = verify_artifacts(location.path, self._spec)
        self._np, ort, tokenizer_class = _load_runtime()
        native = ort.capi.onnxruntime_pybind11_state
        # Native status exceptions share no narrower Python base class than
        # Exception. Name only ORT's failures; never catch arbitrary Python bugs.
        self._native_errors = tuple(getattr(native, name) for name in (
            "Fail", "InvalidArgument", "NoSuchFile", "NoModel", "EngineError", "RuntimeException",
            "InvalidProtobuf", "ModelLoaded", "NotImplemented", "InvalidGraph", "EPFail",
            "ModelRequiresCompilation", "NotFound", "DeviceReset"))
        self._native_cancelled = native.ModelLoadCanceled
        options = ort.SessionOptions()
        self._threads = session_threads()
        options.intra_op_num_threads, options.inter_op_num_threads = self._threads
        options.log_severity_level = 3
        self._tokenizer = tokenizer_class.from_file(str(directory / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=self._spec["max_length"])
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        try:
            self._session = ort.InferenceSession(str(directory / "model.onnx"), sess_options=options,
                                                providers=["CPUExecutionProvider"], enable_fallback=False)
        except self._native_cancelled as error:
            raise CancelledError("Загрузка локальной модели отменена.") from error
        except self._native_errors as error:
            raise AnalysisInputError(f"ONNX Runtime не смог загрузить локальную модель ({type(error).__name__}). "
                                     + ("Проверьте доступную память; если ошибка повторяется, переустановите приложение."
                                        if frozen() else
                                        "Проверьте доступную память и зависимости requirements/semantic.lock.")) from error
        inputs = {item.name for item in self._session.get_inputs()}
        if inputs != {"input_ids", "attention_mask", "token_type_ids"} or "last_hidden_state" not in {
                item.name for item in self._session.get_outputs()}:
            raise AnalysisInputError("Локальная модель имеет несовместимый интерфейс ONNX.")
        # Настройки скорости в manifest не входят: он участвует в отпечатке
        # результата, а число потоков зависит от машины и на значения не влияет.
        self._manifest = {**deepcopy(self._spec), "runtime": _runtime_versions(),
                          "execution_provider": "CPUExecutionProvider", "output_dtype": "float32"}

    def manifest(self):
        """Portable metadata, without local paths, credentials or changing timestamps."""
        return deepcopy(self._manifest)

    def encode(self, texts, kind="passage", cancel=None, progress=None):
        checkpoint(cancel)
        if not isinstance(kind, str) or kind not in self._spec["prefixes"]:
            raise AnalysisInputError("Тип текста локальной модели: query или passage.")
        if (not isinstance(texts, (list, tuple))
                or any(not isinstance(text, str) or not text.strip() for text in texts)):
            raise AnalysisInputError("Локальная модель принимает список непустых текстов.")
        np = self._np
        result = np.empty((len(texts), self._spec["dimensions"]), dtype=np.float32)
        batch_size = runtime_batch_size(self._spec["batch_size"])
        # Батчи собираются из текстов близкой длины: токенизатор дополняет батч
        # до самой длинной последовательности в нём. Порядок результата прежний.
        order = length_order(texts)
        done = 0
        for start in range(0, len(order), batch_size):
            checkpoint(cancel)
            positions = order[start:start + batch_size]
            batch = [texts[position] for position in positions]
            tokens = self._tokenizer.encode_batch([self._spec["prefixes"][kind] + text for text in batch])
            feeds = {name: np.asarray([getattr(row, field) for row in tokens], dtype=np.int64)
                     for name, field in (("input_ids", "ids"), ("attention_mask", "attention_mask"),
                                         ("token_type_ids", "type_ids"))}
            checkpoint(cancel)
            try:
                hidden = self._session.run(["last_hidden_state"], feeds)[0]
            except self._native_cancelled as error:
                raise CancelledError("Вычисление локальной модели отменено.") from error
            except self._native_errors as error:
                raise AnalysisInputError(f"ONNX Runtime не выполнил локальный расчёт ({type(error).__name__}). "
                                         + ("Проверьте доступную память; если ошибка повторяется, переустановите приложение."
                                            if frozen() else
                                            "Проверьте доступную память и зависимости requirements/semantic.lock.")) from error
            checkpoint(cancel)
            if hidden.shape != (*feeds["input_ids"].shape, self._spec["dimensions"]):
                raise AnalysisInputError("Локальная модель вернула эмбеддинг неправильного размера.")
            mask = feeds["attention_mask"].astype(np.float32)[..., None]
            if not np.isfinite(hidden).all() or (mask.sum(axis=1) <= 0).any():
                raise AnalysisInputError("Локальная модель вернула некорректный эмбеддинг.")
            # The pinned model returns token vectors, not sentence embeddings.
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or (norms <= 0).any():
                raise AnalysisInputError("Локальная модель вернула нулевой или нечисловой эмбеддинг.")
            vectors = pooled / norms
            for row, position in enumerate(positions):
                result[position] = vectors[row]
            done += len(positions)
            if progress is not None:
                progress(done, len(texts))
        checkpoint(cancel)
        return result
