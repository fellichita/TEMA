"""Offline multilingual embeddings with pinned artifacts and immutable NPY caches.

The encoder performs no downloads, imports no hub client and accepts no remote
model code. Scientific relevance thresholds are separate from model inference.
"""

from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Protocol
import unicodedata

from app.identity import default_data_dir, validate_data_dir
from app.runtime.inference import (
    RequestedCudaUnavailable, batch_size as runtime_batch_size, execution_providers, length_order,
    require_requested_provider, session_threads,
)
from app.runtime.model_resources import frozen, resolve_model

MAX_TEXT_CHARACTERS = 220_000
MAX_UNITS = 100_000
NORMALIZER_VERSION = "nfkc-whitespace-v1"
CUDA_BATCHING_VERSION = "token-length-window-v1"


class Cancellation(Protocol):
    def is_set(self) -> bool: ...


class EncoderError(ValueError):
    """A safe actionable model/cache error without credentials or document text."""


def checkpoint(cancel: Cancellation | None) -> None:
    if cancel is not None and cancel.is_set():
        raise CancelledError("Локальный расчёт отменён.")


def model_directory(data_dir: Path | None = None) -> Path:
    return (default_data_dir() if data_dir is None else Path(data_dir)) / "models" / "multilingual-e5-small"


def load_spec() -> dict[str, Any]:
    spec = json.loads(Path(__file__).with_name("encoder-spec.json").read_text(encoding="utf-8"))
    if (spec.get("schema_version") != 1 or spec.get("model_id") != "intfloat/multilingual-e5-small"
            or not re.fullmatch(r"[a-f0-9]{40}", spec.get("revision", ""))
            or spec.get("dimensions") != 384 or spec.get("max_length") != 512
            or spec.get("prefixes") != {"query": "query: ", "passage": "passage: "}
            or spec.get("normalizer_version") != NORMALIZER_VERSION):
        raise EncoderError("Повреждена встроенная спецификация многоязычной модели.")
    if (len(spec["files"]) != 3
            or {item["name"] for item in spec["files"]} != {"model.onnx", "tokenizer.json", "MODEL_CARD.md"}):
        raise EncoderError("Неверный состав многоязычной модели.")
    for item in spec["files"]:
        if (not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]) or type(item["bytes"]) is not int
                or not 0 < item["bytes"] <= 500_000_000
                or item["remote_path"] not in {"onnx/model.onnx", "onnx/tokenizer.json", "README.md"}):
            raise EncoderError("Неверная контрольная сумма или путь многоязычной модели.")
    return spec


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def spec_fingerprint(spec: dict[str, Any] | None = None) -> str:
    return hashlib.sha256(_canonical(load_spec() if spec is None else spec)).hexdigest()


def encoder_fingerprint(spec: dict[str, Any], provider: str, *, batch_size: int | None = None) -> str:
    """Keep historical CPU vectors; isolate CUDA kernels by effective batch."""
    if provider == "CPUExecutionProvider":
        return spec_fingerprint(spec)
    identity: dict[str, Any] = {"spec": spec, "provider": provider}
    if provider == "CUDAExecutionProvider":
        effective_batch_size = (runtime_batch_size(spec["batch_size"]) if batch_size is None
                                else batch_size)
        if type(effective_batch_size) is not int or effective_batch_size < 1:
            raise ValueError("CUDA batch size must be a positive integer.")
        identity["batching"] = CUDA_BATCHING_VERSION
        identity["batch_size"] = effective_batch_size
    return hashlib.sha256(_canonical(identity)).hexdigest()


# Category Cs is exactly the surrogate block; one compiled scan replaces a
# per-character category lookup over every abstract of the corpus.
_SURROGATE = re.compile("[\ud800-\udfff]")


def normalize_text(text: str) -> str:
    if not isinstance(text, str) or len(text) > MAX_TEXT_CHARACTERS:
        raise EncoderError("Текст отсутствует или превышает лимит 220 000 символов.")
    if _SURROGATE.search(text):
        raise EncoderError("Текст содержит некорректные Unicode-символы.")
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    if not normalized or len(normalized) > MAX_TEXT_CHARACTERS:
        raise EncoderError("Для эмбеддинга нужен непустой текст.")
    return normalized


def verify_artifacts(directory: Path, spec: dict[str, Any] | None = None,
                     cancel: Cancellation | None = None) -> Path:
    spec = load_spec() if spec is None else spec
    directory = Path(directory)
    for item in spec["files"]:
        checkpoint(cancel)
        path = directory / item["name"]
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"]:
                raise EncoderError("Отсутствует или повреждён файл многоязычной модели. "
                                   + ("Переустановите приложение." if frozen() else
                                      "Установите модель через настройки приложения."))
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    checkpoint(cancel)
                    digest.update(chunk)
            if digest.hexdigest() != item["sha256"]:
                raise EncoderError("Контрольная сумма многоязычной модели не совпадает.")
        except OSError as error:
            raise EncoderError("Не удалось прочитать многоязычную модель.") from error
    return directory


@dataclass(frozen=True)
class TextUnit:
    text: str
    start: int
    end: int
    token_count: int


@dataclass(frozen=True)
class ChunkedText:
    units: tuple[TextUnit, ...]
    truncated: bool
    total_tokens: int


class MultilingualEncoder:
    """FP32 ONNX encoder with a recorded execution provider and bounded batches."""

    def __init__(self, directory: Path | None = None, *, cancel: Cancellation | None = None):
        self.spec = load_spec()
        # Freeze the setting for this encoder: the GPU cache key and actual
        # inference batches must use the same value even if env changes later.
        self._batch_size = runtime_batch_size(self.spec["batch_size"])
        location = resolve_model("multilingual-e5-small", self.spec["revision"], explicit_dir=directory,
                                 development_default=model_directory())
        directory = verify_artifacts(location.path, self.spec, cancel)
        os.environ["ORT_DISABLE_TELEMETRY"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import numpy as np
        import onnxruntime as ort  # type: ignore[import-untyped]
        from tokenizers import Tokenizer

        ort.disable_telemetry_events()
        self._np = np
        self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self._raw_tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self._raw_tokenizer.no_truncation()
        self._raw_tokenizer.no_padding()
        if self._tokenizer.token_to_id(self.spec["pad_token"]) != self.spec["pad_id"]:
            raise EncoderError("Токенизатор не соответствует многоязычной модели.")
        self._tokenizer.enable_truncation(max_length=512)
        self._tokenizer.enable_padding(pad_id=self.spec["pad_id"], pad_token=self.spec["pad_token"])
        options = ort.SessionOptions()
        self._threads = session_threads()
        options.intra_op_num_threads, options.inter_op_num_threads = self._threads
        options.log_severity_level = 3
        checkpoint(cancel)
        providers = list(execution_providers())
        try:
            self._session = ort.InferenceSession(str(directory / "model.onnx"), sess_options=options,
                                                providers=providers, enable_fallback=False)
        except Exception as error:
            raise EncoderError("ONNX Runtime не смог загрузить проверенную многоязычную модель.") from error
        # The provider the session really got, not the one that was asked for: a
        # GPU changes the last decimals, so the values depend on this answer.
        self.provider = self._session.get_providers()[0]
        try:
            require_requested_provider(self.provider)
        except RequestedCudaUnavailable as error:
            raise EncoderError(str(error)) from None
        # CUDA benefits from grouping by actual token count instead of character
        # count. Keep encodings unpadded until each bounded group is formed.
        if self.provider == "CUDAExecutionProvider":
            self._tokenizer.no_padding()
        checkpoint(cancel)
        inputs = {item.name for item in self._session.get_inputs()}
        if inputs not in ({"input_ids", "attention_mask"}, {"input_ids", "attention_mask", "token_type_ids"}):
            raise EncoderError("Неизвестный интерфейс ONNX-модели.")
        self._inputs = inputs
        if "last_hidden_state" not in {item.name for item in self._session.get_outputs()}:
            raise EncoderError("ONNX-модель не возвращает ожидаемые token embeddings.")
        # The processor keeps the historical fingerprint, so every embedding cached
        # by an earlier version stays valid. Another provider computes slightly
        # different values, so it gets its own fingerprint and its own cache: the
        # two are never mixed, and a saved result says which one produced it.
        self.fingerprint = encoder_fingerprint(self.spec, self.provider, batch_size=self._batch_size)

    def manifest(self) -> dict[str, Any]:
        """Параметры модели и вычисления, от которых зависит результат.

        Число потоков не входит в manifest. На CPU размер батча также не
        влияет на fingerprint. На CUDA размер батча может менять последние
        знаки FP32, поэтому он включён в fingerprint внутри manifest.
        Провайдер записан и явно, и в fingerprint.
        """
        return {**json.loads(_canonical(self.spec)), "fingerprint": self.fingerprint,
                "execution_provider": self.provider,
                "tokenizer_parallelism": False, "dtype": "float32"}

    def performance(self) -> dict[str, Any]:
        """Диагностика скорости для журнала; в отпечаток не входит."""
        intra, inter = self._threads
        return {"intra_op_threads": intra, "inter_op_threads": inter,
                "batch_size": self._batch_size,
                "cuda_token_bucketing": self.provider == "CUDAExecutionProvider",
                "spec_batch_size": self.spec["batch_size"]}

    def chunk_text(self, text: str, *, kind: str = "passage", max_chunks: int = 32,
                   cancel: Cancellation | None = None) -> ChunkedText:
        if kind not in self.spec["prefixes"] or type(max_chunks) is not int or not 1 <= max_chunks <= 256:
            raise EncoderError("Некорректные параметры разбиения текста.")
        text = normalize_text(text)
        checkpoint(cancel)
        encoded = self._raw_tokenizer.encode(text, add_special_tokens=False)
        prefix = self.spec["prefixes"][kind]
        # Reserve prefix and special tokens; verify each boundary after re-tokenizing.
        capacity = 512 - len(self._raw_tokenizer.encode(prefix).ids) - 4
        units: list[TextUnit] = []
        start = 0
        while start < len(encoded.ids) and len(units) < max_chunks:
            checkpoint(cancel)
            end = min(len(encoded.ids), start + capacity)
            char_start = encoded.offsets[start][0]
            char_end = encoded.offsets[end - 1][1]
            piece = text[char_start:char_end]
            length = len(self._raw_tokenizer.encode(prefix + piece).ids)
            while length > 512 and end > start + 1:
                end -= 1
                char_end = encoded.offsets[end - 1][1]
                piece = text[char_start:char_end]
                length = len(self._raw_tokenizer.encode(prefix + piece).ids)
            if length > 512 or not piece:
                raise EncoderError("Не удалось безопасно разбить текст на токены модели.")
            units.append(TextUnit(piece, char_start, char_end, length))
            if end == len(encoded.ids):
                return ChunkedText(tuple(units), False, len(encoded.ids))
            start = max(start + 1, end - 32)
        return ChunkedText(tuple(units), start < len(encoded.ids), len(encoded.ids))

    def encode(self, texts: list[str] | tuple[str, ...], *, kind: str = "passage",
               cancel: Cancellation | None = None, progress=None):
        if not isinstance(texts, (list, tuple)) or len(texts) > MAX_UNITS or kind not in self.spec["prefixes"]:
            raise EncoderError("Некорректный запрос эмбеддингов или превышен лимит 100 000 текстовых фрагментов.")
        np = self._np
        result = np.empty((len(texts), 384), dtype=np.float32)
        prefix = self.spec["prefixes"][kind]
        # Start with a cheap character-length order. On CUDA, re-sort a bounded
        # window by exact token count after one tokenizer pass: multilingual
        # texts of equal character length can have very different token counts.
        # Original result positions are restored below.
        order = length_order(texts)
        size = self._batch_size
        cuda = self.provider == "CUDAExecutionProvider"
        window = size * 4 if cuda else size
        done = 0
        for start in range(0, len(order), window):
            checkpoint(cancel)
            positions = order[start:start + window]
            normalized = [prefix + normalize_text(texts[position]) for position in positions]
            tokenized = self._tokenizer.encode_batch(normalized)
            if cuda:
                ranked = sorted(zip(positions, tokenized, strict=True),
                                key=lambda item: (len(item[1].ids), item[0]))
            else:
                ranked = list(zip(positions, tokenized, strict=True))
            for offset in range(0, len(ranked), size):
                checkpoint(cancel)
                group = ranked[offset:offset + size]
                if cuda:
                    padded_length = max(len(row.ids) for _, row in group)
                    for _, row in group:
                        row.pad(padded_length, pad_id=self.spec["pad_id"],
                                pad_token=self.spec["pad_token"])
                feeds = {name: np.asarray([getattr(row, field) for _, row in group], dtype=np.int64)
                         for name, field in (("input_ids", "ids"), ("attention_mask", "attention_mask"),
                                             ("token_type_ids", "type_ids")) if name in self._inputs}
                try:
                    hidden = self._session.run(["last_hidden_state"], feeds)[0]
                except Exception as error:
                    raise EncoderError("Ошибка локального расчёта многоязычной модели.") from error
                checkpoint(cancel)
                if hidden.shape != (*feeds["input_ids"].shape, 384) or not np.isfinite(hidden).all():
                    raise EncoderError("Модель вернула некорректный token embedding.")
                mask = feeds["attention_mask"].astype(np.float32)[..., None]
                counts = mask.sum(axis=1)
                if (counts <= 0).any():
                    raise EncoderError("Токенизатор вернул пустую маску.")
                # ORT normally returns an owned writable FP32 array. Reuse it to
                # avoid another [batch, sequence, 384] allocation for masking;
                # retain compatibility with foreign read-only/aliased outputs.
                if hidden.dtype == np.float32 and hidden.flags.owndata and hidden.flags.writeable:
                    np.multiply(hidden, mask, out=hidden)
                else:
                    hidden = hidden * mask
                pooled = hidden.sum(axis=1) / counts
                norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                if not np.isfinite(pooled).all() or not np.isfinite(norms).all() or (norms <= 0).any():
                    raise EncoderError("Модель вернула нулевой или нечисловой эмбеддинг.")
                vectors = pooled / norms
                for row_index, (position, _) in enumerate(group):
                    result[position] = vectors[row_index]
                done += len(group)
                if progress:
                    progress(done, len(texts))
        checkpoint(cancel)
        return result


def validate_vectors(vectors, rows: int, *, cancel: Cancellation | None = None) -> None:
    import numpy as np

    if vectors.shape != (rows, 384) or vectors.dtype != np.dtype("float32") or not 0 <= rows <= MAX_UNITS:
        raise EncoderError("Индекс содержит несовместимые векторы.")
    for start in range(0, rows, 512):
        checkpoint(cancel)
        batch = vectors[start:start + 512]
        if not np.isfinite(batch).all() or not np.allclose(np.linalg.norm(batch, axis=1), 1, atol=1e-4):
            raise EncoderError("Индекс содержит нечисловые или ненормированные векторы.")


class EmbeddingCache:
    """Bounded derived vectors and immutable arrays, never pickle.

    last_hit means a verified aggregate hit. Rebuilding a damaged or differently
    ordered aggregate from verified per-text vectors still sets it to False.
    """

    def __init__(self, directory: Path, fingerprint: str, *, max_bytes: int | None = None):
        from app.pilot.embedding_store import DEFAULT_CACHE_BYTES

        if not re.fullmatch(r"[a-f0-9]{64}", fingerprint):
            raise EncoderError("Некорректный fingerprint модели.")
        self.directory = validate_data_dir(Path(directory))
        self.fingerprint = fingerprint
        self.max_bytes = DEFAULT_CACHE_BYTES if max_bytes is None else max_bytes
        if type(self.max_bytes) is not int or not 128 * 1024 <= self.max_bytes <= 4 * 1024**3:
            raise EncoderError("Лимит кеша должен быть от 128 КиБ до 4 ГиБ.")
        self.last_hit = False

    def key(self, texts: list[str] | tuple[str, ...], kind: str, *, cancel: Cancellation | None = None) -> str:
        if not isinstance(texts, (list, tuple)) or kind not in {"query", "passage"} or len(texts) > MAX_UNITS:
            raise EncoderError("Некорректный запрос кеша.")
        digest = hashlib.sha256(_canonical({"fingerprint": self.fingerprint, "kind": kind,
                                            "normalizer": NORMALIZER_VERSION, "rows": len(texts)}))
        for text in texts:
            checkpoint(cancel)
            # JSON framing distinguishes ['ab', 'c'] from ['a', 'bc'].
            digest.update(_canonical(normalize_text(text)))
            digest.update(b"\n")
        return digest.hexdigest()

    def _unit_key(self, normalized: str, kind: str) -> str:
        return hashlib.sha256(_canonical({"fingerprint": self.fingerprint, "kind": kind,
                                          "normalizer": NORMALIZER_VERSION, "text": normalized})).hexdigest()

    def _read(self, key: str, rows: int, cancel: Cancellation | None):
        import numpy as np
        from app.pilot.reports import open_local_regular

        manifest_path, array_path = self.directory / f"{key}.json", self.directory / f"{key}.npy"
        try:
            identities = [path.stat(follow_symlinks=False) for path in (manifest_path, array_path)]
            if (any(not stat.S_ISREG(info.st_mode) for info in identities)
                    or identities[0].st_size > 2048 or identities[1].st_size > rows * 384 * 4 + 1024):
                return None
            with open_local_regular(manifest_path) as stream:
                manifest = json.loads(stream.read(2049))
            if manifest != {"schema_version": 1, "key": key, "fingerprint": self.fingerprint,
                            "rows": rows, "dimensions": 384, "dtype": "float32",
                            "sha256": manifest.get("sha256")}:
                return None
            with open_local_regular(array_path) as stream:
                opened = os.fstat(stream.fileno())
                if (opened.st_dev, opened.st_ino) != (identities[1].st_dev, identities[1].st_ino):
                    return None
                digest = hashlib.sha256()
                total = 0
                while chunk := stream.read(1024 * 1024):
                    checkpoint(cancel)
                    total += len(chunk)
                    if total > rows * 384 * 4 + 1024:
                        return None
                    digest.update(chunk)
                if digest.hexdigest() != manifest["sha256"]:
                    return None
                # Parse only a bounded numeric header, then map this same open
                # regular file. A filename swap cannot redirect mmap to a FIFO
                # or an unverified file; object/pickle payloads are never loaded.
                stream.seek(0)
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=1024)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=1024)
                else:
                    return None
                if shape != (rows, 384) or fortran or dtype != np.dtype("float32"):
                    return None
                array = (np.memmap(stream, dtype=dtype, mode="r", offset=stream.tell(), shape=shape)
                         if rows else np.empty(shape, dtype=dtype))
                array.flags.writeable = False
            validate_vectors(array, rows, cancel=cancel)
            for path, before in zip((manifest_path, array_path), identities, strict=True):
                after = path.stat(follow_symlinks=False)
                if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    return None
            return array
        except (OSError, ValueError, TypeError, AttributeError, KeyError, OverflowError):
            return None

    def encode(self, encoder, texts: list[str] | tuple[str, ...], *, kind: str = "passage",
               cancel: Cancellation | None = None, progress=None):
        if encoder.fingerprint != self.fingerprint:
            raise EncoderError("Кеш принадлежит другой версии модели.")
        checkpoint(cancel)
        key = self.key(texts, kind, cancel=cancel)
        from app.pilot.embedding_store import DerivedCache
        import numpy as np

        with DerivedCache(self.directory, self.max_bytes, cancel) as store:
            cached = self._read(key, len(texts), cancel)
            self.last_hit = cached is not None
            normalized = []
            for text in texts:
                checkpoint(cancel)
                normalized.append(normalize_text(text))
            keys = [self._unit_key(text, kind) for text in normalized]
            unique = dict(zip(keys, normalized, strict=True))
            if cached is not None:
                # A legacy aggregate may seed reusable units only after full
                # hash/dtype/shape/norm verification. Existing stores are lazy.
                if not store.store_exists:
                    first = dict(zip(keys, range(len(keys)), strict=True))
                    store.write([(unit, cached[index].astype("<f4", copy=False).tobytes())
                                 for unit, index in list(first.items())[-store.max_rows:]])
                checkpoint(cancel)
                return cached
            if len(texts) * 384 * 4 + 4096 + store.db_limit >= self.max_bytes:
                raise EncoderError("Запрошенный массив превышает лимит производного кеша.")
            found = store.read(list(unique))
            missing = [unit for unit in unique if unit not in found]
            if missing:
                computed = encoder.encode([unique[unit] for unit in missing], kind=kind, cancel=cancel,
                                          progress=progress)
                validate_vectors(computed, len(missing), cancel=cancel)
                found.update((unit, computed[index]) for index, unit in enumerate(missing))
                start = max(0, len(missing) - store.max_rows)
                store.write([(missing[index], computed[index].astype("<f4", copy=False).tobytes())
                             for index in range(start, len(missing))])
            vectors = np.empty((len(texts), 384), dtype=np.float32)
            for index, unit in enumerate(keys):
                if index % 512 == 0:
                    checkpoint(cancel)
                vectors[index] = found[unit]
            validate_vectors(vectors, len(texts), cancel=cancel)
            store.reserve(vectors.nbytes + 4096)
            try:
                return self._publish(key, vectors, cancel)
            except OSError:
                raise EncoderError("Не удалось сохранить производный кеш эмбеддингов. Проверьте доступ и свободное место.") from None

    def _publish(self, key, vectors, cancel):
        array_path, manifest_path = self.directory / f"{key}.npy", self.directory / f"{key}.json"
        import numpy as np

        with tempfile.TemporaryDirectory(prefix=".embedding-", dir=self.directory) as temporary:
            array_temp = Path(temporary) / "vectors.npy"
            with array_temp.open("xb") as stream:
                np.save(stream, vectors, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            if os.name != "nt":
                os.chmod(array_temp, 0o600)
            with array_temp.open("rb") as stream:
                checksum = hashlib.sha256()
                while chunk := stream.read(1024 * 1024):
                    checkpoint(cancel)
                    checksum.update(chunk)
                digest = checksum.hexdigest()
            manifest = {"schema_version": 1, "key": key, "fingerprint": self.fingerprint,
                        "rows": len(vectors), "dimensions": 384, "dtype": "float32", "sha256": digest}
            manifest_temp = Path(temporary) / "manifest.json"
            with manifest_temp.open("xb") as stream:
                stream.write(_canonical(manifest))
                stream.flush()
                os.fsync(stream.fileno())
            if os.name != "nt":
                os.chmod(manifest_temp, 0o600)
            checkpoint(cancel)
            os.replace(array_temp, array_path)
            checkpoint(cancel)
            os.replace(manifest_temp, manifest_path)
        checkpoint(cancel)
        # Reopen through the same bounded, numeric-only verification path as a
        # cache hit. A post-publication filename swap must not expose a mapping.
        published = self._read(key, len(vectors), cancel)
        if published is None:
            raise EncoderError("Не удалось подтвердить целостность опубликованного кеша эмбеддингов.")
        return published
