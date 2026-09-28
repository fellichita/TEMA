"""Model-independent integrity tests; real-model quality is a separate live smoke."""

from concurrent.futures import CancelledError
import hashlib
import json
import os
from pathlib import Path
from threading import Event
from time import perf_counter

import httpx
import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from app.pilot.encoder import (
    EmbeddingCache, EncoderError, MAX_TEXT_CHARACTERS, MultilingualEncoder,
    load_spec, normalize_text, spec_fingerprint, validate_vectors, verify_artifacts,
)
from app.pilot.embedding_store import DerivedCache
from app.pilot import embedding_store as store_module
from app.runtime.inference import batch_size as runtime_batch_size
from app.sqlite_runtime import sqlite3
from scripts import install_pilot_model
from tests.platform_support import require_symlinks


class CountingEncoder:
    fingerprint = "a" * 64

    def __init__(self):
        self.calls = 0
        self.inputs = []

    def encode(self, texts, *, kind="passage", cancel=None, progress=None):
        self.calls += 1
        self.inputs.append((tuple(texts), kind))
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index, int(hashlib.sha256(text.encode()).hexdigest()[:4], 16) % 384] = 1
        return vectors


def small_encoder():
    encoder = object.__new__(MultilingualEncoder)
    encoder.spec = load_spec()
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "<pad>": 1, "[CLS]": 2, "[SEP]": 3,
                                           "query": 4, "passage": 5, ":": 6, "текст": 7}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    encoder._raw_tokenizer = tokenizer
    encoder._tokenizer = Tokenizer.from_str(tokenizer.to_str())
    encoder._tokenizer.enable_truncation(max_length=512)
    encoder._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
    encoder._np = np
    encoder._inputs = {"input_ids", "attention_mask"}
    encoder._batch_size = runtime_batch_size(encoder.spec["batch_size"])
    encoder.provider = "CPUExecutionProvider"
    encoder.fingerprint = spec_fingerprint(encoder.spec)
    return encoder


def test_spec_uses_official_fp32_files_with_exact_revision_and_license():
    spec = load_spec()
    assert spec["model_id"] == "intfloat/multilingual-e5-small"
    assert spec["license"] == "MIT"
    assert len(spec["revision"]) == 40
    assert spec["files"][0]["remote_path"] == "onnx/model.onnx"
    assert spec["files"][0]["bytes"] == 470268510
    assert "Not independently measured" in spec["reference_parity"]
    assert spec_fingerprint() == spec_fingerprint()


def test_artifact_verification_rejects_wrong_hash_even_when_size_matches(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"bad")
    spec = {"files": [{"name": "weights", "bytes": 3, "sha256": hashlib.sha256(b"yes").hexdigest()}]}
    with pytest.raises(EncoderError, match="сумма"):
        verify_artifacts(tmp_path, spec)
    assert path.read_bytes() == b"bad"


def test_artifact_verification_rejects_symlink_and_honors_cancel(tmp_path):
    require_symlinks()
    target = tmp_path / "actual"
    target.write_bytes(b"yes")
    (tmp_path / "weights").symlink_to(target)
    spec = {"files": [{"name": "weights", "bytes": 3, "sha256": hashlib.sha256(b"yes").hexdigest()}]}
    with pytest.raises(EncoderError):
        verify_artifacts(tmp_path, spec)
    cancel = Event()
    cancel.set()
    with pytest.raises(CancelledError):
        verify_artifacts(tmp_path, spec, cancel)


@pytest.mark.parametrize("text", ["", "\t\n", None, "x" * (MAX_TEXT_CHARACTERS + 1), "\ud800"],
                         ids=["empty", "whitespace", "none", "over-limit", "invalid-unicode"])
def test_invalid_text_fails_before_native_inference(text):
    with pytest.raises(EncoderError):
        normalize_text(text)


def test_cache_keys_include_framing_order_role_model_and_unicode_normalization(tmp_path):
    cache = EmbeddingCache(tmp_path, "a" * 64)
    assert cache.key(["ab", "c"], "query") != cache.key(["a", "bc"], "query")
    assert cache.key(["one", "two"], "query") != cache.key(["two", "one"], "query")
    assert cache.key(["same"], "query") != cache.key(["same"], "passage")
    assert cache.key(["Ａ  B"], "query") == cache.key(["A B"], "query")
    assert cache.key(["same"], "query") != EmbeddingCache(tmp_path, "b" * 64).key(["same"], "query")


def test_valid_cache_is_readonly_mmap_and_reused_without_inference(tmp_path):
    encoder = CountingEncoder()
    cache = EmbeddingCache(tmp_path, encoder.fingerprint)
    initial = cache.encode(encoder, ["память", "energy"])
    assert isinstance(initial, np.memmap)
    assert not initial.flags.writeable
    repeated = cache.encode(encoder, ["память", "energy"])
    assert cache.last_hit
    assert encoder.calls == 1
    np.testing.assert_array_equal(initial, repeated)
    if os.name != "nt":
        assert all(path.stat().st_mode & 0o077 == 0 for path in tmp_path.iterdir())


@pytest.mark.parametrize("damage", ["missing_array", "missing_manifest", "torn_array", "bad_shape", "nonfinite"])
@pytest.mark.parametrize("reusable_vectors", [True, False])
def test_torn_or_invalid_aggregate_uses_only_independently_verified_vectors(tmp_path, damage, reusable_vectors):
    encoder = CountingEncoder()
    cache = EmbeddingCache(tmp_path, encoder.fingerprint)
    cache.encode(encoder, ["text"])
    if not reusable_vectors:
        (tmp_path / "text-vectors-v1.sqlite3").unlink()
    key = cache.key(["text"], "passage")
    array, manifest = tmp_path / f"{key}.npy", tmp_path / f"{key}.json"
    if damage == "missing_array":
        array.unlink()
    elif damage == "missing_manifest":
        manifest.unlink()
    elif damage == "torn_array":
        array.write_bytes(b"torn")
    else:
        values = np.zeros((1, 12 if damage == "bad_shape" else 384), dtype=np.float32)
        values[0, 0] = np.nan if damage == "nonfinite" else 1
        np.save(array, values, allow_pickle=False)
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        metadata["sha256"] = hashlib.sha256(array.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(metadata))
    vectors = cache.encode(encoder, ["text"])
    assert not cache.last_hit
    assert encoder.calls == (1 if reusable_vectors else 2)
    validate_vectors(vectors, 1)


def test_object_npy_is_never_unpickled(tmp_path):
    encoder = CountingEncoder()
    cache = EmbeddingCache(tmp_path, encoder.fingerprint)
    key = cache.key(["text"], "passage")
    array = tmp_path / f"{key}.npy"
    np.save(array, np.array([["not a numeric vector"]], dtype=object), allow_pickle=True)
    manifest = {"schema_version": 1, "key": key, "fingerprint": encoder.fingerprint,
                "rows": 1, "dimensions": 384, "dtype": "float32", "sha256": hashlib.sha256(array.read_bytes()).hexdigest()}
    (tmp_path / f"{key}.json").write_text(json.dumps(manifest))
    validate_vectors(cache.encode(encoder, ["text"]), 1)
    assert encoder.calls == 1


def test_cancelled_cache_does_not_invoke_encoder_or_create_files(tmp_path):
    encoder = CountingEncoder()
    cache = EmbeddingCache(tmp_path / "cache", encoder.fingerprint)
    cancel = Event()
    cancel.set()
    with pytest.raises(CancelledError):
        cache.encode(encoder, ["text"], cancel=cancel)
    assert encoder.calls == 0
    assert not cache.directory.exists()


def test_per_text_cache_reuses_normalized_duplicates_and_preserves_requested_order(tmp_path):
    encoder = CountingEncoder()
    cache = EmbeddingCache(tmp_path, encoder.fingerprint)
    first = cache.encode(encoder, ["one", "two", "one", "Ａ\t B"])
    assert encoder.inputs == [(("one", "two", "A B"), "passage")]
    second = cache.encode(encoder, ["two", "changed", "A B", "one", "changed"])
    assert not cache.last_hit
    assert encoder.inputs[-1] == (("changed",), "passage")
    assert encoder.calls == 2
    np.testing.assert_array_equal(second[[0, 2, 3]], first[[1, 3, 0]])
    np.testing.assert_array_equal(second[1], second[4])
    assert isinstance(second, np.memmap) and not second.flags.writeable


def test_unit_cache_separates_role_and_model_fingerprint(tmp_path):
    encoder = CountingEncoder()
    cache = EmbeddingCache(tmp_path, encoder.fingerprint)
    cache.encode(encoder, ["one", "two"])
    cache.encode(encoder, ["two", "one"], kind="query")
    assert encoder.inputs[-1] == (("two", "one"), "query")
    other = CountingEncoder()
    other.fingerprint = "b" * 64
    EmbeddingCache(tmp_path, other.fingerprint).encode(other, ["one", "two"])
    assert other.calls == 1
    assert encoder.calls == 2


@pytest.mark.parametrize("damage", ["hash", "norm", "nonfinite", "oversized"])
def test_bad_vector_row_is_recomputed_while_other_verified_rows_are_reused(tmp_path, damage):
    encoder = CountingEncoder()
    cache = EmbeddingCache(tmp_path, encoder.fingerprint)
    first = cache.encode(encoder, ["one", "two"])
    unit = cache._unit_key("one", "passage")
    invalid = np.zeros(384, dtype="<f4")
    invalid[0] = np.nan if damage == "nonfinite" else 2
    data = b"x" * 1_000_000 if damage == "oversized" else invalid.tobytes()
    checksum = hashlib.sha256(unit.encode() + data).hexdigest() if damage != "hash" else "0" * 64
    connection = sqlite3.connect(tmp_path / "text-vectors-v1.sqlite3", isolation_level=None)
    connection.execute("UPDATE vectors SET payload=?,sha256=? WHERE key=?", (data, checksum, unit))
    connection.close()
    second = cache.encode(encoder, ["two", "one", "new"])
    assert encoder.inputs[-1] == (("one", "new"), "passage")
    np.testing.assert_array_equal(second[0], first[1])
    validate_vectors(second, 3)


@pytest.mark.parametrize("damage", ["bytes", "oversized", "view", "trigger", "virtual", "table", "columns", "wal"])
def test_untrusted_or_damaged_vector_database_is_discarded_before_recomputation(tmp_path, damage):
    encoder = CountingEncoder()
    cache = EmbeddingCache(tmp_path, encoder.fingerprint, max_bytes=256 * 1024)
    cache.encode(encoder, ["one", "two"])
    path = tmp_path / "text-vectors-v1.sqlite3"
    if damage in {"bytes", "oversized"}:
        path.write_bytes(b"garbage" if damage == "bytes" else b"x" * (256 * 1024))
    elif damage == "wal":
        path.with_name(path.name + "-wal").write_bytes(b"x" * (256 * 1024))
    else:
        connection = sqlite3.connect(path, isolation_level=None)
        sql = {
            "view": "CREATE VIEW unexpected AS SELECT * FROM vectors",
            "trigger": "CREATE TRIGGER unexpected AFTER INSERT ON vectors BEGIN UPDATE vectors SET payload=zeroblob(1536); END",
            "virtual": "CREATE VIRTUAL TABLE unexpected USING fts5(body)",
            "table": "CREATE TABLE unexpected (data TEXT)",
            "columns": "ALTER TABLE vectors ADD COLUMN unexpected TEXT",
        }[damage]
        connection.execute(sql)
        connection.close()
    vectors = cache.encode(encoder, ["two", "one"])
    assert encoder.inputs[-1] == (("two", "one"), "passage")
    assert encoder.calls == 2
    validate_vectors(vectors, 2)
    connection = sqlite3.connect(path)
    assert set(connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master")) == store_module._SCHEMA
    connection.close()


def test_replaced_database_between_open_and_read_cannot_supply_cached_vectors(tmp_path, monkeypatch):
    encoder = CountingEncoder()
    cache = EmbeddingCache(tmp_path, encoder.fingerprint)
    cache.encode(encoder, ["one", "two"])
    original = DerivedCache._connect
    replaced = []
    blocked = []

    def swap(store):
        opened = original(store)
        if not replaced:
            replacement = tmp_path / "replacement.sqlite3"
            replacement.write_bytes(store.path.read_bytes())
            try:
                os.replace(replacement, store.path)
            except PermissionError:
                # Windows can prevent the replacement while SQLite owns it.
                blocked.append(True)
                replacement.unlink()
            replaced.append(True)
        return opened

    monkeypatch.setattr(DerivedCache, "_connect", swap)
    cache.encode(encoder, ["two", "one"])
    assert encoder.calls == (1 if blocked else 2)


@pytest.mark.parametrize("name", ["text-vectors-v1.sqlite3", "text-vectors-v1.sqlite3-journal", "text-vectors-v1.sqlite3-wal"])
def test_cache_database_and_sidecar_symlinks_never_overwrite_their_target(tmp_path, name):
    require_symlinks()
    directory = tmp_path / "cache"
    directory.mkdir()
    target = tmp_path / "outside"
    target.write_bytes(b"Preserve this unrelated file")
    (directory / name).symlink_to(target)
    encoder = CountingEncoder()
    vectors = EmbeddingCache(directory, encoder.fingerprint).encode(encoder, ["public text"])
    validate_vectors(vectors, 1)
    assert target.read_bytes() == b"Preserve this unrelated file"
    assert not (directory / name).is_symlink()


def test_database_replaced_with_symlink_before_sqlite_open_is_not_written(tmp_path, monkeypatch):
    require_symlinks()
    directory = tmp_path / "cache"
    target = tmp_path / "unrelated.sqlite3"
    original = store_module.sqlite3.connect
    connection = original(target)
    connection.execute("CREATE TABLE preserve (value TEXT)")
    connection.close()
    before = target.read_bytes()
    swapped = []

    def connect(*args, **kwargs):
        if not swapped:
            path = directory / "text-vectors-v1.sqlite3"
            path.unlink()
            path.symlink_to(target)
            swapped.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(store_module.sqlite3, "connect", connect)
    encoder = CountingEncoder()
    result = EmbeddingCache(directory, encoder.fingerprint).encode(encoder, ["public text"])
    validate_vectors(result, 1)
    assert target.read_bytes() == before


def test_replaced_aggregate_while_mapping_is_rejected_without_reading_replacement(tmp_path, monkeypatch):
    encoder = CountingEncoder()
    cache = EmbeddingCache(tmp_path, encoder.fingerprint)
    initial = cache.encode(encoder, ["one"])
    key = cache.key(["one"], "passage")
    original = np.memmap
    replaced = []
    blocked = []

    def swap(stream, *args, **kwargs):
        if not isinstance(stream, (str, Path)) and not replaced:
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"Unverified replacement data")
            try:
                os.replace(replacement, tmp_path / (key + ".npy"))
            except PermissionError:
                blocked.append(True)
                replacement.unlink()
            replaced.append(True)
        return original(stream, *args, **kwargs)

    monkeypatch.setattr(np, "memmap", swap)
    checked = cache._read(key, 1, None)
    if blocked:
        np.testing.assert_array_equal(checked, initial)
    else:
        assert checked is None
    np.testing.assert_array_equal(initial, CountingEncoder().encode(["one"]))


def test_empty_embedding_aggregate_remains_readonly_without_invoking_model(tmp_path):
    encoder = CountingEncoder()
    cache = EmbeddingCache(tmp_path, encoder.fingerprint)
    initial = cache.encode(encoder, [])
    repeated = cache.encode(encoder, [])
    assert initial.shape == repeated.shape == (0, 384)
    assert not initial.flags.writeable and not repeated.flags.writeable
    assert cache.last_hit and encoder.calls == 0


def test_verified_legacy_aggregate_seeds_units_without_native_inference(tmp_path):
    encoder = CountingEncoder()
    cache = EmbeddingCache(tmp_path, encoder.fingerprint)
    cache.encode(encoder, ["one", "two"])
    (tmp_path / "text-vectors-v1.sqlite3").unlink()
    cache.encode(encoder, ["one", "two"])
    assert cache.last_hit and encoder.calls == 1
    cache.encode(encoder, ["two", "one", "three"])
    assert encoder.inputs[-1] == (("three",), "passage")


def test_physical_quota_counts_freelist_journal_and_aggregate_staging(tmp_path, monkeypatch):
    limit = 256 * 1024
    observations = []

    def observe():
        observations.append(sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()))

    original_connect, original_fsync = store_module.sqlite3.connect, os.fsync

    class ObservedConnection(sqlite3.Connection):
        def execute(self, *args, **kwargs):
            result = super().execute(*args, **kwargs)
            observe()
            return result

        def executemany(self, *args, **kwargs):
            result = super().executemany(*args, **kwargs)
            observe()
            return result

    def connect(*args, **kwargs):
        return original_connect(*args, factory=ObservedConnection, **kwargs)

    def fsync(descriptor):
        observe()
        return original_fsync(descriptor)

    monkeypatch.setattr(store_module.sqlite3, "connect", connect)
    monkeypatch.setattr(os, "fsync", fsync)
    encoder = CountingEncoder()
    cache = EmbeddingCache(tmp_path, encoder.fingerprint, max_bytes=limit)
    for batch in range(12):
        cache.encode(encoder, [f"batch-{batch}-item-{index}" for index in range(30)])
    connection = original_connect(tmp_path / "text-vectors-v1.sqlite3", isolation_level=None)
    connection.execute("DELETE FROM vectors")
    assert connection.execute("PRAGMA freelist_count").fetchone()[0] > 0
    connection.close()
    cache.encode(encoder, ["last"])
    assert max(observations) <= limit
    assert (tmp_path / "text-vectors-v1.sqlite3").stat().st_size <= limit // 4
    assert not list(tmp_path.glob("*.sqlite3-*"))
    assert not list(tmp_path.glob(".embedding-*"))
    print(json.dumps({"quota_bytes": limit, "peak_observed_bytes": max(observations), "observations": len(observations)}))


def test_locked_old_aggregate_refuses_quota_growth_safely(tmp_path, monkeypatch):
    encoder = CountingEncoder()
    cache = EmbeddingCache(tmp_path, encoder.fingerprint, max_bytes=128 * 1024)
    held = cache.encode(encoder, [f"item-{index}" for index in range(35)])
    unlink = Path.unlink

    def locked(path, *args, **kwargs):
        if path.suffix in {".npy", ".json"}:
            raise PermissionError("Simulated Windows mapping")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked)
    with pytest.raises(EncoderError, match="лимита"):
        cache.encode(encoder, ["different"])
    assert sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()) <= cache.max_bytes
    validate_vectors(held, 35)


def test_cancelled_vector_transaction_rolls_back_and_lock_wait_is_cancellable(tmp_path, monkeypatch):
    encoder = CountingEncoder()
    cache = EmbeddingCache(tmp_path, encoder.fingerprint)
    cache.encode(encoder, ["one"])
    key = cache._unit_key("new", "passage")
    data = CountingEncoder().encode(["new"])[0].tobytes()
    cancel = Event()
    original = DerivedCache._connect

    def observed(store):
        connection, identity = original(store)
        connection.set_trace_callback(lambda sql: cancel.set() if sql.startswith("INSERT OR REPLACE") else None)
        return connection, identity

    monkeypatch.setattr(DerivedCache, "_connect", observed)
    with DerivedCache(tmp_path, cache.max_bytes, cancel) as store:
        with pytest.raises(CancelledError):
            store.write([(key, data)])
    monkeypatch.setattr(DerivedCache, "_connect", original)
    cancel.clear()
    with DerivedCache(tmp_path, cache.max_bytes, cancel) as store:
        assert key not in store.read([key])
        assert cache._unit_key("one", "passage") in store.read([cache._unit_key("one", "passage")])

        class CancelAfterPolling:
            calls = 0

            def is_set(self):
                self.calls += 1
                return self.calls > 4

        start = perf_counter()
        with pytest.raises(CancelledError):
            with DerivedCache(tmp_path, cache.max_bytes, CancelAfterPolling()):
                pytest.fail("The first lease still owns the cache")
        assert perf_counter() - start < 1


def test_cancellation_between_array_and_manifest_does_not_publish_a_valid_hit(tmp_path, monkeypatch):
    encoder = CountingEncoder()
    cache = EmbeddingCache(tmp_path, encoder.fingerprint)
    cancel = Event()
    original = os.replace

    def replace(source, target):
        result = original(source, target)
        if Path(source).name == "vectors.npy":
            cancel.set()
        return result

    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(CancelledError):
        cache.encode(encoder, ["one"], cancel=cancel)
    assert not list(tmp_path.glob("*.json"))
    monkeypatch.setattr(os, "replace", original)
    result = cache.encode(encoder, ["one"])
    assert not cache.last_hit and encoder.calls == 1  # Independently committed vectors are valid.
    validate_vectors(result, 1)


@pytest.mark.parametrize("replacement", ["regular", "nonregular"])
def test_final_publish_rejects_replaced_array_before_exposing_mapping(tmp_path, monkeypatch, replacement):
    encoder = CountingEncoder()
    cache = EmbeddingCache(tmp_path, encoder.fingerprint)
    original = os.replace

    def replace(source, target):
        result = original(source, target)
        if Path(source).name == "manifest.json":
            array = Path(target).with_suffix(".npy")
            array.unlink()
            if replacement == "regular":
                array.write_bytes(b"Unverified cache replacement")
            elif hasattr(os, "mkfifo"):
                os.mkfifo(array)
            else:
                array.mkdir()
        return result

    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(EncoderError, match="целостность опубликованного"):
        cache.encode(encoder, ["one"])


def test_changed_one_of_five_hundred_units_encodes_only_the_missing_text(tmp_path):
    encoder = CountingEncoder()
    cache = EmbeddingCache(tmp_path, encoder.fingerprint)
    texts = [f"public synthetic unit {index}" for index in range(500)]
    initial = cache.encode(encoder, texts)
    texts[211] = "changed public synthetic unit"
    start = perf_counter()
    updated = cache.encode(encoder, texts)
    seconds = perf_counter() - start
    assert len(encoder.inputs[0][0]) == 500 and encoder.inputs[1][0] == (texts[211],)
    np.testing.assert_array_equal(initial[:211], updated[:211])
    np.testing.assert_array_equal(initial[212:], updated[212:])
    print(json.dumps({"synthetic_initial_encoded": 500, "synthetic_changed_encoded": 1, "changed_seconds": seconds}))


def test_long_text_chunks_keep_offsets_overlap_and_model_token_limit():
    encoder = small_encoder()
    text = "текст " * 1400
    chunks = encoder.chunk_text(text)
    normalized = normalize_text(text)
    assert len(chunks.units) >= 3
    assert chunks.truncated is False
    assert chunks.units[-1].end == len(normalized)
    for index, unit in enumerate(chunks.units):
        assert unit.text == normalized[unit.start:unit.end]
        assert unit.token_count <= 512
        if index:
            assert unit.start < chunks.units[index - 1].end
    limited = encoder.chunk_text(text, max_chunks=1)
    assert limited.truncated is True


@pytest.mark.parametrize("bad", ["nan", "zero", "shape"])
def test_invalid_native_embedding_is_rejected(bad):
    encoder = small_encoder()

    class Session:
        def run(self, names, feeds):
            shape = (*feeds["input_ids"].shape, 12 if bad == "shape" else 384)
            return [np.full(shape, np.nan if bad == "nan" else 0, dtype=np.float32)]

    encoder._session = Session()
    with pytest.raises(EncoderError):
        encoder.encode(["текст"])


def test_masked_mean_and_l2_do_not_include_padding():
    encoder = small_encoder()

    class Session:
        def run(self, names, feeds):
            result = np.zeros((*feeds["input_ids"].shape, 384), dtype=np.float32)
            result[..., 0] = 1
            result[..., 1] = np.where(feeds["attention_mask"], 0, 1000)
            return [result]

    encoder._session = Session()
    result = encoder.encode(["текст", "текст " * 30], kind="query")
    assert result.shape == (2, 384)
    np.testing.assert_array_equal(result[:, 0], [1, 1])
    np.testing.assert_array_equal(result[:, 1:], 0)


def test_cuda_token_buckets_reduce_padding_and_preserve_vectors(monkeypatch):
    monkeypatch.setenv("TRENDANALIZER_INFERENCE_BATCH", "4")
    # Equal character lengths, but one versus sixteen word tokens. Character
    # ordering mixes them in every batch; token ordering groups them exactly.
    short = "текст" * 20
    long = "текст " * 15 + "тексттекст"
    assert len(short) == len(long) == 100
    texts = [short, long] * 8

    class Session:
        def __init__(self):
            self.padded = 0
            self.useful = 0

        def run(self, _names, feeds):
            ids, mask = feeds["input_ids"], feeds["attention_mask"]
            self.padded += ids.size
            self.useful += int(mask.sum())
            hidden = np.zeros((*ids.shape, 384), dtype=np.float32)
            hidden[..., 0] = np.where(mask, ids + 1, 1000)
            hidden[..., 1] = np.where(mask, 1, 1000)
            return [hidden]

    cpu = small_encoder()
    cpu._session = Session()
    expected = cpu.encode(texts)
    cuda = small_encoder()
    cuda.provider = "CUDAExecutionProvider"
    cuda._tokenizer.no_padding()
    cuda._session = Session()
    actual = cuda.encode(texts)

    np.testing.assert_array_equal(actual, expected)
    assert cpu._session.useful == cuda._session.useful == 200
    assert cuda._session.padded == 200 < cpu._session.padded == 320


def test_readonly_native_output_uses_safe_masking_fallback():
    encoder = small_encoder()

    class Session:
        def run(self, _names, feeds):
            hidden = np.broadcast_to(np.ones((1, 1, 384), dtype=np.float32),
                                     (*feeds["input_ids"].shape, 384))
            assert not hidden.flags.writeable
            return [hidden]

    encoder._session = Session()
    vectors = encoder.encode(["текст", "текст текст"])
    np.testing.assert_allclose(vectors[:, 0], [1 / np.sqrt(384)] * 2, rtol=1e-6)


def test_inplace_masking_matches_previous_pooling_bit_for_bit():
    encoder = small_encoder()
    observed = []

    class Session:
        def run(self, _names, feeds):
            hidden = np.arange(np.prod((*feeds["input_ids"].shape, 384)),
                               dtype=np.float32).reshape((*feeds["input_ids"].shape, 384))
            hidden = np.sin(hidden).astype(np.float32)
            observed.append((hidden.copy(), feeds["attention_mask"].copy()))
            return [hidden]

    encoder._session = Session()
    result = encoder.encode(["текст", "текст текст", "текст текст текст"])
    hidden, attention = observed[0]
    mask = attention.astype(np.float32)[..., None]
    pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
    expected = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
    np.testing.assert_array_equal(result, expected)


def test_cuda_token_bucketing_keeps_model_length_limit(monkeypatch):
    monkeypatch.setenv("TRENDANALIZER_INFERENCE_BATCH", "2")
    encoder = small_encoder()
    encoder.provider = "CUDAExecutionProvider"
    encoder._tokenizer.no_padding()

    class Session:
        def run(self, _names, feeds):
            assert feeds["input_ids"].shape[1] == 512
            hidden = np.ones((*feeds["input_ids"].shape, 384), dtype=np.float32)
            return [hidden]

    encoder._session = Session()
    assert encoder.encode(["текст " * 700]).shape == (1, 384)


@pytest.mark.parametrize(("url", "allowed"), [
    ("https://huggingface.co/intfloat/model", True), ("https://cas-bridge.xethub.hf.co/file", True),
    ("https://huggingface.co.evil.example/file", False), ("http://huggingface.co/file", False),
    ("https://user:pass@huggingface.co/file", False), ("https://127.0.0.1/file", False),
    ("https://huggingface.co:bad/file", False), ("https://huggingface.co:444/file", False),
])
def test_model_redirect_allowlist(url, allowed):
    assert install_pilot_model._allowed_url(url) is allowed


def test_model_download_checks_streamed_hash_and_never_follows_untrusted_redirect(tmp_path):
    item = {"name": "small", "remote_path": "onnx/model.onnx", "bytes": 3,
            "sha256": hashlib.sha256(b"yes").hexdigest()}
    spec = load_spec()
    requests = []

    def handle(request):
        requests.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://127.0.0.1/metadata"})

    with httpx.Client(transport=httpx.MockTransport(handle), follow_redirects=False) as client:
        with pytest.raises(EncoderError, match="сервер"):
            install_pilot_model._download(client, spec, item, tmp_path / "new")
    assert len(requests) == 1
    assert not (tmp_path / "new").exists()


def test_invalid_existing_model_is_not_overwritten(tmp_path):
    (tmp_path / "valuable").write_text("keep")
    with pytest.raises(EncoderError, match="не перезаписывается"):
        install_pilot_model.install(tmp_path)
    assert (tmp_path / "valuable").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("payload", [b"yes", b"bad", b"too large"])
def test_streamed_model_bytes_enforce_size_hash_and_report_progress(tmp_path, payload):
    spec = load_spec()
    item = {"name": "small", "remote_path": "onnx/model.onnx", "bytes": 3,
            "sha256": hashlib.sha256(b"yes").hexdigest()}
    progress = []

    def handle(request):
        return httpx.Response(200, stream=httpx.ByteStream(payload))

    with httpx.Client(transport=httpx.MockTransport(handle), follow_redirects=False) as client:
        if payload == b"yes":
            install_pilot_model._download(client, spec, item, tmp_path / "download",
                                          progress=lambda done, total: progress.append((done, total)))
            assert (tmp_path / "download").read_bytes() == b"yes"
            assert progress == [(3, 3)]
        else:
            with pytest.raises(EncoderError):
                install_pilot_model._download(client, spec, item, tmp_path / "download")


def test_cancellation_after_native_batch_discards_result():
    encoder = small_encoder()
    cancel = Event()

    class Session:
        def run(self, names, feeds):
            cancel.set()
            return [np.ones((*feeds["input_ids"].shape, 384), dtype=np.float32)]

    encoder._session = Session()
    with pytest.raises(CancelledError):
        encoder.encode(["текст"], cancel=cancel)
