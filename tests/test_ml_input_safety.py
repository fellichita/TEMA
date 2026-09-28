"""Regressions for malformed text, JSON depth and early cancellation."""

import html
import json
import re
import subprocess
import sys
import unicodedata
from concurrent.futures import CancelledError
from copy import deepcopy
from pathlib import Path
from threading import Event

import pytest

from app.input_safety import MAX_ABSTRACT_CHARACTERS, MAX_TITLE_CHARACTERS
from app.ml import engine, semantic, service
from app.ml.contracts import AnalysisInputError, AnalysisOptions
from app.ml.corpus import MAX_JSON_DEPTH, _check_json_depth, read_snapshot, unpack_snapshot
from app.ml.text import clean
from tests.mvp_fixture import snapshot
from tests.test_ml_cli_errors import assert_input_failure, run_cli

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = AnalysisOptions(topic="photonic neuromorphic computing")


@pytest.mark.parametrize("value", [
    None, "", "<>", "<><p>word</p>", "<<>", "word <incomplete", "<<<<<",
    "<p a='>'>one</p>\\n<p>two</p>", "before <tag\nattribute> after",
    "ＡＢＣ &amp; &#65; &#x41; &lt;tag&gt; &apos; &copy;", "&#38;lt;tag&#38;gt;",
    "&amp;lt; &#0; &#xD800; &#1114112; &#128; &notit; &#x; &#;",
    "&#00065without-semicolon and &#x0000041tail", "\t first\nsecond\r third  ",
])
def test_clean_preserves_existing_text_and_one_pass_html_semantics(value):
    normalized = unicodedata.normalize("NFKC", html.unescape(value or ""))
    expected = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", normalized.replace("\\n", " "))).strip()
    assert clean(value) == expected


@pytest.mark.parametrize("value,expected", [
    ("&#" + "9" * 10_000 + ";", "\ufffd"),
    ("&#" + "0" * 10_000 + "65;", "A"),
    ("&#x" + "f" * 10_000 + ";", "\ufffd"),
    ("&#x" + "0" * 10_000 + "41;", "A"),
])
def test_long_numeric_entities_are_handled_without_big_integer_conversion(value, expected):
    assert clean(value) == expected


def test_clean_accepts_joined_and_unicode_expanded_valid_fields():
    # Callers may clean joined fields or already NFKC-expanded source text.
    title = "a" * MAX_TITLE_CHARACTERS
    abstract = "\ufb03" * MAX_ABSTRACT_CHARACTERS
    assert clean(title + ". " + abstract) == title + ". " + "ffi" * MAX_ABSTRACT_CHARACTERS
    assert clean(clean(abstract)) == "ffi" * MAX_ABSTRACT_CHARACTERS


def test_maximum_unclosed_markup_finishes_in_a_bounded_subprocess():
    # The previous regex takes tens of seconds for this permitted abstract and
    # holds the GIL. The subprocess timeout also protects the test runner.
    completed = subprocess.run(
        [sys.executable, "-c", ("from app.ml.text import clean; "
         "text='<'*200000; assert clean(text)==text; print('ok')")],
        cwd=ROOT, capture_output=True, text=True, timeout=5, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "ok\n"


@pytest.mark.parametrize("field,limit", [
    ("title", MAX_TITLE_CHARACTERS), ("abstract", MAX_ABSTRACT_CHARACTERS),
])
def test_snapshot_enforces_raw_text_limits_before_cleaning(tmp_path, monkeypatch, field, limit):
    data = snapshot()
    document = data["batches"][0]["documents"][0]["document"]
    document[field] = "<" * (limit + 1)
    path = tmp_path / "oversized.json"
    payload = json.dumps(data).encode()
    path.write_bytes(payload)
    monkeypatch.setattr(engine, "clean", lambda value: pytest.fail("Oversized snapshot reached cleaning"))
    with pytest.raises(AnalysisInputError, match=f"document.{field}.*{limit}"):
        service.run_analysis(None, OPTIONS, snapshot_path=path)
    assert path.read_bytes() == payload


@pytest.mark.parametrize("field,limit", [
    ("title", MAX_TITLE_CHARACTERS), ("abstract", MAX_ABSTRACT_CHARACTERS),
])
def test_snapshot_accepts_boundary_without_rewriting_the_source(field, limit):
    data = snapshot()
    document = data["batches"][0]["documents"][0]["document"]
    document[field] = "<" * limit
    before = deepcopy(data)
    corpus = unpack_snapshot(data)
    assert corpus["entries"][0]["document"][field] == "<" * limit
    assert data == before


@pytest.mark.parametrize("field,limit", [
    ("title", MAX_TITLE_CHARACTERS), ("abstract", MAX_ABSTRACT_CHARACTERS),
])
def test_direct_preparation_preserves_usable_version_behind_oversized_one(field, limit):
    original = snapshot()["batches"][0]["documents"][0]
    damaged = deepcopy(original)
    damaged["revision_id"] = "damaged"
    damaged["document"][field] = "<" * (limit + 1)
    damaged["document"]["abstract"] += " extra"  # Prefer it if raw guards are bypassed.
    entries = [damaged, original]
    before = deepcopy(entries)
    studies, diagnostic = engine.prepare(entries, OPTIONS)
    assert len(studies) == 1
    assert studies[0]["title"] == original["document"]["title"]
    assert studies[0]["abstract"] == original["document"]["abstract"]
    assert len(studies[0]["versions"]) == 2
    assert diagnostic["rejected"] == {}
    assert entries == before
    assert engine.prepare([damaged], OPTIONS)[1]["rejected"] == {f"oversized_{field}_requires_review": 1}


def test_semantic_eligibility_checks_size_before_cleaning(monkeypatch):
    document = snapshot()["batches"][0]["documents"][0]["document"]
    document["title"] = "<" * (MAX_TITLE_CHARACTERS + 1)
    monkeypatch.setattr(semantic, "clean", lambda value: pytest.fail("Oversized text reached semantic cleaning"))
    assert semantic._document_skip_reason(document, 2025) == "oversized_title_requires_review"


def test_oversized_retraction_version_still_withdraws_usable_identity():
    original = snapshot()["batches"][0]["documents"][0]
    withdrawn = deepcopy(original)
    withdrawn["document"]["title"] = "RETRACTED: " + "<" * (MAX_TITLE_CHARACTERS + 1)
    assert engine.prepare([original, withdrawn], OPTIONS)[0] == []


@pytest.mark.parametrize("url", ["https://example.org:invalid/study", "https://bad host/study"])
def test_bad_url_is_diagnosed_and_does_not_mask_usable_version(url):
    original = snapshot()["batches"][0]["documents"][0]
    invalid = deepcopy(original)
    invalid["revision_id"] = "invalid-url"
    invalid["document"]["url"] = url
    invalid["document"]["abstract"] += " More measured results from the photonic device."
    assert engine.prepare([invalid], OPTIONS)[1]["rejected"] == {"invalid_source_url": 1}
    studies, diagnostic = engine.prepare([invalid, original], OPTIONS)
    assert len(studies) == 1 and studies[0]["url"] == original["document"]["url"]
    assert diagnostic["rejected"] == {}


def nested(depth):
    value = "[brackets in a string do not add depth]"
    for _ in range(depth):
        value = [value]
    return value


def test_json_depth_has_a_defined_boundary_including_unconsumed_metadata(tmp_path):
    _check_json_depth(nested(MAX_JSON_DEPTH))
    with pytest.raises(AnalysisInputError, match="Вложенность"):
        _check_json_depth(nested(MAX_JSON_DEPTH + 1))
    data = snapshot()
    data["history"]["request"]["extra"] = nested(MAX_JSON_DEPTH)
    path = tmp_path / "deep-metadata.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(AnalysisInputError, match="Вложенность"):
        read_snapshot(path)


def test_ordinary_nested_metadata_remains_verbatim(tmp_path):
    data = snapshot()
    data["history"]["request"]["extra"] = nested(30)
    path = tmp_path / "metadata.json"
    payload = json.dumps(data).encode()
    path.write_bytes(payload)
    assert read_snapshot(path)["provenance"]["request"] == data["history"]["request"]
    assert path.read_bytes() == payload


@pytest.mark.parametrize("topic_args", [[], ["--topic", OPTIONS.topic]])
def test_decoder_recursion_is_a_readable_cli_input_error(tmp_path, topic_args):
    source = tmp_path / "deep.json"
    source.write_bytes(b"[" * 10_000 + b"0" + b"]" * 10_000)
    output = tmp_path / "result.json"
    completed = run_cli(tmp_path, "--snapshot", source, *topic_args, "--output", output)
    assert_input_failure(completed, output, "Вложенность")


def test_cancelled_inspection_and_retraction_scan_stop_before_reading_input(tmp_path):
    cancel = Event()
    cancel.set()
    with pytest.raises(CancelledError):
        service.inspect_snapshot(tmp_path / "not-read.json", cancel=cancel)
    with pytest.raises(CancelledError):
        engine.analysis_entries(snapshot()["batches"][0]["documents"], 2025, cancel=cancel)


def test_cancellation_during_cleaning_stops_before_scope_evaluation(monkeypatch):
    original = snapshot()["batches"][0]["documents"][0]
    cancel = Event()
    real_clean = engine.clean

    def cancelling_clean(value):
        result = real_clean(value)
        if value == original["document"]["abstract"]:
            cancel.set()
        return result

    monkeypatch.setattr(engine, "clean", cancelling_clean)
    monkeypatch.setattr(engine, "scope_check", lambda *args: pytest.fail("Cancelled input reached scope"))
    with pytest.raises(CancelledError):
        engine.prepare([original], OPTIONS, cancel)


def test_shared_input_rules_are_part_of_the_implementation_fingerprint(tmp_path, monkeypatch):
    from app.ml import provenance

    directory = tmp_path / "app" / "ml"
    directory.mkdir(parents=True)
    (directory / "engine.py").write_text("model-v1", encoding="utf-8")
    shared = directory.parent / "input_safety.py"
    shared.write_text("validation-v1", encoding="utf-8")
    monkeypatch.setattr(provenance, "__file__", str(directory / "provenance.py"))
    first = provenance.implementation_fingerprint()
    shared.write_text("validation-v2", encoding="utf-8")
    assert provenance.implementation_fingerprint() != first
