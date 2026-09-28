"""Regression checks for the cached legacy text helpers."""

from app.ml.text import _denied_implementation, sentences


def test_sentence_cache_does_not_expose_mutable_shared_results():
    source = "The optical system was measured. Results were reproduced."
    first = sentences(source)
    first.clear()
    assert sentences(source) == [
        "The optical system was measured.",
        "Results were reproduced.",
    ]


def test_cached_denial_preserves_not_required_and_external_executor_meaning():
    assert not _denied_implementation("Optical neural inference does not require a GPU implementation.")
    assert _denied_implementation("Optical neural inference is not implemented on the chip.")
    assert _denied_implementation("Optical neural inference is not implemented on the chip.")
