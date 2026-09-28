"""The local E5 publication score is bounded, versioned, and restart-safe."""

from __future__ import annotations

from concurrent.futures import CancelledError
import hashlib
import math
from threading import Event

import pytest

from app.pilot.encoder import EncoderError
from app.pilot.publication_confidence import (MAX_PUBLICATIONS, ordinal_score,
                                              completed_cached_scores, score_publications,
                                              validate_confidence)
from app.runtime.jobs import TaskCancelled


def _publication(number: int) -> dict:
    return {"publication_id": hashlib.sha256(str(number).encode()).hexdigest(),
            "title": f"Battery study {number}", "summary": "Solid electrolyte cells."}


class _Context:
    run_id = "a" * 32

    def __init__(self):
        self.cancel_event = Event()
        self.saved = {}
        self.progresses = []

    def check_cancelled(self):
        if self.cancel_event.is_set():
            raise TaskCancelled()

    def load_checkpoint(self, stage):
        return self.saved.get(stage)

    def checkpoint(self, stage, value):
        self.saved[stage] = value

    def progress(self, stage, message, completed=0, total=0):
        self.progresses.append((stage, message, completed, total))


class _Encoder:
    fingerprint = "test-e5-fingerprint"

    def __init__(self, similarities):
        self.similarities = similarities
        self.calls = []

    def encode(self, texts, *, kind, cancel):
        self.calls.append((kind, list(texts)))
        assert not cancel.is_set()
        if kind == "query":
            return [(1.0, 0.0) for _ in texts]
        return [(value, math.sqrt(1 - value * value)) for value in self.similarities[:len(texts)]]


def test_ordinal_score_uses_fixed_clipped_half_up_scale():
    assert [ordinal_score(value) for value in (0.6, 0.75, 0.825, 0.9, 0.95)] == [0, 0, 50, 100, 100]
    assert ordinal_score(0.768) == 12
    assert ordinal_score(0.882) == 88
    assert ordinal_score(float("nan")) is None
    assert ordinal_score(1.01) is None
    assert ordinal_score(True) is None


def test_score_publications_batches_only_fifteen_and_persists_each_score():
    context, encoder = _Context(), _Encoder([0.85] * 20)
    items = [_publication(number) for number in range(20)]
    scored = score_publications(items, query="solid-state batteries", encoder=encoder, context=context)
    assert len(scored) == MAX_PUBLICATIONS == len(context.saved)
    assert encoder.calls[0] == ("query", ["solid-state batteries"])
    assert encoder.calls[1][0] == "passage" and len(encoder.calls[1][1]) == MAX_PUBLICATIONS
    assert len(encoder.calls) == 2
    assert all(score["score"] == 67 and score["basis"] == "title_and_summary"
               and score["evidence_quote"].startswith("Battery study")
               and "не вероятность" in score["reason"] for score in scored.values())
    assert context.progresses[-1][2:] == (MAX_PUBLICATIONS, MAX_PUBLICATIONS)
    assert all(progress[0] == "publish" for progress in context.progresses)

    # A missing model can still read completed checkpoints. A changed title
    # invalidates only that publication's saved score.
    assert score_publications(items, query="solid-state batteries", encoder=None, context=context) == scored
    changed = [dict(items[0], title="Changed title"), *items[1:]]
    reused = score_publications(changed, query="solid-state batteries", encoder=None, context=context)
    assert items[0]["publication_id"] not in reused and len(reused) == MAX_PUBLICATIONS - 1


def test_complete_cpu_checkpoint_can_skip_model_load_only_for_same_fingerprint():
    context, encoder = _Context(), _Encoder([0.85, 0.86])
    items = [_publication(1), _publication(2)]
    scored = score_publications(items, query="battery", encoder=encoder, context=context)
    assert completed_cached_scores(items, query="battery", english_query=None,
                                   expected_fingerprint=encoder.fingerprint, context=context) == scored
    assert completed_cached_scores(items, query="battery", english_query=None,
                                   expected_fingerprint="other-model", context=context) is None
    changed = [items[0], dict(items[1], summary="Different evidence")]
    assert completed_cached_scores(changed, query="battery", english_query=None,
                                   expected_fingerprint=encoder.fingerprint, context=context) is None


def test_original_and_english_queries_are_deduplicated_and_part_of_checkpoint_identity():
    item, context = _publication(1), _Context()
    encoder = _Encoder([0.85])
    first = score_publications([item], query="твердотельные аккумуляторы",
                               english_query="solid-state batteries", encoder=encoder, context=context)
    assert encoder.calls[0] == ("query", ["твердотельные аккумуляторы", "solid-state batteries"])
    assert len(first) == 1
    changed = score_publications([item], query="твердотельные аккумуляторы",
                                 english_query="solid batteries", encoder=None, context=context)
    assert changed == {}
    duplicate = _Encoder([0.85])
    score_publications([item], query="solid-state batteries", english_query="solid-state batteries",
                       encoder=duplicate, context=_Context())
    assert duplicate.calls[0] == ("query", ["solid-state batteries"])


def test_encoder_error_keeps_publications_unscored_without_fake_number():
    class Broken(_Encoder):
        def encode(self, texts, *, kind, cancel):
            raise EncoderError("Model unavailable")

    context = _Context()
    assert score_publications([_publication(1)], query="battery", encoder=Broken([]), context=context) == {}
    assert context.saved == {}


def test_user_cancellation_is_not_misreported_as_model_failure():
    class Cancelling(_Encoder):
        def encode(self, texts, *, kind, cancel):
            context.cancel_event.set()
            raise CancelledError()

    context = _Context()
    with pytest.raises(TaskCancelled):
        score_publications([_publication(1)], query="battery", encoder=Cancelling([]), context=context)


def test_publication_confidence_requires_exact_grounding_and_honest_basis():
    base = {"score": 75, "reason": "Fixed local E5 relevance scale.",
            "evidence_quote": "solid electrolyte", "basis": "title_and_summary"}
    assert validate_confidence(base, "Battery study", "A SOLID   electrolyte cell") == base
    assert validate_confidence({**base, "evidence_quote": "invented result"},
                               "Battery study", "A solid electrolyte cell") is None
    assert validate_confidence({**base, "basis": "title"},
                               "Battery study", "A solid electrolyte cell") is None
    assert validate_confidence({**base, "score": True},
                               "Battery study", "A solid electrolyte cell") is None
