"""Query expansion must preserve identity and source request semantics."""

from datetime import date, datetime, timezone
from pathlib import Path
from threading import Event
from uuid import uuid4

import numpy as np
import pytest

from app.pilot.multisource.contracts import QueryTerm, TechnologyConcept, load_policy
from app.pilot.multisource.matching import exact_confirmed_match, propose_semantic_matches
from app.pilot.multisource.queries import build_manual_profile, compile_wordstat_monthly, normalize_term
from app.pilot.multisource.store import object_digest
from app.runtime.jobs import TaskCancelled, TaskFailure


NOW = datetime(2026, 9, 19, tzinfo=timezone.utc)
HASH = "a" * 64


def _concept(label: str, *, aliases: tuple[QueryTerm, ...] = ()) -> TechnologyConcept:
    return TechnologyConcept(concept_id=uuid4(), label=label, definition="Технологический подход",
                             identity_status="confirmed", confirmed_at=NOW,
                             provenance_hashes=(HASH,), aliases=aliases)


def test_manual_profile_requires_no_cloud_and_keeps_meaningful_punctuation() -> None:
    profile = build_manual_profile("  память в ДНК ", "Метод записи данных", seed_terms=("ДНК память", "C++"),
                                   primary_phrase="ДНК память", confirmed_at=NOW)
    assert profile.original_query == "память в ДНК"
    assert tuple(item.text for item in profile.terms) == ("ДНК память", "C++")
    assert profile.scientific_plan_hash is None
    assert normalize_term("  3D   NAND ") == "3D NAND"
    assert normalize_term("C++") == "C++"


def test_profile_version_requires_explicit_previous_artifact_reference() -> None:
    first = build_manual_profile("молекулярная память", "Запись данных", seed_terms=("ДНК память",),
                                 primary_phrase="ДНК память", confirmed_at=NOW)
    with pytest.raises(TaskFailure):
        build_manual_profile(first.original_query, first.definition, seed_terms=("ДНК память",), previous=first)
    second = build_manual_profile(first.original_query, first.definition, seed_terms=("ДНК память",),
                                  primary_phrase="ДНК память", confirmed_at=NOW, previous=first,
                                  previous_hash=object_digest(first))
    assert second.profile_id == first.profile_id and second.version == 2 and second.supersedes_hash == object_digest(first)
    with pytest.raises(TaskFailure):
        build_manual_profile("другая область", first.definition, seed_terms=("ДНК память",),
                             previous=first, previous_hash=object_digest(first))


def test_monthly_requests_are_bounded_and_primary_is_not_selected_after_results() -> None:
    policy, _ = load_policy()
    profile = build_manual_profile("хранение данных", "Молекулярные носители",
                                   seed_terms=("ДНК хранение", "молекулярное хранение", "архивирование ДНК", "C++"),
                                   primary_phrase="ДНК хранение", confirmed_at=NOW)
    requests = compile_wordstat_monthly(profile, policy, from_date=date(2025, 1, 1), to_date=date(2026, 8, 31))
    assert len(requests) == policy.wordstat_seed_limit == 3
    assert requests[0].phrase == "ДНК хранение"
    assert requests[0].api_body("b1g-example") == {"phrase": "ДНК хранение", "period": "PERIOD_MONTHLY",
                                                  "fromDate": "2025-01-01T00:00:00Z",
                                                  "toDate": "2026-08-31T00:00:00Z", "folderId": "b1g-example"}
    assert requests[0].request_hash != requests[1].request_hash
    with pytest.raises(TaskFailure):
        compile_wordstat_monthly(profile, policy, from_date=date(2025, 1, 2), to_date=date(2026, 8, 31))
    with pytest.raises(TaskFailure):
        compile_wordstat_monthly(profile, policy, from_date=date(2025, 1, 1), to_date=date(2026, 8, 30))


def test_wordstat_operator_in_primary_is_not_sent_as_monthly_query() -> None:
    policy, _ = load_policy()
    profile = build_manual_profile("язык C++", "Технология компиляции", seed_terms=("C++", "компиляторы"),
                                   primary_phrase="C++", confirmed_at=NOW)
    with pytest.raises(TaskFailure, match="оператор"):
        compile_wordstat_monthly(profile, policy, from_date=date(2025, 1, 1), to_date=date(2026, 8, 31))


def test_exact_match_only_uses_reviewed_aliases_and_exposes_ambiguity() -> None:
    rag = QueryTerm(text="RAG", language="en", role="technology", origin="user",
                    status="proposed")
    a = _concept("retrieval augmented generation", aliases=(rag,))
    assert exact_confirmed_match("RAG", (a,)).state == "none"
    confirmed = rag.model_copy(update={"status": "confirmed", "confirmed_at": NOW})
    b = _concept("retrieval augmented generation", aliases=(confirmed,))
    assert exact_confirmed_match(" rag ", (b,)).concept_id == b.concept_id
    c = _concept("regional analysis group", aliases=(confirmed,))
    result = exact_confirmed_match("RAG", (b, c))
    assert result.state == "ambiguous" and result.concept_id is None and len(result.candidate_ids) == 2
    older = b.model_copy(update={"confirmed_at": datetime(2025, 1, 1, tzinfo=timezone.utc)})
    assert exact_confirmed_match("RAG", (older,), cutoff=datetime(2025, 2, 1, tzinfo=timezone.utc)).state == "none"
    assert exact_confirmed_match(older.label, (older,), cutoff=datetime(2025, 2, 1,
                                                                        tzinfo=timezone.utc)).concept_id == older.concept_id


class _Encoder:
    def encode(self, texts, *, kind="passage", cancel=None, progress=None):
        result = np.zeros((len(texts), 384), dtype=np.float32)
        for index in range(len(texts)):
            result[index, 0 if index == 1 else index % 384] = 1
        return result


def test_semantic_matching_returns_review_only_top_five_and_honours_cancel() -> None:
    concepts = tuple(_concept(f"метод {index}") for index in range(8))
    proposals = propose_semantic_matches("молекулярное хранение", concepts, _Encoder())
    assert len(proposals) == 5 and all(item.status == "proposed" for item in proposals)
    assert proposals[0].similarity == 1.0
    cancel = Event()
    cancel.set()
    with pytest.raises(TaskCancelled):
        propose_semantic_matches("молекулярное хранение", concepts, _Encoder(), cancel=cancel)


def test_no_unsafe_automatic_normalization_or_guessing(tmp_path: Path) -> None:
    del tmp_path
    with pytest.raises(TaskFailure):
        normalize_term("+++")
    with pytest.raises(TaskFailure):
        build_manual_profile("ДНК", "Метод записи", seed_terms=("ДНК", "днк"), confirmed_at=NOW)
