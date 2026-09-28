"""V3 boundaries reject false completeness and preserve immutable provenance."""

import hashlib
from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from app.pilot.contracts import (
    AnalysisResult, AnalysisRun, AnalysisStage, Candidate, Claim, CorpusSnapshot,
    Coverage, DocumentRevisionRef, Evidence, QueryPlan, SearchQuery, TrendCard,
    content_hash, verify_evidence_text,
)

NOW = datetime(2026, 9, 10, tzinfo=timezone.utc)
HASH = "a" * 64
YEARS = tuple(range(2020, 2026))


def plan(**changes):
    values = dict(original_query="  технологии в ИИ  ", language="ru", definition="AI mechanisms",
                  english_query="AI technologies", subdirections=("Learning methods",),
                  queries=(SearchQuery(source="openalex", text="AI technologies"),),
                  completed_years=YEARS, as_of=date(2026, 9, 10), planner_version="test-1")
    return QueryPlan(**(values | changes))


def coverage(**changes):
    values = dict(source="openalex", purpose="history", query_hash=HASH, state="complete",
                  requested_years=YEARS, completed_years=YEARS, pagination_exhausted=True,
                  comparable=True, scanned_records=1, accepted_records=1)
    return Coverage(**(values | changes))


def revision(**changes):
    return DocumentRevisionRef(**(dict(revision_id="revision-1", study_id="study-1", source="openalex",
        source_id="W1", text_hash=HASH, observed_at=NOW, publication_year=2025) | changes))


def snapshot(**changes):
    return CorpusSnapshot(**(dict(snapshot_id="discovery-1", plan_hash=plan().plan_hash, purpose="discovery",
        created_at=NOW, as_of=NOW.date(), documents=(revision(),),
        coverage=(coverage(purpose="discovery"),), normalizer_version="1", deduplication_version="1") | changes))


def candidate(**changes):
    return Candidate(**(dict(candidate_id="candidate-1", plan_hash=plan().plan_hash, label="Mechanism",
        definition="A specific mechanism", admission_rule_version="1", admission_rule_hash=HASH,
        discovery_snapshot_id="discovery-1", discovery_study_ids=("study-1",),
        specificity="specific_technology") | changes))


def evidence(**changes):
    return Evidence(**(dict(evidence_id="evidence-1", revision_id="revision-1", study_id="study-1",
        source="openalex", source_url="https://openalex.org/W1", retrieved_at=NOW, text_hash=HASH,
        text_field="abstract", start=0, end=4, quote="text") | changes))


def card(**changes):
    return TrendCard(**(dict(candidate=candidate(), category="early_signal", quality="partial",
        claims=(), evidence=(evidence(),), limitations=("History incomplete",)) | changes))


def result(**changes):
    return AnalysisResult(**(dict(result_id="result-1", run_id="run-1", query_plan=plan(), created_at=NOW,
        quality="partial", snapshots=(snapshot(),), cards=(card(),), limitations=("History incomplete",)) | changes))


def test_query_hash_normalizes_surrounding_whitespace_and_is_stable_after_json_roundtrip():
    first = plan()
    assert first.original_query == "технологии в ИИ"
    assert first.plan_hash == plan(original_query="технологии в ИИ").plan_hash
    assert QueryPlan.model_validate_json(first.model_dump_json()).plan_hash == first.plan_hash
    assert plan(definition="Another interpretation").plan_hash != first.plan_hash
    assert plan(exclusions=("medical AI",)).plan_hash != first.plan_hash
    assert content_hash({"b": 2, "a": 1}) == content_hash({"a": 1, "b": 2})


@pytest.mark.parametrize("query", ["", "   ", "a\nb", "a\x00b", "ab\u200bb", "a" * 501])
def test_invalid_queries_are_rejected_before_expensive_work(query):
    with pytest.raises(ValidationError):
        plan(original_query=query)


@pytest.mark.parametrize("years", [tuple(range(2021, 2027)), (2020, 2021, 2022, 2023, 2025, 2025),
                                   (2019, 2020, 2021, 2022, 2023, 2025), tuple(reversed(YEARS))])
def test_historical_window_rejects_current_gaps_duplicates_and_reversed_years(years):
    with pytest.raises(ValidationError):
        plan(completed_years=years)


def test_contract_rejects_unknown_fields_mutation_naive_times_and_boolean_counts():
    with pytest.raises(ValidationError):
        plan(fake_topics=("hardcoded",))
    with pytest.raises(ValidationError):
        plan().definition = "changed"
    with pytest.raises(ValidationError):
        revision(observed_at=NOW.replace(tzinfo=None))
    with pytest.raises(ValidationError):
        coverage(scanned_records=True)


@pytest.mark.parametrize("changes", [dict(pagination_exhausted=False), dict(limit_reached=True),
    dict(completed_years=YEARS[:-1]), dict(scanned_records=2, unresolved_records=1),
    dict(scanned_records=3), dict(completed_years=(*YEARS, 2019))])
def test_complete_coverage_cannot_hide_caps_unresolved_pages_or_missing_years(changes):
    with pytest.raises(ValidationError):
        coverage(**changes)


def test_discovery_and_other_sources_never_certify_reference_history():
    assert coverage().complete_history
    assert not coverage(purpose="discovery").complete_history
    assert not coverage(source="crossref").complete_history
    assert not coverage(comparable=False).complete_history
    with pytest.raises(ValidationError):
        coverage(state="partial")


def test_quotations_preserve_spaces_offsets_unicode_and_archived_text_version():
    text = " Важный результат.\n"
    quote = " Важный "
    item = evidence(quote=quote, end=len(quote), text_hash=hashlib.sha256(text.encode()).hexdigest())
    assert item.quote == quote
    verify_evidence_text(item, text)
    with pytest.raises(ValueError, match="archived text"):
        verify_evidence_text(item, text + " ")
    with pytest.raises(ValidationError):
        evidence(quote=quote, end=len(quote) + 1)


@pytest.mark.parametrize("url", ["javascript:alert(1)", "file:///etc/passwd", "https://secret:key@example.org/"])
def test_evidence_display_urls_reject_executables_files_and_credentials(url):
    with pytest.raises(ValidationError):
        evidence(source_url=url)


def test_supported_and_numeric_claims_require_their_distinct_provenance():
    with pytest.raises(ValidationError):
        Claim(claim_id="c", role="problem", text="An assertion", support="supported")
    with pytest.raises(ValidationError):
        Claim(claim_id="c", role="summary", text="Ten papers", support="unverified", kind="numeric")
    claim = Claim(claim_id="c", role="case", text="Case", support="supported",
                  evidence_ids=("evidence-1",), grounding_method="human-review/1")
    assert card(claims=(claim,)).claims[0] == claim
    with pytest.raises(ValidationError):
        card(claims=(claim,), evidence=())


def test_snapshots_reject_duplicate_revisions_wrong_purpose_and_future_publications():
    with pytest.raises(ValidationError):
        snapshot(documents=(revision(), revision()))
    with pytest.raises(ValidationError):
        snapshot(coverage=(coverage(),))
    with pytest.raises(ValidationError):
        snapshot(documents=(revision(publicly_available_at=date(2026, 9, 11)),))


def test_run_success_cannot_cross_cancellation_fence_or_hide_missing_output():
    with pytest.raises(ValidationError):
        AnalysisStage(run_id="r", stage="plan", attempt_id="1", state="succeeded", updated_at=NOW,
                      input_hash=HASH)
    with pytest.raises(ValidationError):
        AnalysisRun(run_id="r", plan_hash=HASH, state="succeeded", created_at=NOW, updated_at=NOW,
                    quality="partial", result_id="res", cancellation_requested_at=NOW)
    run = AnalysisRun(run_id="r", plan_hash=HASH, state="succeeded", created_at=NOW, updated_at=NOW,
                      quality="partial", result_id="res")
    assert run.state == "succeeded" and run.quality == "partial"


def test_result_validates_query_memberships_and_immutable_revision_definitions():
    assert AnalysisResult.model_validate_json(result().model_dump_json()) == result()
    with pytest.raises(ValidationError):
        result(snapshots=(snapshot(plan_hash="b" * 64),))
    with pytest.raises(ValidationError):
        result(cards=(card(candidate=candidate(discovery_study_ids=("unknown",))),))
    with pytest.raises(ValidationError):
        result(cards=(card(evidence=(evidence(study_id="different"),)),))
    with pytest.raises(ValidationError):
        result(snapshots=(snapshot(), snapshot(snapshot_id="second", documents=(revision(study_id="other"),))))


def test_incomplete_cards_and_empty_results_cannot_claim_complete_quality():
    with pytest.raises(ValidationError):
        result(quality="complete")
    with pytest.raises(ValidationError):
        result(cards=(), quality="complete")
    assert result(cards=(), quality="insufficient_data").quality == "insufficient_data"
    with pytest.raises(ValidationError):
        card(category="confirmed_trend", quality="complete", historical_snapshot_id="history",
             assessment_hash=HASH)


def test_one_finished_history_query_does_not_hide_another_truncated_query():
    claims = tuple(Claim(claim_id=role, role=role, text=f"Supported {role}", support="supported",
                         evidence_ids=("evidence-1",), grounding_method="human-review/1")
                   for role in ("problem", "advantage", "case"))
    confirmed = card(category="confirmed_trend", quality="complete", claims=claims,
                     historical_snapshot_id="history", assessment_hash=HASH)
    partial_coverage = coverage(state="partial", reasons=("limit",), limit_reached=True,
                                pagination_exhausted=False)
    partial = snapshot(snapshot_id="history", purpose="history", coverage=(coverage(), partial_coverage))
    with pytest.raises(ValidationError, match="Incomplete historical coverage"):
        result(cards=(confirmed,), quality="complete", snapshots=(snapshot(), partial))
