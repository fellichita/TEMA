"""Distinguish a saved job from evaluated candidates and a legitimate empty TOP."""

import pytest

from app.pilot.completion import evaluation_progress, label_stage_limitations
from app.pilot.export import read_result_package
from tests.test_pilot_evidence import candidate, query_plan
from tests.test_pilot_service_signals import runtime_for, source_documents, wait_success


def card(identifier, category="unassessed_cluster", *, assessed=False, specificity="uncertain"):
    return {"candidate": {"candidate_id": identifier, "specificity": specificity}, "category": category,
            "assessment_hash": "verified-artifact" if assessed else None,
            "historical_snapshot_id": "verified-history" if assessed else None}


def test_original_live_shape_is_not_a_completed_scientific_analysis():
    result = {"cards": [card(str(index)) for index in range(30)], "top_trend_ids": [],
              "candidate_queue": [{}] * 453, "quality": "partial"}
    progress = evaluation_progress(result)
    assert progress.state == "not_assessed"
    assert progress.pending == progress.unresolved_mechanisms == 30
    assert progress.history_assessed == 0 and progress.history_required == 30
    assert "Научная оценка кандидатов не завершена" in progress.message
    assert "Исторические оценки: 0 из 30" in progress.details


@pytest.mark.parametrize("category", ["established_topic", "declining", "transient_burst", "insufficient_evidence"])
def test_assessed_negative_result_and_pending_queue_are_not_execution_failures(category):
    result = {"cards": [card("evaluated", category, assessed=True, specificity="specific_technology")],
              "top_trend_ids": [], "quality": "partial", "candidate_queue": [{}] * 453}
    progress = evaluation_progress(result)
    assert progress.state == "assessed" and progress.pending == 0
    assert "Автоматическая проверка выполнена;" in progress.message
    assert "не независимая научная валидация" in progress.message
    if category == "insufficient_evidence":
        assert "недостаточно доказательств" in progress.details


def test_zero_cards_off_scope_and_partial_evaluation_remain_distinct():
    assert evaluation_progress({"cards": []}).state == "no_candidates"
    queue_only = evaluation_progress({"cards": [], "candidate_queue": [{}]})
    assert "не найдены" not in queue_only.message
    assert evaluation_progress({"cards": [card("outside", "off_scope")]}).state == "assessed"
    progress = evaluation_progress({"cards": [card("done", "weak_signal_candidate", assessed=True,
        specificity="specific_technology"), card("pending")]})
    assert progress.state == "partially_assessed" and progress.pending == 1
    assert progress.history_assessed == 1 and progress.history_required == 2


def test_safe_label_failures_count_only_affected_unresolved_definitions():
    checkpoints = [
        {"failure_code": "read_timeout", "fallback_candidate_ids": ["a", "a", "retained"],
         "limitation": "PRIVATE remote detail"},
        {"failure_code": "read_timeout", "fallback_candidate_ids": ["b"]},
        {"failure_code": "budget_exceeded", "fallback_candidate_ids": ["c"]},
        {"failure_code": "untrusted private exception", "fallback_candidate_ids": []},
    ]
    messages = label_stage_limitations(checkpoints, {"a", "b", "c"})
    assert len(messages) == 2
    assert "время ожидания" in messages[0] and "Кандидатов: 2" in messages[0]
    assert "бюджета" in messages[1] and "Кандидатов: 1" in messages[1]
    assert "PRIVATE" not in str(messages) and "untrusted" not in str(messages)


def test_legacy_checkpoint_fallback_and_retained_definition_do_not_invent_error_types():
    messages = label_stage_limitations([{"label_status": "mixed_frozen_and_lexical", "candidates": [
        {"candidate_id": "frozen", "specificity": "specific_technology"},
        {"candidate_id": "lexical", "specificity": "uncertain"}]}], {"frozen", "lexical"})
    assert len(messages) == 1 and "Кандидатов: 1" in messages[0]
    assert "локальные названия" in messages[0]


def test_article_query_fallback_explains_the_missing_mechanism_name():
    messages = label_stage_limitations([{"failure_code": "mechanism_alias_required",
        "fallback_candidate_ids": ["paper"], "limitation": "private provider text"}], {"paper"})
    assert len(messages) == 1 and "название конкретного механизма" in messages[0]
    assert "не принято как имя технологии" in messages[0]
    assert "private" not in messages[0]


def test_scope_precheck_does_not_claim_the_ai_proposed_a_rejected_name():
    messages = label_stage_limitations([{"failure_code": "evidence_rejected",
        "fallback_candidate_ids": ["unanchored"], "receipt": None}], {"unanchored"})
    assert len(messages) == 1
    assert "Связь механизма с областью запроса не подтверждена" in messages[0]
    assert "Предложенное AI" not in messages[0]


def test_real_service_saves_unassessed_result_with_honest_completion_and_portable_reason(tmp_path, monkeypatch):
    runtime, calls, _providers = runtime_for(tmp_path, monkeypatch, source_documents())
    monkeypatch.setattr(runtime, "_discover", lambda _context, snapshot, _plan:
        {"candidates": [candidate(snapshot).model_dump(mode="json")], "review_queue": []})
    try:
        job = runtime.start(query_plan().original_query, query_plan().english_query)
        result, artifacts = wait_success(runtime, job)
        assert result.top_trend_ids == () and not artifacts
        assert result.cards[0].category == "unassessed_cluster"
        row = runtime.get(job)
        assert row["stage"] == "complete" and row["completed"] == row["total"] == 1
        assert "Научная оценка кандидатов не завершена" in row["message"]
        assert "Исторические оценки: 0 из 1" in row["message"]
        assert any("непроверенные локальные названия" in item for item in result.limitations)
        assert not any(request.max_results == 1 for request in calls)
        path = tmp_path / "partial.trendresult"
        runtime.export_result(job, str(path))
        with read_result_package(path) as package:
            assert package.result == result
            assert evaluation_progress(package.result.model_dump(mode="json")).state == "not_assessed"
    finally:
        runtime.close()


def test_real_service_completed_negative_assessment_remains_successful(tmp_path, monkeypatch):
    runtime, _calls, _providers = runtime_for(tmp_path, monkeypatch, source_documents((1, 1, 1, 2, 100, 3)))
    try:
        job = runtime.start(query_plan().original_query, query_plan().english_query)
        result, artifacts = wait_success(runtime, job)
        assert not result.top_trend_ids and artifacts
        progress = evaluation_progress(result.model_dump(mode="json"))
        assert progress.state == "assessed" and progress.pending == 0
        assert "Автоматическая проверка выполнена;" in runtime.get(job)["message"]
    finally:
        runtime.close()
