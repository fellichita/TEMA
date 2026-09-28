"""A real reachable confirmed path requires explicit attributed source review."""

from datetime import datetime, UTC
from threading import Event

import pytest
from pydantic import ValidationError

from app.pilot.contracts import Evidence, TrendCard
from app.pilot.evidence import EvidenceError
from app.pilot.history import assess_snapshot
from app.pilot.methodology import rank_candidates
from app.pilot.review import FieldReview, ReviewDecision, apply_novelty_review, read_review, record_review, verify_review_record
from tests import test_pilot_history
from tests.test_pilot_antecedents import bundle_for
from tests.test_pilot_evidence import document, query_plan, snapshot

scenario = test_pilot_history.scenario


def decision_for(bundle, passport, **changes):
    current = next(item for item in passport.evidence if item.text_field == "abstract")
    earlier = tuple(item.evidence_id for item in bundle.evidence if item.text_field == "abstract")[:1]
    values = dict(reviewer_name="Test reviewer", reviewed_at=datetime.now(UTC), candidate_id=passport.candidate.candidate_id,
        admission_rule_hash=passport.candidate.admission_rule_hash, bundle_hash=bundle.bundle_hash,
        kind="new_combination", rationale="The reviewed experiments support a new combination within the stated research scope.",
        mechanism_comparison="The reviewer compared the archived membrane mechanisms and their reported experimental constraints.",
        terminology_review="Earlier terminology and the corresponding source extracts were reviewed explicitly.",
        current_evidence_ids=(current.evidence_id,), earlier_evidence_ids=earlier,
        earlier_analogues_checked=True, terminology_changes_checked=True)
    return ReviewDecision(**(values | changes))


def test_explicit_expert_review_and_full_reference_history_reach_confirmed_top(scenario, tmp_path):
    archive, _, _, historical, item, context, passport = scenario
    bundle, _ = bundle_for(scenario, (document(81, year=2010),))
    decision = decision_for(bundle, passport)
    expanded, novelty = apply_novelty_review(decision, bundle, passport, archive)
    artifact, card, _ = assess_snapshot(item, query_plan(), historical, archive, context,
        passport=expanded, verified_novelty=novelty, antecedents=bundle)
    assert artifact.assessment.growth_confirmed and card.category == "confirmed_trend"
    assert rank_candidates((artifact.assessment,)) == (artifact.assessment,)
    claim = next(claim for claim in card.claims if claim.role == "novelty")
    assert claim.grounding_method == "reviewed-novelty/manual-v1" and "Test reviewer" in claim.text
    record = record_review(tmp_path / "reviews", decision, bundle, source_card=passport, reviewed_card=card,
        artifact=artifact, historical_snapshot=historical, archive=archive)
    assert read_review(tmp_path / "reviews", record.review_id, archive=archive) == record
    verify_review_record(record, archive)


@pytest.mark.parametrize("kind", ["established", "renamed"])
def test_old_methods_with_growth_become_renewed_interest_not_confirmed_new_trends(scenario, kind):
    archive, _, _, historical, item, context, passport = scenario
    bundle, _ = bundle_for(scenario, (document(81, year=2010),))
    expanded, novelty = apply_novelty_review(decision_for(bundle, passport, kind=kind), bundle, passport, archive)
    artifact, card, _ = assess_snapshot(item, query_plan(), historical, archive, context,
        passport=expanded, verified_novelty=novelty, antecedents=bundle)
    assert card.category == "renewed_interest" and rank_candidates((artifact.assessment,)) == ()


def test_verified_old_method_without_growth_is_established_topic(scenario):
    archive, docs, _, _, item, context, passport = scenario
    declining = tuple(type(doc).model_validate(doc.model_dump() | {"publication_year": 2020 + (index % 3)})
                      for index, doc in enumerate(docs))
    historical = snapshot(declining, archive, purpose="history")
    bundle, _ = bundle_for(scenario, (document(81, year=2010),))
    expanded, novelty = apply_novelty_review(decision_for(bundle, passport, kind="established"), bundle, passport, archive)
    artifact, card, _ = assess_snapshot(item, query_plan(), historical, archive, context,
        passport=expanded, verified_novelty=novelty, antecedents=bundle)
    assert not artifact.assessment.growth_confirmed and card.category == "established_topic"


def test_incomplete_older_search_cannot_support_new_mechanism_even_with_reviewer_name(scenario):
    archive, _, _, _, _, _, passport = scenario
    bundle, _ = bundle_for(scenario, (document(81, year=2010),), exhausted=False)
    with pytest.raises(EvidenceError, match="Неполный поиск"):
        apply_novelty_review(decision_for(bundle, passport, kind="new_mechanism"), bundle, passport, archive)
    # A positive older source still supports an attributed established-method
    # decision; it does not rely on exhaustiveness or absence claims.
    _, novelty = apply_novelty_review(decision_for(bundle, passport, kind="established"), bundle, passport, archive)
    assert novelty.kind == "established"


def test_absence_of_matches_never_automatically_generates_a_decision(scenario):
    archive, _, _, _, _, _, passport = scenario
    bundle, _ = bundle_for(scenario)
    assert bundle.operational_status == "none_found_within_queries"
    with pytest.raises(EvidenceError, match="более раннего источника"):
        apply_novelty_review(decision_for(bundle, passport, kind="renamed"), bundle, passport, archive)
    with pytest.raises(EvidenceError, match="отдельной проверки"):
        apply_novelty_review(decision_for(bundle, passport, earlier_analogues_checked=False), bundle, passport, archive)


def test_review_requires_content_quote_and_explicit_comparison_of_found_analogues(scenario):
    archive, _, _, _, _, _, passport = scenario
    bundle, _ = bundle_for(scenario, (document(81, year=2010),))
    with pytest.raises(EvidenceError, match="хотя бы с одним"):
        apply_novelty_review(decision_for(bundle, passport, earlier_evidence_ids=()), bundle, passport, archive)
    title = next(item.evidence_id for item in passport.evidence if item.text_field == "title")
    with pytest.raises(EvidenceError, match="Название работы недостаточно"):
        apply_novelty_review(decision_for(bundle, passport, current_evidence_ids=(title,)), bundle, passport, archive)


def test_forged_quotes_missing_ids_and_changed_definition_are_rejected(scenario):
    archive, _, _, _, _, _, passport = scenario
    bundle, _ = bundle_for(scenario, (document(81, year=2010),))
    with pytest.raises(EvidenceError, match="отсутствует"):
        apply_novelty_review(decision_for(bundle, passport, current_evidence_ids=("invented-evidence",)), bundle, passport, archive)
    original = next(item for item in passport.evidence if item.text_field == "abstract")
    forged = Evidence.model_validate(original.model_dump() | {"quote": "X" * len(original.quote)})
    altered = TrendCard.model_validate(passport.model_dump() | {"evidence": tuple(
        forged if item.evidence_id == original.evidence_id else item for item in passport.evidence)})
    with pytest.raises(ValueError, match="does not match"):
        apply_novelty_review(decision_for(bundle, altered), bundle, altered, archive)
    with pytest.raises(EvidenceError, match="другому определению"):
        apply_novelty_review(decision_for(bundle, passport, admission_rule_hash="f" * 64), bundle, passport, archive)


def test_review_journal_detects_tampering_and_keeps_superseded_entries(scenario, tmp_path):
    archive, _, _, historical, item, context, passport = scenario
    bundle, _ = bundle_for(scenario, (document(81, year=2010),))

    def save(decision):
        expanded, novelty = apply_novelty_review(decision, bundle, passport, archive)
        artifact, card, _ = assess_snapshot(item, query_plan(), historical, archive, context,
            passport=expanded, verified_novelty=novelty, antecedents=bundle)
        return record_review(tmp_path, decision, bundle, source_card=passport, reviewed_card=card,
            artifact=artifact, historical_snapshot=historical, archive=archive)

    first = save(decision_for(bundle, passport))
    second = save(decision_for(bundle, passport, kind="renamed", supersedes_review_id=first.review_id))
    assert first.review_id != second.review_id and len(list(tmp_path.glob("*.json"))) == 2
    target = tmp_path / (first.review_id + ".json")
    target.write_bytes(target.read_bytes().replace(b"Test reviewer", b"Fake reviewer"))
    with pytest.raises(EvidenceError, match="повреждена"):
        read_review(tmp_path, first.review_id, archive=archive)


def test_empty_rationale_and_string_booleans_cannot_be_submitted(scenario):
    _, _, _, _, _, _, passport = scenario
    bundle, _ = bundle_for(scenario)
    with pytest.raises(ValidationError):
        decision_for(bundle, passport, earlier_analogues_checked="true")
    with pytest.raises(ValidationError):
        decision_for(bundle, passport, mechanism_comparison="")


def test_expert_verification_obeys_cancellation_and_default_deadline(scenario, monkeypatch):
    from app.pilot import review
    from app.runtime.jobs import TaskCancelled
    from tests.test_pilot_evidence import Context

    archive, _, _, _, _, _, passport = scenario
    bundle, _ = bundle_for(scenario, (document(81, year=2010),))
    decision = decision_for(bundle, passport)
    context = Context()
    context.cancel_event.set()
    with pytest.raises(TaskCancelled):
        apply_novelty_review(decision, bundle, passport, archive, context=context)
    times = iter((0.0, 121.0))
    monkeypatch.setattr(review.time, "monotonic", lambda: next(times))
    with pytest.raises(EvidenceError, match="превысила"):
        apply_novelty_review(decision, bundle, passport, archive)


@pytest.mark.parametrize("cancel_at", ["entry", "publication"])
def test_cancelled_review_never_publishes_journal_entry(scenario, tmp_path, monkeypatch, cancel_at):
    from app.pilot import review
    from app.runtime.jobs import TaskCancelled

    archive, _, _, historical, item, context, passport = scenario
    bundle, _ = bundle_for(scenario, (document(81, year=2010),))
    decision = decision_for(bundle, passport)
    expanded, novelty = apply_novelty_review(decision, bundle, passport, archive)
    artifact, card, _ = assess_snapshot(item, query_plan(), historical, archive, context,
        passport=expanded, verified_novelty=novelty, antecedents=bundle)
    cancel = Event()
    if cancel_at == "entry":
        cancel.set()
    else:
        original_fsync = review.os.fsync

        def fsync_then_cancel(descriptor):
            original_fsync(descriptor)
            cancel.set()

        monkeypatch.setattr(review.os, "fsync", fsync_then_cancel)
    directory = tmp_path / "new-journal"
    with pytest.raises(TaskCancelled):
        record_review(directory, decision, bundle, source_card=passport, reviewed_card=card,
            artifact=artifact, historical_snapshot=historical, archive=archive, cancel=cancel)
    assert not list(directory.glob("*.json")) and not list(directory.glob("*.tmp"))


def test_superseded_review_uses_the_same_cancel_context_before_publication(scenario, tmp_path, monkeypatch):
    from app.pilot import review
    from app.runtime.jobs import TaskCancelled

    archive, _, _, historical, item, context, passport = scenario
    bundle, _ = bundle_for(scenario, (document(81, year=2010),))
    first_decision = decision_for(bundle, passport)
    expanded, novelty = apply_novelty_review(first_decision, bundle, passport, archive)
    artifact, card, _ = assess_snapshot(item, query_plan(), historical, archive, context,
        passport=expanded, verified_novelty=novelty, antecedents=bundle)
    first = record_review(tmp_path / "reviews", first_decision, bundle, source_card=passport,
        reviewed_card=card, artifact=artifact, historical_snapshot=historical, archive=archive)
    decision = decision_for(bundle, passport, supersedes_review_id=first.review_id)
    expanded, novelty = apply_novelty_review(decision, bundle, passport, archive)
    artifact, card, _ = assess_snapshot(item, query_plan(), historical, archive, context,
        passport=expanded, verified_novelty=novelty, antecedents=bundle)
    cancel = Event()
    original_read = review.read_review
    shared = []

    def cancel_before_older_read(*args, **kwargs):
        shared.append(kwargs["context"])
        cancel.set()
        return original_read(*args, **kwargs)

    monkeypatch.setattr(review, "read_review", cancel_before_older_read)
    with pytest.raises(TaskCancelled):
        record_review(tmp_path / "reviews", decision, bundle, source_card=passport,
            reviewed_card=card, artifact=artifact, historical_snapshot=historical, archive=archive, cancel=cancel)
    assert len(shared) == 1
    assert [path.stem for path in (tmp_path / "reviews").glob("*.json")] == [first.review_id]


def field_review_for(passport, role="advantage", **changes):
    source = next(item for item in passport.evidence if item.text_field == "abstract")
    values = dict(role=role, source_evidence_id=source.evidence_id, text_field="abstract",
        quote="The lithium selective membrane reduces energy consumption in laboratory experiments.",
        verdict="supported", rationale="I checked the full source context: the stated result belongs to the candidate membrane experiment.",
        context_checked=True)
    return FieldReview(**(values | changes))


def test_review_preserves_old_decision_serialization_when_no_field_verdicts(scenario):
    _, _, _, _, _, _, passport = scenario
    bundle, _ = bundle_for(scenario, (document(81, year=2010),))
    decision = decision_for(bundle, passport)
    assert "field_reviews" not in decision.model_dump(mode="json")
    assert ReviewDecision.model_validate(decision.model_dump(mode="json")).decision_hash == decision.decision_hash


def test_manual_context_review_resolves_missing_fields_and_replays_journal(scenario, tmp_path):
    archive, _, _, historical, item, context, passport = scenario
    passport = passport.model_copy(update={"claims": tuple(claim for claim in passport.claims
                                            if claim.role not in {"advantage", "application"})})
    bundle, _ = bundle_for(scenario, (document(81, year=2010),))
    decision = decision_for(bundle, passport, field_reviews=(field_review_for(passport),
        field_review_for(passport, "application", application_kind="research")))
    expanded, novelty = apply_novelty_review(decision, bundle, passport, archive)
    assert expanded.candidate == passport.candidate
    assert len([claim for claim in expanded.claims if claim.role == "application"]) == 1
    manual = next(claim for claim in expanded.claims if claim.role == "advantage")
    assert manual.support == "supported" and manual.grounding_method == "reviewed-evidence/manual-v1"
    assert decision.reviewer_name in manual.text
    artifact, card, _ = assess_snapshot(item, query_plan(), historical, archive, context,
        passport=expanded, verified_novelty=novelty, antecedents=bundle)
    assert artifact.inputs.application.kind == "research"
    assert card.category == "confirmed_trend"
    record = record_review(tmp_path, decision, bundle, source_card=passport, reviewed_card=card,
        artifact=artifact, historical_snapshot=historical, archive=archive)
    assert read_review(tmp_path, record.review_id, archive=archive) == record
    verify_review_record(record, archive)
    altered = record.model_copy(update={"decision": decision.model_copy(update={"field_reviews": (
        decision.field_reviews[0].model_copy(update={"rationale": "A substituted expert rationale that was never part of the signed local decision."}),)})})
    with pytest.raises(EvidenceError, match="не воспроизводится"):
        verify_review_record(altered, archive)


def test_expert_rejection_replaces_old_supported_role_and_blocks_confirmation(scenario):
    archive, _, _, historical, item, context, passport = scenario
    bundle, _ = bundle_for(scenario, (document(81, year=2010),))
    decision = decision_for(bundle, passport, field_reviews=(field_review_for(passport, verdict="contradicted"),))
    expanded, novelty = apply_novelty_review(decision, bundle, passport, archive)
    assert [claim.support for claim in expanded.claims if claim.role == "advantage"] == ["contradicted"]
    artifact, card, _ = assess_snapshot(item, query_plan(), historical, archive, context,
        passport=expanded, verified_novelty=novelty, antecedents=bundle)
    assert card.category != "confirmed_trend" and not artifact.assessment.growth_confirmed


@pytest.mark.parametrize("changes", [dict(context_checked=False), dict(context_checked="true"),
    dict(rationale=" " * 40), dict(text_field="title"), dict(application_kind="deployment")])
def test_field_review_requires_explicit_context_valid_role_and_rationale(scenario, changes):
    *_, passport = scenario
    with pytest.raises(ValidationError):
        field_review_for(passport, **changes)


def test_field_review_rejects_fabricated_quotes_foreign_sources_duplicate_roles_and_legacy(scenario):
    archive, _, _, _, _, _, passport = scenario
    bundle, _ = bundle_for(scenario, (document(81, year=2010),))
    for changes in (dict(quote="Invented technology advantages that do not occur in the archived source text."),
                    dict(source_evidence_id=bundle.evidence[0].evidence_id)):
        decision = decision_for(bundle, passport, field_reviews=(field_review_for(passport, **changes),))
        with pytest.raises(EvidenceError):
            apply_novelty_review(decision, bundle, passport, archive)
    field = field_review_for(passport)
    with pytest.raises(ValidationError, match="одного поля"):
        decision_for(bundle, passport, field_reviews=(field, field))
    with pytest.raises(EvidenceError, match="3.1.0"):
        apply_novelty_review(decision_for(bundle, passport, field_reviews=(field,)), bundle, passport, archive,
                             methodology_version="3.0.0")
