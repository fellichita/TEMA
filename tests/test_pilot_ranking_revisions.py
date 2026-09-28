"""Selection limits never erase scientific reviews or unexamined hypotheses."""

import pytest
from pydantic import ValidationError

from app.pilot.antecedents import collect_antecedents
from app.pilot.contracts import AnalysisResult
from app.pilot.evidence import admission_hash, build_passport
from app.pilot.export import export_result, read_result_package
from app.pilot.history import assess_snapshot
from app.pilot.review import ReviewRecord, apply_novelty_review
from app.pilot.service import PilotService, _rank_cards, _top_trend_ids
from app.runtime.credentials import CredentialStore
from tests.test_pilot_antecedents import Provider
from tests.test_pilot_evidence import Context, document
from tests.test_pilot_export import make_result
from tests.test_pilot_review import decision_for


def test_rare_queue_is_portable_and_requires_archived_membership(tmp_path):
    result, archive, artifacts = make_result(tmp_path)
    rare = result.cards[0].candidate.model_copy(update={"candidate_id": "rare-protein-mechanism"})
    result = AnalysisResult.model_validate(result.model_dump(mode="python") | {"candidate_queue": (rare,)})
    saved = export_result(tmp_path / "queue.trendresult", result, archive, artifacts)
    with read_result_package(saved.path) as package:
        assert package.result.candidate_queue == (rare,)
    unrelated = rare.model_copy(update={"discovery_study_ids": ("doi:10.9999/not-archived",)})
    with pytest.raises(ValidationError, match="discovery provenance"):
        AnalysisResult.model_validate(result.model_dump(mode="python") | {"candidate_queue": (unrelated,)})


def test_sixteenth_expert_review_preserves_all_passports_and_reselects_top(tmp_path, monkeypatch):
    # Synthetic catalog: identical source mechanisms with distinct IDs exercise
    # the review/transport boundary, not discovery precision or concept dedup.
    result, archive, _ = make_result(tmp_path / "profile", historical=True)
    historical = next(item for item in result.snapshots if item.purpose == "history")
    discovery = next(item for item in result.snapshots if item.purpose == "discovery")
    cards, artifacts, reviews = [], [], []
    snapshots = {item.snapshot_id: item for item in result.snapshots}
    credentials = CredentialStore()
    monkeypatch.setattr(credentials, "get", lambda _: None)
    prepared = None
    values = None
    for index in range(16):
        item = result.cards[0].candidate.model_copy(update={"candidate_id": f"catalog-{index:02}"})
        item = item.model_copy(update={"admission_rule_hash": admission_hash(item)})
        context = Context()
        passport = build_passport(item, discovery, archive, context)
        base_artifact, source_card, _ = assess_snapshot(item, result.query_plan, historical, archive,
                                                       context, passport=passport)
        bundle = collect_antecedents(item, result.query_plan, archive, credentials, Context(),
                                    provider_factory=lambda _: Provider((document(81, year=2010),)))
        decision = decision_for(bundle, source_card)
        if index == 15:
            cards.append(source_card)
            artifacts.append(base_artifact)
            prepared = {"kind": "antecedents", "bundle": bundle.model_dump(mode="json")}
            values = decision.model_dump(mode="json")
            continue
        expanded, novelty = apply_novelty_review(decision, bundle, source_card, archive)
        artifact, card, _ = assess_snapshot(item, result.query_plan, historical, archive, context,
                                          passport=expanded, verified_novelty=novelty, antecedents=bundle)
        assert card.category == "confirmed_trend"
        cards.append(card)
        artifacts.append(artifact)
        snapshots[bundle.snapshot.snapshot_id] = bundle.snapshot
        reviews.append(ReviewRecord(created_at=decision.reviewed_at, decision=decision, bundle=bundle,
            source_card=source_card, reviewed_card=card, artifact=artifact, historical_snapshot=historical))
    _rank_cards(cards, artifacts)
    original = AnalysisResult.model_validate(result.model_dump(mode="python") | {
        "cards": tuple(cards), "snapshots": tuple(snapshots.values()), "top_trend_ids": _top_trend_ids(cards)})
    runtime = PilotService(tmp_path / "profile", credentials)
    try:
        saved = runtime._library.save_result(original, tuple(artifacts), tuple(reviews))
        assert prepared is not None and values is not None
        prepared["source_run_id"] = saved["id"]
        monkeypatch.setattr(runtime.coordinator, "result", lambda _: prepared)
        updated = runtime.apply_review("prepared-expert-job", values)
        revised = AnalysisResult.model_validate(updated["payload"]["result"])
        assert len(revised.cards) == 16
        assert all(card.category == "confirmed_trend" for card in revised.cards)
        assert len(revised.top_trend_ids) == 15
        assert len(updated["payload"]["reviews"]) == len(updated["payload"]["assessments"]) == 16
        assert sum(card["category"] == "confirmed_trend" for card in runtime.result(saved["id"])["result"]["cards"]) == 15
        # Unknown IDs, duplicated selections and old-format overflow remain invalid.
        for changes in ({"top_trend_ids": ("absent",)}, {"top_trend_ids": ("catalog-00", "catalog-00")},
                        {"methodology_version": "3.0.0"}):
            with pytest.raises(ValidationError):
                AnalysisResult.model_validate(revised.model_dump(mode="python") | changes)
    finally:
        runtime.close()
