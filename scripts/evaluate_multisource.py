"""Offline, rights-aware evaluation of frozen multisource profiles.

This script is never imported by the app. Protocol fixtures test the evaluator
only; they are not human labels or empirical quality evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.runtime.backup import ArchiveError
from app.runtime.jobs import TaskFailure
from scripts.evaluate_pilot import EvaluationError, digest, read_json, write_json

ARMS = ("C", "B", "S", "S+Q", "S+A", "S+Q+A", "S+Q+A+F")
MODES = ("end-to-end", "fixed-pool")
CUTS = ("T0", "T1", "T2")
_AREA = re.compile(r"[a-z][a-z0-9_]{2,79}\Z")
_SUITE_FIELDS = {"schema_version", "protocol_version", "status", "application_must_not_load_this_file",
                 "study_design", "areas", "cuts", "arms", "modes", "top_k",
                 "random_control_families_per_area", "primary_review_task_cap",
                 "minimum_second_reviewer_fraction", "maximum_policy_variants", "policy_presets",
                 "rubric", "ranking_label", "longitudinal_label", "missing_labels_are_unknown",
                 "default_without_R1_gate"}
_RUBRIC = ("concept_specific", "scope_relevant", "factual_claims_supported",
           "not_merely_renamed", "next_step_useful")
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_FAMILY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}\Z")
_SOURCES = {"S": frozenset(), "S+Q": frozenset({"wordstat"}),
            "S+A": frozenset({"arxiv"}), "S+Q+A": frozenset({"wordstat", "arxiv"}),
            "S+Q+A+F": frozenset({"wordstat", "arxiv", "cordis", "investment_csv"})}


class ArmCapture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    arm: Literal["C", "B", "S", "S+Q", "S+A", "S+Q+A", "S+Q+A+F"]
    state: Literal["succeeded", "failed", "not_executed"]
    artifact: str | None = Field(default=None, max_length=240)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    root_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    input_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    reason: str | None = Field(default=None, max_length=300)
    spent_micro: int = Field(ge=0, strict=True)
    external_calls: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def consistent(self) -> "ArmCapture":
        if self.state == "succeeded":
            if (not self.artifact or not self.sha256 or not self.root_hash or not self.input_hash
                    or self.reason is not None or Path(self.artifact).is_absolute()
                    or Path(self.artifact).suffix not in {".trendsignals", ".trendresult"}):
                raise ValueError("Successful arm needs one relative, hashed frozen artifact")
            if (self.arm == "C") != (Path(self.artifact).suffix == ".trendresult"):
                raise ValueError("Scientific C and signal arms require their own package kinds")
        elif (any(value is not None for value in (self.artifact, self.sha256, self.root_hash, self.input_hash))
              or not self.reason or self.spent_micro or self.external_calls):
            raise ValueError("Failed/not-executed arms need a reason and no fabricated output or spend")
        return self


class CaseCapture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    area_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    scope_query: str = Field(min_length=2, max_length=500)
    scope_confirmed_by: str = Field(min_length=2, max_length=200)
    scope_frozen_at: datetime
    budget_plan_frozen_at: datetime
    budget_caps_micro: dict[str, int]
    family_map: dict[str, str]
    control_population: tuple[str, ...] = Field(default=(), max_length=1000)
    fixed_pool_families: tuple[str, ...] = Field(default=(), max_length=105)
    end_to_end: tuple[ArmCapture, ...] = Field(min_length=7, max_length=7)
    fixed_pool: tuple[ArmCapture, ...] = Field(min_length=7, max_length=7)

    @model_validator(mode="after")
    def consistent(self) -> "CaseCapture":
        if (set(self.budget_caps_micro) != set(ARMS)
                or any(type(value) is not int or value < 0 for value in self.budget_caps_micro.values())
                or any(tuple(item.arm for item in group) != ARMS for group in (self.end_to_end, self.fixed_pool))
                or any(item.spent_micro > self.budget_caps_micro[item.arm]
                       for group in (self.end_to_end, self.fixed_pool) for item in group)
                or any(item.spent_micro or item.external_calls for item in self.fixed_pool)):
            raise ValueError("All seven arms need predeclared, non-reallocated budgets")
        if (len(self.family_map) != len(set(self.family_map)) or any(not 1 <= len(key) <= 2048
                or not _FAMILY.fullmatch(value) for key, value in self.family_map.items())
                or len(self.control_population) != len(set(self.control_population))
                or len(self.fixed_pool_families) != len(set(self.fixed_pool_families))
                or not set(self.control_population).issubset(self.family_map.values())
                or not set(self.fixed_pool_families).issubset(self.family_map.values())):
            raise ValueError("Concept-family mappings or frozen pool are invalid")
        return self


class CutCapture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    suite_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    cut: Literal["T0", "T1", "T2"]
    collection_started_at: datetime
    knowledge_cutoff: datetime
    policy_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    policy_frozen_at: datetime | None = None
    previous_cutoff: datetime | None = None
    t1_review_completed_at: datetime | None = None
    t1_review_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    t1_review_file: str | None = Field(default=None, max_length=240)
    cases: tuple[CaseCapture, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def consistent(self) -> "CutCapture":
        if (self.knowledge_cutoff < self.collection_started_at
                or any(moment.utcoffset() != timedelta(0) for moment in (
                    self.collection_started_at, self.knowledge_cutoff,
                    *(value for value in (self.policy_frozen_at, self.previous_cutoff,
                    self.t1_review_completed_at) if value is not None)))
                or any(case.scope_frozen_at.utcoffset() != timedelta(0)
                       or case.budget_plan_frozen_at.utcoffset() != timedelta(0)
                       or case.scope_frozen_at >= self.collection_started_at
                       or case.budget_plan_frozen_at >= self.collection_started_at for case in self.cases)):
            raise ValueError("Frozen inputs need UTC timestamps before collection")
        if self.cut == "T0":
            if (self.previous_cutoff is not None or self.t1_review_completed_at is not None
                    or self.t1_review_digest is not None or self.t1_review_file is not None):
                raise ValueError("T0 cannot borrow future review or cutoff data")
        else:
            if (self.previous_cutoff is None or self.collection_started_at <= self.previous_cutoff
                    or self.knowledge_cutoff < self.previous_cutoff + timedelta(days=14)):
                raise ValueError("Validation/blind cuts need at least fourteen days")
        if self.cut == "T2" and (self.policy_frozen_at is None or self.t1_review_completed_at is None
                or self.t1_review_digest is None or self.t1_review_file is None
                or self.policy_frozen_at >= self.collection_started_at
                or self.t1_review_completed_at > self.policy_frozen_at
                or self.previous_cutoff is None or self.t1_review_completed_at <= self.previous_cutoff):
            raise ValueError("Blind T2 needs policy freeze after completed T1 review, before collection")
        if self.cut == "T1" and (self.t1_review_completed_at is not None
                or self.t1_review_digest is not None or self.t1_review_file is not None):
            raise ValueError("T1 cannot contain its own future review evidence")
        return self


def load_suite(path: Path) -> dict[str, Any]:
    """Validate a protocol without interpreting proposed directions as data."""
    suite = read_json(path)
    if not isinstance(suite, dict) or set(suite) != _SUITE_FIELDS:
        raise EvaluationError("Multisource protocol has unknown or missing fields")
    if (type(suite["schema_version"]) is not int or suite["schema_version"] != 1
            or suite["protocol_version"] != "multisource-evaluation/1.0.0"
            or suite["status"] != "protocol_prepared_real_inputs_and_review_pending"
            or suite["application_must_not_load_this_file"] is not True
            or suite["study_design"] != "prospective"
            or suite["arms"] != list(ARMS) or suite["modes"] != list(MODES)
            or type(suite["top_k"]) is not int or suite["top_k"] != 5
            or type(suite["random_control_families_per_area"]) is not int
            or suite["random_control_families_per_area"] != 5
            or type(suite["primary_review_task_cap"]) is not int or suite["primary_review_task_cap"] != 650
            or type(suite["minimum_second_reviewer_fraction"]) not in (int, float)
            or suite["minimum_second_reviewer_fraction"] != 0.25
            or type(suite["maximum_policy_variants"]) is not int or suite["maximum_policy_variants"] != 3
            or suite["rubric"] != list(_RUBRIC) or suite["ranking_label"] != "concept_useful_now"
            or suite["longitudinal_label"] != "independent_confirmation_180d"
            or suite["missing_labels_are_unknown"] is not True
            or suite["default_without_R1_gate"] != "scientific"):
        raise EvaluationError("Multisource protocol no longer matches the frozen evaluation design")
    areas = suite["areas"]
    if (not isinstance(areas, list) or len(areas) != 3
            or any(not isinstance(area, dict) or set(area) != {"area_id", "proposed_direction"}
                   or not isinstance(area["area_id"], str) or not _AREA.fullmatch(area["area_id"])
                   or not isinstance(area["proposed_direction"], str)
                   or not 10 <= len(area["proposed_direction"].strip()) <= 500 for area in areas)
            or len({area["area_id"] for area in areas}) != 3
            or len({area["proposed_direction"].casefold().strip() for area in areas}) != 3):
        raise EvaluationError("Three distinct proposed technology areas are required")
    cuts = suite["cuts"]
    if (not isinstance(cuts, list) or len(cuts) != 3
            or any(not isinstance(cut, dict) or set(cut) != {"id", "role", "minimum_days_after_previous"}
                   or cut["id"] != expected or cut["role"] != role
                   or type(cut["minimum_days_after_previous"]) is not int
                   or cut["minimum_days_after_previous"] != gap
                   for cut, expected, role, gap in zip(cuts, CUTS,
                        ("development", "validation", "blind"), (0, 14, 14), strict=True))):
        raise EvaluationError("T0/T1/T2 prospective ordering is required")
    presets = suite["policy_presets"]
    expected_presets = (("base", "0.25", 30, "0.6666666667"),
                        ("sensitive", "0.15", 15, "0.6666666667"),
                        ("conservative", "0.40", 50, "1"))
    if (not isinstance(presets, list) or len(presets) != 3
            or any(not isinstance(item, dict) or set(item) != {"id", "search_yoy_fraction",
                "minimum_recent_count", "persistence_fraction"}
                or (item["id"], item["search_yoy_fraction"], item["minimum_recent_count"],
                    item["persistence_fraction"]) != expected
                for item, expected in zip(presets, expected_presets, strict=True))):
        raise EvaluationError("Only three predeclared policy variants are allowed")
    return suite


def capture_path(suite_path: Path, cut: str, explicit: Path | None = None) -> Path:
    if cut not in CUTS:
        raise EvaluationError("Choose T0, T1 or T2")
    return explicit if explicit is not None else Path(__file__).resolve().parents[1] / "build" / "multisource" / "captures" / (cut + ".json")


def _contained_file(directory: Path, relative: str) -> Path:
    selected = directory / relative
    path = selected.resolve()
    if (Path(relative).is_absolute() or not path.is_relative_to(directory.resolve())
            or selected.is_symlink() or not path.is_file()):
        raise EvaluationError("Frozen evaluation file is absent or outside its capture directory")
    return path


def _sha256(path: Path) -> str:
    from app.pilot.reports import open_local_regular

    hasher = hashlib.sha256()
    with open_local_regular(path) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_capture(path: Path, suite: dict[str, Any], cut: str, *, allow_blind: bool = False) -> CutCapture:
    """Reject missing cases, future inputs, adaptive budgets and unsealed T2."""
    try:
        capture = CutCapture.model_validate(read_json(path))
    except (OSError, ValueError, TypeError):
        raise EvaluationError("Frozen cut capture is missing or invalid") from None
    if (capture.cut != cut or capture.suite_hash != digest(suite)
            or tuple(case.area_id for case in capture.cases) != tuple(area["area_id"] for area in suite["areas"])
            or capture.knowledge_cutoff > datetime.now(UTC)):
        raise EvaluationError("Captured cases, protocol hash or cutoff do not match this replay")
    if cut == "T2" and not allow_blind:
        raise EvaluationError("Blind T2 needs explicit --allow-blind")
    if cut != "T0":
        previous = "T0" if cut == "T1" else "T1"
        previous_path = path.with_name(previous + ".json")
        try:
            prior = CutCapture.model_validate(read_json(previous_path))
        except (OSError, ValueError, TypeError):
            raise EvaluationError("Previous frozen cut is unavailable") from None
        if (prior.cut != previous or prior.suite_hash != capture.suite_hash
                or prior.knowledge_cutoff != capture.previous_cutoff):
            raise EvaluationError("Previous frozen cutoff was changed or omitted")
    if cut == "T2":
        assert (capture.t1_review_file is not None and capture.t1_review_digest is not None
                and capture.t1_review_completed_at is not None and capture.policy_frozen_at is not None
                and capture.previous_cutoff is not None)
        review_path = _contained_file(path.parent, capture.t1_review_file)
        if _sha256(review_path) != capture.t1_review_digest:
            raise EvaluationError("T1 review evidence changed after policy selection")
        review = read_json(review_path)
        if not isinstance(review, dict) or review.get("suite_hash") != capture.suite_hash:
            raise EvaluationError("T1 human review is incomplete or not tied to this protocol")
        from scripts.multisource_review import verify_t1_completion

        completed_at = verify_t1_completion(review_path, review)
        if (completed_at != capture.t1_review_completed_at
                or completed_at > capture.policy_frozen_at
                or review.get("knowledge_cutoff") != capture.previous_cutoff.isoformat()):
            raise EvaluationError("T1 review ended after policy freeze or belongs to another cutoff")
    return capture


def _family(case: CaseCapture, source_id: str) -> str:
    family = case.family_map.get(source_id)
    if family is None:
        raise EvaluationError("A returned concept has no frozen family mapping")
    return family


def _ranked(case: CaseCapture, identities: list[tuple[str, str]]) -> list[dict[str, str]]:
    rows = [{"family_id": _family(case, identity), "payload_hash": payload_hash,
             "source_id": identity} for identity, payload_hash in identities[:5]]
    if len({row["family_id"] for row in rows}) != len(rows):
        raise EvaluationError("One concept family occupies multiple top-five slots")
    return rows


def _scientific_before_cutoff(result: Any, cutoff: datetime) -> None:
    """A linked scientific base obeys the same boundary as comparator C."""
    if (result.created_at > cutoff or result.query_plan.as_of > cutoff.date()
            or any(snapshot.created_at > cutoff or snapshot.as_of > cutoff.date()
                   or any(document.observed_at > cutoff
                          or document.publicly_available_at is not None
                          and document.publicly_available_at > cutoff.date()
                          or document.publication_year is not None
                          and document.publication_year > cutoff.year
                          for document in snapshot.documents) for snapshot in result.snapshots)
            or any(evidence.retrieved_at > cutoff for card in result.cards for evidence in card.evidence)):
        raise EvaluationError("Scientific result or linked evidence appeared after the frozen cutoff")


def _source_timestamps(package: Any, cutoff: datetime) -> set[str]:
    """Enforce an actual frozen knowledge boundary beyond structural CAS checks."""
    from app.pilot.contracts import AnalysisResult
    from app.pilot.multisource.contracts import (ArxivImportReceipt, ArxivVersion, CapitalEvent,
        CapitalImportReceipt, FundingMetric, QueryProfile, QueryTerm, SearchMetric, SearchObservation,
        SourceSnapshot, TechnologyAssociation,
        TechnologyConcept, WordstatImportReceipt)
    from app.runtime.jobs import TaskFailure

    profile, store = package.profile, package.store
    if profile.knowledge_cutoff > cutoff or profile.decision_at > cutoff:
        raise EvaluationError("Signal profile was collected after the frozen cutoff")
    if package.base_payload is not None:
        _scientific_before_cutoff(AnalysisResult.model_validate(package.base_payload["result"]), cutoff)
    query = store.get_object(profile.query_profile_hash, QueryProfile)
    if query.confirmed_at is None or query.confirmed_at > cutoff:
        raise EvaluationError("Query identity was confirmed after its cutoff")
    moments = [term.confirmed_at for term in query.terms if term.confirmed_at is not None]
    sources: set[str] = set()
    for object_hash in profile.concept_artifact_hashes:
        concept = store.get_object(object_hash, TechnologyConcept)
        if concept.confirmed_at is not None:
            moments.append(concept.confirmed_at)
        moments.extend(term.confirmed_at for term in concept.aliases if term.confirmed_at is not None)
    for object_hash in profile.association_artifact_hashes:
        item = store.get_object(object_hash, TechnologyAssociation)
        if item.reviewed_at is not None:
            moments.append(item.reviewed_at)
    for object_hash in profile.source_snapshot_hashes:
        snapshot = store.get_object(object_hash, SourceSnapshot)
        moments.extend((snapshot.observed_at, snapshot.available_at))
        if snapshot.source not in {"scientific", "user"}:
            sources.add(snapshot.source)
    for object_hash in profile.metric_artifact_hashes:
        metric: SearchMetric | FundingMetric
        try:
            metric = store.get_object(object_hash, SearchMetric)
        except TaskFailure:
            metric = store.get_object(object_hash, FundingMetric)
        moments.extend((metric.decision_at, metric.knowledge_cutoff))
    for object_hash in profile.import_receipt_hashes:
        receipt: WordstatImportReceipt | ArxivImportReceipt | CapitalImportReceipt
        for kind in (WordstatImportReceipt, ArxivImportReceipt, CapitalImportReceipt):
            try:
                receipt = store.get_object(object_hash, kind)
                break
            except TaskFailure:
                continue
        else:
            raise EvaluationError("Unknown source receipt in frozen profile")
        moments.append(receipt.completed_at)
        snapshot = store.get_object(receipt.snapshot_hash, SourceSnapshot)
        moments.extend((snapshot.observed_at, snapshot.available_at))
        sources.add(snapshot.source)
        if isinstance(receipt, WordstatImportReceipt):
            for ref in receipt.term_hashes:
                term = store.get_object(ref, QueryTerm)
                if term.confirmed_at is not None:
                    moments.append(term.confirmed_at)
            for ref in receipt.observation_hashes:
                observation = store.get_object(ref, SearchObservation)
                moments.extend((observation.available_at, observation.observed_at))
                if observation.period_start > cutoff.date():
                    raise EvaluationError("Search observation starts after its frozen cutoff")
        elif isinstance(receipt, ArxivImportReceipt):
            if receipt.as_of > cutoff.date():
                raise EvaluationError("arXiv import used a future as-of date")
            for ref in receipt.version_hashes:
                version = store.get_object(ref, ArxivVersion)
                moments.extend((version.updated_at, version.published_at))
                document = package.archive.get(version.revision_id)
                moments.append(document.fetched_at)
        else:
            for ref in receipt.event_hashes:
                event = store.get_object(ref, CapitalEvent)
                moments.extend((event.available_at, event.observed_at))
    if any(moment > cutoff for moment in moments):
        raise EvaluationError("Alias, source revision, event or import was available after its cutoff")
    return sources


def _date_key(value: date | None, *, precision: str = "day") -> tuple[int, int, int] | None:
    if value is None:
        return None
    return value.year, value.month, value.day if precision == "day" else 0


def _baseline_latest(package: Any, finding: Any, science_cards: dict[str, Any]) -> tuple[int, int, int] | None:
    from app.pilot.multisource.contracts import CapitalEvent, SearchObservation
    from app.runtime.jobs import TaskFailure

    latest: list[tuple[int, int, int]] = []
    for evidence_hash in finding.observation_hashes:
        try:
            observation = package.store.get_object(evidence_hash, SearchObservation)
            key = _date_key(observation.period_end, precision="month")
        except TaskFailure:
            try:
                event = package.store.get_object(evidence_hash, CapitalEvent)
                if event.event_date_precision == "day":
                    key = _date_key(event.agreement_at or event.announced_at or event.closed_at)
                elif event.event_date_precision == "month" and event.event_month is not None:
                    year, month = map(int, event.event_month.split("-"))
                    key = (year, month, 0)
                elif event.event_date_precision == "year" and event.event_year is not None:
                    key = (event.event_year, 0, 0)
                else:
                    key = None
            except TaskFailure:
                try:
                    document = package.archive.get(evidence_hash)
                    key = (_date_key(document.publication_date) if document.publication_date is not None else
                           (document.publication_year, 0, 0) if document.publication_year is not None else None)
                except TaskFailure:
                    key = None  # A description hash is not an independent dated observation.
        if key is not None:
            latest.append(key)
    if finding.scientific_candidate_id in science_cards:
        card = science_cards[finding.scientific_candidate_id]
        for evidence in card.evidence:
            document = package.archive.get(evidence.revision_id)
            key = (_date_key(document.publication_date) if document.publication_date is not None else
                   (document.publication_year, 0, 0) if document.publication_year is not None else None)
            if key is not None:
                latest.append(key)
    return max(latest) if latest else None


def _baseline_rank(package: Any, case: CaseCapture) -> tuple[list[dict[str, str]], set[str]]:
    from app.pilot.contracts import AnalysisResult
    from app.pilot.multisource.contracts import TechnologyConcept

    profile = package.profile
    concepts = {str(item.concept_id): item for item in (
        package.store.get_object(digest, TechnologyConcept) for digest in profile.concept_artifact_hashes)}
    science_cards: dict[str, Any] = {}
    if package.base_payload is not None:
        base = AnalysisResult.model_validate(package.base_payload["result"])
        science_cards = {card.candidate.candidate_id: card for card in base.cards}
    ranked = []
    all_families = set()
    seen_families: set[str] = set()
    for finding in profile.findings:
        identity = str(finding.concept_id)
        family = _family(case, identity)
        if family in seen_families:
            raise EvaluationError("One concept family is duplicated in a baseline profile")
        seen_families.add(family)
        if family not in case.fixed_pool_families:
            continue  # B is a ranking comparator on the frozen review pool.
        all_families.add(family)
        concept = concepts[identity]
        if concept.identity_status != "confirmed" or concept.confirmed_at is None:
            continue
        latest = _baseline_latest(package, finding, science_cards)
        if latest is not None:
            neutral = {"label": concept.label, "definition": concept.definition,
                       "latest_observation": latest, "aliases": [item.text for item in concept.aliases
                       if item.status == "confirmed"]}
            ranked.append((latest, identity, digest(neutral)))
    if all_families != set(case.fixed_pool_families):
        raise EvaluationError("Baseline B did not receive every frozen review-pool concept")
    ranked.sort(key=lambda item: (-item[0][0], -item[0][1], -item[0][2], _family(case, item[1])))
    return _ranked(case, [(identity, payload_hash) for _, identity, payload_hash in ranked]), all_families


def _arm_output(capture_path: Path, case: CaseCapture, row: ArmCapture, mode: str,
                cutoff: datetime, policy_hash: str) -> dict[str, Any]:
    from app.pilot.contracts import content_hash
    from app.pilot.export import read_result_package
    from app.pilot.multisource.contracts import QueryProfile, TechnologyConcept
    from app.pilot.multisource.export import read_signal_package
    from scripts.evaluate_pilot import shown_top_cards

    output: dict[str, Any] = {"arm": row.arm, "state": row.state,
                              "reported_spend_micro": row.spent_micro,
                              "reported_external_calls": row.external_calls,
                              "ranked": [], "all_family_ids": [], "observed_sources": []}
    if row.state != "succeeded":
        output["reason"] = row.reason
        return output
    assert row.artifact is not None and row.sha256 is not None and row.root_hash is not None and row.input_hash is not None
    artifact = _contained_file(capture_path.parent, row.artifact)
    if _sha256(artifact) != row.sha256:
        raise EvaluationError("Frozen artifact bytes changed after capture")
    output.update(artifact=row.artifact, artifact_sha256=row.sha256, root_hash=row.root_hash,
                  input_hash=row.input_hash)
    if row.arm == "C":
        with read_result_package(artifact) as package:
            result = package.result
            _scientific_before_cutoff(result, cutoff)
            if (content_hash(result) != row.root_hash or result.query_plan.plan_hash != row.input_hash
                    or " ".join(result.query_plan.original_query.casefold().split()) !=
                       " ".join(case.scope_query.casefold().split())):
                raise EvaluationError("Scientific comparator belongs to another scope, plan or cutoff")
            cards = shown_top_cards(result.model_dump(mode="json"))
            output["ranked"] = _ranked(case, [(card["candidate"]["candidate_id"], digest(card))
                                               for card in cards])
            all_families = [_family(case, card.candidate.candidate_id) for card in result.cards]
            if len(set(all_families)) != len(all_families):
                raise EvaluationError("One concept family is duplicated in a scientific result")
            output["all_family_ids"] = sorted(all_families)
            output["observed_sources"] = ["scientific"]
            return output
    with read_signal_package(artifact) as package:
        profile = package.profile
        if (package.profile_hash != row.root_hash or profile.query_profile_hash != row.input_hash
                or profile.policy_hash != policy_hash
                or profile.knowledge_cutoff > cutoff):
            raise EvaluationError("Signal root or input hash differs from the captured output")
        query = package.store.get_object(profile.query_profile_hash, QueryProfile)
        if " ".join(query.original_query.casefold().split()) != " ".join(case.scope_query.casefold().split()):
            raise EvaluationError("Signal profile belongs to a different frozen scope")
        actual_sources = _source_timestamps(package, cutoff)
        if row.arm != "B":
            allowed = _SOURCES[row.arm]
            if not actual_sources.issubset(allowed) or profile.base_result_hash is None:
                raise EvaluationError("An arm used a disabled source or lost its scientific baseline")
            if mode == "end-to-end":
                forbidden = {"wordstat", "arxiv", "cordis", "investment_csv"} - allowed
                terms = [*query.terms]
                for digest_ref in profile.concept_artifact_hashes:
                    terms.extend(package.store.get_object(digest_ref, TechnologyConcept).aliases)
                if any(term.origin in forbidden for term in terms):
                    raise EvaluationError("Discovery alias from a disabled channel leaked into an arm")
            by_id = {item.finding_id: item for item in profile.findings}
            identities = [(str(by_id[identifier].concept_id), digest(by_id[identifier].model_dump(mode="json")))
                          for identifier in profile.attention_ids]
            output["ranked"] = _ranked(case, identities)
            all_families = [_family(case, str(item.concept_id)) for item in profile.findings]
            if len(set(all_families)) != len(all_families):
                raise EvaluationError("One concept family is duplicated in a source profile")
            output["all_family_ids"] = sorted(all_families)
        else:
            output["ranked"], families = _baseline_rank(package, case)
            output["all_family_ids"] = sorted(families)
        if mode == "fixed-pool" and set(output["all_family_ids"]) != set(case.fixed_pool_families):
            raise EvaluationError("Fixed-pool arm did not receive the same frozen candidate universe")
        output["observed_sources"] = sorted(actual_sources | ({"scientific"} if profile.base_result_hash else set()))
        return output


def _case_outputs(capture_path: Path, case: CaseCapture, mode: str,
                  cutoff: datetime, policy_hash: str) -> list[dict[str, Any]]:
    group = case.end_to_end if mode == "end-to-end" else case.fixed_pool
    return [_arm_output(capture_path, case, row, mode, cutoff, policy_hash) for row in group]


def _pool(case: CaseCapture, outputs: list[dict[str, Any]], suite_hash: str, cut: str) -> list[str]:
    top = {item["family_id"] for output in outputs if output["arm"] != "B" for item in output["ranked"]}
    population = {family for output in outputs if output["arm"] != "B" for family in output["all_family_ids"]} - top
    if set(case.control_population) != population:
        raise EvaluationError("Random-control population omits or adds concept families")
    selected = sorted(population, key=lambda family: digest({"suite": suite_hash, "cut": cut,
                                                               "area": case.area_id, "family": family}))[:5]
    expected = top | set(selected)
    if set(case.fixed_pool_families) != expected:
        raise EvaluationError("Frozen review pool differs from TOP-5 union plus random controls")
    return sorted(expected)


def replay(suite_path: Path, capture_file: Path, cut: str, mode: str, output: Path, *,
           offline: bool, allow_blind: bool = False) -> Path:
    """Verify pre-captured results without opening a service, model or network."""
    if mode not in MODES or cut not in CUTS or offline is not True:
        raise EvaluationError("Replay requires one cut, one mode and --offline")
    suite = load_suite(suite_path)
    capture = load_capture(capture_file, suite, cut, allow_blind=allow_blind)
    rows: list[dict[str, Any]] = []
    try:
        for case in capture.cases:
            discovery = _case_outputs(capture_file, case, "end-to-end", capture.knowledge_cutoff,
                                      capture.policy_hash)
            pool = _pool(case, discovery, capture.suite_hash, cut)
            selected = (discovery if mode == "end-to-end" else
                        _case_outputs(capture_file, case, "fixed-pool", capture.knowledge_cutoff,
                                      capture.policy_hash))
            rows.append({"area_id": case.area_id, "scope_query": case.scope_query,
                         "pool_family_ids": pool, "outputs": selected})
    except (OSError, ValueError) as error:
        if isinstance(error, EvaluationError):
            raise
        raise EvaluationError("Frozen artifact failed its transitive verification") from None
    except RuntimeError:
        raise EvaluationError("Frozen artifact failed its transitive verification") from None
    incomplete = any(row["state"] != "succeeded" for case in rows for row in case["outputs"])
    manifest = {"schema_version": 1, "evaluator_version": "multisource-evaluation/1.0.0",
                "suite_hash": capture.suite_hash, "capture_sha256": _sha256(capture_file),
                "suite_path": str(suite_path.resolve()),
                "capture_path": str(capture_file.resolve()), "cut": cut, "mode": mode,
                "knowledge_cutoff": capture.knowledge_cutoff.isoformat(),
                "policy_hash": capture.policy_hash,
                "status": "replayed_partial_unlabeled" if incomplete else "replayed_unlabeled",
                "human_R1": "not_evaluated", "L2": "pending", "cases": rows}
    destination = output / "manifest.json"
    write_json(destination, manifest)
    return destination


def verified_manifest(path: Path, *, allow_blind: bool = False) -> dict[str, Any]:
    """Recompute the entire offline replay before trusting review/scoring input."""
    manifest = read_json(path)
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("evaluator_version") != "multisource-evaluation/1.0.0"
            or manifest.get("cut") not in CUTS or manifest.get("mode") not in MODES
            or not isinstance(manifest.get("suite_path"), str)
            or not isinstance(manifest.get("capture_path"), str)):
        raise EvaluationError("Review requires a complete multisource replay manifest")
    if manifest["cut"] == "T2" and not allow_blind:
        raise EvaluationError("Blind T2 review requires explicit --allow-blind")
    with tempfile.TemporaryDirectory(prefix="trendanalyser-evaluation-verify-") as temporary:
        reconstructed = replay(Path(manifest["suite_path"]), Path(manifest["capture_path"]),
                               manifest["cut"], manifest["mode"], Path(temporary),
                               offline=True, allow_blind=allow_blind)
        if read_json(reconstructed) != manifest:
            raise EvaluationError("Replay manifest was edited or its frozen inputs changed")
    return manifest


def inventory(suite_path: Path, *, capture: Path | None = None) -> dict[str, Any]:
    suite = load_suite(suite_path)
    if capture is not None:
        raise EvaluationError("Use a cut-specific capture with replay; inventory scans all three cuts")
    rows = []
    for cut in CUTS:
        path = capture_path(suite_path, cut)
        rows.append({"cut": cut, "state": "present_unverified" if path.is_file() else "missing",
                     "capture_path": str(path)})
    return {"suite_hash": digest(suite), "status": "inputs_pending" if any(row["state"] == "missing" for row in rows)
            else "captures_present_unverified", "cuts": rows, "human_R1": "not_evaluated", "L2": "pending"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("inventory", help="Report real frozen input availability without network")
    check.add_argument("--suite", type=Path, required=True)
    replay = commands.add_parser("replay", help="Verify frozen outputs and replay deterministic views offline")
    replay.add_argument("--suite", type=Path, required=True)
    replay.add_argument("--capture", type=Path)
    replay.add_argument("--cut", choices=CUTS, required=True)
    replay.add_argument("--mode", choices=MODES, required=True)
    replay.add_argument("--offline", action="store_true", required=True)
    replay.add_argument("--allow-blind", action="store_true")
    replay.add_argument("--output", type=Path, required=True)
    packets = commands.add_parser("review-packets", help="Prepare blinded packets from a verified replay")
    packets.add_argument("--manifest", type=Path, required=True)
    packets.add_argument("--output", type=Path, required=True)
    packets.add_argument("--allow-blind", action="store_true")
    score = commands.add_parser("score", help="Score independent human labels with unknown bounds")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--labels", type=Path, required=True)
    score.add_argument("--outcomes", type=Path)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--allow-blind", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inventory":
            print(json.dumps(inventory(args.suite), ensure_ascii=False, indent=2))
            return 0
        if args.command == "replay":
            path = capture_path(args.suite, args.cut, args.capture)
            created = replay(args.suite, path, args.cut, args.mode, args.output,
                             offline=args.offline, allow_blind=args.allow_blind)
            print(json.dumps({"status": read_json(created)["status"], "manifest": str(created)}, ensure_ascii=False))
            return 0
        if args.command == "review-packets":
            from scripts.multisource_review import review_packets

            created = review_packets(args.manifest, args.output, allow_blind=args.allow_blind)
            print(json.dumps({"status": "review_packets_prepared", "packets": str(created)}, ensure_ascii=False))
            return 0
        if args.command == "score":
            from scripts.multisource_review import score

            created = score(args.manifest, args.labels, args.output, outcomes=args.outcomes,
                            allow_blind=args.allow_blind)
            print(json.dumps({"status": read_json(created)["status"], "report": str(created)}, ensure_ascii=False))
            return 0
        raise EvaluationError("Unknown evaluator command")
    except (EvaluationError, ArchiveError, TaskFailure, OSError, ValueError) as error:
        print(json.dumps({"status": "not_evaluated", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
