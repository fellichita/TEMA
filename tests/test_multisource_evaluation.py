"""Evaluator fixtures test safeguards; they are never human quality labels."""

import json
import hashlib
import csv
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from scripts.evaluate_multisource import (ARMS, CUTS, MODES, _ranked, _scientific_before_cutoff, build_parser, inventory,
                                          load_capture, load_suite, replay)
from scripts.evaluate_pilot import EvaluationError
from scripts.evaluate_pilot import digest
from scripts.multisource_review import review_packets, score, verify_t1_completion


SUITE_PATH = Path(__file__).parent / "evaluation" / "multisource-v1.json"


def _write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def test_protocol_is_separate_from_app_and_has_no_real_outcomes() -> None:
    suite = load_suite(SUITE_PATH)
    assert suite["application_must_not_load_this_file"] is True
    assert tuple(suite["arms"]) == ARMS and tuple(suite["modes"]) == MODES
    assert tuple(item["id"] for item in suite["cuts"]) == CUTS
    assert "labels" not in suite and "captures" not in suite
    report = inventory(SUITE_PATH)
    assert report["human_R1"] == "not_evaluated" and report["L2"] == "pending"
    assert report["status"] == "inputs_pending"
    assert all(row["state"] == "missing" for row in report["cuts"])


@pytest.mark.parametrize("change", [
    lambda suite: suite.update(top_k=0),
    lambda suite: suite["arms"].remove("S+A"),
    lambda suite: suite["cuts"][2].update(minimum_days_after_previous=0),
    lambda suite: suite["policy_presets"].append(deepcopy(suite["policy_presets"][0])),
    lambda suite: suite["areas"][1].update(area_id=suite["areas"][0]["area_id"]),
    lambda suite: suite.update(missing_labels_are_unknown=False),
    lambda suite: suite.update(application_must_not_load_this_file=False),
    lambda suite: suite.update(outcomes=[{"quality": 1}]),
])
def test_protocol_cannot_silently_change_denominator_or_become_a_fake_result(tmp_path: Path, change) -> None:
    suite = deepcopy(load_suite(SUITE_PATH))
    change(suite)
    with pytest.raises(EvaluationError):
        load_suite(_write(tmp_path, suite))


def test_replay_parser_requires_cut_mode_and_offline_flag() -> None:
    parser = build_parser()
    required = ("replay", "--suite", str(SUITE_PATH), "--cut", "T0", "--mode", "end-to-end",
                "--offline", "--output", "/tmp/multisource-replay")
    args = parser.parse_args(required)
    assert args.cut == "T0" and args.mode == "end-to-end" and args.offline
    for missing in ("--cut", "--mode", "--offline", "--output"):
        start = list(required)
        index = start.index(missing)
        del start[index:index + (1 if missing == "--offline" else 2)]
        with pytest.raises(SystemExit):
            parser.parse_args(start)


def _capture(suite: dict, *, cutoff: datetime | None = None) -> dict:
    from app.pilot.multisource.contracts import load_policy

    ending = cutoff or datetime.now(UTC)
    absent = [{"arm": arm, "state": "not_executed", "reason": "real input unavailable",
               "spent_micro": 0, "external_calls": 0} for arm in ARMS]
    cases = [{"area_id": area["area_id"], "scope_query": area["proposed_direction"],
              "scope_confirmed_by": "test-reviewer", "scope_frozen_at": (ending - timedelta(days=2)).isoformat(),
              "budget_plan_frozen_at": (ending - timedelta(days=2)).isoformat(),
              "budget_caps_micro": {arm: 0 for arm in ARMS}, "family_map": {},
              "control_population": [], "fixed_pool_families": [],
              "end_to_end": deepcopy(absent), "fixed_pool": deepcopy(absent)} for area in suite["areas"]]
    return {"schema_version": 1, "suite_hash": digest(suite), "cut": "T0",
            "collection_started_at": (ending - timedelta(days=1)).isoformat(),
            "knowledge_cutoff": ending.isoformat(), "policy_hash": load_policy()[1],
            "cases": cases}


def test_offline_replay_keeps_all_missing_arms_and_three_cases_in_denominator(tmp_path: Path, monkeypatch) -> None:
    import socket

    suite = load_suite(SUITE_PATH)
    capture_file = _write(tmp_path, _capture(suite))
    monkeypatch.setattr(socket.socket, "connect", lambda *_: pytest.fail("network access during replay"))
    path = replay(SUITE_PATH, capture_file, "T0", "end-to-end", tmp_path / "out", offline=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "replayed_partial_unlabeled"
    assert manifest["human_R1"] == "not_evaluated" and manifest["L2"] == "pending"
    assert len(manifest["cases"]) == 3
    assert all(len(case["outputs"]) == 7 and all(item["state"] == "not_executed"
               for item in case["outputs"]) for case in manifest["cases"])


def test_missing_case_adaptive_budget_and_future_cutoff_fail_closed(tmp_path: Path) -> None:
    suite = load_suite(SUITE_PATH)
    capture = _capture(suite)
    capture["cases"].pop()
    with pytest.raises(EvaluationError):
        load_capture(_write(tmp_path, capture), suite, "T0")
    capture = _capture(suite)
    capture["cases"][0]["budget_caps_micro"]["S+Q"] = -1
    with pytest.raises(EvaluationError):
        load_capture(_write(tmp_path, capture), suite, "T0")
    capture = _capture(suite)
    capture["cases"][0]["end_to_end"][3]["spent_micro"] = 1
    with pytest.raises(EvaluationError):
        load_capture(_write(tmp_path, capture), suite, "T0")
    capture = _capture(suite, cutoff=datetime.now(UTC) + timedelta(days=1))
    with pytest.raises(EvaluationError):
        load_capture(_write(tmp_path, capture), suite, "T0")


def test_shared_family_cannot_take_two_top_slots() -> None:
    from scripts.evaluate_multisource import CaseCapture

    suite = load_suite(SUITE_PATH)
    case = CaseCapture.model_validate(_capture(suite)["cases"][0] |
                                      {"family_map": {"concept-a": "same", "concept-b": "same"}})
    with pytest.raises(EvaluationError, match="multiple top-five"):
        _ranked(case, [("concept-a", "a" * 64), ("concept-b", "b" * 64)])


def test_linked_scientific_base_cannot_hide_a_future_snapshot(tmp_path: Path) -> None:
    from tests.test_pilot_export import make_result

    result, _, _ = make_result(tmp_path, historical=True)
    cutoff = result.created_at
    _scientific_before_cutoff(result, cutoff)
    future = result.snapshots[0].model_copy(update={"created_at": cutoff + timedelta(seconds=1)})
    impossible = result.model_copy(update={"snapshots": (future, *result.snapshots[1:])})
    with pytest.raises(EvaluationError, match="linked evidence"):
        _scientific_before_cutoff(impossible, cutoff)
    observed = result.snapshots[0].documents[0].model_copy(
        update={"observed_at": cutoff + timedelta(seconds=1)})
    future_document = result.snapshots[0].model_copy(update={"documents": (
        observed, *result.snapshots[0].documents[1:])})
    impossible_document = result.model_copy(update={"snapshots": (future_document, *result.snapshots[1:])})
    with pytest.raises(EvaluationError, match="linked evidence"):
        _scientific_before_cutoff(impossible_document, cutoff)


def test_blind_replay_needs_completed_t1_review_and_prior_cut(tmp_path: Path) -> None:
    suite = load_suite(SUITE_PATH)
    capture = _capture(suite)
    capture["cut"] = "T2"
    capture["previous_cutoff"] = (datetime.now(UTC) - timedelta(days=14)).isoformat()
    capture["t1_review_completed_at"] = (datetime.now(UTC) - timedelta(days=16)).isoformat()
    capture["policy_frozen_at"] = (datetime.now(UTC) - timedelta(days=15)).isoformat()
    capture["t1_review_file"] = "t1-review.json"
    capture["t1_review_digest"] = "a" * 64
    capture["collection_started_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    capture["cases"] = [{**case, "scope_frozen_at": (datetime.now(UTC) - timedelta(days=2)).isoformat(),
                          "budget_plan_frozen_at": (datetime.now(UTC) - timedelta(days=2)).isoformat()}
                         for case in capture["cases"]]
    with pytest.raises(EvaluationError):
        load_capture(_write(tmp_path, capture), suite, "T2", allow_blind=False)
    with pytest.raises(EvaluationError):
        load_capture(_write(tmp_path, capture), suite, "T2", allow_blind=True)


def test_replay_verifies_a_real_portable_root_and_rejects_policy_mismatch(tmp_path: Path) -> None:
    from tests.test_multisource_export import _published

    service, run, _ = _published(tmp_path, share=True)
    try:
        package = tmp_path / "captured.trendsignals"
        service.export_signal(run, str(package))
        view = service.signal_result(run)
        profile = view["profile"]
        profile_hash = view["profile_hash"]
    finally:
        service.close()
    suite = load_suite(SUITE_PATH)
    capture = _capture(suite)
    case = capture["cases"][2]
    case["scope_query"] = "Молекулярная память"
    concept_id = profile["findings"][0]["concept_id"]
    case["family_map"] = {concept_id: "dna-memory"}
    case["end_to_end"][1] = {"arm": "B", "state": "succeeded", "artifact": package.name,
                              "sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
                              "root_hash": profile_hash,
                              "input_hash": profile["query_profile_hash"],
                              "spent_micro": 0, "external_calls": 0}
    capture_file = _write(tmp_path, capture)
    result = replay(SUITE_PATH, capture_file, "T0", "end-to-end", tmp_path / "replay", offline=True)
    outputs = json.loads(result.read_text(encoding="utf-8"))["cases"][2]["outputs"]
    assert outputs[1]["root_hash"] == profile_hash
    assert outputs[1]["ranked"] == []  # B cannot add a family outside the S/C review pool.
    capture["policy_hash"] = "a" * 64
    with pytest.raises(EvaluationError, match="Signal root"):
        replay(SUITE_PATH, _write(tmp_path, capture), "T0", "end-to-end", tmp_path / "bad", offline=True)


def _real_b_replay(tmp_path: Path) -> Path:
    from tests.test_multisource_export import _published

    service, run, _ = _published(tmp_path, share=True)
    try:
        package = tmp_path / "captured.trendsignals"
        service.export_signal(run, str(package))
        view = service.signal_result(run)
    finally:
        service.close()
    suite = load_suite(SUITE_PATH)
    capture = _capture(suite)
    case = capture["cases"][2]
    case["scope_query"] = "Молекулярная память"
    identity = view["profile"]["findings"][0]["concept_id"]
    case["family_map"] = {identity: "dna-memory"}
    case["end_to_end"][1] = {"arm": "B", "state": "succeeded", "artifact": package.name,
                              "sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
                              "root_hash": view["profile_hash"],
                              "input_hash": view["profile"]["query_profile_hash"],
                              "spent_micro": 0, "external_calls": 0}
    return replay(SUITE_PATH, _write(tmp_path, capture), "T0", "end-to-end",
                  tmp_path / "replay", offline=True)


def _real_s_replay(tmp_path: Path) -> Path:
    from app.pilot.export import export_result
    from app.pilot.multisource.contracts import TechnologyConcept
    from app.pilot.multisource.export import read_signal_package
    from app.pilot.multisource.queries import build_manual_profile
    from app.pilot.multisource.store import SignalStore
    from app.pilot.multisource.wordstat import DynamicsMapping, import_wordstat_csv
    from app.pilot.service import PilotService
    from app.runtime.credentials import CredentialStore
    from tests.test_pilot_export import make_result

    data_dir = tmp_path / "app"
    scientific, archive, assessments = make_result(data_dir, historical=True)
    result_file = export_result(tmp_path / "science.trendresult", scientific, archive, assessments).path
    service = PilotService(data_dir, CredentialStore())
    try:
        science_id = service.import_result(str(result_file))["id"]
        card = scientific.cards[0]
        query = build_manual_profile(card.candidate.label, card.candidate.definition,
                                     seed_terms=(card.candidate.label,), primary_phrase=card.candidate.label,
                                     confirmed_at=datetime.now(UTC))
        store = SignalStore(data_dir)
        query_hash = store.put_object(query)
        concept = TechnologyConcept(concept_id=uuid4(), label=card.candidate.label,
                                    definition=card.candidate.definition, identity_status="confirmed",
                                    confirmed_at=datetime.now(UTC), provenance_hashes=(query_hash,))
        concept_hash = store.put_object(concept)
        rows = ["Месяц;Запросов;Доля"]
        for year in (2025, 2026):
            for month in range(1, 13):
                if year == 2026 and month > 8:
                    break
                increased = year == 2026 and month in (6, 7, 8)
                rows.append(f"{month:02d}.{year};{60 if increased else 30};"
                            f"{'0,20%' if increased else '0,10%'}")
        source = tmp_path / "wordstat.csv"
        source.write_text("\n".join(rows) + "\n", encoding="utf-8")
        mapping = DynamicsMapping(date_column="Месяц", count_column="Запросов", share_column="Доля",
                                  date_format="MM.YYYY", share_unit="percent", phrase=card.candidate.label,
                                  expected_from=date(2025, 1, 1), expected_to=date(2026, 8, 1))
        receipt = import_wordstat_csv(store, source, query_hash, kind="dynamics", mapping=mapping,
                                      encoding="utf-8-sig", delimiter=";", retention="local_allowed",
                                      export_right="share_allowed", license_ref="generated test data")
        run = service.start_signals(query_hash, (concept_hash,), base_result_run_id=science_id,
                                    wordstat_receipt_hash=receipt,
                                    scientific_links=({"concept_id": str(concept.concept_id),
                                                       "candidate_id": card.candidate.candidate_id},))
        service.coordinator.wait()
        assert service.get(run)["state"] == "succeeded"
        package = tmp_path / "captured.trendsignals"
        service.export_signal(run, str(package))
        view = service.signal_result(run)
    finally:
        service.close()
    with read_signal_package(package) as portable:
        selected = bool(portable.profile.attention_ids)
    suite = load_suite(SUITE_PATH)
    capture = _capture(suite)
    case = capture["cases"][2]
    case["scope_query"] = card.candidate.label
    identity = str(concept.concept_id)
    case["family_map"] = {identity: "dna-memory"}
    case["control_population"] = [] if selected else ["dna-memory"]
    case["fixed_pool_families"] = ["dna-memory"]
    for index, arm in ((1, "B"), (3, "S+Q")):
        case["end_to_end"][index] = {"arm": arm, "state": "succeeded", "artifact": package.name,
                                       "sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
                                       "root_hash": view["profile_hash"],
                                       "input_hash": view["profile"]["query_profile_hash"],
                                       "spent_micro": 0, "external_calls": 0}
    return replay(SUITE_PATH, _write(tmp_path, capture), "T0", "end-to-end",
                  tmp_path / "replay", offline=True)


def _fill_labels(path: Path, *, first: str = "yes", second: str = "yes",
                 same_reviewer: bool = False, adjudicate: bool = False) -> None:
    packet = json.loads((path.parent / "packets.json").read_text(encoding="utf-8"))
    task_kind = {item["task_id"]: item["kind"] for item in packet["tasks"]}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    assert fields is not None
    for row in rows:
        row.update({"reviewer_id": "analyst-1" if row["phase"] == "primary" or same_reviewer else "analyst-2",
                    "factual_claims_supported": "yes", "next_step_useful": "yes",
                    "explanation": "Independent source review", "refs": "sha256:verified-source",
                    "label_available_at": datetime.now(UTC).isoformat()})
        if task_kind[row["task_id"]] == "concept":
            row.update({"concept_specific": "yes", "scope_relevant": "yes",
                        "not_merely_renamed": "yes",
                        "concept_useful_now": first if row["phase"] == "primary" else second})
        else:
            row.update({"payload_supported": "yes", "critical_unsupported": "no"})
    if adjudicate:
        concept_row = next(row for row in rows if task_kind[row["task_id"]] == "concept"
                           and row["phase"] == "primary")
        rows.append({**concept_row, "phase": "adjudication", "reviewer_id": "analyst-3",
                     "concept_useful_now": "yes"})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_review_packets_and_score_require_two_real_independent_reviews(tmp_path: Path) -> None:
    manifest = _real_s_replay(tmp_path)
    packet_file = review_packets(manifest, tmp_path / "review")
    packet = json.loads(packet_file.read_text(encoding="utf-8"))
    assert {item["kind"] for item in packet["tasks"]} == {"concept", "payload"}
    assert all("arm" not in item and "rank" not in item for item in packet["tasks"])
    labels = packet_file.parent / "labels.csv"
    pending = json.loads(score(manifest, labels, tmp_path / "pending.json").read_text(encoding="utf-8"))
    assert pending["status"] == "not_evaluated"
    assert pending["per_area"][2]["arms"][1]["upper_bound"] == .2
    assert pending["L2"]["status"] == "censored"
    _fill_labels(labels, same_reviewer=True)
    with pytest.raises(EvaluationError, match="distinct people"):
        score(manifest, labels, tmp_path / "same-person.json")
    _fill_labels(labels, first="yes", second="no")
    disputed = json.loads(score(manifest, labels, tmp_path / "disputed.json").read_text(encoding="utf-8"))
    assert disputed["status"] == "review_incomplete"
    assert disputed["per_area"][2]["arms"][1]["upper_bound"] == .2
    _fill_labels(labels, first="unknown", second="unknown")
    unknown = json.loads(score(manifest, labels, tmp_path / "unknown.json").read_text(encoding="utf-8"))
    assert unknown["status"] == "review_incomplete"
    assert unknown["per_area"][2]["arms"][1]["useful_slots_at_5"] is None
    _fill_labels(labels, first="yes", second="no", adjudicate=True)
    resolved = json.loads(score(manifest, labels, tmp_path / "resolved.json").read_text(encoding="utf-8"))
    assert resolved["status"] == "review_complete"
    assert resolved["per_area"][2]["arms"][1]["useful_slots_at_5"] == .2
    assert resolved["default_screen_change_eligible"] is False
    packet["tasks"][0]["material"] = {"invented": True}
    packet_file.write_text(json.dumps(packet), encoding="utf-8")
    with pytest.raises(EvaluationError, match="edited"):
        score(manifest, labels, tmp_path / "altered-packet.json")


def test_review_packets_reject_edited_replay_and_empty_labels_never_pass(tmp_path: Path) -> None:
    suite = load_suite(SUITE_PATH)
    capture = _write(tmp_path, _capture(suite))
    manifest = replay(SUITE_PATH, capture, "T0", "end-to-end", tmp_path / "replay", offline=True)
    packet = review_packets(manifest, tmp_path / "review")
    report = json.loads(score(manifest, packet.parent / "labels.csv", tmp_path / "quality.json")
                        .read_text(encoding="utf-8"))
    assert report["status"] == "not_evaluated"
    assert all(row["lower_bound"] == 0 and row["upper_bound"] == 1 for row in report["macro"])
    edited = json.loads(manifest.read_text(encoding="utf-8"))
    edited["cases"][0]["outputs"].pop()
    manifest.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(EvaluationError, match="edited"):
        review_packets(manifest, tmp_path / "tampered")


def test_t1_completion_replays_its_human_csv_before_policy_freeze(tmp_path: Path) -> None:
    _real_s_replay(tmp_path)
    suite = load_suite(SUITE_PATH)
    t1 = json.loads((tmp_path / "suite.json").read_text(encoding="utf-8"))
    t0_cutoff = datetime.fromisoformat(t1["knowledge_cutoff"]) - timedelta(days=15)
    (tmp_path / "T0.json").write_text(json.dumps(_capture(suite, cutoff=t0_cutoff)), encoding="utf-8")
    t1["cut"] = "T1"
    t1["previous_cutoff"] = t0_cutoff.isoformat()
    (tmp_path / "T1.json").write_text(json.dumps(t1), encoding="utf-8")
    manifest = replay(SUITE_PATH, tmp_path / "T1.json", "T1", "end-to-end",
                      tmp_path / "t1", offline=True)
    packets = review_packets(manifest, tmp_path / "t1" / "review")
    labels = packets.parent / "labels.csv"
    _fill_labels(labels)
    report_path = score(manifest, labels, tmp_path / "t1" / "quality.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "review_complete"
    assert verify_t1_completion(report_path, report) == datetime.fromisoformat(report["review_completed_at"])
    labels.write_text(labels.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="changed"):
        verify_t1_completion(report_path, report)
