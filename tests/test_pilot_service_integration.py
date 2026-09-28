"""Independent integration regressions for restored budgets and optional sources."""

from datetime import UTC, date, datetime
import os
from threading import Event, Thread
from types import SimpleNamespace

import httpx
import pytest

from app.pilot.contracts import CorpusSnapshot, Coverage, content_hash
from app.pilot.service import PilotService
from app.pilot.settings import PilotSettings
from app.runtime.budget import BudgetLimits, BudgetService, RequestAllowance
from app.runtime.budget_admin import BudgetAdmin
from app.runtime.credentials import CredentialStore
from app.runtime.jobs import TaskCancelled, TaskFailure
from tests.test_pilot_evidence import candidate, document
from tests.test_pilot_history import frozen


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    credentials = CredentialStore()
    monkeypatch.setattr(credentials, "get", lambda _: None)
    monkeypatch.setattr("app.pilot.encoder.verify_artifacts", lambda _, **kwargs: None)
    service = PilotService(tmp_path / "data", credentials)
    try:
        yield service
    finally:
        service.close()


def restore_conservative_day(pilot):
    # These runs are about the paid ledger, so the profile is put on the paid
    # provider first: the default is the local model, and switching providers is
    # itself a settings change that reconfigures the day budget.
    pilot.configure(PilotSettings(provider="deepseek").model_dump(mode="json"))
    scope = "day/deepseek/" + datetime.now(UTC).date().isoformat()

    def seed():
        budget = BudgetService(pilot.coordinator._connection)
        budget.create_scope(scope, BudgetLimits(96, 800_000, 120_000, PilotSettings().day_cost_micro), currency="USD")
        budget.reserve("uncertain-charge", (scope,), RequestAllowance(200, 30, 20_000))
        budget.mark_sent("uncertain-charge")
        budget.mark_restored()

    pilot.coordinator._executor.submit(seed).result()
    BudgetAdmin(pilot.coordinator).acknowledge_restore(scope, confirmed=True)
    return scope


def test_new_analysis_cannot_reset_zero_headroom_after_backup_reconciliation(pilot, monkeypatch):
    scope = restore_conservative_day(pilot)

    def inspect_before_sources(plan, context, *_):
        snapshot = BudgetService(context.connection).snapshot(scope)
        assert snapshot.limits.cost_micro == snapshot.used.cost_micro == 20_000
        assert snapshot.remaining.cost_micro == 0
        raise TaskFailure("Controlled unavailable source after budget inspection")

    monkeypatch.setattr("app.pilot.service.collect_snapshot", inspect_before_sources)
    run = pilot.start("lithium selective membranes", "lithium selective membranes")
    pilot.coordinator.wait()
    assert "Controlled unavailable" in pilot.get(run)["error"]
    status = BudgetAdmin(pilot.coordinator).status()
    day = next(item for item in status["scopes"] if item["scope_id"] == scope)
    assert day["remaining"]["cost_micro"] == 0


def test_saving_unchanged_daily_preference_cannot_reset_reconciled_limit(pilot):
    scope = restore_conservative_day(pilot)
    # These runs exercise the paid ledger, so the provider is explicit.
    values = PilotSettings(provider="deepseek", history_enabled=False,
                           patents_enabled=True).model_dump(mode="json")
    pilot.configure(values)
    day = next(item for item in BudgetAdmin(pilot.coordinator).status()["scopes"] if item["scope_id"] == scope)
    assert day["limits"]["cost_micro"] == 20_000 and day["remaining"]["cost_micro"] == 0
    values["day_cost_micro"] = 3_000_000
    pilot.configure(values)
    day = next(item for item in BudgetAdmin(pilot.coordinator).status()["scopes"] if item["scope_id"] == scope)
    assert day["limits"]["cost_micro"] == 3_000_000
    assert day["used"]["cost_micro"] == 20_000


def test_restored_failed_run_resumes_using_explicitly_reconciled_run_scope(pilot, monkeypatch):
    attempts = []

    def collect(plan, *_):
        attempts.append(plan.plan_hash)
        if len(attempts) == 1:
            raise TaskFailure("First source attempt intentionally unavailable")
        return CorpusSnapshot(snapshot_id="resumed-empty", plan_hash=plan.plan_hash, purpose="discovery",
            as_of=plan.as_of, created_at=datetime.now(UTC), documents=(),
            coverage=(Coverage(source="openalex", purpose="discovery", query_hash=content_hash(plan.queries[0]),
                state="unavailable", requested_years=plan.completed_years, pagination_exhausted=False,
                comparable=False, reasons=("source_unavailable",)),),
            normalizer_version="integration-fixture", deduplication_version="doi-source-id-v1")

    monkeypatch.setattr("app.pilot.service.collect_snapshot", collect)
    monkeypatch.setattr(pilot, "_discover", lambda *_: dict(candidates=[], input_records=0, unique_studies=0,
        retained_studies=0, quality="insufficient_data", limitations=["No source records"]))
    run = pilot.start("lithium selective membranes", "lithium selective membranes")
    pilot.coordinator.wait()
    assert pilot.get(run)["state"] == "failed"
    pilot.coordinator._executor.submit(lambda: BudgetService(pilot.coordinator._connection).mark_restored()).result()
    admin = BudgetAdmin(pilot.coordinator)
    for scope in admin.status()["scopes"]:
        admin.acknowledge_restore(scope["scope_id"], additional_allowance=BudgetLimits(1, 100, 30, 1_000), confirmed=True)
    pilot.resume(run)
    pilot.coordinator.wait()
    assert pilot.get(run)["state"] == "succeeded", pilot.get(run)["error"]
    assert len(attempts) == 2
    run_budget = next(item for item in admin.status()["scopes"] if item["scope_id"] == "run/" + run)
    assert run_budget["limits"]["cost_micro"] == 1_000 and run_budget["limits"]["calls"] == 1


@pytest.fixture
def paid_workflow(pilot, monkeypatch):
    """Real coordinator, LLM adapter and ledger; only HTTP and the clock are fake."""
    # A profile defaults to the local model, which has no ledger of money to
    # exercise: this whole fixture is about the paid provider.
    pilot.configure(PilotSettings(provider="deepseek").model_dump(mode="json"))
    from app.pilot.llm import LlmClient
    from app.pilot.query import plan_query
    from app.sqlite_runtime import sqlite3
    from tests.test_pilot_llm import Answer, response_payload

    class Clock:
        current = datetime(2026, 9, 11, 23, 59, 59, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            return cls.current.astimezone(tz)

    state = SimpleNamespace(clock=Clock, sent=[], budget=None, database_path=None, action=None)
    monkeypatch.setattr("app.pilot.service.datetime", Clock)
    monkeypatch.setattr(pilot.credentials, "get", lambda name: "synthetic-test-key" if name == "deepseek_api_key" else None)

    def respond(request):
        # Scope allocation must already be durable when network dispatch begins.
        # HTTP's worker owns a separate read connection. Production ledger
        # access stays on the coordinator; check_same_thread is never disabled.
        connection = sqlite3.connect(state.database_path)
        try:
            pending = connection.execute("SELECT request_id FROM pilot_budget_requests WHERE state='sent'").fetchall()
            assert len(pending) == 1
            allocated = tuple(row[0] for row in connection.execute(
                "SELECT scope_id FROM pilot_budget_allocations WHERE request_id=? ORDER BY scope_id", (pending[0][0],)))
        finally:
            connection.close()
        state.sent.append((pending[0][0], allocated))
        return httpx.Response(200, json=response_payload())

    http = httpx.Client(transport=httpx.MockTransport(respond))

    def client_factory(config, credentials, budget, **kwargs):
        state.budget = budget
        state.database_path = budget.connection.execute("PRAGMA database_list").fetchone()[2]
        return LlmClient(config, credentials, budget, http_client=http, **kwargs)

    def pay(client, scopes, request_id, cancel):
        return client.generate_json(Answer, system_prompt="Return a short fact.", user_content="Synthetic input",
            prompt_version="boundary-test/1", request_id=request_id, scope_ids=scopes, cancel=cancel)

    state.pay = pay

    def paid_plan(query, client, **kwargs):
        state.action(client, kwargs["scope_ids"], kwargs["request_id"], kwargs["cancel"])
        return plan_query(query, None, **kwargs)

    def collect(plan, *_):
        return CorpusSnapshot(snapshot_id="boundary-empty", plan_hash=plan.plan_hash, purpose="discovery",
            as_of=plan.as_of, created_at=datetime.now(UTC), documents=(),
            coverage=(Coverage(source="openalex", purpose="discovery", query_hash=content_hash(plan.queries[0]),
                state="unavailable", requested_years=plan.completed_years, pagination_exhausted=False,
                comparable=False, reasons=("source_unavailable",)),),
            normalizer_version="boundary-fixture", deduplication_version="doi-source-id-v1")

    monkeypatch.setattr("app.pilot.service.LlmClient", client_factory)
    monkeypatch.setattr("app.pilot.service.plan_query", paid_plan)
    monkeypatch.setattr("app.pilot.service.collect_snapshot", collect)
    monkeypatch.setattr(pilot, "_discover", lambda *_: dict(candidates=[], input_records=0, unique_studies=0,
        retained_studies=0, quality="insufficient_data", limitations=[]))
    try:
        yield state
    finally:
        http.close()


def test_paid_calls_crossing_utc_midnight_use_their_own_day_and_one_run_cap(pilot, paid_workflow):
    state = paid_workflow

    def two_days(client, scopes, request_id, cancel):
        state.pay(client, scopes, request_id + "/first", cancel)
        state.clock.current = datetime(2026, 9, 12, 0, 0, 1, tzinfo=UTC)
        state.pay(client, scopes, request_id + "/second", cancel)

    state.action = two_days
    run = pilot.start("lithium selective membranes", "lithium selective membranes")
    pilot.coordinator.wait()
    assert pilot.get(run)["state"] == "succeeded", pilot.get(run)["error"]
    assert len(state.sent) == 2
    assert state.sent[0][1] == ("day/deepseek/2026-09-11", "run/" + run)
    assert state.sent[1][1] == ("day/deepseek/2026-09-12", "run/" + run)
    scopes = {item["scope_id"]: item for item in BudgetAdmin(pilot.coordinator).status()["scopes"]}
    assert scopes["day/deepseek/2026-09-11"]["used"]["calls"] == 1
    assert scopes["day/deepseek/2026-09-12"]["used"]["calls"] == 1
    assert scopes["run/" + run]["used"]["calls"] == 2


@pytest.mark.parametrize("barrier", ["reduced_day", "run_cap", "currency"])
def test_midnight_cannot_reset_existing_day_run_or_currency_fences(pilot, paid_workflow, barrier):
    state = paid_workflow

    def capped(client, scopes, request_id, cancel):
        from app.pilot.service import _configure_day_budget

        state.pay(client, scopes, request_id + "/first", cancel)
        state.clock.current = datetime(2026, 9, 12, 0, 0, 1, tzinfo=UTC)
        if barrier == "run_cap":
            state.budget.update_limits(scopes[0], BudgetLimits(1, 200_000, 30_000, 500_000))
        else:
            state.budget.create_scope("day/deepseek/2026-09-12", BudgetLimits(96, 800_000, 120_000, 2_000_000),
                                      currency="RUB" if barrier == "currency" else "USD")
            if barrier == "reduced_day":
                _configure_day_budget(state.budget.connection, "deepseek", "USD", 0)
        state.pay(client, scopes, request_id + "/blocked", cancel)

    state.action = capped
    run = pilot.start("lithium selective membranes", "lithium selective membranes")
    pilot.coordinator.wait()
    row = pilot.get(run)
    assert row["state"] == "failed"
    assert ("Валюта" if barrier == "currency" else "Лимит") in row["error"]
    assert len(state.sent) == 1  # No retry and no second network dispatch.
    scopes = {item["scope_id"]: item for item in BudgetAdmin(pilot.coordinator).status()["scopes"]}
    assert scopes["run/" + run]["used"]["calls"] == 1
    assert scopes["day/deepseek/2026-09-12"]["used"]["calls"] == 0
    if barrier == "reduced_day":
        assert scopes["day/deepseek/2026-09-12"]["limits"]["cost_micro"] == 0
    if barrier == "currency":
        assert scopes["day/deepseek/2026-09-12"]["currency"] == "RUB"


def test_explicit_next_day_resume_preserves_run_usage_and_selects_current_day(pilot, paid_workflow):
    state = paid_workflow

    def first_attempt(client, scopes, request_id, cancel):
        state.pay(client, scopes, request_id, cancel)
        raise TaskFailure("Controlled interruption after a paid response")

    state.action = first_attempt
    run = pilot.start("lithium selective membranes", "lithium selective membranes")
    pilot.coordinator.wait()
    assert pilot.get(run)["state"] == "failed"
    state.clock.current = datetime(2026, 9, 12, 0, 0, 1, tzinfo=UTC)
    state.action = state.pay
    pilot.resume(run)
    pilot.coordinator.wait()
    assert pilot.get(run)["state"] == "succeeded", pilot.get(run)["error"]
    assert len(state.sent) == 2 and state.sent[0][0] != state.sent[1][0]
    assert state.sent[1][1] == ("day/deepseek/2026-09-12", "run/" + run)
    scopes = {item["scope_id"]: item for item in BudgetAdmin(pilot.coordinator).status()["scopes"]}
    assert scopes["run/" + run]["used"]["calls"] == 2
    assert scopes["day/deepseek/2026-09-11"]["used"]["calls"] == 1
    assert scopes["day/deepseek/2026-09-12"]["used"]["calls"] == 1


def test_disposable_discovery_cache_schema_damage_is_rebuilt_without_poisoning_query(pilot, monkeypatch):
    from app.pilot.service import WORKFLOW_VERSION

    settings = PilotSettings(provider="deepseek")
    query = "lithium selective membranes"
    payload = dict(query=query, english_query=query, english_source="user",
                   as_of=date.today().isoformat(), settings=settings.model_dump(mode="json"))
    key = content_hash({"workflow": WORKFLOW_VERSION, "request": payload})
    pilot._cache_write(key, "discovery", {"invalid_snapshot_schema": True})
    collected = []

    def collect(plan, *_):
        collected.append(plan.plan_hash)
        return CorpusSnapshot(snapshot_id="fresh-empty-snapshot", plan_hash=plan.plan_hash, purpose="discovery",
            as_of=plan.as_of, created_at=datetime.now(UTC), documents=(),
            coverage=(Coverage(source="openalex", purpose="discovery", query_hash=content_hash(plan.queries[0]),
                state="unavailable", requested_years=plan.completed_years, pagination_exhausted=False,
                comparable=False, reasons=("source_unavailable",)),),
            normalizer_version="integration-fixture", deduplication_version="doi-source-id-v1")

    monkeypatch.setattr("app.pilot.service.collect_snapshot", collect)
    monkeypatch.setattr(pilot, "_discover", lambda *_: dict(candidates=[], input_records=0, unique_studies=0,
        retained_studies=0, quality="insufficient_data", limitations=["No source records"]))
    run = pilot.start(query, query)
    pilot.coordinator.wait()
    assert pilot.get(run)["state"] == "succeeded", pilot.get(run)["error"]
    assert len(collected) == 1
    assert pilot.result(run)["result"]["quality"] == "insufficient_data"


def test_optional_patent_source_without_credentials_does_not_crash_completed_analysis(pilot, monkeypatch):
    from app.pilot.encoder import EncoderError

    def unavailable_encoder(*_args, **_kwargs):
        # This test isolates optional patent collection; its temporary profile
        # has no model files for the separate publication relevance score.
        raise EncoderError("Scientific encoder unavailable in this fixture")

    monkeypatch.setattr("app.pilot.encoder.MultilingualEncoder", unavailable_encoder)

    def collect(plan, context, archive, credentials):
        reference = archive.put(document(31, year=2025))
        return CorpusSnapshot(snapshot_id="observed-documents", plan_hash=plan.plan_hash, purpose="discovery",
            as_of=plan.as_of, created_at=datetime.now(UTC), documents=(reference,),
            coverage=(Coverage(source="openalex", purpose="discovery", query_hash=content_hash(plan.queries[0]),
                state="complete", requested_years=plan.completed_years, completed_years=plan.completed_years,
                pagination_exhausted=True, comparable=False, scanned_records=1, accepted_records=1),),
            normalizer_version="integration-fixture", deduplication_version="doi-source-id-v1")

    def discover(context, snapshot, plan):
        selected = frozen(candidate(snapshot))
        return dict(candidates=[selected.model_dump(mode="json")], input_records=1, unique_studies=1,
                    retained_studies=1, quality="partial", limitations=["One source document"])

    pilot.configure(PilotSettings(provider="deepseek", history_enabled=False, patents_enabled=True,
                                  external_ai_allowed=False).model_dump(mode="json"))
    monkeypatch.setattr("app.pilot.service.collect_snapshot", collect)
    monkeypatch.setattr(pilot, "_discover", discover)
    # Isolate optional-source orchestration with a prevalidated specific candidate;
    # local labels deliberately remain uncertain without an external label review.
    monkeypatch.setattr("app.pilot.evidence.label_candidates", lambda candidates, *args, **kwargs: candidates)
    run = pilot.start("lithium selective membranes", "lithium selective membranes")
    pilot.coordinator.wait()
    assert pilot.get(run)["state"] == "succeeded", pilot.get(run)["error"]
    saved = pilot.coordinator.checkpoint_value(run, "patents_0")
    assert saved["coverage"]["state"] == "unavailable"
    assert "missing_epo_credentials" in saved["coverage"]["reasons"]
    payload = pilot.result(run)
    assert payload["result"]["cards"][0]["category"] == "weak_signal_candidate"
    assert payload["assessments"][0]["assessment"]["growth_confirmed"] is False
    assert payload["assessments"][0]["inputs"]["history"]["coverage"]["state"] == "unavailable"


def test_service_keeps_persistence_failure_visible_without_followup_database_read(pilot, monkeypatch):
    def fail_storage(context, _):
        context.connection.execute("PRAGMA query_only=ON")
        return {}

    monkeypatch.setattr(pilot.coordinator, "_processor", fail_storage)
    run = pilot.coordinator.submit({})
    pilot.coordinator.wait()

    def no_read(*_):
        pytest.fail("An unconfirmed failure must not issue another DB read for optional clarification")

    monkeypatch.setattr(pilot.coordinator, "checkpoint_value", no_read)
    row = pilot.get(run)
    assert row["state"] == "failed" and row["persistence_error"]
    assert row["clarification"] is None


def test_arxiv_import_cancelled_during_archiving_does_not_publish_catalogue(pilot, tmp_path, monkeypatch):
    from app.pilot.supplemental import SupplementalLibrary
    from tests.test_pilot_enrichment import atom_entry, write_atom

    path = write_atom(tmp_path, [atom_entry(), atom_entry(identity="2501.12346v1")])
    cancel = Event()
    put = pilot.archive.put

    def cancel_after_one_revision(*args, **kwargs):
        reference = put(*args, **kwargs)
        cancel.set()
        return reference

    monkeypatch.setattr(pilot.archive, "put", cancel_after_one_revision)
    library = SupplementalLibrary(pilot.data_dir, pilot.archive)
    with pytest.raises(TaskCancelled):
        library.import_arxiv(path, cancel=cancel)
    assert not list(library.directory.glob("*.json"))


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Native Unix pipe boundary")
def test_local_result_fifo_is_rejected_without_blocking_ui_executor(tmp_path):
    from app.pilot.library import read_artifact

    fifo = tmp_path / ("a" * 64 + ".json")
    os.mkfifo(fifo)
    outcomes = []

    def read():
        try:
            read_artifact(tmp_path, "a" * 64)
        except (TaskFailure, OSError) as error:
            outcomes.append(error)

    thread = Thread(target=read, daemon=True)
    thread.start()
    thread.join(0.25)
    blocked = thread.is_alive()
    if blocked:
        # Unblock the old implementation so a failing regression leaves no thread.
        descriptor = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        try:
            os.write(descriptor, b"{}")
        finally:
            os.close(descriptor)
        thread.join(1)
    assert not thread.is_alive()
    assert not blocked, "Opening a replaced catalogue FIFO blocks polling and shutdown indefinitely"
    assert len(outcomes) == 1


def test_queued_auxiliary_action_does_not_start_after_controller_close(tmp_path):
    from concurrent.futures import CancelledError
    from app.backend.config import BackendSettings
    from app.backend.service import Backend
    from app.ui.controller import Controller
    from tests.test_pilot_enrichment import atom_entry, write_atom

    class Scheduler:
        def after(self, *_):
            return None

    controller = Controller(Scheduler(), factory=lambda: Backend(BackendSettings(data_dir=tmp_path / "data")))
    path = write_atom(tmp_path, [atom_entry()])
    controller._invoke("open", (), {})
    controller.closing = True
    try:
        with pytest.raises((CancelledError, TaskCancelled, TaskFailure)):
            controller._invoke("pilot_import_arxiv", (path,), {})
        assert not list((tmp_path / "data" / "supplemental").glob("*.json"))
    finally:
        controller._close_backend()
        controller.executor.shutdown(wait=True)


@pytest.mark.parametrize("same_timestamp", [False, True], ids=["distinct-times", "equal-times"])
def test_imported_result_pagination_parses_only_selected_recent_page(pilot, tmp_path, monkeypatch, same_timestamp):
    from app.pilot import library as module
    from app.pilot.library import ResultLibrary
    from tests.test_pilot_export import make_result

    result, archive, artifacts = make_result(tmp_path / "source")
    library = ResultLibrary(tmp_path / "source", archive)
    identifiers = []
    for number in range(5):
        output = result.model_copy(update={"result_id": "pagination-" + str(number)})
        saved = library.save_result(output, artifacts)
        identifier = saved["id"]
        path = library.directory / (identifier.removeprefix("import-") + ".json")
        # NTFS file times have coarser precision than one nanosecond. Exercise
        # both distinct import times and the library's deterministic tie order.
        stamp = 1_700_000_000_000_000_000 + (0 if same_timestamp else number * 1_000_000_000)
        os.utime(path, ns=(stamp, stamp))
        identifiers.append(identifier)
    read = module.read_artifact
    opened = []

    def tracked(*args):
        opened.append(args[1])
        return read(*args)

    monkeypatch.setattr(module, "read_artifact", tracked)
    assert library.count() == 5 and opened == []
    expected = sorted(identifiers, reverse=True) if same_timestamp else list(reversed(identifiers))
    assert [row["id"] for row in library.list_rows(offset=1, limit=2)] == expected[1:3]
    assert len(opened) == 2
    assert library.list_rows(offset=50) == [] and len(opened) == 2


def test_service_reuses_imported_verification_between_result_and_document_pages(pilot, tmp_path, monkeypatch):
    from app.pilot import library as module
    from app.pilot.export import export_result
    from tests.test_pilot_export import make_result

    result, archive, assessments = make_result(tmp_path / "source", historical=True)
    package = export_result(tmp_path / "saved.zip", result, archive, assessments)
    run_id = pilot.import_result(str(package.path))["id"]
    original_verify = module.verify_result
    replays = []

    def tracked(*args, **kwargs):
        replays.append(args[0].result_id)
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(module, "verify_result", tracked)
    pilot.result(run_id)
    assert pilot.documents(run_id, 0, 2)["total"] == 4
    assert len(pilot.documents(run_id, 2, 2)["items"]) == 2
    assert len(replays) == 1


@pytest.mark.parametrize("state", ["succeeded", "failed"])
def test_review_lookup_reuses_or_resumes_match_older_than_hundred_recent_runs(pilot, tmp_path, monkeypatch, state):
    import json

    from app.pilot.export import export_result
    from tests.test_pilot_export import make_result

    result, archive, assessments = make_result(tmp_path / "source", historical=True)
    package = export_result(tmp_path / "saved.zip", result, archive, assessments)
    source_id = pilot.import_result(str(package.path))["id"]
    candidate_id = result.cards[0].candidate.candidate_id
    monkeypatch.setattr(pilot, "_run_antecedents", lambda *_: {"kind": "controlled-review"})
    review_id = pilot.begin_review(source_id, candidate_id)
    pilot.coordinator.wait()

    def seed_later_runs():
        connection = pilot.coordinator._connection
        connection.execute("UPDATE analysis_runs SET state=? WHERE id=?", (state, review_id))
        for index in range(110):
            connection.execute("INSERT INTO analysis_runs(id,attempt,state,input_json,created_at,updated_at) "
                "VALUES(?,1,'succeeded',?,'2099-01-01','2099-01-01')",
                ("later-" + str(index), json.dumps({"payload": {"query": "unrelated"}})))

    pilot.coordinator._executor.submit(seed_later_runs).result()
    assert review_id not in {row["id"] for row in pilot.coordinator.list_runs()}
    assert pilot.begin_review(source_id, candidate_id) == review_id
    pilot.coordinator.wait()
    assert pilot.coordinator.get(review_id)["attempt"] == (2 if state == "failed" else 1)


def test_cancelled_model_preflight_cannot_admit_a_late_analysis(pilot, monkeypatch):
    checked = []

    def cancel_during_verification(directory, *, cancel):
        checked.append(cancel)
        cancel.set()

    monkeypatch.setattr("app.pilot.encoder.verify_artifacts", cancel_during_verification)
    with pytest.raises(TaskCancelled):
        pilot.start("lithium selective membranes", "lithium selective membranes")
    assert checked == [pilot.model_cancel]
    assert pilot.coordinator.list_runs() == []


def test_catalogue_capacity_refuses_new_entry_without_silent_omission(tmp_path):
    from app.pilot.archive import DocumentArchive
    from app.pilot.library import MAX_IMPORTS, ResultLibrary, publish_catalogue_artifact

    library = ResultLibrary(tmp_path, DocumentArchive(tmp_path / "revisions"))
    library.directory.mkdir()
    for number in range(MAX_IMPORTS):
        (library.directory / (f"{number:064x}" + ".json")).write_text("{}")
    assert library.count() == MAX_IMPORTS
    with pytest.raises(TaskFailure, match="1000"):
        publish_catalogue_artifact(library.directory, {"new": "cannot exceed capacity"})
    assert library.count() == MAX_IMPORTS
    (library.directory / ("f" * 64 + ".json")).write_text("{}")
    with pytest.raises(TaskFailure, match="1000"):
        library.list_rows()


def test_duplicate_catalogue_payload_remains_idempotent_at_capacity(tmp_path):
    from app.pilot.library import MAX_IMPORTS, publish_catalogue_artifact

    digest = publish_catalogue_artifact(tmp_path, {"existing": "unchanged"})
    for number in range(MAX_IMPORTS - 1):
        (tmp_path / (f"{number:064x}" + ".json")).write_text("{}")
    assert publish_catalogue_artifact(tmp_path, {"existing": "unchanged"}) == digest
    assert len(list(tmp_path.glob("*.json"))) == MAX_IMPORTS


def test_catalogue_reader_checks_exact_canonical_bytes_not_just_json_semantics(tmp_path):
    from app.pilot.library import read_artifact, write_artifact

    identifier = write_artifact(tmp_path, {"query": "real archived object"})
    path = tmp_path / (identifier + ".json")
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(TaskFailure, match="повреждён"):
        read_artifact(tmp_path, identifier)


def test_supplemental_entries_are_lazy_and_cancel_before_next_large_read(pilot, monkeypatch):
    from app.pilot import supplemental as module
    from app.pilot.library import write_artifact
    from app.pilot.supplemental import SupplementalImport, SupplementalLibrary
    from tests.test_pilot_evidence import Context

    library = SupplementalLibrary(pilot.data_dir, pilot.archive)
    for number in range(3):
        value = SupplementalImport(kind="arxiv", created_at=datetime.now(UTC), documents=(), limitations=(str(number),))
        write_artifact(library.directory, value.model_dump(mode="json"))
    read = module.read_artifact
    opened = []

    def tracked(*args):
        opened.append(args[1])
        return read(*args)

    monkeypatch.setattr(module, "read_artifact", tracked)
    context = Context()
    entries = library.entries(cancel=context)
    assert opened == []
    next(entries)
    assert len(opened) == 1
    context.cancel_event.set()
    with pytest.raises(TaskCancelled):
        next(entries)
    assert len(opened) == 1


def test_supplemental_text_limit_is_reported_without_pretending_complete_search(pilot, monkeypatch):
    from app.pilot.library import write_artifact
    from app.pilot.supplemental import SupplementalImport, SupplementalLibrary
    from tests.test_pilot_evidence import Context

    reference = pilot.archive.put(document(42).model_copy(update={"source": "arxiv", "source_id": "2501.00042"}))
    record = SupplementalImport(kind="arxiv", created_at=datetime.now(UTC), documents=(reference,))
    library = SupplementalLibrary(pilot.data_dir, pilot.archive)
    write_artifact(library.directory, record.model_dump(mode="json"))
    monkeypatch.setattr("app.pilot.supplemental.MAX_MATCH_TEXT_BYTES", 10)
    class CandidateInput:
        label = "lithium selective membranes"
        synonyms = (label,)
        exclusions = ()

    result = library.match(CandidateInput(), Context())
    assert result["limited"] and result["limit_reason"] == "text_limit"
    assert result["items"] == [] and result["text_bytes"] == 0 and result["scanned"] == 1


@pytest.mark.parametrize("invalid", [{"deepseek_api_key": "key with whitespace"}, {"unknown_key": "token"}])
def test_invalid_second_key_cannot_partially_change_first_key_budget_or_settings(pilot, monkeypatch, invalid):
    from app.pilot.settings import load_settings

    scope = restore_conservative_day(pilot)
    monkeypatch.delattr(pilot.credentials, "get")
    pilot.credentials.set("openalex_api_key", "original-key")
    before = load_settings(pilot.data_dir)
    keys = {"openalex_api_key": "replacement-key", **invalid}
    values = before.model_dump(mode="json") | {"day_cost_micro": 3_000_000}
    with pytest.raises(ValueError):
        pilot.configure(values, keys)
    assert pilot.credentials.get("openalex_api_key") == "original-key"
    assert load_settings(pilot.data_dir) == before
    budget = next(item for item in BudgetAdmin(pilot.coordinator).status()["scopes"] if item["scope_id"] == scope)
    assert budget["limits"]["cost_micro"] == budget["used"]["cost_micro"] == 20_000
    assert budget["remaining"]["cost_micro"] == 0


def test_unavailable_secure_storage_preflight_keeps_daily_budget_and_settings(pilot, monkeypatch):
    from app.pilot.settings import load_settings
    from app.runtime.credentials import CredentialUnavailable

    scope = restore_conservative_day(pilot)
    before = load_settings(pilot.data_dir)
    monkeypatch.setattr(pilot.credentials, "_load_backend", lambda: None)
    values = before.model_dump(mode="json") | {"day_cost_micro": 3_000_000}
    with pytest.raises(CredentialUnavailable):
        pilot.configure(values, {"openalex_api_key": "new-key"}, persistent=True)
    assert load_settings(pilot.data_dir) == before
    assert not pilot.credentials._session
    budget = next(item for item in BudgetAdmin(pilot.coordinator).status()["scopes"] if item["scope_id"] == scope)
    assert budget["limits"]["cost_micro"] == 20_000


def test_a_local_run_is_not_fenced_by_the_paid_request_budget():
    """The local naming stage needs one request per candidate, not one per four.

    A run attempts at most 60 candidates. Under the paid fence of 24 requests a
    local run left two thirds of them with an unverified lexical name, so none
    of them could become a specific technology, receive a historical assessment
    or enter the TOP, and the result reported an empty list.
    """
    from app.pilot.contracts import QueryLimits
    from app.pilot.service import _day_budget_limits, _run_budget_limits

    limits = QueryLimits()
    paid = _run_budget_limits("deepseek", limits, 500_000)
    local = _run_budget_limits("local", limits, 500_000)
    assert paid == BudgetLimits(limits.llm_calls, limits.input_tokens, limits.output_tokens, 500_000)
    # 60 namings and 60 passports, each entitled to one corrective retry.
    assert local.calls >= 4 * 60
    assert local.input_tokens > paid.input_tokens and local.output_tokens > paid.output_tokens
    # The tariff is zero by construction, so the user's money cap is untouched.
    assert local.cost_micro == paid.cost_micro == 500_000
    assert _day_budget_limits("deepseek", 2_000_000) == BudgetLimits(96, 800000, 120000, 2_000_000)
    assert _day_budget_limits("local", 2_000_000).calls >= local.calls


def test_a_local_day_opened_under_the_paid_fence_is_raised_not_kept(pilot):
    from app.pilot.service import _day_budget_limits, _day_budget_scope

    settings = PilotSettings(provider="local")
    scope = "day/local/" + datetime.now(UTC).date().isoformat()
    expected = _day_budget_limits("local", settings.day_cost_micro)

    def upgrade():
        budget = BudgetService(pilot.coordinator._connection)
        budget.create_scope(scope, BudgetLimits(96, 800_000, 120_000, settings.day_cost_micro), currency="USD")
        assert _day_budget_scope(budget, settings) == scope
        return budget.snapshot(scope).limits

    assert pilot.coordinator._executor.submit(upgrade).result() == expected
