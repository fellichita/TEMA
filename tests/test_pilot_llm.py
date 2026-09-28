"""Real HTTP adapter + real SQLite ledger, with deterministic provider responses."""

import gzip
import json
import time
from dataclasses import replace
from threading import Event, Thread

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Json

from app.pilot.llm import LlmCancelled, LlmClient, LlmError, MissingCredential, ProviderConfig
from app.runtime.budget import BudgetExceeded, BudgetLimits, BudgetUsage, BudgetService, BudgetStateError
from app.runtime.credentials import CredentialStore
from app.sqlite_runtime import sqlite3

SECRET = "test-credential-do-not-log-1234567890"
CONFIG = ProviderConfig(provider="deepseek", model="deepseek-v4-flash", model_version="fixture-model-1",
    pricing_version="fixture-price-1", currency="USD", input_per_million_micro=440000,
    output_per_million_micro=1320000)
YANDEX_CONFIG = ProviderConfig(provider="yandex", model="gpt://folder123/deepseek-v4-flash",
    model_version="fixture-model", pricing_version="fixture-rub", currency="RUB",
    input_per_million_micro=300_000_000, output_per_million_micro=500_000_000)
YANDEX_LIMITS = BudgetLimits(24, 200000, 30000, 100_000_000)


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str


def response_payload(content=None, **changes):
    return dict(model="deepseek-v4-flash", usage=dict(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                choices=[dict(finish_reason="stop", message=dict(role="assistant", content=(
                    json.dumps({"answer": "An actual JSON answer"}) if content is None else content)))]) | changes


@pytest.fixture
def clients(tmp_path):
    resources = []

    def factory(handler, *, config=CONFIG, limits=None, **kwargs):
        connection = sqlite3.connect(tmp_path / f"ledger-{len(resources)}.sqlite3", isolation_level=None)
        budget = BudgetService(connection)
        budget.create_scope("run:one", limits or BudgetLimits(24, 200000, 30000, 1_000_000), currency=config.currency)
        credentials = CredentialStore()
        credentials.set(config.credential_name, SECRET, persistent=False)
        http = httpx.Client(transport=httpx.MockTransport(handler))
        client = LlmClient(config, credentials, budget, http_client=http, **kwargs)
        resources.append((client, http, connection, credentials))
        return client, budget

    yield factory
    for client, http, connection, credentials in resources:
        client.close()
        http.close()
        connection.close()
        credentials.close()


def generate(client, **changes):
    values = dict(system_prompt="Return a short fact.", user_content="User research text", prompt_version="test/1",
                  request_id="request:one", scope_ids=("run:one",), cancel=Event(), max_output_tokens=3000)
    return client.generate_json(Answer, **(values | changes))


def test_request_is_json_non_thinking_bounded_and_budgeted_before_network(clients):
    requests = []
    database_path = None

    def handler(request):
        requests.append(request)
        # HTTP runs on its own thread. Inspect the durable state through a
        # separate reader, preserving the ledger's coordinator-thread guard.
        reader = sqlite3.connect(database_path)
        try:
            assert reader.execute("SELECT state FROM pilot_budget_requests WHERE request_id='request:one'").fetchone()[0] == "sent"
        finally:
            reader.close()
        body = json.loads(request.content)
        assert str(request.url) == "https://api.deepseek.com/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {SECRET}"
        assert body["thinking"] == {"type": "disabled"}
        assert body["stream"] is False and body["max_tokens"] == 3000
        assert body["response_format"] == {"type": "json_object"}
        assert "tools" not in body and len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert "JSON schema" in body["messages"][0]["content"]
        return httpx.Response(200, json=response_payload())

    client, budget = clients(handler)
    database_path = budget.connection.execute("PRAGMA database_list").fetchone()[2]
    completion = generate(client)
    assert isinstance(completion.value, Answer)
    assert completion.receipt.prompt_version == "test/1"
    assert completion.receipt.configured_model_version == "fixture-model-1"
    assert completion.receipt.pricing_version == "fixture-price-1"
    assert completion.receipt.cost_micro == 71
    assert json.loads(json.dumps(completion.receipt.to_dict()))["usage"]["total_tokens"] == 120
    assert budget.reservation("request:one").state == "settled"
    assert budget.snapshot("run:one").used == BudgetUsage(1, 100, 20, 71)
    assert SECRET not in repr(client) + repr(completion)
    assert len(requests) == 1


def test_yandex_uses_only_its_fixed_endpoint_project_logging_and_explicit_tariff(clients):
    config = ProviderConfig(provider="yandex", model="gpt://folder123/deepseek-v4-flash",
        model_version="fixture-model", pricing_version="fixture-rub", currency="RUB",
        input_per_million_micro=300_000_000, output_per_million_micro=500_000_000)

    def handler(request):
        assert str(request.url) == "https://ai.api.cloud.yandex.net/v1/chat/completions"
        assert request.headers["Authorization"] == f"Api-Key {SECRET}"
        assert request.headers["x-project"] == "folder123"
        assert request.headers["x-data-logging-enabled"] == "false"
        body = json.loads(request.content)
        assert body["reasoning_effort"] == "none" and body["store"] is False
        return httpx.Response(200, json=response_payload(model=config.model))

    client, budget = clients(handler, config=config, limits=BudgetLimits(24, 200000, 30000, 100_000_000))
    assert generate(client).receipt.cost_micro == 40000
    assert budget.reservation("request:one").currency == "RUB"


@pytest.mark.parametrize("changes", [dict(provider="custom"), dict(model="https://evil.example/v1"),
    dict(model="deepseek-chat"), dict(currency="RUB"), dict(input_per_million_micro=0),
    dict(output_per_million_micro=True), dict(model_version="model with spaces"), dict(authentication="iam_token")])
def test_provider_configuration_has_no_arbitrary_endpoints_old_aliases_or_unstated_free_pricing(changes):
    with pytest.raises(ValueError):
        replace(CONFIG, **changes)


@pytest.mark.parametrize("status", [301, 307, 401, 403, 429, 500, 503])
def test_http_error_is_never_retried_redirected_or_refunded_without_usage(clients, status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, headers={"Location": "https://evil.example/"},
                              text=f"Provider echoed {SECRET}")

    client, budget = clients(handler)
    with pytest.raises(LlmError) as caught:
        generate(client)
    assert SECRET not in str(caught.value)
    assert len(requests) == 1
    reservation = budget.reservation("request:one")
    assert reservation.state == "unknown" and reservation.charged == reservation.reserved
    with pytest.raises(BudgetStateError):
        generate(client)
    assert len(requests) == 1


def test_timeout_keeps_exact_reservation_and_hides_network_exception_payload(clients):
    def handler(request):
        raise httpx.ReadTimeout(f"sensitive request {SECRET}", request=request)

    client, budget = clients(handler)
    with pytest.raises(LlmError) as caught:
        generate(client)
    assert caught.value.code == "read_timeout" and SECRET not in str(caught.value)
    assert budget.reservation("request:one").state == "unknown"
    assert budget.snapshot("run:one").used.calls == 1


def test_completion_read_window_does_not_inherit_short_connection_timeout(clients):
    requests = []

    def handler(request):
        requests.append(request)
        timeout = request.extensions["timeout"]
        # Reproduce the admission of a provider that needs 31 s before headers,
        # without adding a 31 s artificial sleep to each test suite.
        if timeout["read"] <= 31:
            raise httpx.ReadTimeout("fixture generation exceeded read timeout", request=request)
        assert timeout["read"] == 120 and timeout["connect"] == 10
        assert timeout["write"] == 30 and timeout["pool"] == 10
        return httpx.Response(200, json=response_payload())

    client, budget = clients(handler)
    assert generate(client).value.answer
    assert len(requests) == 1 and budget.reservation("request:one").state == "settled"


@pytest.mark.parametrize(("exception", "code"), [
    (httpx.ConnectTimeout, "connect_timeout"), (httpx.WriteTimeout, "write_timeout"),
    (httpx.PoolTimeout, "pool_timeout"), (httpx.RemoteProtocolError, "transport_failed")])
def test_transport_diagnostics_preserve_safe_subtype_and_unknown_budget(clients, exception, code):
    calls = []

    def handler(request):
        calls.append(request)
        raise exception(f"{SECRET} provider request body", request=request)

    client, budget = clients(handler)
    with pytest.raises(LlmError) as caught:
        generate(client)
    assert caught.value.code == code and SECRET not in str(caught.value)
    assert len(calls) == 1 and budget.reservation("request:one").state == "unknown"


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_stalled_completion_has_prompt_cancellation_and_hard_wall_deadline(clients, stop):
    entered, release, finished, cancel = Event(), Event(), Event(), Event()
    calls = []

    def handler(request):
        calls.append(request)
        entered.set()
        try:
            assert release.wait(3), "Test must release the simulated stalled socket."
            return httpx.Response(200, json=response_payload())
        finally:
            finished.set()

    def cancel_on_dispatch():
        if entered.wait(2):
            cancel.set()

    client, budget = clients(handler, deadline_seconds=.15 if stop == "deadline" else 150)
    canceller = Thread(target=cancel_on_dispatch, daemon=True) if stop == "cancel" else None
    if canceller:
        canceller.start()
    started = time.monotonic()
    try:
        with pytest.raises(LlmError) as caught:
            generate(client, cancel=cancel)
        assert caught.value.code == ("cancelled" if stop == "cancel" else "response_deadline")
        assert time.monotonic() - started < .75
        assert budget.reservation("request:one").state == "unknown"
        cancel.clear()
        with pytest.raises(LlmError) as blocked:
            generate(client, request_id="request:two")
        assert blocked.value.code == "transport_unavailable"
        assert budget.snapshot("run:one").used.calls == 1 and len(calls) == 1
    finally:
        release.set()
        assert finished.wait(2)
        if canceller:
            canceller.join(2)


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 100, "completion_tokens": 20},
    {"prompt_tokens": True, "completion_tokens": 20, "total_tokens": 21},
    {"prompt_tokens": "100", "completion_tokens": 20, "total_tokens": 120},
    {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 999},
    {"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100}])
def test_missing_invalid_or_contradictory_usage_holds_reservation(clients, usage):
    client, budget = clients(lambda _: httpx.Response(200, json=response_payload(usage=usage)))
    with pytest.raises(LlmError) as caught:
        generate(client)
    assert caught.value.code == "usage_unknown"
    reservation = budget.reservation("request:one")
    assert reservation.state == "unknown" and reservation.charged == reservation.reserved


@pytest.mark.parametrize("body", [b"", b"[]", b'{"usage":{},"usage":{}}', b'{"answer":NaN}',
                                   b'{"secret":"invalid UTF8 \xff"}'])
def test_untrusted_provider_json_is_strict_and_invalid_usage_cannot_be_released(clients, body):
    client, budget = clients(lambda _: httpx.Response(200, content=body))
    with pytest.raises(LlmError):
        generate(client)
    assert budget.reservation("request:one").state == "unknown"


@pytest.mark.parametrize("content", ["", "   ", '{"answer": true}', '{"answer":"x","new_endpoint":"evil"}',
                                     '{"answer":"a","answer":"b"}', '```json\n{}\n```'])
def test_unusable_output_with_valid_usage_is_charged_once_without_hidden_repair(clients, content):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=response_payload(content))

    client, budget = clients(handler)
    with pytest.raises(LlmError):
        generate(client)
    assert budget.reservation("request:one").state == "settled"
    assert budget.snapshot("run:one").used == BudgetUsage(1, 100, 20, 71)
    assert len(calls) == 1


def test_tools_and_incomplete_generation_are_rejected_without_executing_any_action(clients):
    payload = response_payload()
    payload["choices"][0]["finish_reason"] = "tool_calls"
    payload["choices"][0]["message"]["tool_calls"] = [{"function": {"name": "delete_everything"}}]
    client, budget = clients(lambda _: httpx.Response(200, json=payload))
    with pytest.raises(LlmError, match="прерван"):
        generate(client)
    assert budget.reservation("request:one").state == "settled"


def test_output_overrun_records_actual_usage_and_requires_reconciliation(clients):
    usage = dict(prompt_tokens=100, completion_tokens=4000, total_tokens=4100)
    client, budget = clients(lambda _: httpx.Response(200, json=response_payload(usage=usage)))
    with pytest.raises(LlmError) as caught:
        generate(client)
    assert caught.value.code == "usage_overrun"
    assert budget.reservation("request:one").charged.output_tokens == 4000
    assert budget.snapshot("run:one").requires_reconciliation


def test_cancel_before_reserve_and_after_reserve_refunds_only_proven_unsent(clients, monkeypatch):
    requests = []
    client, budget = clients(lambda request: requests.append(request))
    cancel = Event()
    cancel.set()
    with pytest.raises(LlmCancelled):
        generate(client, cancel=cancel)
    assert budget.snapshot("run:one").used.calls == 0
    cancel.clear()
    reserve = budget.reserve

    def reserve_and_cancel(*args, **kwargs):
        value = reserve(*args, **kwargs)
        cancel.set()
        return value

    monkeypatch.setattr(budget, "reserve", reserve_and_cancel)
    with pytest.raises(LlmCancelled):
        generate(client, cancel=cancel)
    assert budget.reservation("request:one").state == "released"
    assert not requests


def test_cancel_after_dispatch_preserves_unknown_reservation(clients):
    cancel = Event()

    def handler(_):
        cancel.set()
        return httpx.Response(200, json=response_payload())

    client, budget = clients(handler)
    with pytest.raises(LlmCancelled):
        generate(client, cancel=cancel)
    assert budget.reservation("request:one").state == "unknown"


def test_missing_credentials_exhausted_budget_and_huge_prompt_never_contact_provider(clients, monkeypatch):
    requests = []
    client, budget = clients(lambda request: requests.append(request))
    monkeypatch.setattr(client._credentials, "get", lambda _: None)
    with pytest.raises(MissingCredential):
        generate(client)
    assert budget.snapshot("run:one").used.calls == 0
    with pytest.raises(LlmError, match="размер"):
        generate(client, user_content="x" * 200000)
    client2, budget2 = clients(lambda request: requests.append(request), limits=BudgetLimits(0, 0, 0, 0))
    with pytest.raises(BudgetExceeded):
        generate(client2)
    assert budget2.snapshot("run:one").used.calls == 0 and not requests


def test_oversized_and_compressed_response_bombs_are_bounded_before_json_decode(clients):
    client, budget = clients(lambda _: httpx.Response(200, headers={"content-encoding": "gzip"},
        stream=httpx.ByteStream(gzip.compress(b"x" * 10000))), max_response_bytes=128)
    with pytest.raises(LlmError):
        generate(client)
    assert budget.reservation("request:one").state == "unknown"


def test_provider_cannot_echo_credentials_into_saved_completion_or_exception(clients):
    client, budget = clients(lambda _: httpx.Response(200, json=response_payload(json.dumps({"answer": SECRET}))))
    with pytest.raises(LlmError) as caught:
        generate(client)
    assert SECRET not in str(caught.value) and client.last_receipt is None
    assert budget.reservation("request:one").state == "settled"


@pytest.mark.parametrize("encoding", ["envelope", "content", "typed_json"])
@pytest.mark.parametrize("location", ["nested_value", "nested_key"])
def test_escaped_credentials_are_rejected_after_every_supported_json_decode(clients, encoding, location):
    class NestedAnswer(BaseModel):
        answer: dict[str, list[str]]

    class EncodedAnswer(BaseModel):
        answer: Json[dict[str, list[str]]]

    nested = {"summary": [f"prefix {SECRET} suffix"]} if location == "nested_value" else {SECRET: ["public text"]}
    escaped = "".join(f"\\u{ord(char):04x}" for char in SECRET)
    schema = NestedAnswer
    if encoding == "typed_json":
        content = json.dumps({"answer": json.dumps(nested).replace(SECRET, escaped)})
        schema = EncodedAnswer
    else:
        content = json.dumps({"answer": nested})
        if encoding == "content":
            content = content.replace(SECRET, escaped)
    wire = json.dumps(response_payload(content))
    if encoding == "envelope":
        wire = wire.replace(SECRET, escaped)
    assert SECRET not in wire
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, content=wire.encode("utf-8"))

    client, budget = clients(handler)
    with pytest.raises(LlmError) as caught:
        client.generate_json(schema, system_prompt="Return JSON", user_content="Synthetic research",
            prompt_version="test/1", request_id="request:one", scope_ids=("run:one",), cancel=Event())
    assert caught.value.code == "unsafe_response"
    assert client.last_receipt is None
    assert SECRET not in str(caught.value) + repr(client)
    assert budget.reservation("request:one").state == "settled"
    assert budget.snapshot("run:one").used == BudgetUsage(1, 100, 20, CONFIG.quote(100, 20))
    saved_requests = budget.connection.execute("SELECT * FROM pilot_budget_requests").fetchall()
    saved_events = budget.connection.execute("SELECT * FROM pilot_budget_events").fetchall()
    assert SECRET not in repr(saved_requests) + repr(saved_events)
    assert len(requests) == 1


def test_decoded_nested_public_json_is_preserved_without_false_credential_detection(clients):
    class PublicAnswer(BaseModel):
        answer: Json[dict[str, list[str]]]

    public = {"secret": ["Статья о защите API-ключей", "Public research only"]}
    client, budget = clients(lambda _: httpx.Response(200, json=response_payload(
        json.dumps({"answer": json.dumps(public, ensure_ascii=True)}))))
    completion = client.generate_json(PublicAnswer, system_prompt="Return JSON", user_content="Public research",
        prompt_version="test/1", request_id="request:one", scope_ids=("run:one",), cancel=Event())
    assert completion.value.answer == public
    assert completion.receipt is client.last_receipt
    assert budget.reservation("request:one").state == "settled"


def test_a_failed_new_request_does_not_expose_the_previous_call_receipt(clients):
    client, _ = clients(lambda _: httpx.Response(200, json=response_payload()))
    assert generate(client).receipt == client.last_receipt
    cancel = Event()
    cancel.set()
    with pytest.raises(LlmCancelled):
        generate(client, request_id="new", cancel=cancel)
    assert client.last_receipt is None


def test_response_model_drift_is_detected_without_silent_fallback_or_refund(clients):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response_payload(model="deepseek-flash"))

    client, budget = clients(handler)
    with pytest.raises(LlmError) as caught:
        generate(client)
    assert caught.value.code == "model_mismatch"
    assert len(requests) == 1 and budget.reservation("request:one").state == "settled"


@pytest.mark.parametrize(("requested", "returned", "accepted"), [
    ("gpt://folder123/deepseek-v4-flash", "gpt://folder123/deepseek-v4-flash", True),
    ("gpt://folder123/deepseek-v4-flash", "deepseek-v4-flash", True),
    ("gpt://folder123/deepseek-v4-flash", "gpt://deepseek-v4-flash/latest", True),
    ("gpt://folder123/aliceai-llm-flash", "aliceai-llm-flash", True),
    ("gpt://folder123/aliceai-llm-flash", "gpt://aliceai-llm-flash/latest", True),
    ("gpt://folder123/deepseek-v4-flash/latest", "gpt://folder123/deepseek-v4-flash/latest", True),
    ("gpt://folder123/deepseek-v4-flash/version1", "gpt://folder123/deepseek-v4-flash/version1", True),
    ("gpt://folder123/deepseek-v4-flash", "deepseek-flash", False),
    ("gpt://folder123/deepseek-v4-flash", "aliceai-llm-flash", False),
    ("gpt://folder123/deepseek-v4-flash", "gpt://folder999/deepseek-v4-flash", False),
    ("gpt://folder123/deepseek-v4-flash", "gpt://folder123/deepseek-v4-flash/latest", False),
    ("gpt://folder123/deepseek-v4-flash", "deepseek-v4-flash/latest", False),
    ("gpt://folder123/deepseek-v4-flash", "gpt://deepseek-v4-flash/rc", False),
    ("gpt://folder123/deepseek-v4-flash", "gpt://deepseek-v4-flash/pinned", False),
    ("gpt://folder123/deepseek-v4-flash", "gpt://aliceai-llm-flash/latest", False),
    ("gpt://folder123/deepseek-v4-flash", "gpt://deepseek-v4-flash", False),
    ("gpt://folder123/deepseek-v4-flash", "gpt://deepseek-v4-flash/", False),
    ("gpt://folder123/deepseek-v4-flash", "gpt:///latest", False),
    ("gpt://folder123/deepseek-v4-flash", "gpt://deepseek-v4-flash//latest", False),
    ("gpt://folder123/deepseek-v4-flash", "gpt://deepseek-v4-flash/latest/extra", False),
    ("gpt://folder123/deepseek-v4-flash/latest", "deepseek-v4-flash", False),
    ("gpt://folder123/deepseek-v4-flash/latest", "gpt://deepseek-v4-flash/latest", False),
    ("gpt://folder123/deepseek-v4-flash/latest", "gpt://folder123/deepseek-v4-flash", False),
    ("gpt://folder123/deepseek-v4-flash/version1", "deepseek-v4-flash", False),
    ("gpt://folder123/deepseek-v4-flash/version1", "gpt://deepseek-v4-flash/latest", False),
    ("gpt://folder123/deepseek-v4-flash/version1", "deepseek-v4-flash/version1", False),
    ("gpt://folder123/deepseek-v4-flash/version1", "gpt://folder123/deepseek-v4-flash/version2", False),
])
def test_yandex_model_identity_preserves_explicit_folder_family_and_version(requested, returned, accepted):
    config = replace(YANDEX_CONFIG, model=requested)
    assert config.accepts_returned_model(returned) is accepted


@pytest.mark.parametrize(("returned", "accepted"), [
    ("deepseek-v4-flash", True),
    ("deepseek-flash", False),
    ("gpt://folder123/deepseek-v4-flash", False),
    ("gpt://deepseek-v4-flash/latest", False),
    (None, False),
    (123, False),
])
def test_direct_deepseek_model_identity_remains_exact(returned, accepted):
    assert CONFIG.accepts_returned_model(returned) is accepted


@pytest.mark.parametrize("name", ["deepseek-v4-flash", "aliceai-llm-flash"])
@pytest.mark.parametrize("response_format", ["bare", "public_uri"])
def test_yandex_accepts_response_model_format_and_audits_original_identifiers(clients, name, response_format):
    config = replace(YANDEX_CONFIG, model=f"gpt://folder123/{name}")
    returned_model = name if response_format == "bare" else f"gpt://{name}/latest"
    requests = []

    def handler(request):
        requests.append(request)
        assert json.loads(request.content)["model"] == config.model
        assert request.headers["x-project"] == "folder123"
        return httpx.Response(200, json=response_payload(model=returned_model))

    client, budget = clients(handler, config=config, limits=YANDEX_LIMITS)
    completion = generate(client)
    receipt = completion.receipt
    assert receipt == client.last_receipt
    assert receipt.requested_model == config.model and receipt.returned_model == returned_model
    assert receipt.configured_model_version == config.model_version
    assert receipt.pricing_version == config.pricing_version
    assert receipt.cost_basis == "configured_upper_tariff" and receipt.currency == "RUB"
    assert receipt.cost_micro == 40000
    assert budget.snapshot("run:one").used == BudgetUsage(1, 100, 20, 40000)
    assert budget.reservation("request:one").state == "settled" and len(requests) == 1


@pytest.mark.parametrize(("requested", "returned"), [
    ("gpt://folder123/deepseek-v4-flash", "aliceai-llm-flash"),
    ("gpt://folder123/deepseek-v4-flash", "deepseek-flash"),
    ("gpt://folder123/deepseek-v4-flash", "gpt://folder999/deepseek-v4-flash"),
    ("gpt://folder123/deepseek-v4-flash", "gpt://folder123/deepseek-v4-flash/latest"),
    ("gpt://folder123/deepseek-v4-flash", "gpt://deepseek-v4-flash/rc"),
    ("gpt://folder123/deepseek-v4-flash", "gpt://deepseek-v4-flash/pinned"),
    ("gpt://folder123/deepseek-v4-flash", "gpt://aliceai-llm-flash/latest"),
    ("gpt://folder123/deepseek-v4-flash", "gpt://deepseek-v4-flash"),
    ("gpt://folder123/deepseek-v4-flash", "gpt://deepseek-v4-flash/"),
    ("gpt://folder123/deepseek-v4-flash", "gpt:///latest"),
    ("gpt://folder123/deepseek-v4-flash", "gpt://deepseek-v4-flash//latest"),
    ("gpt://folder123/deepseek-v4-flash", "gpt://deepseek-v4-flash/latest/extra"),
    ("gpt://folder123/deepseek-v4-flash/latest", "deepseek-v4-flash"),
    ("gpt://folder123/deepseek-v4-flash/latest", "gpt://deepseek-v4-flash/latest"),
    ("gpt://folder123/deepseek-v4-flash/version1", "deepseek-v4-flash"),
    ("gpt://folder123/deepseek-v4-flash/version1", "gpt://deepseek-v4-flash/latest"),
    ("gpt://folder123/deepseek-v4-flash/version1", "gpt://folder123/deepseek-v4-flash/version2"),
])
def test_yandex_model_mismatch_preserves_safe_receipt_and_charges_once(clients, requested, returned):
    config = replace(YANDEX_CONFIG, model=requested)
    requests = []
    provider_debug = "provider-only-debug-do-not-persist"

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response_payload(model=returned, debug=provider_debug))

    client, budget = clients(handler, config=config, limits=YANDEX_LIMITS)
    with pytest.raises(LlmError) as caught:
        generate(client)
    assert caught.value.code == "model_mismatch"
    receipt = client.last_receipt
    assert receipt is not None
    assert receipt.requested_model == requested and receipt.returned_model == returned
    assert receipt.configured_model_version == config.model_version
    assert receipt.pricing_version == config.pricing_version and receipt.cost_micro == 40000
    assert receipt.cost_basis == "configured_upper_tariff" and receipt.currency == "RUB"
    audit = json.dumps(receipt.to_dict()) + str(caught.value)
    for private_text in (SECRET, provider_debug, "User research text", "Return a short fact."):
        assert private_text not in audit
    assert returned not in str(caught.value)
    assert budget.reservation("request:one").state == "settled"
    assert budget.snapshot("run:one").used == BudgetUsage(1, 100, 20, 40000)
    with pytest.raises(BudgetStateError):
        generate(client)
    assert len(requests) == 1


@pytest.mark.parametrize("model_fields", [
    {}, {"model": None}, {"model": False}, {"model": 123}, {"model": []}, {"model": {}},
    {"model": ""}, {"model": "deepseek v4 flash"}, {"model": "deepseek-v4-flash\n"},
    {"model": "<model>deepseek-v4-flash</model>"}, {"model": "deepseek-в4-flash"},
    {"model": "a" * 161},
], ids=["missing", "null", "boolean", "integer", "list", "object", "empty", "spaces",
        "newline", "markup", "non_ascii", "too_long"])
def test_unusable_response_model_is_charged_without_persisting_a_receipt(clients, model_fields):
    payload = response_payload()
    payload.pop("model")
    payload.update(model_fields)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    client, budget = clients(handler, config=YANDEX_CONFIG, limits=YANDEX_LIMITS)
    with pytest.raises(LlmError) as caught:
        generate(client)
    assert caught.value.code == "model_mismatch" and client.last_receipt is None
    assert budget.reservation("request:one").state == "settled"
    assert budget.snapshot("run:one").used == BudgetUsage(1, 100, 20, 40000)
    assert len(requests) == 1


@pytest.mark.parametrize("echo", ["raw_model", "escaped_model", "raw_debug", "escaped_debug",
                                 "escaped_nested_debug", "escaped_field_name"])
def test_credential_echo_cannot_enter_model_mismatch_receipt_or_error(clients, echo):
    payload = response_payload(model=SECRET if echo.endswith("model") else "aliceai-llm-flash")
    if echo in {"raw_debug", "escaped_debug"}:
        payload["debug"] = SECRET
    elif echo == "escaped_nested_debug":
        payload["debug"] = {"details": [{"value": SECRET}]}
    elif echo == "escaped_field_name":
        payload["debug"] = {SECRET: "public text"}
    response_text = json.dumps(payload)
    if echo.startswith("escaped"):
        response_text = response_text.replace(SECRET, "".join(f"\\u{ord(char):04x}" for char in SECRET))
        assert SECRET not in response_text
        if echo == "escaped_model":
            assert json.loads(response_text)["model"] == SECRET
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, content=response_text.encode("utf-8"))

    client, budget = clients(handler, config=YANDEX_CONFIG, limits=YANDEX_LIMITS)
    with pytest.raises(LlmError) as caught:
        generate(client)
    assert caught.value.code == "unsafe_response" and client.last_receipt is None
    assert SECRET not in str(caught.value) + repr(client)
    saved_requests = budget.connection.execute("SELECT * FROM pilot_budget_requests").fetchall()
    saved_events = budget.connection.execute("SELECT * FROM pilot_budget_events").fetchall()
    assert SECRET not in repr(saved_requests) + repr(saved_events)
    assert budget.reservation("request:one").state == "settled"
    assert budget.snapshot("run:one").used == BudgetUsage(1, 100, 20, 40000)
    assert len(requests) == 1


def test_currency_mismatch_is_rejected_before_reserving_or_sending(clients):
    requests = []
    client, budget = clients(lambda request: requests.append(request))
    budget.create_scope("rubles", BudgetLimits(10, 200000, 30000, 1000000), currency="RUB")
    with pytest.raises(LlmError) as caught:
        generate(client, scope_ids=("rubles",))
    assert caught.value.code == "currency_mismatch"
    assert budget.snapshot("rubles").used.calls == 0 and not requests
