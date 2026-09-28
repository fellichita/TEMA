"""Credential boundaries: persistent backend policy, inheritance and safe failures."""

import os
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest

from app.runtime import credentials
from app.runtime.credentials import (
    KEYRING_NAMESPACE, LEGACY_ENVIRONMENT, CredentialStore, CredentialUnavailable,
)


@pytest.fixture(autouse=True)
def clean_credentials_environment(monkeypatch):
    for name in LEGACY_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


def install_backend(monkeypatch, *, secure=True, failing=False):
    class MemoryKeyring:
        def __init__(self):
            self.values = {}

        def get_password(self, service, username):
            if failing:
                raise RuntimeError("backend leaked private-token")
            return self.values.get((service, username))

        def set_password(self, service, username, password):
            if failing:
                raise RuntimeError("backend leaked " + password)
            self.values[(service, username)] = password

        def delete_password(self, service, username):
            if failing:
                raise RuntimeError("backend leaked private-token")
            self.values.pop((service, username), None)

    MemoryKeyring.__module__ = "keyring.backends.macOS" if secure else "keyrings.alt.file"
    MemoryKeyring.__name__ = "Keyring" if secure else "PlaintextKeyring"
    backend = MemoryKeyring()
    monkeypatch.setattr(credentials.sys, "platform", "darwin")
    monkeypatch.setattr(credentials.importlib, "import_module", lambda name: SimpleNamespace(get_keyring=lambda: backend))
    return backend


def test_import_is_lazy_and_session_storage_requires_no_keyring(monkeypatch):
    def forbidden_import(name):
        raise AssertionError("Session credentials must not load keyring")

    monkeypatch.setattr(credentials.importlib, "import_module", forbidden_import)
    store = CredentialStore()
    store.set("openalex_api_key", "private-token")
    assert store.get("openalex_api_key") == "private-token"
    assert "private-token" not in repr(store)
    store.delete("openalex_api_key")
    store.close()


def test_secure_backend_uses_only_main2_namespace(monkeypatch):
    backend = install_backend(monkeypatch)
    store = CredentialStore()
    store.set("deepseek_api_key", "private-token", persistent=True)
    assert backend.values == {(KEYRING_NAMESPACE, "deepseek_api_key"): "private-token"}
    assert store.storage_status.persistent_available
    assert store.get("deepseek_api_key") == "private-token"
    store.set("deepseek_api_key", "temporary-token")
    assert store.get("deepseek_api_key") == "temporary-token"
    store.delete("deepseek_api_key")
    assert store.get("deepseek_api_key") == "private-token"
    store.delete("deepseek_api_key", persistent=True)
    assert store.get("deepseek_api_key") is None
    assert not backend.values


def test_plaintext_backend_is_rejected_without_automatic_fallback(monkeypatch):
    backend = install_backend(monkeypatch, secure=False)
    store = CredentialStore()
    assert store.storage_status.reason == "backend_not_allowed"
    with pytest.raises(CredentialUnavailable):
        store.set("openalex_api_key", "private-token", persistent=True)
    assert not backend.values
    assert store.get("openalex_api_key") is None
    store.set("openalex_api_key", "session-token", persistent=False)
    assert store.get("openalex_api_key") == "session-token"


def test_windows_backend_is_allowed_only_on_windows(monkeypatch):
    backend = install_backend(monkeypatch)
    type(backend).__module__ = "keyring.backends.Windows"
    type(backend).__name__ = "WinVaultKeyring"
    assert not CredentialStore().storage_status.persistent_available
    monkeypatch.setattr(credentials.sys, "platform", "win32")
    store = CredentialStore()
    store.set("openalex_api_key", "private-token", persistent=True)
    assert store.storage_status.persistent_available


def test_backend_chaining_is_rejected_even_when_system_backend_exists(monkeypatch):
    backend = install_backend(monkeypatch)
    type(backend).__module__ = "keyring.backends.chainer"
    type(backend).__name__ = "ChainerBackend"
    assert CredentialStore().storage_status.reason == "backend_not_allowed"


def test_backend_module_initialization_failure_is_safe(monkeypatch):
    monkeypatch.setattr(credentials.sys, "platform", "darwin")

    def broken_import(name):
        raise RuntimeError("private-token")

    monkeypatch.setattr(credentials.importlib, "import_module", broken_import)
    store = CredentialStore()
    assert store.storage_status.reason == "storage_unavailable"


def test_failing_system_keyring_omits_secret_and_keeps_session_choice_explicit(monkeypatch):
    install_backend(monkeypatch, failing=True)
    store = CredentialStore()
    with pytest.raises(CredentialUnavailable) as caught:
        store.set("openalex_api_key", "private-token", persistent=True)
    assert "private-token" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert not store.storage_status.persistent_available
    with pytest.raises(CredentialUnavailable):
        store.get("openalex_api_key")
    store.set("openalex_api_key", "session-token", persistent=False)
    assert store.get("openalex_api_key") == "session-token"


def test_missing_dependency_allows_explicit_session_storage(monkeypatch):
    monkeypatch.setattr(credentials.sys, "platform", "darwin")

    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(credentials.importlib, "import_module", missing)
    store = CredentialStore()
    assert store.storage_status.reason == "dependency_unavailable"
    with pytest.raises(CredentialUnavailable):
        store.set("openalex_api_key", "private-token", persistent=True)
    store.set("openalex_api_key", "private-token")
    assert store.get("openalex_api_key") == "private-token"


@pytest.mark.parametrize("value", ["", "token\n", "token\x00", "token value", "x" * 16_385, 42])
def test_invalid_secret_fails_without_echoing_input(value):
    with pytest.raises(ValueError) as caught:
        CredentialStore().set("openalex_api_key", value)
    assert "token" not in str(caught.value)


def test_unknown_credential_name_rejected():
    with pytest.raises(ValueError):
        CredentialStore().get("arbitrary-secret-file")


def test_environment_import_is_once_and_scrubs_even_rejected_values(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "private-token")
    monkeypatch.setenv("YANDEX_API_KEY", "first-token")
    monkeypatch.setenv("YANDEX_CLOUD_API_KEY", "second-token")
    monkeypatch.setenv("EPO_OPS_KEY", "bad\nvalue")
    store = CredentialStore()
    result = store.import_legacy_environment()
    assert result.imported == ("openalex_api_key", "yandex_api_key")
    assert result.rejected == ("EPO_OPS_KEY",)
    assert not any(name in os.environ for name in LEGACY_ENVIRONMENT)
    assert store.get("openalex_api_key") == "private-token"
    assert store.get("yandex_api_key") == "first-token"
    monkeypatch.setenv("OPENALEX_API_KEY", "replacement-token")
    assert store.import_legacy_environment().imported == ()
    assert "OPENALEX_API_KEY" not in os.environ
    assert store.get("openalex_api_key") == "private-token"
    assert "private-token" not in repr(result)


def test_real_child_inherits_no_api_credentials_or_secret_arguments(monkeypatch):
    for name in LEGACY_ENVIRONMENT:
        monkeypatch.setenv(name, "test-private-marker-739")
    store = CredentialStore()
    store.import_legacy_environment()
    result = subprocess.run(
        [sys.executable, "-c", (
            "import os, sys; "
            "names = ('OPENALEX_API_KEY', 'EPO_OPS_KEY', 'EPO_OPS_SECRET', 'DEEPSEEK_API_KEY', "
            "'YANDEX_API_KEY', 'YANDEX_CLOUD_API_KEY', 'YANDEX_IAM_TOKEN', 'OPENAI_API_KEY'); "
            "assert not any(name in os.environ for name in names); "
            "assert not any('test-private-marker-' + '739' in arg for arg in sys.argv); print('clean')"
        )], capture_output=True, text=True, check=True, timeout=15,
    )
    assert result.stdout.strip() == "clean"
    assert result.stderr == ""
    assert store.get("openalex_api_key") == "test-private-marker-739"


def test_store_cannot_be_pickled_or_accessed_from_child_identity(monkeypatch):
    store = CredentialStore()
    store.set("deepseek_api_key", "private-token")
    with pytest.raises(TypeError):
        pickle.dumps(store)
    parent = os.getpid()
    monkeypatch.setattr(credentials.os, "getpid", lambda: parent + 1)
    with pytest.raises(CredentialUnavailable):
        store.get("deepseek_api_key")


def test_closed_store_is_inaccessible():
    store = CredentialStore()
    store.set("deepseek_api_key", "private-token")
    store.close()
    with pytest.raises(CredentialUnavailable):
        store.get("deepseek_api_key")


def test_validate_all_updates_rejects_invalid_second_key_without_writing_first(monkeypatch):
    backend = install_backend(monkeypatch)
    store = CredentialStore()
    store.set("openalex_api_key", "old-key", persistent=False)
    with pytest.raises(ValueError):
        store.validate_updates({"openalex_api_key": "new-key", "unknown": "bad-key"}, persistent=True)
    assert store.get("openalex_api_key") == "old-key"
    assert not backend.values


def test_valid_preflight_does_not_read_or_write_system_credentials(monkeypatch):
    backend = install_backend(monkeypatch, failing=True)
    store = CredentialStore()
    store.validate_updates({"openalex_api_key": "new-key", "deepseek_api_key": ""}, persistent=True)
    assert not backend.values and not store._session
    # OS denial may still happen later; validation does not promise a transaction.
    with pytest.raises(CredentialUnavailable):
        store.set("openalex_api_key", "new-key", persistent=True)


def test_preflight_checks_backend_only_when_changes_require_persistence(monkeypatch):
    install_backend(monkeypatch, secure=False)
    store = CredentialStore()
    store.validate_updates({"openalex_api_key": ""}, persistent=True)
    store.validate_updates({"openalex_api_key": "session-key"}, persistent=False)
    with pytest.raises(CredentialUnavailable):
        store.validate_updates({"openalex_api_key": "system-key"}, persistent=True)
    assert not store._session


@pytest.mark.parametrize("keys", [{"openalex_api_key": None}, {"openalex_api_key": 0},
    {"openalex_api_key": "contains whitespace"}, {"unknown": ""}, [], "invalid"])
def test_preflight_rejects_invalid_types_and_blanks_for_unknown_names(keys):
    with pytest.raises(ValueError):
        CredentialStore().validate_updates(keys)
