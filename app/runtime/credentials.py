"""Parent-process credentials with an explicitly selected persistence policy.

Call ``import_legacy_environment`` before spawning any child or starting workers.
The store is deliberately not serializable. CPU work receives neither this store
nor credentials. Python strings cannot be reliably zeroed in memory; this module
does not pretend to be an operating-system process sandbox.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Never, Protocol, SupportsIndex, cast

from app.identity import KEYRING_NAMESPACE

LEGACY_ENVIRONMENT = {
    "OPENALEX_API_KEY": "openalex_api_key",
    "EPO_OPS_KEY": "epo_ops_key",
    "EPO_OPS_SECRET": "epo_ops_secret",
    "DEEPSEEK_API_KEY": "deepseek_api_key",
    "YANDEX_API_KEY": "yandex_api_key",
    "YANDEX_CLOUD_API_KEY": "yandex_api_key",
    "YANDEX_IAM_TOKEN": "yandex_iam_token",
    "WORDSTAT_API_KEY": "wordstat_api_key",
    "OPENAI_API_KEY": "openai_api_key",
}
CREDENTIAL_NAMES = frozenset(LEGACY_ENVIRONMENT.values())
_SECURE_BACKENDS = {
    "darwin": ("keyring.backends.macOS", "Keyring"),
    "win32": ("keyring.backends.Windows", "WinVaultKeyring"),
}


class CredentialUnavailable(RuntimeError):
    """A safe, value-free failure suitable for display to a user."""


class SystemKeyring(Protocol):
    def get_password(self, service: str, username: str) -> str | None: ...

    def set_password(self, service: str, username: str, password: str) -> None: ...

    def delete_password(self, service: str, username: str) -> None: ...


@dataclass(frozen=True)
class CredentialStorageStatus:
    persistent_available: bool
    reason: str


@dataclass(frozen=True)
class EnvironmentImport:
    imported: tuple[str, ...]
    rejected: tuple[str, ...]


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or name not in CREDENTIAL_NAMES:
        raise ValueError("Неизвестный тип ключа доступа.")


def _validate_secret(secret: str) -> None:
    if (
        not isinstance(secret, str) or not secret or len(secret) > 16_384
        or any(character.isspace() or ord(character) < 32 for character in secret)
    ):
        raise ValueError("Некорректное значение ключа доступа.")


class CredentialStore:
    """A thread-safe store owned by its creating process.

    ``set(..., persistent=False)`` explicitly opts into session-only storage.
    A failed persistent write raises; it never silently creates a plaintext or
    session fallback. Only exact system backend classes are accepted, including
    when the optional keyring package is configured with a custom backend.
    """

    def __init__(self) -> None:
        self._owner_pid = os.getpid()
        self._session: dict[str, str] = {}
        self._lock = threading.RLock()
        self._backend: SystemKeyring | None = None
        self._loaded = False
        self._imported_environment = False
        self._closed = False
        self._status = CredentialStorageStatus(False, "not_checked")

    def __repr__(self) -> str:
        return f"CredentialStore(namespace={KEYRING_NAMESPACE!r}, closed={self._closed})"

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        raise TypeError("Хранилище ключей нельзя передавать в другой процесс.")

    def _check_owner(self) -> None:
        if os.getpid() != self._owner_pid or self._closed:
            raise CredentialUnavailable("Хранилище ключей недоступно в этом процессе или закрыто.")

    def _load_backend(self) -> SystemKeyring | None:
        if self._loaded:
            return self._backend
        self._loaded = True
        expected = _SECURE_BACKENDS.get(sys.platform)
        if expected is None:
            self._status = CredentialStorageStatus(False, "platform_not_supported")
            return None
        try:
            module = importlib.import_module("keyring")
        except ImportError:
            self._status = CredentialStorageStatus(False, "dependency_unavailable")
            return None
        except Exception:
            self._status = CredentialStorageStatus(False, "storage_unavailable")
            return None
        try:
            backend = module.get_keyring()
            if (type(backend).__module__, type(backend).__name__) != expected:
                self._status = CredentialStorageStatus(False, "backend_not_allowed")
                return None
            self._backend = cast(SystemKeyring, backend)
            self._status = CredentialStorageStatus(True, "system_keyring")
        except Exception:
            self._status = CredentialStorageStatus(False, "storage_unavailable")
        return self._backend

    @property
    def storage_status(self) -> CredentialStorageStatus:
        with self._lock:
            self._check_owner()
            self._load_backend()
            return self._status

    def _persistent_failure(self) -> CredentialUnavailable:
        self._status = CredentialStorageStatus(False, "storage_unavailable")
        return CredentialUnavailable(
            "Системное хранилище ключей недоступно. Можно явно выбрать хранение на текущую сессию."
        )

    def get(self, name: str) -> str | None:
        _validate_name(name)
        with self._lock:
            self._check_owner()
            if name in self._session:
                return self._session[name]
            backend = self._load_backend()
            if backend is None:
                return None
            try:
                value = backend.get_password(KEYRING_NAMESPACE, name)
                if value is not None:
                    _validate_secret(value)
                self._status = CredentialStorageStatus(True, "system_keyring")
                return value
            except Exception:
                raise self._persistent_failure() from None

    def set(self, name: str, secret: str, *, persistent: bool = False) -> None:
        _validate_name(name)
        _validate_secret(secret)
        with self._lock:
            self._check_owner()
            if not persistent:
                self._session[name] = secret
                return
            backend = self._load_backend()
            if backend is None:
                raise CredentialUnavailable("Защищённое системное хранилище ключей недоступно.")
            try:
                backend.set_password(KEYRING_NAMESPACE, name, secret)
                self._status = CredentialStorageStatus(True, "system_keyring")
            except Exception:
                raise self._persistent_failure() from None
            self._session.pop(name, None)

    def validate_updates(self, keys: Mapping[str, str], *, persistent: bool = False) -> None:
        """Validate an entire settings submission before any budget/key mutation.

        Empty strings mean unchanged. This performs no credential reads or writes;
        checking an allowed backend cannot guarantee a later OS write succeeds.
        System keychains provide no atomic transaction across several credentials.
        """
        if not isinstance(keys, Mapping) or type(persistent) is not bool:
            raise ValueError("Некорректный набор ключей или режим хранения.")
        with self._lock:
            self._check_owner()
            changed = False
            for name, secret in keys.items():
                _validate_name(name)
                if secret == "":
                    continue
                _validate_secret(secret)
                changed = True
            if changed and persistent and self._load_backend() is None:
                raise CredentialUnavailable("Защищённое системное хранилище ключей недоступно.")

    def delete(self, name: str, *, persistent: bool = False) -> None:
        """Delete session credentials, and system credentials only when requested."""
        _validate_name(name)
        with self._lock:
            self._check_owner()
            if persistent:
                backend = self._load_backend()
                if backend is None:
                    raise CredentialUnavailable("Не удалось удалить ключ из системного хранилища.")
                try:
                    if backend.get_password(KEYRING_NAMESPACE, name) is not None:
                        backend.delete_password(KEYRING_NAMESPACE, name)
                except Exception:
                    raise self._persistent_failure() from None
            self._session.pop(name, None)

    def import_legacy_environment(self) -> EnvironmentImport:
        """Import once into memory, always removing known names before returning.

        Call during single-threaded bootstrap. Never write values back into the
        environment. Aliases use the first non-empty valid value in mapping order.
        Repeated calls scrub newly inserted names but do not import them again.
        """
        with self._lock:
            self._check_owner()
            removed = {name: os.environ.pop(name, None) for name in LEGACY_ENVIRONMENT}
            if self._imported_environment:
                return EnvironmentImport((), ())
            self._imported_environment = True
            imported: set[str] = set()
            rejected: set[str] = set()
            for environment_name, name in LEGACY_ENVIRONMENT.items():
                secret = removed[environment_name]
                if not secret:
                    continue
                try:
                    _validate_secret(secret)
                except ValueError:
                    rejected.add(environment_name)
                    continue
                if name not in self._session:
                    self._session[name] = secret
                    imported.add(name)
            return EnvironmentImport(tuple(sorted(imported)), tuple(sorted(rejected)))

    def close(self) -> None:
        with self._lock:
            self._session.clear()
            self._backend = None
            self._closed = True
