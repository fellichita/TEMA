"""Parent-owned application session; no credentials are passed to CPU processes."""

from threading import RLock

from app.runtime.credentials import CredentialStore

_lock = RLock()
_credentials: CredentialStore | None = None


def credentials() -> CredentialStore:
    global _credentials
    with _lock:
        if _credentials is None:
            _credentials = CredentialStore()
            _credentials.import_legacy_environment()
        return _credentials
