"""Per-analysis scope adapter; no encoder imports or process-global monkeypatches."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol, TypedDict


class ScopeDecision(TypedDict):
    admitted: bool
    route: str
    lexical_admitted: bool
    semantic_only: bool
    similarity: float | None
    scored: bool
    requires_review: bool


class ScopePolicy(Protocol):
    def decision(self, topic: str, title: str, abstract: str) -> ScopeDecision: ...


_POLICY: ContextVar[ScopePolicy | None] = ContextVar("ml_semantic_scope_policy", default=None)


@contextmanager
def semantic_scope(policy: ScopePolicy):
    """Restore the previous policy on success, cancellation, or an exception."""
    token = _POLICY.set(policy)
    try:
        yield policy
    finally:
        _POLICY.reset(token)


def current_semantic_policy() -> ScopePolicy | None:
    return _POLICY.get()
