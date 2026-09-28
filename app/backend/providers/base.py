from collections.abc import Iterator
from threading import Event
from typing import Protocol

from app.backend.contracts import SearchRequest, SourcePage


class DocumentProvider(Protocol):
    def iter_pages(self, request: SearchRequest, cancel: Event) -> Iterator[SourcePage]:
        """Отдавать нормализованные порции с учётом отмены и max_results."""
        ...

    def close(self) -> None:
        ...
