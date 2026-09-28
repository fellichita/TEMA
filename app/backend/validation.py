"""Pure input checks shared by the Python API and CLI before storage access."""

from collections.abc import Sequence

from app.backend.contracts import SearchRequest
from app.backend.errors import BackendError


def validate_pagination(limit: int, offset: int = 0) -> None:
    if type(limit) is not int or not 1 <= limit <= 1000 or type(offset) is not int or offset < 0:
        raise BackendError("invalid_pagination", "Размер страницы должен быть 1–1000, смещение — неотрицательным.")


def validate_search_query(query: str | None) -> None:
    if query is not None and (not isinstance(query, str) or len(query) > 500):
        raise BackendError("invalid_query", "Поисковая строка должна содержать не больше 500 символов.")
    if query is not None and "\x00" in query:
        # SQLite LIKE truncates its pattern at NUL, potentially turning a
        # specific search into "%". Reject it instead of changing the query.
        raise BackendError("invalid_query", "Поисковая строка не должна содержать нулевой символ.")


def collection_requests(request: SearchRequest, sources: Sequence[str]) -> tuple[SearchRequest, ...]:
    if (isinstance(sources, str) or not isinstance(sources, Sequence) or not sources
            or any(not isinstance(source, str) for source in sources)
            or len(sources) != len(set(sources))):
        raise BackendError("invalid_source", "Укажите непустой список разных источников.")
    values = SearchRequest.model_validate(request).model_dump()
    return tuple(SearchRequest.model_validate(values | {"source": source}) for source in sources)
