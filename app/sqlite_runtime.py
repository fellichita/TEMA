"""Use one tested SQLite DB-API driver without replacing global stdlib modules."""

import sqlite3 as _stdlib
from typing import TYPE_CHECKING

MIN_SQLITE_VERSION = (3, 53, 2)


def is_patched(version: tuple[int, ...]) -> bool:
    """Require the FTS5 security fixes as well as the earlier WAL-reset fix."""
    return version >= MIN_SQLITE_VERSION


if TYPE_CHECKING:
    import sqlite3
elif is_patched(_stdlib.sqlite_version_info):
    sqlite3 = _stdlib
else:
    try:
        import pysqlite3 as sqlite3
    except ImportError:
        raise RuntimeError("Нужен SQLite 3.53.2 или новее с исправлениями безопасности. "
                           "Соберите runtime по docs/guides/checks.md.") from None
    if not is_patched(sqlite3.sqlite_version_info):
        raise RuntimeError("SQLite в окружении не содержит необходимых исправлений безопасности. "
                           "Нужна версия 3.53.2 или новее. "
                           "Установите закреплённый runtime main2 из build/wheels.")
