#!/bin/bash
# Finder and Terminal start in different directories; always use this checkout.
TREND_ROOT="$(cd -- "$(dirname -- "$0")/../.." && pwd -P)" || exit 1
cd -- "$TREND_ROOT" || exit 1

TREND_PYTHON=""
for TREND_CANDIDATE in "$TREND_ROOT/.venv/bin/python" \
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 \
    /opt/homebrew/bin/python3.13 /usr/local/bin/python3.13 python3.13; do
    if "$TREND_CANDIDATE" -E -s -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 13))' \
        >/dev/null 2>&1; then
        TREND_PYTHON="$TREND_CANDIDATE"
        break
    fi
done

if [ -z "$TREND_PYTHON" ]; then
    printf '%s\n' 'Нужен Python 3.13 с Tkinter: установите macOS universal2 installer с python.org.' \
        'Подробности: docs/guides/macos-setup.md'
    TREND_EXIT=1
else
    "$TREND_PYTHON" -E -s "$TREND_ROOT/scripts/mac_bootstrap.py" "$@"
    TREND_EXIT=$?
fi

if [ "$TREND_EXIT" -ne 0 ] && [ -t 0 ]; then
    printf '\n%s' 'Запуск остановлен. Нажмите Enter, чтобы закрыть это окно. '
    IFS= read -r TREND_IGNORED
fi
exit "$TREND_EXIT"
