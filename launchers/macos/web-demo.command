#!/bin/bash
set -u
TREND_ROOT="$(cd -- "$(dirname -- "$0")/../.." && pwd -P)" || exit 1
cd -- "$TREND_ROOT" || exit 1
"$TREND_ROOT/.venv/bin/python" -E -s -m scripts.run_web_demo "$@"
status=$?
if [ "$status" -ne 0 ]; then
    printf '\nЗапуск остановлен с ошибкой. Нажмите Enter, чтобы закрыть окно.\n'
    read -r _
fi
exit "$status"
