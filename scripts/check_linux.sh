#!/usr/bin/env bash
# Same X11 desktop for local/container checks and GitHub Actions.
set -euo pipefail
if [[ "${1:-}" != --inside-display ]]; then
  exec xvfb-run -a -s '-screen 0 1920x1200x24 -dpi 96' bash "$0" --inside-display "$@"
fi
shift
openbox --sm-disable >"${TMPDIR:-/tmp}/trendanalyser-openbox.log" 2>&1 &
manager_pid=$!
trap 'kill "$manager_pid" 2>/dev/null || true' EXIT
for attempt in {1..50}; do
  wmctrl -m >/dev/null 2>&1 && break
  sleep 0.1
done
wmctrl -m
if [[ "${1:-}" == --resize-performance ]]; then
  shift
  python -m tools.measure_resize "$@"
else
  python -m tools.run_checks --fail-on-skip \
    --exclude-test tests/test_run_web_demo.py::test_windows_children_keep_system_root_whatever_its_spelling \
    "Windows environment variable check is inapplicable on Linux" "$@"
fi
