"""Browser clock behaviour under delayed callbacks and system sleep."""

import os
from pathlib import Path
import shutil
import subprocess

def test_browser_clocks_keep_phase_and_resume_after_sleep():
    node = os.environ.get("TREND_TEST_NODE") or shutil.which("node")
    assert node, "Node.js is required for the browser clock simulation"
    script = Path(__file__).with_name("browser_clock_check.cjs")
    result = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr or result.stdout
