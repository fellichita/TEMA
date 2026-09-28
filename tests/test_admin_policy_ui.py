"""Saved radar policy warnings in the owner run drawer."""

import os
from pathlib import Path
import shutil
import subprocess


def test_admin_run_drawer_warns_about_old_radar_selection_policy():
    node = os.environ.get("TREND_TEST_NODE") or shutil.which("node")
    assert node, "Node.js is required for the admin run drawer simulation"
    script = Path(__file__).with_name("browser_admin_policy_check.cjs")
    result = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr or result.stdout
