"""Isolated subprocess environments for local diagnostics."""

import os


def diagnostic_environment() -> dict[str, str]:
    """Diagnostics need no API secrets or developer import path overrides."""
    excluded = {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}
    environment = {
        key: value for key, value in os.environ.items()
        if key not in excluded and not key.upper().endswith("_KEY")
        and not any(word in key.upper() for word in ("SECRET", "TOKEN", "PASSWORD"))
    }
    environment["ORT_DISABLE_TELEMETRY"] = "1"
    return environment
