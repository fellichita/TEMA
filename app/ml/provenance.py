"""Single definition of the implementation inputs recorded in saved results."""

import hashlib
from pathlib import Path


def implementation_fingerprint():
    directory = Path(__file__).parent
    files = sorted([*directory.glob("*.py"), directory.parent / "input_safety.py"])

    def digest(value):
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    return digest(repr([(path.name, digest(path.read_text(encoding="utf-8"))) for path in files]))
