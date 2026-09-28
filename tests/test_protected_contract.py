import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def protected_bytes(path: Path) -> bytes:
    """The file's content, independent of the checkout's line endings.

    The baseline was taken on LF bytes; a Windows checkout with autocrlf stores
    the identical content as CRLF. Comparing raw bytes would report that
    difference as a modified protected file.
    """
    return path.read_bytes().replace(b"\r\n", b"\n")


def test_user_protected_functions_and_regression_expectations_remain_identical():
    baseline = json.loads((ROOT / "tests/fixtures/contracts/protected-baseline.json").read_text(encoding="utf-8"))
    for key, expected in baseline["functions"].items():
        file, name = key.split(":")
        source = (ROOT / file).read_text(encoding="utf-8")
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest() == expected, key
    for file, expected in baseline["tests"].items():
        exception = baseline.get("approved_placement_migration", {}).get(file)
        if exception:
            tree = ast.parse((ROOT / file).read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in exception["allowed_functions"]:
                    node.body = [ast.Pass()]
            actual = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
            assert actual == exception["unmodified_rest_sha256"], file
        else:
            assert hashlib.sha256(protected_bytes(ROOT / file)).hexdigest() == expected, file
