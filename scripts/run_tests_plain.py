"""Run test_* functions from a test file without pytest.

Usage: python scripts/run_tests_plain.py tests/test_file.py [name_substring]
Exits non-zero if any test fails. unittest.SkipTest marks a skip.
"""
import importlib.util
import sys
import traceback
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    path = Path(sys.argv[1])
    only = sys.argv[2] if len(sys.argv) > 2 else ""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    failed = 0
    for name in sorted(n for n in dir(module) if n.startswith("test_") and only in n):
        try:
            getattr(module, name)()
            print(f"PASS {name}", flush=True)
        except unittest.SkipTest as exc:
            print(f"SKIP {name}: {exc}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {name}", flush=True)
            traceback.print_exc()
    print(f"failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
