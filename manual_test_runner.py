"""Manual test runner that executes all tests with explicit output.
This script avoids using pytest's runner and prints detailed
information about each test, including parameter sets and results.
"""

import importlib
import inspect
import sys
import traceback
from pathlib import Path


BASE_DIR = Path(__file__).parent


def iter_test_modules():
    """Yield dotted module paths for all test modules."""
    # modules in tests/ directory
    tests_dir = BASE_DIR / "tests"
    if tests_dir.exists():
        for path in tests_dir.glob("test_*.py"):
            yield f"tests.{path.stem}"
    # top-level test_*.py files excluding this runner
    for path in BASE_DIR.glob("test_*.py"):
        if path.name != Path(__file__).name:
            yield path.stem


def get_param_sets(func):
    """Return a list of parameter dictionaries for a test function."""
    params = []
    marks = getattr(func, "pytestmark", [])
    for mark in marks:
        if mark.name == "parametrize":
            names = [n.strip() for n in mark.args[0].split(",")]
            for values in mark.args[1]:
                params.append(dict(zip(names, values)))
    return params or [None]


def run():
    total = passed = failed = 0
    for mod_name in iter_test_modules():
        module = importlib.import_module(mod_name)
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test"):
                for param in get_param_sets(func):
                    total += 1
                    doc = inspect.getdoc(func) or "No description"
                    print(f"\nRunning {mod_name}.{name}")
                    print(f"  Description: {doc}")
                    if param:
                        print(f"  Parameters: {param}")
                    try:
                        if param:
                            func(**param)
                        else:
                            func()
                        print("  Result: PASSED")
                        passed += 1
                    except Exception as exc:
                        print(f"  Result: FAILED -> {exc}")
                        traceback.print_exc()
                        failed += 1
    print(f"\nSummary: {total} total, {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(run())
