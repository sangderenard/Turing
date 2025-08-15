import os
import time
from pathlib import Path

import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--tape-viz",
        action="store_true",
        help="Enable TapeVisualizer output during tests",
    )
    parser.addoption(
        "--run-operators",
        action="store_true",
        help="Run tests that execute the operator simulation",
    )


def pytest_configure(config):
    if config.getoption("--tape-viz"):
        os.environ["TAPE_VIZ"] = "1"
    config.addinivalue_line(
        "markers",
        "operators: tests that run the operator simulation",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-operators"):
        return
    skip = pytest.mark.skip(reason="need --run-operators option to run")
    for item in items:
        if "operators" in item.keywords:
            item.add_marker(skip)


_RUN_LOG = "pytest_run_times.log"


def pytest_sessionstart(session):
    session._start_time = time.time()


def pytest_sessionfinish(session, exitstatus):
    duration = time.time() - session._start_time
    log_file = Path(session.config.rootpath) / _RUN_LOG
    history = int(os.environ.get("PYTEST_RUN_TIME_HISTORY", "50"))
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {duration:.2f}\n"
    with log_file.open("a") as fh:
        fh.write(line)
    with log_file.open("r+") as fh:
        lines = fh.readlines()
        if len(lines) > history:
            fh.seek(0)
            fh.writelines(lines[-history:])
            fh.truncate()


# ---------------------------------------------------------------------------
# Helpers for cell pressure tests
# ---------------------------------------------------------------------------


@pytest.fixture(params=[7, 11, 13, 17])
def stride(request):
    """Provide a selection of prime strides for pressure simulations."""
    return request.param
