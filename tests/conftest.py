import os
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
