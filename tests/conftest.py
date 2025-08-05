import os

def pytest_addoption(parser):
    parser.addoption(
        "--tape-viz",
        action="store_true",
        help="Enable TapeVisualizer output during tests",
    )

def pytest_configure(config):
    if config.getoption("--tape-viz"):
        os.environ["TAPE_VIZ"] = "1"
