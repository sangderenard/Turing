import pytest
from unittest.mock import Mock, MagicMock, call
from sympy import Integer, Float
import sys
import os

# Add src to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.transmogrifier.cells.cellsim.api.saline import SalinePressureAPI as SalineHydraulicSystem

class MockCell:
    def __init__(self, salinity, pressure, left=None, right=None, leftmost=None, rightmost=None, label=''):
        self.salinity = salinity
        self.pressure = pressure
        self.left = left
        self.right = right
        self.leftmost = leftmost
        self.rightmost = rightmost
        self.label = label

    def __repr__(self):
        return f"MockCell(s={self.salinity}, p={self.pressure})"

@pytest.fixture
def mock_sim():
    """Pytest fixture for a mock simulation object."""
    sim = Mock()
    sim.cells = [MockCell(10, 20), MockCell(30, 40)]
    sim.bitbuffer = MagicMock()
    sim.bitbuffer.mask_size = 100
    sim.system_lcm = 1
    sim.lcm = MagicMock(return_value=1)
    sim.bitbuffer.intceil.side_effect = lambda x, y: x 
    sim.engine = None
    return sim
