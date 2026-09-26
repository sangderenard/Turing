"""Compatibility import for the chemistry package.

The implementation moved to :mod:`src.common.chemistry` when the tested
prototype became a reusable engine subsystem. Existing chamber examples may
keep importing this module while they migrate; there is one implementation.
"""

from src.common.chemistry import *  # noqa: F401,F403
from src.common.chemistry import __all__
