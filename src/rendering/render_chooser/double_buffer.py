import warnings

from src.common.double_buffer import *  # re-export everything

warnings.warn(
    "DoubleBuffer has moved to src.common.double_buffer; "
    "import from there instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [name for name in globals() if not name.startswith('_')]
