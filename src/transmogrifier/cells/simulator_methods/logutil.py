import logging
import sys

# Base logger for simulator methods (off by default)
logger = logging.getLogger("simulator_methods")
logger.addHandler(logging.NullHandler())

# Dedicated analysis logger (active and isolated by default)
ANALYSIS = 25
if not hasattr(logging, "ANALYSIS"):
	logging.addLevelName(ANALYSIS, "ANALYSIS")

def _analysis(self, message, *args, **kws):
	if self.isEnabledFor(ANALYSIS):
		self._log(ANALYSIS, message, args, **kws)

logging.Logger.analysis = _analysis  # type: ignore[attr-defined]

analysis_logger = logging.getLogger("simulator_methods.analysis")
analysis_logger.setLevel(ANALYSIS)
analysis_logger.propagate = False
if not analysis_logger.handlers:
	_h = logging.StreamHandler(stream=sys.stdout)
	_fmt = logging.Formatter("[%(levelname)s] %(message)s")
	_h.setFormatter(_fmt)
	analysis_logger.addHandler(_h)
