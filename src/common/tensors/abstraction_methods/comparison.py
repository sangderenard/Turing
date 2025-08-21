from __future__ import annotations

from typing import Any


def greater(self, value: Any) -> "AbstractTensor":
    result = type(self)(track_time=self.track_time)
    result.data = self.greater_(value)
    return result


def greater_equal(self, value: Any) -> "AbstractTensor":
    result = type(self)(track_time=self.track_time)
    result.data = self.greater_equal_(value)
    return result
