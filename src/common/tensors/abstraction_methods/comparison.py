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


def less(self, value: Any) -> "AbstractTensor":
    result = type(self)(track_time=self.track_time)
    result.data = self.less_(value)
    return result


def less_equal(self, value: Any) -> "AbstractTensor":
    result = type(self)(track_time=self.track_time)
    result.data = self.less_equal_(value)
    return result


def equal(self, value: Any) -> "AbstractTensor":
    result = type(self)(track_time=self.track_time)
    result.data = self.equal_(value)
    return result


def not_equal(self, value: Any) -> "AbstractTensor":
    result = type(self)(track_time=self.track_time)
    result.data = self.not_equal_(value)
    return result
