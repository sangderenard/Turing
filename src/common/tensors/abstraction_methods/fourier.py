from __future__ import annotations

from typing import Any, Optional


def fft(self, n: Optional[int] = None, axis: int = -1, norm: Optional[str] = None):
    """Return the discrete Fourier transform of ``self``.

    Records an autograd node so gradients propagate through the transform
    using the inverse FFT in the backward pass.
    """
    from ..abstraction import AbstractTensor

    finalize = AbstractTensor._pre_autograd(
        "fft", [self], params={"n": n, "axis": axis, "norm": norm}
    )
    out = self.__class__(track_time=self.track_time, tape=getattr(self, "_tape", None))
    if not hasattr(self, "fft_"):
        raise NotImplementedError(f"{self.__class__.__name__} must implement fft_()")
    out.data = self.fft_(n=n, axis=axis, norm=norm)
    return finalize(out)


def ifft(self, n: Optional[int] = None, axis: int = -1, norm: Optional[str] = None):
    """Return the inverse discrete Fourier transform of ``self``."""
    from ..abstraction import AbstractTensor

    finalize = AbstractTensor._pre_autograd(
        "ifft", [self], params={"n": n, "axis": axis, "norm": norm}
    )
    out = self.__class__(track_time=self.track_time, tape=getattr(self, "_tape", None))
    if not hasattr(self, "ifft_"):
        raise NotImplementedError(f"{self.__class__.__name__} must implement ifft_()")
    out.data = self.ifft_(n=n, axis=axis, norm=norm)
    return finalize(out)


def rfft(self, n: Optional[int] = None, axis: int = -1, norm: Optional[str] = None):
    """Return the FFT of a real-valued ``self``."""
    from ..abstraction import AbstractTensor

    finalize = AbstractTensor._pre_autograd(
        "rfft", [self], params={"n": n, "axis": axis, "norm": norm}
    )
    out = self.__class__(track_time=self.track_time, tape=getattr(self, "_tape", None))
    if not hasattr(self, "rfft_"):
        raise NotImplementedError(f"{self.__class__.__name__} must implement rfft_()")
    out.data = self.rfft_(n=n, axis=axis, norm=norm)
    return finalize(out)


def irfft(self, n: Optional[int] = None, axis: int = -1, norm: Optional[str] = None):
    """Return the inverse FFT for a real-input spectrum ``self``."""
    from ..abstraction import AbstractTensor

    finalize = AbstractTensor._pre_autograd(
        "irfft", [self], params={"n": n, "axis": axis, "norm": norm}
    )
    out = self.__class__(track_time=self.track_time, tape=getattr(self, "_tape", None))
    if not hasattr(self, "irfft_"):
        raise NotImplementedError(f"{self.__class__.__name__} must implement irfft_()")
    out.data = self.irfft_(n=n, axis=axis, norm=norm)
    return finalize(out)


@classmethod
def rfftfreq(
    cls, n: int, d: float = 1.0, *, like: Any | None = None
):
    """Return the sample frequencies for a real-input FFT."""
    from ..abstraction import AbstractTensor

    base = like if isinstance(like, cls) else cls.get_tensor(0)
    finalize = AbstractTensor._pre_autograd(
        "rfftfreq", [base] if isinstance(like, cls) else [], params={"n": n, "d": d}
    )
    out = base.__class__(track_time=getattr(base, "track_time", False), tape=getattr(base, "_tape", None))
    if not hasattr(base, "rfftfreq_"):
        raise NotImplementedError(f"{base.__class__.__name__} must implement rfftfreq_()")
    out.data = base.rfftfreq_(n, d=d)
    return finalize(out)


@classmethod
def fftfreq(
    cls, n: int, d: float = 1.0, *, like: Any | None = None
):
    """Return the sample frequencies for a complex FFT."""
    from ..abstraction import AbstractTensor

    base = like if isinstance(like, cls) else cls.get_tensor(0)
    finalize = AbstractTensor._pre_autograd(
        "fftfreq", [base] if isinstance(like, cls) else [], params={"n": n, "d": d}
    )
    out = base.__class__(track_time=getattr(base, "track_time", False), tape=getattr(base, "_tape", None))
    if not hasattr(base, "fftfreq_"):
        raise NotImplementedError(f"{base.__class__.__name__} must implement fftfreq_()")
    out.data = base.fftfreq_(n, d=d)
    return finalize(out)

