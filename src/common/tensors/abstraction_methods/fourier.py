from __future__ import annotations

from typing import Any, Optional


class FFTNamespace:
    """Namespace for FFT-related functions with backend dispatch.

    Provides callables that resolve to backend-specific underscore methods
    (e.g., ``fft_``, ``rfft_``, ``rfftfreq_``). The namespace itself is
    callable and forwards to ``fft`` so ``AbstractTensor.fft(x)`` works
    the same as ``AbstractTensor.fft.fft(x)``.
    """

    def __init__(self, tensor_cls: Any):
        self._cls = tensor_cls

    # Allow calling the namespace directly as fft(...)
    def __call__(self, x: Any, n: Optional[int] = None, axis: int = -1, norm: Optional[str] = None):
        return self.fft(x, n=n, axis=axis, norm=norm)

    # --- Complex FFTs -----------------------------------------------------
    def fft(self, x: Any, n: Optional[int] = None, axis: int = -1, norm: Optional[str] = None):
        t = self._cls.get_tensor(x)
        finalize = self._cls._pre_autograd("fft", [t], params={"n": n, "axis": axis, "norm": norm})
        out = t.__class__(track_time=t.track_time, tape=getattr(t, "_tape", None))
        if not hasattr(t, "fft_"):
            raise NotImplementedError(f"{t.__class__.__name__} must implement fft_()")
        out.data = t.fft_(n=n, axis=axis, norm=norm)
        return finalize(out)

    def ifft(self, x: Any, n: Optional[int] = None, axis: int = -1, norm: Optional[str] = None):
        t = self._cls.get_tensor(x)
        finalize = self._cls._pre_autograd("ifft", [t], params={"n": n, "axis": axis, "norm": norm})
        out = t.__class__(track_time=t.track_time, tape=getattr(t, "_tape", None))
        if not hasattr(t, "ifft_"):
            raise NotImplementedError(f"{t.__class__.__name__} must implement ifft_()")
        out.data = t.ifft_(n=n, axis=axis, norm=norm)
        return finalize(out)

    # --- Real-input FFTs --------------------------------------------------
    def rfft(self, x: Any, n: Optional[int] = None, axis: int = -1, norm: Optional[str] = None):
        t = self._cls.get_tensor(x)
        finalize = self._cls._pre_autograd("rfft", [t], params={"n": n, "axis": axis, "norm": norm})
        out = t.__class__(track_time=t.track_time, tape=getattr(t, "_tape", None))
        if not hasattr(t, "rfft_"):
            raise NotImplementedError(f"{t.__class__.__name__} must implement rfft_()")
        out.data = t.rfft_(n=n, axis=axis, norm=norm)
        return finalize(out)

    def irfft(self, x: Any, n: Optional[int] = None, axis: int = -1, norm: Optional[str] = None):
        t = self._cls.get_tensor(x)
        finalize = self._cls._pre_autograd("irfft", [t], params={"n": n, "axis": axis, "norm": norm})
        out = t.__class__(track_time=t.track_time, tape=getattr(t, "_tape", None))
        if not hasattr(t, "irfft_"):
            raise NotImplementedError(f"{t.__class__.__name__} must implement irfft_()")
        out.data = t.irfft_(n=n, axis=axis, norm=norm)
        return finalize(out)

    # --- Frequency helpers ------------------------------------------------
    def rfftfreq(self, n: int, d: float = 1.0, *, like: Any | None = None):
        base = like if isinstance(like, self._cls) else self._cls.get_tensor(0)
        finalize = self._cls._pre_autograd("rfftfreq", [base] if isinstance(like, self._cls) else [], params={"n": n, "d": d})
        out = base.__class__(track_time=getattr(base, "track_time", False), tape=getattr(base, "_tape", None))
        if not hasattr(base, "rfftfreq_"):
            raise NotImplementedError(f"{base.__class__.__name__} must implement rfftfreq_()")
        out.data = base.rfftfreq_(n, d=d)
        return finalize(out)

    def fftfreq(self, n: int, d: float = 1.0, *, like: Any | None = None):
        base = like if isinstance(like, self._cls) else self._cls.get_tensor(0)
        finalize = self._cls._pre_autograd("fftfreq", [base] if isinstance(like, self._cls) else [], params={"n": n, "d": d})
        out = base.__class__(track_time=getattr(base, "track_time", False), tape=getattr(base, "_tape", None))
        if not hasattr(base, "fftfreq_"):
            raise NotImplementedError(f"{base.__class__.__name__} must implement fftfreq_()")
        out.data = base.fftfreq_(n, d=d)
        return finalize(out)
