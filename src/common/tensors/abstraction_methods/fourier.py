from __future__ import annotations


def fft(self) -> "AbstractTensor":
    """Placeholder forward Fourier transform.

    Backends should provide an implementation. This stub simply dispatches
    to the backend via ``_apply_operator`` with the ``"fft"`` op name.
    """
    return self._apply_operator("fft", self, None)


def ifft(self) -> "AbstractTensor":
    """Placeholder inverse Fourier transform.

    Backends should provide an implementation. This stub dispatches to
    the backend via ``_apply_operator`` with the ``"ifft"`` op name.
    """
    return self._apply_operator("ifft", self, None)
