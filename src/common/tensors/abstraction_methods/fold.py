from __future__ import annotations

from typing import Tuple

from ..abstraction import AbstractTensor


def _to2(x):
    return (x, x) if isinstance(x, int) else x


def _to3(x):
    return (x, x, x) if isinstance(x, int) else x


def _zeros_from(reference: AbstractTensor, shape: tuple[int, ...]) -> AbstractTensor:
    """Allocate a different shape without losing the reference backend context."""

    result = type(reference)(
        track_time=getattr(reference, "track_time", False),
        tape=getattr(reference, "_tape", None),
    )
    if hasattr(reference, "_program"):
        result._program = reference._program
    result.data = reference.zeros_(shape)
    return result


def fold2d(
    cols: AbstractTensor,
    output_size: Tuple[int, int, int, int],
    kernel_size: Tuple[int, int] | int,
    *,
    stride: Tuple[int, int] | int = 1,
    padding: Tuple[int, int] | int = 0,
    dilation: Tuple[int, int] | int = 1,
) -> AbstractTensor:
    """Pure-AbstractTensor reference implementation of 2D fold (col2im).

    Shapes follow unfold2d/col2im conventions:
      - cols: (N, C*kH*kW, L) with L = Hout*Wout
      - output_size: (N, C, H, W)

    Implemented using only AbstractTensor ops: reshape, indexing and addition.
    """
    if hasattr(cols, "fold2d_"):
        finalize = AbstractTensor._pre_autograd(
            "fold2d",
            [cols],
            params={
                "output_size": tuple(output_size),
                "kernel_size": _to2(kernel_size),
                "stride": _to2(stride),
                "padding": _to2(padding),
                "dilation": _to2(dilation),
            },
        )
        result = type(cols)(
            track_time=getattr(cols, "track_time", False),
            tape=getattr(cols, "_tape", None),
        )
        if hasattr(cols, "_program"):
            result._program = cols._program
        result.data = cols.fold2d_(
            output_size,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        return finalize(result)

    kH, kW = _to2(kernel_size)
    sH, sW = _to2(stride)
    pH, pW = _to2(padding)
    dH, dW = _to2(dilation)

    N, C, H, W = output_size
    Hpad, Wpad = H + 2 * pH, W + 2 * pW
    eKH = (kH - 1) * dH + 1
    eKW = (kW - 1) * dW + 1
    Hout = (Hpad - eKH) // sH + 1
    Wout = (Wpad - eKW) // sW + 1

    # (N, C, kH, kW, Hout, Wout)
    cols6 = cols.reshape(N, C, kH, kW, Hout, Wout)

    ypad = _zeros_from(cols, (N, C, Hpad, Wpad))
    for i in range(kH):
        hi = i * dH
        hi_end = hi + sH * Hout
        for j in range(kW):
            wj = j * dW
            wj_end = wj + sW * Wout
            # target slice (view)
            tgt = ypad[:, :, hi:hi_end:sH, wj:wj_end:sW]
            # add contribution
            ypad[:, :, hi:hi_end:sH, wj:wj_end:sW] = tgt + cols6[:, :, i, j, :, :]

    # crop padding
    if pH or pW:
        return ypad[:, :, pH : pH + H, pW : pW + W]
    return ypad


def fold3d(
    cols: AbstractTensor,
    output_size: Tuple[int, int, int, int, int],
    kernel_size: Tuple[int, int, int] | int,
    *,
    stride: Tuple[int, int, int] | int = 1,
    padding: Tuple[int, int, int] | int = 0,
    dilation: Tuple[int, int, int] | int = 1,
) -> AbstractTensor:
    """Pure-AbstractTensor reference implementation of 3D fold (col2vol).

    Shapes:
      - cols: (N, C*kD*kH*kW, L) with L = Dout*Hout*Wout
      - output_size: (N, C, D, H, W)
    """
    kD, kH, kW = _to3(kernel_size)
    sD, sH, sW = _to3(stride)
    pD, pH, pW = _to3(padding)
    dD, dH, dW = _to3(dilation)

    N, C, D, H, W = output_size
    Dpad, Hpad, Wpad = D + 2 * pD, H + 2 * pH, W + 2 * pW
    eKD = (kD - 1) * dD + 1
    eKH = (kH - 1) * dH + 1
    eKW = (kW - 1) * dW + 1
    Dout = (Dpad - eKD) // sD + 1
    Hout = (Hpad - eKH) // sH + 1
    Wout = (Wpad - eKW) // sW + 1

    cols8 = cols.reshape(N, C, kD, kH, kW, Dout, Hout, Wout)

    ypad = _zeros_from(cols, (N, C, Dpad, Hpad, Wpad))
    for kd in range(kD):
        d0 = kd * dD
        d1 = d0 + sD * Dout
        for kh in range(kH):
            h0 = kh * dH
            h1 = h0 + sH * Hout
            for kw in range(kW):
                w0 = kw * dW
                w1 = w0 + sW * Wout
                tgt = ypad[:, :, d0:d1:sD, h0:h1:sH, w0:w1:sW]
                ypad[:, :, d0:d1:sD, h0:h1:sH, w0:w1:sW] = tgt + cols8[:, :, kd, kh, kw, :, :, :]

    if pD or pH or pW:
        return ypad[:, :, pD : pD + D, pH : pH + H, pW : pW + W]
    return ypad


def unfold2d(
    x: AbstractTensor,
    kernel_size: Tuple[int, int] | int,
    *,
    stride: Tuple[int, int] | int = 1,
    padding: Tuple[int, int] | int = 0,
    dilation: Tuple[int, int] | int = 1,
) -> AbstractTensor:
    """Pure-AbstractTensor unfold (im2col) for 2D.

    Returns (N, C*kH*kW, Hout*Wout).
    """
    if hasattr(x, "unfold2d_"):
        finalize = AbstractTensor._pre_autograd(
            "unfold2d",
            [x],
            params={
                "kernel_size": _to2(kernel_size),
                "stride": _to2(stride),
                "padding": _to2(padding),
                "dilation": _to2(dilation),
            },
        )
        result = type(x)(
            track_time=getattr(x, "track_time", False),
            tape=getattr(x, "_tape", None),
        )
        if hasattr(x, "_program"):
            result._program = x._program
        result.data = x.unfold2d_(
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        return finalize(result)

    kH, kW = _to2(kernel_size)
    sH, sW = _to2(stride)
    pH, pW = _to2(padding)
    dH, dW = _to2(dilation)

    N, C, H, W = x.shape
    Hpad, Wpad = H + 2 * pH, W + 2 * pW
    eKH = (kH - 1) * dH + 1
    eKW = (kW - 1) * dW + 1
    Hout = (Hpad - eKH) // sH + 1
    Wout = (Wpad - eKW) // sW + 1

    # Zero-pad via assignment (keeps graph alive)
    xpad = _zeros_from(x, (N, C, Hpad, Wpad))
    xpad[:, :, pH : pH + H, pW : pW + W] = x

    # Allocate cols6 and fill by slicing
    cols6 = _zeros_from(x, (N, C, kH, kW, Hout, Wout))
    for i in range(kH):
        hi = i * dH
        hi_end = hi + sH * Hout
        for j in range(kW):
            wj = j * dW
            wj_end = wj + sW * Wout
            patch = xpad[:, :, hi:hi_end:sH, wj:wj_end:sW]
            cols6[:, :, i, j, :, :] = patch

    return cols6.reshape(N, C * kH * kW, Hout * Wout)


def unfold3d(
    x: AbstractTensor,
    kernel_size: Tuple[int, int, int] | int,
    *,
    stride: Tuple[int, int, int] | int = 1,
    padding: Tuple[int, int, int] | int = 0,
    dilation: Tuple[int, int, int] | int = 1,
) -> AbstractTensor:
    """Pure-AbstractTensor unfold (im2col) for 3D.

    Returns (N, C*kD*kH*kW, Dout*Hout*Wout).
    """
    kD, kH, kW = _to3(kernel_size)
    sD, sH, sW = _to3(stride)
    pD, pH, pW = _to3(padding)
    dD, dH, dW = _to3(dilation)

    N, C, D, H, W = x.shape
    Dpad, Hpad, Wpad = D + 2 * pD, H + 2 * pH, W + 2 * pW
    eKD = (kD - 1) * dD + 1
    eKH = (kH - 1) * dH + 1
    eKW = (kW - 1) * dW + 1
    Dout = (Dpad - eKD) // sD + 1
    Hout = (Hpad - eKH) // sH + 1
    Wout = (Wpad - eKW) // sW + 1

    xpad = _zeros_from(x, (N, C, Dpad, Hpad, Wpad))
    xpad[:, :, pD : pD + D, pH : pH + H, pW : pW + W] = x

    cols8 = _zeros_from(x, (N, C, kD, kH, kW, Dout, Hout, Wout))
    for kd in range(kD):
        d0 = kd * dD
        d1 = d0 + sD * Dout
        for kh in range(kH):
            h0 = kh * dH
            h1 = h0 + sH * Hout
            for kw in range(kW):
                w0 = kw * dW
                w1 = w0 + sW * Wout
                patch = xpad[:, :, d0:d1:sD, h0:h1:sH, w0:w1:sW]
                cols8[:, :, kd, kh, kw, :, :, :] = patch

    return cols8.reshape(N, C * kD * kH * kW, Dout * Hout * Wout)
