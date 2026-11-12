"""Register the Studio exposure card's authored QR in a saved photo PNG.

The client supplies the exact module grid from its existing QR authoring code.
This is registration against that fiducial, not a second QR encoder or a
brightness-based selection of chart patches.
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from skimage.feature import match_template


def _refine_corners(gray, grid, initial):
    """Fit the authored module samples through a projective square-to-image map."""
    yy, xx = np.mgrid[:33, :33]
    u = ((xx + .5) / 33).ravel()
    v = ((yy + .5) / 33).ravel()
    signal = grid.ravel().astype(np.float64)
    targets = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64)
    rows = []
    for (a, b), (x, y) in zip(targets, initial):
        rows.extend([[a, b, 1, 0, 0, 0, -a * x, -b * x],
                     [0, 0, 0, a, b, 1, -a * y, -b * y]])
    def samples(flat):
        corners = flat.reshape(4, 2)
        values = corners.ravel()
        rows = []
        for (a, b), (x, y) in zip(targets, corners):
            rows.extend([[a, b, 1, 0, 0, 0, -a * x, -b * x],
                         [0, 0, 0, a, b, 1, -a * y, -b * y]])
        try:
            h = np.linalg.solve(rows, values)
        except np.linalg.LinAlgError:
            return np.zeros_like(signal)
        den = h[6] * u + h[7] * v + 1
        px = (h[0] * u + h[1] * v + h[2]) / den
        py = (h[3] * u + h[4] * v + h[5]) / den
        return map_coordinates(gray, [py, px], order=1, mode="constant", cval=0)
    def residual(flat):
        measured = samples(flat)
        dark, light = measured[signal > 0], measured[signal == 0]
        return measured - np.where(signal > 0, dark.mean(), light.mean())
    start = np.asarray(initial, np.float64).ravel()
    fit = least_squares(residual, start, bounds=(start - 24, start + 24),
                        max_nfev=80, ftol=1e-5, xtol=1e-5)
    measured = samples(fit.x)
    quality = float(np.corrcoef(measured, signal)[0, 1])
    return fit.x.reshape(4, 2), quality


def register_calibration_qr(png: bytes, modules: list[str], *, full_card: bool = False) -> dict:
    if (not isinstance(modules, list) or len(modules) != 33
            or any(not isinstance(row, str) or len(row) != 33
                   or set(row) - {"0", "1"} for row in modules)):
        raise ValueError("invalid calibration QR module grid")
    with Image.open(BytesIO(png)) as opened:
        if opened.format != "PNG" or opened.width > 1024 or opened.height > 1024:
            raise ValueError("invalid calibration image")
        gray = np.asarray(opened.convert("L"), dtype=np.float32) / 255.0
    # This built-in calibration pose places the card's upper-right fiducial in
    # the upper-right image quadrant. The bright central ring otherwise makes
    # a stronger false template match than the noisy rendered QR itself.
    offset_x = 0 if full_card else int(gray.shape[1] * .55)
    region = (gray if full_card else
              gray[:int(gray.shape[0] * .42), offset_x:int(gray.shape[1] * .95)])
    grid = np.array([[int(bit) for bit in row] for row in modules], dtype=np.uint8)
    if not 250 <= int(grid.sum()) <= 550:
        raise ValueError("invalid calibration QR modules")
    best = None
    # The calibration camera faces the planar card. Search a bounded range of
    # projected sizes; the exact QR pattern rejects other dark scene features.
    sizes = (range(250, min(501, *region.shape) + 1, 4) if full_card else
             range(80, min(181, *region.shape) + 1, 2))
    for side in sizes:
        template = np.asarray(Image.fromarray((grid if full_card else 1 - grid) * 255).resize(
            (side, side), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        response = match_template(region, template, pad_input=False)
        y, x = np.unravel_index(np.argmax(response), response.shape)
        score = float(response[y, x])
        if best is None or score > best[0]:
            best = (score, int(x) + offset_x, int(y), side)
    if best is None or best[0] < (.45 if full_card else .25):
        raise ValueError(f"Exposure target QR was not found at readable scale (match {best[0]:.2f})")
    score, x, y, side = best
    corners = np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]], np.float64)
    if full_card:
        corners, quality = _refine_corners(gray, grid, corners)
        if not np.isfinite(quality) or quality < .45:
            raise ValueError("Spectral QR perspective fit did not verify the authored pattern")
        score = quality
    return {"corners": [{"x": float(x), "y": float(y)} for x, y in corners],
            "score": score, "method": "authored-qr-projective-v1" if full_card else "authored-qr-template-v1"}
