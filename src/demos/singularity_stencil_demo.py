"""Singularity-stencil experiment: predict, then measure.

Run ``python -m src.demos.singularity_stencil_demo [out.png]``.

Panels:
1. Measured winding W(r, a) for F_a = z**2 - a with the centre removed.
   Prediction: W = 0 below the curve r = sqrt(a), W = 2 above it.
2. Noise robustness: fraction of trials still reporting W = 2 on r = 1, a = 0.3,
   against a pointwise detector (|F| < eps on a 0.5-spaced grid).
3. Deconstruction: the recursive box stencil splits the charge-2 contour into
   two charge-1 boxes at +-sqrt(a), drawn over the field phase.
"""

from __future__ import annotations

import sys

import torch

from src.common.singularity_stencil import (
    add_fields,
    decompose,
    detection_rate,
    naive_grid_zeros,
    phase_diagram,
    predicted_winding,
    smooth_noise,
    z2_minus_a,
)


def naive_rate(a: float, sigmas, trials: int = 64, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for sigma in sigmas:
        hits = 0
        for _ in range(trials):
            f = add_fields(z2_minus_a(a), smooth_noise(sigma, generator=gen))
            hits += int(naive_grid_zeros(f, extent=2.0, spacing=0.5, eps=0.02).shape[0] > 0)
        out.append(hits / trials)
    return out


def main(out_path: str = "singularity_stencil.png") -> None:
    a = torch.linspace(0.0, 1.0, 201, dtype=torch.float64)
    r = torch.linspace(0.01, 1.2, 240, dtype=torch.float64)
    w, _, step = phase_diagram(a, r, n=512)
    measured = torch.round(w)
    pred = predicted_winding(a, r)
    reliable = step < torch.pi / 2
    agree = (measured == pred) | ~reliable
    print(f"phase diagram: {int(reliable.sum())}/{reliable.numel()} reliable cells, "
          f"{int((~agree).sum())} disagree with prediction")

    sigmas = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0]
    stencil = detection_rate(0.3, 1.0, sigmas, trials=64)
    naive = naive_rate(0.3, sigmas, trials=64)
    for s, p, q in zip(sigmas, stencil, naive):
        print(f"sigma={s:4.2f}  stencil W=2: {p:5.2f}   pointwise |F|<eps: {q:5.2f}")

    dec = decompose(z2_minus_a(0.3), (-2.0, -2.0, 2.0, 2.0), min_size=1e-3)
    print(f"decomposition: total {dec.total} -> "
          + " + ".join(f"{c.winding}@({c.center[0]:+.4f},{c.center[1]:+.4f})" for c in dec.charges))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return

    ink, muted, grid = "#1f2328", "#59636e", "#d0d7de"
    blue_light, blue_dark, gray = "#cfe2ff", "#1f5fbf", "#9aa4ae"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    ax = axes[0]
    img = torch.where(reliable, measured / 2.0, torch.full_like(measured, 0.5))
    cmap = ListedColormap([blue_light, gray, blue_dark])
    ax.imshow(img.numpy(), origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1,
              extent=[float(a[0]), float(a[-1]), float(r[0]), float(r[-1])], interpolation="nearest")
    ax.plot(a.numpy(), torch.sqrt(a).numpy(), color="white", lw=2, ls="--")
    ax.text(0.62, 0.25, "W = 0", color=ink)
    ax.text(0.15, 0.95, "W = 2", color="white")
    ax.text(0.5, 0.56, "dashed: predicted r = √a", color=ink, fontsize=9)
    ax.set_xlabel("a  (zeros at ±√a)")
    ax.set_ylabel("stencil radius r")
    ax.set_title("Measured winding, centre removed", loc="left", color=ink)

    ax = axes[1]
    ax.plot(sigmas, stencil, color=blue_dark, lw=2, marker="o", ms=5, label="ring stencil reports W = 2")
    ax.plot(sigmas, naive, color="#c2410c", lw=2, marker="s", ms=5, label="grid point with |F| < 0.02")
    ax.axvline(0.7, color=muted, lw=1, ls=":")
    ax.text(0.72, 0.5, "ring margin\nmin|F| = 0.7", color=muted, fontsize=9)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("smooth-noise amplitude σ")
    ax.set_ylabel("fraction of trials")
    ax.grid(True, color=grid, lw=0.6)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Detecting the zeros of z² − 0.3", loc="left", color=ink)

    ax = axes[2]
    g = torch.linspace(-1.2, 1.2, 400, dtype=torch.float64)
    X, Y = torch.meshgrid(g, g, indexing="xy")
    fx, fy = z2_minus_a(0.3)(X, Y)
    ax.imshow(torch.atan2(fy, fx).numpy(), origin="lower", extent=[-1.2, 1.2, -1.2, 1.2],
              cmap="twilight", interpolation="bilinear")
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="white", lw=2))
    ax.text(-0.35, 1.03, "W = 2", color="white")
    for c in dec.charges:
        ax.add_patch(plt.Circle(c.center, 0.08, fill=False, color="white", lw=2))
        ax.text(c.center[0] - 0.12, c.center[1] + 0.12, f"W = {c.winding}", color="white")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Field phase: 2 deconstructs into 1 + 1", loc="left", color=ink)

    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
