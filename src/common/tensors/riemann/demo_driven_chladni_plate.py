"""Drive a parametric plate in the eigenbasis of the existing LaplaceND."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..abstraction import AbstractTensor as AT
from ..abstract_convolution.laplace_nd import GridDomain, RectangularTransform
from ..autograd import autograd
from .manifold import ManifoldPackage


def advance_modal_plate(
    displacement,
    velocity,
    eigenvalues,
    drive_projection,
    *,
    time_value: float,
    dt: float,
    drive_frequency: float,
    drive_strength: float,
    damping: float,
    frequency_scale: float,
):
    """One vectorized driven Kirchhoff-plate step in LaplaceND eigenmodes."""
    modal_frequency = frequency_scale * abs(eigenvalues)
    forcing = (
        drive_strength
        * np.sin(drive_frequency * time_value)
        * drive_projection
    )
    acceleration = (
        forcing
        - 2.0 * damping * modal_frequency * velocity
        - modal_frequency * modal_frequency * displacement
    )
    velocity = velocity + dt * acceleration
    displacement = displacement + dt * velocity
    return displacement, velocity


def build_plate(resolution: int, modes: int, backend: str):
    with AT.use_backend(backend), autograd.no_grad():
        transform = RectangularTransform(
            Lx=1.0, Ly=1.0, Lz=0.04, device="cpu"
        )
        sample = AT.tensor([1.0])
        grid = GridDomain.generate_grid_domain(
            coordinate_system="rectangular",
            N_u=resolution,
            N_v=resolution,
            N_w=2,
            Lx=1.0,
            Ly=1.0,
            Lz=0.04,
            device="cpu",
            cls=type(sample),
            precision=sample.dtype,
        )
        grid.transform = transform
        manifold = ManifoldPackage(
            transform,
            grid,
            laplace_kwargs={"boundary_conditions": ("dirichlet",) * 6},
            num_eigenpairs=modes,
        )
        manifold.build()
        eigenvalues, eigenvectors = manifold.eigenpairs()
        # eigh returns low frequency last for this negative Laplacian. Present
        # the modal state in increasing |lambda| order.
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        u = np.asarray(grid.U.tolist()).reshape(-1)
        v = np.asarray(grid.V.tolist()).reshape(-1)
        driver = np.exp(-90.0 * ((u - 0.31) ** 2 + (v - 0.43) ** 2))
        driver_tensor = AT.tensor(
            driver, dtype=eigenvectors.get_dtype(),
            device=eigenvectors.get_device(),
        )
        drive_projection = eigenvectors.swapaxes(0, 1) @ driver_tensor
        return manifold, eigenvalues, eigenvectors, drive_projection


def run_demo(
    *,
    resolution: int = 6,
    modes: int = 18,
    frames: int = 900,
    steps_per_frame: int = 4,
    dt: float = 0.002,
    damping: float = 0.012,
    drive_strength: float = 18.0,
    backend: str = "numpy",
    live: bool = False,
    output: str | Path | None = "chladni_laplace_nd.png",
):
    manifold, eigenvalues, eigenvectors, drive_projection = build_plate(
        resolution, modes, backend
    )
    with AT.use_backend(backend), autograd.no_grad():
        scale = 7.0 / max(float(abs(eigenvalues[0]).item()), 1e-9)
        modal_frequencies = scale * abs(eigenvalues)
        driven_mode = min(5, modes - 1)
        drive_frequency = float(modal_frequencies[driven_mode].item())
        q = eigenvalues * 0.0
        q_velocity = eigenvalues * 0.0
        activity = AT.zeros((resolution * resolution * 2,))

        if live or output is not None:
            import matplotlib.pyplot as plt

            figure, axes = plt.subplots(1, 3, figsize=(13, 4.7))
        for frame in range(frames):
            for substep in range(steps_per_frame):
                time_value = (frame * steps_per_frame + substep) * dt
                q, q_velocity = advance_modal_plate(
                    q,
                    q_velocity,
                    eigenvalues,
                    drive_projection,
                    time_value=time_value,
                    dt=dt,
                    drive_frequency=drive_frequency,
                    drive_strength=drive_strength,
                    damping=damping,
                    frequency_scale=scale,
                )
            displacement = eigenvectors @ q
            speed = eigenvectors @ q_velocity
            activity = 0.992 * activity + 0.008 * speed * speed

            if (live and frame % 3 == 0) or frame == frames - 1:
                displacement_2d = np.asarray(
                    displacement.tolist()
                ).reshape(resolution, resolution, 2).mean(axis=2)
                activity_2d = np.asarray(
                    activity.tolist()
                ).reshape(resolution, resolution, 2).mean(axis=2)
                nodes = np.exp(-18.0 * np.sqrt(activity_2d))
                for axis in axes:
                    axis.clear()
                axes[0].imshow(
                    displacement_2d.T, origin="lower", cmap="RdBu_r",
                    interpolation="bicubic",
                )
                axes[0].set_title("instantaneous driven plate")
                axes[1].imshow(
                    activity_2d.T, origin="lower", cmap="magma",
                    interpolation="bicubic",
                )
                axes[1].set_title("time-averaged modal activity")
                axes[2].imshow(
                    nodes.T, origin="lower", cmap="gray",
                    interpolation="bicubic",
                )
                axes[2].set_title("Chladni nodal accumulation")
                for axis in axes:
                    axis.set_xticks([])
                    axis.set_yticks([])
                figure.suptitle(
                    "Parametric GridDomain → LaplaceND stencil → eigenvectors → driven plate\n"
                    f"mode {driven_mode} · ω={drive_frequency:.3g} · "
                    f"{len(eigenvalues)} vectorized modes"
                )
                figure.tight_layout()
                if live:
                    plt.pause(0.001)
        if output is not None:
            figure.savefig(Path(output), dpi=160)
        if live:
            plt.show()
        elif output is not None:
            plt.close(figure)
        return {
            "eigenvalues": np.asarray(eigenvalues.tolist()),
            "drive_frequency": drive_frequency,
            "modal_displacement": np.asarray(q.tolist()),
            "output": None if output is None else Path(output),
            "laplace_package": manifold.laplace_package(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=6)
    parser.add_argument("--modes", type=int, default=18)
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--steps-per-frame", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--damping", type=float, default=0.012)
    parser.add_argument("--drive-strength", type=float, default=18.0)
    parser.add_argument("--backend", choices=("numpy", "c", "torch"), default="numpy")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", default="chladni_laplace_nd.png")
    args = parser.parse_args()
    result = run_demo(**vars(args))
    print(
        f"modes={len(result['eigenvalues'])} "
        f"drive_frequency={result['drive_frequency']:.6g} "
        f"output={result['output']}"
    )


if __name__ == "__main__":
    main()
