from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from ..abstraction import AbstractTensor as AT
from .whiteboard_runtime import run_batched_vjp, BatchVJPResult
from .fluxspring import ParamWheel


@dataclass
class _QueuedJob:
    """Internal helper bundling a job with its residual kind."""

    job: Any
    kind: str = "main"  # either "main" or "spectral"


@dataclass
class SlotBackpropQueue:
    """Track residuals and VJP jobs per parameter slot.

    Parameters
    ----------
    wheels:
        Sequence of :class:`ParamWheel` objects whose slots will be updated
        when gradients are applied.
    """

    wheels: Sequence[ParamWheel]
    main_residuals: Dict[int, AT | None] = field(init=False)
    spectral_residuals: Dict[int, AT | None] = field(init=False)
    jobs: Dict[int, List[_QueuedJob]] = field(init=False)

    def __post_init__(self) -> None:
        slots = len(self.wheels[0].params) if self.wheels else 0
        self.main_residuals = {i: None for i in range(slots)}
        self.spectral_residuals = {i: None for i in range(slots)}
        self.jobs = {i: [] for i in range(slots)}

    # ------------------------------------------------------------------
    def add_residual(self, slot: int, *, main: AT | None = None, spectral: AT | None = None) -> None:
        """Accumulate residuals for ``slot``.

        Residuals are summed if multiple contributions arrive before the slot is
        processed.
        """

        if main is not None:
            t = AT.get_tensor(main)
            prev = self.main_residuals.get(slot)
            self.main_residuals[slot] = t if prev is None else prev + t
        if spectral is not None:
            t = AT.get_tensor(spectral)
            prev = self.spectral_residuals.get(slot)
            self.spectral_residuals[slot] = t if prev is None else prev + t

    # ------------------------------------------------------------------
    def queue_job(self, slot: int, job: Any, *, kind: str = "main") -> None:
        """Queue a JVP/VJP job for ``slot``.

        Parameters
        ----------
        slot:
            Slot index that the ``job`` is associated with.
        job:
            Any object understood by :func:`run_batched_vjp`.
        kind:
            Which residual buffer to apply when the slot is processed.
            ``"main"`` (default) uses :attr:`main_residuals`; ``"spectral"``
            uses :attr:`spectral_residuals`.
        """

        self.jobs.setdefault(slot, []).append(_QueuedJob(job, kind))

    # ------------------------------------------------------------------
    def process_slot(
        self,
        slot: int,
        *,
        sys: Any,
        lr: float = 0.01,
        run_vjp=run_batched_vjp,
    ) -> BatchVJPResult | None:
        """Drain and process jobs for ``slot`` applying gradients.

        Parameters
        ----------
        slot:
            Slot index being evicted this tick.
        sys:
            System passed through to :func:`run_batched_vjp`.
        lr:
            Learning rate used when applying gradients to parameters.
        run_vjp:
            Callable compatible with :func:`run_batched_vjp` used to compute
            gradients.  Primarily for dependency injection in tests.
        """

        qjobs = self.jobs.get(slot, [])
        if not qjobs:
            self.main_residuals[slot] = None
            self.spectral_residuals[slot] = None
            return None

        main_res = self.main_residuals.get(slot)
        spec_res = self.spectral_residuals.get(slot)
        jobs = []
        for qj in qjobs:
            if qj.kind == "spectral":
                qj.job.residual = spec_res
            else:
                qj.job.residual = main_res
            jobs.append(qj.job)

        batch = run_vjp(sys=sys, jobs=jobs)
        g_tensor = batch.grads_per_source_tensor
        if g_tensor is not None:
            g_tensor = AT.get_tensor(g_tensor)
            for idx, w in enumerate(self.wheels):
                grad = g_tensor[idx]
                p = w.params[slot]
                # Store gradient on the internal attribute expected by the
                # autograd helpers.  ``grad`` is a read-only property, so we set
                # ``_grad`` directly.
                p._grad = AT.get_tensor(grad)  # type: ignore[attr-defined]
                w.apply_slot(slot, lambda p_, g_=p._grad: p_ - lr * g_)
        self.jobs[slot] = []
        self.main_residuals[slot] = None
        self.spectral_residuals[slot] = None
        return batch
