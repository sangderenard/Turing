from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from ..abstraction import AbstractTensor as AT
from .whiteboard_runtime import run_batched_vjp, BatchVJPResult, _WBJob
from .fluxspring import ParamWheel


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
    jobs: Dict[int, List[_WBJob]] = field(init=False)
    spectral_jobs: Dict[int, List[_WBJob]] = field(init=False)
    slots: int = field(init=False)

    def __post_init__(self) -> None:
        self.slots = len(self.wheels[0].params) if self.wheels else 0
        self.main_residuals = {i: None for i in range(self.slots)}
        self.spectral_residuals = {i: None for i in range(self.slots)}
        self.jobs = {i: [] for i in range(self.slots)}
        self.spectral_jobs = {i: [] for i in range(self.slots)}

    # ------------------------------------------------------------------
    def _slot_for(self, *, tick: int, row_idx: int) -> int:
        """Return slot index for ``row_idx`` at ``tick``.

        Parameters
        ----------
        tick:
            Global tick counter.
        row_idx:
            Row index within the parameter tensor.
        """

        return (tick - row_idx) % self.slots if self.slots else 0

    # ------------------------------------------------------------------
    def add_residual(
        self,
        slot: int | None = None,
        *,
        tick: int | None = None,
        row_idx: int = 0,
        main: AT | None = None,
        spectral: AT | None = None,
    ) -> None:
        """Accumulate residuals for a slot determined by ``tick``/``row_idx``.

        Callers may supply ``slot`` directly or allow it to be computed via the
        ``tick`` and ``row_idx`` pair.  Residuals are summed if multiple
        contributions arrive before the slot is processed.
        """

        if slot is None:
            if tick is None:
                raise ValueError("add_residual requires either slot or tick")
            slot = self._slot_for(tick=tick, row_idx=row_idx)

        if main is not None:
            t = AT.get_tensor(main)
            prev = self.main_residuals.get(slot)
            self.main_residuals[slot] = t if prev is None else prev + t
        if spectral is not None:
            t = AT.get_tensor(spectral)
            prev = self.spectral_residuals.get(slot)
            self.spectral_residuals[slot] = t if prev is None else prev + t

    # ------------------------------------------------------------------
    def queue_job(
        self,
        slot: int | None,
        job: _WBJob,
        *,
        tick: int | None = None,
        row_idx: int = 0,
        kind: str = "main",
    ) -> None:
        """Queue a VJP job for a computed slot.

        Parameters
        ----------
        slot:
            Explicit slot index. If ``None``, the index is derived from
            ``tick`` and ``row_idx``.
        job:
            :class:`_WBJob` instance to run via :func:`run_batched_vjp`.
        kind:
            Which residual buffer to apply when the slot is processed.
            ``"main"`` (default) uses :attr:`main_residuals`; ``"spectral"``
            uses :attr:`spectral_residuals`.
        """

        if not isinstance(job, _WBJob):
            raise TypeError("queue_job expects a _WBJob instance")

        if slot is None:
            if tick is None:
                raise ValueError("queue_job requires either slot or tick")
            slot = self._slot_for(tick=tick, row_idx=row_idx)

        if kind == "spectral":
            self.spectral_jobs.setdefault(slot, []).append(job)
        else:
            self.jobs.setdefault(slot, []).append(job)

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

        main_jobs = self.jobs.get(slot, [])
        spec_jobs = self.spectral_jobs.get(slot, [])
        if not main_jobs and not spec_jobs:
            self.main_residuals[slot] = None
            self.spectral_residuals[slot] = None
            return None

        main_res = self.main_residuals.get(slot)
        spec_res = self.spectral_residuals.get(slot)
        jobs: List[_WBJob] = []
        for j in main_jobs:
            jobs.append(_WBJob(j.job_id, j.op, j.src_ids, main_res, j.param_lens, j.fn))
        for j in spec_jobs:
            jobs.append(_WBJob(j.job_id, j.op, j.src_ids, spec_res, j.param_lens, j.fn))

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
        self.spectral_jobs[slot] = []
        self.main_residuals[slot] = None
        self.spectral_residuals[slot] = None
        return batch
