from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple
import logging

from ..abstraction import AbstractTensor as AT
from .whiteboard_runtime import run_batched_vjp, BatchVJPResult, _WBJob
from .fluxspring import ParamWheel, spiral_slot

logger = logging.getLogger(__name__)


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

        return spiral_slot(tick, row_idx, self.slots)

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

        main_shape = None
        if main is not None:
            t = AT.get_tensor(main)
            main_shape = getattr(t, "shape", None)
            prev = self.main_residuals.get(slot)
            self.main_residuals[slot] = t if prev is None else prev + t
        spectral_shape = None
        if spectral is not None:
            t = AT.get_tensor(spectral)
            spectral_shape = getattr(t, "shape", None)
            prev = self.spectral_residuals.get(slot)
            self.spectral_residuals[slot] = t if prev is None else prev + t

        logger.debug(
            "add_residual: slot=%s tick=%s row_idx=%d main_shape=%s spectral_shape=%s",
            slot,
            tick,
            row_idx,
            main_shape,
            spectral_shape,
        )

    # ------------------------------------------------------------------
    def queue_job(
        self,
        slot: int | None,
        job: _WBJob,
        *,
        tick: int | None = None,
        row_idx: int = 0,
        kind: str = "main",
        param_schema: Tuple[str, ...] | None = None,
        fn_args: Tuple[Any, ...] | None = None,
        fn_kwargs: Dict[str, Any] | None = None,
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
        param_schema:
            Optional tuple of attribute names describing the parameter layout
            for this job. If provided, it overrides ``job.param_schema``.
        fn_args:
            Optional positional arguments supplied to the job's callable.
            When provided, they replace ``job.fn_args``.
        fn_kwargs:
            Optional keyword arguments supplied to the job's callable.  When
            provided, they replace ``job.fn_kwargs``.
        """

        if not isinstance(job, _WBJob):
            raise TypeError("queue_job expects a _WBJob instance")
        new_schema = param_schema if param_schema is not None else job.param_schema
        new_args = fn_args if fn_args is not None else job.fn_args
        new_kwargs = fn_kwargs if fn_kwargs is not None else job.fn_kwargs
        if (
            new_schema != job.param_schema
            or new_args != job.fn_args
            or new_kwargs != job.fn_kwargs
        ):
            job = _WBJob(
                job.job_id,
                job.op,
                job.src_ids,
                job.residual,
                job.param_lens,
                job.fn,
                new_schema,
                new_args,
                new_kwargs,
            )

        if slot is None:
            if tick is None:
                raise ValueError("queue_job requires either slot or tick")
            slot = self._slot_for(tick=tick, row_idx=row_idx)

        if kind == "spectral":
            self.spectral_jobs.setdefault(slot, []).append(job)
        else:
            self.jobs.setdefault(slot, []).append(job)

        logger.debug(
            "queue_job: slot=%s job_id=%s kind=%s",
            slot,
            getattr(job, "job_id", None),
            kind,
        )

    # ------------------------------------------------------------------
    def process_slot(
        self,
        slot: int,
        *,
        sys: Any,
        row_slots: Sequence[int] | Dict[int, int] | None = None,
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
        row_slots:
            Optional mapping of wheel index to slot indices corresponding to
            the parameter versions used when the slot was queued.  When
            provided, gradients are applied to these slot indices instead of
            ``slot``.
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
            logger.debug("process_slot: slot=%d empty", slot)
            return None

        main_res = self.main_residuals.get(slot)
        spec_res = self.spectral_residuals.get(slot)
        logger.debug(
            "process_slot: slot=%d main_jobs=%d spec_jobs=%d main_res=%s spec_res=%s",
            slot,
            len(main_jobs),
            len(spec_jobs),
            main_res,
            spec_res,
        )
        jobs: List[_WBJob] = []
        for j in main_jobs:
            res = main_res if main_res is not None else j.residual
            logger.debug(
                "process_slot: enqueue main job=%s residual=%s", j.job_id, res
            )
            jobs.append(
                _WBJob(
                    j.job_id,
                    j.op,
                    j.src_ids,
                    res,
                    j.param_lens,
                    j.fn,
                    j.param_schema,
                    j.fn_args,
                    j.fn_kwargs,
                )
            )
        for j in spec_jobs:
            res = spec_res if spec_res is not None else j.residual
            logger.debug(
                "process_slot: enqueue spectral job=%s residual=%s", j.job_id, res
            )
            jobs.append(
                _WBJob(
                    j.job_id,
                    j.op,
                    j.src_ids,
                    res,
                    j.param_lens,
                    j.fn,
                    j.param_schema,
                    j.fn_args,
                    j.fn_kwargs,
                )
            )

        if sys is not None and getattr(sys, "nodes", None) is not None:
            for nid, node in (
                sys.nodes.items() if isinstance(sys.nodes, dict) else enumerate(sys.nodes)
            ):
                p = getattr(node, "p", None)
                if p is None:
                    continue
                logger.debug(
                    "process_slot: node_id=%s param_id=%s requires_grad=%s value=%s",
                    nid,
                    id(p),
                    getattr(p, "requires_grad", False),
                    p,
                )

        batch = run_vjp(sys=sys, jobs=jobs)
        g_tensor = batch.grads_per_source_tensor
        if g_tensor is not None:
            g_tensor = AT.get_tensor(g_tensor)
            logger.debug("process_slot: slot=%d g_tensor=%s", slot, g_tensor)
            if isinstance(row_slots, dict):
                slots_seq = [row_slots.get(i, slot) for i in range(len(self.wheels))]
            elif row_slots is None:
                slots_seq = [slot] * len(self.wheels)
            else:
                slots_seq = list(row_slots)
            for idx, w in enumerate(self.wheels):
                grad = g_tensor[idx]
                s_idx = slots_seq[idx]
                p = w.params[s_idx]
                before = AT.get_tensor(p)
                p._grad = AT.get_tensor(grad)  # type: ignore[attr-defined]
                w.apply_slot(s_idx, lambda p_, g_=p._grad: p_ - lr * g_)
                after = AT.get_tensor(w.params[s_idx])
                logger.debug(
                    "process_slot: apply idx=%d slot=%d grad=%s before=%s after=%s",
                    idx,
                    s_idx,
                    grad,
                    before,
                    after,
                )
        else:
            logger.debug(
                "process_slot: slot=%d no g_tensor queued_jobs=%d",
                slot,
                len(jobs),
            )
        self.jobs[slot] = []
        self.spectral_jobs[slot] = []
        self.main_residuals[slot] = None
        self.spectral_residuals[slot] = None
        logger.debug("process_slot: slot=%d cleared", slot)
        return batch
