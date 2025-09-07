from __future__ import annotations

"""Utility helpers for batching whiteboard jobs.

The job batcher is intentionally lightweight: callers provide a sequence of
job-like objects and a metadata dictionary describing how to evaluate those
jobs.  When possible the batcher stacks the per-job arguments and executes a
single vectorised call.  If the jobs are incompatible for batching a sequential
fallback path is taken.

Expected ``op_meta`` keys
-------------------------
``fn``
    Callable implementing the scalar operation.  Used for the fallback path
    and as the default vectorised callable.
``vectorized_fn`` (optional)
    Callable that accepts stacked arguments.  Defaults to ``fn`` when omitted.
``sub_fns`` (optional)
    Sequence of callables to be used per job.  These are forwarded untouched to
    both the vectorised and sequential paths.  When present the length must
    match ``len(jobs)``.
``sub_fn_args`` (optional)
    Per-job argument tuples for ``sub_fns``. Length must match ``len(jobs)``
    when provided.
``sub_fn_kwargs`` (optional)
    Per-job keyword argument dicts for ``sub_fns``. Length must match
    ``len(jobs)`` when provided.

Each job is expected to provide ``args`` (tuple), ``kwargs`` (dict) and an
optional ``residual`` attribute.  ``can_batch`` performs a shallow structural
check to ensure all arguments can be stacked.  Stacking is performed using
:func:`AbstractTensor.stack` when available, with a best-effort fall back to a
Python list when stacking fails.
"""

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..abstraction import AbstractTensor


class JobBatcher:
    """Batch execution helper for whiteboard jobs."""

    @staticmethod
    def _stack(items: Sequence[Any]) -> Any:
        """Stack ``items`` along a new leading dimension using ``AbstractTensor``.

        ``AbstractTensor.stack`` is attempted first; if it fails (for example
        because the inputs are scalars or do not share a compatible backend) the
        original ``items`` sequence is returned unchanged.  This mirrors the
        relaxed stacking behaviour used elsewhere in the repository."""

        try:
            return AbstractTensor.stack(list(items), dim=0)
        except Exception:
            return list(items)

    @staticmethod
    def can_batch(jobs: Sequence[Any], op_meta: Dict[str, Any]) -> bool:
        """Return ``True`` when ``jobs`` are structurally compatible for batching.

        The check is intentionally conservative – we only verify that every job
        exposes the same number of positional arguments, the same set of keyword
        argument keys and that corresponding arguments have matching ``shape``
        attributes when present.  Residuals are considered stackable when all
        non-``None`` residuals share a common ``shape`` as well.
        """

        if not jobs:
            return False

        # Validate sub-function array lengths when provided
        sub_fns = op_meta.get("sub_fns")
        sub_fn_args = op_meta.get("sub_fn_args")
        sub_fn_kwargs = op_meta.get("sub_fn_kwargs")
        expected_len = len(jobs)
        for arr in (sub_fns, sub_fn_args, sub_fn_kwargs):
            if arr is not None and len(arr) != expected_len:
                return False

        ref_args = getattr(jobs[0], "args", ())
        ref_kwargs = getattr(jobs[0], "kwargs", {})
        n_args = len(ref_args)
        kw_keys = set(ref_kwargs.keys())

        for j in jobs[1:]:
            args = getattr(j, "args", ())
            kwargs = getattr(j, "kwargs", {})
            if len(args) != n_args or set(kwargs.keys()) != kw_keys:
                return False
            for i in range(n_args):
                s0 = getattr(ref_args[i], "shape", None)
                si = getattr(args[i], "shape", None)
                if s0 != si:
                    return False
            for k in kw_keys:
                s0 = getattr(ref_kwargs[k], "shape", None)
                si = getattr(kwargs[k], "shape", None)
                if s0 != si:
                    return False

        # Residual shapes (ignore None)
        res = [getattr(j, "residual", None) for j in jobs]
        res = [r for r in res if r is not None]
        if res:
            base = getattr(res[0], "shape", None)
            for r in res[1:]:
                if getattr(r, "shape", None) != base:
                    return False

        return True

    @classmethod
    def run_vectorized(cls, jobs: Sequence[Any], op_meta: Dict[str, Any]) -> List[Any]:
        """Execute ``jobs`` either in a vectorised batch or sequentially.

        Returns a list of per-job results.  If ``can_batch`` is ``False`` a
        sequential loop is used instead.  ``op_meta`` must provide a ``fn``
        callable used for the sequential path; ``vectorized_fn`` is preferred for
        the batched call when present.
        """

        if not jobs:
            return []

        vec_fn = op_meta.get("vectorized_fn") or op_meta.get("fn")
        base_fn = op_meta.get("fn")
        sub_fns = op_meta.get("sub_fns")
        sub_fn_args = op_meta.get("sub_fn_args")
        sub_fn_kwargs = op_meta.get("sub_fn_kwargs")

        if cls.can_batch(jobs, op_meta) and callable(vec_fn):
            # Stack positional arguments and kwargs
            num_args = len(getattr(jobs[0], "args", ()))
            batch_args = [cls._stack([getattr(j, "args", ())[i] for j in jobs]) for i in range(num_args)]

            kw_keys = getattr(jobs[0], "kwargs", {}).keys()
            batch_kwargs: Dict[str, Any] = {
                k: cls._stack([getattr(j, "kwargs", {}).get(k) for j in jobs]) for k in kw_keys
            }

            # Residuals are passed as a stacked keyword argument when present
            residuals = [getattr(j, "residual", None) for j in jobs]
            if any(r is not None for r in residuals):
                batch_kwargs["residual"] = cls._stack([
                    r if r is not None else 0 for r in residuals
                ])

            if sub_fns is not None:
                batch_kwargs["sub_fns"] = sub_fns
            if sub_fn_args is not None:
                batch_kwargs["sub_fn_args"] = sub_fn_args
            if sub_fn_kwargs is not None:
                batch_kwargs["sub_fn_kwargs"] = sub_fn_kwargs

            out = vec_fn(*batch_args, **batch_kwargs)
            return list(out) if isinstance(out, Iterable) else [out]

        # Fallback: run sequentially
        if not callable(base_fn):
            raise TypeError("op_meta['fn'] must be callable for sequential fallback")

        results: List[Any] = []
        for idx, job in enumerate(jobs):
            args = getattr(job, "args", ())
            kwargs = dict(getattr(job, "kwargs", {}))
            residual = getattr(job, "residual", None)
            if residual is not None:
                kwargs.setdefault("residual", residual)
            if sub_fns is not None:
                kwargs.setdefault("sub_fns", sub_fns[idx])
            if sub_fn_args is not None:
                kwargs.setdefault("sub_fn_args", sub_fn_args[idx])
            if sub_fn_kwargs is not None:
                kwargs.setdefault("sub_fn_kwargs", sub_fn_kwargs[idx])
            results.append(base_fn(*args, **kwargs))
        return results
