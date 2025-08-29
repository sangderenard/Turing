"""FusedProgram IR and runner.

This module implements the initial scaffolding for the unified program
intermediate representation described in ``docs/FUSED_PROGRAM_IR.md``.
The IR captures a linear sequence of tensor operations detached from the
Autograd tape and can be replayed deterministically with a ``training``
flag to alter mode sensitive operators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set, Optional, Tuple
import inspect
import difflib

import networkx as nx

from ..abstraction import AbstractTensor as AT
from ..graph_translator import GraphTranslator
from ....transmogrifier.ilpscheduler import ILPScheduler
from ..autograd import autograd
from .optimizer import Adam, adam_step
from collections import deque
import sys
import ast
import random


# ---------------------------------------------------------------------------
# Dataclasses mirroring the design document
# ---------------------------------------------------------------------------


@dataclass
class Meta:
    """Per-id snapshot of tensor metadata."""

    shape: Iterable[int] | None = None
    dtype: str | None = None
    device: str | None = None


@dataclass
class OpStep:
    """Single linearised tensor operation."""

    step_id: int
    op_name: str
    input_ids: List[int]
    attrs: Dict[str, Any] = field(default_factory=dict)
    result_id: int = -1
    mode_sensitive: bool = False
    level: Optional[int] = None


@dataclass
class FusedProgram:
    """Unified program representation for AbstractTensor graphs."""

    version: int
    feeds: Set[int]
    steps: List[OpStep]
    outputs: Dict[str, int]
    state_in: Set[int] | None = None
    meta: Dict[int, Meta] | None = None
    # Optional map of additional outputs (e.g., updated params for training)
    extras: Dict[str, int] | None = None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_fused_program(
    graph: nx.DiGraph,
    *,
    outputs: Dict[str, int] | None = None,
    version: int = 1,
    scheduler_cls: type[ILPScheduler] = ILPScheduler,
) -> FusedProgram:
    """Construct a :class:`FusedProgram` from ``graph``.

    The ``graph`` is expected to contain ``tensor`` and ``op`` nodes with the
    following minimal attributes:

    - tensor nodes: ``kind='tensor'``, optional ``shape``, ``dtype`` and
      ``device``.
    - op nodes: ``kind='op'``, ``op_name`` (method on ``AbstractTensor``),
      optional ``attrs`` dict and ``mode_sensitive`` bool.

    Feeds are inferred as tensor nodes with no producing op predecessors.
    Steps are linearised according to ASAP levels from ``ILPScheduler``.
    """

    outputs = outputs or {}

    translator = GraphTranslator(graph)
    order = translator.schedule(scheduler_cls)

    feeds: Set[int] = set()
    steps: List[OpStep] = []
    meta: Dict[int, Meta] = {}

    for nid, data in graph.nodes(data=True):
        if data.get("kind") == "tensor":
            pred_ops = [p for p in graph.predecessors(nid) if graph.nodes[p].get("kind") == "op"]
            if not pred_ops:
                if isinstance(nid, int):
                    feeds.add(nid)
            m = Meta(
                shape=tuple(data.get("shape", [])) or None,
                dtype=data.get("dtype"),
                device=data.get("device"),
            )
            if any(v is not None for v in (m.shape, m.dtype, m.device)):
                if isinstance(nid, int):
                    meta[nid] = m

    sid_counter = 0
    for nid in order:
        data = graph.nodes[nid]
        if data.get("kind") != "op":
            continue

        ctx = data.get("ctx", {}) or {}

        # Prefer ordered inputs from the autograd ctx when available
        ctx_inputs = ctx.get("inputs")
        if ctx_inputs is not None:
            try:
                input_ids = [int(id(x)) for x in ctx_inputs]
            except Exception:
                # Fallback to graph predecessors if ctx is not materialized
                input_ids = [
                    int(tid)
                    for tid in graph.predecessors(nid)
                    if graph.nodes[tid].get("kind") == "tensor"
                ]
        else:
            input_ids = [
                int(tid)
                for tid in graph.predecessors(nid)
                if graph.nodes[tid].get("kind") == "tensor"
            ]

        # Prefer result id from ctx when available
        ctx_res = ctx.get("result")
        if ctx_res is not None:
            try:
                result_id = int(id(ctx_res))
            except Exception:
                result_id = nid
        else:
            result_candidates = [
                int(tid)
                for tid in graph.successors(nid)
                if graph.nodes[tid].get("kind") == "tensor"
            ]
            result_id = result_candidates[0] if result_candidates else nid

        # Op name and attributes (support autograd graph fields)
        op_name = data.get("op_name") or data.get("op")
        if not op_name:
            raise ValueError(f"Missing op name for node {nid}")
        attrs = data.get("attrs") or (ctx.get("params") or {})

        step = OpStep(
            step_id=sid_counter,
            op_name=str(op_name),
            input_ids=input_ids,
            attrs=dict(attrs),
            result_id=result_id,
            mode_sensitive=bool(data.get("mode_sensitive", False)),
            level=data.get("level"),
        )
        steps.append(step)
        sid_counter += 1

    return FusedProgram(
        version=version,
        feeds=feeds,
        steps=steps,
        outputs=outputs,
        meta=meta or None,
    )


# ---------------------------------------------------------------------------
# Program runner
# ---------------------------------------------------------------------------


class ProgramRunner:
    """Execute a :class:`FusedProgram` using ``AbstractTensor`` ops."""

    def __init__(self, program: FusedProgram) -> None:
        self.program = program

    def __call__(
        self,
        feeds: Dict[int, AT],
        *,
        training: bool = False,
    ) -> Dict[str, AT]:
        prog = self.program
        store: Dict[int, Any] = {}

        missing = prog.feeds - set(feeds)
        if missing:
            raise KeyError(f"Missing feeds: {sorted(missing)}")
        store.update(feeds)

        # Validate feed invariants: all AbstractTensor feeds must be attached to current tape
        # so that strict connectivity checks see a single per-step tape.
        mismatched: Dict[int, Any] = {}
        for k, v in list(store.items()):
            if hasattr(v, "__dict__") and hasattr(v, "data"):
                vt = getattr(v, "_tape", None)
                if vt is not None and vt is not autograd.tape:
                    mismatched[k] = v
        if mismatched:
            ids = ", ".join(str(k) for k in mismatched.keys())
            raise RuntimeError(
                (
                    "ProgramRunner: feed tensors are attached to a different tape than the current training tape.\n"
                    f"Feed ids on mismatched tapes: [{ids}]\n"
                    "Recapture with current inputs+targets before running, or run eager mode."
                )
            )

        # Default execution class (for creation ops without inputs)
        default_cls: Optional[type] = None
        if store:
            try:
                any_val = next(iter(store.values()))
                default_cls = any_val.__class__
            except Exception:
                default_cls = None

        # Neighbor context maps for diagnostics
        producers: Dict[int, OpStep] = {}
        consumers: Dict[int, List[OpStep]] = {}
        for st in prog.steps:
            producers[st.result_id] = st
            for iid in st.input_ids:
                consumers.setdefault(iid, []).append(st)

        # Preflight: identify dangling inputs (not produced and not in feeds)
        produced: Set[int] = set(producers.keys())
        known: Set[int] = set(prog.feeds) | produced
        dangling: Dict[int, List[int]] = {}
        for st in prog.steps:
            miss = [i for i in st.input_ids if i not in known]
            if miss:
                dangling[st.step_id] = miss
        if dangling:
            details = "; ".join(
                f"step {sid}({producers.get(sid).op_name if producers.get(sid) else '?'}) -> inputs {miss}"
                for sid, miss in dangling.items()
            )
            raise ValueError(
                f"Program has dangling inputs (no feed or producer): {details}"
            )

        for step in prog.steps:
            try:
                args = [store[i] for i in step.input_ids]
            except KeyError as ke:
                missing_inputs = [i for i in step.input_ids if i not in store]
                raise KeyError(
                    f"Step {step.step_id} ({step.op_name}) missing inputs: {missing_inputs}"
                ) from ke

            cls = args[0].__class__ if args else (default_cls or AT)
            fn = getattr(cls, step.op_name, None)
            if fn is None:
                # Suggest close method names
                candidates = [n for n in dir(cls) if not n.startswith("_")]
                sugg = difflib.get_close_matches(step.op_name, candidates, n=5, cutoff=0.6)
                raise AttributeError(
                    (
                        f"Step {step.step_id} op '{step.op_name}' not found on class {cls.__name__}. "
                        f"Level={step.level}. Close matches: {sugg}"
                    )
                )

            # Build call kwargs with optional training flag if accepted
            call_kwargs = dict(step.attrs or {})
            if step.mode_sensitive:
                try:
                    sig = inspect.signature(fn)
                    params = list(sig.parameters.values())
                    # discard implicit self if present
                    names = [p.name for p in params[1:]] if (params and params[0].name == 'self') else [p.name for p in params]
                    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
                    alias = None
                    for cand in ("training", "train", "is_training", "in_training"):
                        if cand in names:
                            alias = cand
                            break
                    if alias is not None:
                        call_kwargs[alias] = training
                    elif accepts_var_kw:
                        call_kwargs["training"] = training
                except Exception:
                    # Best effort: only pass if common name
                    call_kwargs["training"] = training

            # Validate attribute names against callable signature
            try:
                sig = inspect.signature(fn)
                params = list(sig.parameters.values())
                param_names = [p.name for p in params]
                accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
                unknown = [k for k in call_kwargs.keys() if k not in param_names]
                if unknown and not accepts_var_kw:
                    # Suggest close matches per key
                    sugg = {k: difflib.get_close_matches(k, param_names, n=3, cutoff=0.6) for k in unknown}
                    raise TypeError(
                        (
                            f"Step {step.step_id} ({step.op_name}) on {cls.__name__} does not accept attrs {unknown}. "
                            f"Level={step.level}. Suggestions: {sugg}. Signature: {sig}"
                        )
                    )
            except ValueError:
                # Builtins or C-accelerated functions without signatures: skip
                pass

            try:
                result = fn(*args, **call_kwargs)
            except Exception as e:
                # Gather neighbor context for diagnostics
                prev_producers = [producers.get(i).step_id for i in step.input_ids if producers.get(i)]
                next_consumers = [s.step_id for s in consumers.get(step.result_id, [])]

                # Summarize inputs for deep diagnostics
                def _summ(x: Any) -> Tuple[str, Any, Any, Any, Any]:
                    clsn = getattr(x, '__class__', type(x)).__name__
                    shape = getattr(x, 'shape', None)
                    dtype = getattr(x, 'dtype', None)
                    device = getattr(x, 'device', None)
                    backend = getattr(x, '_backend', getattr(x, '__module__', None))
                    return clsn, shape, dtype, device, backend

                input_summ = {
                    i: _summ(store.get(i)) for i in step.input_ids
                }
                # Backend class consistency check
                backend_classes = {getattr(v, '__class__', type(v)) for v in store.values() if v is not None}
                mixed_backends = len(backend_classes) > 1

                raise RuntimeError(
                    (
                        f"Execution failed at step {step.step_id} ({step.op_name}) level={step.level} on class {cls.__name__}.\n"
                        f"Inputs: ids={step.input_ids} producers={prev_producers}\n"
                        f"Result: id={step.result_id} consumers={next_consumers}\n"
                        f"Attrs={step.attrs}; mode_sensitive={step.mode_sensitive}; training={training}\n"
                        f"Input summaries: {input_summ}\n"
                        f"Mixed backend classes across store={mixed_backends}. Error: {e}"
                    )
                ) from e

            store[step.result_id] = result

        # Ensure all requested outputs are present
        missing_out = {name: i for name, i in prog.outputs.items() if i not in store}
        if missing_out:
            raise KeyError(f"Missing outputs: {missing_out}")
        return {name: store[i] for name, i in prog.outputs.items()}


__all__ = [
    "Meta",
    "OpStep",
    "FusedProgram",
    "build_fused_program",
    "ProgramRunner",
    "IRGraphedModel",
]


# ---------------------------------------------------------------------------
# IRGraphedModel – thin wrapper for capture + run
# ---------------------------------------------------------------------------


class IRGraphedModel:
    """Effortless wrapper that captures a model into a FusedProgram and exposes
    train(), infer(), and config() helpers.

    Notes
    -----
    - Designed for models built on the AbstractTensor surface so the autograd
      tape records operator names and inputs/outputs for building the program.
    - Pure PyTorch/NumPy models without AbstractTensor wrappers are currently
      not captured by this path; a conversion layer is planned.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.loss_fn = None  # set in configure
        self.optimizer_cls = Adam
        self.lr = 1e-2
        self.epochs = 0
        self.program: Optional[FusedProgram] = None
        self.runner: Optional[ProgramRunner] = None
        self.feed_store: Dict[int, Any] = {}
        self.outputs: Dict[str, int] = {}
        self.inputs_any: Any = None
        self.targets_any: Any = None
        self.input_feed_ids: List[int] = []  # feed ids considered external inputs (not params)
        # Training material sources
        self.training_data_hook: Optional[callable] = None
        self.training_queue: deque[tuple[Any, Any]] = deque()
        self.stdin_parser: Optional[callable] = None  # (bytes) -> (inputs, targets) or None
        self.stdin_blocking: bool = True
        # Interactive discriminator and recycling policy
        self.interactive: bool = True
        self.interactive_encoding: str = "utf-8"
        self.recycle_enabled: bool = True
        self.queue_maxlen: int = 256
        self.max_sample_age: int = 1000  # steps
        # Optimizer state (explicit, recordable): per-parameter m, v and scalar t
        self.opt_m: List[Any] = []
        self.opt_v: List[Any] = []
        self.opt_t: Any | None = None

    # ------------------------------ user surface -----------------------------
    def config(self, **kwargs) -> "IRGraphedModel":
        """Configure loss/optimizer/epochs and other knobs.

        Recognized keys: loss_fn, optimizer (class), lr (float), epochs (int).
        Unknown keys are ignored for forward compatibility.
        """
        self.loss_fn = kwargs.get("loss_fn", self.loss_fn)
        self.optimizer_cls = kwargs.get("optimizer", self.optimizer_cls)
        self.lr = float(kwargs.get("lr", self.lr))
        self.epochs = int(kwargs.get("epochs", self.epochs))
        # Optional data sources
        if "training_data_hook" in kwargs:
            self.training_data_hook = kwargs["training_data_hook"]
        if "stdin_parser" in kwargs:
            self.stdin_parser = kwargs["stdin_parser"]
        if "stdin_blocking" in kwargs:
            self.stdin_blocking = bool(kwargs["stdin_blocking"])
        if "interactive" in kwargs:
            self.interactive = bool(kwargs["interactive"])
        if "interactive_encoding" in kwargs:
            self.interactive_encoding = str(kwargs["interactive_encoding"])
        if "recycle" in kwargs:
            self.recycle_enabled = bool(kwargs["recycle"])
        if "queue_maxlen" in kwargs:
            self.queue_maxlen = int(kwargs["queue_maxlen"])
        if "max_sample_age" in kwargs:
            self.max_sample_age = int(kwargs["max_sample_age"])
        return self

    # ---- training material management ---------------------------------
    def set_data_hook(self, hook: callable) -> "IRGraphedModel":
        self.training_data_hook = hook
        return self

    def enqueue_training_sample(self, inputs: Any, targets: Any, *, step: Optional[int] = None) -> None:
        meta_step = int(step) if step is not None else -1
        self.training_queue.append((inputs, targets, meta_step))
        # Enforce max length by ejecting oldest
        while len(self.training_queue) > max(1, int(self.queue_maxlen)):
            try:
                self.training_queue.popleft()
            except Exception:
                break

    def set_stdin_parser(self, parser: callable) -> "IRGraphedModel":
        self.stdin_parser = parser
        return self

    def _default_discriminator(self, inputs: Any) -> Any:
        """Fallback target builder: run model and threshold outputs to {0,1}.

        Produces a same-shaped tensor of 0/1 floats, suitable for MSE.
        """
        fwd = getattr(self.model, "forward", None)
        model_callable = fwd if callable(fwd) else self.model
        pred = model_callable(inputs)
        # Threshold at 0.0 -> 0/1
        try:
            zeros = pred.zeros_like()
            ones = pred.ones_like()
            targets = pred.greater_(0.0)
            targets = targets.where_(targets, ones, zeros) if hasattr(targets, 'where_') else (targets * (ones - zeros) + zeros)
            return targets
        except Exception:
            # best effort: use comparison and cast via get_tensor
            like = AT.get_tensor(pred)
            try:
                mask = (pred > 0)
            except Exception:
                mask = 0
            return like.ensure_tensor(mask)

    # ---- utf-8 interactive discriminator --------------------------------
    def _tensor_to_bytes(self, t: Any) -> bytes:
        obj = getattr(t, "data", t)
        # Try tolist-like
        try:
            arr = obj.tolist() if hasattr(obj, "tolist") else obj
        except Exception:
            arr = obj
        # Flatten into 1-D list
        def _flatten(x):
            if isinstance(x, (list, tuple)):
                for v in x:
                    yield from _flatten(v)
            else:
                yield x
        flat = list(_flatten(arr))
        # Coerce numeric to 0..255 ints
        out = bytearray()
        for v in flat:
            try:
                if hasattr(v, "item"):
                    v = v.item()
                ival = int(round(float(v)))
                ival = max(0, min(255, ival))
            except Exception:
                ival = 0
            out.append(ival)
        return bytes(out)

    def _bytes_to_tensor(self, b: bytes, like: Any) -> Any:
        vals = list(b)
        # Use default backend selection; caller may convert as needed
        return AT.get_tensor(vals)

    def _decode_text(self, pred: Any) -> str:
        data = self._tensor_to_bytes(pred)
        try:
            return data.decode(self.interactive_encoding, errors="replace")
        except Exception:
            return data.decode("utf-8", errors="replace")

    def _encode_text(self, text: str, like: Any) -> Any:
        b = text.encode(self.interactive_encoding, errors="ignore")
        tgt = self._bytes_to_tensor(b, like)
        # Match length: pad/truncate to like length when available
        try:
            L_like = int(getattr(like, "numel", lambda: None)() or len(self._tensor_to_bytes(like)))
        except Exception:
            L_like = None
        if L_like is not None:
            cur = list(b)
            if len(cur) < L_like:
                cur = cur + [0] * (L_like - len(cur))
            elif len(cur) > L_like:
                cur = cur[:L_like]
            tgt = AT.get_tensor(cur)
        return tgt

    def _interactive_prompt_pair(self, base_inputs: Any | None) -> Optional[tuple[Any, Any]]:
        """Show model output as text and ask user for expected response.

        Returns a single (inputs, targets) pair or None if prompt aborted.
        """
        try:
            inputs = base_inputs if base_inputs is not None else self._random_like(self.inputs_any)
        except Exception:
            inputs = base_inputs
        if inputs is None:
            return None
        # Prefer fused runner output if available; fallback to direct forward
        pred = None
        if self.runner is not None and self.program is not None and self.input_feed_ids:
            try:
                feeds = dict(self.feed_store)
                for fid in self.input_feed_ids:
                    feeds[fid] = inputs
                out = self.runner(feeds, training=False)
                pred = out.get("pred", next(iter(out.values()))) if out else None
            except Exception:
                pred = None
        if pred is None:
            try:
                pred = getattr(self.model, "forward", self.model)(inputs)
            except Exception:
                return None
        text = self._decode_text(pred)
        try:
            sys.stdout.write("\nModel output (decoded):\n")
            sys.stdout.write(text + "\n")
            sys.stdout.write("Enter expected output (utf-8), then newline:\n> ")
            sys.stdout.flush()
        except Exception:
            pass
        line = sys.stdin.readline()
        if not line:
            return None
        expected = line.rstrip("\n")
        tgt = self._encode_text(expected, pred)
        return (inputs, tgt)

    def _random_like(self, ref: Any, batch: int | None = None) -> Any:
        """Create a random tensor shaped like ``ref`` (optionally override batch)."""
        like = AT.get_tensor(ref) if ref is not None else AT.get_tensor()
        shape = getattr(ref, "shape", None)
        if not shape:
            # default to vector of length 8
            data = [random.random() for _ in range(8)]
            return like.ensure_tensor(data)
        shape = list(shape)
        if batch is not None and len(shape) >= 1:
            shape[0] = batch
        # build nested Python lists with uniform [0,1)
        def _build(sz):
            if not sz:
                return random.random()
            return [_build(sz[1:]) for _ in range(sz[0])]
        data = _build(shape)
        return like.ensure_tensor(data)

    def _read_stdin_pair(self) -> Optional[tuple[Any, Any]]:
        """Blocking or non-blocking read of one training pair from stdin.

        Format options supported by default parser:
        - Two python literals separated by '||' (e.g., "[... ] || [...]")
        - A single python literal that is a 2-tuple/list: "([..], [..])"
        - If only inputs are provided, targets are built by _default_discriminator
        """
        # Non-blocking mode cannot be reliably implemented portably; we keep blocking by default
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if self.stdin_parser:
            try:
                pair = self.stdin_parser(line)
                if pair is not None:
                    return pair
            except Exception:
                pass
        try:
            s = line.decode("utf-8", errors="ignore").strip()
            if "||" in s:
                a, b = s.split("||", 1)
                inp = ast.literal_eval(a.strip())
                tgt = ast.literal_eval(b.strip())
                inp_t = AT.get_tensor(inp)
                tgt_t = AT.get_tensor(tgt)
                return (inp_t, tgt_t)
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)) and len(obj) == 2:
                inp_t = AT.get_tensor(obj[0])
                tgt_t = AT.get_tensor(obj[1])
                return (inp_t, tgt_t)
            # Inputs only; produce targets via discriminator
            inp_t = AT.get_tensor(obj)
            tgt_t = self._default_discriminator(inp_t)
            return (inp_t, tgt_t)
        except Exception:
            return None

    def _next_training_pair(self, step: int) -> Optional[tuple[Any, Any]]:
        # 1) Hook
        if self.training_data_hook is not None:
            try:
                pair = self.training_data_hook(step)
                if pair is not None:
                    return pair
            except Exception:
                pass
        # 2) Queue (recycled material) with freshness policy
        if self.training_queue and self.recycle_enabled:
            try:
                # Eject stale
                fresh: deque = deque()
                while self.training_queue:
                    item = self.training_queue.popleft()
                    if not isinstance(item, tuple):
                        continue
                    if len(item) == 3:
                        x, y, t = item
                        if t >= 0 and self.max_sample_age > 0 and (step - t) > self.max_sample_age:
                            continue  # drop stale
                        fresh.append((x, y, t))
                    elif len(item) >= 2:
                        fresh.append((item[0], item[1], step))
                # Refill up to capacity
                self.training_queue.extend(fresh)
                return self.training_queue.popleft() if self.training_queue else None
            except Exception:
                pass
        # 3) Interactive discriminator: show decoded output and ask for expected
        if self.interactive:
            pair = self._interactive_prompt_pair(self.inputs_any)
            if pair is not None:
                # Recycle by default: enqueue with current step
                try:
                    self.enqueue_training_sample(pair[0], pair[1], step=step)
                except Exception:
                    pass
                return pair
        # 4) Default discriminator on random input (if we have a captured input to mimic shape)
        if self.inputs_any is not None:
            try:
                rnd = self._random_like(self.inputs_any)
                tgt = self._default_discriminator(rnd)
                return (rnd, tgt)
            except Exception:
                pass
        # 5) Stdin (blocking)
        return self._read_stdin_pair()

    @staticmethod
    def _mse(a, b):
        try:
            return ((a - b) ** 2).mean()
        except Exception as e:
            raise RuntimeError(f"Default MSE loss failed: {e}") from e

    def capture(self, inputs: Any, targets: Any | None = None, *, outputs: Optional[Dict[str, int]] = None) -> "IRGraphedModel":
        """Run a single forward (and optional loss) to build a FusedProgram.

        - Resets the autograd tape.
        - Executes model forward with supplied inputs.
        - If targets are provided (and loss_fn configured), computes loss and
          marks it on the tape for better diagnostics.
        - Builds a FusedProgram and a ProgramRunner, and captures feed values.
        """
        # Reset tape for a clean capture
        autograd.tape = autograd.__class__().tape
        self.inputs_any = inputs
        self.targets_any = targets

        # Forward
        fwd = getattr(self.model, "forward", None)
        model_callable = fwd if callable(fwd) else self.model
        if not callable(model_callable):
            raise TypeError("Model is not callable and has no forward() method")

        try:
            pred = model_callable(inputs)
        except Exception as e:
            raise RuntimeError(f"Model forward failed: {e}") from e

        # Loss (optional)
        loss = None
        if targets is not None:
            lf = self.loss_fn or self._mse
            try:
                loss = lf(pred, targets)
            except Exception as e:
                raise RuntimeError(f"Loss computation failed: {e}") from e
            try:
                autograd.tape.mark_loss(loss)
            except Exception:
                pass

            # Use autograd.grad to get parameter gradients (recorded forward already exists)
            params: List[Any] = []
            try:
                if hasattr(self.model, "parameters"):
                    params = list(self.model.parameters())
            except Exception:
                params = []
            if params:
                try:
                    grads = autograd.grad(loss, params, retain_graph=True)
                except Exception as e:
                    raise RuntimeError(f"autograd.grad failed during capture: {e}") from e
            else:
                grads = []

        # If we have gradients, also record a single optimizer update (functional, recordable)
        extras: Dict[str, int] = {}
        if targets is not None and grads:
            # Prepare optimizer state as feeds so the program accepts them externally
            if not self.opt_m or len(self.opt_m) != len(params):
                self.opt_m = [p.zeros_like() for p in params]
            if not self.opt_v or len(self.opt_v) != len(params):
                self.opt_v = [p.zeros_like() for p in params]
            if self.opt_t is None:
                self.opt_t = AT.get_tensor(0.0)
            # Register state tensors on current tape as feeds
            for s in self.opt_m + self.opt_v + [self.opt_t]:
                try:
                    autograd.tape.create_tensor_node(s)
                except Exception:
                    pass
            # Record adam updates per param
            new_params: List[Any] = []
            new_m: List[Any] = []
            new_v: List[Any] = []
            # Shared t across all params
            t_new_any = None
            for i, (p, g, m, v) in enumerate(zip(params, grads, self.opt_m, self.opt_v)):
                if p is None or g is None:
                    new_params.append(p); new_m.append(m); new_v.append(v)
                    continue
                p_new, m_new, v_new, t_new = adam_step(p, g, m, v, self.opt_t, lr=self.lr)
                new_params.append(p_new)
                new_m.append(m_new)
                new_v.append(v_new)
                t_new_any = t_new
                extras[f"param{ i }_new"] = id(p_new)
                extras[f"opt_m{ i }_new"] = id(m_new)
                extras[f"opt_v{ i }_new"] = id(v_new)
            if t_new_any is not None:
                extras["opt_t_new"] = id(t_new_any)

        # Build program from the full autograd tape graph (includes backward/optimizer ops if any)
        try:
            g = autograd.tape.graph
        except Exception as e:
            raise RuntimeError(f"Failed to export graph from tape: {e}") from e

        # Determine default outputs if not provided
        out_map: Dict[str, int] = {}
        try:
            pred_id = id(pred)
            out_map["pred"] = pred_id
        except Exception:
            pass
        if loss is not None:
            try:
                out_map["loss"] = id(loss)
            except Exception:
                pass
        # Updated parameters from optimizer (if any)
        if extras:
            out_map.update(extras)
        if outputs:
            out_map.update(outputs)
        self.outputs = out_map

        prog = build_fused_program(g, outputs=out_map)
        self.program = prog
        self.runner = ProgramRunner(prog)

        # Identify input feeds (exclude parameters/stateful)
        try:
            param_ids = set(getattr(autograd.tape, "_parameters", {}).keys())
            self.input_feed_ids = sorted([fid for fid in prog.feeds if fid not in param_ids])
        except Exception:
            self.input_feed_ids = sorted(list(prog.feeds))

        # Bind feed values from tape refs for quick replay
        # Prefer tape tensor refs when available; fallback to id->object via graph
        feed_vals: Dict[int, Any] = {}
        refs = getattr(autograd.tape, "_tensor_refs", {})
        for fid in prog.feeds:
            val = refs.get(fid)
            if val is None:
                # best effort: try to reconstruct from model/inputs/targets ids
                if id(inputs) == fid:
                    val = inputs
                elif targets is not None and id(targets) == fid:
                    val = targets
                elif self.opt_t is not None and id(self.opt_t) == fid:
                    val = self.opt_t
                else:
                    # search in opt state arrays
                    for s in (self.opt_m + self.opt_v):
                        if id(s) == fid:
                            val = s
                            break
            if val is None:
                # leave missing; ProgramRunner will emit precise diagnostics
                continue
            feed_vals[fid] = val
        self.feed_store = feed_vals
        return self

    # Exhaustive capture over a sequence of forced predicate outcomes.
    # plans: list of sequences, each sequence is list[(op_name, bool)] that will be forced in order.
    def capture_exhaustive(self, inputs: Any, targets: Any | None, plans: List[List[Tuple[str, bool]]]) -> List[FusedProgram]:
        from ..branch_oracle import BRANCH_ORACLE
        progs: List[FusedProgram] = []
        saved = []
        for seq in plans:
            autograd.tape = autograd.__class__().tape
            try:
                BRANCH_ORACLE.reset()
                BRANCH_ORACLE.force_sequence(seq)
            except Exception:
                pass
            self.capture(inputs, targets)
            if self.program is not None:
                progs.append(self.program)
                saved.append((self.program, dict(self.outputs)))
        # Restore oracle
        try:
            BRANCH_ORACLE.reset()
        except Exception:
            pass
        return progs

    # ------------------------------ execution API ----------------------------
    def infer(self, **kwargs) -> Dict[str, Any]:
        """Replay the program for inference.

        kwargs:
          - feeds: optional dict[id->value] to override defaults
          - training: bool flag (defaults to False)
        """
        if self.runner is None or self.program is None:
            raise RuntimeError("Program is not captured; call capture() first")
        feeds = dict(self.feed_store)
        # Convenience: allow direct 'inputs=' override, mapped to input feed ids
        if "inputs" in kwargs and self.input_feed_ids:
            inp = kwargs["inputs"]
            for fid in self.input_feed_ids:
                feeds[fid] = inp
        feeds.update(kwargs.get("feeds", {}))
        training = bool(kwargs.get("training", False))
        return self.runner(feeds, training=training)

    def train(self, **kwargs) -> Dict[str, Any]:
        """Minimal training loop using autograd + Adam.

        kwargs:
          - epochs: int
          - lr: float
          - loss_fn: callable(optional)
          - inputs/targets: override captured data
        Returns a small dict with final loss and step count.
        """
        epochs = int(kwargs.get("epochs", self.epochs or 1))
        lr = float(kwargs.get("lr", self.lr))
        lf = kwargs.get("loss_fn", self.loss_fn) or self._mse
        inputs = kwargs.get("inputs", self.inputs_any)
        targets = kwargs.get("targets", self.targets_any)

        # Build parameter list
        params = []
        try:
            if hasattr(self.model, "parameters"):
                params = list(self.model.parameters())
        except Exception:
            params = []
        if not params:
            raise ValueError("Model has no parameters() or it returned an empty list")

        opt = self.optimizer_cls(params, lr=lr)

        hist_loss = None
        for step in range(1, epochs + 1):
            # Reset tape each step for a clean capture
            autograd.tape = autograd.__class__().tape
            pair = (inputs, targets)
            if pair[0] is None or pair[1] is None:
                npair = self._next_training_pair(step)
                if npair is None:
                    raise RuntimeError("No training material available: hook/queue/stdin all empty")
                pair = npair
            inputs, targets = pair
            # Use fused program for the forward pass when available
            if self.runner is not None and self.program is not None and self.input_feed_ids:
                run_feeds = dict(self.feed_store)
                for fid in self.input_feed_ids:
                    run_feeds[fid] = inputs
                out_map = self.runner(run_feeds, training=True)
                pred = out_map.get("pred", next(iter(out_map.values())))
            else:
                pred = getattr(self.model, "forward", self.model)(inputs)
            loss = lf(pred, targets)
            try:
                loss.backward()
            except Exception as e:
                raise RuntimeError(f"backward() failed at step {step}: {e}") from e

            grads = []
            for p in params:
                g = getattr(p, "grad", None)
                if g is None:
                    raise RuntimeError("Parameter has no gradient after backward(); check graph connectivity")
                grads.append(g)
            new_params = opt.step(params, grads)
            for p, np_ in zip(params, new_params):
                AT.copyto(p, np_)
            try:
                hist_loss = float(loss.item())
            except Exception:
                hist_loss = None
            # Enqueue this pair for potential recycling with age metadata
            try:
                if self.recycle_enabled:
                    self.enqueue_training_sample(inputs, targets, step=step)
            except Exception:
                pass

        # Optionally refresh program after training
        try:
            self.capture(inputs, targets, outputs=self.outputs)
        except Exception:
            pass
        return {"epochs": epochs, "lr": lr, "final_loss": hist_loss}
