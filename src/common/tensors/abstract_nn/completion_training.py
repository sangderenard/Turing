"""
completion_training.py
----------------------

Modular completion-training utilities for interactive and document-sourced
training loops built on top of IRGraphedModel + FusedProgram.

Features
- Interactive mode: decode model output to text, prompt for expected response,
  encode to bytes-length target, train step, and loop forever.
- Document source mode: sample n pairs of (input_len, output_len) byte spans
  from a corpus using the project's Random utility, then train.
- Pluggable loss function and optimizer; uses Adam + MSE by default.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

from .fused_program import IRGraphedModel
from .optimizer import Adam
from ..abstraction import AbstractTensor as AT
from ..abstraction_methods.random import Random
from ..autograd import autograd


def _tensor_to_bytes(t: Any) -> bytes:
    obj = getattr(t, "data", t)
    try:
        arr = obj.tolist() if hasattr(obj, "tolist") else obj
    except Exception:
        arr = obj
    # Flatten
    def _flat(x):
        if isinstance(x, (list, tuple)):
            for v in x:
                yield from _flat(v)
        else:
            yield x
    flat = list(_flat(arr))
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


def decode_text(pred: Any, encoding: str = "utf-8") -> str:
    b = _tensor_to_bytes(pred)
    try:
        return b.decode(encoding, errors="replace")
    except Exception:
        return b.decode("utf-8", errors="replace")


def encode_text(text: str, length: Optional[int] = None, *, encoding: str = "utf-8", like: Any | None = None) -> Any:
    b = text.encode(encoding, errors="ignore")
    if length is not None:
        cur = list(b)
        if len(cur) < length:
            cur = cur + [0] * (length - len(cur))
        elif len(cur) > length:
            cur = cur[:length]
        b = bytes(cur)
    vals = list(b)
    # Use like's backend when provided
    return AT.get_tensor(vals)


def sample_document_pairs(
    document: bytes | str,
    *,
    input_len: int,
    output_len: int,
    batch: int,
    seed: Optional[int] = None,
) -> List[Tuple[Any, Any]]:
    """Return a list of (input_tensor, target_tensor) pairs from a text corpus.

    Chooses random offsets such that input covers [pos, pos+input_len) and
    target covers [pos+input_len, pos+input_len+output_len). If the sampled
    region extends beyond the corpus, wraps around.
    """
    data = document.encode("utf-8") if isinstance(document, str) else document
    n = len(data)
    if n == 0:
        return []
    rng = Random(seed=seed)
    pairs: List[Tuple[Any, Any]] = []
    for _ in range(int(batch)):
        pos = rng.randint(0, max(0, n - 1))
        x_bytes = bytearray()
        y_bytes = bytearray()
        for i in range(input_len):
            x_bytes.append(data[(pos + i) % n])
        for j in range(output_len):
            y_bytes.append(data[(pos + input_len + j) % n])
        x = AT.get_tensor(list(x_bytes))
        y = AT.get_tensor(list(y_bytes))
        pairs.append((x, y))
    return pairs


class CompletionTrainer:
    def __init__(
        self,
        gm: IRGraphedModel,
        *,
        loss_fn: Optional[Callable[[Any, Any], Any]] = None,
        optimizer_cls: Callable[..., Any] = Adam,
        lr: float = 1e-2,
    ) -> None:
        self.gm = gm
        self.loss_fn = loss_fn or (lambda a, b: ((a - b) ** 2).mean())
        # Build optimizer on model params
        params = []
        try:
            if hasattr(gm.model, "parameters"):
                params = list(gm.model.parameters())
        except Exception:
            params = []
        if not params:
            raise ValueError("Model has no parameters() or it returned an empty list")
        self.params = params
        self.opt = optimizer_cls(params, lr=lr)

    def _forward_pred(self, inputs: Any, *, training: bool) -> Any:
        gm = self.gm
        if gm.runner is not None and gm.program is not None and gm.input_feed_ids:
            feeds = dict(gm.feed_store)
            for fid in gm.input_feed_ids:
                feeds[fid] = inputs
            out = gm.runner(feeds, training=training)
            return out.get("pred", next(iter(out.values())))
        return getattr(gm.model, "forward", gm.model)(inputs)

    def train_step(self, inputs: Any, targets: Any) -> float:
        # Fresh tape and bake loss into the captured program so forward+loss share the same tape.
        autograd.tape = autograd.__class__().tape
        gm = self.gm
        gm.capture(inputs, targets)
        if gm.runner is not None and gm.program is not None:
            out = gm.runner(dict(gm.feed_store), training=True)
            if "loss" not in out:
                raise RuntimeError("Fused program missing 'loss' output; capture must include targets.")
            loss = out["loss"]
            # Align strict checks to the loss' tape
            lt = getattr(loss, "_tape", None)
            if lt is not None:
                autograd.tape = lt
            # Apply parameter updates returned by the program if present
            # We rely on ordered parameters() and names param{i}_new
            new_params: List[Any] = []
            i = 0
            while True:
                key = f"param{ i }_new"
                if key not in out:
                    break
                new_params.append(out[key])
                i += 1
            if new_params:
                for p, np_ in zip(self.params, new_params):
                    AT.copyto(p, np_)
            # Update optimizer state (adam) if provided
            try:
                if hasattr(gm, "opt_m") and hasattr(gm, "opt_v"):
                    gm.opt_m = []
                    gm.opt_v = []
                    i = 0
                    while True:
                        mk = f"opt_m{ i }_new"; vk = f"opt_v{ i }_new"
                        if mk not in out or vk not in out:
                            break
                        gm.opt_m.append(out[mk])
                        gm.opt_v.append(out[vk])
                        i += 1
                    if "opt_t_new" in out:
                        gm.opt_t = out["opt_t_new"]
            except Exception:
                pass
        else:
            pred = getattr(gm.model, "forward", gm.model)(inputs)
            loss = self.loss_fn(pred, targets)
        loss.backward()
        grads = []
        for p in self.params:
            g = getattr(p, "grad", None)
            if g is None:
                raise RuntimeError("Parameter has no gradient; check graph connectivity")
            grads.append(g)
        # Optimizer step only when not handled by program
        if gm.runner is None or gm.program is None:
            new_params = self.opt.step(self.params, grads)
            for p, np_ in zip(self.params, new_params):
                AT.copyto(p, np_)
        try:
            return float(loss.item())
        except Exception:
            return float("nan")

    def interactive_loop(self, *, initial_input: Any, encoding: str = "utf-8") -> None:
        """Endless loop: show model output as text, accept expected reply, train step,
        then feed expected reply as next input.
        """
        last_input = initial_input
        step = 0
        while True:
            step += 1
            autograd.tape = autograd.__class__().tape
            pred = self._forward_pred(last_input, training=False)
            text = decode_text(pred, encoding=encoding)
            try:
                print("\nModel output (decoded):\n" + text)
                exp = input("Enter expected output (utf-8) then Enter:\n> ")
            except (EOFError, KeyboardInterrupt):
                print("\nStopped by user.")
                break
            # Prepare targets with length match
            try:
                L = int(getattr(pred, "numel", lambda: None)() or len(_tensor_to_bytes(pred)))
            except Exception:
                L = None
            targets = encode_text(exp, length=L, encoding=encoding, like=pred)
            loss_val = self.train_step(last_input, targets)
            try:
                print(f"[step {step}] loss={loss_val:.4e}")
            except Exception:
                pass
            last_input = targets

    def train_from_document(
        self,
        document: bytes | str,
        *,
        input_len: int,
        output_len: int,
        epochs: int = 1,
        batch: int = 32,
        encoding: str = "utf-8",
    ) -> List[float]:
        """Train for a number of epochs using random (input, output) byte slices
        from a document as completion examples. Returns per-epoch loss averages.
        """
        epoch_losses: List[float] = []
        for ep in range(1, int(epochs) + 1):
            pairs = sample_document_pairs(
                document,
                input_len=input_len,
                output_len=output_len,
                batch=batch,
            )
            total = 0.0
            count = 0
            for x, y in pairs:
                loss_val = self.train_step(x, y)
                if loss_val == loss_val:  # not NaN
                    total += loss_val
                    count += 1
            avg = total / max(1, count)
            epoch_losses.append(avg)
            try:
                print(f"[epoch {ep}] avg_loss={avg:.4e} n={count}")
            except Exception:
                pass
        return epoch_losses


__all__ = [
    "CompletionTrainer",
    "sample_document_pairs",
    "encode_text",
    "decode_text",
]
