"""The chess network's evaluation compiled to native code.

The net's own ``ConvValueNet.forward_both`` and the masked softmax of
``evaluate`` are traced through the SSA tensor backend (parameters and the
board batch are SSA inputs) and compiled by the repository's forward entry,
``compile_native_graph_forward``.  Nothing is rewritten: the traced code is
the same method the Python path runs.  The compiled function takes a fixed
batch; ``NativeEvaluator.evaluate`` pads a smaller batch and splits a larger
one.

    python examples/meowdoku/native_forward.py --channels 2 --hidden 8 --batch 1   # small first
    python examples/meowdoku/native_forward.py                                       # the real net
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku.chess_net import PLANES, POLICY, ConvValueNet  # noqa: E402

PARAMETER_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "W5", "b5", "W6", "b6")
BUILD = ROOT / "build" / "meowdoku" / "native_forward"


def _trace(channels: int, hidden: int, batch: int, name: str):
    from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
    shell = ConvValueNet(channels=channels, hidden=hidden)
    program = SSATensorProgram(name)
    shapes = {"x": (batch, PLANES, 8, 8), "mask": (batch, POLICY)}
    shapes.update({p: tuple(getattr(shell, p).shape) for p in PARAMETER_NAMES})
    bindings = {key: SSATensorOperations.input(program, shape) for key, shape in shapes.items()}
    for p in PARAMETER_NAMES:
        setattr(shell, p, bindings[p])
    value, logits = shell.forward_both(bindings["x"])
    probs = (logits + (bindings["mask"] - 1.0) * 1e4).softmax(dim=1)
    return (value, probs), bindings, shapes


class NativeEvaluator:
    """``evaluate(boards, masks)`` with the same answers as the net's own,
    run by the compiled library with the net's current weights."""

    def __init__(self, net: ConvValueNet, batch: int = 32, directory: Path = BUILD,
                 contract: str | None = None):
        """``contract`` names the work-contract preset the compile runs
        under (``develop`` when None): ``fast`` licenses FMA contraction
        and fast-math, ``eigen`` links Eigen's GEMM for the kernels."""
        from src.compiler.llvm_training_runtime import compile_native_graph_forward
        from src.compiler.work_contract import PRESETS, scoped_work_contract
        self.net, self.batch, self.contract = net, int(batch), contract or "develop"
        name = f"chess_eval_c{net.channels}_h{net.hidden}_b{self.batch}_{self.contract}"
        outputs, bindings, self.shapes = _trace(net.channels, net.hidden, self.batch, name)
        started = time.perf_counter()
        with scoped_work_contract(PRESETS[self.contract]):
            self.forward = compile_native_graph_forward(
                outputs, bindings=bindings, source="ConvValueNet.forward_both + masked softmax",
                name=name, directory=Path(directory) / name)
        self.compile_seconds = time.perf_counter() - started
        self.ids = dict(self.forward.input_value_ids)
        self.value_id, self.policy_id = self.forward.output_value_ids

    def _weights(self) -> dict[int, np.ndarray]:
        with self.net.lock:
            return {self.ids[p]: np.ascontiguousarray(np.asarray(getattr(self.net, p).numpy(), dtype=np.float64))
                    for p in PARAMETER_NAMES}

    def _resident(self):
        """One prepared execution kept for the life of the evaluator: its
        buffers are bound once, and every call writes into them in place
        and calls the entry.  Preparation (allocation and extent
        derivation) is paid on the first call only."""
        if getattr(self, "_execution", None) is None:
            from src.compiler.ssa_llvm_backend import prepare_artifact_execution
            self._execution = prepare_artifact_execution(self.forward.artifact, {
                **self._weights(),
                self.ids["x"]: np.zeros((self.batch, PLANES, 8, 8)),
                self.ids["mask"]: np.ones((self.batch, POLICY))})
            self._buffers = {key: np.asarray(self._execution.buffers[value]) for key, value in self.ids.items()}
            self._value_out = np.asarray(self._execution.buffers[self.value_id])
            self._policy_out = np.asarray(self._execution.buffers[self.policy_id])
        return self._execution

    def _refresh_weights(self) -> None:
        """Copy the net's current weights into the resident buffers.  Adam
        writes the parameters in place, so nothing cheaper than the copy
        (about half a millisecond) proves they are unchanged."""
        with self.net.lock:
            for p in PARAMETER_NAMES:
                np.copyto(self._buffers[p].reshape(-1),
                          np.asarray(getattr(self.net, p).numpy(), dtype=np.float64).reshape(-1))

    def evaluate(self, boards: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        execution = self._resident()
        self._refresh_weights()
        planes = self.net.planes(boards)
        masks = np.asarray(masks, dtype=np.float64)
        x = self._buffers["x"].reshape(self.batch, PLANES, 8, 8)
        m = self._buffers["mask"].reshape(self.batch, POLICY)
        values, policies = [], []
        for start in range(0, planes.shape[0], self.batch):
            chunk = planes[start:start + self.batch]
            count = len(chunk)
            x[:count] = chunk; x[count:] = 0.0
            m[:count] = masks[start:start + self.batch]; m[count:] = 1.0
            execution.run()
            values.append(self._value_out.reshape(-1)[:count].copy())
            policies.append(self._policy_out.reshape(self.batch, POLICY)[:count].copy())
        return np.concatenate(values), np.concatenate(policies)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args(argv)
    from meowdoku.chess_net import encode, legal_mask
    from meowdoku.games import Chess
    net = ConvValueNet(channels=args.channels, hidden=args.hidden, seed=3)
    if args.channels == 32 and args.hidden == 128:
        from meowdoku.chess_net import WEIGHTS
        print("weights loaded:", net.load(WEIGHTS), flush=True)
    native = NativeEvaluator(net, batch=args.batch)
    print(f"compiled in {native.compile_seconds:.1f}s: {native.forward.artifact.library_path}", flush=True)
    rng = np.random.default_rng(0)
    game = Chess("probe")
    boards, masks = [], []
    for _ in range(args.batch + 3):                     # crosses a batch boundary
        moves = game.all_legal_moves()
        if not moves:
            game = Chess("probe"); moves = game.all_legal_moves()
        boards.append(encode(game.pieces, game.turn)); masks.append(legal_mask(moves))
        src, dst = moves[int(rng.integers(len(moves)))]
        game.apply_move(src, dst); game.turn = "black" if game.turn == "white" else "white"
    boards, masks = np.stack(boards), np.stack(masks)
    t = time.perf_counter(); v0, p0 = net.evaluate(boards, masks); python_s = time.perf_counter() - t
    t = time.perf_counter(); v1, p1 = native.evaluate(boards, masks); native_s = time.perf_counter() - t
    print(f"positions {len(boards)} · python {python_s * 1e3:.1f} ms · native {native_s * 1e3:.1f} ms", flush=True)
    print(f"max |value diff| {np.abs(v0 - v1).max():.3e} · max |policy diff| {np.abs(p0 - p1).max():.3e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
