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

    def __init__(self, net: ConvValueNet, batch: int = 32, directory: Path = BUILD):
        from src.compiler.llvm_training_runtime import compile_native_graph_forward
        self.net, self.batch = net, int(batch)
        name = f"chess_eval_c{net.channels}_h{net.hidden}_b{self.batch}"
        outputs, bindings, self.shapes = _trace(net.channels, net.hidden, self.batch, name)
        started = time.perf_counter()
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

    def evaluate(self, boards: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from src.compiler.ssa_llvm_backend import prepare_artifact_execution
        planes = self.net.planes(boards)
        masks = np.asarray(masks, dtype=np.float64)
        n = planes.shape[0]
        weights = self._weights()
        values, policies = [], []
        for start in range(0, n, self.batch):
            x = np.zeros((self.batch, PLANES, 8, 8)); m = np.ones((self.batch, POLICY))
            chunk = planes[start:start + self.batch]
            x[:len(chunk)] = chunk; m[:len(chunk)] = masks[start:start + self.batch]
            run = prepare_artifact_execution(self.forward.artifact, {
                **weights, self.ids["x"]: x, self.ids["mask"]: m}).run()
            values.append(np.asarray(run.buffers[self.value_id]).reshape(-1)[:len(chunk)])
            policies.append(np.asarray(run.buffers[self.policy_id]).reshape(self.batch, POLICY)[:len(chunk)])
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
