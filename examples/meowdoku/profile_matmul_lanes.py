"""Profile the chess network's matrix products: numpy vs LLVM vs LLVM+Eigen.

Lanes
  numpy    the numpy backend (its BLAS)
  llvm     the repository compiler, default work contract: the authored
           compensated ``matmul_double`` / ``batched_matmul_indexed_double``
  eigen    the repository compiler under the ``eigen`` work contract: the same
           emitted program with ``matmul_double`` linked from Eigen

Two levels
  gemm     each matrix product of the net alone, as one compiled program
  forward  the whole evaluation (``native_forward.NativeEvaluator``) at
           several batch sizes: how the cost scales when more work is
           compiled into one call

Native timings are bare ``run()`` calls on a resident execution (buffers
bound once), median of repeats; every lane is checked against numpy.

    python examples/meowdoku/profile_matmul_lanes.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

BUILD = ROOT / "build" / "meowdoku" / "profile_lanes"
LANES = ("llvm", "eigen")
CONTRACT = {"llvm": "develop", "eigen": "eigen"}


def _median_ms(fn, repeats: int = 15) -> float:
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)) * 1e3


def _compile_gemm(m: int, k: int, n: int, lane: str):
    from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
    from src.compiler.llvm_training_runtime import compile_native_graph_forward
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    name = f"gemm_{m}x{k}x{n}_{lane}"
    program = SSATensorProgram(name)
    a = SSATensorOperations.input(program, (m, k))
    b = SSATensorOperations.input(program, (k, n))
    with scoped_work_contract(PRESETS[CONTRACT[lane]]):
        forward = compile_native_graph_forward(a @ b, bindings={"a": a, "b": b}, source="a @ b",
                                               name=name, directory=BUILD / name)
    rng = np.random.default_rng(0)
    A, B = rng.standard_normal((m, k)), rng.standard_normal((k, n))
    ids = forward.input_value_ids
    execution = prepare_artifact_execution(forward.artifact, {ids["a"]: A, ids["b"]: B})
    out = forward.output_value_ids[0]
    return execution, out, A, B


def gemm_table() -> None:
    shapes = {
        "conv1 (W1 @ cols, batch folded)": (32, 117, 2048),
        "conv2 / residual": (32, 288, 2048),
        "policy head": (64, 32, 2048),
        "dense": (32, 2048, 128),
    }
    print("\nGEMM alone (ms, median; GF/s = 2mkn / time)")
    print(f"{'shape':34s} {'m x k x n':>16s} {'numpy':>14s} {'llvm':>14s} {'eigen':>14s}   max |diff| llvm / eigen")
    for label, (m, k, n) in shapes.items():
        row, diffs = {}, {}
        rng = np.random.default_rng(0)
        A, B = rng.standard_normal((m, k)), rng.standard_normal((k, n))
        reference = A @ B
        row["numpy"] = _median_ms(lambda: A @ B)
        for lane in LANES:
            execution, out, A2, B2 = _compile_gemm(m, k, n, lane)
            row[lane] = _median_ms(execution.run)
            diffs[lane] = float(np.abs(np.asarray(execution.buffers[out]).reshape(m, n) - (A2 @ B2)).max())
        flops = 2.0 * m * k * n
        cells = "".join(f"{row[l]:7.2f} ({flops / row[l] / 1e6:4.1f})" for l in ("numpy", *LANES))
        print(f"{label:34s} {f'{m}x{k}x{n}':>16s} {cells}   {diffs['llvm']:.1e} / {diffs['eigen']:.1e}")


def forward_table(batches=(1, 32, 128)) -> None:
    from meowdoku.chess_net import ConvValueNet, WEIGHTS, encode, legal_mask
    from meowdoku.games import Chess
    from meowdoku.native_forward import NativeEvaluator
    net = ConvValueNet()
    net.load(WEIGHTS)
    print("\nWhole forward (ms per call, median; us per position in brackets)")
    print(f"{'batch':>5s} {'numpy':>16s} {'llvm':>16s} {'eigen':>16s}   compile s llvm / eigen   max |value diff| llvm / eigen")
    for batch in batches:
        rng = np.random.default_rng(batch)
        game, boards, masks = Chess("p"), [], []
        for _ in range(batch):
            moves = game.all_legal_moves()
            if not moves:
                game = Chess("p"); moves = game.all_legal_moves()
            boards.append(encode(game.pieces, game.turn)); masks.append(legal_mask(moves))
            src, dst = moves[int(rng.integers(len(moves)))]
            game.apply_move(src, dst); game.turn = "black" if game.turn == "white" else "white"
        boards, masks = np.stack(boards), np.stack(masks)
        reference, _ = net.evaluate(boards, masks)
        row = {"numpy": _median_ms(lambda: net.evaluate(boards, masks), 7)}
        compile_s, diffs = {}, {}
        for lane in LANES:
            evaluator = NativeEvaluator(net, batch=batch, directory=BUILD, contract=CONTRACT[lane])
            values, _ = evaluator.evaluate(boards, masks)
            row[lane] = _median_ms(evaluator._execution.run, 7)
            compile_s[lane] = evaluator.compile_seconds
            diffs[lane] = float(np.abs(values - reference).max())
        cells = "".join(f"{row[l]:8.2f} ({row[l] * 1e3 / batch:5.0f})" for l in ("numpy", *LANES))
        print(f"{batch:5d} {cells}   {compile_s['llvm']:5.1f} / {compile_s['eigen']:5.1f}            "
              f"{diffs['llvm']:.1e} / {diffs['eigen']:.1e}")


if __name__ == "__main__":
    gemm_table()
    forward_table()
