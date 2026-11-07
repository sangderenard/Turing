"""Trimmed token-pilot variants: profile, distil, evaluate, cost.

The variants (``pilot_law`` VARIANTS: ``t32.h32.q1.g32.ahard`` ...) are one
text with the default network; this tool measures them.  Run from the
repository root with TURING_EIGEN_INCLUDE set; every command is one process,
one compile at a time.

  profile  VARIANT [R...]     native op-by-op cost of the forward at batch R (cumulative cuts)
  record   EPISODES SECONDS   fly the TEACHER (the live bullet_hell weights) on the real law, record
                              (observation, teacher output) -> build/meowdoku/autopilot/teacher_records.npz
  distill  VARIANT            fit VARIANT to the records (warm start from the teacher where shapes allow),
                              save beside the live weights as bullet_hell.<VARIANT>.npz, report imitation error
  play     VARIANT SEEDS SECONDS   fly VARIANT (``token`` = the teacher) on the real law, score / survival
  cost     VARIANT [R...]     native in-law cost of observation + network for R piloted lanes

The live weights file is never written (``distill`` writes the variant's own file).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parents[1]), "C:/dev/Powershell/engine_toy"]

from meowdoku.s3_train import autopilot as ap  # noqa: E402
from meowdoku.s3_train import pilot_law as pl  # noqa: E402

RECORDS = ap.BUILD / "teacher_records.npz"
COLS = ("lane", "slot", "kind", "owner", "alive", "x", "y", "z", "vx", "vy", "vz", "hp", "spd", "bank", "fuel", "energy",
        "lives", "timer", "beam", "weapon")


def teacher_params() -> dict:
    params = ap.load_params(ap.weights_path("bullet_hell", pl.DEFAULT_VARIANT))
    if params is None:
        raise SystemExit("no live bullet_hell weights")
    return params


def _native(out, bindings, name, directory):
    from src.compiler.llvm_training_runtime import compile_native_graph_forward
    return compile_native_graph_forward(out, bindings=bindings, source="pilot_variants", name=name, directory=directory)


def _time_execution(ex, repeats: int) -> float:
    """Seconds per run: the best of five blocks (the machine is shared)."""
    ex.run()
    best = float("inf")
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(repeats):
            ex.run()
        best = min(best, (time.perf_counter() - t) / repeats)
    return best


def profile(variant: str, batches: list[int]) -> None:
    """Cumulative cuts of the forward, natively (eigen contract): each row is the
    network up to that op, so a row minus the one above is that op's cost."""
    from src.common.tensors import AbstractTensor as AT
    from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    spec = pl.spec_of(variant)
    params = ap.init_token_params(0, 5, variant)
    T, H, act = spec["tokens"], spec["h"], spec["act"]
    names = pl.TOKEN_PARAMS
    stages = []

    def stage(label):
        def deco(fn):
            stages.append((label, fn))
            return fn
        return deco

    def unpack(x, n, p):
        tokens = x[:, pl.SHIP_G:pl.SHIP_G + T * pl.TOKEN_F].reshape((n * T, pl.TOKEN_F))
        mask = x[:, pl.SHIP_G + T * pl.TOKEN_F:].reshape((n, T, 1))
        return tokens, mask

    @stage("embed layer 1 GEMM (+bias)")
    def _(x, n, p):
        return unpack(x, n, p)[0] @ p["We1"] + p["be1"]

    @stage("embed layer 1 gate")
    def _(x, n, p):
        return pl.gate(unpack(x, n, p)[0] @ p["We1"] + p["be1"], act)

    @stage("embed layer 2 GEMM (+bias)")
    def _(x, n, p):
        return pl.gate(unpack(x, n, p)[0] @ p["We1"] + p["be1"], act) @ p["We2"] + p["be2"]

    def embedded(x, n, p):
        return pl.embed(unpack(x, n, p)[0], p["We1"], p["be1"], p["We2"], p["be2"], act)

    @stage("embed layer 2 gate")
    def _(x, n, p):
        return embedded(x, n, p)

    @stage("attention score GEMV")
    def _(x, n, p):
        return embedded(x, n, p) @ p["Q1"]

    @stage("attention score squash")
    def _(x, n, p):
        s = embedded(x, n, p) @ p["Q1"]
        return s.tanh() if act == "tanh" else s.clip(-1.0, 1.0)

    @stage("attention exp * mask")
    def _(x, n, p):
        s = embedded(x, n, p) @ p["Q1"]
        s = s.tanh() if act == "tanh" else s.clip(-1.0, 1.0)
        return (s * 4.0).exp().reshape((n, T, 1)) * unpack(x, n, p)[1]

    @stage("pooling (all queries)")
    def _(x, n, p):
        tokens, mask = unpack(x, n, p)
        qs = (p["Q1"], p["Q2"]) if spec["q"] == 2 else (p["Q1"],)
        return AT.cat(pl.pool(embedded(x, n, p), mask, n, T, qs, act), dim=1)

    def full(x, n, p):
        return pl.token_policy_forward(x, *(p[k] for k in names), act=act)

    @stage("ship branch + head")
    def _(x, n, p):
        return full(x, n, p)

    print(f"profile {variant}: tokens {T}, embed {H}, queries {spec['q']}, head {spec['g']}, gate {act}", flush=True)
    for R in batches:
        print(f"-- batch {R}", flush=True)
        previous = 0.0
        for label, fn in stages:
            with scoped_work_contract(PRESETS["eigen"]):
                program = SSATensorProgram(f"prof_{abs(hash((variant, R, label))) % 10**8}")
                shapes = {"x": (R, pl.features_of(T)), **{k: tuple(params[k].shape) for k in names}}
                b = {k: SSATensorOperations.input(program, shape) for k, shape in shapes.items()}
                with program.activate():
                    out = fn(b["x"], R, b)
                fw = _native(out, b, "prof_stage",
                             ap.BUILD / "native" / f"prof_{os.getpid()}_{abs(hash((variant, R, label))) % 10**8}")
            ids = fw.input_value_ids
            rng = np.random.default_rng(0)
            x = rng.normal(0, 0.5, (R, pl.features_of(T)))
            x[:, -T:] = rng.random((R, T)) < 0.5
            init = {ids["x"]: x, **{ids[k]: np.ascontiguousarray(params[k]) for k in names}}
            ex = prepare_artifact_execution(fw.artifact, init)
            us = _time_execution(ex, max(20, 400 // R)) * 1e6 / R
            print(f"   {label:34s} {us:8.1f} us/row   (+{us - previous:7.1f})", flush=True)
            previous = us


def cost(variant: str, batches: list[int]) -> None:
    """In-law cost of one decision: the observation plus the network, natively
    (the module's variant is the process's MEOWDOKU_PILOT_ARCH)."""
    from meowdoku.s2_titles.scroller import scroller as sc
    from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    assert variant == pl.ACTIVE_VARIANT, f"run with MEOWDOKU_PILOT_ARCH={variant}"
    params = ap.init_token_params(0, 5, variant) if variant != pl.DEFAULT_VARIANT else teacher_params()
    cols = sc.initial_columns()
    for R in batches:
        rows = (1, 1 + R)
        for part in ("observation", "network", "decide"):
            with scoped_work_contract(PRESETS["eigen"]):
                program = SSATensorProgram(f"cost_{part}_{R}")
                shapes = {n: np.shape(cols[n]) for n in COLS}
                shapes.update({n: tuple(params[n].shape) for n in pl.TOKEN_PARAMS})
                shapes["obs"] = (R, pl.TOKEN_FEATURES)
                b = {n: SSATensorOperations.input(program, s) for n, s in shapes.items()}
                with program.activate():
                    if part == "observation":
                        out = pl.observation(rows, {n: b[n] for n in COLS})
                    elif part == "network":
                        out = pl.token_policy_forward(b["obs"], *(b[n] for n in pl.TOKEN_PARAMS))
                    else:
                        out = pl.decide(rows, {n: b[n] for n in COLS}, tuple(b[n] for n in pl.TOKEN_PARAMS))
                fw = _native(out, b, f"cost_{part}_{R}", ap.BUILD / "native" / f"cost_{os.getpid()}_{part}_{R}")
            ids = fw.input_value_ids
            init = {ids[n]: np.ascontiguousarray(params[n]) if n in params else
                    (np.zeros(shapes[n]) if n == "obs" else np.array(cols[n], dtype=np.float64)) for n in shapes}
            ex = prepare_artifact_execution(fw.artifact, init)
            rng = np.random.default_rng(1)
            a, x, y = (np.asarray(ex.buffers[ids[k]]) for k in ("alive", "x", "y"))
            if part == "network":
                xin = rng.normal(0, 0.5, (R, pl.TOKEN_FEATURES))
                xin[:, -pl.TOKENS:] = rng.random((R, pl.TOKENS)) < 0.5
                np.asarray(ex.buffers[ids["obs"]])[...] = xin
            else:       # a busy board: bullets scattered about the player so most tokens are real
                lo, hi = sc.BULLETS
                a[lo:hi] = 1.0
                x[lo:hi] = 4.5 + rng.normal(0, 3, hi - lo)
                y[lo:hi] = 11 + rng.normal(0, 3, hi - lo)
            per = _time_execution(ex, max(10, 200 // R))
            print(f"COST {variant:22s} R={R:3d} {part:11s} {per * 1e3:8.3f} ms/call  {per / R * 1e6:8.1f} us/lane", flush=True)


# ------------------------------------------------------------ distil / play

def record(episodes: int, seconds: float, noise: float = 0.35) -> None:
    """Fly the teacher on the real law, recording (64-token observation, teacher
    output).  Off-path states come from taking, for a few decisions at a time,
    a random control instead of the teacher's (the labels stay the teacher's)."""
    params = teacher_params()
    run = ap.numpy_product(1, *ap.widths_of(params), arch=ap.arch_key(params))
    xs, ys = [], []
    for e in range(episodes):
        rng = np.random.default_rng(900 + e)
        hold = {"left": 0, "out": None}

        def act(game, f, rng=rng, hold=hold):
            out = run(f, params)[0]
            xs.append(np.asarray(f, dtype=np.float64)); ys.append(out)
            if hold["left"] > 0:
                hold["left"] -= 1
                return hold["out"]
            if rng.random() < 0.04 * noise / 0.35:
                hold["left"] = int(rng.integers(3, 12))
                hold["out"] = np.clip(out + rng.normal(0, 0.8, out.shape), -1.0, 1.0)
            return out
        r = ap.run_episode("bullet_hell", act, 900 + e, seconds)
        print(f"record ep {e}: score {r['score']:.0f} survival {r['survival']:.0f}s rows {len(xs)}", flush=True)
    ap.BUILD.mkdir(parents=True, exist_ok=True)
    np.savez(RECORDS, x=np.stack(xs), y=np.stack(ys))


def distill(variant: str, epochs: int = 40, seed: int = 0) -> dict:
    data = np.load(RECORDS)
    x64, y = data["x"], data["y"]
    spec = pl.spec_of(variant)
    x = pl.slice_tokens(x64, 64, spec["tokens"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    hold = order[:max(256, len(x) // 10)]
    fit = order[len(hold):]
    teacher = teacher_params()
    start = ap.init_token_params(seed, 5, variant)
    # warm start where the shapes are the teacher's (same widths: only the tokens / gate / query changed)
    same = all(start[n].shape == teacher[n].shape for n in pl.TOKEN_PARAMS if n != "Q2")
    if same:
        start = {**teacher, "variant": start.get("variant", np.array(variant))} if variant != pl.DEFAULT_VARIANT else dict(teacher)
        if spec["q"] == 1:
            start["Q2"] = np.zeros((1, 1))
            # the one-query head reads one pooled block: take the teacher's first
            H = spec["h"]
            start["Wh1"] = np.concatenate([teacher["Wh1"][:H], teacher["Wh1"][2 * H:]], axis=0)
    trainer = ap.PolicyTrainer(start, lr=1e-3 if same else 2e-3)

    def error(rows):
        out = ap.eager_forward(trainer.params(), x[rows])
        return float(np.mean((out - y[rows]) ** 2)), float(np.mean(np.abs(out - y[rows]))), \
            float(np.mean(np.sign(out[:, 2]) == np.sign(y[rows, 2])))
    first = error(hold)
    started = time.perf_counter()
    for epoch in range(epochs):
        perm = rng.permutation(fit)
        for at in range(0, len(perm), 256):
            trainer.step(x[perm[at:at + 256]], y[perm[at:at + 256]])
        if epoch % 10 == 9:
            print(f"  {variant} epoch {epoch + 1}: held-out mse/mae/fire-agree {error(hold)}", flush=True)
    mse, mae, fire = error(hold)
    params = trainer.params()
    path = ap.weights_path("bullet_hell", variant)
    assert variant == pl.DEFAULT_VARIANT or path.name != "bullet_hell.npz"
    if variant != pl.DEFAULT_VARIANT:
        ap.save_params(params, path)
    result = {"variant": variant, "warm": bool(same), "rows": int(len(x)), "mse_before": first[0], "mse": mse, "mae": mae,
              "fire_agree": fire, "train_s": round(time.perf_counter() - started, 1), "saved": str(path),
              "params": int(sum(params[n].size for n in pl.TOKEN_PARAMS))}
    print("DISTILL " + json.dumps(result), flush=True)
    return result


def play(variant: str, seeds: int, seconds: float) -> None:
    params = teacher_params() if variant == pl.DEFAULT_VARIANT else ap.load_params(ap.weights_path("bullet_hell", variant))
    assert ap.variant_of(params) == variant, (ap.variant_of(params), variant)
    tokens = pl.spec_of(variant)["tokens"]
    run = ap.numpy_product(1, *ap.widths_of(params), arch=ap.arch_key(params))

    def act(game, f):
        return run(pl.slice_tokens(f, 64, tokens), params)[0]
    episodes = []
    for seed in range(seeds):
        r = ap.run_episode("bullet_hell", act, 100 + seed, seconds)
        episodes.append(r)
        print(f"PLAY {variant} seed {100 + seed}: score {r['score']:.0f} survival {r['survival']:.0f}s lives {r['lives']:.0f}", flush=True)
    s = ap.summarize(episodes)
    print("PLAYSUM " + json.dumps({"variant": variant, **{k: round(v, 1) for k, v in s.items()}}), flush=True)


def main(argv=None) -> None:
    import src.common.tensors.numpy_backend  # noqa: F401
    argv = list(sys.argv[1:] if argv is None else argv)
    command, rest = argv[0], argv[1:]
    if command == "profile":
        profile(rest[0], [int(r) for r in rest[1:]] or [1, 16])
    elif command == "cost":
        cost(rest[0], [int(r) for r in rest[1:]] or [1, 16, 64])
    elif command == "record":
        record(int(rest[0]), float(rest[1]))
    elif command == "distill":
        distill(rest[0], int(rest[1]) if len(rest) > 1 else 40)
    elif command == "play":
        play(rest[0], int(rest[1]), float(rest[2]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
