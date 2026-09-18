"""Nested LLVM training and graph-derived outer learning-rate demo.

The inner process is a complete one-step AbstractNN XOR training motion:
forward, MSE loss, graph-generated backward, and SGD update execute inside one
LLVM entry point.  The outer ProcessGraph evaluates the post-step model and
uses the repository's ProcessGraph autograd to derive the influence of the
inner learning rate on that outer loss.  A host-side outer optimizer step is
kept deliberately outside the hot loop.

Run from the repository root::

    python -m examples.xor_project.nested_llvm_meta_learning

After the first build, keep training from the carried inner state until
interrupted::

    python -m examples.xor_project.nested_llvm_meta_learning --stage run --outer-steps 0 --report-every 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


PARAMETER_IDS = (1, 2, 3, 4)


@dataclass(frozen=True)
class CompiledMotion:
    product: ProcessGraphBackwardProduct
    artifact: LLVMFunctionArtifact
    outputs: Mapping[str, int]


@dataclass(frozen=True)
class RuntimeMotion:
    artifact: LLVMFunctionArtifact
    outputs: Mapping[str, int]
    saved_binding_count: int


def _manifest_path(directory: Path, name: str) -> Path:
    return directory / f"{name}.json"


def _write_manifest(
    path: Path,
    artifact: LLVMFunctionArtifact,
    outputs: Mapping[str, int],
    *,
    saved_binding_count: int,
    library_path: Path | None = None,
) -> None:
    selected_library = artifact.library_path or library_path
    if selected_library is None:
        raise RuntimeError("cannot manifest an uncompiled LLVM artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "name": artifact.name,
        "library_path": str(selected_library.resolve()),
        "buffer_order": list(artifact.buffer_order),
        "buffer_shapes": [list(shape) for shape in artifact.buffer_shapes],
        "buffer_dtypes": list(artifact.buffer_dtypes),
        "extent_order": [list(item) for item in artifact.extent_order],
        "needs_text_sink": artifact.needs_text_sink,
        "training_steps_value_id": artifact.training_steps_value_id,
        "learning_rate_value_id": artifact.learning_rate_value_id,
        "outputs": dict(outputs),
        "saved_binding_count": int(saved_binding_count),
    }, indent=2, sort_keys=True), encoding="utf-8")


def _read_manifest(path: Path) -> RuntimeMotion:
    from src.compiler.ssa_llvm_backend import LLVMFunctionArtifact

    data = json.loads(path.read_text(encoding="utf-8"))
    artifact = LLVMFunctionArtifact(
        name=str(data["name"]),
        llvm_ir="",
        buffer_order=tuple(map(int, data["buffer_order"])),
        buffer_shapes=tuple(tuple(shape) for shape in data["buffer_shapes"]),
        extent_order=tuple(tuple(item) for item in data["extent_order"]),
        shortfalls=(),
        buffer_dtypes=tuple(map(str, data["buffer_dtypes"])),
        needs_text_sink=bool(data["needs_text_sink"]),
        library_path=Path(data["library_path"]),
        training_steps_value_id=data["training_steps_value_id"],
        learning_rate_value_id=data["learning_rate_value_id"],
    )
    return RuntimeMotion(
        artifact=artifact,
        outputs={str(key): int(value) for key, value in data["outputs"].items()},
        saved_binding_count=int(data["saved_binding_count"]),
    )


def xor_values() -> dict[int, np.ndarray]:
    return {
        0: np.asarray([
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ]),
        1: np.asarray([
            [.15, -.2, .35, -.4, .25, .1, -.3, .45],
            [-.3, .4, .2, -.1, .5, -.35, .15, .25],
        ]),
        2: np.asarray([[.05, -.1, .08, .02, -.04, .06, .03, -.07]]),
        3: np.asarray([
            [.3], [-.25], [.4], [-.35], [.2], [.15], [-.45], [.5],
        ]),
        4: np.asarray([[.02]]),
        5: np.asarray([[0.0], [1.0], [1.0], [0.0]]),
    }


def _xor_loss(x, w1, b1, w2, b2, target):
    from examples.xor_project.train_xor import build_model
    from src.common.tensors.abstract_nn import MSELoss

    model = build_model(x)
    model.layers[0].W, model.layers[0].b = w1, b1
    model.layers[1].W, model.layers[1].b = w2, b2
    return MSELoss()(model.forward(x), target)


def _compile_motion(
    output,
    *,
    bindings: Mapping[str, Any],
    wrt: tuple[int, ...],
    name: str,
    directory: Path,
    compile_native: bool = True,
) -> CompiledMotion:
    from src.compiler.process_graph_autograd import (
        abstract_tensor_program_to_process_graph,
        compile_process_graph_backward,
        lower_training_motion_to_repository_ssa,
    )
    from src.compiler.ssa_llvm_backend import (
        compile_artifact,
        emit_ssa_function_to_llvm,
    )

    forward = abstract_tensor_program_to_process_graph(output, bindings=bindings)
    product = compile_process_graph_backward(
        forward,
        wrt=wrt,
        packaging="combined",
    )
    if product.motion is None:
        raise RuntimeError("combined graph-autograd request produced no motion")
    lowering = lower_training_motion_to_repository_ssa(
        product.motion,
        function_name=name,
    )
    emitted = emit_ssa_function_to_llvm(
        lowering.module,
        lowering.function_name,
        entry_name=lowering.function_name,
    )
    if emitted.shortfalls:
        raise RuntimeError(f"{name} LLVM shortfalls: {emitted.shortfalls!r}")
    artifact = (
        compile_artifact(emitted, directory=directory)
        if compile_native else emitted
    )
    return CompiledMotion(product, artifact, dict(lowering.outputs))


def compile_inner(directory: Path) -> tuple[CompiledMotion, LLVMFunctionArtifact]:
    from src.common.tensors.accelerator_backends.ssa_backend import (
        SSATensorOperations,
        SSATensorProgram,
    )
    from src.compiler.ssa_llvm_backend import compile_artifact, with_native_sgd_loop

    program = SSATensorProgram("nested_xor_inner")
    shapes = ((4, 2), (2, 8), (1, 8), (8, 1), (1, 1), (4, 1))
    x, w1, b1, w2, b2, target = [
        SSATensorOperations.input(program, shape) for shape in shapes
    ]
    loss = _xor_loss(x, w1, b1, w2, b2, target)
    motion = _compile_motion(
        loss,
        bindings={
            "x": x,
            "W1": w1,
            "b1": b1,
            "W2": w2,
            "b2": b2,
            "target": target,
        },
        wrt=PARAMETER_IDS,
        name="nested_xor_inner",
        directory=directory / "inner_motion",
        compile_native=False,
    )
    loop = with_native_sgd_loop(
        motion.artifact,
        parameter_gradient_pairs=tuple(
            (parameter, motion.outputs[f"grad_{parameter}"])
            for parameter in PARAMETER_IDS
        ),
        entry_name="nested_xor_inner_step",
    )
    return motion, compile_artifact(loop, directory=directory / "inner_step")


def compile_outer(directory: Path) -> CompiledMotion:
    from src.common.tensors.accelerator_backends.ssa_backend import (
        SSATensorOperations,
        SSATensorProgram,
    )

    program = SSATensorProgram("nested_xor_outer")
    shapes = (
        (4, 2),
        (2, 8), (1, 8), (8, 1), (1, 1),
        (4, 1),
        (2, 8), (1, 8), (8, 1), (1, 1),
        (1,),
    )
    (
        x, w1, b1, w2, b2, target,
        inner_dw1, inner_db1, inner_dw2, inner_db2,
        inner_learning_rate,
    ) = [SSATensorOperations.input(program, shape) for shape in shapes]
    updated = (
        w1 - inner_learning_rate * inner_dw1,
        b1 - inner_learning_rate * inner_db1,
        w2 - inner_learning_rate * inner_dw2,
        b2 - inner_learning_rate * inner_db2,
    )
    outer_loss = _xor_loss(x, *updated, target)
    return _compile_motion(
        outer_loss,
        bindings={
            "x": x,
            "W1.before_inner_step": w1,
            "b1.before_inner_step": b1,
            "W2.before_inner_step": w2,
            "b2.before_inner_step": b2,
            "target": target,
            "saved.inner_dW1": inner_dw1,
            "saved.inner_db1": inner_db1,
            "saved.inner_dW2": inner_dw2,
            "saved.inner_db2": inner_db2,
            "outer.inner_learning_rate": inner_learning_rate,
        },
        wrt=(10,),
        name="nested_xor_outer",
        directory=directory / "outer_motion",
        compile_native=False,
    )


def _run_inner(
    artifact: LLVMFunctionArtifact,
    initial: Mapping[int, np.ndarray],
    learning_rate: float,
) -> LLVMExecution:
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution

    execution = prepare_artifact_execution(artifact, {
        **{key: value.copy() for key, value in initial.items()},
        artifact.training_steps_value_id: 1,
        artifact.learning_rate_value_id: float(learning_rate),
    })
    return execution.run()


def _run_outer(
    motion: RuntimeMotion,
    initial: Mapping[int, np.ndarray],
    saved_inner_gradients: Mapping[int, np.ndarray],
    learning_rate: float,
) -> LLVMExecution:
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution

    feeds = {key: value.copy() for key, value in initial.items()}
    feeds.update({
        6: saved_inner_gradients[1],
        7: saved_inner_gradients[2],
        8: saved_inner_gradients[3],
        9: saved_inner_gradients[4],
        10: np.asarray([learning_rate]),
    })
    return prepare_artifact_execution(motion.artifact, feeds).run()


def run_demo(
    *,
    directory: Path,
    inner_learning_rate: float = 0.5,
    outer_learning_rate: float = 0.25,
    outer_steps: int = 1,
    report_every: int = 1,
    stateful_inner: bool = True,
) -> dict[str, float]:
    if outer_steps < 0:
        raise ValueError("outer_steps must be non-negative (zero means indefinite)")
    if report_every <= 0:
        raise ValueError("report_every must be positive")
    initial = xor_values()
    inner_state = {key: value.copy() for key, value in initial.items()}
    inner = _read_manifest(_manifest_path(directory, "inner"))
    outer = _read_manifest(_manifest_path(directory, "outer"))
    inner_step = inner.artifact
    inner_loss_id = inner.outputs["loss_0"]
    current_learning_rate = float(inner_learning_rate)
    step = 0
    inner_loss = float("nan")
    outer_loss = float("nan")
    outer_gradient = float("nan")
    try:
        while outer_steps == 0 or step < outer_steps:
            step += 1
            before = {key: value.copy() for key, value in inner_state.items()}
            inner_execution = _run_inner(
                inner_step, before, current_learning_rate,
            )
            inner_loss = float(inner_execution.buffers[inner_loss_id])
            saved_inner_gradients = {
                parameter: inner_execution.buffers[
                    inner.outputs[f"grad_{parameter}"]
                ].copy()
                for parameter in PARAMETER_IDS
            }
            outer_execution = _run_outer(
                outer, before, saved_inner_gradients, current_learning_rate,
            )
            outer_loss = float(
                outer_execution.buffers[outer.outputs["loss_0"]]
            )
            outer_gradient = float(
                outer_execution.buffers[outer.outputs["grad_10"]][0]
            )
            next_learning_rate = (
                current_learning_rate - outer_learning_rate * outer_gradient
            )
            native_updated = {
                parameter: inner_execution.buffers[parameter].copy()
                for parameter in PARAMETER_IDS
            }
            if stateful_inner:
                inner_state.update(native_updated)
            else:
                inner_state = {
                    key: value.copy() for key, value in initial.items()
                }
            if step == 1 or step % report_every == 0:
                print(
                    f"outer step {step:6d}  "
                    f"inner_loss={inner_loss:.12f}  "
                    f"outer_loss={outer_loss:.12f}  "
                    f"dL/dlr={outer_gradient:+.12f}  "
                    f"inner_lr={current_learning_rate:.12f}"
                )
            current_learning_rate = next_learning_rate
    except KeyboardInterrupt:
        print(f"stopped after {step} outer steps")

    print(f"outer saved bindings: {outer.saved_binding_count}")
    print(f"inner state carried:  {stateful_inner}")
    return {
        "inner_loss": inner_loss,
        "outer_loss": outer_loss,
        "outer_gradient": outer_gradient,
        "updated_inner_learning_rate": current_learning_rate,
        "outer_steps": float(step),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-directory",
        type=Path,
        default=Path(".turing-cache") / "nested-xor-meta-learning",
    )
    parser.add_argument("--inner-learning-rate", type=float, default=0.5)
    parser.add_argument("--outer-learning-rate", type=float, default=0.25)
    parser.add_argument(
        "--outer-steps",
        type=int,
        default=1,
        help="outer learning cycles; zero runs until Ctrl+C",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=1,
        help="print one loss/gradient line every N outer cycles",
    )
    parser.add_argument(
        "--reset-inner-each-step",
        action="store_true",
        help="restart from the original XOR weights instead of carrying state",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "compile-inner", "compile-outer", "run"),
        default="all",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    directory = args.build_directory.resolve()
    if args.stage == "compile-inner":
        motion, step = compile_inner(directory)
        _write_manifest(
            _manifest_path(directory, "inner"),
            step,
            motion.outputs,
            saved_binding_count=len(
                motion.product.adjoint.saved_value_contracts
            ),
        )
        if step.library_path is None or not step.library_path.is_file():
            raise RuntimeError("inner LLVM linker produced no runtime library")
        os._exit(0)
    if args.stage == "compile-outer":
        from src.compiler.ssa_llvm_backend import compile_artifact

        motion = compile_outer(directory)
        build_directory = directory / "outer_motion"
        expected_library = build_directory / f"{motion.artifact.name}.dll"
        _write_manifest(
            _manifest_path(directory, "outer"),
            motion.artifact,
            motion.outputs,
            saved_binding_count=len(
                motion.product.adjoint.saved_value_contracts
            ),
            library_path=expected_library,
        )
        compile_artifact(motion.artifact, directory=build_directory)
        if not expected_library.is_file():
            raise RuntimeError("outer LLVM linker produced no runtime library")
        os._exit(0)
    if args.stage == "all":
        base = [
            sys.executable,
            "-m", "examples.xor_project.nested_llvm_meta_learning",
            "--build-directory", str(directory),
        ]
        subprocess.run([*base, "--stage", "compile-inner"], check=True)
        subprocess.run([*base, "--stage", "compile-outer"], check=True)
    run_demo(
        directory=directory,
        inner_learning_rate=args.inner_learning_rate,
        outer_learning_rate=args.outer_learning_rate,
        outer_steps=args.outer_steps,
        report_every=args.report_every,
        stateful_inner=not args.reset_inner_each_step,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
