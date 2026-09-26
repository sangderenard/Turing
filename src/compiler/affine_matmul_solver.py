"""Solve FusedProgram pieces as affine systems and certify matmul replacement.

The analyzer probes every step in isolation, composes locally affine state
transitions, and independently probes the entire program.  A replacement is
certified only when held-out probes satisfy the requested tolerance.

Run the demonstration with::

    python -m src.compiler.affine_matmul_solver
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from ..common.tensors.fused_ir import FusedProgram, Meta, OpStep, ordered_feed_ids
from .fused_program_python_backend import compile_single_region_python


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return np.asarray(value, dtype=np.float64)


def _size(meta: Mapping[int, Meta], value_id: int, value: Any | None = None) -> int:
    entry = meta.get(value_id)
    if entry is not None and entry.shape is not None:
        shape = tuple(int(item) for item in entry.shape)
        return int(np.prod(shape, dtype=np.int64)) if shape else 1
    if value is None:
        raise KeyError(f"value {value_id} has neither metadata nor a sample")
    return int(_array(value).size)


def _shape(meta: Mapping[int, Meta], value_id: int, value: Any | None = None) -> tuple[int, ...]:
    entry = meta.get(value_id)
    if entry is not None and entry.shape is not None:
        return tuple(int(item) for item in entry.shape)
    if value is None:
        raise KeyError(f"value {value_id} has neither metadata nor a sample")
    return tuple(_array(value).shape)


def _execute(program: FusedProgram, feeds: Mapping[int, Any], *, name: str) -> Dict[str, np.ndarray]:
    feed_ids = ordered_feed_ids(program)
    missing = set(feed_ids) - set(feeds)
    if missing:
        raise KeyError(f"{name} is missing feeds: {sorted(missing)}")
    labels = {feed_id: f"feed_{index}" for index, feed_id in enumerate(feed_ids)}
    compiled = compile_single_region_python(
        program, labels, dialect="numpy", function_name=name,
    ).callable
    raw = compiled(*(_array(feeds[feed_id]) for feed_id in feed_ids))
    values = tuple(raw) if len(program.outputs) > 1 else (raw,)
    return {
        output_name: np.asarray(value, dtype=np.float64)
        for (output_name, _), value in zip(program.outputs.items(), values)
    }


def _single_step_program(program: FusedProgram, step: OpStep) -> FusedProgram:
    return FusedProgram(
        version=program.version,
        feeds=set(step.input_ids),
        steps=[step],
        outputs={"result": step.result_id},
        meta=None if program.meta is None else dict(program.meta),
        extras=None if program.extras is None else dict(program.extras),
    )


def _flatten(values: Sequence[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in values])


def _unflatten(vector: np.ndarray, ids: Sequence[int], shapes: Mapping[int, tuple[int, ...]]) -> Dict[int, np.ndarray]:
    result: Dict[int, np.ndarray] = {}
    offset = 0
    for value_id in ids:
        shape = shapes[value_id]
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        result[value_id] = vector[offset:offset + count].reshape(shape)
        offset += count
    if offset != len(vector):
        raise ValueError("flat vector does not match declared input shapes")
    return result


@dataclass(frozen=True)
class AffinePiece:
    """One isolated step fitted as ``result = matrix @ inputs + bias``."""

    step_id: int
    operation: str
    input_ids: tuple[int, ...]
    output_id: int
    matrix: np.ndarray
    bias: np.ndarray
    maximum_error: float
    certified: bool

    @property
    def homogeneous_matrix(self) -> np.ndarray:
        matrix = np.zeros((len(self.bias) + 1, self.matrix.shape[1] + 1))
        matrix[:-1, :-1] = self.matrix
        matrix[:-1, -1] = self.bias
        matrix[-1, -1] = 1.0
        return matrix


@dataclass(frozen=True)
class MatmulReplacement:
    """A certified whole-program affine map over flattened boundary values."""

    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    input_shapes: Dict[int, tuple[int, ...]]
    output_shapes: Dict[int, tuple[int, ...]]
    matrix: np.ndarray
    bias: np.ndarray
    maximum_error: float
    certified: bool

    def __call__(self, inputs: Mapping[int, Any] | np.ndarray) -> Dict[int, np.ndarray]:
        if isinstance(inputs, Mapping):
            vector = _flatten([_array(inputs[value_id]) for value_id in self.input_ids])
        else:
            vector = np.asarray(inputs, dtype=np.float64).reshape(-1)
        output = self.matrix @ vector + self.bias
        return _unflatten(output, self.output_ids, self.output_shapes)

    @property
    def homogeneous_matrix(self) -> np.ndarray:
        """Matrix H for ``H @ [flattened_inputs, 1]``."""

        result = np.zeros((len(self.bias) + 1, self.matrix.shape[1] + 1))
        result[:-1, :-1] = self.matrix
        result[:-1, -1] = self.bias
        result[-1, -1] = 1.0
        return result

    def to_fused_program(self) -> tuple[FusedProgram, Dict[int, np.ndarray]]:
        """Materialize a one-input/one-output reshape→matmul→bias replacement."""

        if not self.certified:
            raise ValueError("an uncertified affine fit cannot replace the program")
        if len(self.input_ids) != 1 or len(self.output_ids) != 1:
            raise ValueError("FusedProgram replacement currently requires one input and one output")
        input_id = self.input_ids[0]
        output_id = self.output_ids[0]
        used = {input_id, output_id}
        next_id = max(used, default=0) + 1
        flat_id, matrix_id, product_id, bias_id, affine_id, replacement_output = range(
            next_id, next_id + 6
        )
        input_size = self.matrix.shape[1]
        output_size = self.matrix.shape[0]
        metadata = {
            input_id: Meta(shape=self.input_shapes[input_id], dtype="float64"),
            flat_id: Meta(shape=(input_size,), dtype="float64"),
            matrix_id: Meta(shape=(input_size, output_size), dtype="float64"),
            product_id: Meta(shape=(output_size,), dtype="float64"),
            bias_id: Meta(shape=(output_size,), dtype="float64"),
            affine_id: Meta(shape=(output_size,), dtype="float64"),
            replacement_output: Meta(shape=self.output_shapes[output_id], dtype="float64"),
        }
        steps = [
            OpStep(0, "reshape", [input_id], {"new_shape": (input_size,)}, flat_id),
            OpStep(1, "matmul", [flat_id, matrix_id], {}, product_id),
            OpStep(2, "add", [product_id, bias_id], {}, affine_id),
            OpStep(3, "reshape", [affine_id], {"new_shape": self.output_shapes[output_id]}, replacement_output),
        ]
        program = FusedProgram(
            version=1,
            feeds={input_id, matrix_id, bias_id},
            steps=steps,
            outputs={"matmul_replacement": replacement_output},
            meta=metadata,
            extras={"capture_feed_origins": {
                input_id: {"binding_name": "input"},
                matrix_id: {"binding_name": "affine_matrix_transposed"},
                bias_id: {"binding_name": "affine_bias"},
            }},
        )
        return program, {
            matrix_id: np.ascontiguousarray(self.matrix.T),
            bias_id: np.ascontiguousarray(self.bias),
        }


@dataclass(frozen=True)
class AffineProgramAnalysis:
    pieces: tuple[AffinePiece, ...]
    replacement: MatmulReplacement
    composed_matrix: np.ndarray | None
    composed_bias: np.ndarray | None
    composition_error: float | None

    @property
    def local_blockers(self) -> tuple[AffinePiece, ...]:
        return tuple(piece for piece in self.pieces if not piece.certified)

    @property
    def fully_replaceable(self) -> bool:
        return self.replacement.certified

    def to_mapping(self) -> dict[str, Any]:
        return {
            "fully_replaceable": self.fully_replaceable,
            "global_maximum_error": self.replacement.maximum_error,
            "input_dimension": self.replacement.matrix.shape[1],
            "output_dimension": self.replacement.matrix.shape[0],
            "piece_count": len(self.pieces),
            "local_blockers": [
                {
                    "step_id": piece.step_id,
                    "operation": piece.operation,
                    "maximum_error": piece.maximum_error,
                }
                for piece in self.local_blockers
            ],
            "composition_error": self.composition_error,
            "matrix": self.replacement.matrix.tolist(),
            "bias": self.replacement.bias.tolist(),
            "homogeneous_matrix": self.replacement.homogeneous_matrix.tolist(),
        }


def _fit_affine(
    evaluate,
    dimension: int,
    *,
    probe_count: int,
    seed: int,
    atol: float,
    rtol: float,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    zero = np.zeros((dimension,), dtype=np.float64)
    bias = np.asarray(evaluate(zero), dtype=np.float64).reshape(-1)
    matrix = np.empty((len(bias), dimension), dtype=np.float64)
    for column in range(dimension):
        basis = np.zeros((dimension,), dtype=np.float64)
        basis[column] = 1.0
        matrix[:, column] = np.asarray(evaluate(basis)).reshape(-1) - bias
    rng = np.random.default_rng(seed)
    probes = [
        rng.normal(size=dimension),
        rng.uniform(-1.5, 1.5, size=dimension),
        np.linspace(-0.75, 1.25, dimension) if dimension else zero,
    ]
    probes.extend(rng.normal(size=dimension) for _ in range(max(0, probe_count - len(probes))))
    maximum_error = 0.0
    certified = True
    for probe in probes:
        actual = np.asarray(evaluate(probe), dtype=np.float64).reshape(-1)
        predicted = matrix @ probe + bias
        error = float(np.max(np.abs(actual - predicted))) if actual.size else 0.0
        maximum_error = max(maximum_error, error)
        certified = certified and bool(np.allclose(actual, predicted, atol=atol, rtol=rtol))
    return matrix, bias, maximum_error, certified


def analyze_affine_replacement(
    program: FusedProgram,
    feed_values: Mapping[int, Any],
    *,
    variable_feed_ids: Iterable[int] | None = None,
    probe_count: int = 8,
    seed: int = 0,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> AffineProgramAnalysis:
    """Fit isolated steps and the whole program, then compare composition.

    Feeds omitted from ``variable_feed_ids`` are held as coefficients. By
    default feeds with a ``capture_feed_origins`` binding are variables; if no
    origins exist, every feed is variable.
    """

    metadata = dict(program.meta or {})
    missing = set(program.feeds) - set(feed_values)
    if missing:
        raise KeyError(f"feed_values is missing program feeds: {sorted(missing)}")
    if variable_feed_ids is None:
        origins = dict((program.extras or {}).get("capture_feed_origins", {}) or {})
        selected = {
            feed_id for feed_id in program.feeds
            if feed_id in origins or str(feed_id) in origins
        }
        variable_ids = tuple(sorted(selected or set(program.feeds)))
    else:
        variable_ids = tuple(dict.fromkeys(int(item) for item in variable_feed_ids))
    invalid = set(variable_ids) - set(program.feeds)
    if invalid:
        raise ValueError(f"variable_feed_ids are not program feeds: {sorted(invalid)}")

    store: Dict[int, np.ndarray] = {
        feed_id: _array(feed_values[feed_id]) for feed_id in program.feeds
    }
    shapes: Dict[int, tuple[int, ...]] = {
        value_id: _shape(metadata, value_id, value)
        for value_id, value in store.items()
    }
    state_ids = set(variable_ids)
    pieces: list[AffinePiece] = []
    for index, step in enumerate(program.steps):
        step_program = _single_step_program(program, step)
        input_ids = tuple(dict.fromkeys(step.input_ids))
        active_ids = tuple(value_id for value_id in input_ids if value_id in state_ids)
        for value_id in input_ids:
            shapes.setdefault(value_id, _shape(metadata, value_id, store[value_id]))
        active_shapes = {value_id: shapes[value_id] for value_id in active_ids}
        dimension = sum(_size(metadata, value_id, store[value_id]) for value_id in active_ids)

        def evaluate_piece(vector: np.ndarray) -> np.ndarray:
            boundary = {value_id: store[value_id] for value_id in input_ids}
            boundary.update(_unflatten(vector, active_ids, active_shapes))
            return _execute(step_program, boundary, name=f"affine_piece_{index}")["result"].reshape(-1)

        matrix, bias, error, certified = _fit_affine(
            evaluate_piece, dimension, probe_count=probe_count,
            seed=seed + index + 1, atol=atol, rtol=rtol,
        )
        actual = _execute(
            step_program,
            {value_id: store[value_id] for value_id in input_ids},
            name=f"sample_piece_{index}",
        )["result"]
        store[step.result_id] = actual
        shapes[step.result_id] = _shape(metadata, step.result_id, actual)
        pieces.append(AffinePiece(
            step_id=step.step_id,
            operation=step.op_name,
            input_ids=active_ids,
            output_id=step.result_id,
            matrix=matrix,
            bias=bias,
            maximum_error=error,
            certified=certified,
        ))
        state_ids.add(step.result_id)

    input_shapes = {value_id: shapes[value_id] for value_id in variable_ids}
    output_ids = tuple(program.outputs.values())
    output_shapes = {value_id: shapes[value_id] for value_id in output_ids}
    input_dimension = sum(_size(metadata, value_id, store[value_id]) for value_id in variable_ids)

    def evaluate_program(vector: np.ndarray) -> np.ndarray:
        boundary = {feed_id: store[feed_id] for feed_id in program.feeds}
        boundary.update(_unflatten(vector, variable_ids, input_shapes))
        outputs = _execute(program, boundary, name="affine_whole_program")
        return _flatten([outputs[name] for name in program.outputs])

    global_matrix, global_bias, global_error, global_certified = _fit_affine(
        evaluate_program, input_dimension, probe_count=probe_count,
        seed=seed + 10_000, atol=atol, rtol=rtol,
    )
    replacement = MatmulReplacement(
        input_ids=variable_ids,
        output_ids=output_ids,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        matrix=global_matrix,
        bias=global_bias,
        maximum_error=global_error,
        certified=global_certified,
    )

    composed_matrix: np.ndarray | None = None
    composed_bias: np.ndarray | None = None
    composition_error: float | None = None
    if all(piece.certified for piece in pieces):
        ordered_state_ids = list(variable_ids) + [
            step.result_id for step in program.steps
            if step.result_id not in variable_ids
        ]
        offsets: Dict[int, slice] = {}
        offset = 0
        for value_id in ordered_state_ids:
            count = _size(metadata, value_id, store[value_id])
            offsets[value_id] = slice(offset, offset + count)
            offset += count
        homogeneous = np.eye(offset + 1, dtype=np.float64)
        for piece in pieces:
            transition = np.eye(offset + 1, dtype=np.float64)
            output_slice = offsets[piece.output_id]
            transition[output_slice, :] = 0.0
            column = 0
            for input_id in piece.input_ids:
                input_slice = offsets[input_id]
                width = input_slice.stop - input_slice.start
                transition[output_slice, input_slice] = piece.matrix[:, column:column + width]
                column += width
            transition[output_slice, -1] = piece.bias
            homogeneous = transition @ homogeneous
        output_rows = np.concatenate([
            np.arange(offsets[value_id].start, offsets[value_id].stop)
            for value_id in output_ids
        ])
        input_columns = np.concatenate([
            np.arange(offsets[value_id].start, offsets[value_id].stop)
            for value_id in variable_ids
        ])
        composed_matrix = homogeneous[np.ix_(output_rows, input_columns)]
        composed_bias = homogeneous[output_rows, -1]
        composition_error = float(max(
            np.max(np.abs(composed_matrix - global_matrix)),
            np.max(np.abs(composed_bias - global_bias)),
        ))

    return AffineProgramAnalysis(
        pieces=tuple(pieces),
        replacement=replacement,
        composed_matrix=composed_matrix,
        composed_bias=composed_bias,
        composition_error=composition_error,
    )


def _demo_program(nonlinear: bool = False) -> tuple[FusedProgram, Dict[int, np.ndarray]]:
    metadata = {
        1: Meta((3,), "float64"),
        2: Meta((3,), "float64"),
        3: Meta((3,), "float64"),
        4: Meta((3,), "float64"),
        5: Meta((3,), "float64"),
        6: Meta((3,), "float64"),
    }
    steps = [
        OpStep(0, "mul", [1, 2], {}, 4),
        OpStep(1, "add", [4, 3], {}, 5),
        OpStep(2, "mul" if nonlinear else "neg", [5, 5] if nonlinear else [5], {}, 6),
    ]
    return FusedProgram(
        version=1,
        feeds={1, 2, 3},
        steps=steps,
        outputs={"result": 6},
        meta=metadata,
        extras={"capture_feed_origins": {1: {"binding_name": "x"}}},
    ), {
        1: np.asarray((0.2, -0.4, 0.7)),
        2: np.asarray((2.0, -1.0, 0.5)),
        3: np.asarray((1.0, 3.0, -2.0)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nonlinear", action="store_true")
    args = parser.parse_args(argv)
    program, feeds = _demo_program(args.nonlinear)
    analysis = analyze_affine_replacement(program, feeds, variable_feed_ids=(1,))
    print(json.dumps(analysis.to_mapping(), indent=2))
    return 0 if analysis.fully_replaceable else 2


if __name__ == "__main__":  # pragma: no cover - exercised as a runnable
    raise SystemExit(main())


__all__ = [
    "AffinePiece", "AffineProgramAnalysis", "MatmulReplacement",
    "analyze_affine_replacement", "main",
]
