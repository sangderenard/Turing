"""Math lab: the user writes math; the compiler's own batch pipeline makes products.

    r = sqrt(x*x + y*y)
    e = x*y + 2*z

goes through, exactly as spelled elsewhere in this tree:

    parse_math                          whitelist parser (ast; never eval/exec of user text)
    compile_sympy_equations             sympy equations -> repository SSA
    symbolic_abstract_tensor_source     the AbstractTensor stage, as Python source (law.at.py)
    autograd.forward_capture + emit_tape_source(backend="numpy")
                                        the NumPy product given to the user (law.numpy.py)
    graph_wasm._compile                 AT callable traced under SSATensorProgram ->
                                        compile_graph_forward_to_wasm (law.wasm + law.abi.json)
    ssa_webgpu_backend.emit_tensor_program_passes
                                        WGSL under the `fast` contract (law.wgsl, law.api.yaml,
                                        law.wgsl.abi.json)
    native_package.piece_from_law       batched native library (native/*.dll, law.native.piece)

Every target is attempted on its own: a refusal on one is recorded in
``products.json`` (schema ``turing.math-lab-products.v1``) and never blocks the
others.  ``profile_server`` times the sympy lambdify reference, the NumPy
product and the native product on the server; the wasm and WGSL products are
timed in the visitor's browser by the page, not here.

    python examples/meowdoku/math_lab.py compile --text "r = sqrt(x*x+y*y)" --out DIR
    python examples/meowdoku/math_lab.py profile DIR
    python examples/meowdoku/math_lab.py run --text "..." --out DIR
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import importlib.util
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _path in (str(ROOT), str(ROOT / "examples")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import sympy

PRODUCTS_SCHEMA = "turing.math-lab-products.v1"
PROFILE_SCHEMA = "turing.math-lab-profile.v1"
WGSL_SCHEMA = "turing.math-lab-wgsl.v1"
ALL_TARGETS = ("numpy", "wasm", "wgsl", "native")

MAX_LINES = 32
MAX_CHARS = 4096
MAX_DEPTH = 64
MAX_SYMBOLS = 64
MAX_EXPONENT = 16
MAX_NAME = 32

# name -> (sympy function, argument count)
FUNCTIONS: dict[str, tuple[Callable[..., Any], int]] = {
    "sqrt": (sympy.sqrt, 1), "sin": (sympy.sin, 1), "cos": (sympy.cos, 1),
    "exp": (sympy.exp, 1), "log": (sympy.log, 1), "abs": (sympy.Abs, 1),
    "atan2": (sympy.atan2, 2), "tanh": (sympy.tanh, 1),
    "min": (sympy.Min, 2), "max": (sympy.Max, 2), "floor": (sympy.floor, 1),
}
# Measured refusals, by name (probe: every target refuses at the symbolic stage).
REFUSED_FUNCTIONS = {
    "tan": "the compiler refuses Tan: no exact Python spelling (compile_sympy_equations)",
}
# An argument named like a generated temporary is emitted WRONG by the AbstractTensor
# stage (probe: input `t1` -> `t1 = 2.0; t2 = t1 * t1`), so these names are refused.
_RESERVED_NAME = re.compile(r"^(?:[tv]\d+|np|AbstractTensor|law|self|math)$")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_PYTHON_KEYWORDS = frozenset(__import__("keyword").kwlist)


class MathSyntaxError(ValueError):
    """The text is outside the whitelisted math language."""

    def __init__(self, message: str, line: int | None = None, col: int | None = None,
                 construct: str | None = None):
        where = f" (line {line}, col {col})" if line is not None else ""
        super().__init__(f"{message}{where}")
        self.reason, self.line, self.col, self.construct = message, line, col, construct


# ------------------------------------------------------------------ parser

def _pos(node: ast.AST, line_no: int) -> tuple[int, int]:
    return line_no, int(getattr(node, "col_offset", 0)) + 1


def _construct(node: ast.AST) -> str:
    return type(node).__name__


def _name_ok(name: str, line_no: int, col: int, what: str) -> None:
    if (len(name) > MAX_NAME or not _NAME.match(name) or name in _PYTHON_KEYWORDS
            or name in FUNCTIONS or name in REFUSED_FUNCTIONS):
        raise MathSyntaxError(
            f"{what} name {name!r} is not allowed (ASCII letter then letters/digits/_, "
            f"<= {MAX_NAME} chars, not a keyword or function name)", line_no, col, "Name")
    if _RESERVED_NAME.match(name):
        raise MathSyntaxError(
            f"{what} name {name!r} is reserved (collides with generated temporaries)",
            line_no, col, "Name")


def _integer_exponent(node: ast.AST, line_no: int) -> int:
    sign = 1
    inner = node
    if isinstance(inner, ast.UnaryOp) and isinstance(inner.op, (ast.USub, ast.UAdd)):
        sign = -1 if isinstance(inner.op, ast.USub) else 1
        inner = inner.operand
    if not (isinstance(inner, ast.Constant) and type(inner.value) is int):
        raise MathSyntaxError("exponent must be an integer literal", *_pos(node, line_no),
                              _construct(node))
    value = sign * inner.value
    if abs(value) > MAX_EXPONENT:
        raise MathSyntaxError(f"exponent {value} exceeds |n| <= {MAX_EXPONENT}",
                              *_pos(node, line_no), "Pow")
    return value


def _build(node: ast.AST, line_no: int, symbols: dict[str, sympy.Symbol],
           outputs: set[str], depth: int) -> sympy.Expr:
    if depth > MAX_DEPTH:
        raise MathSyntaxError(f"expression deeper than {MAX_DEPTH}", *_pos(node, line_no),
                              _construct(node))
    if isinstance(node, ast.Constant):
        value = node.value
        if type(value) is int:
            if abs(value) >= 2 ** 53:
                raise MathSyntaxError("integer constant too large", *_pos(node, line_no), "Constant")
            return sympy.Integer(value)
        if type(value) is float:
            if value != value or value in (float("inf"), float("-inf")):
                raise MathSyntaxError("non-finite constant", *_pos(node, line_no), "Constant")
            return sympy.Float(value)
        raise MathSyntaxError(f"constant of type {type(value).__name__} is not allowed",
                              *_pos(node, line_no), "Constant")
    if isinstance(node, ast.Name):
        if node.id in outputs:
            raise MathSyntaxError(
                f"output {node.id!r} is used on a right-hand side; all equations step "
                "simultaneously from the inputs, so an output cannot feed another equation",
                *_pos(node, line_no), "Name")
        if node.id in FUNCTIONS or node.id in REFUSED_FUNCTIONS:
            raise MathSyntaxError(f"function name {node.id!r} used without a call",
                                  *_pos(node, line_no), "Name")
        _name_ok(node.id, *_pos(node, line_no), "input")
        if node.id not in symbols:
            if len(symbols) >= MAX_SYMBOLS:
                raise MathSyntaxError(f"more than {MAX_SYMBOLS} symbols", *_pos(node, line_no), "Name")
            symbols[node.id] = sympy.Symbol(node.id)
        return symbols[node.id]
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise MathSyntaxError(f"operator {_construct(node.op)} is not allowed",
                                  *_pos(node, line_no), _construct(node.op))
        operand = _build(node.operand, line_no, symbols, outputs, depth + 1)
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            base = _build(node.left, line_no, symbols, outputs, depth + 1)
            return base ** sympy.Integer(_integer_exponent(node.right, line_no))
        ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
               ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}
        for kind, apply in ops.items():
            if isinstance(node.op, kind):
                return apply(_build(node.left, line_no, symbols, outputs, depth + 1),
                             _build(node.right, line_no, symbols, outputs, depth + 1))
        raise MathSyntaxError(f"operator {_construct(node.op)} is not allowed (use + - * / **)",
                              *_pos(node, line_no), _construct(node.op))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise MathSyntaxError(f"only plain named function calls are allowed, not a call of {_construct(node.func)}",
                                  *_pos(node, line_no), "Call")
        name = node.func.id
        if name in REFUSED_FUNCTIONS:
            raise MathSyntaxError(f"function {name!r} refused: {REFUSED_FUNCTIONS[name]}",
                                  *_pos(node, line_no), "Call")
        if name not in FUNCTIONS:
            raise MathSyntaxError(
                f"function {name!r} is not in the table ({', '.join(sorted(FUNCTIONS))})",
                *_pos(node, line_no), "Call")
        if node.keywords:
            raise MathSyntaxError("keyword arguments are not allowed", *_pos(node, line_no), "Call")
        function, arity = FUNCTIONS[name]
        if len(node.args) != arity:
            raise MathSyntaxError(f"{name} takes {arity} argument(s), got {len(node.args)}",
                                  *_pos(node, line_no), "Call")
        return function(*(_build(a, line_no, symbols, outputs, depth + 1) for a in node.args))
    raise MathSyntaxError(f"{_construct(node)} is not allowed", *_pos(node, line_no), _construct(node))


def parse_math(text: str) -> list[sympy.Equality]:
    """``name = expression`` lines to sympy equations.  Never evals the text."""

    if not isinstance(text, str):
        raise MathSyntaxError("math text must be a string")
    if len(text) > MAX_CHARS:
        raise MathSyntaxError(f"input is {len(text)} characters; the limit is {MAX_CHARS}")
    lines = [(i, line.strip()) for i, line in enumerate(text.splitlines(), 1)]
    lines = [(i, line) for i, line in lines if line and not line.startswith("#")]
    if not lines:
        raise MathSyntaxError("no equations")
    if len(lines) > MAX_LINES:
        raise MathSyntaxError(f"{len(lines)} lines; the limit is {MAX_LINES}")
    # Pass 1: targets, so a right-hand-side use of an output is recognised wherever it sits.
    parsed: list[tuple[int, str, ast.expr]] = []
    for line_no, line in lines:
        try:
            module = ast.parse(line)
        except (SyntaxError, RecursionError, MemoryError, ValueError) as error:
            col = getattr(error, "offset", None)
            raise MathSyntaxError(f"cannot parse: {type(error).__name__}", line_no, col, "Syntax") from None
        if len(module.body) != 1 or not isinstance(module.body[0], ast.Assign):
            raise MathSyntaxError("each line must be exactly `name = expression`", line_no, 1,
                                  _construct(module.body[0]) if module.body else "Empty")
        statement = module.body[0]
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            raise MathSyntaxError("the left side must be a single plain name", line_no, 1, "Assign")
        target = statement.targets[0].id
        _name_ok(target, line_no, 1, "output")
        parsed.append((line_no, target, statement.value))
    outputs = [target for _, target, _ in parsed]
    for (line_no, target, _), seen in zip(parsed, range(len(parsed))):
        if outputs.index(target) != seen:
            raise MathSyntaxError(f"output {target!r} is defined twice", line_no, 1, "Assign")
    output_set = set(outputs)
    symbols: dict[str, sympy.Symbol] = {}
    equations = []
    for line_no, target, value in parsed:
        try:
            rhs = _build(value, line_no, symbols, output_set, 1)
        except RecursionError:
            raise MathSyntaxError(f"expression deeper than {MAX_DEPTH}", line_no, 1, "depth") from None
        if not rhs.free_symbols:
            raise MathSyntaxError(f"equation for {target!r} has no input symbol", line_no, 1, "Assign")
        equations.append(sympy.Eq(sympy.Symbol(target), rhs))
    return equations


# ------------------------------------------------------------------ products

@dataclasses.dataclass
class MathProducts:
    out_dir: Path
    record: dict
    equations: list = dataclasses.field(default_factory=list, repr=False)

    @property
    def inputs(self) -> list[str]:
        return list(self.record["inputs"])

    @property
    def outputs(self) -> list[str]:
        return list(self.record["outputs"])

    def target(self, name: str) -> dict:
        return self.record["targets"][name]

    def path(self, name: str) -> Path | None:
        file = self.record["targets"].get(name, {}).get("file")
        return None if not file else self.out_dir / file

    def to_dict(self) -> dict:
        return self.record


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first_line(error: BaseException) -> str:
    text = str(error).strip().splitlines()
    return (text[0] if text else "")[:400]


def _sample(names, rows: int, seed: int, dtype=np.float64) -> dict[str, np.ndarray]:
    """Positive, bounded columns so sqrt/log/pow(-n) laws stay in domain."""
    rng = np.random.default_rng(seed)
    return {n: rng.uniform(0.5, 2.0, rows).astype(dtype) for n in names}


def _reference(equations, inputs, outputs) -> Callable[..., tuple]:
    by_name = {str(eq.lhs): eq.rhs for eq in equations}
    function = sympy.lambdify([sympy.Symbol(n) for n in inputs],
                              [by_name[n] for n in outputs], modules="numpy", cse=True)

    def evaluate(*columns):
        return tuple(np.broadcast_to(np.asarray(v, dtype=np.float64), np.shape(columns[0])).copy()
                     for v in function(*columns))
    return evaluate


def _as_tuple(result) -> tuple:
    return tuple(result) if isinstance(result, (tuple, list)) else (result,)


def _parity(got: tuple, ref: tuple) -> dict:
    max_abs, max_ulp = 0.0, 0.0
    for a, b in zip(got, ref):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        finite = np.isfinite(a) & np.isfinite(b)
        if not finite.any():
            continue
        diff = np.abs(a[finite] - b[finite])
        max_abs = max(max_abs, float(diff.max()))
        spacing = np.spacing(np.abs(b[finite]))
        spacing[spacing == 0] = np.spacing(0.0)
        max_ulp = max(max_ulp, float((diff / spacing).max()))
    return {"max_abs_err": max_abs, "max_ulp_err": max_ulp}


def _load_numpy_product(path: Path, name: str) -> Callable:
    spec = importlib.util.spec_from_file_location(f"math_lab_{path.parent.name}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


class _Stage:
    def __init__(self):
        self.name = "start"

    def __call__(self, name: str):
        self.name = name


def _attempt(function: Callable[[_Stage], dict]) -> dict:
    stage = _Stage()
    started = time.perf_counter()
    try:
        record = function(stage)
        record.update({"state": "complete", "stage": stage.name})
    except BaseException as error:  # noqa: BLE001 - a refusal is a record, not a crash
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        record = {"state": "refused", "stage": stage.name, "error_type": type(error).__name__,
                  "error": _first_line(error)}
    record["seconds"] = round(time.perf_counter() - started, 3)
    return record


def _file_fields(out_dir: Path, path: Path) -> dict:
    return {"file": path.relative_to(out_dir).as_posix(), "sha256": _sha(path),
            "bytes": path.stat().st_size}


def compile_math(text: str, out_dir, *, name: str = "user_law", batch: int = 4096,
                 targets=ALL_TARGETS, dtype: str = "float64") -> MathProducts:
    """Run the chain for ``text`` into ``out_dir``; see the module docstring."""

    if dtype != "float64":
        raise ValueError("math lab compiles float64 laws")
    if not _NAME.match(name) or _RESERVED_NAME.match(name):
        raise ValueError(f"bad law name {name!r}")
    batch = int(batch)
    if batch < 1:
        raise ValueError("batch must be positive")
    unknown = [t for t in targets if t not in ALL_TARGETS]
    if unknown:
        raise ValueError(f"unknown targets {unknown}; choose from {ALL_TARGETS}")
    equations = parse_math(text)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema": PRODUCTS_SCHEMA, "name": name, "batch": batch, "dtype": dtype,
        "text": text, "equations": [f"{eq.lhs} = {eq.rhs}" for eq in equations],
        "inputs": [], "outputs": [], "files": {}, "targets": {},
    }
    state: dict[str, Any] = {}

    def symbolic(stage: _Stage) -> dict:
        from src.compiler.symbolic_equation_compiler import compile_sympy_equations
        from src.compiler.vehicle_python_compilation import (
            _abstract_tensor_stage_callable, symbolic_abstract_tensor_source)
        stage("compile_sympy_equations")
        compilation = compile_sympy_equations(equations, name=name, dtype=dtype)
        stage("symbolic_abstract_tensor_source")
        source = symbolic_abstract_tensor_source(compilation, name)
        callable_ = _abstract_tensor_stage_callable(compilation, name)
        header = (f'"""{name}: AbstractTensor stage of the user law (batch columns in, batch columns out).\n\n'
                  + "\n".join(f"    {line}" for line in record["equations"]) + '\n"""\n'
                  "from src.common.tensors.abstraction import AbstractTensor\n\n\n")
        path = out_dir / "law.at.py"
        path.write_text(header + source, encoding="utf-8", newline="\n")
        state.update(compilation=compilation, callable=callable_)
        record["inputs"] = [str(n) for n in compilation.function.metadata["argument_names"]]
        record["outputs"] = [str(n) for n in compilation.function.metadata["output_names"]]
        return _file_fields(out_dir, path)

    record["symbolic"] = _attempt(symbolic)
    symbolic_failed = record["symbolic"]["state"] != "complete"
    if not symbolic_failed:
        record["files"]["at"] = record["symbolic"]["file"]
    inputs, outputs = record["inputs"], record["outputs"]

    def numpy_target(stage: _Stage) -> dict:
        from src.common.tensors.accelerator_backends.tape_to_source import emit_tape_source
        from src.common.tensors.autograd import autograd
        from src.common.tensors.numpy_backend import NumPyTensorOperations
        stage("forward_capture")
        columns = _sample(inputs, batch, 0)
        sample = {n: NumPyTensorOperations.tensor(columns[n]) for n in inputs}
        with autograd.forward_capture() as tape:
            produced = _as_tuple(state["callable"](*sample.values()))
        stage("emit_tape_source")
        source = emit_tape_source(tape, sample, dict(zip(outputs, produced)),
                                  backend="numpy", function_name=name)
        header = (f'"""{name}: compiled NumPy product of the user law.\n\n'
                  + "\n".join(f"    {line}" for line in record["equations"])
                  + f"\n\nCall {name}({', '.join(inputs)}) with equal-length float64 arrays; returns the "
                  f"outputs ({', '.join(outputs)}) in that order.\nGenerated by AbstractTensor tape tracing "
                  'and tape_to_source (numpy dialect); regenerate rather than edit.\n"""\n')
        path = out_dir / "law.numpy.py"
        path.write_text(header + source, encoding="utf-8", newline="\n")
        stage("selfcheck")
        probe = _sample(inputs, 256, 1)
        got = _as_tuple(_load_numpy_product(path, name)(*(probe[n] for n in inputs)))
        parity = _parity(got, _reference(equations, inputs, outputs)(*(probe[n] for n in inputs)))
        return {**_file_fields(out_dir, path), "precision": "float64",
                "selfcheck_rows": 256, **{f"selfcheck_{k}": v for k, v in parity.items()}}

    def wasm_target(stage: _Stage) -> dict:
        from meowdoku import graph_wasm
        contract = "develop"
        stage("trace+compile_graph_forward_to_wasm")
        original = graph_wasm.TRACE_HOME
        graph_wasm.TRACE_HOME = out_dir / "_wasm_trace"
        try:
            abi = graph_wasm._compile(
                "law", {n: (batch,) for n in inputs},
                lambda b: state["callable"](*(b[n] for n in inputs)),
                outputs, contract, out_dir, {"batch": batch, "name": name})
        finally:
            graph_wasm.TRACE_HOME = original
            shutil.rmtree(out_dir / "_wasm_trace", ignore_errors=True)
        stage("write")
        wasm = out_dir / "law.wasm"
        if wasm.read_bytes()[:4] != b"\0asm":
            raise RuntimeError("emitted file is not WebAssembly")
        record["files"]["wasm_abi"] = "law.abi.json"
        return {**_file_fields(out_dir, wasm), "abi_file": "law.abi.json", "contract": contract,
                "precision": "/".join(sorted(set(abi.get("dtypes", [])))),
                "imports": "host math (libm) where the law needs it"}

    def wgsl_target(stage: _Stage) -> dict:
        from src.common.tensors.accelerator_backends.ssa_backend import (
            SSATensorOperations, SSATensorProgram)
        from src.compiler import ssa_webgpu_backend as webgpu
        from src.compiler.work_contract import PRESETS, scoped_work_contract
        contract = "fast"
        stage("trace")
        with scoped_work_contract(PRESETS[contract]):
            program = SSATensorProgram(name)
            bindings = {n: SSATensorOperations.input(program, (batch,)) for n in inputs}
            with program.activate():
                produced = _as_tuple(state["callable"](*(bindings[n] for n in inputs)))
            values = [item.data.value for item in produced]
            stage("emit_tensor_program_passes")
            compiled = webgpu.emit_tensor_program_passes(program.module, name, values, count=batch,
                                                         name=f"{name}_wgsl")
        if not compiled.complete:
            raise RuntimeError("WGSL incomplete: " + "; ".join(s.format() for s in compiled.shortfalls))
        stage("write")
        ids = {n: int(bindings[n].data.value.id) for n in inputs}
        out_ids = [int(v.id) for v in values]
        wgsl = out_dir / "law.wgsl"
        wgsl.write_text(compiled.source, encoding="utf-8", newline="\n")
        api = out_dir / "law.api.yaml"
        compiled.api.write(api)
        abi = {
            "schema": WGSL_SCHEMA, "contract": contract, "precision": "float32", "name": name,
            "count": batch, "workgroup_size": list(compiled.workgroup_size),
            "groups": list(compiled.groups), "entries": list(compiled.entries),
            "slots": compiled.slots, "dispatches": compiled.dispatches,
            "lane_base": compiled.lane_base, "arena_elements": compiled.arena_elements,
            "bindings": compiled.bindings, "slots_per_binding": compiled.slots_per_binding,
            "storage_bindings_per_pass": compiled.bindings + 1,
            "binding_budget": webgpu.MAX_STORAGE_BINDINGS_PER_STAGE,
            "arguments": inputs, "outputs": outputs,
            "feed_slots": {n: compiled.feed_slots[ids[n]] for n in inputs if ids[n] in compiled.feed_slots},
            "output_slots": {n: compiled.output_slots[i] for n, i in zip(outputs, out_ids)},
            "backend_identities": [d.as_record() for d in compiled.backend_identity_decisions],
        }
        abi_path = out_dir / "law.wgsl.abi.json"
        abi_path.write_text(json.dumps(abi, indent=1), encoding="utf-8")
        record["files"]["wgsl_abi"] = abi_path.name
        record["files"]["wgsl_api"] = api.name
        return {**_file_fields(out_dir, wgsl), "abi_file": abi_path.name, "api_file": api.name,
                "contract": contract, "precision": "float32", "passes": len(compiled.entries),
                "dispatches": compiled.dispatches}

    def native_target(stage: _Stage) -> dict:
        from src.compiler.native_package import piece_from_law
        stage("piece_from_law")
        piece = piece_from_law(state["compilation"], name, batch, directory=out_dir / "native")
        library = Path(piece.artifact.library_path)
        stage("save piece")
        piece_path = out_dir / "law.native.piece"
        piece.save(piece_path, retain_compilation=False)
        record["files"]["native_piece"] = piece_path.name
        return {**_file_fields(out_dir, library), "piece_file": piece_path.name,
                "piece_sha256": _sha(piece_path), "precision": "float64", "batch": batch,
                "route": "native_package.piece_from_law (batched LLVM)"}

    runners = {"numpy": numpy_target, "wasm": wasm_target, "wgsl": wgsl_target, "native": native_target}
    for target in targets:
        if symbolic_failed:
            record["targets"][target] = {
                "state": "refused", "stage": record["symbolic"]["stage"],
                "error_type": record["symbolic"]["error_type"], "error": record["symbolic"]["error"],
                "seconds": 0.0}
            continue
        record["targets"][target] = _attempt(runners[target])
    (out_dir / "products.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return MathProducts(out_dir, record, equations)


# ------------------------------------------------------------------ profile

def _median_ns_per_row(run: Callable[[], Any], rows: int, repeats: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        run()
    times = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        run()
        times.append(time.perf_counter_ns() - started)
    return statistics.median(times) / rows


def profile_server(products, rows=(1_000, 100_000), repeats: int = 20, seed: int = 0) -> dict:
    """Server-side timing and parity: sympy lambdify (reference), the NumPy
    product and the native product.  wasm/WGSL are timed in the browser."""

    if not isinstance(products, MathProducts):
        directory = Path(products)
        record = json.loads((directory / "products.json").read_text(encoding="utf-8"))
        products = MathProducts(directory, record, parse_math(record["text"]))
    record, out_dir = products.record, products.out_dir
    inputs, outputs, batch = record["inputs"], record["outputs"], int(record["batch"])
    reference = _reference(products.equations, inputs, outputs)
    runners: dict[str, Callable[..., tuple] | dict] = {"sympy_lambdify": reference}
    native_kernel_ns = None
    refused: dict[str, str] = {}
    numpy_state = record["targets"].get("numpy", {})
    if numpy_state.get("state") == "complete":
        runners["numpy_product"] = _load_numpy_product(out_dir / numpy_state["file"], record["name"])
    else:
        refused["numpy_product"] = numpy_state.get("error", "not compiled")
    native_state = record["targets"].get("native", {})
    if native_state.get("state") == "complete":
        try:
            from src.compiler.native_law_kernels import LLVMPiece
            piece = LLVMPiece.load(out_dir / native_state["piece_file"], retain_compilation=False)

            def native(*columns):
                # The piece is fixed at ``batch`` rows and ignores extra rows: chunk, padding the last.
                count = len(columns[0])
                pieces = []
                for start in range(0, count, batch):
                    chunk = [np.resize(c[start:start + batch], batch) for c in columns]
                    pieces.append(piece(*chunk))
                return tuple(np.concatenate([p[k] for p in pieces])[:count] for k in range(len(outputs)))
            runners["native_product"] = native
            kernel_columns = _sample(inputs, batch, seed)
            from src.compiler.ssa_llvm_backend import prepare_artifact_execution
            execution = prepare_artifact_execution(piece.artifact, {
                value_id: kernel_columns[n].copy() for value_id, n in zip(piece.argument_ids, piece.argument_names)})
            native_kernel_ns = _median_ns_per_row(execution.run, batch, max(repeats, 50), warmup=5)
        except Exception as error:  # noqa: BLE001
            refused["native_product"] = f"{type(error).__name__}: {_first_line(error)}"
    else:
        refused["native_product"] = native_state.get("error", "not compiled")
    result: dict[str, Any] = {
        "schema": PROFILE_SCHEMA, "name": record["name"], "inputs": inputs, "outputs": outputs,
        "batch": batch, "repeats": repeats, "seed": seed, "reference": "sympy_lambdify",
        "timing": "median wall ns per row after 3 warm-up runs; native is called in "
                  f"chunks of {batch} rows (last chunk padded), each call preparing its execution",
        "rows": {}, "refused": refused,
        "native_kernel_only_ns_per_row": native_kernel_ns,
        "native_kernel_only_note": f"execution.run() on buffers already bound, {batch} rows; excludes per-call preparation",
        "browser_targets": "wasm and WGSL are timed in the visitor's browser by the page",
    }
    for count in rows:
        columns = _sample(inputs, int(count), seed)
        args = [columns[n] for n in inputs]
        ref = reference(*args)
        entry: dict[str, Any] = {}
        for label, run in runners.items():
            got = _as_tuple(run(*args))
            entry[label] = {
                "ns_per_row": _median_ns_per_row(lambda run=run: run(*args), int(count), repeats),
                **_parity(got, ref)}
        base = entry["sympy_lambdify"]["ns_per_row"]
        for label, item in entry.items():
            item["speedup_vs_sympy"] = base / item["ns_per_row"]
        result["rows"][str(count)] = entry
    (out_dir / "profile.server.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


# ------------------------------------------------------------------ subprocess job

def run_job(text: str, out_dir, timeout_s: float = 120, **compile_options) -> dict:
    """Compile + profile in a subprocess with a hard timeout (a queue worker can never hang)."""

    out_dir = Path(out_dir)
    try:
        parse_math(text)
    except MathSyntaxError as error:
        return {"state": "rejected", "error_type": "MathSyntaxError", "error": str(error)}
    out_dir.mkdir(parents=True, exist_ok=True)
    source = out_dir / "law.math.txt"
    source.write_text(text, encoding="utf-8")
    command = [sys.executable, str(Path(__file__).resolve()), "run", "--text-file", str(source),
               "--out", str(out_dir)]
    for key, value in compile_options.items():
        command += [f"--{key.replace('_', '-')}", str(value) if not isinstance(value, (list, tuple))
                    else ",".join(value)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace", cwd=str(ROOT))
    try:
        output, _ = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
        else:
            process.kill()
        process.communicate()
        return {"state": "timeout", "timeout_s": timeout_s}
    products_path, profile_path = out_dir / "products.json", out_dir / "profile.server.json"
    if process.returncode != 0 or not products_path.exists():
        return {"state": "error", "returncode": process.returncode,
                "error": "\n".join(output.strip().splitlines()[-6:])[-1200:]}
    result = {"state": "complete", "products": json.loads(products_path.read_text(encoding="utf-8"))}
    if profile_path.exists():
        result["profile"] = json.loads(profile_path.read_text(encoding="utf-8"))
    return result


# ------------------------------------------------------------------ CLI

def _targets(value: str):
    return tuple(t for t in value.split(",") if t)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("compile", "run"):
        p = sub.add_parser(command)
        p.add_argument("--text")
        p.add_argument("--text-file")
        p.add_argument("--out", required=True)
        p.add_argument("--name", default="user_law")
        p.add_argument("--batch", type=int, default=4096)
        p.add_argument("--targets", type=_targets, default=ALL_TARGETS)
        if command == "run":
            p.add_argument("--rows", default="1000,100000")
            p.add_argument("--repeats", type=int, default=20)
    p = sub.add_parser("profile")
    p.add_argument("directory")
    p.add_argument("--rows", default="1000,100000")
    p.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    if args.command == "profile":
        result = profile_server(args.directory, rows=tuple(int(r) for r in args.rows.split(",")),
                                repeats=args.repeats)
        print(json.dumps(result, indent=1))
        return 0
    text = args.text if args.text is not None else Path(args.text_file).read_text(encoding="utf-8")
    try:
        products = compile_math(text, args.out, name=args.name, batch=args.batch, targets=args.targets)
    except MathSyntaxError as error:
        print(f"MathSyntaxError: {error}")
        return 2
    print(json.dumps({k: {"state": v["state"], "seconds": v["seconds"], **({"error": v["error"]} if "error" in v else {})}
                      for k, v in products.record["targets"].items()}, indent=1))
    if args.command == "run":
        print(json.dumps(profile_server(products, rows=tuple(int(r) for r in args.rows.split(",")),
                                        repeats=args.repeats), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
