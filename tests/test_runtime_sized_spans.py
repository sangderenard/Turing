"""A span that states its rank but not its extents compiles as a tensor.

``program_abi`` may declare a parameter ``storage: span`` with ``rank: R`` and
no ``shape``: the extents arrive with each call, in the ``extents`` vector of
``void entry(void **buffers, <int> *extents)``.  This is the boundary a host
that owns its own tensor handles uses -- it hands Turing raw buffers and
extents and nothing else -- so Turing must neither guess a shape nor fall back
to scalar arithmetic.

Wrong record (main 364f179): ``def f(x, y): return x * y`` with rank-1
rank-only spans lowered as one scalar multiply, and the C, LLVM and Fortran
products all returned ``x[0] * y[0]``, reported complete.  The rule now pinned
here: an elementwise result over runtime-sized operands of one rank is a
runtime-sized span of that rank (``_tensor_descriptor_rule``), each backend
loops over the extents the host supplies, operands whose extents disagree
raise validation error 77, and the artifact states its calling contract
(``runtime_abi``) so the host can allocate the result.
"""
from __future__ import annotations

import pathlib
import tempfile
import warnings

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

REPO = pathlib.Path(__file__).resolve().parents[1]
BASE_CONTRACT = REPO / "extraction_contracts" / "program_extraction.yaml"
TENSOR_TYPE = "src.common.tensors.abstraction.AbstractTensor"


def _contract(tmp_path, name, rank, shape=None, parameters=("x", "y")):
    raw = yaml.safe_load(BASE_CONTRACT.read_text(encoding="utf-8"))
    values = []
    for parameter in parameters:
        if rank == 0:
            values.append({
                "function": "f", "parameter": parameter, "storage": "scalar",
                "dtype": "float64", "python_type": "builtins.float",
            })
            continue
        value = {
            "function": "f", "parameter": parameter, "storage": "span",
            "dtype": "float64", "rank": rank, "python_type": TENSOR_TYPE,
        }
        if shape is not None:
            value["shape"] = list(shape)
        values.append(value)
    raw["program_abi"] = {"records": {}, "bindings": [], "values": values}
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return ExtractionContract(path)


def _lower(tmp_path, source, name, rank, shape=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, exports = lower_ast_source_to_ssa(
            source, "f", name=name,
            extraction_contract=_contract(tmp_path, name, rank, shape),
            progress=lambda _: None,
        )
    root = module.functions[exports[0]]
    returned = next(
        instruction.args for block in root.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
    )
    parameters = dict(root.metadata["parameter_names"])
    return module, exports[0], parameters, returned[0]


def _run(backend, module, entry, feeds, tmp_path):
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, entry)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(pathlib.Path(tempfile.mkdtemp(dir=tmp_path)) / "n")
        return artifact, artifact.prepare_execution(feeds).run()
    if backend == "llvm":
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        artifact = emit_ssa_function_to_llvm(module, entry)
        assert artifact.complete, [s.reason for s in artifact.shortfalls]
        artifact = compile_artifact(artifact)
        return artifact, prepare_artifact_execution(artifact, feeds).run()
    from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
    from src.compiler.ssa_fortran_backend import FortranCoreNative, emit_module

    fortran = emit_module(module, progress=lambda _: None)
    assert getattr(fortran, "complete", True), fortran.shortfalls
    built = compile_fortran_module_c_shell(
        fortran, {}, pathlib.Path(tempfile.mkdtemp(dir=tmp_path)),
        library=True, entrypoint=entry, name=entry[:48], standalone=False,
    )
    record = next(
        item for item in fortran.api.entry_points if str(item.name) == entry
    )
    native = FortranCoreNative(built.executable_path, record, fortran.api)
    return fortran, native.prepare_execution(feeds).run()


def _backends():
    import shutil

    found = ["c", "llvm"]
    if shutil.which("gfortran"):
        found.append("fortran")
    return found


BACKENDS = _backends()


def _lower_integer_span(tmp_path, source, name, shape=None):
    raw = yaml.safe_load(BASE_CONTRACT.read_text(encoding="utf-8"))
    span = {"function": "f", "parameter": "x", "storage": "span",
            "dtype": "int64", "rank": 1, "python_type": TENSOR_TYPE}
    if shape is not None:
        span["shape"] = list(shape)
    raw["program_abi"] = {"records": {}, "bindings": [], "values": [
        span,
        {"function": "f", "parameter": "n", "storage": "scalar",
         "dtype": "int64", "rank": 0, "shape": [],
         "python_type": "builtins.int"},
    ]}
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, exports = lower_ast_source_to_ssa(
            source, "f", name=name,
            extraction_contract=ExtractionContract(path),
            progress=lambda _: None,
        )
    entry = exports[0]
    root = module.functions[entry]
    parameters = dict(root.metadata["parameter_names"])
    returned = next(
        instruction.args[0] for block in root.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
    )
    return module, entry, parameters, returned


@pytest.mark.parametrize("expression,expected", [
    ("x * n", [10, -15, 20, 25]),
    ("x - n", [-3, -8, -1, 0]),
    ("x % n", [2, 2, 4, 0]),
])
@pytest.mark.parametrize("shape", [None, (4,)])
def test_integer_span_scalar_arithmetic_matches_c_and_llvm(
    tmp_path, expression, expected, shape,
):
    module, entry, parameters, result = _lower_integer_span(
        tmp_path, f"def f(x, n):\n    return {expression}\n",
        "integer_span_arithmetic", shape,
    )
    assert result.dtype == "int64"
    feeds = {parameters["x"]: np.array([2, -3, 4, 5], dtype=np.int64),
             parameters["n"]: np.array([5], dtype=np.int64)}
    for backend in ("c", "llvm"):
        artifact, execution = _run(backend, module, entry, feeds, tmp_path)
        assert np.asarray(execution.buffers[result.id]).reshape(-1).tolist() == expected
        if backend == "llvm":
            spelling = {"x * n": "mul i64", "x - n": "sub i64",
                        "x % n": "srem i64"}[expression]
            assert spelling in artifact.llvm_ir
        abi = artifact.runtime_abi()
        assert abi["schema"] == "turing-runtime-abi-v1"
        buffers = {item["value_id"]: item for item in abi["buffers"]}
        output = buffers[result.id]
        assert (output["dtype"], output["rank"]) == ("int64", 1)
        if shape is None:
            assert output["shape"] is None
            assert output["extents_from"] == buffers[parameters["x"]]["index"]


def test_c620_integer_mask_arithmetic_uses_the_span_dtype(tmp_path):
    source = (
        REPO / "tests" / "integer_span_corpus" /
        "c620_integer_mask_arithmetic.py"
    ).read_text(encoding="utf-8")
    module, entry, parameters, result = _lower_integer_span(
        tmp_path, source, "c620_integer_mask_arithmetic",
    )
    feeds = {parameters["x"]: np.array([2, -3, 4, 5], dtype=np.int64),
             parameters["n"]: np.array([5], dtype=np.int64)}
    for backend in ("c", "llvm"):
        _artifact, execution = _run(backend, module, entry, feeds, tmp_path)
        assert np.asarray(execution.buffers[result.id]).reshape(-1).tolist() == [1, 0, 1, 2]


@pytest.mark.parametrize(
    "program,expected",
    (("c411_rank_two_first_row.py", 7.0),
     ("c412_rank_two_last_row.py", 4.0)),
)
def test_fortran_rank_two_span_reads_the_authored_cell(
    tmp_path, program, expected,
):
    source = (
        REPO / "tests" / "fortran_span_corpus" / program
    ).read_text(encoding="utf-8")
    module, entry, parameters, result = _lower(
        tmp_path, source, program.removesuffix(".py"), 2, shape=(2, 3),
    )
    values = np.asarray([[1.0, 7.0, 3.0], [4.0, 2.0, 9.0]])
    feeds = {parameters["x"]: values}
    for backend in BACKENDS:
        _artifact, execution = _run(backend, module, entry, feeds, tmp_path)
        got = np.asarray(execution.buffers[result.id]).reshape(-1).tolist()
        assert got == [expected], (backend, program, got)

#: A representative AbstractTensor operation: several elementwise steps,
#: a broadcast scalar constant, and an intermediate that is not an output.
REPRESENTATIVE = "def f(x, y):\n    return (x - y) * (x + y) * 0.5\n"


# -- item 1: the sanctioned entry, rank known, extents at run time ---------

def test_rank_only_product_is_a_span_not_a_scalar(tmp_path):
    """The planner states the result's rank; the region posts it."""

    module, entry, _parameters, result = _lower(
        tmp_path, REPRESENTATIVE, "rank_only_product", 1,
    )
    accounting = dict(result.accounting or {})
    assert accounting.get("program_abi_storage") == "span"
    assert int(accounting.get("program_abi_rank") or 0) == 1
    region_results = [
        instruction.res
        for name, function in module.functions.items() if name != entry
        for block in function.blocks.values()
        for instruction in block.instrs
        if str(instruction.op).casefold() == "mul"
    ]
    assert region_results
    assert all(
        int((value.accounting or {}).get("program_abi_rank") or 0) == 1
        for value in region_results
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rank,extents", [(1, (7,)), (2, (2, 3))])
def test_representative_op_runs_at_extents_it_never_saw(
    tmp_path, backend, rank, extents,
):
    module, entry, parameters, result = _lower(
        tmp_path, REPRESENTATIVE, f"rep_{backend}_{rank}", rank,
    )
    rng = np.random.default_rng(11)
    x = rng.uniform(-4, 4, size=extents)
    y = rng.uniform(-4, 4, size=extents)
    _artifact, execution = _run(
        backend, module, entry,
        {parameters["x"]: x, parameters["y"]: y}, tmp_path,
    )
    got = np.asarray(execution.buffers[result.id])
    assert got.size == x.size
    np.testing.assert_allclose(
        got.reshape(-1), ((x - y) * (x + y) * 0.5).reshape(-1),
    )


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b != "fortran"])
def test_disagreeing_runtime_extents_raise(tmp_path, backend):
    """(2, 3) against (3, 2) has equal element counts and is still wrong."""

    module, entry, parameters, _result = _lower(
        tmp_path, "def f(x, y):\n    return x * y\n", f"bad_{backend}", 2,
    )
    feeds = {
        parameters["x"]: np.ones((2, 3)), parameters["y"]: np.ones((3, 2)),
    }
    with pytest.raises(Exception, match="77"):
        _run(backend, module, entry, feeds, tmp_path)


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_runtime_spans_of_different_ranks_are_refused(tmp_path, backend):
    """A runtime broadcast (rank 1 against rank 2) is not stated: refused by
    name, never one scalar multiply."""

    raw = yaml.safe_load(BASE_CONTRACT.read_text(encoding="utf-8"))
    raw["program_abi"] = {"records": {}, "bindings": [], "values": [
        {"function": "f", "parameter": name, "storage": "span",
         "dtype": "float64", "rank": rank, "python_type": TENSOR_TYPE}
        for name, rank in (("x", 1), ("y", 2))
    ]}
    path = tmp_path / "mixed.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, exports = lower_ast_source_to_ssa(
            "def f(x, y):\n    return x * y\n", "f", name=f"mixed_{backend}",
            extraction_contract=ExtractionContract(path),
            progress=lambda _: None,
        )
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, exports[0])
    else:
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm

        artifact = emit_ssa_function_to_llvm(module, exports[0])
    assert not artifact.complete


# -- item 3: the exported calling contract ---------------------------------

@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_runtime_abi_states_rank_extents_and_result_derivation(
    tmp_path, backend,
):
    module, entry, parameters, result = _lower(
        tmp_path, REPRESENTATIVE, f"abi_{backend}", 2,
    )
    x = np.arange(6.0).reshape(2, 3)
    artifact, _execution = _run(
        backend, module, entry,
        {parameters["x"]: x, parameters["y"]: x + 1.0}, tmp_path,
    )
    abi = artifact.runtime_abi()
    assert abi["schema"] == "turing-runtime-abi-v1"
    assert abi["extents"]["type"] == (
        "int64_t" if backend == "c" else "int32_t"
    )
    buffers = {item["value_id"]: item for item in abi["buffers"]}
    for name in ("x", "y"):
        record = buffers[parameters[name]]
        assert record["role"] == "input"
        assert record["dtype"] == "float64"
        assert record["rank"] == 2
        assert record["shape"] is None
        assert record["extents_from"] == "fed"
    output = buffers[result.id]
    assert output["role"] == "output"
    assert output["rank"] == 2 and output["shape"] is None
    source = abi["buffers"][output["extents_from"]]
    assert source["value_id"] in {parameters["x"], parameters["y"]}
    # Every slot measures a public buffer, and every runtime-sized input is
    # measured, so a host can fill the vector from its own handles alone.
    measured = {slot["buffer"] for slot in abi["extents"]["order"]}
    assert {buffers[parameters["x"]]["index"],
            buffers[parameters["y"]]["index"]} <= measured


# -- item 4: the operator matrix --------------------------------------------

OPERATIONS = {
    "add": "x + y", "sub": "x - y", "mul": "x * y", "div": "x / y",
    "scale": "x * 2.5", "rsub": "1.5 - x", "neg": "-x",
    "fma": "x + y * 2.0", "poly": "(x - y) * (x + y)", "lt": "x < y",
    "abs": "abs(x)", "pow2": "x ** 2", "floordiv": "x // y", "mod": "x % y",
}
GEOMETRIES = {
    "rank0": (0, None, ()),
    "fixed4": (1, (4,), (4,)),
    "fixed23": (2, (2, 3), (2, 3)),
    "runtime1": (1, None, (5,)),
    "runtime2": (2, None, (2, 3)),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
@pytest.mark.parametrize("operation", sorted(OPERATIONS))
def test_operator_matrix(tmp_path, backend, geometry, operation):
    rank, shape, extents = GEOMETRIES[geometry]
    source = f"def f(x, y):\n    return {OPERATIONS[operation]}\n"
    module, entry, parameters, result = _lower(
        tmp_path, source, f"m_{operation}_{geometry}", rank, shape,
    )
    rng = np.random.default_rng(1)
    feeds_python = [
        rng.integers(1, 9, size=extents).astype(np.float64) if extents
        else float(rng.integers(1, 9))
        for _ in ("x", "y")
    ]
    namespace: dict = {}
    exec(source, namespace)
    expected = np.asarray(namespace["f"](*feeds_python))
    feeds = {
        parameters[name]: np.asarray(value)
        for name, value in zip(("x", "y"), feeds_python) if name in parameters
    }
    _artifact, execution = _run(backend, module, entry, feeds, tmp_path)
    got = np.asarray(execution.buffers[result.id])
    assert got.size == expected.size
    np.testing.assert_allclose(
        got.reshape(-1).astype(float), expected.reshape(-1).astype(float),
    )


# -- deployable SPIR-V from the same sanctioned entry ------------------------

def test_rank_only_region_deploys_as_elementwise_spirv(tmp_path):
    """The lowered region of the representative op is elementwise over
    rank-2 runtime-sized spans, so it deploys one element per invocation."""

    import shutil
    import subprocess

    from src.compiler.shader_deployment import spirv_deployment
    from src.compiler.ssa_spirv_backend import emit_deployable_compute_shader

    module, entry, _parameters, _result = _lower(
        tmp_path, REPRESENTATIVE, "rep_spirv", 2,
    )
    (region_name,) = [name for name in module.functions if name != entry]
    region = module.functions[region_name]
    outputs = [
        argument for block in region.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
        for argument in instruction.args
    ]
    shader = emit_deployable_compute_shader(region, outputs=outputs)
    assert shader.complete, shader.shortfalls
    record = spirv_deployment(shader).to_mapping()
    spans = [
        item["realizes"][0] for item in record["resources"]
        if item["realizes"][0]["rank"]
    ]
    assert len(spans) == 3 and all(view["shape"] is None for view in spans)
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        source = tmp_path / "region.spvasm"
        source.write_text(shader.source)
        binary = tmp_path / "region.spv"
        assembled = subprocess.run(
            ["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
            capture_output=True, text=True,
        )
        assert assembled.returncode == 0, assembled.stderr
        validated = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
            capture_output=True, text=True,
        )
        assert validated.returncode == 0, validated.stderr


def test_reduction_regions_deploy_static_and_runtime_sized(tmp_path):
    """``x.mean() - y.prod()`` lowers to one region per reduction.  With a
    static shape each reduction region unrolls; over runtime-sized spans it
    deploys as one invocation looping over the host's element count."""

    import shutil
    import subprocess

    from src.compiler.ssa_spirv_backend import emit_deployable_compute_shader

    source = "def f(x, y):\n    return x.mean() - y.prod()\n"

    def regions(name, shape):
        module, entry, _parameters, _result = _lower(tmp_path, source, name, 1, shape)
        for region_name, region in module.functions.items():
            if region_name == entry:
                continue
            outputs = [
                argument for block in region.blocks.values()
                for instruction in block.instrs if instruction.op == "Ret"
                for argument in instruction.args
            ]
            ops = {
                instruction.op for block in region.blocks.values()
                for instruction in block.instrs
            }
            yield ops, emit_deployable_compute_shader(region, outputs=outputs)

    static = list(regions("red_static", (5,)))
    assert {"mean", "prod"} <= set().union(*(ops for ops, _ in static))
    for index, (_ops, shader) in enumerate(static):
        assert shader.complete, shader.shortfalls
        if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
            continue
        text = tmp_path / f"region{index}.spvasm"
        text.write_text(shader.source)
        binary = tmp_path / f"region{index}.spv"
        assembled = subprocess.run(
            ["spirv-as", "--target-env", "vulkan1.1", str(text), "-o", str(binary)],
            capture_output=True, text=True,
        )
        assert assembled.returncode == 0, assembled.stderr
        validated = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
            capture_output=True, text=True,
        )
        assert validated.returncode == 0, validated.stderr

    looped = [
        shader for ops, shader in regions("red_runtime", None)
        if ops & {"mean", "prod"}
    ]
    assert {shader.reduction for shader in looped} == {"mean", "prod"}
    for shader in looped:
        assert shader.complete, shader.shortfalls
        if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
            continue
        text = tmp_path / f"{shader.reduction}.spvasm"
        text.write_text(shader.source)
        binary = tmp_path / f"{shader.reduction}.spv"
        for command in (
            ["spirv-as", "--target-env", "vulkan1.1", str(text), "-o", str(binary)],
            ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
        ):
            ran = subprocess.run(command, capture_output=True, text=True)
            assert ran.returncode == 0, ran.stderr


# -- builtin max/min of one tensor ------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("source,expected", [
    ("def f(x, y):\n    return max(y) + min(x)\n", 9.0 + 1.0),
    ("def f(x, y):\n    return y.max() - x.min()\n", 9.0 - 1.0),
])
def test_builtin_max_of_one_static_tensor_is_its_largest_element(
    tmp_path, backend, source, expected,
):
    """``max(y)`` iterates a rank-1 tensor and returns one element, as
    ``y.max()`` does; it is not an elementwise selector of one operand."""

    module, entry, parameters, result = _lower(
        tmp_path, source, f"one_max_{backend}", 1, shape=(5,),
    )
    x = np.array([1.0, 5.0, 2.0, 4.0, 3.0])
    y = np.array([3.0, 9.0, -1.0, 2.0, 0.0])
    _artifact, execution = _run(
        backend, module, entry,
        {parameters["x"]: x, parameters["y"]: y}, tmp_path,
    )
    got = np.asarray(execution.buffers[int(result.id)]).reshape(-1)
    np.testing.assert_allclose(got, [expected])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("source", [
    "def f(x, y):\n    return max(y)\n",
    "def f(x, y):\n    return y.max(axis=0)\n",
])
def test_rank_two_max_reductions_are_refused_not_guessed(tmp_path, backend, source):
    """``max(y)`` of a (2, 3) tensor raises in Python, and no backend spells
    ``y.max(axis=0)`` yet: each is refused by name, never one row or a
    broadcast whole-array maximum."""

    module, entry, parameters, _result = _lower(
        tmp_path, source, f"two_max_{backend}", 2, shape=(2, 3),
    )
    with pytest.raises(AssertionError):
        _run(
            backend, module, entry,
            {parameters["y"]: np.arange(6.0).reshape(2, 3)}, tmp_path,
        )
