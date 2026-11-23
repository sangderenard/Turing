"""AbstractTensor atomic updates: the declared read, combine and store.

``atomic_add`` / ``atomic_or`` return the old element and write the
caller-owned tensor in place.  Compiled, the pursued body is one
``source_atomic_site`` region, so a concurrent target can emit it as one
read-modify-write and a serial target runs it exactly as written.
"""

from pathlib import Path
import tempfile
import textwrap

import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstraction_methods.indexing import (
    ATOMIC_UPDATE_OPERATIONS, atomic_add, atomic_or,
)

ROOT = Path(__file__).resolve().parents[1]


def test_eager_atomic_returns_old_value_and_writes_in_place():
    words = AbstractTensor.get_tensor(np.array([5, 9, 0, 0], dtype=np.uint32))
    observed = [
        words.atomic_add(0, 1), words.atomic_add(0, 1),
        words.atomic_or(1, 6), words.atomic_or(1, 1),
    ]
    assert [int(value) for value in observed] == [5, 6, 9, 15]
    assert words.tolist() == [7, 15, 0, 0]


def test_atomic_declarations_name_the_authored_functions():
    assert ATOMIC_UPDATE_OPERATIONS == {
        (atomic_add.__module__, atomic_add.__qualname__): "add",
        (atomic_or.__module__, atomic_or.__qualname__): "or",
    }


def _lower(source: str):
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contract = ExtractionContract(
        ROOT / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({"records": {}, "bindings": [], "values": [
        {"function": "claim", "parameter": "words", "storage": "span",
         "dtype": "uint32", "rank": 1, "shape": [4],
         "python_type": "src.common.tensors.abstraction.AbstractTensor"},
        {"function": "claim", "parameter": "i", "storage": "scalar",
         "dtype": "int64", "rank": 0, "shape": [],
         "python_type": "builtins.int"},
    ]}).with_execution_file(
        ROOT / "extraction_contracts/vehicle_full_native_execution.yaml"
    )
    return lower_ast_source_to_ssa(
        textwrap.dedent(source), "claim", extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        runtime_closure_only=True, progress=lambda _: None,
        progress_bars=False,
    )


def test_compiled_atomic_site_is_one_region_and_runs_on_caller_storage():
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm,
        prepare_artifact_execution,
    )

    module, _outputs, exports = _lower('''
        from src.common.tensors.abstraction import AbstractTensor
        def claim(words: AbstractTensor, i: int) -> int:
            a = AbstractTensor.atomic_add(words, i, 1)
            return AbstractTensor.atomic_or(words, i, 6)
    ''')
    for operation in ("atomic_add", "atomic_or"):
        regions = [
            function for name, function in module.functions.items()
            if f"__{operation}__specialized_" in name
            and "__planned_region_" in name
        ]
        # One reserved region holds the read, the combine and the store.
        assert len(regions) == 1, (operation, sorted(module.functions))
        ops = [
            instruction.op for block in regions[0].blocks.values()
            for instruction in block.instrs
        ]
        assert ops.count("Load") == 1 and ops.count("Store") == 1, ops

    entry = exports[0]
    function = module.functions[entry]
    names = dict(function.metadata["parameter_names"])
    emitted = emit_ssa_function_to_llvm(module, entry)
    assert emitted.shortfalls == ()
    artifact = compile_artifact(emitted, directory=tempfile.mkdtemp())
    output = next(
        instruction.args[0].id for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
    )
    words = np.array([5, 9, 0, 0], dtype=np.uint32)
    observed = []
    for i in (0, 0, 1):
        feeds = {
            names["words"]: words,
            names["i"]: np.array([i], dtype=np.int64),
        }
        result = prepare_artifact_execution(artifact, feeds).run()
        observed.append(int(result.buffers[output].reshape(-1)[0]))
    # add then or, each returning the element it found: 5+1=6, 6|6=6;
    # 6+1=7, 7|6=7; 9+1=10, 10|6=14.  The caller's array is written.
    assert observed == [6, 7, 10]
    assert words.tolist() == [7, 14, 0, 0]


def test_literal_index_atomic_and_store_keep_the_uint32_element():
    """``atomicAdd(words[0], ..)``: a literal index on a uint32 span.

    The literal index used to take ``index_assign_double``, whose in-place
    output formal revised the span to physical float64 beside its uint32
    dtype; ir_indexing then refused the disagreement.  It is one element's
    address in its own dtype, as a runtime index is.
    """

    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm,
        prepare_artifact_execution,
    )

    module, _outputs, exports = _lower('''
        from src.common.tensors.abstraction import AbstractTensor
        def claim(words: AbstractTensor, i: int) -> int:
            words[2] = i
            return AbstractTensor.atomic_add(words, 0, 1)
    ''')
    entry = exports[0]
    function = module.functions[entry]
    names = dict(function.metadata["parameter_names"])
    emitted = emit_ssa_function_to_llvm(module, entry)
    assert emitted.shortfalls == ()
    artifact = compile_artifact(emitted, directory=tempfile.mkdtemp())
    output = next(
        instruction.args[0].id for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
    )
    words = np.array([5, 9, 0, 0], dtype=np.uint32)
    observed = []
    for i in (3, 4):
        result = prepare_artifact_execution(artifact, {
            names["words"]: words,
            names["i"]: np.array([i], dtype=np.int64),
        }).run()
        observed.append(int(result.buffers[output].reshape(-1)[0]))
    assert observed == [5, 6]
    assert words.tolist() == [7, 9, 4, 0]


def test_declared_uint32_argument_wraps_in_the_callee():
    """``one: 'uint32' = 1`` crosses the call as a uint32, not a Python int.

    The specialization used to fold the argument to the bare literal 1, so
    the callee's ``old + value`` was widened to int64 and the uint32 store
    of 0xffffffff + 1 faulted (NativeIntOverflow) instead of wrapping to 0
    as GLSL's ``atomicAdd`` does.  The fact now carries the declared dtype.
    """

    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm,
        prepare_artifact_execution,
    )

    module, _outputs, exports = _lower('''
        from src.common.tensors.abstraction import AbstractTensor
        def claim(words: AbstractTensor, i: int) -> int:
            one: 'uint32' = 1
            six: 'uint32' = 6
            a = AbstractTensor.atomic_add(words, i, one)
            return AbstractTensor.atomic_or(words, 1, six)
    ''')
    for name, function in module.functions.items():
        if "__atomic_" in name and "__planned_region_" in name:
            ops = [
                instruction for block in function.blocks.values()
                for instruction in block.instrs
            ]
            assert not any(instruction.op == "Cast" for instruction in ops)
            combine = next(
                instruction for instruction in ops
                if instruction.op in {"Add", "BitOr"}
            )
            assert combine.res.dtype == "uint32"
    entry = exports[0]
    function = module.functions[entry]
    names = dict(function.metadata["parameter_names"])
    emitted = emit_ssa_function_to_llvm(module, entry)
    assert emitted.shortfalls == ()
    artifact = compile_artifact(emitted, directory=tempfile.mkdtemp())
    output = next(
        instruction.args[0].id for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
    )
    words = np.array([0xFFFFFFFF, 9, 0, 0], dtype=np.uint32)
    observed = []
    for _ in range(2):
        result = prepare_artifact_execution(artifact, {
            names["words"]: words,
            names["i"]: np.array([0], dtype=np.int64),
        }).run()
        observed.append(int(result.buffers[output].reshape(-1)[0]))
    assert observed == [9, 15]
    # 0xffffffff + 1 wraps to 0, then 1; 9 | 6 == 15.
    assert words.tolist() == [1, 15, 0, 0]
