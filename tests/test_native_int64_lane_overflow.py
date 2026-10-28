"""int64 AbstractTensor lanes follow Python ``int``, never a silent wrap.

``-tokens`` / ``abs(tokens)`` on an int64 tensor went through the double
unary kernel (2**53 + 1 came back 2**53; after an int64 producer the lane's
bits were negated as a double's sign).  ``tokens * 1103515245`` wrapped every
lane that left int64.  Under ``INT64_LANE_POLICY == "exact"`` each lane is
computed on the integer path and a lane Python would carry past int64
reports ``NativeIntOverflow`` (code 86) on C and LLVM.  The "wrap" policy is
NumPy's lane and the same programs then match NumPy.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

CONTRACT = "extraction_contracts/program_extraction.yaml"

PROGRAMS = {
    "neg": "def root(tokens):\n    return -tokens\n",
    "abs": "def root(tokens):\n    return abs(tokens)\n",
    "abs_method": "def root(tokens):\n    return tokens.abs()\n",
    "neg_after_op": "def root(tokens):\n    shifted = tokens - 1\n    return -shifted\n",
    "abs_after_op": "def root(tokens):\n    shifted = tokens * 2\n    return abs(shifted)\n",
    "lcg": "def root(tokens):\n    return tokens * 1103515245 + 12345\n",
    "add_sub": "def root(tokens):\n    return (tokens + tokens) - 1\n",
}

PYTHON = {
    "neg": lambda x: -x,
    "abs": abs,
    "abs_method": abs,
    "neg_after_op": lambda x: -(x - 1),
    "abs_after_op": lambda x: abs(x * 2),
    "lcg": lambda x: x * 1103515245 + 12345,
    "add_sub": lambda x: (x + x) - 1,
}

IN_RANGE = [-7, 3, 2**53 + 1]
#: The LCG's product leaves int64 for any lane past about 2**33.
IN_RANGE_FOR = {"lcg": [-7, 3, 2**32]}
#: One lane per program whose Python value leaves int64.
OVERFLOWING = {
    "neg": [-7, 3, -2**63],
    "abs": [-7, 3, -2**63],
    "abs_method": [-7, 3, -2**63],
    "neg_after_op": [-7, 3, -2**63 + 1],
    "abs_after_op": [-7, 3, 2**62],
    "lcg": [-7, 3, 2**40],
    "add_sub": [-7, 3, 2**62 + 1],
}


def _lower(name):
    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contract = ExtractionContract(CONTRACT).with_program_abi({
        "records": {}, "bindings": [],
        "values": [{
            "function": "root", "parameter": "tokens", "storage": "span",
            "dtype": "int64", "rank": 1, "shape": [3],
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
    })
    module, _outputs, exports = lower_ast_source_to_ssa(
        PROGRAMS[name], "root", name=f"int64_lane_{name}",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None, progress_bars=False,
    )
    entry = exports[0]
    function = module.functions[entry]
    parameter = dict(function.metadata["parameter_names"])["tokens"]
    returned = next(
        instruction.args[0]
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
    )
    return module, entry, parameter, int(returned.id)


def _run(backend, name, lanes):
    from tests.test_runtime_sized_spans import _run as run_backend

    module, entry, parameter, returned = _lower(name)
    _artifact, execution = run_backend(
        backend, module, entry,
        {parameter: np.asarray(lanes, dtype=np.int64)},
        pathlib.Path(tempfile.mkdtemp()),
    )
    result = np.asarray(execution.buffers[returned])
    assert result.dtype == np.int64, result.dtype
    return result.reshape(-1).tolist()


@pytest.mark.parametrize("backend", ["c", "llvm"])
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_int64_lanes_match_python(backend, name):
    lanes = IN_RANGE_FOR.get(name, IN_RANGE)
    expected = [PYTHON[name](x) for x in lanes]
    assert _run(backend, name, lanes) == expected


@pytest.mark.parametrize("backend", ["c", "llvm"])
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_int64_lane_past_int64_is_reported_not_wrapped(backend, name):
    from src.compiler.native_faults import NativeIntOverflow

    lanes = OVERFLOWING[name]
    values = [PYTHON[name](x) for x in lanes]
    assert any(not -2**63 <= value < 2**63 for value in values)
    with pytest.raises(NativeIntOverflow):
        _run(backend, name, lanes)


@pytest.mark.parametrize("backend", ["c", "llvm"])
@pytest.mark.parametrize("name", ["neg", "abs", "lcg"])
def test_wrap_policy_is_numpy_lane(backend, name, monkeypatch):
    from src.compiler import ssa_llvm_backend

    monkeypatch.setattr(ssa_llvm_backend, "INT64_LANE_POLICY", "wrap")
    lanes = OVERFLOWING[name]
    expected = {
        "neg": lambda x: -x,
        "abs": np.abs,
        "lcg": lambda x: x * np.int64(1103515245) + np.int64(12345),
    }[name]
    with np.errstate(over="ignore"):
        wrapped = expected(np.asarray(lanes, dtype=np.int64)).tolist()
    assert _run(backend, name, lanes) == wrapped
