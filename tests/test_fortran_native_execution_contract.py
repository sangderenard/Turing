"""The existing packed Fortran caller must obey emitted storage records."""
import numpy as np
import pytest
from src.compiler.compiled_program_api import CompiledProgramAPI, EntryPoint, Parameter
from src.compiler.ssa_fortran_backend import FortranCoreNative


def _unbound_core(parameters, metadata=None):
    # Allocation tests must not load/execute a library. Use the actual caller's
    # preparation method, inspecting the exact slots it would pass to native.
    core = FortranCoreNative.__new__(FortranCoreNative)
    core._entry_record = EntryPoint("root", "root", "control", tuple(parameters))
    core._api = CompiledProgramAPI("probe", "fortran", "root", (core._entry_record,), metadata or {})
    core._packed = None
    return core


@pytest.mark.parametrize("dtype,c_type,ctype,value", (
    ("bool", "bool", "c_bool", True),
    ("int64", "int64_t", "c_int64", 2**40 + 7),
    ("float32", "float", "c_float", 1.25),
    ("uint8", "uint8_t", "c_uint8", 251),
))
def test_fortran_slots_use_declared_native_type(dtype, c_type, ctype, value):
    core = _unbound_core([Parameter("t3", "input", dtype, c_type, ctype, "value")])
    execution = core.prepare_execution({3: value})
    assert execution.buffers[3].dtype == np.dtype(dtype)
    assert execution.buffers[3].item() == value
    assert execution._slots[0] is execution.buffers[3]


def test_fortran_fixed_shape_and_extent_cells_are_retained():
    core = _unbound_core([
        Parameter("extent_2", "extent", "int32", "int32_t", "c_int32", "value"),
        Parameter("extent_3", "extent", "int32", "int32_t", "c_int32", "value"),
        Parameter("t5", "workspace", "int64", "int64_t", "c_int64", "reference",
                  shape=(2, 3), extents=("extent_2", "extent_3")),
    ])
    execution = core.prepare_execution({})
    assert execution.buffers[5].size == 6
    assert [slot.item() for slot in execution._slots[:2]] == [2, 3]
    assert all(int(slot.ctypes.data) == execution._pointers[index]
               for index, slot in enumerate(execution._slots))


def test_fortran_missing_and_wrong_sized_feeds_refuse_before_native():
    core = _unbound_core([Parameter("t3", "input", "float64", "double", "c_double",
                                  "reference", shape=(2,))])
    with pytest.raises(ValueError, match="missing Fortran feed"):
        core.prepare_execution({})
    with pytest.raises(ValueError, match="ABI needs 2"):
        core.prepare_execution({3: [1.0]})


def test_fortran_capacity_is_the_minimum_actual_column_row_count():
    core = _unbound_core([
        Parameter("t1", "workspace", "float64", "double", "c_double", "reference", shape=(4,)),
        Parameter("t2", "workspace", "float64", "double", "c_double", "reference", shape=(6,)),
        Parameter("t3", "workspace", "int64", "int64_t", "c_int64", "value"),
    ], {"sequence_tables": {"root": [{"sequence_id": 7, "column_value_ids": [1, 2],
                                      "column_shapes": [[], [2]]}]},
        "sequence_runtime_bindings": {"root": [{"sequence_id": 7,
            "column_parameters": ["t1", "t2"], "capacity_parameter": "t3"}]}})
    execution = core.prepare_execution({})
    assert execution.buffers[3].item() == 3
    with pytest.raises(ValueError, match="capacity exceeds"):
        core.prepare_execution({3: 4})
