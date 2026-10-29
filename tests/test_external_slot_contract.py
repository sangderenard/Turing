"""A runtime slot binds a Python callable only where the execution contract allows it."""

import pytest

from src.compiler.external_functions import _slot_address, external_slot_rows


def _row(python_callbacks):
    return {
        "slot": 0, "external": "f", "buffer_shapes": ((4,), (4,)),
        "argument_slots": (0,), "output_slots": (1,),
        "python_callbacks": python_callbacks,
    }


def _host_f(t):
    return t


@pytest.mark.parametrize("python_callbacks", ["reject", None])
def test_python_callable_slot_is_refused_without_contract_permission(python_callbacks):
    with pytest.raises(ValueError, match="call back into the host interpreter"):
        _slot_address(_row(python_callbacks), _host_f, [], [])


def test_python_callable_slot_is_bound_when_the_contract_allows_callbacks():
    keep = []
    address = _slot_address(_row("contract_only"), _host_f, keep, [])
    assert address and len(keep) == 1


def test_native_address_slot_needs_no_callback_permission():
    assert _slot_address(_row("reject"), 4096, [], []) == 4096


def test_slot_rows_carry_the_lowered_module_execution_contract():
    class _Function:
        metadata = {"llvm_piece": {
            "external": "f", "external_identity": None, "buffer_order": (1, 2),
            "extent_order": (), "argument_slots": (0,), "output_slots": (1,),
            "buffer_shapes": ((4,), (4,)), "derivative": None,
        }}

    class _Module:
        functions = {"f": _Function()}
        metadata = {"extraction_contract": {"execution": {"python_callbacks": "reject"}}}

    (row,) = external_slot_rows(_Module(), ("f",))
    assert row["python_callbacks"] == "reject"
