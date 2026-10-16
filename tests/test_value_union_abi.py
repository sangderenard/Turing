import pytest

from src.transmogrifier.ssa import (
    SSAValueUnionAlternative,
    SSAValueUnionDescriptor,
    SSAValueUnionStorage,
    SSAValueUnionTable,
)


def test_value_union_table_keeps_per_arm_physical_storage():
    descriptor = SSAValueUnionDescriptor(
        union_id=40,
        identity="builtins.int | builtins.str",
        tag_value_id=41,
        alternatives=(
            SSAValueUnionAlternative(
                "builtins.int", SSAValueUnionStorage.SCALAR,
                dtype="int64", value_ids=(42,),
            ),
            SSAValueUnionAlternative(
                "builtins.str", SSAValueUnionStorage.SPAN,
                dtype="int64", shape=(-1,), value_ids=(43,),
            ),
        ),
    )
    table = SSAValueUnionTable(owner="row_admits")
    assert table.register(descriptor) is descriptor
    assert table.by_id(40) == descriptor
    assert table.member_claims(41) == ((40, "tag", ("tag",)),)
    assert table.member_claims(42) == (
        (40, "builtins.int", ("value", 0)),
    )


def test_value_union_refuses_unrepresented_payload_arm():
    with pytest.raises(ValueError, match="requires payload value ids"):
        SSAValueUnionAlternative(
            "builtins.str", SSAValueUnionStorage.SPAN, dtype="int64",
        )


def test_value_union_none_arm_has_no_payload():
    arm = SSAValueUnionAlternative(
        "builtins.NoneType", SSAValueUnionStorage.NONE,
    )
    assert arm.value_ids == ()
    with pytest.raises(ValueError, match="cannot own payload storage"):
        SSAValueUnionAlternative(
            "builtins.NoneType", SSAValueUnionStorage.NONE,
            dtype="int64", value_ids=(9,),
        )
