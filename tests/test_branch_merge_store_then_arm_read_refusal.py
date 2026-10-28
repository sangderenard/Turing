"""A store through a branch-merged list, read later through the shared arm.

``a = [7, 8, 9]; x = a if n > 2 else <display>; grow(x, n); x[0] = ...;
a[0]`` has no exact order yet; it is refused by name (equal and unequal
length), never lowered to a stale read.
"""
import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

TEMPLATE = '''
def grow(xs: list[int], k: int) -> int:
    xs.append(k)
    return len(xs)
def root(n: int) -> int:
    a = [7, 8, 9]
    if n > 2:
        x = a
    else:
        x = {arm}
    m = grow(x, n)
    x[0] = 100 + n
    return m * 100000 + a[0] * 10 + a[1]
'''


@pytest.mark.parametrize("arm", ["[4, 5, 6]", "[4, 5]"])
def test_store_through_merge_then_shared_arm_read_is_refused(arm):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    with pytest.raises(
        ValueError, match="store_through_handle_merge_then_arm_read",
    ):
        lower_ast_source_to_ssa(
            TEMPLATE.format(arm=arm), "root", name="refused_store_then_read",
            extraction_contract=str(
                ROOT / "extraction_contracts" / "program_extraction.yaml"
            ),
            progress=lambda _: None,
        )
