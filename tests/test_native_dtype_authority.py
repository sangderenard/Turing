"""Every scalar layout view retains the authority's width and identity."""
import pytest

from src.transmogrifier import dtype_layout as authority


SCALARS = tuple(layout for layout in authority.DTYPES.values()
                if layout.kind != "ptr")
ALIASES = tuple((alias, layout) for layout in SCALARS for alias in layout.aliases)


@pytest.mark.parametrize("alias,layout", ALIASES)
def test_scalar_aliases_keep_exact_storage_at_every_layout_handoff(alias, layout):
    assert authority.llvm_type_for_dtype(alias) == layout.llvm_type
    assert authority.dtype_byte_size(alias) == layout.byte_size
    assert authority.c_abi_type(alias) == (layout.c_type, layout.ctypes_name)
    assert authority.c_abi_type_table()[alias] == (layout.c_type, layout.ctypes_name)
    assert authority.c_lane_numpy_dtype(alias) == layout.numpy_dtype
    assert authority.shell_numpy_dtype(alias) == layout.numpy_dtype
    assert authority.shell_numpy_dtype_table()[alias] == layout.numpy_dtype


def test_reverse_c_storage_does_not_identify_unsigned_bytes_as_booleans():
    assert authority.c_lane_dtype_for_storage("uint8_t") == "uint8"
    for layout in SCALARS:
        assert authority.c_lane_dtype_for_storage(layout.c_type) == layout.name


def test_width_only_llvm_layouts_include_narrow_integers_without_inventing_sign():
    sizes = authority.llvm_type_bytes_table()
    assert sizes["i8"] == 1
    assert sizes["i16"] == 2
    assert sizes["i32"] == 4
    assert sizes["i64"] == 8
    assert authority.by_llvm_type("i8").name == "int8"
    assert authority.by_llvm_type("i16").name == "int16"
    assert authority.c_lane_storage_for_llvm_type("i8") == "int8_t"
    assert authority.c_lane_storage_for_llvm_type("i16") == "int16_t"
    assert authority.llvm_abi_numpy_table()["i8"] == "int8"
    assert authority.llvm_abi_numpy_table()["i16"] == "int16"


@pytest.mark.parametrize("lookup", [authority.c_abi_type, authority.llvm_type_for_dtype,
    authority.c_lane_numpy_dtype, authority.dtype_byte_size,
    authority.c_lane_dtype_for_storage, authority.c_lane_storage_for_llvm_type])
def test_explicit_undeclared_type_cannot_take_floating_fallback(lookup):
    with pytest.raises(KeyError, match="declared|storage|LLVM"):
        lookup("not_a_declared_dtype")


def test_supported_lane_tables_are_views_of_declared_backend_support():
    for alias, layout in ALIASES:
        if layout.fortran_kind is None:
            assert alias not in authority.fortran_kind_table()
        else:
            assert authority.fortran_kind_table()[alias] == layout.fortran_kind
        if layout.wasm_type is None:
            assert alias not in authority.wasm_type_table()
        else:
            assert authority.wasm_type_table()[alias][0] == layout.wasm_type


def test_absent_type_keeps_the_existing_default():
    assert authority.c_abi_type(None) == ("double", "c_double")
    assert authority.llvm_type_for_dtype(None) == "double"
    assert authority.c_lane_numpy_dtype(None) == "float64"
    assert authority.dtype_byte_size(None) == 8
