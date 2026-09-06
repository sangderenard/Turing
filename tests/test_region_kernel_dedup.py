"""Byte-identical region kernels share one emitted ``.wasm`` file.

A region program's WebAssembly binary is independent of its value-ids and its
module name -- only the topology, the constants, and the dtype enter the bytes.
So regions that compute the same algorithm over different data collapse to one
kernel file, while each region keeps its own method card (its own field-slot
wiring). This is the dedup that turns a program repeating one operation over
thousands of regions into a handful of distinct kernels.
"""
from __future__ import annotations

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.control_source import ControlProgram, StatementBlock
from src.compiler.wasm_class_modules import emit_control_region_modules
from src.compiler.wasm_class_coordinator import build_class_inventory


def _region(data_id, value_id):
    return FusedProgram(
        version=1, feeds={data_id, value_id},
        steps=[OpStep(0, "index_set", [data_id, value_id], {"slices": 2},
                      data_id + 900)],
        outputs={f"data{data_id}": data_id + 900},
        extras={"capture_feed_origins": {
            data_id: {"binding_name": f"d{data_id}"},
            value_id: {"binding_name": f"v{value_id}"}}},
    )


def test_byte_identical_regions_share_one_kernel_file():
    control = ControlProgram(
        StatementBlock(("__scheduled_region_0__", "__scheduled_region_1__")),
        region_indices=(0, 1),
    )
    # Same topology + same constant, different value-ids: identical bytes.
    _modules, manifest = emit_control_region_modules(
        control, {0: _region(1, 2), 1: _region(11, 22)},
        owner_name="buf", module_dir=".", dtype="int64",
    )
    assert manifest["region_count"] == 2
    assert manifest["unique_kernels"] == 1
    kernels = [entry["kernel"] for entry in manifest["modules"]]
    assert kernels[0] == kernels[1]
    assert manifest["modules"][0]["url"] == manifest["modules"][1]["url"]
    # Still one method card per region (per-region field-slot wiring).
    inventory = build_class_inventory(manifest)
    assert len(inventory.methods) == 2


def test_different_constants_do_not_dedup():
    control = ControlProgram(
        StatementBlock(("__scheduled_region_0__", "__scheduled_region_1__")),
        region_indices=(0, 1),
    )
    a = _region(1, 2)
    b = FusedProgram(
        version=1, feeds={11, 22},
        steps=[OpStep(0, "index_set", [11, 22], {"slices": 5}, 911)],
        outputs={"data11": 911},
        extras={"capture_feed_origins": {
            11: {"binding_name": "d11"}, 22: {"binding_name": "v22"}}},
    )
    _modules, manifest = emit_control_region_modules(
        control, {0: a, 1: b}, owner_name="buf", module_dir=".", dtype="int64",
    )
    # Different subscript constant -> different bytes -> two kernels.
    assert manifest["unique_kernels"] == 2
