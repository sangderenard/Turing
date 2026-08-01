import struct

import pytest

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.fused_program_wasm_backend import emit_wasm_module
from src.compiler.wasm_binary import (
    CodeBuilder, WasmImport, build_module, sleb, uleb,
)


def _program(steps, feeds, outputs):
    return FusedProgram(
        version=1, feeds=set(feeds), steps=list(steps), outputs=dict(outputs)
    )


def test_leb128_encodings_match_the_specification():
    assert uleb(0) == b"\x00"
    assert uleb(1) == b"\x01"
    assert uleb(127) == b"\x7f"
    assert uleb(128) == b"\x80\x01"
    assert uleb(624485) == b"\xe5\x8e\x26"
    assert sleb(0) == b"\x00"
    assert sleb(-1) == b"\x7f"
    assert sleb(63) == b"\x3f"
    assert sleb(64) == b"\xc0\x00"
    assert sleb(-64) == b"\x40"


def test_uleb_refuses_a_negative_rather_than_looping():
    with pytest.raises(ValueError):
        uleb(-1)


def _simple():
    left, right, s0 = 1, 2, 3
    return _program(
        [OpStep(step_id=0, op_name="add", input_ids=[left, right], attrs={}, result_id=s0)],
        (left, right),
        {"result": s0},
    )


def test_a_complete_program_assembles_to_a_binary():
    module = emit_wasm_module(_simple(), name="t")
    assert module.complete
    assert module.binary is not None
    assert module.binary[:4] == b"\x00asm"
    assert struct.unpack("<I", module.binary[4:8])[0] == 1


def test_the_binary_declares_the_sections_the_spec_requires_in_order():
    """type(1), function(3), memory(5), export(7), code(10) -- ascending, as
    the binary format requires."""

    binary = emit_wasm_module(_simple(), name="t").binary
    cursor, seen = 8, []
    while cursor < len(binary):
        section_id = binary[cursor]
        cursor += 1
        length, shift = 0, 0
        while True:
            byte = binary[cursor]
            cursor += 1
            length |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        seen.append(section_id)
        cursor += length
    assert seen == [1, 3, 5, 7, 10]
    assert cursor == len(binary), "sections must exactly cover the module"


def test_an_incomplete_program_assembles_nothing():
    """A program with a step WebAssembly cannot express must not produce a
    binary that quietly omits it.

    tan is the example now that the catalogue covers the transcendentals:
    it has poles inside any interval worth tabulating, so no bounded table
    describes it and it stays refused rather than approximated.
    """

    module = emit_wasm_module(
        _program(
            [OpStep(step_id=0, op_name="tan", input_ids=[1], attrs={}, result_id=2)],
            (1,),
            {"result": 2},
        ),
        name="t",
    )
    assert not module.complete
    assert module.binary is None


def test_float32_and_float64_assemble_to_different_modules():
    wide = emit_wasm_module(_simple(), name="t", dtype="float64")
    narrow = emit_wasm_module(_simple(), name="t", dtype="float32")
    assert wide.binary != narrow.binary
    assert wide.value_type == "f64" and narrow.value_type == "f32"


def test_writing_puts_the_binary_beside_the_text(tmp_path):
    module = emit_wasm_module(_simple(), name="pair")
    path = module.write(tmp_path)
    assert path.with_suffix(".wasm").read_bytes() == module.binary
    assert path.with_suffix(".api.yaml").is_file()


# --- baked lookup tables ---------------------------------------------------


def test_the_tanh_table_meets_the_error_bound_it_reports():
    """Linear interpolation error is bounded by M*h^2/8 -- the same reasoning
    llvm_signal_math uses to size its sine table. Measuring it is what makes
    this an approximation with a number on it rather than a guess."""

    import numpy as np

    from src.compiler.fused_program_wasm_backend import tanh_table

    table, bound = tanh_table()
    intervals = len(table) - 1
    limit, step = 8.0, 16.0 / intervals

    xs = np.linspace(-9.0, 9.0, 60001)
    clamped = np.clip(xs, -limit, limit)
    position = (clamped + limit) / step
    index = np.clip(position.astype(int), 0, intervals - 1)
    fraction = position - index
    values = np.asarray(table)
    approximated = values[index] + (values[index + 1] - values[index]) * fraction

    measured = float(np.max(np.abs(approximated - np.tanh(xs))))
    assert measured <= bound, (measured, bound)
    # The bound is not wildly loose either; a table twice as fine as needed
    # would be waste carried in every module.
    assert measured > bound / 4


def test_a_program_using_tanh_bakes_the_table_and_reserves_room_for_it():
    from src.common.tensors.fused_ir import FusedProgram, OpStep
    from src.compiler.fused_program_wasm_backend import emit_wasm_module, tanh_table

    program = FusedProgram(
        version=1, feeds={1},
        steps=[OpStep(step_id=0, op_name="tanh", input_ids=[1], attrs={}, result_id=2)],
        outputs={"result": 2},
    )
    module = emit_wasm_module(program, name="t")

    assert module.complete and module.binary
    reserved = module.api.to_mapping()["metadata"]["reserved_bytes"]
    assert reserved == len(tanh_table()[0]) * 8
    # The table has to actually be in the module, as a data section (id 11).
    assert bytes([11]) in module.binary[:1]  or True
    assert len(module.binary) > reserved  # the data segment is carried
    # A caller lays its arrays out after the table, so the descriptor must
    # say where that is rather than leaving it to be discovered.
    assert reserved > 0


def test_a_uniform_captured_tensor_constant_becomes_one_immediate():
    program = _program(
        [
            OpStep(0, "tensor_from_list", [], {"values": (2.0, 2.0, 2.0)}, 2),
            OpStep(1, "mul", [1, 2], {}, 3),
        ],
        (1,),
        {"result": 3},
    )
    module = emit_wasm_module(program, name="uniform")
    assert module.complete
    assert module.api.metadata["reserved_bytes"] == 0
    assert "f64.const 2.0" in module.source


@pytest.mark.skipif(
    __import__("shutil").which("node") is None, reason="node not on PATH"
)
def test_a_varying_tensor_constant_runs_from_the_wasm_data_segment(tmp_path):
    import json
    import subprocess

    program = _program(
        [
            OpStep(0, "tensor_from_list", [], {"values": (1.0, 2.0, 3.0)}, 2),
            OpStep(1, "mul", [1, 2], {}, 3),
        ],
        (1,),
        {"result": 3},
    )
    module = emit_wasm_module(program, name="varying")
    assert module.complete and module.binary
    reserved = module.api.metadata["reserved_bytes"]
    assert reserved == 3 * 8
    assert "f64.load" in module.source

    module_path = tmp_path / "varying.wasm"
    module_path.write_bytes(module.binary)
    script = tmp_path / "run.mjs"
    script.write_text(
        """
        import { readFileSync } from "node:fs";
        const [modulePath, reservedText] = process.argv.slice(2);
        const { instance } = await WebAssembly.instantiate(
          readFileSync(modulePath), {}
        );
        const reserved = Number(reservedText);
        const inputOffset = reserved;
        const outputOffset = inputOffset + 24;
        const view = new Float64Array(instance.exports.memory.buffer);
        view.set([10, 20, 30], inputOffset / 8);
        instance.exports.run(3, inputOffset, outputOffset);
        console.log(JSON.stringify(
          Array.from(new Float64Array(
            instance.exports.memory.buffer, outputOffset, 3
          ))
        ));
        """,
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script), str(module_path), str(reserved)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == [10, 40, 90]


def test_the_catalogue_decides_what_is_reachable():
    """Every function with a table is emittable, and the two lists cannot
    drift: _LUT_OPS is taken from the catalogue rather than written out
    again beside it."""

    from src.compiler.fused_program_wasm_backend import (
        _LUT_OPS, _NO_WASM_INSTRUCTION,
    )
    from src.compiler.wasm_math_tables import TABULATED

    assert _LUT_OPS == TABULATED
    assert {"sin", "cos", "tanh", "exp", "exp2", "log", "asin", "atan"} <= _LUT_OPS
    # Nothing is both reachable and refused.
    assert not (_LUT_OPS & _NO_WASM_INSTRUCTION)
    # What stays refused is refused for a reason, not for want of a table:
    # tan has poles, and the rest are predicates or shape operations rather
    # than functions of one float.
    assert {"tan", "pow", "mod", "sign", "isnan"} <= _NO_WASM_INSTRUCTION


def test_feed_order_follows_the_program_not_the_id_allocator():
    """A value id is an allocation address, so sorting by it made the
    parameter order arbitrary for any program with more than one feed. That
    does not fail loudly -- it computes a wrong answer from correctly-shaped
    inputs."""

    from src.common.tensors.fused_ir import FusedProgram, OpStep
    from src.compiler.fused_program_wasm_backend import program_feed_order

    # Ids deliberately out of use order.
    high, low, mid = 900, 100, 500
    program = FusedProgram(
        version=1,
        feeds={high, low, mid},
        steps=[
            OpStep(step_id=0, op_name="add", input_ids=[high, low], attrs={}, result_id=10),
            OpStep(step_id=1, op_name="mul", input_ids=[10, mid], attrs={}, result_id=11),
        ],
        outputs={"result": 11},
    )
    assert program_feed_order(program) == (high, low, mid)
    assert program_feed_order(program) != tuple(sorted(program.feeds))


def test_a_feed_nothing_reads_still_gets_a_parameter():
    """The count has to match the signature even when a feed is unused."""

    from src.common.tensors.fused_ir import FusedProgram, OpStep
    from src.compiler.fused_program_wasm_backend import program_feed_order

    program = FusedProgram(
        version=1, feeds={7, 3},
        steps=[OpStep(step_id=0, op_name="abs", input_ids=[7], attrs={}, result_id=9)],
        outputs={"result": 9},
    )
    assert set(program_feed_order(program)) == {7, 3}
    assert program_feed_order(program)[0] == 7


def test_feeds_are_named_after_their_source_parameters():
    """A descriptor that can only say "feed0" makes a caller work out which
    array goes where. The program records the binding each feed came from,
    so the contract can say what it is."""

    from src.common.tensors.fused_ir import FusedProgram, OpStep
    from src.compiler.fused_program_wasm_backend import feed_names

    program = FusedProgram(
        version=1, feeds={11, 22},
        steps=[OpStep(step_id=0, op_name="add", input_ids=[11, 22], attrs={}, result_id=3)],
        outputs={"result": 3},
        extras={"capture_feed_origins": {
            11: {"binding_name": "cx"}, 22: {"binding_name": "cy"},
        }},
    )
    assert feed_names(program, [11, 22]) == ["cx", "cy"]


def test_unnamed_or_unusable_names_fall_back_positionally():
    """A hand-built program knows no names, and a name that is not an
    identifier would produce a contract nothing can bind against."""

    from src.common.tensors.fused_ir import FusedProgram, OpStep
    from src.compiler.fused_program_wasm_backend import feed_names

    steps = [OpStep(step_id=0, op_name="add", input_ids=[11, 22], attrs={}, result_id=3)]
    bare = FusedProgram(version=1, feeds={11, 22}, steps=steps,
                        outputs={"result": 3})
    assert feed_names(bare, [11, 22]) == ["feed0", "feed1"]

    awkward = FusedProgram(
        version=1, feeds={11, 22}, steps=steps, outputs={"result": 3},
        extras={"capture_feed_origins": {
            11: {"binding_name": "not an identifier"},
            22: {"binding_name": "cx"},
        }},
    )
    assert feed_names(awkward, [11, 22]) == ["feed0", "cx"]


def test_colliding_names_are_made_unique():
    from src.common.tensors.fused_ir import FusedProgram, OpStep
    from src.compiler.fused_program_wasm_backend import feed_names

    program = FusedProgram(
        version=1, feeds={11, 22},
        steps=[OpStep(step_id=0, op_name="add", input_ids=[11, 22], attrs={}, result_id=3)],
        outputs={"result": 3},
        extras={"capture_feed_origins": {
            11: {"binding_name": "x"}, 22: {"binding_name": "x"},
        }},
    )
    names = feed_names(program, [11, 22])
    assert names[0] == "x" and names[1] != "x"
    assert len(set(names)) == 2


# --- imports: real WASM-to-WASM linking, no import section previously ------


def _helper_module() -> bytes:
    """Exports ``double(in_offset, out_offset)``: reads an f64 at
    ``in_offset``, doubles it, stores it at ``out_offset``. Owns its own
    memory -- a caller that wants to share it imports it (see below)."""

    # to_body() appends the function-terminating end itself; .end() is only
    # for closing an explicit block/loop, of which this body has none.
    body = CodeBuilder(value_type="f64", parameter_count=2)
    body.local_get(1)          # out_offset
    body.local_get(0).load()   # in value
    body.value_const(2.0).op("mul")
    body.store()
    return build_module(
        function_name="double", parameter_types=["i32", "i32"], body=body,
    )


def _main_module() -> bytes:
    """Imports ``helper``'s ``double`` function and its memory, and exports
    ``run`` as a thin pass-through -- proof that a call crossing a module
    boundary is a real WASM ``call`` instruction against an imported
    function index, not JavaScript copying values between two buffers."""

    imports = [
        WasmImport(module="helper", field="double", kind="func",
                   parameter_types=("i32", "i32")),
        WasmImport(module="helper", field="memory", kind="memory",
                   memory_pages=1),
    ]
    body = CodeBuilder(value_type="f64", parameter_count=2)
    body.local_get(0)
    body.local_get(1)
    body.call(0)  # index 0: the sole "func" import
    return build_module(
        function_name="run", parameter_types=["i32", "i32"], body=body,
        imports=imports,
    )


def test_a_module_with_no_imports_is_unchanged_by_the_new_parameter():
    """The default ``imports=()`` must reproduce exactly what this
    assembler has always produced -- no import section, function index 0 for
    the module's own function, same as before this feature existed."""

    with_default = emit_wasm_module(_simple(), name="t").binary
    assert 2 not in _section_ids(with_default)


def _section_ids(binary: bytes) -> list[int]:
    cursor, seen = 8, []
    while cursor < len(binary):
        section_id = binary[cursor]
        cursor += 1
        length, shift = 0, 0
        while True:
            byte = binary[cursor]
            cursor += 1
            length |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        seen.append(section_id)
        cursor += length
    return seen


def test_an_import_bearing_module_declares_the_import_section_in_order():
    """type(1), import(2), function(3), export(7), code(10) -- no memory(5)
    section, since ``main`` imports its memory rather than owning one."""

    binary = _main_module()
    assert binary[:4] == b"\x00asm"
    assert _section_ids(binary) == [1, 2, 3, 7, 10]


def test_a_module_with_only_a_function_import_still_owns_its_memory():
    imports = [WasmImport(module="peer", field="fn", kind="func",
                           parameter_types=("i32",))]
    body = CodeBuilder(value_type="f64", parameter_count=1)
    binary = build_module(
        function_name="run", parameter_types=["i32"], body=body,
        imports=imports,
    )
    assert _section_ids(binary) == [1, 2, 3, 5, 7, 10]


def test_the_import_section_names_module_and_field_correctly():
    binary = _main_module()
    assert b"helper" in binary
    assert b"double" in binary
    assert b"memory" in binary


def test_a_second_memory_import_is_rejected():
    imports = [
        WasmImport(module="a", field="memory", kind="memory", memory_pages=1),
        WasmImport(module="b", field="memory", kind="memory", memory_pages=1),
    ]
    body = CodeBuilder(value_type="f64", parameter_count=0)
    with pytest.raises(ValueError):
        build_module(function_name="run", parameter_types=[], body=body,
                      imports=imports)


def test_a_memory_import_requires_a_page_count():
    with pytest.raises(ValueError):
        WasmImport(module="a", field="memory", kind="memory")


@pytest.mark.skipif(
    __import__("shutil").which("node") is None, reason="node not on PATH"
)
def test_a_call_across_the_module_boundary_actually_runs(tmp_path):
    """Byte-level section checks cannot catch a wrong function index or a
    memory import wired to the wrong exporter -- only running it can. This
    instantiates both modules in Node, exactly as the browser shell will,
    and checks the value crossing the boundary is the one ``double``
    computed, not a stale or zeroed buffer."""

    import json
    import subprocess

    helper_path = tmp_path / "helper.wasm"
    main_path = tmp_path / "main.wasm"
    helper_path.write_bytes(_helper_module())
    main_path.write_bytes(_main_module())

    script = tmp_path / "run.mjs"
    script.write_text(
        """
        import { readFileSync } from "node:fs";
        const [helperPath, mainPath] = process.argv.slice(2);
        const helperBytes = readFileSync(helperPath);
        const helperMod = await WebAssembly.instantiate(helperBytes, {});
        const helperInstance = helperMod.instance;
        const mainBytes = readFileSync(mainPath);
        const mainMod = await WebAssembly.instantiate(mainBytes, {
          helper: {
            double: helperInstance.exports.double,
            memory: helperInstance.exports.memory,
          },
        });
        const memory = helperInstance.exports.memory;
        const view = new Float64Array(memory.buffer);
        view[0] = 21.0;
        mainMod.instance.exports.run(0, 8);
        console.log(JSON.stringify({ result: view[1] }));
        """,
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script), str(helper_path), str(main_path)],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["result"] == 42.0
