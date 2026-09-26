from src.compiler.nodus_canvas_kpn import (
    RegionEdge,
    canvas_from_regions,
    collapse_canvas_regions,
    parse_canvas,
    region_from_module,
    serialize_canvas,
)
from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue


def _mix_function():
    """tanh(a + b) — builtin-spellable add, plugin-spellable tanh."""

    a = SSAValue(0, "float32", ())
    b = SSAValue(1, "float32", ())
    summed = SSAValue(2, "float32", ())
    activated = SSAValue(3, "float32", ())
    block = BasicBlock("entry", [
        Instr("add", [a, b], summed),
        Instr("tanh", [summed], activated),
        Instr("Ret", [], None),
    ])
    return Function("mix", [a, b], {"entry": block}), (activated,)


def _polish_function():
    """sqrt(abs(a - b)) — Handler spelling for sub, plugin rows for the rest."""

    a = SSAValue(0, "float32", ())
    b = SSAValue(1, "float32", ())
    diff = SSAValue(2, "float32", ())
    mag = SSAValue(3, "float32", ())
    root = SSAValue(4, "float32", ())
    block = BasicBlock("entry", [
        Instr("Sub", [a, b], diff),
        Instr("abs", [diff], mag),
        Instr("sqrt", [mag], root),
        Instr("Ret", [], None),
    ])
    return Function("polish", [a, b], {"entry": block}), (root,)


def _gray_function():
    """x ^ (x >> 1) — reuses x, so it is NOT stack-serializable without dup."""

    x = SSAValue(0, "uint32", ())
    shifted = SSAValue(1, "uint32", ())
    gray = SSAValue(2, "uint32", ())
    block = BasicBlock("entry", [
        Instr("shr", [x], shifted, attributes={"right_scalar": 1}),
        Instr("bitxor", [x, shifted], gray),
        Instr("Ret", [], None),
    ])
    return Function("gray", [x], {"entry": block}), (gray,)


REAL_CANVAS_SAMPLE = """CANVAS V4
1000 720 56
OFFSET 0 0
MODULE 32 99 420 664 0 2 0 0 Table
MODULE 549 125 420 636 2 0 0 0 Table
ROWS 0 4 1 10 1 0 1 10 4 0 2 0 1 0 2 0 1 0
ROWS 1 2 0 0 1 0 0 0 1 0
EDGE 0 0 1 0 0 0
EDGE 0 1 1 1 0 0
NODE 13 0 0 0
NODE 14 1 0 0
"""


def test_real_canvas_document_roundtrips_byte_exactly():
    doc = parse_canvas(REAL_CANVAS_SAMPLE)
    assert len(doc.modules) == 2
    assert doc.modules[0].label == "Table"
    assert len(doc.modules[0].rows) == 4
    assert doc.modules[0].rows[0].tool == 10  # TableNumber builtin
    assert len(doc.edges) == 2 and len(doc.nodes) == 2
    assert serialize_canvas(doc) == REAL_CANVAS_SAMPLE


def test_regions_write_read_roundtrip_with_kpn_edge():
    regions = {
        "mix": _mix_function(),
        "polish": _polish_function(),
    }
    edges = [RegionEdge("mix", 0, "polish", 0)]
    doc, shortfalls = canvas_from_regions(regions, edges)
    assert shortfalls == ()

    # The builtin arithmetic spells as a builtin row; abs/sqrt/tanh spell as
    # canonical-vocabulary plugin rows.
    mix_rows = doc.modules[0].rows
    assert [r.kind for r in mix_rows] == [0, 1, 1, 2]
    assert mix_rows[1].tool == 1  # ModuleToolKind::Add
    assert mix_rows[2].plugin_id == "abstract_tensor.tanh"
    polish_rows = doc.modules[1].rows
    assert polish_rows[1].tool == 2  # ModuleToolKind::Subtract (Handler "Sub")
    assert [r.plugin_id for r in polish_rows[2:4]] == [
        "abstract_tensor.abs", "abstract_tensor.sqrt",
    ]

    # KPN edge carries the source contact's declared type in the wire
    # encoding (ValueTypeId + 1), so a float32 channel is genuinely typed
    # rather than colliding with the ABI's 0 = untyped/wildcard sentinel.
    assert doc.edges[0].a_module == 0 and doc.edges[0].b_module == 1
    assert doc.edges[0].type_id == 1  # VT_FLOAT32 (0) + 1
    assert doc.nodes[0].input_types == [1, 1]
    assert doc.nodes[0].output_types == [1]
    # Unified contact space: mix has 2 inputs, so its output is contact 2.
    assert doc.edges[0].a_contact == 2
    assert doc.edges[0].b_contact == 0

    # Full textual round trip back to SSA.
    parsed = parse_canvas(serialize_canvas(doc))
    mix = region_from_module(parsed, 0)
    assert mix.complete
    ops = [i.op for i in next(iter(mix.function.blocks.values())).instrs]
    assert ops == ["add", "tanh", "Ret"]
    assert len(mix.function.args) == 2 and len(mix.outputs) == 1

    polish = region_from_module(parsed, 1)
    assert polish.complete
    ops = [i.op for i in next(iter(polish.function.blocks.values())).instrs]
    assert ops == ["sub", "abs", "sqrt", "Ret"]


def test_value_reuse_region_is_withheld_not_misspelled():
    doc, shortfalls = canvas_from_regions({"gray": _gray_function()})
    # The module exists (topology preserved) but its rows are withheld.
    assert doc.modules[0].rows == []
    reasons = " ".join(s.reason for s in shortfalls)
    assert "scalar immediate" in reasons or "round-trip" in reasons


def test_opaque_plugin_row_is_a_named_reconstruction_shortfall():
    text = (
        "CANVAS V4\n1000 720 56\nOFFSET 0 0\n"
        "MODULE 0 0 400 300 1 1 0 0 mystery\n"
        "ROWS 0 3 0 0 1 0 1 0 1 1 kpath.utf16_stack_to_toolpath 2 0 1 0\n"
        "NODE 1 0 1 0 1 0\n"
    )
    doc = parse_canvas(text)
    region = region_from_module(doc, 0)
    assert not region.complete
    assert any("opaque to" in s.reason for s in region.shortfalls)


def _pipeline_chain(name, dtype, arg_count, ops):
    values = [SSAValue(i, dtype, ()) for i in range(arg_count)]
    instrs = []
    for op, operand_indices in ops:
        result = SSAValue(len(values), dtype, ())
        instrs.append(Instr(op, [values[i] for i in operand_indices], result))
        values.append(result)
    instrs.append(Instr("Ret", [], None))
    fn = Function(
        name, values[:arg_count], {"entry": BasicBlock("entry", instrs)},
    )
    return fn, (values[-1],)


def test_collapse_reduces_pipeline_to_fanout_boundaries_and_one_deep_table():
    regions = {
        "blend": _pipeline_chain("blend", "float32", 2, [
            ("add", (0, 1)), ("tanh", (2,)),
        ]),
        "contrast": _pipeline_chain("contrast", "float32", 2, [
            ("Sub", (0, 1)), ("abs", (2,)), ("sqrt", (3,)),
        ]),
        "gate": _pipeline_chain("gate", "float32", 2, [
            ("greater", (0, 1)),
        ]),
        "mixdown": _pipeline_chain("mixdown", "float32", 3, [
            ("mul", (1, 2)), ("add", (0, 3)), ("tanh", (4,)),
        ]),
        "polish": _pipeline_chain("polish", "float32", 1, [
            ("abs", (0,)), ("sqrt", (1,)), ("cos", (2,)),
        ]),
    }
    edges = [
        RegionEdge("blend", 0, "gate", 0),
        RegionEdge("contrast", 0, "gate", 1),
        RegionEdge("blend", 0, "mixdown", 1),      # blend fans out (2 uses)
        RegionEdge("gate", 0, "mixdown", 0),       # gate: single consumer
        RegionEdge("contrast", 0, "mixdown", 2),   # contrast fans out (2 uses)
        RegionEdge("mixdown", 0, "polish", 0),     # mixdown: single consumer
    ]
    live, channels, log = collapse_canvas_regions(regions, edges)

    # gate and mixdown fold into polish; fan-out producers survive.
    assert set(live) == {"blend", "contrast", "polish"}
    assert len(log) == 2
    compound_fn, compound_outs = live["polish"]
    # polish absorbed mixdown, which absorbed gate: 4 external contacts
    # (gate's two, blend's mixdown feed, contrast's mixdown feed).
    assert len(compound_fn.args) == 4
    ops = [i.op for i in next(iter(compound_fn.blocks.values())).instrs]
    assert ops == [
        "greater",            # inlined gate
        "mul", "add", "tanh",  # inlined mixdown
        "abs", "sqrt", "cos",  # polish itself
        "Ret",
    ]
    # Surviving channels all target the compound's renumbered contacts.
    assert {(c.source_region, c.target_region) for c in channels} == {
        ("blend", "polish"), ("contrast", "polish"),
    }
    assert sorted(c.target_contact for c in channels) == [0, 1, 2, 3]

    # The deep table spells as one self-verified row program with segmented
    # input pulls, and reconstructs completely.
    doc, shortfalls = canvas_from_regions(
        {name: live[name] for name in ("blend", "contrast", "polish")},
        channels,
    )
    assert shortfalls == ()
    deep_rows = doc.modules[2].rows
    assert sum(1 for r in deep_rows if r.kind == 1) == 7  # 7 compound ops
    parsed = parse_canvas(serialize_canvas(doc))
    deep = region_from_module(parsed, 2)
    assert deep.complete
    deep_ops = [
        i.op for i in next(iter(deep.function.blocks.values())).instrs
    ]
    assert deep_ops == [
        "greater", "mul", "add", "tanh", "abs", "sqrt", "cos", "Ret",
    ]


def test_unmodeled_records_survive_translation():
    text = REAL_CANVAS_SAMPLE + "MODULE_UUID 0 12345\nFRAMELED 0 1 2 7 0\n"
    doc = parse_canvas(text)
    assert doc.unmodeled == ["MODULE_UUID 0 12345", "FRAMELED 0 1 2 7 0"]
    out = serialize_canvas(doc)
    assert out.endswith("MODULE_UUID 0 12345\nFRAMELED 0 1 2 7 0\n")
