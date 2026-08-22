import re

import pytest

from src.compiler.llvm_optimizing_pipeline import (
    OptimizationProfile,
    REFERENCE_PROFILE,
    annotate_pointer_parameters,
    apply_fast_math,
    compare_profiles,
    optimize_ir,
)
from src.compiler.ssa_fortran_backend import (
    compile_module,
    emit_function,
    emit_module,
    fortran_compiler,
    supported_tensor_operations,
)
from src.transmogrifier.ssa import (
    BasicBlock,
    Function,
    IRModule,
    Instr,
    SSAValue,
)


ELEMENTWISE_IR = """
define void @axpy(ptr %a, ptr %b, ptr %out, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %body ]
  %done = icmp sge i32 %i, %n
  br i1 %done, label %exit, label %body
body:
  %i64 = sext i32 %i to i64
  %ap = getelementptr inbounds double, ptr %a, i64 %i64
  %bp = getelementptr inbounds double, ptr %b, i64 %i64
  %op = getelementptr inbounds double, ptr %out, i64 %i64
  %av = load double, ptr %ap, align 8
  %bv = load double, ptr %bp, align 8
  %m = fmul double %av, 2.500000e+00
  %s = fadd double %m, %bv
  store double %s, ptr %op, align 8
  %i.next = add nsw i32 %i, 1
  br label %loop
exit:
  ret void
}
"""


def test_fortran_api_describes_scalar_source_projection():
    text = SSAValue(0, dtype="unknown")
    encoded_length = SSAValue(5, dtype="int64")
    function = Function("projection", [text, encoded_length], {
        "entry": BasicBlock("entry", [Instr("Ret", [], None)], []),
    }, metadata={
        "parameter_names": (("text", 0),),
        "scalar_source_transforms": ((5, "text", "utf8_length"),),
    })

    emitted = emit_module(
        IRModule({"projection": function}),
        name="scalar_source_projection",
        extra_roots=("projection",),
    )

    assert emitted.complete
    parameters = {
        parameter.name: parameter
        for parameter in emitted.api.entry_point("projection").parameters
    }
    assert parameters["t5"].source_name == "text"
    assert parameters["t5"].source_transform == "utf8_length"


def test_fortran_call_folds_nested_row_address_to_array_section():
    child = SSAValue(0, "int", (12,))
    offset = SSAValue(1, "int")
    row = SSAValue(2, "int", (3,))
    callee_row = SSAValue(3, "int", (3,))
    callee = Function(
        "consume_row",
        [callee_row],
        {"entry": BasicBlock("entry", [], None)},
    )
    caller = Function(
        "select_row",
        [child, offset],
        {
            "entry": BasicBlock(
                "entry",
                [
                    Instr("GetElementPtr", [child, offset], row),
                    Instr(
                        "Call", [row], None,
                        attributes={"callee": "consume_row"},
                    ),
                ],
                None,
            )
        },
    )

    source = emit_module({callee.name: callee, caller.name: caller}).source

    assert "call consume_row(" in source
    assert "t0(t1 + 1)" in source


def _elementwise_function():
    a = SSAValue(0, "float64", (64,))
    b = SSAValue(1, "float64", (64,))
    scaled = SSAValue(3, "float64", (64,))
    summed = SSAValue(4, "float64", (64,))
    activated = SSAValue(5, "float64", (64,))
    total = SSAValue(2, "float64")
    block = BasicBlock(
        "entry",
        [
            Instr("mul", [a], scaled, attributes={"right_scalar": 2.5}),
            Instr("add", [scaled, b], summed),
            Instr("tanh", [summed], activated),
            Instr("sum", [activated], total),
            Instr("Ret", [], SSAValue(9)),
        ],
    )
    function = Function("axpy_tanh_sum", [a, b], {"entry": block})
    return function, activated, total


def _loop_function():
    x = SSAValue(0, "float64")
    n = SSAValue(1, "float64")
    seed = SSAValue(2, "float64")
    acc = SSAValue(3, "float64")
    nxt = SSAValue(4, "float64")
    cond = SSAValue(5, "bool")
    blocks = {
        "entry": BasicBlock(
            "entry",
            [
                Instr("const", [], seed, attributes={"constant": 0.0}),
                Instr("Br", [], SSAValue(90), attributes={"target": "loop"}),
            ],
        ),
        "loop": BasicBlock(
            "loop",
            [
                Instr(
                    "Phi",
                    [],
                    acc,
                    attributes={"incoming": [("entry", seed), ("body", nxt)]},
                ),
                Instr("Lt", [acc, n], cond),
                Instr(
                    "CondBr",
                    [cond],
                    SSAValue(91),
                    attributes={"true": "body", "false": "exit"},
                ),
            ],
        ),
        "body": BasicBlock(
            "body",
            [
                Instr("add", [acc, x], nxt),
                Instr("Br", [], SSAValue(92), attributes={"target": "loop"}),
            ],
        ),
        "exit": BasicBlock("exit", [Instr("Ret", [], SSAValue(93))]),
    }
    return Function("accumulate", [x, n], blocks), acc


def test_fortran_emits_integer_bitwise_operators_without_numeric_projection():
    left = SSAValue(0, "int64")
    right = SSAValue(1, "int64")
    union = SSAValue(2, "int64")
    function = Function(
        "bitwise_union",
        [left, right],
        {
            "entry": BasicBlock(
                "entry",
                [
                    Instr("bitor", [left, right], union),
                    Instr("Ret", [union], SSAValue(3)),
                ],
            )
        },
    )

    source = emit_function(function, outputs=[union]).source

    assert "ior(" in source
    assert "real(ior(" not in source


def test_fortran_nested_bitwise_operands_use_one_integer_kind():
    left = SSAValue(0, "int")
    right = SSAValue(1, "int")
    mask = SSAValue(2, "float64")
    union = SSAValue(3, "int")
    inverted = SSAValue(4, "int")
    result = SSAValue(5, "int")
    function = Function(
        "nested_bits",
        [left, right, mask],
        {
            "entry": BasicBlock(
                "entry",
                [
                    Instr("bitor", [left, right], union),
                    Instr("invert", [mask], inverted),
                    Instr("bitand", [union, inverted], result),
                    Instr("Ret", [result], SSAValue(6)),
                ],
            )
        },
    )

    source = emit_function(function, outputs=[result]).source

    assert "ior(int(t0, c_int64_t), int(t1, c_int64_t))" in source
    assert "iand(int(" in source


def test_fortran_repository_right_shift_preserves_signed_source_semantics():
    value = SSAValue(0, "int")
    amount = SSAValue(1, "int")
    result = SSAValue(2, "int")
    function = Function(
        "signed_shift",
        [value, amount],
        {"entry": BasicBlock("entry", [Instr("Shr", [value, amount], result)])},
    )

    emitted = emit_function(function, outputs=[result])

    assert emitted.complete, [item.format() for item in emitted.shortfalls]
    assert "shifta(" in emitted.source
    assert "shiftr(" not in emitted.source


def test_repository_integer_or_is_not_coerced_through_logical_merge():
    left = SSAValue(0, "int", ())
    right = SSAValue(1, "int", ())
    result = SSAValue(2, "int", ())
    function = Function(
        "integer_or",
        [left, right],
        {"entry": BasicBlock("entry", [Instr("Or", [left, right], result)])},
    )

    emitted = emit_function(function, outputs=[result])

    assert emitted.complete, [item.format() for item in emitted.shortfalls]
    assert "ior(" in emitted.source
    assert "merge(" not in emitted.source


def test_fortran_truthiness_of_inlined_bitmask_compares_with_zero():
    value = SSAValue(0, "int")
    bit = SSAValue(1, "int")
    # Control-expression lowering may record the intermediate as bool because
    # Python consumes it through truthiness; its producer still defines an
    # integer bit mask.
    mask = SSAValue(2, "bool")
    absent = SSAValue(3, "bool")
    function = Function(
        "bitmask_not",
        [value, bit],
        {
            "entry": BasicBlock(
                "entry",
                [
                    Instr("bitand", [value, bit], mask),
                    Instr("LNot", [mask], absent),
                ],
            )
        },
    )

    emitted = emit_function(function, outputs=[absent])

    assert emitted.complete, [item.format() for item in emitted.shortfalls]
    assert "== 0_c_int64_t" in emitted.source
    assert ".not. int(iand(" not in emitted.source


def test_fortran_logical_composition_truth_converts_inlined_bitmask():
    value = SSAValue(0, "int")
    bit = SSAValue(1, "int")
    flag = SSAValue(2, "bool")
    mask = SSAValue(3, "bool")
    result = SSAValue(4, "bool")
    function = Function(
        "bitmask_and_flag",
        [value, bit, flag],
        {
            "entry": BasicBlock(
                "entry",
                [
                    Instr("bitand", [value, bit], mask),
                    Instr("LAnd", [mask, flag], result),
                ],
            )
        },
    )

    emitted = emit_function(function, outputs=[result])

    assert emitted.complete, [item.format() for item in emitted.shortfalls]
    assert "/= 0_c_int64_t" in emitted.source
    assert ".and." in emitted.source


# ---------------------------------------------------------------- Fortran


def test_elementwise_ssa_emits_whole_array_fortran():
    function, activated, total = _elementwise_function()
    module = emit_module(
        IRModule({"axpy_tanh_sum": function}),
        name="turing_demo",
        outputs={"axpy_tanh_sum": [activated, total]},
    )
    source = module.source

    assert module.complete, [s.format() for s in module.shortfalls]
    assert 'bind(C, name="axpy_tanh_sum")' in source
    assert "use, intrinsic :: iso_c_binding" in source
    # Whole-array expressions rather than explicit element loops, and the
    # single-use chain folded into one statement.
    assert "t5 = tanh(((t0 * 2.5_c_double) + t1))" in source
    assert "t2 = sum(t5)" in source


def test_empty_array_constant_has_a_typed_fortran_constructor():
    empty = SSAValue(0, "float64", (0, 3))
    function = Function(
        "empty_constant",
        [],
        {
            "entry": BasicBlock(
                "entry",
                [
                    Instr("Const", [], empty, attributes={"values": ()}),
                    Instr("Ret", [], SSAValue(1)),
                ],
            )
        },
    )

    module = emit_module(
        IRModule({function.name: function}),
        outputs={function.name: [empty]},
    )

    assert module.complete, [shortfall.format() for shortfall in module.shortfalls]
    assert "reshape([real(c_double) ::], [0, 3])" in module.source


def test_captured_scalar_in_values_is_not_treated_as_an_array():
    scalar = SSAValue(0, "float64", ())
    function = Function(
        "captured_scalar_constant",
        [],
        {
            "entry": BasicBlock(
                "entry",
                [
                    Instr("Const", [], scalar, attributes={"values": 2.0}),
                    Instr("Ret", [], SSAValue(1)),
                ],
            )
        },
    )

    module = emit_module(
        IRModule({function.name: function}),
        outputs={function.name: [scalar]},
    )

    assert module.complete, [shortfall.format() for shortfall in module.shortfalls]
    assert "2.0_c_double" in module.source


def test_imported_llvm_scalar_literals_emit_fortran_constants():
    zero = SSAValue(0, "i32", ())
    positive_infinity = SSAValue(1, "float64", ())
    function = Function(
        "llvm_constants",
        [],
        {
            "entry": BasicBlock(
                "entry",
                [
                    Instr(
                        "Const", [], zero,
                        attributes={"llvm_literal": "i32 0"},
                    ),
                    Instr(
                        "Const", [], positive_infinity,
                        attributes={
                            "llvm_literal": "double 0x7FF0000000000000"
                        },
                    ),
                ],
            )
        },
    )

    module = emit_module(
        {function.name: function},
        outputs={function.name: [zero, positive_infinity]},
    )

    assert module.complete, [item.format() for item in module.shortfalls]
    assert "ieee_value(0.0_c_double, ieee_positive_inf)" in module.source


def test_fortran_module_omits_helpers_replaced_by_native_tensor_ops():
    argument = SSAValue(0, "float64", (4,))
    result = SSAValue(1, "float64", (4,))
    public = Function(
        "public_numeric",
        [argument],
        {
            "entry": BasicBlock(
                "entry",
                [Instr("add", [argument], result, attributes={"right_scalar": 1})],
            )
        },
        metadata={"named_outputs": (("result", 1),)},
    )
    dead_helper = Function(
        "llvm_helper_replaced_by_native_expression",
        [argument],
        {"entry": BasicBlock("entry", [Instr("einsum", [argument], result)])},
    )

    module = emit_module(
        {public.name: public, dead_helper.name: dead_helper},
        outputs={public.name: [result]},
    )

    assert module.complete, [item.format() for item in module.shortfalls]
    assert tuple(item.name for item in module.subroutines) == (public.name,)


def test_fortran_module_derives_outputs_from_ssa_metadata():
    argument = SSAValue(0, "float64", (4,))
    result = SSAValue(1, "float64", (4,))
    function = Function(
        "metadata_outputs",
        [argument],
        {
            "entry": BasicBlock(
                "entry",
                [Instr("add", [argument], result, attributes={"right_scalar": 1})],
            )
        },
        metadata={"named_outputs": (("result", 1),)},
    )

    module = emit_module({function.name: function})

    assert module.complete, [item.format() for item in module.shortfalls]
    assert "intent(out) :: t1(extent_4)" in module.source


def test_scalar_reduction_is_identity_and_logical_arithmetic_is_numeric():
    logical = SSAValue(0, "bool", ())
    numeric = SSAValue(1, "bool", ())
    reduced = SSAValue(2, "float64", (1,))
    function = Function(
        "scalar_numpy_semantics",
        [],
        {
            "entry": BasicBlock(
                "entry",
                [
                    Instr("Const", [], logical, attributes={"value": True}),
                    Instr(
                        "add", [logical], numeric,
                        attributes={"right_scalar": 0},
                    ),
                    Instr(
                        "Call", [numeric], reduced,
                        attributes={"tensor_operation": "sum"},
                    ),
                ],
            )
        },
        metadata={"named_outputs": (("result", 2),)},
    )

    module = emit_module({function.name: function})

    assert module.complete, [item.format() for item in module.shortfalls]
    assert "sum(" not in module.source
    assert ".true._c_bool +" not in module.source


def test_fortran_converts_logical_values_at_cast_phi_and_store_boundaries():
    logical = SSAValue(0, "bool")
    cast_result = SSAValue(1, "float64")
    cast_function = Function("logical_cast", [logical], {
        "entry": BasicBlock("entry", [
            Instr("Cast", [logical], cast_result),
            Instr("Ret", [cast_result], SSAValue(2)),
        ]),
    })
    cast_source = emit_function(
        cast_function, outputs=[cast_result]
    ).source
    assert "merge(1.0_c_double, 0.0_c_double, t0)" in cast_source
    assert "real(t0" not in cast_source

    initial = SSAValue(0, "bool")
    current = SSAValue(1, "float64")
    phi_function = Function("logical_phi", [initial], {
        "entry": BasicBlock("entry", [
            Instr("Br", [], SSAValue(90), attributes={"target": "join"}),
        ], ["join"]),
        "join": BasicBlock("join", [
            Instr(
                "Phi", [initial], current,
                attributes={"incoming_blocks": ("entry",)},
            ),
            Instr("Ret", [current], SSAValue(91)),
        ]),
    })
    phi_source = emit_function(phi_function, outputs=[current]).source
    assert "t1 = merge(1.0_c_double, 0.0_c_double, t0)" in phi_source

    collection = SSAValue(0, "float64", (4,))
    stored = SSAValue(1, "bool")
    index = SSAValue(2, "int32")
    address = SSAValue(3, "float64")
    store_function = Function("logical_store", [collection, stored, index], {
        "entry": BasicBlock("entry", [
            Instr("GetElementPtr", [collection, index], address),
            Instr("Store", [stored, address], None),
        ]),
    })
    store_source = emit_function(
        store_function, array_base_ids={collection.id}
    ).source
    assert (
        "t0(t2 + 1) = merge(1.0_c_double, 0.0_c_double, t1)"
        in store_source
    )


def test_index_set_accepts_a_scalar_value_without_reducing_it():
    base = SSAValue(0, "float64", (5,))
    value = SSAValue(1, "float64", ())
    result = SSAValue(2, "float64", (5,))
    function = Function(
        "scalar_index_set",
        [base, value],
        {
            "entry": BasicBlock(
                "entry",
                [Instr(
                    "Call", [base, value], result,
                    attributes={"tensor_operation": "index_set", "slices": 0},
                )],
            )
        },
        metadata={"named_outputs": (("result", 2),)},
    )

    module = emit_module({function.name: function})

    assert module.complete, [item.format() for item in module.shortfalls]
    assert "sum(t1)" not in module.source
    assert "t2(1) = t1" in module.source


def test_ieee_classification_ops_emit_logical_fortran_expressions():
    value = SSAValue(0, "float64", (4,))
    isnan = SSAValue(1, "bool", (4,))
    isfinite = SSAValue(2, "bool", (4,))
    function = Function(
        "classify",
        [value],
        {
            "entry": BasicBlock(
                "entry",
                [
                    Instr(
                        "Call",
                        [value],
                        isnan,
                        attributes={"tensor_operation": "isnan"},
                    ),
                    Instr(
                        "Call",
                        [value],
                        isfinite,
                        attributes={"tensor_operation": "isfinite"},
                    ),
                    Instr("Ret", [], SSAValue(3)),
                ],
            )
        },
    )

    module = emit_module(
        IRModule({function.name: function}),
        outputs={function.name: [isnan, isfinite]},
    )

    assert module.complete, [shortfall.format() for shortfall in module.shortfalls]
    assert "use, intrinsic :: ieee_arithmetic" in module.source
    assert "ieee_is_nan(t0)" in module.source
    assert "ieee_is_finite(t0)" in module.source


def test_single_use_temporaries_are_fused_into_one_expression():
    """One SSA chain must become one array statement, not one per step.

    A statement per SSA step materialises an array temporary per step, turning
    a single fused traversal into N passes over memory.  At array sizes past
    cache that dominates the runtime, so this is a correctness-of-intent test
    for the emitter, not a cosmetic one.
    """

    function, activated, total = _elementwise_function()
    source = emit_function(function, outputs=[activated, total]).source

    # The intermediate values must not appear as declared arrays at all.
    assert "t3(" not in source
    assert "t4(" not in source
    # Exactly two assignments survive: the fused elementwise chain, and the
    # reduction that genuinely cannot fuse into it.
    assignments = re.findall(r"^\s*t\d+ = ", source, flags=re.MULTILINE)
    assert len(assignments) == 2, source


def test_bind_c_arrays_are_explicit_shape_over_a_passed_extent():
    """A bind(C) procedure may not take assumed-shape dummies.

    ``x(:)`` needs a descriptor a C caller cannot construct before Fortran 2018
    / TS 29113, and gfortran rejects it outright.  Arrays must be explicit-shape
    over an extent supplied by the caller.
    """

    function, activated, total = _elementwise_function()
    source = emit_function(function, outputs=[activated, total]).source

    # One extent parameter per distinct dimension size, named after the
    # size itself (dimension_extents), not a single shared "n_elements" --
    # this is what lets a different-shaped op like matmul share the same
    # emitter instead of being permanently out of reach.
    assert "integer(c_int), intent(in), value :: extent_64" in source
    assert "intent(in) :: t0(extent_64)" in source
    assert "intent(out) :: t5(extent_64)" in source
    assert "intent(out) :: t2" in source
    # No assumed-shape dummy may survive anywhere in a bind(C) subroutine,
    # and nothing may reach the heap.
    assert "(:)" not in source
    assert "allocatable" not in source


def test_fill_initializes_a_large_predeclared_span_without_array_literals():
    span = SSAValue(20, "float64", (70_000,))
    function = Function("zero_span", [], {
        "entry": BasicBlock("entry", [
            Instr("Fill", [], span, attributes={"fill_value": 0.0}),
            Instr("Ret", [], SSAValue(99)),
        ]),
    })

    source = emit_function(function, outputs=[span]).source

    assert "intent(out) :: t20(extent_70000)" in source
    assert "t20 = 0.0_c_double" in source
    assert "[" not in source
    assert "allocatable" not in source


def test_argument_output_alias_emits_one_inout_fortran_arena():
    arena = SSAValue(20, "float64", (4,))
    function = Function("advance", [arena], {
        "entry": BasicBlock("entry", [
            Instr("Ret", [], SSAValue(99)),
        ]),
    })

    module = emit_module(
        IRModule({"advance": function}),
        outputs={"advance": [arena]},
    )
    source = module.source
    entry = module.api.entry_point("advance")

    assert "subroutine advance(extent_4, t20)" in source
    assert "intent(inout) :: t20(extent_4)" in source
    assert source.count(":: t20(extent_4)") == 1
    assert [parameter.role for parameter in entry.parameters] == [
        "extent",
        "inout",
    ]


def test_argument_assigned_by_linked_result_is_an_inout_slot():
    frame_slot = SSAValue(20, "float64")
    function = Function("wrapper", [frame_slot], {
        "entry": BasicBlock("entry", [
            Instr("Cast", [frame_slot], frame_slot),
            Instr("Ret", [], SSAValue(99)),
        ]),
    })

    source = emit_function(function).source

    assert "intent(inout) :: t20" in source
    assert "intent(in), value :: t20" not in source


def test_linked_sequence_result_may_publish_into_explicit_frame_slot():
    formal_output = SSAValue(1, "int64", (8,))
    callee = Function("produce", [formal_output], {
        "entry": BasicBlock("entry", [Instr("Ret", [], SSAValue(90))]),
    })
    caller_output = SSAValue(7, "int64", (8,))
    semantic_result = SSAValue(40, "aggregate")
    caller = Function("consume", [caller_output], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call", [caller_output], semantic_result,
                attributes={
                    "callee": "produce",
                    "ssa_output_argument": 0,
                    "result_aliases_frame": True,
                },
            ),
            Instr("Ret", [], SSAValue(91)),
        ]),
    })

    emitted = emit_module(
        IRModule({"produce": callee, "consume": caller}),
        outputs={"produce": [formal_output]},
        extra_roots=("consume",),
    )

    assert emitted.complete, [item.format() for item in emitted.shortfalls]
    assert "call produce(extent_8, t7)" in emitted.source
    assert "UNSUPPORTED Call" not in emitted.source


def test_linked_sequence_result_declares_non_public_frame_as_local_array():
    formal_output = SSAValue(1, "int64", (8,))
    callee = Function("produce_local", [formal_output], {
        "entry": BasicBlock("entry", [Instr("Ret", [], SSAValue(90))]),
    })
    caller_frame = SSAValue(7, "int64", (8,))
    caller = Function("consume_local", [], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call", [caller_frame], caller_frame,
                attributes={
                    "callee": "produce_local",
                    "ssa_output_argument": 0,
                    "result_aliases_frame": True,
                },
            ),
            Instr("Ret", [], SSAValue(91)),
        ]),
    })

    emitted = emit_module(
        IRModule({"produce_local": callee, "consume_local": caller}),
        outputs={"produce_local": [formal_output]},
        extra_roots=("consume_local",),
    )

    assert emitted.complete, [item.format() for item in emitted.shortfalls]
    assert "integer(c_int64_t) :: t7(extent_8)" in emitted.source
    assert "call produce_local(extent_8, t7)" in emitted.source


def test_fortran_shortens_internal_procedure_names_but_preserves_bind_symbol():
    callee_name = "program__specialized_" + "a" * 70
    caller_name = "program__wrapper_" + "b" * 70
    callee = Function(callee_name, [], {
        "entry": BasicBlock("entry", [Instr("Ret", [], None)]),
    })
    caller = Function(caller_name, [], {
        "entry": BasicBlock("entry", [
            Instr("Call", [], None, attributes={"callee": callee_name}),
            Instr("Ret", [], None),
        ]),
    })

    emitted = emit_module(
        IRModule({caller_name: caller, callee_name: callee}),
        extra_roots=(caller_name,),
    )
    symbols = emitted.api.metadata["fortran_internal_symbols"]

    assert len(symbols[callee_name]) <= 63
    assert len(symbols[caller_name]) <= 63
    assert f'bind(C, name="{callee_name}")' in emitted.source
    assert f"call {symbols[callee_name]}()" in emitted.source
    assert emitted.api.entry_point(callee_name).symbol == callee_name


def test_inout_region_load_and_phi_retain_resident_arena_rank():
    arena = SSAValue(20, "float64", (4,))
    region = Function("advance", [arena], {
        "entry": BasicBlock("entry", [
            Instr("Ret", [], SSAValue(99)),
        ]),
    })
    aggregate = SSAValue(10, "aggregate")
    address = SSAValue(11, "pointer")
    rank_lost_load = SSAValue(12, "float64")
    rank_lost_phi = SSAValue(13, "float64")
    caller_arena = SSAValue(1, "float64", (4,))
    caller = Function("cycle", [caller_arena], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call", [caller_arena], aggregate,
                attributes={
                    "callee": "advance",
                    "result_convention": "ssa.aggregate",
                },
            ),
            Instr(
                "GetElementPtr", [aggregate], address,
                attributes={"aggregate_index": 0},
            ),
            Instr("Load", [address], rank_lost_load),
            Instr(
                "Phi", [rank_lost_load], rank_lost_phi,
                attributes={"incoming": (("entry", rank_lost_load),)},
            ),
            Instr("Ret", [], SSAValue(100)),
        ]),
    })

    source = emit_module(
        IRModule({"advance": region, "cycle": caller}),
        outputs={"advance": [arena]},
    ).source

    assert "real(c_double) :: t12(extent_4)" in source
    assert "real(c_double) :: t13(extent_4)" in source
    assert "t12 = t1" in source
    assert "call advance(extent_4, t12)" in source


def test_repeated_aggregate_output_identity_is_one_native_argument():
    repeated = SSAValue(30, "float64")
    callee = Function(
        "repeat_value",
        [],
        {"entry": BasicBlock("entry", [Instr("Ret", [], SSAValue(99))])},
    )
    aggregate = SSAValue(10, "aggregate")
    first_address = SSAValue(11, "pointer")
    second_address = SSAValue(12, "pointer")
    projected = SSAValue(13, "float64")
    caller = Function("consume_repeat", [], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call", [], aggregate,
                attributes={
                    "callee": "repeat_value",
                    "result_convention": "ssa.aggregate",
                },
            ),
            Instr(
                "GetElementPtr", [aggregate], first_address,
                attributes={"aggregate_index": 0},
            ),
            Instr("Load", [first_address], projected),
            Instr(
                "GetElementPtr", [aggregate], second_address,
                attributes={"aggregate_index": 1},
            ),
            Instr("Load", [second_address], projected),
            Instr("Ret", [], SSAValue(100)),
        ]),
    })

    source = emit_module(
        IRModule({"repeat_value": callee, "consume_repeat": caller}),
        outputs={
            "repeat_value": [repeated, repeated],
            "consume_repeat": [projected],
        },
    ).source

    assert "subroutine repeat_value(t30)" in source
    assert "call repeat_value(t13)" in source
    assert "call repeat_value(t13, t13)" not in source


def test_aggregate_projection_inherits_callee_output_dtype():
    predicate = SSAValue(30, "bool")
    callee = Function(
        "make_predicate",
        [],
        {"entry": BasicBlock("entry", [Instr("Ret", [], SSAValue(99))])},
    )
    aggregate = SSAValue(10, "aggregate")
    address = SSAValue(11, "pointer")
    untyped_projection = SSAValue(12)
    caller = Function("consume_predicate", [], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call", [], aggregate,
                attributes={
                    "callee": "make_predicate",
                    "result_convention": "ssa.aggregate",
                },
            ),
            Instr(
                "GetElementPtr", [aggregate], address,
                attributes={"aggregate_index": 0},
            ),
            Instr("Load", [address], untyped_projection),
            Instr("Ret", [], SSAValue(100)),
        ]),
    })

    source = emit_module(
        IRModule({"make_predicate": callee, "consume_predicate": caller}),
        outputs={
            "make_predicate": [predicate],
            "consume_predicate": [untyped_projection],
        },
    ).source

    assert "logical(c_bool), intent(out) :: t12" in source
    assert "call make_predicate(t12)" in source


def test_generic_index_addresses_lower_to_fortran_loads_and_stores():
    arena = SSAValue(20, "float64", (3, 4))
    row = SSAValue(21, "int32")
    column = SSAValue(22, "int32")
    value = SSAValue(23, "float64")
    address = SSAValue(24, "pointer")
    stored = SSAValue(25, "void")
    loaded = SSAValue(26, "float64")
    function = Function("indexed_arena", [arena, row, column, value], {
        "entry": BasicBlock("entry", [
            Instr("GetElementPtr", [arena, row, column], address),
            Instr("Store", [value, address], stored),
            Instr("Load", [address], loaded),
            Instr("Ret", [], SSAValue(99)),
        ]),
    })

    source = emit_function(function, outputs=[loaded]).source

    assert "intent(inout) :: t20(extent_3, extent_4)" in source
    assert "t20(t21 + 1, t22 + 1) = t23" in source
    assert "t26 = t20(t21 + 1, t22 + 1)" in source
    assert "UNSUPPORTED" not in source


def test_numeric_where_mask_and_scalar_branch_are_conformed_for_fortran():
    mask = SSAValue(20, "float64", (4,))
    when_true = SSAValue(21, "float64", (4,))
    when_false = SSAValue(22, "float64", (1,))
    result = SSAValue(23, "float64", (4,))
    function = Function("numeric_where", [mask, when_true, when_false], {
        "entry": BasicBlock("entry", [
            Instr("where", [mask, when_true, when_false], result),
            Instr("Ret", [], SSAValue(99)),
        ]),
    })

    source = emit_function(function, outputs=[result]).source

    assert "merge(t21, t22(1), (t20 /= 0.0_c_double))" in source


def test_where_promotes_mixed_branch_kinds_to_the_result_dtype():
    mask = SSAValue(30, "bool", (4,))
    integers = SSAValue(31, "int64", (1,))
    reals = SSAValue(32, "float64", (1,))
    result = SSAValue(33, "float64", (4,))
    function = Function("promoted_where", [mask, integers, reals], {
        "entry": BasicBlock("entry", [
            Instr("where", [mask, integers, reals], result),
            Instr("Ret", [], SSAValue(99)),
        ]),
    })

    source = emit_function(function, outputs=[result]).source

    assert "merge(real(t31(1), c_double), t32(1), t30)" in source


def test_api_describes_transitive_callee_extents_from_final_signature():
    field = SSAValue(0, "float64", (4,))
    scalar_temporary = SSAValue(1, "float64", (1,))
    result = SSAValue(2, "float64", (4,))
    callee = Function("region", [field], {
        "entry": BasicBlock("entry", [
            Instr("sum", [field], scalar_temporary),
            Instr("add", [field], result, attributes={"right_scalar": 1.0}),
            Instr("Ret", [], SSAValue(90)),
        ]),
    })

    aggregate = SSAValue(10, "aggregate")
    address = SSAValue(11, "pointer")
    public_result = SSAValue(12, "float64", (4,))
    caller = Function("whole_program", [field], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call", [field], aggregate,
                attributes={
                    "callee": "region",
                    "result_convention": "ssa.aggregate",
                },
            ),
            Instr(
                "GetElementPtr", [aggregate], address,
                attributes={"aggregate_index": 0},
            ),
            Instr("Load", [address], public_result),
            Instr("Ret", [], SSAValue(91)),
        ]),
    })

    outer_aggregate = SSAValue(20, "aggregate")
    outer_address = SSAValue(21, "pointer")
    outer_result = SSAValue(22, "float64", (4,))
    outer = Function("outer_program", [field], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call", [field], outer_aggregate,
                attributes={
                    "callee": "whole_program",
                    "result_convention": "ssa.aggregate",
                },
            ),
            Instr(
                "GetElementPtr", [outer_aggregate], outer_address,
                attributes={"aggregate_index": 0},
            ),
            Instr("Load", [outer_address], outer_result),
            Instr("Ret", [], SSAValue(92)),
        ]),
    })

    module = emit_module(
        IRModule({
            "region": callee,
            "whole_program": caller,
            "outer_program": outer,
        }),
        outputs={
            "region": [result],
            "whole_program": [public_result],
            "outer_program": [outer_result],
        },
    )
    control_api = module.api.entry_point("whole_program")
    outer_api = module.api.entry_point("outer_program")

    assert [
        parameter.name
        for parameter in control_api.parameters
        if parameter.role == "extent"
    ] == ["extent_1", "extent_4"]
    assert "subroutine whole_program(extent_1, extent_4" in module.source
    assert [
        parameter.name
        for parameter in outer_api.parameters
        if parameter.role == "extent"
    ] == ["extent_1", "extent_4"]
    assert "subroutine outer_program(extent_1, extent_4" in module.source
    assert "call whole_program(extent_1, extent_4" in module.source


def test_unreferenced_block_labels_are_not_emitted():
    """A label nothing branches to is a warning and pure noise."""

    function, activated, total = _elementwise_function()
    straight_line = emit_function(function, outputs=[activated, total]).source
    assert "continue" not in straight_line

    loop, acc = _loop_function()
    with_branches = emit_function(loop, outputs=[acc]).source
    assert "continue" in with_branches


def test_control_flow_lowers_phi_into_predecessor_assignments():
    function, acc = _loop_function()
    source = emit_function(function, outputs=[acc]).source

    # Phi is realised by assigning in each predecessor before the branch,
    # never by a Fortran construct pretending to be a phi.
    entry_section = source.split("! block loop")[0]
    body_section = source.split("! block body")[1]
    assert "t3 = t2" in entry_section
    assert "t3 = t4" in body_section
    # The comparison feeding the branch is single-use, so it folds into the
    # condition rather than occupying a named temporary.
    assert "if ((t3 < t1)) then" in source
    assert source.count("goto") >= 3


def test_compact_phi_incoming_blocks_emit_predecessor_assignments():
    initial = SSAValue(0, "int32")
    updated = SSAValue(1, "int32")
    current = SSAValue(2, "int32")
    function = Function("compact_phi", [], {
        "entry": BasicBlock("entry", [
            Instr("const", [], initial, attributes={"constant": 0}),
            Instr("Br", [], SSAValue(90), attributes={"target": "loop"}),
        ], ["loop"]),
        "loop": BasicBlock("loop", [
            Instr(
                "Phi", [initial, updated], current,
                attributes={"incoming_blocks": ("entry", "latch")},
            ),
            Instr("Br", [], SSAValue(91), attributes={"target": "latch"}),
        ], ["latch"]),
        "latch": BasicBlock("latch", [
            Instr("add", [current], updated, attributes={"right_scalar": 1}),
            Instr("Br", [], SSAValue(92), attributes={"target": "loop"}),
        ], ["loop"]),
    })

    source = emit_function(function).source

    assert "t2 = t0" in source
    assert "t2 = t1" in source


def test_values_live_across_blocks_are_not_fused_away():
    """Inlining must never move a computation across control flow."""

    function, acc = _loop_function()
    source = emit_function(function, outputs=[acc]).source

    # t4 is defined in 'body' and consumed by the loop-header phi, so it must
    # remain a real assignment inside the block that computes it.
    assert "t4 = (t3 + t0)" in source
    assert "real(c_double) :: t4" in source


def test_unsupported_operation_is_reported_not_guessed():
    value = SSAValue(0, "float64", (8,))
    result = SSAValue(1, "float64", (8,))
    block = BasicBlock("entry", [Instr("einsum", [value], result)])
    function = Function("mystery", [value], {"entry": block})

    subroutine = emit_function(function, outputs=[result])
    assert not subroutine.complete
    assert subroutine.shortfalls[0].op == "einsum"
    assert "UNSUPPORTED einsum" in subroutine.source


def test_shapeless_axis_operation_is_reported_instead_of_dividing_by_zero():
    value = SSAValue(0, "float64", ())
    result = SSAValue(1, "float64", ())
    function = Function(
        "shapeless_stack",
        [value],
        {
            "entry": BasicBlock(
                "entry",
                [Instr(
                    "Call",
                    [value],
                    result,
                    attributes={"tensor_operation": "stack", "dim": 0},
                )],
            )
        },
    )

    subroutine = emit_function(function, outputs=[result])

    assert not subroutine.complete
    assert subroutine.shortfalls[0].op == "stack"
    assert "UNSUPPORTED Call" in subroutine.source


def test_control_markers_and_validation_failure_emit_without_external_symbols():
    function = Function(
        "guarded_deployment",
        [],
        {
            "entry": BasicBlock(
                "entry",
                [
                    Instr("Deploy", [], None),
                    Instr("Join", [], None),
                    Instr(
                        "Call",
                        [],
                        None,
                        attributes={
                            "callee": "turing_validation_error",
                            "error_code": 7,
                        },
                    ),
                    Instr("Ret", [], SSAValue(0)),
                ],
            )
        },
        metadata={"control_ir": True},
    )

    module = emit_module(IRModule({function.name: function}))

    assert module.complete, [item.format() for item in module.shortfalls]
    assert "! Deploy deployment boundary" in module.source
    assert "! Join deployment boundary" in module.source
    assert "error stop 7" in module.source
    assert "call turing_validation_error" not in module.source


def test_extent_uses_inferred_rank_for_a_lossy_control_value_occurrence():
    resident = SSAValue(0, "float64", (8,))
    rank_lost_reference = SSAValue(0, "float64", ())
    extent = SSAValue(1, "int32", ())
    function = Function(
        "resident_extent",
        [resident],
        {
            "entry": BasicBlock(
                "entry",
                [
                    Instr(
                        "Call",
                        [rank_lost_reference],
                        extent,
                        attributes={"tensor_operation": "extent", "dim": 0},
                    ),
                    Instr("Ret", [], SSAValue(2)),
                ],
            )
        },
    )

    subroutine = emit_function(function, outputs=[extent])

    assert subroutine.complete, [item.format() for item in subroutine.shortfalls]
    assert "t1 = size(t0, 1)" in subroutine.source


def test_dynamic_array_extent_is_an_explicit_pointer_length_abi_pair():
    values = SSAValue(0, "float64", ())
    extent = SSAValue(1, "int32", ())
    function = Function(
        "dynamic_extent",
        [values],
        {
            "entry": BasicBlock(
                "entry",
                [
                    Instr(
                        "Call",
                        [values],
                        extent,
                        attributes={"tensor_operation": "extent", "dim": 0},
                    ),
                    Instr("Ret", [], SSAValue(2)),
                ],
            )
        },
    )

    # Whole-object assembly supplies this set from address use across the
    # method.  Exercise the per-function emitter directly with that fact too.
    subroutine = emit_function(
        function,
        outputs=[extent],
        array_base_ids={values.id},
    )

    assert subroutine.complete, [item.format() for item in subroutine.shortfalls]
    assert "extent_dynamic_0" in subroutine.extent_names
    assert "intent(in) :: t0(extent_dynamic_0)" in subroutine.source
    assert "t1 = extent_dynamic_0" in subroutine.source


def test_dynamic_rank_two_contract_preserves_rank_with_leading_extent():
    values = SSAValue(
        0,
        "float64",
        (),
        accounting={"program_abi_storage": "span", "program_abi_rank": 2},
    )
    row = SSAValue(1, "int32")
    column = SSAValue(2, "int32")
    address = SSAValue(3, "ptr")
    loaded = SSAValue(4, "float64")
    function = Function(
        "dynamic_matrix",
        [values, row, column],
        {
            "entry": BasicBlock(
                "entry",
                [
                    Instr("GetElementPtr", [values, row, column], address),
                    Instr("Load", [address], loaded),
                    Instr("Ret", [], SSAValue(5)),
                ],
            )
        },
    )

    subroutine = emit_function(
        function,
        outputs=[loaded],
        array_base_ids={values.id},
    )

    assert subroutine.complete, [item.format() for item in subroutine.shortfalls]
    assert "extent_dynamic_0_1" in subroutine.extent_names
    assert "extent_dynamic_0_2" in subroutine.extent_names
    assert (
        "intent(in) :: t0(extent_dynamic_0_1, extent_dynamic_0_2)"
        in subroutine.source
    )
    assert "t4 = t0(t1 + 1, t2 + 1)" in subroutine.source


def test_cast_like_rank_closure_is_monotonic():
    scalar_value = SSAValue(0, "float64")
    reference = SSAValue(1, "float64")
    result = SSAValue(2, "float64")
    caller = Function(
        "cast_like_rank",
        [scalar_value, reference],
        {"entry": BasicBlock("entry", [
            Instr("CastLike", [scalar_value, reference], result),
            Instr("Call", [result], None, attributes={"callee": "rank_consumer"}),
            Instr("Ret", [], SSAValue(3)),
        ])},
    )
    array = SSAValue(
        0,
        "float64",
        accounting={"program_abi_storage": "span", "program_abi_rank": 2},
    )
    callee = Function(
        "rank_consumer",
        [array],
        {"entry": BasicBlock("entry", [Instr("Ret", [], SSAValue(1))])},
    )

    module = emit_module(
        {caller.name: caller, callee.name: callee},
    )

    assert module.complete, [item.format() for item in module.shortfalls]
    assert "call rank_consumer" in module.source


def test_dynamic_rank_two_workspace_is_described_without_becoming_input():
    workspace = SSAValue(
        0,
        "float64",
        (),
        accounting={
            "linked_call_frame_storage": "callee.buffer",
            "ssa_call_rank": 2,
        },
    )
    row = SSAValue(1, "int32")
    column = SSAValue(2, "int32")
    address = SSAValue(3, "ptr")
    loaded = SSAValue(4, "float64")
    function = Function(
        "dynamic_workspace",
        [workspace, row, column],
        {
            "entry": BasicBlock(
                "entry",
                [
                    Instr("GetElementPtr", [workspace, row, column], address),
                    Instr("Load", [address], loaded),
                    Instr("Ret", [], SSAValue(5)),
                ],
            )
        },
    )

    module = emit_module({function.name: function})
    entry = module.api.entry_point(function.name)
    parameter = next(item for item in entry.parameters if item.name == "t0")

    assert parameter.role == "workspace"
    assert parameter.extent is None
    assert len(parameter.extents) == 2
    assert parameter.extents[0].endswith("_0_1")
    assert parameter.extents[1].endswith("_0_2")


def test_python_float_is_scalar_extraction_from_a_resident_array():
    values = SSAValue(
        0,
        "float64",
        (),
        accounting={"program_abi_storage": "span", "program_abi_rank": 1},
    )
    scalar = SSAValue(1, "float64")
    function = Function(
        "scalar_extract",
        [values],
        {"entry": BasicBlock("entry", [
            Instr(
                "Cast",
                [values],
                scalar,
                attributes={
                    "extraction_identity": "builtins.float",
                    "source_operator": "float",
                    "target_dtype": "float64",
                },
            ),
            Instr("Ret", [scalar], None),
        ])},
    )

    module = emit_module(
        {function.name: function},
        outputs={function.name: (scalar,)},
    )
    source = module.source

    assert "intent(in) :: t0(extent_dynamic_" in source
    assert "real(c_double), intent(out) :: t1" in source
    assert "t1 = real(t0(1), c_double)" in source


def test_resident_representation_boundaries_and_native_transpose_emit_directly():
    matrix = SSAValue(0, "float64", (2, 3))
    detached = SSAValue(1, "float64", (2, 3))
    host = SSAValue(2, "float64", (2, 3))
    listed = SSAValue(3, "float64", (2, 3))
    transposed = SSAValue(4, "float64", (3, 2))
    function = Function(
        "representation_boundaries",
        [matrix],
        {
            "entry": BasicBlock(
                "entry",
                [
                    Instr("Call", [matrix], detached, attributes={"tensor_operation": "detach"}),
                    Instr("Call", [detached], host, attributes={"tensor_operation": "cpu"}),
                    Instr("Call", [host], listed, attributes={"tensor_operation": "tolist"}),
                    Instr(
                        "Call",
                        [listed],
                        transposed,
                        attributes={
                            "tensor_operation": "transpose",
                            "dim0": 0,
                            "dim1": 1,
                        },
                    ),
                    Instr("Ret", [], SSAValue(5)),
                ],
            )
        },
    )

    subroutine = emit_function(function, outputs=[transposed])

    assert subroutine.complete, [item.format() for item in subroutine.shortfalls]
    assert "transpose(t0)" in subroutine.source
    assert {"cpu", "detach", "tolist", "transpose"} <= supported_tensor_operations()


@pytest.mark.parametrize(
    ("operation", "dtype", "expression"),
    [
        ("double", "float64", "real(t0, c_double)"),
        ("float", "float32", "real(t0, c_float)"),
        ("int", "int32", "int(t0, c_int32_t)"),
        ("long", "int64", "int(t0, c_int64_t)"),
        ("to_dtype", "int64", "int(t0, c_int64_t)"),
    ],
)
def test_resident_dtype_conversions_use_explicit_fortran_kinds(
    operation, dtype, expression
):
    source = SSAValue(0, "float64", ())
    result = SSAValue(1, dtype, ())
    attributes = {"tensor_operation": operation}
    if operation == "to_dtype":
        attributes["dtype"] = dtype
    function = Function(
        "convert_value",
        [source],
        {"entry": BasicBlock(
            "entry",
            [Instr("Call", [source], result, attributes=attributes)],
        )},
    )

    subroutine = emit_function(function, outputs=[result])

    assert subroutine.complete, [item.format() for item in subroutine.shortfalls]
    assert expression in subroutine.source


@pytest.mark.parametrize(
    ("dtype", "expression"),
    [("bool", "(.not. t0)"), ("int32", "not(t0)")],
)
def test_invert_uses_the_dtype_appropriate_fortran_operation(dtype, expression):
    source = SSAValue(0, dtype, ())
    result = SSAValue(1, dtype, ())
    function = Function(
        "invert_value",
        [source],
        {"entry": BasicBlock("entry", [Instr(
            "Call", [source], result,
            attributes={"tensor_operation": "invert"},
        )])},
    )

    subroutine = emit_function(function, outputs=[result])

    assert subroutine.complete, [item.format() for item in subroutine.shortfalls]
    assert expression in subroutine.source


@pytest.mark.skipif(
    fortran_compiler() is None, reason="no Fortran compiler installed"
)
def test_generated_fortran_actually_compiles(tmp_path):
    function, activated, total = _elementwise_function()
    module = emit_module(
        IRModule({"axpy_tanh_sum": function}),
        name="turing_demo",
        outputs={"axpy_tanh_sum": [activated, total]},
    )
    library = compile_module(module, directory=tmp_path)
    assert library.exists()


# ------------------------------------------------------------------- LLVM


def test_reference_profile_reproduces_the_unoptimized_path():
    result = optimize_ir(ELEMENTWISE_IR, REFERENCE_PROFILE)
    assert result.profile.opt == 0
    assert not result.profile.annotate_noalias
    # Scalar double arithmetic only: no packed SIMD.
    assert result.vector_instruction_count == 0


def test_optimizing_profile_produces_packed_simd():
    reference, optimized = compare_profiles(ELEMENTWISE_IR)

    assert reference.vector_instruction_count == 0
    assert optimized.vectorized
    # Packed forms, not the scalar sd/ss forms that also use xmm registers.
    assert re.search(r"\bv?(add|mul)pd\b", optimized.assembly)


def test_noalias_removes_runtime_alias_versioning_not_vectorization():
    """``noalias`` buys smaller code, not the ability to vectorize.

    LLVM's loop vectorizer handles unproven aliasing by versioning the loop
    behind a runtime memory check, so it vectorizes with or without the
    attribute.  Asserting that ``noalias`` *enables* SIMD would be wrong.  What
    it actually removes is the check and the scalar fallback copy — which is
    the advantage Fortran gets for free from its argument rules.
    """

    without = optimize_ir(
        ELEMENTWISE_IR,
        OptimizationProfile(annotate_noalias=False),
    )
    with_noalias = optimize_ir(
        ELEMENTWISE_IR,
        OptimizationProfile(annotate_noalias=True),
    )

    assert without.vectorized and with_noalias.vectorized
    assert with_noalias.vector_instruction_count == (
        without.vector_instruction_count
    )
    # The saving is the elided versioning path.
    assert len(with_noalias.assembly.splitlines()) < len(
        without.assembly.splitlines()
    )


def test_pointer_annotation_is_confined_to_pointer_parameters():
    annotated = annotate_pointer_parameters(ELEMENTWISE_IR)
    assert "ptr noalias %a" in annotated
    assert "ptr noalias %out" in annotated
    # The integer parameter must not be touched.
    assert "i32 %n" in annotated
    assert "i32 noalias" not in annotated
    # Applying twice must not double-annotate.
    assert annotate_pointer_parameters(annotated) == annotated


def test_fast_math_is_off_by_default_and_explicit_when_requested():
    assert not OptimizationProfile().fast_math
    assert "fast" not in optimize_ir(ELEMENTWISE_IR).hardened_ir.split("fadd")[0]

    fast = apply_fast_math(ELEMENTWISE_IR)
    assert "fmul fast" in fast
    assert "fadd fast" in fast
