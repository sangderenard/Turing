from fractions import Fraction
import typing

import pytest

from src.common.tensors.topological_reducer import PRECISION_SINGULAR_NAMES
from src.common.tensors.extended_precision import Precision
from src.compiler.ir_identities import (
    FMA_MANDATORY,
    PRECISION_PIPELINE_METADATA,
    SECTION_ISOLATION,
    apply_precision_pipeline,
    precision_backend_shortfalls,
)
from src.transmogrifier.ssa import (
    BasicBlock,
    Function,
    IRModule,
    Instr,
    SSACallRecord,
    SSAValue,
)


def value(identifier):
    return SSAValue(identifier, dtype="float64")


def _lowered_limbs(function, value_id):
    return dict(function.metadata["precision_lowered_values"])[value_id]


def _fraction_limbs(number, width):
    remainder = Fraction(number)
    result = []
    for _index in range(width):
        limb = float(remainder)
        result.append(limb)
        remainder -= Fraction.from_float(limb)
    return result


def _evaluate_scalar_block(function, arguments):
    environment = {
        int(formal.id): float(actual)
        for formal, actual in zip(function.args, arguments)
    }
    for instruction in function.blocks["entry"].instrs:
        if instruction.res is None:
            continue
        operation = str(instruction.op)
        operands = [environment[int(arg.id)] for arg in instruction.args]
        if operation == "Const":
            result = float(instruction.attributes["constant"])
        elif operation == "Add":
            result = operands[0] + operands[1]
        elif operation == "Sub":
            result = operands[0] - operands[1]
        elif operation == "Mul":
            result = operands[0] * operands[1]
        elif operation == "Div":
            result = operands[0] / operands[1]
        elif operation == "Neg":
            result = -operands[0]
        elif operation == "Fma":
            result = float(
                Fraction.from_float(operands[0])
                * Fraction.from_float(operands[1])
                + Fraction.from_float(operands[2])
            )
        else:
            raise AssertionError(f"unexpected scalar operation {operation!r}")
        environment[int(instruction.res.id)] = result
    return environment


def test_precision_width_annotation_is_a_runtime_generic_alias():
    annotation = Precision[4]

    assert typing.get_origin(annotation) is Precision
    assert typing.get_args(annotation) == (4,)


def test_live_precision_wrapper_keeps_width_until_explicit_collapse():
    from src.common.tensors.abstraction import AbstractTensor

    wide = (Precision.of(AbstractTensor.get_tensor([1.0, 2.0]), 2) + 3.0) * 2.0

    assert isinstance(wide, Precision)
    assert wide.limbs == 2
    assert wide.to_float_lists() == [[8.0, 10.0], [0.0, 0.0]]
    assert wide.collapse().tolist() == [8.0, 10.0]
    with pytest.raises(AttributeError, match="collapse"):
        wide.mean()


def test_shader_capability_selection_does_not_claim_abstract_precision_ops():
    from src.compiler.machine_targets import targets

    precision_ops = set(PRECISION_SINGULAR_NAMES.values())
    registered = targets()
    assert not registered["glsl"].capabilities.supports(precision_ops)
    assert not registered["webgpu"].capabilities.supports(precision_ops)


def test_precision_pipeline_carries_derived_limbs_across_call_and_zero_extends_scalar():
    precision_add = PRECISION_SINGULAR_NAMES["Add"]
    caller_x, caller_y, derived = value(1), value(2), value(3)
    call_result = value(4)
    ordinary = value(5)
    callee_x, callee_y, callee_result = value(10), value(11), value(12)

    caller = Function("caller", [caller_x, caller_y], {
        "entry": BasicBlock("entry", [
            Instr(precision_add, [caller_x, caller_y], derived, attributes={
                "precision_limbs": 2,
                "precision_element": "float64",
            }),
            Instr("Add", [caller_x, caller_y], ordinary),
            Instr("Call", [derived, ordinary], call_result, attributes={
                "callee": "callee",
                "source_linked": True,
                "plan_callsite_id": 99,
            }),
        ])
    })
    callee = Function("callee", [callee_x, callee_y], {
        "entry": BasicBlock("entry", [
            Instr(precision_add, [callee_x, callee_y], callee_result,
                  attributes={"precision_limbs": 2}),
        ])
    })
    module = IRModule(
        {"caller": caller, "callee": callee},
        call_table={"caller": (SSACallRecord(
            caller="caller",
            callsite_id=99,
            callee_reference=None,
            callee_name="callee",
            callee_symbol="callee",
            argument_bindings=((3, 10), (5, 11)),
            callee_storage_value_ids=(10, 11),
            frame_bindings=((10, "caller_value", 3),
                            (11, "caller_value", 5)),
            resolution="native_call",
        ),)},
    )

    receipt = apply_precision_pipeline(module)

    call = next(
        instruction for instruction in caller.blocks["entry"].instrs
        if instruction.op == "Call"
    )
    assert len(callee.args) == 4
    assert len(call.args) == 4
    assert call.args[2].id != derived.id
    assert call.args[3].id != ordinary.id
    zero_definition = next(
        instruction for instruction in caller.blocks["entry"].instrs
        if instruction.res is not None
        and instruction.res.id == call.args[3].id
    )
    assert zero_definition.op == "Const"
    assert zero_definition.attributes["constant"] == 0.0

    record = module.call_table["caller"][0]
    assert (call.args[2].id, callee.args[2].id) in record.argument_bindings
    assert (call.args[3].id, callee.args[3].id) in record.argument_bindings
    assert receipt["call_records_refreshed"] == 1
    assert receipt["exact_only"] is True
    assert receipt["status"] == "lowered"
    assert {
        FMA_MANDATORY, SECTION_ISOLATION,
    }.issubset(receipt["section_contracts"][0]["obligations"])
    assert all(
        instruction.op not in set(PRECISION_SINGULAR_NAMES.values())
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )


def test_precision_pipeline_is_idempotent():
    x, y, result = value(1), value(2), value(3)
    function = Function("f", [x, y], {
        "entry": BasicBlock("entry", [Instr(
            PRECISION_SINGULAR_NAMES["Mul"], [x, y], result,
            attributes={"precision_limbs": 2},
        )])
    })
    module = IRModule({"f": function})

    first = apply_precision_pipeline(module)
    instruction_count = len(function.blocks["entry"].instrs)
    second = apply_precision_pipeline(module)

    assert second is first
    assert module.metadata[PRECISION_PIPELINE_METADATA] is first
    assert len(function.blocks["entry"].instrs) == instruction_count
    assert precision_backend_shortfalls(module, "llvm", ("f",)) == ()
    # wasm delivers isolation by construction (deterministic IEEE, no
    # reassociation licence); what it cannot deliver is the single-rounding
    # fma this flavour's sections demand.
    wasm = precision_backend_shortfalls(module, "wasm", ("f",))
    assert wasm[0]["missing"] == (FMA_MANDATORY,)
    # ieee_fma plus parenthesised expressions plus the per-artifact
    # -ffp-contract=off policy: fortran now meets both obligations.
    assert precision_backend_shortfalls(module, "fortran", ("f",)) == ()


def test_precision_pipeline_refuses_only_width_beyond_supported_ssa_lowering():
    x, y, result = value(1), value(2), value(3)
    module = IRModule({"f": Function("f", [x, y], {
        "entry": BasicBlock("entry", [Instr(
            PRECISION_SINGULAR_NAMES["Add"], [x, y], result,
            attributes={"precision_limbs": 5},
        )])
    })})

    with pytest.raises(ValueError, match="binary64: four limbs"):
        apply_precision_pipeline(module)


@pytest.mark.parametrize("width", (3, 4))
def test_precision_pipeline_lowers_every_multiply_limb_at_width_three_and_four(
    width,
):
    x, y, result = value(1), value(2), value(3)
    function = Function("f", [x, y], {
        "entry": BasicBlock("entry", [Instr(
            PRECISION_SINGULAR_NAMES["Mul"], [x, y], result,
            attributes={"precision_limbs": width},
        )])
    })
    module = IRModule({"f": function})

    receipt = apply_precision_pipeline(module)

    assert len(function.args) == 2 * width
    result_limbs = _lowered_limbs(function, result.id)
    assert len(result_limbs) == len(set(result_limbs)) == width
    assert sum(
        instruction.op == "Fma"
        for instruction in function.blocks["entry"].instrs
    ) == width * width
    assert receipt["section_contracts"][0]["limbs"] == width
    assert precision_backend_shortfalls(module, "llvm", ("f",)) == ()


@pytest.mark.parametrize("width", (3, 4))
@pytest.mark.parametrize("operator", ("Add", "Sub", "neg"))
def test_precision_pipeline_preserves_requested_width_for_linear_operations(
    width, operator,
):
    x, y, result = value(1), value(2), value(3)
    arguments = [x] if operator == "neg" else [x, y]
    function = Function("f", [x, y], {
        "entry": BasicBlock("entry", [Instr(
            PRECISION_SINGULAR_NAMES[operator], arguments, result,
            attributes={"precision_limbs": width},
        )])
    })

    apply_precision_pipeline(IRModule({"f": function}))

    result_limbs = _lowered_limbs(function, result.id)
    assert len(result_limbs) == len(set(result_limbs)) == width


@pytest.mark.parametrize("width", (3, 4))
def test_precision_pipeline_stores_every_array_limb_at_width_three_and_four(
    width,
):
    left, right, output = value(1), value(2), value(3)
    index = SSAValue(4, dtype="int64")
    left_pointer, right_pointer, output_pointer = value(5), value(6), value(7)
    left_value, right_value, result = value(8), value(9), value(10)
    function = Function("f", [left, right, output, index], {
        "entry": BasicBlock("entry", [
            Instr("GetElementPtr", [left, index], left_pointer),
            Instr("Load", [left_pointer], left_value),
            Instr("GetElementPtr", [right, index], right_pointer),
            Instr("Load", [right_pointer], right_value),
            Instr(
                PRECISION_SINGULAR_NAMES["Mul"],
                [left_value, right_value],
                result,
                attributes={"precision_limbs": width},
            ),
            Instr("GetElementPtr", [output, index], output_pointer),
            Instr("Store", [result, output_pointer], None),
        ])
    })

    apply_precision_pipeline(IRModule({"f": function}))

    stores = [
        instruction for instruction in function.blocks["entry"].instrs
        if instruction.op == "Store"
    ]
    assert len(stores) == width
    stored_ids = [int(instruction.args[0].id) for instruction in stores]
    assert stored_ids == list(_lowered_limbs(function, result.id))
    assert len(set(stored_ids)) == width


def test_width_four_ssa_multiplication_carries_more_exact_information_than_three():
    left_number = Fraction("1.23456789012345678901234567890123456789")
    right_number = Fraction("9.87654321098765432109876543210987654321")
    errors = {}

    for width in (3, 4):
        x, y, result = value(1), value(2), value(3)
        function = Function("f", [x, y], {
            "entry": BasicBlock("entry", [Instr(
                PRECISION_SINGULAR_NAMES["Mul"], [x, y], result,
                attributes={"precision_limbs": width},
            )])
        })
        apply_precision_pipeline(IRModule({"f": function}))

        left_limbs = _fraction_limbs(left_number, width)
        right_limbs = _fraction_limbs(right_number, width)
        # Lowering preserves original formals first, then appends each
        # formal's low limbs in source order.
        arguments = [
            left_limbs[0], right_limbs[0],
            *left_limbs[1:], *right_limbs[1:],
        ]
        environment = _evaluate_scalar_block(function, arguments)
        represented = sum(
            (Fraction.from_float(environment[identifier])
             for identifier in _lowered_limbs(function, result.id)),
            Fraction(),
        )
        exact_inputs = (
            sum(map(Fraction.from_float, left_limbs), Fraction())
            * sum(map(Fraction.from_float, right_limbs), Fraction())
        )
        errors[width] = abs(represented - exact_inputs)

    assert errors[4] < errors[3]


def test_width_four_ssa_division_carries_more_exact_information_than_three():
    numerator = Fraction("1.23456789012345678901234567890123456789")
    denominator = Fraction("9.87654321098765432109876543210987654321")
    errors = {}

    for width in (3, 4):
        x, y, result = value(1), value(2), value(3)
        function = Function("f", [x, y], {
            "entry": BasicBlock("entry", [Instr(
                PRECISION_SINGULAR_NAMES["Div"], [x, y], result,
                attributes={"precision_limbs": width},
            )])
        })
        apply_precision_pipeline(IRModule({"f": function}))

        left_limbs = _fraction_limbs(numerator, width)
        right_limbs = _fraction_limbs(denominator, width)
        environment = _evaluate_scalar_block(function, [
            left_limbs[0], right_limbs[0],
            *left_limbs[1:], *right_limbs[1:],
        ])
        represented = sum(
            (Fraction.from_float(environment[identifier])
             for identifier in _lowered_limbs(function, result.id)),
            Fraction(),
        )
        exact_inputs = (
            sum(map(Fraction.from_float, left_limbs), Fraction())
            / sum(map(Fraction.from_float, right_limbs), Fraction())
        )
        errors[width] = abs(represented - exact_inputs)

    assert errors[4] < errors[3]


@pytest.mark.parametrize("width", (2, 3, 4))
def test_source_compiler_runs_precision_pipeline_at_completed_module_seam(width):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm

    module, _outputs, _exports = lower_ast_source_to_ssa(
        f"def f(x: Precision[{width}], y: Precision[{width}]):\n"
        "    return x * y\n",
        "f",
    )

    receipt = module.metadata[PRECISION_PIPELINE_METADATA]
    assert receipt["status"] == "lowered"
    assert receipt["source_operations"] == 1
    assert receipt["section_contracts"][0]["fma_value_ids"]
    region = next(
        function for name, function in module.functions.items()
        if "planned_region" in name
    )
    wrapper = next(
        function for name, function in module.functions.items()
        if name.endswith("__f")
    )
    call = next(
        instruction for block in wrapper.blocks.values()
        for instruction in block.instrs if instruction.op == "Call"
    )
    assert len(region.args) == len(call.args) == 2 * width
    assert all(
        len(limb_ids) == width
        for _value_id, limb_ids in region.metadata["precision_lowered_values"]
    )
    assert any(
        instruction.op == "Fma"
        for block in region.blocks.values() for instruction in block.instrs
    )
    artifact = emit_ssa_function_to_llvm(module, wrapper.name)
    assert artifact.shortfalls == ()
    assert "llvm.fma.f64" in artifact.llvm_ir


def test_split_two_product_flavour_matches_fma_bit_for_bit_and_frees_wasm():
    """Dekker's split spelling is the SAME exact product, not an approximation.

    The error-free-transformation theorem says the fma residual and the
    Veltkamp-split residual are equal, not merely close -- so the two
    flavours must produce bit-identical limbs. The split flavour incurs no
    FMA_MANDATORY, which is what lets wasm (isolation by construction, no
    fma instruction) host a section honestly.
    """

    left_number = Fraction("1.23456789012345678901234567890123456789")
    right_number = Fraction("9.87654321098765432109876543210987654321")
    width = 3
    outcomes = {}

    for flavour in ("fma", "split"):
        x, y, result = value(1), value(2), value(3)
        function = Function("f", [x, y], {
            "entry": BasicBlock("entry", [Instr(
                PRECISION_SINGULAR_NAMES["Mul"], [x, y], result,
                attributes={"precision_limbs": width},
            )])
        })
        module = IRModule({"f": function})
        receipt = apply_precision_pipeline(
            module, two_product_flavor=flavour
        )
        assert receipt["two_product_flavor"] == flavour

        operations = {
            str(instruction.op)
            for block in function.blocks.values()
            for instruction in block.instrs
        }
        if flavour == "split":
            assert "Fma" not in operations
            # No fma was incurred, so the only obligation left is
            # isolation -- which wasm delivers and WGSL cannot.
            assert precision_backend_shortfalls(module, "wasm", ("f",)) == ()
            webgpu = precision_backend_shortfalls(module, "webgpu", ("f",))
            assert webgpu[0]["missing"] == (SECTION_ISOLATION,)
            with pytest.raises(ValueError, match="already lowered"):
                apply_precision_pipeline(module, two_product_flavor="fma")
        else:
            assert "Fma" in operations
            assert precision_backend_shortfalls(module, "wasm", ("f",)) != ()

        left_limbs = _fraction_limbs(left_number, width)
        right_limbs = _fraction_limbs(right_number, width)
        arguments = [
            left_limbs[0], right_limbs[0],
            *left_limbs[1:], *right_limbs[1:],
        ]
        environment = _evaluate_scalar_block(function, arguments)
        outcomes[flavour] = [
            environment[identifier]
            for identifier in _lowered_limbs(function, result.id)
        ]

    assert outcomes["split"] == outcomes["fma"]


def _lowered_precision_mul_module():
    """One precision_mul, lowered under the default "fma" flavour.

    The section that comes out carries both obligations -- FMA_MANDATORY
    (the flavour spells the residual as Fma instructions) and
    SECTION_ISOLATION -- which is exactly the pair "spirv" claims and the
    refusal-only shader lanes do not.
    """

    x, y, result = value(1), value(2), value(3)
    function = Function("f", [x, y], {
        "entry": BasicBlock("entry", [
            Instr(
                PRECISION_SINGULAR_NAMES["Mul"], [x, y], result,
                attributes={"precision_limbs": 2},
            ),
            Instr("Ret", [], None),
        ])
    })
    module = IRModule({"f": function})
    apply_precision_pipeline(module)
    return module, function


def test_spirv_backend_delivers_precision_sections_with_fma_and_nocontraction():
    from src.compiler.ssa_spirv_backend import emit_function

    module, function = _lowered_precision_mul_module()

    emitted = emit_function(function, outputs=(), module=module)

    # BACKEND_PRECISION_CAPABILITIES claims "spirv" honours both
    # obligations, so the guard at the emission entry point must find
    # nothing -- and neither may any per-instruction path, since every
    # instruction the lowering produced now has a SPIR-V spelling.
    assert emitted.complete, [s.format() for s in emitted.shortfalls]
    assert precision_backend_shortfalls(module, "spirv", ("f",)) == ()

    # The single rounding is the GLSL.std.450 extended instruction, never a
    # mul+add pair.
    assert '%glsl = OpExtInstImport "GLSL.std.450"' in emitted.source
    fma_lines = [
        line for line in emitted.source.splitlines()
        if "OpExtInst" in line and " %glsl Fma " in line
    ]
    fma_result_ids = {
        f"%t{instruction.res.id}"
        for instruction in function.blocks["entry"].instrs
        if str(instruction.op) == "Fma"
    }
    assert fma_result_ids
    assert len(fma_lines) == len(fma_result_ids)

    # Every arithmetic result inside the section -- the Fma results
    # INCLUDED, since GLSL.std.450 only guarantees Fma's single rounding
    # under the decoration -- carries NoContraction, placed in the
    # annotations section before any type declaration.
    decorated = {
        line.split()[1]
        for line in emitted.source.splitlines()
        if line.startswith("OpDecorate") and "NoContraction" in line
    }
    assert fma_result_ids <= decorated
    section_arithmetic = [
        instruction for instruction in function.blocks["entry"].instrs
        if instruction.attributes.get("precision_section")
        and str(instruction.op) in {"Add", "Sub", "Mul", "Div", "Neg", "Fma"}
    ]
    assert len(decorated) == len(section_arithmetic)
    assert emitted.source.index("NoContraction") < emitted.source.index("OpTypeFloat")


def test_spirv_backend_leaves_ordinary_instructions_contractible():
    from src.compiler.ssa_spirv_backend import emit_function

    a, b, product, total = value(1), value(2), value(3), value(4)
    function = Function("plain", [a, b], {
        "entry": BasicBlock("entry", [
            Instr("Mul", [a, b], product),
            Instr("Add", [product, b], total),
            Instr("Ret", [], None),
        ])
    })

    emitted = emit_function(function, outputs=[total])

    assert emitted.complete
    assert "NoContraction" not in emitted.source


def test_spirv_precision_module_assembles_and_validates():
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    from src.compiler.ssa_spirv_backend import emit_function

    if shutil.which("spirv-as") is None:
        pytest.skip("no spirv-as on PATH")

    module, function = _lowered_precision_mul_module()
    emitted = emit_function(function, outputs=(), module=module)
    assert emitted.complete

    workdir = Path(tempfile.mkdtemp(prefix="turing_spirv_precision_")).resolve()
    source = emitted.write(workdir)
    binary = workdir / f"{emitted.name}.spv"
    assembled = subprocess.run(
        ["spirv-as", str(source), "-o", str(binary)],
        capture_output=True, text=True,
    )
    assert assembled.returncode == 0, assembled.stderr
    validator = shutil.which("spirv-val")
    if validator is not None:
        validated = subprocess.run(
            [validator, str(binary)], capture_output=True, text=True,
        )
        assert validated.returncode == 0, validated.stderr


def test_webgpu_backend_refuses_precision_sections_loudly():
    from src.compiler.ssa_webgpu_backend import emit_module as emit_webgpu

    module, _function = _lowered_precision_mul_module()

    emitted = emit_webgpu(module, name="f")

    assert not emitted.complete
    refusals = [
        shortfall for shortfall in emitted.shortfalls
        if shortfall.operation == "precision_section"
    ]
    assert refusals, [s.format() for s in emitted.shortfalls]
    # WGSL's fma() may round twice and the language has no contraction
    # control, so BOTH obligations are named in the refusal.
    assert FMA_MANDATORY in refusals[0].reason
    assert SECTION_ISOLATION in refusals[0].reason


def test_webgl_backend_refuses_precision_sections_loudly():
    from src.compiler.ssa_webgl_backend import emit_ssa_webgl_fragment_module

    module, function = _lowered_precision_mul_module()

    # With the module in hand the persisted receipt is authoritative.
    emitted = emit_ssa_webgl_fragment_module(function, module=module)
    refusals = [
        shortfall for shortfall in emitted.shortfalls
        if shortfall.operation == "precision_section"
    ]
    assert refusals, [s for s in emitted.shortfalls]
    assert FMA_MANDATORY in refusals[0].reason
    assert SECTION_ISOLATION in refusals[0].reason

    # A bare Function -- the shape autogenesis actually hands this entry
    # point -- must be refused too, from the precision_section attribute the
    # pipeline stamps on every section instruction; otherwise the fused
    # path would emit the section's plain Add/Mul "successfully" and the
    # residual would silently come back zero.
    emitted_bare = emit_ssa_webgl_fragment_module(function)
    bare_refusals = [
        shortfall for shortfall in emitted_bare.shortfalls
        if shortfall.operation == "precision_section"
    ]
    assert bare_refusals, [s for s in emitted_bare.shortfalls]
    assert FMA_MANDATORY in bare_refusals[0].reason
    assert SECTION_ISOLATION in bare_refusals[0].reason


def test_float32_limb_sections_widen_to_eight_and_split_with_4097():
    """The binary32 ladder: wider sections, its own Veltkamp constant.

    A float32 limb carries 24 bits, so the proven ceiling doubles (eight
    limbs, ladder 2/4/6/8) and Dekker's split must use 2**12 + 1 --
    splitting with the binary64 constant would not degrade the residual,
    it would invalidate the theorem.
    """

    def f32_value(identifier):
        return SSAValue(identifier, dtype="float32")

    x, y, result = f32_value(1), f32_value(2), f32_value(3)
    function = Function("f", [x, y], {
        "entry": BasicBlock("entry", [Instr(
            PRECISION_SINGULAR_NAMES["Mul"], [x, y], result,
            attributes={
                "precision_limbs": 6,
                "precision_element": "float32",
            },
        )])
    })
    module = IRModule({"f": function})
    receipt = apply_precision_pipeline(module, two_product_flavor="split")

    assert receipt["status"] == "lowered"
    contract = receipt["section_contracts"][0]
    assert contract["limbs"] == 6
    assert contract["element"] == "float32"

    constants = {
        float(instruction.attributes.get("constant"))
        for block in function.blocks.values()
        for instruction in block.instrs
        if str(instruction.op) == "Const"
        and instruction.attributes.get("constant") is not None
    }
    assert 4097.0 in constants
    assert 134217729.0 not in constants
    operations = {
        str(instruction.op)
        for block in function.blocks.values()
        for instruction in block.instrs
    }
    assert "Fma" not in operations

    # binary64 keeps its four-limb ceiling: the same width that just
    # lowered for binary32 is refused for binary64 seeds.
    x64, y64, r64 = value(11), value(12), value(13)
    wide = IRModule({"g": Function("g", [x64, y64], {
        "entry": BasicBlock("entry", [Instr(
            PRECISION_SINGULAR_NAMES["Mul"], [x64, y64], r64,
            attributes={"precision_limbs": 6},
        )])
    })})
    with pytest.raises(ValueError, match="ceiling"):
        apply_precision_pipeline(wide)


def test_precision_scalar_operands_keep_element_shape():
    """A wide value against a plain scalar must keep the element shape.

    The dispatch's scalar expansion sliced an already-sliced template, so a
    three-element value against a scalar broadcast (3,) into (2,) and
    crashed; two-element values broadcast by coincidence, which is why the
    original wrapper test missed it. Three elements is the honest shape.
    """

    import numpy as np

    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.extended_precision import Precision

    wide = Precision.of(
        AbstractTensor.get_tensor(np.asarray([0.1, 0.2, 0.3])), 2
    )
    assert (wide * 0.5).collapse().tolist() == [0.05, 0.1, 0.15]
    assert (wide + 1.0).collapse().tolist() == [1.1, 1.2, 1.3]
    quotients = (1.0 / wide).collapse().tolist()
    assert quotients[0] == 10.0 and abs(quotients[2] - 10.0 / 3.0) < 1e-15
