from src.compiler.glsl_source_ingestion import lower_glsl_source_to_ssa
from src.compiler.glsl_source_tables import (
    GLSL_BINARY_TO_SSA,
    GLSL_UNARY_TO_SSA,
    SSA_TO_GLSL_BINARY,
    SSA_TO_GLSL_UNARY,
)
from src.transmogrifier.ssa_registry import Handler


def _instructions(result):
    return result.module.functions["main"].blocks["entry"].instrs


def test_glsl_lexical_tables_are_bidirectional_views_of_existing_handlers():
    assert all(isinstance(value, Handler) for value in GLSL_BINARY_TO_SSA.values())
    assert all(isinstance(value, Handler) for value in GLSL_UNARY_TO_SSA.values())
    assert {
        SSA_TO_GLSL_BINARY[handler]: handler
        for handler in SSA_TO_GLSL_BINARY
    } == dict(GLSL_BINARY_TO_SSA)
    assert {
        SSA_TO_GLSL_UNARY[handler]: handler
        for handler in SSA_TO_GLSL_UNARY
    } == dict(GLSL_UNARY_TO_SSA)


def test_straight_line_glsl_lowers_to_existing_ssa_without_source_ops():
    result = lower_glsl_source_to_ssa(
        """
        #version 330 core
        uniform float gain;
        in float pressure;
        out float intensity;
        void main() {
            float scaled = pressure * gain;
            intensity = sin(scaled) + 0.25;
        }
        """
    )

    assert result.complete, result.shortfall_report()
    instructions = _instructions(result)
    assert [item.op for item in instructions] == [
        Handler.Mul.value,
        Handler.Call.value,
        Handler.Const.value,
        Handler.Add.value,
        Handler.Ret.value,
    ]
    assert instructions[1].attributes == {"callee": "sin"}
    assert instructions[-1].args == [instructions[-2].res]
    assert not ({"sin", "*", "+"} & {item.op for item in instructions})


def test_glsl_conveniences_decompose_into_existing_ssa_handlers():
    result = lower_glsl_source_to_ssa(
        """
        uniform float x;
        uniform float lo;
        uniform float hi;
        out float color;
        void main() {
            float bounded = clamp(x, lo, hi);
            color = mix(bounded, hi, smoothstep(lo, hi, x));
        }
        """
    )

    assert result.complete, result.shortfall_report()
    operations = [item.op for item in _instructions(result)]
    assert "clamp" not in operations
    assert "mix" not in operations
    assert "smoothstep" not in operations
    assert {Handler.Lt.value, Handler.Gt.value, Handler.Select.value} <= set(
        operations
    )
    assert {Handler.Add.value, Handler.Sub.value, Handler.Mul.value} <= set(
        operations
    )


def test_webgl_only_texture_call_uses_existing_shortfall_path_not_fake_ssa():
    result = lower_glsl_source_to_ssa(
        """
        uniform float source;
        out float color;
        void main() {
            color = texture(source, 0.0);
        }
        """
    )

    assert not result.complete
    assert "no exact existing SSA/ProcessGraph operation" in (
        result.shortfall_report()
    )
    operations = [item.op for item in _instructions(result)]
    assert "texture" not in operations
    assert operations == [Handler.Const.value, Handler.Ret.value]


def test_ternary_and_compound_assignment_use_select_and_arithmetic():
    result = lower_glsl_source_to_ssa(
        """
        in float x;
        in float y;
        out float result_value;
        void main() {
            float selected = x < y ? x : y;
            selected += 2.0;
            result_value = selected;
        }
        """
    )

    assert result.complete, result.shortfall_report()
    operations = [item.op for item in _instructions(result)]
    assert operations == [
        Handler.Lt.value,
        Handler.Select.value,
        Handler.Const.value,
        Handler.Add.value,
        Handler.Ret.value,
    ]
