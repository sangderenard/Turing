import ast

from src.compiler.glsl_ast_ingestion import lower_glsl_source_to_ast


def test_glsl_parser_can_build_inspectable_python_ast_without_glsl_ops():
    result = lower_glsl_source_to_ast(
        """
        uniform float x;
        uniform float lo;
        uniform float hi;
        uniform float alpha;
        out float color;
        void main() {
            float bounded = clamp(x, lo, hi);
            color = mix(bounded, lo, alpha);
        }
        """
    )

    assert result.complete, result.shortfall_report()
    dumped = ast.dump(result.module, include_attributes=False)
    assert "clamp" not in dumped
    assert "mix" not in dumped
    assert "IfExp" in dumped
    assert "BinOp" in dumped

    namespace = {}
    exec(compile(result.module, "<glsl-ast>", "exec"), namespace)
    assert namespace["main"](2.0, 0.0, 1.0, 0.25) == 0.75


def test_ast_path_uses_canonical_existing_call_names():
    result = lower_glsl_source_to_ast(
        """
        in float x;
        out float color;
        void main() { color = sin(x) + sqrt(x); }
        """
    )

    calls = [
        node.func.id
        for node in ast.walk(result.module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert result.complete, result.shortfall_report()
    assert sorted(calls) == ["sin", "sqrt"]


def test_ast_path_reports_webgl_only_calls_without_placeholder_nodes():
    result = lower_glsl_source_to_ast(
        """
        uniform float source;
        out float color;
        void main() { color = texture(source, 0.0); }
        """
    )

    assert not result.complete
    assert "texture" not in {
        node.func.id
        for node in ast.walk(result.module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
