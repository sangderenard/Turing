from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep
from src.compiler.sympy_math_renderer import render_reduced_program_mathematics


def test_reduced_program_uses_existing_relational_sympy_target():
    program = FusedProgram(
        version=1,
        feeds={1},
        steps=[
            OpStep(
                step_id=0,
                op_name="tensor_from_list",
                input_ids=[],
                attrs={"values": [2.0, 3.0]},
                result_id=2,
            ),
            OpStep(
                step_id=1,
                op_name="mul",
                input_ids=[1, 2],
                attrs={},
                result_id=3,
            ),
        ],
        outputs={"result": 3},
    )

    document = render_reduced_program_mathematics(
        program, input_names=("x",), program_name="demo"
    )
    mapping = document.to_mapping()

    assert mapping["target"] == "sympy"
    assert mapping["projection"] == "process_graph_to_sympy_relations"
    assert mapping["equation_count"] == 2
    assert mapping["program_relation"] == {
        "head": "And", "arity": 2, "arguments": "equations[*]",
    }
    assert mapping["depiction"] == {
        "kind": "function", "name": "demo",
        "inputs": ["x"], "outputs": ["result"],
    }
    assert mapping["outputs"][0]["name"] == "result"
    assert "value_3" in mapping["outputs"][0]["text"]
    assert "<math" in mapping["outputs"][0]["mathml"]
    assert "MathML" in mapping["outputs"][0]["mathml"]
    assert "&InvisibleTimes;" not in mapping["outputs"][0]["mathml"]
    assert "\u2062" in mapping["outputs"][0]["mathml"]


def test_boolean_output_is_depicted_as_a_predicate():
    program = FusedProgram(
        version=1,
        feeds={1, 2},
        steps=[OpStep(0, "less", [1, 2], {}, 3)],
        outputs={"inside": 3},
        meta={3: Meta(dtype="bool")},
    )

    depiction = render_reduced_program_mathematics(program).to_mapping()["depiction"]

    assert depiction["kind"] == "predicate"


def test_stateful_program_is_depicted_as_a_transition():
    program = FusedProgram(
        version=1,
        feeds={1},
        steps=[OpStep(0, "add", [1], {"right_scalar": 1.0}, 2)],
        outputs={"next": 2},
        state_in={1},
    )

    depiction = render_reduced_program_mathematics(program).to_mapping()["depiction"]

    assert depiction["kind"] == "transition"
