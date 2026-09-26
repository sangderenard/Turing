from types import SimpleNamespace

import pytest

from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep
from src.compiler.control_source import ControlProgram, SequenceBlock, StatementBlock
from src.compiler.parametric_card_program import build_parametric_card_program


def _compilation():
    first = FusedProgram(
        1, {1}, [OpStep(0, "add", [1], {"right_scalar": 1.0}, 2)],
        {"seam": 2}, meta={1: Meta((4,), "float64"), 2: Meta((4,), "float64")},
    )
    second = FusedProgram(
        1, {20}, [OpStep(0, "mul", [20], {"right_scalar": 2.0}, 3)],
        {"result": 3}, meta={20: Meta((4,), "float64"), 3: Meta((4,), "float64")},
    )
    return SimpleNamespace(
        entrypoint="cycle",
        shell_control_program=ControlProgram(
            SequenceBlock((
                StatementBlock(("__scheduled_region_0__",)),
                StatementBlock(("__scheduled_region_1__",)),
            )),
            region_indices=(0, 1),
        ),
        region_programs={0: first, 1: second},
        identity_table={"state": (1,), "seam": (2, 20), "result": (3,)},
        public_input_value_ids={"state": 1},
        public_output_value_ids={"result": 3},
        hierarchical_value_aliases={20: 2},
        map_ir={},
    )


def test_card_program_retains_control_regions_and_exact_alias_edges():
    cards = build_parametric_card_program(_compilation())
    mapping = cards.to_mapping()

    assert mapping["abi"] == "turing.parametric-card-program.v1"
    assert mapping["control"]["region_indices"] == [0, 1]
    assert [card["kind"] for card in mapping["cards"]] == [
        "control", "numerical-region", "numerical-region",
    ]
    seam = next(
        edge for edge in mapping["connections"]
        if edge["from"] == "cycle::region::0"
        and edge["to"] == "cycle::region::1"
    )
    assert seam["bindings"] == [{
        "from": "seam",
        "to": "seam",
        "storage_value_id": 2,
        "rewrite": "alias",
    }]
    assert "__scheduled_region_0__" in mapping["control"]["python"]


def test_card_program_rejects_control_dispatch_to_missing_card():
    compilation = _compilation()
    compilation.shell_control_program = ControlProgram(
        StatementBlock(("__scheduled_region_7__",)),
        region_indices=(7,),
    )

    with pytest.raises(ValueError, match="absent numerical cards: 7"):
        build_parametric_card_program(compilation)


def test_card_feedback_must_name_public_ports():
    with pytest.raises(ValueError, match="unknown public card ports"):
        build_parametric_card_program(
            _compilation(), feedback={"missing": "result"},
        )
