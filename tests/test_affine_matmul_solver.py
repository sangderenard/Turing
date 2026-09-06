import numpy as np

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.abstract_nn import ProgramRunner
from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep
from src.compiler.affine_matmul_solver import analyze_affine_replacement
from src.common.tensors.autograd import GradTape, autograd


def _affine_program():
    meta = {index: Meta((2,), "float64") for index in range(1, 7)}
    program = FusedProgram(
        version=1,
        feeds={1, 2, 3},
        steps=[
            OpStep(0, "mul", [1, 2], {}, 4),
            OpStep(1, "add", [4, 3], {}, 5),
            OpStep(2, "neg", [5], {}, 6),
        ],
        outputs={"result": 6},
        meta=meta,
        extras={"capture_feed_origins": {1: {"binding_name": "x"}}},
    )
    feeds = {
        1: np.asarray((0.25, -0.5)),
        2: np.asarray((2.0, -3.0)),
        3: np.asarray((1.0, 4.0)),
    }
    return program, feeds


def test_isolated_affine_pieces_compose_to_the_whole_program_matrix():
    program, feeds = _affine_program()

    analysis = analyze_affine_replacement(program, feeds, variable_feed_ids=(1,))

    assert analysis.fully_replaceable
    assert not analysis.local_blockers
    assert analysis.composition_error is not None
    assert analysis.composition_error < 1e-12
    np.testing.assert_allclose(analysis.replacement.matrix, ((-2.0, 0.0), (0.0, 3.0)))
    np.testing.assert_allclose(analysis.replacement.bias, (-1.0, -4.0))
    solved = analysis.replacement({1: np.asarray((3.0, 5.0))})
    np.testing.assert_allclose(solved[6], (-7.0, 11.0))


def test_certified_affine_map_materializes_as_matmul_fused_program():
    program, feeds = _affine_program()
    replacement = analyze_affine_replacement(
        program, feeds, variable_feed_ids=(1,),
    ).replacement
    replacement_program, coefficient_feeds = replacement.to_fused_program()
    autograd.tape = GradTape()
    input_tensor = AT.tensor((3.0, 5.0))
    bias_id = next(feed for feed, value in coefficient_feeds.items() if value.ndim == 1)
    matrix_id = next(feed for feed, value in coefficient_feeds.items() if value.ndim == 2)
    matrix_tensor = AT.tensor(coefficient_feeds[matrix_id])
    bias_tensor = AT.tensor(coefficient_feeds[bias_id])
    for tensor in (input_tensor, matrix_tensor, bias_tensor):
        tensor._tape = autograd.tape
        autograd.tape.create_tensor_node(tensor)

    result = ProgramRunner(replacement_program)({
        1: input_tensor, matrix_id: matrix_tensor, bias_id: bias_tensor,
    })

    np.testing.assert_allclose(result["matmul_replacement"].tolist(), (-7.0, 11.0))


def test_nonlinear_piece_prevents_false_global_matmul_certificate():
    meta = {index: Meta((2,), "float64") for index in (1, 2)}
    program = FusedProgram(
        version=1, feeds={1},
        steps=[OpStep(0, "mul", [1, 1], {}, 2)],
        outputs={"square": 2}, meta=meta,
    )

    analysis = analyze_affine_replacement(
        program, {1: np.asarray((0.2, -0.3))}, variable_feed_ids=(1,),
    )

    assert not analysis.fully_replaceable
    assert [piece.operation for piece in analysis.local_blockers] == ["mul"]
    assert analysis.composed_matrix is None


def test_global_probe_can_certify_linear_cancellation_despite_local_blockers():
    meta = {index: Meta((2,), "float64") for index in range(1, 5)}
    program = FusedProgram(
        version=1, feeds={1},
        steps=[
            OpStep(0, "mul", [1, 1], {}, 2),
            OpStep(1, "mul", [1, 1], {}, 3),
            OpStep(2, "sub", [2, 3], {}, 4),
        ],
        outputs={"cancelled": 4}, meta=meta,
    )

    analysis = analyze_affine_replacement(
        program, {1: np.asarray((0.2, -0.3))}, variable_feed_ids=(1,),
    )

    assert analysis.local_blockers
    assert analysis.fully_replaceable
    np.testing.assert_allclose(analysis.replacement.matrix, 0.0)
    np.testing.assert_allclose(analysis.replacement.bias, 0.0)
