from src.common.tensors.abstract_nn.demo_fused_field_training import run_demo


def test_fused_field_program_visibly_learns():
    result = run_demo(
        epochs=120,
        learning_rate=0.03,
        hidden=20,
        training_resolution=10,
        training_phases=4,
        display_resolution=16,
        output=None,
        capture_backward=True,
    )
    assert result["program_steps"] > 1
    assert result["forward_nodes"] > 0
    assert result["backward_nodes"] > 0
    assert result["backward_program_steps"] > 1
    assert result["missing_backward"] == ()
    assert result["final_loss"] < 0.15 * result["initial_loss"]
