import numpy as np

from src.common.tensors.accelerator_backends.mandelbrot_detail_network import (
    detail_scores,
    train_detail_controller,
)


def test_detail_score_rejects_uniform_fields():
    bland = np.zeros((1, 12, 16), dtype=np.float32)
    structured = np.indices((12, 16)).sum(axis=0)[None] % 8
    scores = detail_scores(
        np.concatenate((bland, structured.astype(np.float32))), iterations=8
    )
    assert scores[1] > scores[0] + 0.25


def test_tiny_abstract_nn_controller_trains_and_predicts():
    phase = np.linspace(0.0, 2.0 * np.pi, 15, endpoint=False)
    features = np.column_stack((
        np.sin(phase),
        np.cos(phase),
        np.sin(2.0 * phase),
    ))
    scores = 0.45 + 0.35 * np.sin(phase)
    controller = train_detail_controller(
        features, scores, hidden=6, epochs=4, learning_rate=0.03
    )
    prediction = controller.predict(features)
    assert controller.final_loss < controller.initial_loss
    assert prediction.shape == scores.shape
    assert np.isfinite(prediction).all()
