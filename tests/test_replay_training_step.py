import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.autograd import autograd, GradTape
from src.common.tensors.abstract_nn.losses import MSELoss
from src.common.tensors.abstract_nn.optimizer import Adam
from src.common.tensors.abstract_convolution import demo_ndpca3conv3d_process_diagram as demo


def test_replay_training_step_matches_original():
    np.random.seed(0)
    img_np = np.random.rand(
        demo.BATCH_SIZE, demo.IN_CHANNELS, demo.IMG_D, demo.IMG_H, demo.IMG_W
    ).astype(np.float32)
    img = AbstractTensor.get_tensor(img_np)

    metric_np = np.tile(
        np.eye(3, dtype=np.float32), (demo.IMG_D, demo.IMG_H, demo.IMG_W, 1, 1)
    )
    metric = AbstractTensor.get_tensor(metric_np)
    package = {"metric": {"g": metric, "inv_g": metric}}

    model = demo.DemoModel(like=img, grid_shape=(demo.IMG_D, demo.IMG_H, demo.IMG_W))
    model.package = package
    loss_fn = MSELoss()
    optimizer = Adam(model.parameters(), lr=demo.LEARNING_RATE)

    target_np = np.zeros((demo.BATCH_SIZE, demo.NUM_CLASSES), dtype=np.float32)
    target_np[0, 0] = 1.0
    target = AbstractTensor.get_tensor(target_np)

    initial_params = [p.clone() for p in model.parameters()]

    logits = model.forward(img)
    loss = loss_fn.forward(logits, target)
    grad_pred = loss_fn.backward(logits, target)
    model.backward(grad_pred)
    params = model.parameters()
    grads = model.grads()
    with autograd.no_grad():
        new_params = optimizer.step(params, grads)
        i = 0
        for layer in model.layers:
            layer_params = layer.parameters()
            for j in range(len(layer_params)):
                AbstractTensor.copyto(layer_params[j], new_params[i])
                i += 1
    model.zero_grad()

    diagnostics = {
        "logits": logits.clone(),
        "loss": loss.clone(),
        "grads": [g.clone() for g in grads],
        "updated_params": [p.clone() for p in model.parameters()],
        "rng_state": np.random.get_state(),
    }

    img_id, target_id = id(img), id(target)
    autograd.tape = GradTape()
    autograd.capture_all = True
    for tensor in [img, target, metric, *model.parameters()]:
        autograd.tape.create_tensor_node(tensor)
    logits = model.forward(img)
    loss = loss_fn.forward(logits, target)
    autograd.capture_all = False
    autograd.tape.mark_loss(loss)
    proc = demo.AutogradProcess(autograd.tape)
    proc.build(loss)
    loss_id = proc.tape._loss_id
    autograd.tape = GradTape()

    model_replay = demo.DemoModel(like=img, grid_shape=(demo.IMG_D, demo.IMG_H, demo.IMG_W))
    model_replay.package = package
    for p_new, p_old in zip(model_replay.parameters(), initial_params):
        AbstractTensor.copyto(p_new, p_old)

    demo.replay_training_step(
        proc,
        loss_id,
        img_id,
        target_id,
        img,
        target,
        model_replay.parameters(),
        diagnostics,
    )
    for p_replay, p_expected in zip(
        model_replay.parameters(), diagnostics["updated_params"]
    ):
        assert np.allclose(p_replay.data, p_expected.data, atol=1e-5)

