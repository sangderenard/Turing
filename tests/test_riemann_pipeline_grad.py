import pytest
from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_convolution.ndpca3transform import fit_metric_pca
from src.common.tensors.riemann.geometry_factory import build_geometry
from src.common.tensors.abstract_convolution.metric_steered_conv3d import MetricSteeredConv3DWrapper
from src.common.tensors.abstract_nn.linear_block import LinearBlock
from src.common.tensors.abstract_nn.core import Model
from src.common.tensors import autograd as _autograd

TWOPI = 6.283185307179586


def _build_demo_like_layer():
    AT = AbstractTensor
    Nu = Nv = Nw = 2
    B_samples = 20
    n = 8
    t = AT.arange(0, B_samples, 1, requires_grad=True)
    t = (t / (B_samples - 1) - 0.5) * TWOPI
    base = AT.stack(
        [
            t.sin(),
            t.cos(),
            (2 * t).sin(),
            (0.5 * t).cos(),
            (0.3 * t).sin(),
            (1.7 * t).cos(),
            (0.9 * t).sin(),
            (1.3 * t).cos(),
        ],
        dim=-1,
    )
    scale = AT.get_tensor([2.0, 1.5, 1.2, 0.8, 0.5, 0.3, 0.2, 0.1], requires_grad=True)
    u_samples = base * scale
    weights = (-(t ** 2)).exp()
    M = AT.eye(n)
    diag = AT.get_tensor([1.0, 0.5, 0.25, 2.0, 1.0, 3.0, 0.8, 1.2], requires_grad=True)
    M = M * diag.reshape(1, -1)
    M = M.swapaxes(-1, -2) * diag.reshape(1, -1)
    basis = fit_metric_pca(u_samples, weights=weights, metric_M=M)

    def phi_fn(U, V, W):
        feats = [U, V, W, (U * V), (V * W), (W * U), U.sin(), V.cos()]
        return AT.stack(feats, dim=-1)

    geom_cfg = {
        "key": "pca_nd",
        "grid_shape": (Nu, Nv, Nw),
        "boundary_conditions": (True,) * 6,
        "transform_args": {"pca_basis": basis, "phi_fn": phi_fn, "d_visible": 3},
        "laplace_kwargs": {},
    }
    transform, grid, _ = build_geometry(geom_cfg)
    train_cfg = {
        "B": 2,
        "C": 3,
        "boundary_conditions": ("dirichlet",) * 6,
        "k": 3,
        "eig_from": "g",
        "pointwise": True,
    }
    layer = MetricSteeredConv3DWrapper(
        train_cfg["C"],
        train_cfg["C"],
        geom_cfg["grid_shape"],
        transform,
        boundary_conditions=train_cfg["boundary_conditions"],
        k=train_cfg["k"],
        eig_from=train_cfg["eig_from"],
        pointwise=train_cfg["pointwise"],
        deploy_mode="modulated",
        laplace_kwargs={"lambda_reg": 0.5},
    )
    return layer, grid, train_cfg


def _forward_and_back(layer, grid, train_cfg):
    AT = AbstractTensor
    x = AT.randn((train_cfg["B"], train_cfg["C"], *grid.U.shape), requires_grad=True)
    target = (grid.U + grid.V + grid.W).unsqueeze(0).unsqueeze(0).expand(train_cfg["B"], train_cfg["C"], -1, -1, -1)
    out = layer.forward(x)
    reg = layer.laplace_package["regularization_loss"]
    data_loss = ((out - target) ** 2).mean()
    total = data_loss + reg
    total.backward()
    return layer.laplace_package["local_state_network"], AT


def test_riemann_pipeline_without_lsn_backward_has_no_grads():
    layer, grid, train_cfg = _build_demo_like_layer()
    lsn, AT = _forward_and_back(layer, grid, train_cfg)
    params = lsn.parameters(include_all=True, include_structural=True)
    assert all(getattr(p, "_grad", None) is None for p in params)


def test_riemann_pipeline_with_lsn_backward_updates_params():
    layer, grid, train_cfg = _build_demo_like_layer()
    lsn, AT = _forward_and_back(layer, grid, train_cfg)
    grad_w = getattr(lsn._weighted_padded, "_grad", AT.zeros_like(lsn._weighted_padded))
    grad_m = getattr(lsn._modulated_padded, "_grad", AT.zeros_like(lsn._modulated_padded))
    lsn.backward(grad_w, grad_m, lambda_reg=0.5)
    params = lsn.parameters(include_all=True, include_structural=True)
    assert all(getattr(p, "_grad", None) is not None for p in params)


def test_riemann_pipeline_linear_block_params_receive_grads():
    layer, grid, train_cfg = _build_demo_like_layer()
    AT = AbstractTensor
    B, C = train_cfg["B"], train_cfg["C"]
    end_linear = LinearBlock(C, C, AT.zeros((1,)))
    model = Model([layer, end_linear], [None, None])
    for p in end_linear.parameters():
        if hasattr(p, "zero_grad"):
            p.zero_grad()
        _autograd.autograd.tape.create_tensor_node(p)
    x = AT.randn((B, C, *grid.U.shape), requires_grad=True)
    y = model.forward(x)
    _autograd.autograd.tape.auto_annotate_eval(y)
    assert y.shape == (B, C, *grid.U.shape)
    target = AT.randn(y.shape)
    pred = y.reshape(B, -1)
    target_flat = target.reshape(B, -1)
    loss = ((pred - target_flat) ** 2).mean()
    loss.backward()
    for layer in end_linear.model.layers:
        assert getattr(layer, "gW", None) is not None and layer.gW.abs().sum().item() != 0
        if layer.b is not None:
            assert getattr(layer, "gb", None) is not None and layer.gb.abs().sum().item() != 0
