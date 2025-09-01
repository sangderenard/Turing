"""
riemann_convolutional_demo.py
----------------------------

Demo: Trains a RiemannConvolutional3D layer on a synthetic regression task, ensuring that:
- The LocalStateNetwork parameters (in the Laplacian) are updated during training.
- The metric at each location factors into the convolution kernel.
- The geometry is driven by the transform + Laplacian pipeline.
- Training proceeds until loss < 1e-6 (or max epochs).

This demo audits the full geometry-driven learning pipeline.
"""


from .metric_steered_conv3d import MetricSteeredConv3DWrapper
from .ndpca3transform import PCABasisND, fit_metric_pca
from ..abstraction import AbstractTensor
from ..autograd import autograd
from ..riemann.geometry_factory import build_geometry
# from src.common.tensors.abstract_nn.optimizer import BPIDSGD  # TODO: optional later
from src.common.tensors.abstract_nn.optimizer import Adam, SGD
from src.common.tensors.abstract_nn.linear_block import LinearBlock
from src.common.tensors.abstract_nn.pca_layer import PCATransformLayer
import numpy as np
from pathlib import Path
from skimage import measure
from .laplace_nd import BuildLaplace3D
import os
import math
import time
from PIL import Image
import threading
from queue import Queue, Empty
from learning_tasks import get_task
from .render_cache import FrameCache, apply_colormap


def _normalize(arr):
    arr = np.array(arr)
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def normalize_for_visualization(arr):
    """Prepare an array for image/volume visualisation.

    Parameters
    ----------
    arr : array-like
        Input tensor.  The output is guaranteed to be either 2-D (heatmap) or
        3-D (volume) data suitable for visualisation.
    """

    arr = _to_numpy(arr)

    # Collapse leading batch/channel dimensions for 4D/5D (or higher) inputs
    if arr.ndim in (4, 5):
        arr = arr.mean(axis=(0, 1))
    elif arr.ndim > 5:
        leading = tuple(range(arr.ndim - 3))
        arr = arr.mean(axis=leading)

    # Flatten 1-D vectors to the nearest square for heatmap display
    if arr.ndim == 1:
        n = arr.size
        side = int(np.ceil(np.sqrt(n)))
        padded = np.zeros(side * side, dtype=arr.dtype)
        padded[:n] = arr
        arr = padded.reshape(side, side)

    # Ensure we only ever return 2-D or 3-D data
    if arr.ndim > 3:
        arr = arr.reshape(arr.shape[-3:])

    return arr


def pca_to_rgb(arr):
    """Project ``arr`` to two principal components and return an RGB image."""

    arr = normalize_for_visualization(arr)
    arr = np.array(arr)
    if arr.ndim == 3:
        h, w, d = arr.shape
        flat = arr.reshape(-1, d)
        flat = flat - flat.mean(axis=0, keepdims=True)
        cov = np.cov(flat, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        comps = flat @ eigvecs[:, -2:]
        comps = comps.reshape(h, w, 2)
        r = _normalize(comps[..., 0])
        g = _normalize(comps[..., 1])
        b = np.zeros_like(r)
        img = np.stack([r, g, b], axis=-1)
    else:
        arr2d = _normalize(arr)
        img = np.stack([arr2d] * 3, axis=-1)
    return (img * 255).astype(np.uint8)



def _tensor_to_pil_image(t):
    arr = np.array(t.data if hasattr(t, "data") else t)
    arr = _normalize(arr) * 255
    arr = arr.clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def _to_numpy(t):
    return np.array(t.data if hasattr(t, "data") else t)


def _pca_reduce(coords, k=3):
    """Reduce coordinates to ``k`` dimensions using PCA."""
    coords = coords - coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:k]
    return coords @ eigvecs[:, order]


def render_nd_field(
    field,
    return_heatmap: bool = False,
    return_figures: bool = False,
    dim_reducer=_pca_reduce,
    overlay_hooks=None,
):
    """Render an ``n``-D tensor as a visual representation.

    Imports ``matplotlib`` lazily to avoid conflicting with Tkinter.

    Parameters
    ----------
    field: array-like
        The tensor or array to visualise. It must have at least two
        dimensions.
    return_heatmap: bool, optional
        If ``True`` and ``field`` has three or more dimensions, a 2-D
        heat-map view (mean across the first axis) is also returned.
    return_figures: bool, optional
        When ``True`` the returned values are ``matplotlib`` figures.
        Otherwise image buffers (``numpy`` arrays) compatible with the
        existing animation pipeline are returned.
    dim_reducer: callable or ``None``, optional
        Function used to reduce coordinates with ``ndim > 3`` down to
        three dimensions. If ``None`` and the field has more than three
        dimensions, a ``ValueError`` is raised. The default uses a PCA
        projection.
    overlay_hooks: iterable of callables, optional
        Each hook is invoked as ``hook(ax, coords, values, reduced)``
        allowing future feature overlays (bounding boxes, clustering,
        etc.). ``coords`` are the original ``n``-D coordinates; ``reduced``
        are the 3-D coordinates used for plotting.

    Returns
    -------
    dict
        Keys include ``"scatter3d"`` for 3-D scatter images and
        optionally ``"heatmap"``. The values are either figures or
        ``np.ndarray`` buffers depending on ``return_figures``.
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    arr = np.array(field)
    if arr.ndim < 2:
        raise ValueError("Field must be at least 2-D")

    overlay_hooks = overlay_hooks or []
    figures = {}
    coords = np.indices(arr.shape).reshape(arr.ndim, -1).T
    values = arr.reshape(-1)
    if arr.ndim == 2:
        fig, ax = plt.subplots()
        hm = ax.imshow(arr, cmap="viridis")
        ax.axis("off")
        fig.colorbar(hm, ax=ax)
        for hook in overlay_hooks:
            hook(ax, coords, values, coords)
        figures["heatmap"] = fig
    else:
        reduced = coords
        if arr.ndim > 3:
            if dim_reducer is None:
                raise ValueError("dim_reducer must be provided for ndim>3")
            reduced = dim_reducer(coords, 3)
        x, y, z = reduced.T[:3]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(x, y, z, c=values, cmap="viridis")
        fig.colorbar(sc, ax=ax)
        ax.set_title(f"{arr.ndim}D Field")
        for hook in overlay_hooks:
            hook(ax, coords, values, reduced)
        figures["scatter3d"] = fig
        if return_heatmap:
            heatmap = arr.mean(axis=0)
            hfig, hax = plt.subplots()
            hm = hax.imshow(heatmap, cmap="viridis")
            hax.axis("off")
            hfig.colorbar(hm, ax=hax)
            for hook in overlay_hooks:
                hook(hax, coords, values, reduced)
            figures["heatmap"] = hfig

    if return_figures:
        return figures

    images = {}
    for key, fig in figures.items():
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images[key] = buf
        plt.close(fig)
    return images


def render_laplace_surface(laplace_tensor, grid_shape, threshold=0.0, output_path=None, show=False):
    """Render an isosurface of the Laplace tensor's diagonal.

    Parameters
    ----------
    laplace_tensor: array-like
        Dense Laplace operator matrix.
    grid_shape: tuple[int, int, int]
        Spatial grid dimensions used to reshape the diagonal.
    threshold: float, optional
        Iso-value used when extracting the surface via marching cubes.
    output_path: str or Path, optional
        If provided, the figure is saved to this path.
    show: bool, optional
        When True, display the figure instead of closing it.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    arr = np.array(laplace_tensor.data if hasattr(laplace_tensor, "data") else laplace_tensor)
    diag_field = np.diag(arr).reshape(grid_shape)
    level = float(threshold)
    # Ensure level lies within the data range
    min_val, max_val = diag_field.min(), diag_field.max()
    if level < min_val or level > max_val:
        level = float(diag_field.mean())
    verts, faces, _, _ = measure.marching_cubes(diag_field, level=level)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, cmap="viridis", lw=0.5)
    ax.set_title(f"Laplace Surface (level={level:.2f})")
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def build_config():
    AT = AbstractTensor
    Nu = Nv = Nw = 8
    n = 8
    B = 500
    t = AT.arange(0, B, 1, requires_grad=True)
    autograd.tape.annotate(t, label="riemann_demo.t_arange")
    autograd.tape.auto_annotate_eval(t)
    t = (t / (B - 1) - 0.5) * 6.283185307179586
    autograd.tape.annotate(t, label="riemann_demo.t_scaled")
    autograd.tape.auto_annotate_eval(t)
    base = AT.stack([
        t.sin(), t.cos(), (2 * t).sin(), (0.5 * t).cos(),
        (0.3 * t).sin(), (1.7 * t).cos(), (0.9 * t).sin(), (1.3 * t).cos()
    ], dim=-1)
    autograd.tape.annotate(base, label="riemann_demo.base")
    autograd.tape.auto_annotate_eval(base)
    scale = AT.get_tensor([2.0, 1.5, 1.2, 0.8, 0.5, 0.3, 0.2, 0.1], requires_grad=True)
    autograd.tape.annotate(scale, label="riemann_demo.scale")
    autograd.tape.auto_annotate_eval(scale)
    u_samples = base * scale
    autograd.tape.annotate(u_samples, label="riemann_demo.u_samples")
    autograd.tape.auto_annotate_eval(u_samples)
    weights = (-(t**2)).exp()
    autograd.tape.annotate(weights, label="riemann_demo.weights")
    autograd.tape.auto_annotate_eval(weights)
    M = AT.eye(n)
    autograd.tape.annotate(M, label="riemann_demo.M_eye")
    autograd.tape.auto_annotate_eval(M)
    diag = AT.get_tensor([1.0, 0.5, 0.25, 2.0, 1.0, 3.0, 0.8, 1.2], requires_grad=True)
    autograd.tape.annotate(diag, label="riemann_demo.diag")
    autograd.tape.auto_annotate_eval(diag)
    M = M * diag.reshape(1, -1)
    M = M.swapaxes(-1, -2) * diag.reshape(1, -1)
    autograd.tape.annotate(M, label="riemann_demo.metric_M")
    autograd.tape.auto_annotate_eval(M)
    basis = fit_metric_pca(u_samples, weights=weights, metric_M=M)
    autograd.tape.annotate(basis, label="riemann_demo.basis")
    autograd.tape.auto_annotate_eval(basis)

    def phi_fn(U, V, W):
        feats = [U, V, W, (U * V), (V * W), (W * U), (U.sin()), (V.cos())]
        return AT.stack(feats, dim=-1)

    config = {
        "geometry": {
            "key": "pca_nd",
            "grid_shape": (Nu, Nv, Nw),
            "boundary_conditions": (True,) * 6,
            "transform_args": {"pca_basis": basis, "phi_fn": phi_fn, "d_visible": 3},
            "laplace_kwargs": {},
        },
        "training": {
            "B": 4,
            "C": 1,
            "task": "pixel_art_reconstruct",
            "boundary_conditions": ("dirichlet",) * 6,
            "k": 3,
            "eig_from": "g",
            "pointwise": True,
        },
        "visualization": {
            "render_laplace_surface": False,
            "laplace_threshold": 0.0,
            "laplace_path": "laplace_surface.png",
        },
    }
    return config


def training_worker(
    frame_cache: FrameCache,
    shared_state: dict,
    config=None,
    viz_every=1,
    max_epochs=25,
    visualize_laplace=None,
    laplace_threshold=None,
    laplace_path=None,
    show_laplace=False,
    dpi=200,
    deep_research=False,
    stop_event: threading.Event | None = None,
):
    AT = AbstractTensor
    if config is None:
        config = build_config()
    geom_cfg = config["geometry"]
    train_cfg = config["training"]
    viz_cfg = config.get("visualization", {})
    if visualize_laplace is None:
        visualize_laplace = viz_cfg.get("render_laplace_surface", False)
    if laplace_threshold is None:
        laplace_threshold = viz_cfg.get("laplace_threshold", 0.0)
    if laplace_path is None:
        laplace_path = viz_cfg.get("laplace_path", "laplace_surface.png")
    laplace_path = Path(laplace_path)
    B, C = train_cfg["B"], train_cfg["C"]
    task_name = train_cfg.get("task", "pixel_art_reconstruct")
    task = get_task(task_name)
    num_logits = train_cfg.get("num_logits", task.num_logits)

    # -------------------------------
    # Data queue (start producer)
    # -------------------------------
    data_queue: Queue = Queue(maxsize=10)
    pump_thread = threading.Thread(
        target=task.pump_queue,
        args=(data_queue, geom_cfg.get("grid_shape", (1, 1, 1)), C),
        kwargs={"stop_event": stop_event},
        daemon=True,
    )
    pump_thread.start()

    # Grab one sample to infer grid shape and input dimensionality
    sample_inp, sample_tgt, sample_cat = data_queue.get()
    sample_ndim = np.array(sample_inp).ndim
    if sample_ndim == 2:
        inferred_grid_shape = (1, sample_inp.shape[0], sample_inp.shape[1])
    elif sample_ndim == 3:
        inferred_grid_shape = tuple(sample_inp.shape)
    elif sample_ndim >= 5:
        inferred_grid_shape = tuple(np.array(sample_inp).shape[-3:])
    else:
        inferred_grid_shape = geom_cfg.get("grid_shape", (1, 1, 1))
    geom_cfg["grid_shape"] = inferred_grid_shape
    grid_shape = inferred_grid_shape

    # Reinsert the sample so training sees it later
    data_queue.put((sample_inp, sample_tgt, sample_cat))

    # Build geometry only after determining grid shape
    transform, grid, _ = build_geometry(geom_cfg)

    loss_composer = task.build_loss_composer(C, num_logits)
    if visualize_laplace:
        with autograd.no_grad():
            builder = BuildLaplace3D(
                grid_domain=grid,
                metric_tensor_func=transform.metric_tensor_func,
                boundary_conditions=train_cfg.get("boundary_conditions", ("dirichlet",) * 6),
                **geom_cfg.get("laplace_kwargs", {}),
            )
            laplace_tensor, _, _ = builder.build_general_laplace(
                grid.U, grid.V, grid.W, dense=True, return_package=False
            )
        render_laplace_surface(
            laplace_tensor,
            grid_shape,
            threshold=laplace_threshold,
            output_path=laplace_path,
            show=show_laplace,
        )
    from ..abstraction import AbstractTensor as _AT
    U, V, W = grid.U, grid.V, grid.W
    autograd.tape.annotate(U, label="riemann_demo.grid_U")
    autograd.tape.auto_annotate_eval(U)
    autograd.tape.annotate(V, label="riemann_demo.grid_V")
    autograd.tape.auto_annotate_eval(V)
    autograd.tape.annotate(W, label="riemann_demo.grid_W")
    autograd.tape.auto_annotate_eval(W)
    # initial annotations for target and inputs are deferred to queue samples
    # When AUTOGRAD_STRICT=1, unused tensors trigger connectivity errors.
    # Uncomment one of the lines below to relax those checks:
    # autograd.strict = False                   # disable strict mode globally
    # autograd.whitelist(x, target)             # or whitelist specific tensors
    # autograd.whitelist_labels(r"riemann_demo.*")  # whitelist by label pattern
    # --- Parameter and gradient collection helpers ---
    from ..logger import get_tensors_logger
    logger = get_tensors_logger()
    # ------------------------------------------------------------
    # PRE-EPOCH MODEL CONSTRUCTION (instantiate every layer up front)
    #   1) choose a transform layer (replaces former "input linear shim")
    #   2) metric-steered conv wrapper
    #   3) end LinearBlock (replaces former "output shim")
    # ------------------------------------------------------------

    # 1) Convolutional transform layer completely replaces the start linear shim
    #    We keep the same selection logic you already used, but as a proper layer in the model.
    transform_layer = None
    if sample_ndim == 2:
        from src.common.tensors.abstract_nn.transform_layers import Transform2DLayer
        transform_layer = Transform2DLayer()
    elif sample_ndim == 3 or sample_ndim == 5:
        # Treat 3D fields (and already 5D BCHWD layouts) via 3D transform
        from src.common.tensors.abstract_nn.transform_layers import Transform3DLayer
        transform_layer = Transform3DLayer()
    else:
        # Fallback: dimensionality reduction to a 5D-compatible representation
        transform_layer = PCATransformLayer()

    # 2) Metric-steered conv (kept identical), BUT do not depend on runtime shims
    conv_layer = MetricSteeredConv3DWrapper(
        in_channels=C,
        out_channels=C + num_logits,  # conv produces feature + logits channels
        grid_shape=grid_shape,
        transform=transform,
        boundary_conditions=train_cfg.get("boundary_conditions", ("dirichlet",) * 6),
        k=train_cfg.get("k", 3),
        eig_from=train_cfg.get("eig_from", "g"),
        pointwise=train_cfg.get("pointwise", True),
        deploy_mode="modulated",
        laplace_kwargs={"lambda_reg": 0.5},
    )

    # Build a provisional system to probe conv output shape after transform
    from src.common.tensors.abstract_nn.core import Model
    probe_layers = [transform_layer, conv_layer]
    probe_activations = [None] * len(probe_layers)
    probe_system = Model(probe_layers, probe_activations)
    x0 = AbstractTensor.get_tensor(np.expand_dims(np.array(sample_inp), 0), requires_grad=True)
    y0 = probe_system.forward(x0)

    # Decide LinearBlock I/O sizes.
    # End shim is now a real LinearBlock; operate along feature axis without calling it directly.
    # We keep channel count stable by default; adjust here if your loss expects a different mapping.
    # If y0 is BCHWD, take channels as in/out for channelwise linearization.
    
    def compute_F(tensor):
        spatial_dims = tensor.shape
        F = 1
        for d in spatial_dims:
            F *= d
        return F
    flat_target_size = compute_F(sample_tgt)
    print(f"[DEBUG] sample_tgt.shape={sample_tgt.shape}, flat_target_size={flat_target_size}")
    print(f"[DEBUG] y0.shape={y0.shape}, y0_flat_size={compute_F(y0)}")
    
    if hasattr(y0, "shape") and len(y0.shape) >= 2:
        out_channels_after_conv = int(y0.shape[1])
        end_linear = LinearBlock(out_channels_after_conv, flat_target_size, AbstractTensor.zeros((1,)))


    # Final training system: [transform -> conv -> linear]
    layer_list = [transform_layer, conv_layer, end_linear]
    activations = [None] * len(layer_list)
    system = Model(layer_list, activations)

    # Target shaping system (ONLY for targets; respects "only system.forward")
    target_layers = [transform_layer.__class__()] if transform_layer is not None else []
    target_system = Model(target_layers, [None]*len(target_layers)) if target_layers else None

    def collect_params_and_grads():
        params, grads = [], []
        seen_params = set()
        seen_objs = set()

        def add_param(p):
            pid = id(p)
            if pid in seen_params:
                return
            seen_params.add(pid)
            params.append(p)
            grads.append(getattr(p, '_grad', None))

        # Transform layer params (if any)
        if transform_layer is not None and hasattr(transform_layer, 'parameters'):
            seen_objs.add(id(transform_layer))
            for p in transform_layer.parameters():
                add_param(p)

        # Convolutional weights
        if hasattr(conv_layer, 'conv') and hasattr(conv_layer.conv, 'parameters'):
            seen_objs.add(id(conv_layer.conv))
            for p in conv_layer.conv.parameters():
                add_param(p)

        # LocalStateNetwork (if present)
        lsn = conv_layer.local_state_network if hasattr(conv_layer, 'local_state_network') else None
        if lsn is None:
            raise ValueError("LocalStateNetwork not found")
        if lsn and hasattr(lsn, 'parameters'):
            seen_objs.add(id(lsn))
            for p in lsn.parameters(include_all=True):
                add_param(p)
        else:
            raise ValueError("LocalStateNetwork not found")

        # Fallback: any other objects with .parameters
        if isinstance(conv_layer.laplace_package, dict):
            for v in conv_layer.laplace_package.values():
                vid = id(v)
                if vid in seen_objs or vid in seen_params:
                    continue
                if hasattr(v, 'parameters'):
                    seen_objs.add(vid)
                    for p in v.parameters():
                        add_param(p)
        # End LinearBlock (always present now)
        if end_linear is not None and hasattr(end_linear, 'parameters'):
            for p in end_linear.parameters():
                add_param(p)
        # Log all params and grads, including Nones
        for i, (p, g) in enumerate(zip(params, grads)):
            label = getattr(p, '_label', None)
            logger.info(
                f"Param {i}: label={label}, shape={getattr(p, 'shape', None)}, grad is None={g is None}, grad shape={getattr(g, 'shape', None) if g is not None else None}"
            )
        return params, grads
    # get the expected input shape for the architecture
    # make an input of w/e to kick start the parameter recognition
    # first, to accomodate any network, you must poll the model
    # for the expected input shape

    # Quick connectivity sanity: probe once before training
    expected_shape = system.get_input_shape()
    concrete = tuple(1 if d is None else d for d in expected_shape)
    _probe_x = AbstractTensor.ones(concrete)
    _ = system.forward(_probe_x)
    grad_enabled = getattr(_AT.autograd, '_no_grad_depth', 0) == 0
    print(f"[DEBUG] LSN instance id after forward: {id(conv_layer.local_state_network)} | grad_tracking_enabled={grad_enabled}")
    print(f"[DEBUG] LSN param ids: {[id(p) for p in conv_layer.local_state_network.parameters(include_all=True)]}")
    print(f"[DEBUG] LSN param requires_grad: {[getattr(p, 'requires_grad', None) for p in conv_layer.local_state_network.parameters(include_all=True)]}")
    print(f"[DEBUG] LSN _regularization_loss: {conv_layer.local_state_network._regularization_loss}")
    print(f"[DEBUG] LSN _regularization_loss grad_fn: {getattr(conv_layer.local_state_network._regularization_loss, 'grad_fn', None)}")
    grad_enabled = getattr(_AT.autograd, '_no_grad_depth', 0) == 0
    print(f"[DEBUG] About to call backward on LSN _regularization_loss | grad_tracking_enabled={grad_enabled}")
    lsn = conv_layer.local_state_network
    lsn._regularization_loss.backward()
    grad_w = getattr(lsn._weighted_padded, '_grad', AbstractTensor.zeros_like(lsn._weighted_padded))
    grad_m = getattr(lsn._modulated_padded, '_grad', AbstractTensor.zeros_like(lsn._modulated_padded))
    lsn.backward(grad_w, grad_m, lambda_reg=0.5)
    for i, p in enumerate(lsn.parameters(include_all=True)):
        grad_enabled = getattr(_AT.autograd, '_no_grad_depth', 0) == 0
        print(
            f"[DEBUG] After backward: param {i} id={id(p)} grad={getattr(p, '_grad', None)} | grad_tracking_enabled={grad_enabled}"
        )

    params, _ = collect_params_and_grads()

    def init_optimizer(name: str, params, lr: float):
        name_l = name.lower()
        if name_l == "sgd":
            return SGD(params, lr=lr)
        # if name_l == "bpidsgd":
        #     return BPIDSGD(params, lr=lr)  # disabled: produces NaNs
        return Adam(params, lr=lr)

    opt_name = shared_state.get("optimizer", "Adam")
    lr = shared_state.get("lr", 5e-2)
    optimizer = init_optimizer(opt_name, params, lr)
    current_opt, current_lr = opt_name, lr

    for epoch in range(1, max_epochs + 1):
        if stop_event is not None and stop_event.is_set():
            break
        while shared_state.get("epoch_limit", max_epochs) < epoch:
            if stop_event is not None and stop_event.is_set():
                break
            time.sleep(0.1)
        if stop_event is not None and stop_event.is_set():
            break
        params, _ = collect_params_and_grads()
        # Zero gradients for all params
        for p in params:
            if hasattr(p, 'zero_grad'):
                p.zero_grad()
            elif hasattr(p, '_grad'):
                p._grad = AbstractTensor.zeros_like(p._grad)
        batch_inputs, batch_targets, batch_cats = [], [], []
        while len(batch_inputs) < B and (stop_event is None or not stop_event.is_set()):
            try:
                inp, tgt, cat = data_queue.get(timeout=0.1)
                batch_inputs.append(inp)
                batch_targets.append(tgt)
                batch_cats.append(cat)
            except Empty:
                if not batch_inputs:
                    continue
                break
        if not batch_inputs:
            break
        batch_arr = np.stack(batch_inputs)
        target_arr = np.stack(batch_targets)

        # Wrap inputs/targets; ONLY system.forward from here on
        x = AT.get_tensor(batch_arr, requires_grad=True)
        if target_system is not None:
            # Shape targets using the same family of transform, but via its own system
            target = target_system.forward(AT.get_tensor(target_arr))
        else:
            target = AT.get_tensor(target_arr)
        autograd.tape.annotate(x, label=f"riemann_demo.input_epoch_{epoch}")
        autograd.tape.annotate(target, label=f"riemann_demo.target_epoch_{epoch}")
        autograd.tape.auto_annotate_eval(x)
        autograd.tape.auto_annotate_eval(target)
        y = system.forward(x)
        autograd.tape.auto_annotate_eval(y)
        if deep_research:
            print("[DEEP-RESEARCH] input data:", _to_numpy(x))
            print("[DEEP-RESEARCH] predicted data:", _to_numpy(y))
        pred = y[:, :C]
        loss = loss_composer(y, target, batch_cats)
        LSN_loss = conv_layer.local_state_network._regularization_loss
        print(f"Epoch {epoch}: loss={loss.item()}, LSN_loss={LSN_loss.item()}")
        loss = LSN_loss + loss
        print(f"Total loss={loss.item()}")
        autograd.tape.annotate(loss, label="riemann_demo.loss")
        autograd.tape.auto_annotate_eval(loss)
        loss.backward()
        lsn = conv_layer.local_state_network
        grad_w = getattr(lsn._weighted_padded, '_grad', AbstractTensor.zeros_like(lsn._weighted_padded))
        grad_m = getattr(lsn._modulated_padded, '_grad', AbstractTensor.zeros_like(lsn._modulated_padded))
        lsn.backward(grad_w, grad_m, lambda_reg=0.5)
        params, grads = collect_params_and_grads()
        new_opt = shared_state.get("optimizer", current_opt)
        new_lr = shared_state.get("lr", current_lr)
        if (
            shared_state.get("reset_opt")
            or new_opt != current_opt
            or new_lr != current_lr
        ):
            optimizer = init_optimizer(new_opt, params, new_lr)
            current_opt, current_lr = new_opt, new_lr
            shared_state["reset_opt"] = False
        if deep_research:
            for i, p in enumerate(params):
                print(f"[DEEP-RESEARCH] param {i}:", _to_numpy(p))
            for i, (g, p) in enumerate(zip(grads, params)):
                g = g if g is not None else AT.zeros_like(p)
                print(f"[DEEP-RESEARCH] grad {i}:", _to_numpy(g))
            stacked_params = np.concatenate([_to_numpy(p).reshape(-1) for p in params])
            print("[DEEP-RESEARCH] all params stacked vertically:", stacked_params)
            stacked_grads = np.concatenate([
                _to_numpy(g if g is not None else AT.zeros_like(p)).reshape(-1)
                for g, p in zip(grads, params)
            ])
            print("[DEEP-RESEARCH] all grads stacked vertically:", stacked_grads)
        for p in params:
            label = getattr(p, '_label', None)
            # print(p)
            # print(p._grad)
            assert hasattr(p, '_grad'), f"Parameter {label or p} has no grad attribute"
            
            #assert p._grad is not None, f"Parameter {label or p} grad is None after backward()"
            #assert p._grad.shape == p.shape, f"Parameter {label or p} has incorrect grad shape: grad.shape={getattr(p._grad, 'shape', None)}, param.shape={getattr(p, 'shape', None)}"
        for i, (g, p) in enumerate(zip(grads, params)):
            if g is None:
                g = AbstractTensor.zeros_like(p)
                grads[i] = g
            label = getattr(p, '_label', None)
            #assert hasattr(g, 'shape'), f"Gradient for {label or p} has no shape attribute"
            #assert g.shape == p.shape, f"Gradient for {label or p} has incorrect shape: grad.shape={getattr(g, 'shape', None)}, param.shape={getattr(p, 'shape', None)}"
        # Optimizer returns updated tensors; copy values in-place to preserve
        # parameter identity on the tape so they remain registered.
        new_params = optimizer.step(params, grads)
        from ..abstraction import AbstractTensor as _AT
        for p, new_p in zip(params, new_params):
            _AT.copyto(p, new_p)
        if epoch % 1 == 0 or loss.item() < 1e-6:
            print(f"Epoch {epoch}: loss={loss.item():.2e}")
            # Gradient report
            for i, (p, g) in enumerate(zip(params, grads)):
                label = getattr(p, '_label', f'param_{i}')
                if g is not None:
                    try:
                        g_np = g.data if hasattr(g, 'data') else g
                        g_mean = g_np.mean() if hasattr(g_np, 'mean') else 'n/a'
                        g_norm = (g_np ** 2).sum() ** 0.5 if hasattr(g_np, '__pow__') else 'n/a'
                        print(f"  Grad {i} ({label}): mean={g_mean:.2e}, norm={g_norm:.2e}")
                    except Exception as e:
                        print(f"  Grad {i} ({label}): error reporting grad: {e}")
                else:
                    print(f"  Grad {i} ({label}): None")
        if epoch % viz_every == 0:
            inp_np = np.array(x[0, 0].data if hasattr(x[0, 0], 'data') else x[0, 0])
            pred_np = np.array(pred[0, 0].data if hasattr(pred[0, 0], 'data') else pred[0, 0])
            tgt_np = np.array(target[0, 0].data if hasattr(target[0, 0], 'data') else target[0, 0])
            diff_np = pred_np - tgt_np

            quad1 = pca_to_rgb(inp_np)
            quad2 = pca_to_rgb(pred_np)
            quad3 = pca_to_rgb(tgt_np)
            quad4 = pca_to_rgb(diff_np)

            # Ensure all quads are the same size
            h, w, *_ = quad1.shape
            def resize(img):
                if img.shape[:2] != (h, w):
                    return np.array(
                        Image.fromarray(img).resize((w, h), resample=Image.NEAREST)
                    )
                return img
            quad2 = resize(quad2)
            quad3 = resize(quad3)
            quad4 = resize(quad4)

            # Compose into 2x2 grid for legacy animation
            top = np.concatenate([quad1, quad2], axis=1)
            bottom = np.concatenate([quad3, quad4], axis=1)
            ip_frame = np.concatenate([top, bottom], axis=0)
            # Store individual quadrants for flexible layout
            frame_cache.enqueue("low_input", quad1)
            frame_cache.enqueue("low_prediction", quad2)
            frame_cache.enqueue("high_input", quad3)
            frame_cache.enqueue("high_prediction", quad4)

            # Parameter/gradient visualization
            p_imgs: list[np.ndarray] = []
            g_imgs: list[np.ndarray] = []
            for p, g in zip(params, grads):
                g = g if g is not None else AT.zeros_like(p)
                p_img = normalize_for_visualization(p)
                g_img = normalize_for_visualization(g)
                if p_img.ndim == 3:
                    p_img = p_img.mean(axis=0)
                if g_img.ndim == 3:
                    g_img = g_img.mean(axis=0)
                p_imgs.append(np.atleast_2d((_normalize(p_img) * 255).astype(np.uint8)))
                g_imgs.append(np.atleast_2d((_normalize(g_img) * 255).astype(np.uint8)))
            if p_imgs or g_imgs:
                target_h = max(img.shape[0] for img in p_imgs + g_imgs)
                target_w = max(img.shape[1] for img in p_imgs + g_imgs)

                def pad(img: np.ndarray) -> np.ndarray:
                    h_, w_ = img.shape[:2]
                    pad_h = target_h - h_
                    pad_w = target_w - w_
                    if pad_h or pad_w:
                        return np.pad(img, ((0, pad_h), (0, pad_w)), mode="constant")
                    return img

                pairs = []
                for idx, (p_img, g_img) in enumerate(zip(p_imgs, g_imgs)):
                    p_img = pad(p_img)
                    g_img = pad(g_img)
                    frame_cache.enqueue(f"param{idx}_param", p_img)
                    frame_cache.enqueue(f"param{idx}_grad", g_img)
                    pairs.append(np.concatenate([p_img, g_img], axis=-1))
                params_grads_frame = np.concatenate(pairs, axis=0) if pairs else np.zeros((target_h, target_w * 2), dtype=np.uint8)
            else:
                params_grads_frame = np.zeros((h, w * 2), dtype=np.uint8)

            # Legacy composite frames for animation exports
            frame_cache.enqueue("input_prediction", ip_frame)
            frame_cache.enqueue("params_grads", params_grads_frame)
        if loss.item() < 1e-6:
            print("Converged.")
            break
    # Audit: check that metric and local state network are used
    print("Metric at center voxel:", conv_layer.laplace_package['metric']['g'][grid_shape[0]//2, grid_shape[1]//2, grid_shape[2]//2])
    if 'local_state_network' in conv_layer.laplace_package:
        print("LocalStateNetwork parameters:", list(conv_layer.laplace_package['local_state_network'].parameters()))
    if stop_event is not None:
        stop_event.set()
    if pump_thread.is_alive():
        pump_thread.join()


def display_worker(
    frame_cache: FrameCache,
    stop_event: threading.Event,
    shared_state: dict,
    update_ms: int = 100,
    max_epochs: int | None = None,
) -> None:
    """Simple Tkinter UI that streams cached frames to a live window.

    Parameters
    ----------
    frame_cache:
        Shared cache of frames produced by the training thread.
    stop_event:
        Event used to signal the GUI to exit.  This allows the training thread
        or window close event to request a clean shutdown.
    update_ms:
        Refresh period for the UI in milliseconds.
    """

    import tkinter as tk
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("Riemann Demo")

    def on_close() -> None:
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    controls = tk.Frame(root)
    controls.pack(side=tk.TOP, fill=tk.X)

    grid_frame = tk.Frame(controls)
    grid_frame.pack(side=tk.LEFT)

    # Optimizer selection
    opt_var = tk.StringVar(value=shared_state.get("optimizer", "Adam"))
    opt_menu = tk.OptionMenu(controls, opt_var, "Adam", "SGD")  # "BPIDSGD" disabled
    opt_menu.pack(side=tk.LEFT)

    def on_opt_change(*_):
        shared_state["optimizer"] = opt_var.get()
        shared_state["reset_opt"] = True

    opt_var.trace_add("write", on_opt_change)
    tk.Button(controls, text="Reset Optimizer", command=lambda: shared_state.__setitem__("reset_opt", True)).pack(side=tk.LEFT)

    # Learning rate logarithmic slider
    lr_var = tk.DoubleVar(value=math.log10(shared_state.get("lr", 5e-2)))
    lr_scale = tk.Scale(
        controls,
        from_=-5,
        to=0,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        variable=lr_var,
        label="log10 lr",
    )
    lr_scale.pack(side=tk.LEFT)
    lr_label = tk.Label(controls, text=f"lr={shared_state.get('lr', 5e-2):.1e}")
    lr_label.pack(side=tk.LEFT)

    def on_lr_change(*_):
        lr = 10 ** lr_var.get()
        shared_state["lr"] = lr
        shared_state["reset_opt"] = True
        lr_label.config(text=f"lr={lr:.1e}")

    lr_var.trace_add("write", on_lr_change)

    # Epoch limit slider
    cap_max = max_epochs if max_epochs is not None else 1
    epoch_cap_var = tk.IntVar(value=shared_state.get("epoch_limit", cap_max))
    epoch_scale = tk.Scale(
        controls,
        from_=1,
        to=cap_max,
        orient=tk.HORIZONTAL,
        variable=epoch_cap_var,
        label="Epoch Limit",
    )
    epoch_scale.pack(side=tk.LEFT)

    def on_epoch_change(*_):
        shared_state["epoch_limit"] = epoch_cap_var.get()

    epoch_cap_var.trace_add("write", on_epoch_change)

    grid_vars: list[list[tk.StringVar]] = []
    grid_menus: list[list[tk.OptionMenu]] = []

    def _options() -> list[str]:
        return frame_cache.available_options() or [""]

    def _make_cell(r: int, c: int) -> None:
        var = tk.StringVar()
        opts = _options()
        if opts:
            var.set(opts[0])
        menu = tk.OptionMenu(grid_frame, var, *opts)
        menu.grid(row=r, column=c)
        grid_vars[r].append(var)
        grid_menus[r].append(menu)

    def add_row() -> None:
        r = len(grid_vars)
        grid_vars.append([])
        grid_menus.append([])
        cols = len(grid_vars[0]) if grid_vars and grid_vars[0] else 0
        for c in range(cols):
            _make_cell(r, c)

    def add_column() -> None:
        c = len(grid_vars[0]) if grid_vars and grid_vars[0] else 0
        if not grid_vars:
            grid_vars.append([])
            grid_menus.append([])
        for r in range(len(grid_vars)):
            _make_cell(r, c)

    tk.Button(controls, text="Add Row", command=add_row).pack(side=tk.LEFT)
    tk.Button(controls, text="Add Column", command=add_column).pack(side=tk.LEFT)

    norm_var = tk.StringVar(value="none")
    tk.OptionMenu(controls, norm_var, "none", "min-max", "standard").pack(side=tk.RIGHT)
    cmap_var = tk.StringVar(value="blue_fire")
    tk.OptionMenu(controls, cmap_var, "blue_fire", "fire", "hue", "grayscale").pack(side=tk.RIGHT)
    auto_var = tk.BooleanVar(value=True)
    tk.Checkbutton(controls, text="Autoloop", variable=auto_var).pack(side=tk.RIGHT)
    speed_var = tk.IntVar(value=1)
    tk.Scale(controls, from_=-5, to=5, orient=tk.HORIZONTAL, variable=speed_var, label="Speed").pack(side=tk.RIGHT)

    add_row()
    add_column()

    epoch_label = tk.Label(root, text="Epoch 0", font=("Arial", 16))
    epoch_label.pack()
    image_label = tk.Label(root)
    image_label.pack()

    def _apply_norm(arr: np.ndarray, method: str) -> np.ndarray:
        if method == "none":
            return arr
        data = arr.astype(np.float32)
        if method == "min-max":
            mn, mx = data.min(), data.max()
            if mx - mn > 1e-8:
                data = (data - mn) / (mx - mn)
            else:
                data = np.zeros_like(data)
        elif method == "standard":
            data = (data - data.mean()) / (data.std() + 1e-8)
            mn, mx = data.min(), data.max()
            data = (data - mn) / (mx - mn + 1e-8)
        return (data * 255).clip(0, 255).astype(np.uint8)

    last_opts: list[str] = []
    last_layout: list[list[str]] = []
    last_cmap = cmap_var.get()
    last_norm = norm_var.get()

    def refresh_menus() -> None:
        nonlocal last_opts
        opts = _options()
        if opts != last_opts:
            for row_vars, row_menus in zip(grid_vars, grid_menus):
                for var, menu in zip(row_vars, row_menus):
                    m = menu["menu"]
                    m.delete(0, "end")
                    for s in opts:
                        m.add_command(label=s, command=tk._setit(var, s))
                    if var.get() not in opts and opts:
                        var.set(opts[0])
            last_opts = opts

    frame_index = 0

    def update():
        nonlocal frame_index, last_layout, last_cmap, last_norm
        if stop_event.is_set():
            root.quit()
            return
        changed = frame_cache.process_queue()
        refresh_menus()
        layout = [[var.get() for var in row] for row in grid_vars]
        current_cmap = cmap_var.get()
        current_norm = norm_var.get()
        if layout != last_layout or current_cmap != last_cmap or current_norm != last_norm or changed:
            frame_index = 0
            frame_cache.composite_cache.clear()
            last_layout = [row[:] for row in layout]
            last_cmap = current_cmap
            last_norm = current_norm
        grid = frame_cache.compose_layout_at(layout, frame_index)
        grid = apply_colormap(grid, current_cmap)
        grid = _apply_norm(grid, current_norm)
        groups = frame_cache._group_labels()
        lengths: list[int] = []
        for row in grid_vars:
            for var in row:
                label, _, stat = var.get().partition(":")
                if stat not in ("", "sample", "grid"):
                    continue
                if label in frame_cache.cache:
                    lengths.append(len(frame_cache.cache[label]))
                elif label in groups:
                    group_lengths = [len(frame_cache.cache[l]) for l in groups[label] if frame_cache.cache.get(l)]
                    if group_lengths:
                        lengths.append(min(group_lengths))
        if lengths:
            max_len = min(lengths)
            epoch_label.configure(text=f"Epoch {frame_index % max_len + 1}")
            if auto_var.get():
                speed = speed_var.get()
                if speed:
                    frame_index = (frame_index + speed) % max_len
        pil = Image.fromarray(grid)
        photo = ImageTk.PhotoImage(pil)
        image_label.configure(image=photo)
        image_label.image = photo
        root.after(update_ms, update)

    update()
    root.mainloop()


def main(
    config=None,
    viz_every=1,
    max_epochs=25,
    output_dir="riemann_modular_renders",
    visualize_laplace=None,
    laplace_threshold=None,
    laplace_path=None,
    show_laplace=False,
    dpi=200,
    deep_research=False,
    target_height=512,
    target_width=512,
    update_ms: int = 100,
    cmap: str = "hue",
):
    if config is None:
        config = build_config()
    frame_cache = FrameCache(target_height=target_height, target_width=target_width)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    shared_state = {
        "epoch_limit": max_epochs,
        "optimizer": "Adam",
        "lr": 5e-2,
        "reset_opt": False,
    }
    args = (
        frame_cache,
        shared_state,
        config,
        viz_every,
        max_epochs,
        visualize_laplace,
        laplace_threshold,
        laplace_path,
        show_laplace,
        dpi,
        deep_research,
        stop_event,
    )
    worker = threading.Thread(target=training_worker, args=args)
    worker.start()
    display_worker(frame_cache, stop_event, shared_state, update_ms=update_ms, max_epochs=max_epochs)
    worker.join()
    frame_cache.process_queue()
    frame_cache.save_animation("input_prediction", os.path.join(output_dir, "input_prediction.png"), cmap=cmap)
    frame_cache.save_animation("params_grads", os.path.join(output_dir, "params_grads.png"), cmap=cmap)
    print(f"Exported animations to the '{output_dir}/' directory.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--viz-laplace", action="store_true", help="Render Laplace isosurface")
    parser.add_argument("--laplace-threshold", type=float, help="Isosurface threshold")
    parser.add_argument("--laplace-path", type=str, help="Path to save Laplace surface image")
    parser.add_argument("--show-laplace", action="store_true", help="Display Laplace surface instead of closing")
    parser.add_argument("--viz-every", type=int, default=1, help="Visualization frequency")
    parser.add_argument("--max-epochs", type=int, default=250, help="Maximum training epochs")
    parser.add_argument("--output-dir", type=str, default="riemann_modular_renders", help="Root directory for output frames")
    parser.add_argument("--dpi", type=int, default=200, help="Figure resolution")
    parser.add_argument("--deep-research", action="store_true", help="Emit detailed tensor data")
    parser.add_argument("--target-height", type=int, default=512, help="Render target height")
    parser.add_argument("--target-width", type=int, default=512, help="Render target width")
    parser.add_argument("--update-ms", type=int, default=100, help="UI refresh interval in milliseconds")
    parser.add_argument("--cmap", type=str, default="hue", help="Colormap for exported animations")
    args = parser.parse_args()

    main(
        viz_every=args.viz_every,
        max_epochs=args.max_epochs,
        output_dir=args.output_dir,
        dpi=args.dpi,
        visualize_laplace=args.viz_laplace if args.viz_laplace else None,
        laplace_threshold=args.laplace_threshold,
        laplace_path=args.laplace_path,
        show_laplace=args.show_laplace,
        deep_research=args.deep_research,
        target_height=args.target_height,
        target_width=args.target_width,
        update_ms=args.update_ms,
        cmap=args.cmap,
    )
