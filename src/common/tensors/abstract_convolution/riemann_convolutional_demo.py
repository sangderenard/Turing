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
from src.common.tensors.abstract_nn.optimizer import BPIDSGD, Adam
import numpy as np
from pathlib import Path
from skimage import measure
from .laplace_nd import BuildLaplace3D
import os
from PIL import Image
import threading
from .render_cache import FrameCache


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


def _random_spectral_gaussian(shape):
    """Return gaussian noise with randomized spectrum."""
    arr = np.random.randn(*shape)
    fft = np.fft.fftn(arr)
    phase = np.angle(fft)
    mag = np.random.randn(*shape)
    fft_rand = mag * np.exp(1j * phase)
    return np.fft.ifftn(fft_rand).real


def _low_entropy_variant(target_np, epoch):
    """Generate a low-entropy sample by rolling/flipping the target."""
    x = np.roll(target_np, shift=1, axis=-1)
    if (epoch // 2) % 2 == 0:
        x = np.flip(x, axis=-2)
    x += np.random.normal(scale=0.01, size=x.shape)
    return x

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
    B = 50000
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
            "C": 3,
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
    config=None,
    viz_every=1,
    low_entropy_every=5,
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
    transform, grid, _ = build_geometry(geom_cfg)
    grid_shape = geom_cfg["grid_shape"]
    B, C = train_cfg["B"], train_cfg["C"]
    layer = MetricSteeredConv3DWrapper(
        in_channels=C,
        out_channels=C,
        grid_shape=grid_shape,
        transform=transform,
        boundary_conditions=train_cfg.get("boundary_conditions", ("dirichlet",) * 6),
        k=train_cfg.get("k", 3),
        eig_from=train_cfg.get("eig_from", "g"),
        pointwise=train_cfg.get("pointwise", True),
        deploy_mode="modulated",
        laplace_kwargs={"lambda_reg": 0.5},
    )
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
    grad_enabled = getattr(_AT.autograd, '_no_grad_depth', 0) == 0
    print(f"[DEBUG] LSN instance id at layer creation: {id(layer.local_state_network)} | grad_tracking_enabled={grad_enabled}")
    U, V, W = grid.U, grid.V, grid.W
    autograd.tape.annotate(U, label="riemann_demo.grid_U")
    autograd.tape.auto_annotate_eval(U)
    autograd.tape.annotate(V, label="riemann_demo.grid_V")
    autograd.tape.auto_annotate_eval(V)
    autograd.tape.annotate(W, label="riemann_demo.grid_W")
    autograd.tape.auto_annotate_eval(W)
    target = (U + V + W).unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1, -1)
    autograd.tape.annotate(target, label="riemann_demo.target")
    autograd.tape.auto_annotate_eval(target)
    target_np = np.array(target.data if hasattr(target, "data") else target)
    x = AT.get_tensor(_random_spectral_gaussian((B, C, *grid_shape)), requires_grad=True)
    autograd.tape.annotate(x, label="riemann_demo.input_init")
    autograd.tape.auto_annotate_eval(x)
    # When AUTOGRAD_STRICT=1, unused tensors trigger connectivity errors.
    # Uncomment one of the lines below to relax those checks:
    # autograd.strict = False                   # disable strict mode globally
    # autograd.whitelist(x, target)             # or whitelist specific tensors
    # autograd.whitelist_labels(r"riemann_demo.*")  # whitelist by label pattern
    # --- Parameter and gradient collection helpers ---
    from ..logger import get_tensors_logger
    logger = get_tensors_logger()
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

        # Convolutional weights
        if hasattr(layer.conv, 'parameters'):
            seen_objs.add(id(layer.conv))
            for p in layer.conv.parameters():
                add_param(p)

        # LocalStateNetwork (if present)
        lsn = layer.local_state_network if hasattr(layer, 'local_state_network') else None
        if lsn is None:
            raise ValueError("LocalStateNetwork not found")
        if lsn and hasattr(lsn, 'parameters'):
            seen_objs.add(id(lsn))
            for p in lsn.parameters(include_all=True):
                add_param(p)
        else:
            raise ValueError("LocalStateNetwork not found")

        # Fallback: any other objects with .parameters
        if isinstance(layer.laplace_package, dict):
            for v in layer.laplace_package.values():
                vid = id(v)
                if vid in seen_objs or vid in seen_params:
                    continue
                if hasattr(v, 'parameters'):
                    seen_objs.add(vid)
                    for p in v.parameters():
                        add_param(p)
        # Log all params and grads, including Nones
        for i, (p, g) in enumerate(zip(params, grads)):
            label = getattr(p, '_label', None)
            logger.info(
                f"Param {i}: label={label}, shape={getattr(p, 'shape', None)}, grad is None={g is None}, grad shape={getattr(g, 'shape', None) if g is not None else None}"
            )
        return params, grads
    y = layer.forward(x)
    grad_enabled = getattr(_AT.autograd, '_no_grad_depth', 0) == 0
    print(f"[DEBUG] LSN instance id after forward: {id(layer.local_state_network)} | grad_tracking_enabled={grad_enabled}")
    print(f"[DEBUG] LSN param ids: {[id(p) for p in layer.local_state_network.parameters(include_all=True)]}")
    print(f"[DEBUG] LSN param requires_grad: {[getattr(p, 'requires_grad', None) for p in layer.local_state_network.parameters(include_all=True)]}")
    print(f"[DEBUG] LSN _regularization_loss: {layer.local_state_network._regularization_loss}")
    print(f"[DEBUG] LSN _regularization_loss grad_fn: {getattr(layer.local_state_network._regularization_loss, 'grad_fn', None)}")
    grad_enabled = getattr(_AT.autograd, '_no_grad_depth', 0) == 0
    print(f"[DEBUG] About to call backward on LSN _regularization_loss | grad_tracking_enabled={grad_enabled}")
    lsn = layer.local_state_network
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

    optimizer = Adam(params, lr=5e-2)
    loss_fn = lambda y, t: ((y - t) ** 2).mean() * 100
    for epoch in range(1, max_epochs + 1):
        if stop_event is not None and stop_event.is_set():
            break
        # Zero gradients for all params
        for p in params:
            if hasattr(p, 'zero_grad'):
                p.zero_grad()
            elif hasattr(p, '_grad'):
                p._grad = AbstractTensor.zeros_like(p._grad)
        if epoch % low_entropy_every == 0:
            np_x = _low_entropy_variant(target_np, epoch)
        else:
            np_x = _random_spectral_gaussian((B, C, *grid_shape))
        x = AT.get_tensor(np_x, requires_grad=True)
        autograd.tape.annotate(x, label=f"riemann_demo.input_epoch_{epoch}")
        autograd.tape.auto_annotate_eval(x)
        y = layer.forward(x)
        autograd.tape.auto_annotate_eval(y)
        if deep_research:
            print("[DEEP-RESEARCH] input data:", _to_numpy(x))
            print("[DEEP-RESEARCH] predicted data:", _to_numpy(y))
        loss = loss_fn(y, target)
        LSN_loss = layer.local_state_network._regularization_loss
        print(f"Epoch {epoch}: loss={loss.item()}, LSN_loss={LSN_loss.item()}")
        loss = LSN_loss + loss
        print(f"Total loss={loss.item()}")
        autograd.tape.annotate(loss, label="riemann_demo.loss")
        autograd.tape.auto_annotate_eval(loss)
        # layer.report_orphan_nodes()  # retired / no-op
        # Backward pass (assume .backward() populates ._grad)
        loss.backward()
        lsn = layer.local_state_network
        grad_w = getattr(lsn._weighted_padded, '_grad', AbstractTensor.zeros_like(lsn._weighted_padded))
        grad_m = getattr(lsn._modulated_padded, '_grad', AbstractTensor.zeros_like(lsn._modulated_padded))
        lsn.backward(grad_w, grad_m, lambda_reg=0.5)
        # Re-collect params and grads (in case new tensors were created)
        params, grads = collect_params_and_grads()
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
            # Prepare data for each quadrant
            low_entropy = AT.get_tensor(_low_entropy_variant(target_np, epoch))
            with autograd.no_grad():
                le_pred = layer.forward(low_entropy)
            gaussian = AT.get_tensor(_random_spectral_gaussian((B, C, *grid_shape)))
            with autograd.no_grad():
                gauss_pred = layer.forward(gaussian)

            # Use first batch/channel for visualization and apply PCA projection
            le_inp = np.array(low_entropy[0, 0].data if hasattr(low_entropy[0, 0], 'data') else low_entropy[0, 0])
            le_out = np.array(le_pred[0, 0].data if hasattr(le_pred[0, 0], 'data') else le_pred[0, 0])
            ga_inp = np.array(gaussian[0, 0].data if hasattr(gaussian[0, 0], 'data') else gaussian[0, 0])
            ga_out = np.array(gauss_pred[0, 0].data if hasattr(gauss_pred[0, 0], 'data') else gauss_pred[0, 0])

            quad1 = pca_to_rgb(le_inp)
            quad2 = pca_to_rgb(le_out)
            quad3 = pca_to_rgb(ga_inp)
            quad4 = pca_to_rgb(ga_out)

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
            pairs = []
            for idx, (p, g) in enumerate(zip(params, grads)):
                g = g if g is not None else AT.zeros_like(p)
                p_img = normalize_for_visualization(p)
                g_img = normalize_for_visualization(g)
                if p_img.ndim == 3:
                    p_img = p_img.mean(axis=0)
                if g_img.ndim == 3:
                    g_img = g_img.mean(axis=0)
                p_img = np.atleast_2d(p_img)
                g_img = np.atleast_2d(g_img)
                min_h = min(p_img.shape[0], g_img.shape[0])
                min_w = min(p_img.shape[1], g_img.shape[1])
                p_img = p_img[:min_h, :min_w]
                g_img = g_img[:min_h, :min_w]
                p_img = (_normalize(p_img) * 255).astype(np.uint8)
                g_img = (_normalize(g_img) * 255).astype(np.uint8)
                # Store individual param/grad visuals
                frame_cache.enqueue(f"param{idx}_param", p_img)
                frame_cache.enqueue(f"param{idx}_grad", g_img)
                pair = np.concatenate([p_img, g_img], axis=-1)
                pairs.append(pair)
            if pairs:
                max_w = max(pair.shape[1] for pair in pairs)
                padded_pairs = [
                    np.pad(pair, ((0, 0), (0, max_w - pair.shape[1])), mode="constant")
                    for pair in pairs
                ]
                params_grads_frame = np.concatenate(padded_pairs, axis=0)
            else:
                params_grads_frame = np.zeros((h, w * 2), dtype=np.uint8)

            # Legacy composite frames for animation exports
            frame_cache.enqueue("input_prediction", ip_frame)
            frame_cache.enqueue("params_grads", params_grads_frame)
        if loss.item() < 1e-6:
            print("Converged.")
            break
    # Audit: check that metric and local state network are used
    print("Metric at center voxel:", layer.laplace_package['metric']['g'][grid_shape[0]//2, grid_shape[1]//2, grid_shape[2]//2])
    if 'local_state_network' in layer.laplace_package:
        print("LocalStateNetwork parameters:", list(layer.laplace_package['local_state_network'].parameters()))
    if stop_event is not None:
        stop_event.set()


def display_worker(frame_cache: FrameCache, stop_event: threading.Event, update_ms: int = 100) -> None:
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

    row_frame = tk.Frame(controls)
    row_frame.pack(side=tk.LEFT)
    col_frame = tk.Frame(controls)
    col_frame.pack(side=tk.LEFT)

    row_vars = []
    row_menus = []
    col_vars = []
    col_menus = []

    def add_row():
        var = tk.StringVar()
        opts = frame_cache.available_sources() or [""]
        opt = tk.OptionMenu(row_frame, var, *opts)
        opt.pack()
        row_vars.append(var)
        row_menus.append(opt)

    def add_column():
        var = tk.StringVar()
        opts = frame_cache.available_types() or [""]
        opt = tk.OptionMenu(col_frame, var, *opts)
        opt.pack(side=tk.LEFT)
        col_vars.append(var)
        col_menus.append(opt)

    tk.Button(controls, text="Add Row", command=add_row).pack(side=tk.LEFT)
    tk.Button(controls, text="Add Column", command=add_column).pack(side=tk.LEFT)

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

    def refresh_menus():
        sources = frame_cache.available_sources()
        for var, menu in zip(row_vars, row_menus):
            m = menu["menu"]
            m.delete(0, "end")
            for s in sources:
                m.add_command(label=s, command=tk._setit(var, s))
            if var.get() not in sources and sources:
                var.set(sources[0])
        types = frame_cache.available_types()
        for var, menu in zip(col_vars, col_menus):
            m = menu["menu"]
            m.delete(0, "end")
            for t in types:
                m.add_command(label=t, command=tk._setit(var, t))
            if var.get() not in types and types:
                var.set(types[0])

    frame_index = 0

    def update():
        nonlocal frame_index
        if stop_event.is_set():
            root.quit()
            return
        frame_cache.process_queue()
        refresh_menus()
        layout = []
        selected_labels = []
        for r in row_vars:
            row = []
            for c in col_vars:
                label = f"{r.get()}_{c.get()}"
                row.append(label)
                selected_labels.append(label)
            layout.append(row)
        grid = frame_cache.compose_layout_at(layout, frame_index)
        if selected_labels:
            lengths = [len(frame_cache.cache.get(lbl, [])) for lbl in selected_labels if frame_cache.cache.get(lbl)]
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
    low_entropy_every=5,
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
):
    if config is None:
        config = build_config()
    frame_cache = FrameCache(target_height=target_height, target_width=target_width)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    args = (
        frame_cache,
        config,
        viz_every,
        low_entropy_every,
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
    display_worker(frame_cache, stop_event, update_ms=update_ms)
    worker.join()
    frame_cache.process_queue()
    frame_cache.save_animation("input_prediction", os.path.join(output_dir, "input_prediction.png"))
    frame_cache.save_animation("params_grads", os.path.join(output_dir, "params_grads.png"))
    print(f"Exported animations to the '{output_dir}/' directory.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--viz-laplace", action="store_true", help="Render Laplace isosurface")
    parser.add_argument("--laplace-threshold", type=float, help="Isosurface threshold")
    parser.add_argument("--laplace-path", type=str, help="Path to save Laplace surface image")
    parser.add_argument("--show-laplace", action="store_true", help="Display Laplace surface instead of closing")
    parser.add_argument("--viz-every", type=int, default=1, help="Visualization frequency")
    parser.add_argument("--low-entropy-every", type=int, default=1, help="Low-entropy sample frequency")
    parser.add_argument("--max-epochs", type=int, default=250, help="Maximum training epochs")
    parser.add_argument("--output-dir", type=str, default="riemann_modular_renders", help="Root directory for output frames")
    parser.add_argument("--dpi", type=int, default=200, help="Figure resolution")
    parser.add_argument("--deep-research", action="store_true", help="Emit detailed tensor data")
    parser.add_argument("--target-height", type=int, default=512, help="Render target height")
    parser.add_argument("--target-width", type=int, default=512, help="Render target width")
    parser.add_argument("--update-ms", type=int, default=100, help="UI refresh interval in milliseconds")
    args = parser.parse_args()

    main(
        viz_every=args.viz_every,
        low_entropy_every=args.low_entropy_every,
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
    )
