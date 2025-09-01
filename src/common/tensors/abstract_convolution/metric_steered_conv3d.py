"""
metric_steered_conv3d.py
------------------------

Wrapper that builds the geometry package (via Transform → GridDomain →
BuildLaplace3D) and applies the metric‑steered separable conv (NDPCA3Conv3d).

This is a rename/clarification of the former RiemannConvolutional3D wrapper:
it orchestrates the correct pipeline but does not itself implement a full
geodesic/Riemannian convolution; the primitive operator is metric‑steered.
"""

from .laplace_nd import BuildLaplace3D, GridDomain
from .ndpca3conv import NDPCA3Conv3d
from ..abstraction import AbstractTensor
from ..autograd import autograd
from ..abstract_nn import wrap_module


class MetricSteeredConv3DWrapper:
    """
    Metric‑aware 3D convolutional layer using the canonical geometry pipeline.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    grid_shape : (Nu, Nv, Nw)
    transform : PCANDTransform or compatible (provides metric_tensor_func)
    boundary_conditions : tuple[str, str, str, str, str, str]
    k : int (number of principal directions)
    eig_from : 'g' or 'inv_g'
    pointwise : bool
    deploy_mode : {'raw', 'weighted', 'modulated'}, optional
        Selects which Laplace tensor variant to use. Defaults to 'raw'.
    laplace_kwargs : dict, optional
        Additional keyword arguments forwarded to BuildLaplace3D.
    """

    def parameters(self, include_structural: bool = False):
        params = []
        if hasattr(self.conv, 'parameters') and callable(self.conv.parameters):
            params.extend(self.conv.parameters())
        lp = getattr(self, 'laplace_package', None)
        if isinstance(lp, dict):
            for v in lp.values():
                if hasattr(v, 'parameters') and callable(v.parameters):
                    try:
                        params.extend(v.parameters(include_all=True, include_structural=include_structural))
                    except TypeError:
                        params.extend(v.parameters())
        elif hasattr(lp, 'parameters') and callable(lp.parameters):
            try:
                params.extend(lp.parameters(include_structural=include_structural))
            except TypeError:
                params.extend(lp.parameters())
        if not include_structural:
            tape = getattr(autograd, 'tape', None)
            if tape is not None:
                params = [p for p in params if not tape.is_structural(p)]
        return params

    def zero_grad(self):
        if hasattr(self.conv, 'zero_grad') and callable(self.conv.zero_grad):
            self.conv.zero_grad()
        lp = getattr(self, 'laplace_package', None)
        if isinstance(lp, dict):
            for v in lp.values():
                if hasattr(v, 'zero_grad') and callable(v.zero_grad):
                    v.zero_grad()
        elif hasattr(lp, 'zero_grad') and callable(lp.zero_grad):
            lp.zero_grad()

    def get_input_shape(self):
        if hasattr(self.conv, 'get_input_shape') and callable(self.conv.get_input_shape):
            return self.conv.get_input_shape()
        return (None, getattr(self.conv, 'in_channels', None), *getattr(self.conv, 'grid_shape', (None, None, None)))

    def __init__(
        self,
        in_channels,
        out_channels,
        grid_shape,
        transform,
        boundary_conditions=("dirichlet",) * 6,
        k=3,
        eig_from="g",
        pointwise=True,
        deploy_mode="raw",
        laplace_kwargs=None,
        *,
        grid_sync_mode="snap_to_input",
        pad_value=0.0,
        verbose_shape=False,
    ):
        self.transform = transform
        self.grid_shape = tuple(grid_shape)
        Nu, Nv, Nw = self.grid_shape
        U = AbstractTensor.linspace(-1.0, 1.0, Nu).reshape(Nu, 1, 1) * AbstractTensor.ones((1, Nv, Nw))
        V = AbstractTensor.linspace(-1.0, 1.0, Nv).reshape(1, Nv, 1) * AbstractTensor.ones((Nu, 1, Nw))
        W = AbstractTensor.linspace(-1.0, 1.0, Nw).reshape(1, 1, Nw) * AbstractTensor.ones((Nu, Nv, 1))
        autograd.tape.annotate(U, label="MetricSteeredConv3DWrapper.grid_U")
        autograd.tape.auto_annotate_eval(U)
        autograd.tape.annotate(V, label="MetricSteeredConv3DWrapper.grid_V")
        autograd.tape.auto_annotate_eval(V)
        autograd.tape.annotate(W, label="MetricSteeredConv3DWrapper.grid_W")
        autograd.tape.auto_annotate_eval(W)
        self.grid_domain = GridDomain(
            U,
            V,
            W,
            grid_boundaries=(True,) * 6,
            transform=transform,
            coordinate_system="rectangular",
        )
        self.deploy_mode = deploy_mode
        self.laplace_kwargs = laplace_kwargs or {}
        self.boundary_conditions = boundary_conditions
        self.laplace_package = None
        # Cache builder and LocalStateNetwork so that their parameters are
        # preserved across multiple forward calls.  This ensures gradients from
        # the LocalStateNetwork accumulate instead of being re‑initialized with
        # each package build.
        self.laplace_builder = None
        self.local_state_network = None
        self.conv = NDPCA3Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            like=AbstractTensor.zeros((1, in_channels, Nu, Nv, Nw)),
            grid_shape=(Nu, Nv, Nw),
            boundary_conditions=boundary_conditions,
            k=k,
            eig_from=eig_from,
            pointwise=pointwise,
        )
        self.grid_sync_mode = grid_sync_mode
        self.pad_value = pad_value
        self.verbose_shape = verbose_shape
        wrap_module(self)

    def _shape_tuple(self, t):
        # Works with AbstractTensor (shape() method) and numpy-backed arrays
        if hasattr(t, "shape") and callable(t.shape):
            return tuple(int(s) for s in t.shape())
        if hasattr(t, "shape"):
            return tuple(int(s) for s in t.shape)
        return tuple(int(s) for s in t.size())

    def _refresh_conv_like(self):
        B_like = 1
        C_like = getattr(self.conv, 'in_channels', 1)
        dtype = self.conv.like.get_dtype() if hasattr(self.conv, 'like') and hasattr(self.conv.like, 'get_dtype') else getattr(AbstractTensor, 'float_dtype_', None)
        device = self.conv.like.get_device() if hasattr(self.conv, 'like') and hasattr(self.conv.like, 'get_device') else None
        self.conv.like = AbstractTensor.zeros((B_like, C_like, *self.grid_shape), dtype=dtype, device=device)

    def _snap_grid_to_input(self, x):
        # Make conv + wrapper grid follow input spatial dims
        Din, Hin, Win = self._shape_tuple(x)[-3:]
        new_grid = (int(Din), int(Hin), int(Win))
        if new_grid != self.grid_shape:
            self.grid_shape = new_grid
            self.conv.grid_shape = new_grid
            self.grid_domain = None
            self.laplace_builder = None
            self._refresh_conv_like()
            if self.verbose_shape:
                print(f"[MetricSteeredConv3DWrapper] grid -> {new_grid}")

    def _pad_to_cube(self, x):
        # Optional experiment mode: pad to SxSxS (not required by the math)
        Din, Hin, Win = self._shape_tuple(x)[-3:]
        S = max(Din, Hin, Win)
        padD, padH, padW = S - Din, S - Hin, S - Win
        if (padD | padH | padW) == 0:
            return x, (Din, Hin, Win)
        shape = self._shape_tuple(x)
        B = shape[0] if len(shape) >= 5 else 1
        C = shape[1] if len(shape) >= 5 else (shape[1] if len(shape) >= 2 else 1)
        dtype = x.get_dtype() if hasattr(x, "get_dtype") else getattr(x, "dtype", None)
        device = x.get_device() if hasattr(x, "get_device") else getattr(x, "device", None)
        out = AbstractTensor.full((B, C, S, S, S), self.pad_value, dtype=dtype, device=device)
        d0 = padD // 2
        h0 = padH // 2
        w0 = padW // 2
        d1 = d0 + Din
        h1 = h0 + Hin
        w1 = w0 + Win
        out[..., d0:d1, h0:h1, w0:w1] = x
        return out, (S, S, S)

    def _resample_geometry_to(self, package, src_grid, dst_grid, mode="nearest"):
        # If you prefer to keep geometry at some canonical resolution and map to input,
        # implement a small resampler (nearest/trilinear). Here we sketch nearest.
        if src_grid == dst_grid or not isinstance(package, dict):
            return package
        D0, H0, W0 = src_grid
        D1, H1, W1 = dst_grid

        def _nn_axis(a):
            a = a.reshape((D0, H0, W0))
            device = a.get_device() if hasattr(a, 'get_device') else None
            z = AbstractTensor.linspace(0, D0 - 1, D1, dtype=a.get_dtype() if hasattr(a, 'get_dtype') else None, device=device, requires_grad=False, tape=autograd.tape).round().to_dtype(AbstractTensor.long_dtype_)
            y = AbstractTensor.linspace(0, H0 - 1, H1, dtype=a.get_dtype() if hasattr(a, 'get_dtype') else None, device=device, requires_grad=False, tape=autograd.tape).round().to_dtype(AbstractTensor.long_dtype_)
            x = AbstractTensor.linspace(0, W0 - 1, W1, dtype=a.get_dtype() if hasattr(a, 'get_dtype') else None, device=device, requires_grad=False, tape=autograd.tape).round().to_dtype(AbstractTensor.long_dtype_)
            a = a.index_select(0, z)
            a = a.index_select(1, y)
            a = a.index_select(2, x)
            return a

        out = dict(package)
        for k in list(package.keys()):
            kl = k.lower()
            if any(s in kl for s in ("wu", "wv", "ww", "axis_u", "axis_v", "axis_w", "weight_u", "weight_v", "weight_w")):
                out[k] = _nn_axis(package[k])
        return out

    def _canonicalize_laplace_package(self, package: dict) -> dict:
        if not isinstance(package, dict):
            return package
        D, H, W = self.grid_shape
        want5 = (1, 1, D, H, W)

        def _to_5d(w):
            sh = self._shape_tuple(w)
            if sh == (D, H, W):
                return w.reshape(want5)
            if sh == want5:
                return w
            if D == 1 and sh == (1, H, W):
                return w.reshape(want5)
            n = 1
            for s in sh:
                n *= int(s)
            if n == D * H * W:
                return w.reshape((D, H, W)).reshape(want5)
            raise ValueError(f"Axis weight shape {sh} not compatible with grid {(D, H, W)}")

        out = dict(package)
        for k in list(package.keys()):
            kl = k.lower()
            if any(s in kl for s in ("wu", "wv", "ww", "axis_u", "axis_v", "axis_w", "weight_u", "weight_v", "weight_w")):
                out[k] = _to_5d(package[k])
        return out

    def _build_laplace_package(self, boundary_conditions):
        if self.grid_domain is None or tuple(self.grid_domain.U.shape) != self.grid_shape:
            Nu, Nv, Nw = self.grid_shape
            U = AbstractTensor.linspace(-1.0, 1.0, Nu).reshape(Nu, 1, 1) * AbstractTensor.ones((1, Nv, Nw))
            V = AbstractTensor.linspace(-1.0, 1.0, Nv).reshape(1, Nv, 1) * AbstractTensor.ones((Nu, 1, Nw))
            W = AbstractTensor.linspace(-1.0, 1.0, Nw).reshape(1, 1, Nw) * AbstractTensor.ones((Nu, Nv, 1))
            autograd.tape.annotate(U, label="MetricSteeredConv3DWrapper.grid_U")
            autograd.tape.auto_annotate_eval(U)
            autograd.tape.annotate(V, label="MetricSteeredConv3DWrapper.grid_V")
            autograd.tape.auto_annotate_eval(V)
            autograd.tape.annotate(W, label="MetricSteeredConv3DWrapper.grid_W")
            autograd.tape.auto_annotate_eval(W)
            self.grid_domain = GridDomain(
                U,
                V,
                W,
                grid_boundaries=(True,) * 6,
                transform=self.transform,
                coordinate_system="rectangular",
            )
            self.laplace_builder = None
        if self.laplace_builder is None:
            self.laplace_builder = BuildLaplace3D(
                grid_domain=self.grid_domain,
                wave_speed=343,
                precision=getattr(AbstractTensor, "float_dtype_", None) or self.grid_domain.U.dtype,
                resolution=self.grid_domain.U.shape[0],
                metric_tensor_func=self.transform.metric_tensor_func,
                boundary_conditions=boundary_conditions,
                artificial_stability=1e-10,
                device=getattr(self.grid_domain.U, "device", None),
            )
        builder = self.laplace_builder
        _, _, package = builder.build_general_laplace(
            self.grid_domain.U,
            self.grid_domain.V,
            self.grid_domain.W,
            return_package=True,
            deploy_mode=self.deploy_mode,
            local_state_network=self.local_state_network,
            **self.laplace_kwargs,
        )
        lsn = package.get("local_state_network")
        if lsn is not None:
            self.local_state_network = lsn
        return package

    def forward(self, x):
        autograd.tape.annotate(x, label="MetricSteeredConv3DWrapper.input")
        autograd.tape.auto_annotate_eval(x)

        # 1) Choose how to align lattices
        if self.grid_sync_mode == "pad_to_cube":
            x, new_grid = self._pad_to_cube(x)
            self.grid_shape = new_grid
            self.conv.grid_shape = new_grid
            self.grid_domain = None
            self.laplace_builder = None
            self._refresh_conv_like()
        elif self.grid_sync_mode == "snap_to_input":
            self._snap_grid_to_input(x)
        # else: 'resample_geometry' keeps self.grid_shape as-is

        # 2) Build geometry on self.grid_shape
        package = self._build_laplace_package(self.boundary_conditions)

        # 3) If keeping a canonical geometry grid, resample to the input grid
        if self.grid_sync_mode == "resample_geometry":
            Din, Hin, Win = self._shape_tuple(x)[-3:]
            package = self._resample_geometry_to(package, src_grid=self.grid_shape, dst_grid=(Din, Hin, Win))
            self.grid_shape = (Din, Hin, Win)
            self.conv.grid_shape = self.grid_shape
            self.grid_domain = None
            self.laplace_builder = None
            self._refresh_conv_like()
        package = self._canonicalize_laplace_package(package)

        # 4) Call the conv (package and conv grid now align with x)
        self.laplace_package = package
        out = self.conv.forward(x, package=package)
        autograd.tape.annotate(out, label="MetricSteeredConv3DWrapper.output")
        autograd.tape.auto_annotate_eval(out)
        return out

