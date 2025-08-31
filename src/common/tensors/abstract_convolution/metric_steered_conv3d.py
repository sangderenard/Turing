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
                        params.extend(v.parameters(include_structural=include_structural))
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
    ):
        Nu, Nv, Nw = grid_shape
        self.transform = transform
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
        wrap_module(self)

    def _build_laplace_package(self, boundary_conditions):
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
        # Preserve the LocalStateNetwork instance so its parameters persist
        # across builds and accumulate gradients between forward passes.
        if lsn is not None:
            self.local_state_network = lsn
        if lsn is not None:
            unused = []
            if self.deploy_mode == "weighted":
                if hasattr(lsn, "spatial_layer") and hasattr(lsn.spatial_layer, "parameters"):
                    unused.extend(lsn.spatial_layer.parameters())
                inner = getattr(lsn, "inner_state", None)
                if inner is not None and hasattr(inner, "parameters"):
                    try:
                        unused.extend(inner.parameters(include_all=True, include_structural=True))
                    except TypeError:
                        unused.extend(inner.parameters(include_all=True))
            elif self.deploy_mode == "modulated":
                def gather_weight_branch(net):
                    params = [net.g_weight_layer, net.g_bias_layer]
                    inner_net = getattr(net, "inner_state", None)
                    if inner_net is not None:
                        params.extend(gather_weight_branch(inner_net))
                    return params

                unused = gather_weight_branch(lsn)
            elif self.deploy_mode != "raw":
                raise ValueError("Invalid deploy_mode. Use 'raw', 'weighted', or 'modulated'.")

            # Ensure all non-structural parameters have gradient placeholders
            for p in lsn.parameters(include_all=True, include_structural=True):
                if any(p is u for u in unused):
                    continue
                if getattr(p, "grad", None) is None:
                    try:
                        p._grad = AbstractTensor.zeros_like(p)
                    except Exception:
                        pass

            if unused:
                autograd.structural(*unused)
        return package

    def forward(self, x):
        autograd.tape.annotate(x, label="MetricSteeredConv3DWrapper.input")
        autograd.tape.auto_annotate_eval(x)
        # Build the Laplace package inside the forward pass so gradients
        # from LocalStateNetwork propagate correctly.
        self.laplace_package = self._build_laplace_package(self.boundary_conditions)
        out = self.conv.forward(x, package=self.laplace_package)
        autograd.tape.annotate(out, label="MetricSteeredConv3DWrapper.output")
        autograd.tape.auto_annotate_eval(out)
        return out

