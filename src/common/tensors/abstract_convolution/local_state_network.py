import numpy as np
from collections import defaultdict, deque
import threading

from ..abstraction import AbstractTensor
from ..abstract_nn import Linear, RectConv3d
from ..autograd import autograd

# ``LocalStateNetwork`` intentionally avoids heavyweight frameworks like
# PyTorch.  Convolutional support will be wired in once an abstract 3D
# convolution layer exists.  Until then we fall back to a simple Linear layer
# that acts on the flattened per-voxel state.
# Face Neighbors (6)
U_PLUS = 'u+'
U_MINUS = 'u-'
V_PLUS = 'v+'
V_MINUS = 'v-'
W_PLUS = 'w+'
W_MINUS = 'w-'

# Edge Neighbors (12)
U_PLUS_V_PLUS = 'u+v+'
U_PLUS_V_MINUS = 'u+v-'
U_MINUS_V_PLUS = 'u-v+'
U_MINUS_V_MINUS = 'u-v-'

U_PLUS_W_PLUS = 'u+w+'
U_PLUS_W_MINUS = 'u+w-'
U_MINUS_W_PLUS = 'u-w+'
U_MINUS_W_MINUS = 'u-w-'

V_PLUS_W_PLUS = 'v+w+'
V_PLUS_W_MINUS = 'v+w-'
V_MINUS_W_PLUS = 'v-w+'
V_MINUS_W_MINUS = 'v-w-'

# Corner Neighbors (8)
U_PLUS_V_PLUS_W_PLUS = 'u+v+w+'
U_PLUS_V_PLUS_W_MINUS = 'u+v+w-'
U_PLUS_V_MINUS_W_PLUS = 'u+v-w+'
U_PLUS_V_MINUS_W_MINUS = 'u+v-w-'

U_MINUS_V_PLUS_W_PLUS = 'u-v+w+'
U_MINUS_V_PLUS_W_MINUS = 'u-v+w-'
U_MINUS_V_MINUS_W_PLUS = 'u-v-w+'
U_MINUS_V_MINUS_W_MINUS = 'u-v-w-'


STANDARD_STENCIL = [
    U_PLUS, U_MINUS,
    V_PLUS, V_MINUS,
    W_PLUS, W_MINUS
]

EXTENDED_STENCIL = STANDARD_STENCIL + [
    U_PLUS_V_PLUS, U_PLUS_V_MINUS, U_MINUS_V_PLUS, U_MINUS_V_MINUS,
    U_PLUS_W_PLUS, U_PLUS_W_MINUS, U_MINUS_W_PLUS, U_MINUS_W_MINUS,
    V_PLUS_W_PLUS, V_PLUS_W_MINUS, V_MINUS_W_PLUS, V_MINUS_W_MINUS
]

FULL_STENCIL = EXTENDED_STENCIL + [
    U_PLUS_V_PLUS_W_PLUS, U_PLUS_V_PLUS_W_MINUS, U_PLUS_V_MINUS_W_PLUS, U_PLUS_V_MINUS_W_MINUS,
    U_MINUS_V_PLUS_W_PLUS, U_MINUS_V_PLUS_W_MINUS, U_MINUS_V_MINUS_W_PLUS, U_MINUS_V_MINUS_W_MINUS
]

LAPLACEBELTRAMI_STENCIL = [
    U_PLUS, U_MINUS, V_PLUS, V_MINUS, W_PLUS, W_MINUS,  # Face Neighbors
    U_PLUS_V_PLUS, U_PLUS_W_PLUS, V_PLUS_W_PLUS       # Cross-Term Edges
]

# Integer constants for stencil groups
INT_STANDARD_STENCIL = 1
INT_EXTENDED_STENCIL = 2
INT_FULL_STENCIL = 3
INT_LAPLACEBELTRAMI_STENCIL = 4

# Map integers to their stencil definitions
STENCIL_INT_CODES = {
    INT_STANDARD_STENCIL: STANDARD_STENCIL,
    INT_EXTENDED_STENCIL: EXTENDED_STENCIL,
    INT_FULL_STENCIL: FULL_STENCIL,
    INT_LAPLACEBELTRAMI_STENCIL: LAPLACEBELTRAMI_STENCIL
}

DEFAULT_CONFIGURATION = {
            'padded_raw': [{'func': lambda raw: raw, 'args': ['padded_raw']}],
            'weighted_padded': [{'func': lambda weighted: weighted, 'args': ['weighted_padded']}],
            'modulated_padded': [{'func': lambda modulated: modulated, 'args': ['modulated_padded']}]
        }
class LocalStateNetwork:
    def grads(self, include_structural: bool = False):
        """Return gradients corresponding to non-structural parameters by default."""
        grads = [self.g_weight_layer, self.g_bias_layer]
        # If spatial_layer has grads(), include them
        if hasattr(self.spatial_layer, 'grads') and callable(self.spatial_layer.grads):
            try:
                grads.extend(self.spatial_layer.grads(include_structural=include_structural))
            except TypeError:
                grads.extend(self.spatial_layer.grads())
        if self.inner_state is not None and hasattr(self.inner_state, 'grads'):
            try:
                grads.extend(self.inner_state.grads(include_structural=include_structural))
            except TypeError:
                grads.extend(self.inner_state.grads())
        if not include_structural:
            tape = getattr(autograd, 'tape', None)
            if tape is not None:
                grads = [g for g in grads if not tape.is_structural(g)]
        return grads

    def zero_grad(self):
        """
        Zero all gradients for this network and submodules.
        """
        if hasattr(self.g_weight_layer, 'zero_grad'):
            self.g_weight_layer.zero_grad()
        else:
            # Clear gradient attributes without altering the underlying data
            if hasattr(self.g_weight_layer, 'grad'):
                try:
                    self.g_weight_layer.grad = None  # type: ignore[attr-defined]
                except Exception:
                    pass
            if hasattr(self.g_weight_layer, '_grad'):
                self.g_weight_layer._grad = None
        if hasattr(self.g_bias_layer, 'zero_grad'):
            self.g_bias_layer.zero_grad()
        else:
            if hasattr(self.g_bias_layer, 'grad'):
                try:
                    self.g_bias_layer.grad = None  # type: ignore[attr-defined]
                except Exception:
                    pass
            if hasattr(self.g_bias_layer, '_grad'):
                self.g_bias_layer._grad = None
        if hasattr(self.spatial_layer, 'zero_grad') and callable(self.spatial_layer.zero_grad):
            self.spatial_layer.zero_grad()
        if self.inner_state is not None and hasattr(self.inner_state, 'zero_grad'):
            self.inner_state.zero_grad()
    def parameters(self, include_all: bool = False, include_structural: bool = False):
        """Return learnable parameters, excluding structural ones by default.
        optionally filtering by gradient status.

        Args:
            include_all: If ``True`` return every parameter regardless of whether it
                received a gradient in the last backward pass.  When ``False`` (the
                default) only parameters with ``grad`` set are returned.

        Returns:
            List of parameter tensors.
        """
        params = [self.g_weight_layer, self.g_bias_layer]
        # If spatial_layer has parameters(), include them

        if hasattr(self.spatial_layer, 'parameters') and callable(self.spatial_layer.parameters):
            params.extend(self.spatial_layer.parameters())
        if self.inner_state is not None and hasattr(self.inner_state, 'parameters'):
            try:
                params.extend(self.inner_state.parameters(include_all=include_all, include_structural=include_structural))
            except TypeError:
                params.extend(self.inner_state.parameters(include_all))
        if not include_structural:
            tape = getattr(autograd, 'tape', None)
            if tape is not None:
                params = [p for p in params if not tape.is_structural(p)]
        if include_all:
            return params
        return [p for p in params if getattr(p, "_grad", None) is not None]
    def __init__(self, metric_tensor_func, grid_shape, switchboard_config, cache_ttl=50, custom_hooks=None, recursion_depth=0, max_depth=2, _label_prefix=None, disable_cache=True, spatial_bias=True):
        """
        A mini-network for local state management, caching, NN integration, and procedural switchboarding.

        Args:
            metric_tensor_func: Function for metric tensor computation.
            grid_shape: Shape of the local grid.
            switchboard_config: Dictionary defining procedural processing flows for desired outputs.
            cache_ttl: Time-to-live (TTL) for cached values (default: 5 iterations).
            custom_hooks: Dictionary of hooks for custom tensor metrics.
            spatial_bias: If ``True`` include bias terms in the spatial layer.
        """
        self.metric_tensor_func = metric_tensor_func
        self.grid_shape = grid_shape
        self.switchboard_config = switchboard_config
        self.cache_ttl = cache_ttl
        self.custom_hooks = custom_hooks or {}
        self.recursion_depth = recursion_depth
        self.max_depth = max_depth
        self.disable_cache = disable_cache
        self.spatial_bias = spatial_bias
        # Cache Manager
        self.state_cache = {}  # Key: hashed position, Value: (tensor, iteration_count)
        self.current_iteration = 0  # For cache freshness
        self.cache_lock = threading.Lock()
        num_parameters = 27
        # NN Integration Manager
        self.g_weight_layer = AbstractTensor.ones((3, 3, 3), dtype=AbstractTensor.float_dtype, requires_grad=True)
        autograd.tape.create_tensor_node(self.g_weight_layer)
        self.g_weight_layer._label = f"{_label_prefix+'.' if _label_prefix else ''}LocalStateNetwork.g_weight_layer"

        self.g_bias_layer = AbstractTensor.zeros((3, 3, 3), dtype=AbstractTensor.float_dtype, requires_grad=True)
        autograd.tape.create_tensor_node(self.g_bias_layer)
        self.g_bias_layer._label = f"{_label_prefix+'.' if _label_prefix else ''}LocalStateNetwork.g_bias_layer"

        self._cached_padded_raw = None
        like = AbstractTensor.zeros((1, num_parameters), dtype=AbstractTensor.float_dtype)
        if recursion_depth < max_depth - 1:
            self.spatial_layer = RectConv3d(
                num_parameters,
                num_parameters,
                kernel_size=3,
                padding=1,
                like=like,
                bias=spatial_bias,
            )
            # RectConv3d does not have learnable parameters by default, but if it did, label them here
            self.inner_state = LocalStateNetwork(
                metric_tensor_func,
                grid_shape,
                switchboard_config,
                cache_ttl=cache_ttl,
                custom_hooks=custom_hooks,
                recursion_depth=recursion_depth + 1,
                max_depth=max_depth,
                _label_prefix=f"{_label_prefix+'.' if _label_prefix else ''}LocalStateNetwork.inner_state",
                disable_cache=disable_cache,
                spatial_bias=spatial_bias,
            )
        else:
            self.spatial_layer = Linear(
                num_parameters,
                num_parameters,
                like=like,
                bias=spatial_bias,
                _label_prefix=f"{_label_prefix+'.' if _label_prefix else ''}LocalStateNetwork.spatial_layer",
            )
            self.inner_state = None

        self.nn_generators = defaultdict(deque)


    def forward(self, padded_raw):
        """
        Forward pass to compute weighted_padded and modulated_padded.

        Args:
            padded_raw: Raw state tensor of shape (B, D, H, W, 3, 3)

        Returns:
            weighted_padded: Weighted version of padded_raw.
            modulated_padded: Modulated version with convolutional adjustment.
        """
        if len(padded_raw.shape) == 6:
            padded_raw = padded_raw.unsqueeze(0)

        self._cached_padded_raw = padded_raw

        B, D, H, W, _, _, _ = padded_raw.shape


        g_weight_layer = self.g_weight_layer.reshape((1, 1, 1, 1, 3, 3, 3))
        g_bias_layer = self.g_bias_layer.reshape((1, 1, 1, 1, 3, 3, 3))

        weighted_padded = padded_raw * g_weight_layer + g_bias_layer

        padded_view = padded_raw.reshape((B, D, H, W, -1))

        if isinstance(self.spatial_layer, RectConv3d):
            padded_view = padded_view.transpose(1, 4)
            padded_view = padded_view.transpose(2, 4)
            padded_view = padded_view.transpose(3, 4)
            modulated = self.spatial_layer.forward(padded_view)
            modulated = modulated.transpose(3, 4)
            modulated = modulated.transpose(2, 4)
            modulated = modulated.transpose(1, 4)
            modulated = modulated.reshape((B, D, H, W, -1))
        else:
            flat_view = padded_view.reshape((-1, padded_view.shape[-1]))
            modulated = self.spatial_layer.forward(flat_view)
            modulated = modulated.reshape((B, D, H, W, -1))

        modulated_padded = modulated.reshape((B, D, H, W, 3, 3, 3))

        if self.inner_state is not None:
            _, modulated_padded = self.inner_state.forward(modulated_padded)

        return weighted_padded, modulated_padded

    def backward(self, grad_weighted_padded, grad_modulated_padded):
        """Backward pass for ``LocalStateNetwork``.

        Args:
            grad_weighted_padded: Gradient from the weighted branch.
            grad_modulated_padded: Gradient from the modulated branch.

        Returns:
            Gradient with respect to the original ``padded_raw`` input.
        """
        if self._cached_padded_raw is None:
            raise RuntimeError("LocalStateNetwork.backward called before forward")

        padded_raw = self._cached_padded_raw
        B, D, H, W, _, _, _ = padded_raw.shape

        # Gradient through weight layer
        g_weight_layer = self.g_weight_layer.reshape((1, 1, 1, 1, 3, 3, 3))
        grad_from_weight = grad_weighted_padded * g_weight_layer

        # Accumulate gradients for g_weight_layer and g_bias_layer
        grad_weight = (grad_weighted_padded * padded_raw).sum(dim=(0, 1, 2, 3))
        if getattr(self.g_weight_layer, "_grad", None) is None:
            self.g_weight_layer._grad = grad_weight
        else:
            self.g_weight_layer._grad = self.g_weight_layer._grad + grad_weight

        grad_bias = grad_weighted_padded.sum(dim=(0, 1, 2, 3))
        if getattr(self.g_bias_layer, "_grad", None) is None:
            self.g_bias_layer._grad = grad_bias
        else:
            self.g_bias_layer._grad = self.g_bias_layer._grad + grad_bias

        # Propagate through any inner state network
        grad_mod = grad_modulated_padded
        if self.inner_state is not None:
            zero = grad_mod * 0
            grad_mod = self.inner_state.backward(zero, grad_mod)

        grad_mod = grad_mod.reshape((B, D, H, W, -1))

        # Backward through spatial layer
        if isinstance(self.spatial_layer, RectConv3d):
            grad_mod = grad_mod.transpose(1, 4)
            grad_mod = grad_mod.transpose(2, 4)
            grad_mod = grad_mod.transpose(3, 4)
            grad_padded_view = self.spatial_layer.backward(grad_mod)
            grad_padded_view = grad_padded_view.transpose(3, 4)
            grad_padded_view = grad_padded_view.transpose(2, 4)
            grad_padded_view = grad_padded_view.transpose(1, 4)
            grad_from_mod = grad_padded_view.reshape((B, D, H, W, 3, 3, 3))
        else:
            flat_grad = grad_mod.reshape((-1, grad_mod.shape[-1]))
            grad_flat_in = self.spatial_layer.backward(flat_grad)
            grad_from_mod = grad_flat_in.reshape((B, D, H, W, 3, 3, 3))

        grad_input = grad_from_weight + grad_from_mod

        # Clear cached tensors
        self._cached_padded_raw = None

        return grad_input
    # --------- CACHE MANAGER --------- #
    def check_or_compute(self, pos_hash, *inputs):
        """
        Check the cache for existing data or compute it if stale/missing.
        """
        if self.disable_cache:
            return self.metric_tensor_func(*inputs)
        with self.cache_lock:
            if pos_hash in self.state_cache:
                value, timestamp = self.state_cache[pos_hash]
                if self.current_iteration - timestamp < self.cache_ttl:
                    return value
            value = self.metric_tensor_func(*inputs)
            self.state_cache[pos_hash] = (value, self.current_iteration)
            return value

    def update_metric_function(self, new_metric_tensor_func):
        """
        Update the metric tensor function used by the local state network.
        """
        self.metric_tensor_func = new_metric_tensor_func
        with self.cache_lock:
            self.state_cache.clear()

    def update_iteration(self):
        """Increment the iteration for cache freshness checks."""
        with self.cache_lock:
            self.current_iteration += 1

    # --------- NN INTEGRATION MANAGER --------- #
    def get_nn_output(self, process_id, state_tensor):
        """
        Generator-based asynchronous NN computation.
        """
        flattened_input = state_tensor.view(-1)
        weight_output = (
            self.g_weight_layer.unsqueeze(-1).unsqueeze(-1) * state_tensor
            + self.g_bias_layer.unsqueeze(-1).unsqueeze(-1)
        )
        spatial_output = self.spatial_layer(flattened_input).view(state_tensor.shape)

        self.nn_generators[process_id].append(weight_output)
        yield weight_output

        self.nn_generators[process_id].append(spatial_output)
        yield spatial_output

    def fetch_nn_output(self, process_id):
        """
        Retrieve available NN outputs asynchronously.
        """
        while self.nn_generators[process_id]:
            yield self.nn_generators[process_id].popleft()

    # --------- SWITCHBOARD --------- #
    def compute_value(self, target_param, inputs, dependencies):
        """
        Compute a desired value using the switchboard configuration.
        """
        steps = self.switchboard_config.get(target_param, [])
        current_value = None

        for step in steps:
            func, args = step['func'], step['args']
            current_value = func(*[inputs.get(arg, dependencies.get(arg)) for arg in args])
        return current_value

    # --------- MAIN CALL FUNCTION --------- #
    def __call__(self, grid_u, grid_v, grid_w, partials, additional_params=None):
        """
        Compute tensors for the current local state.

        Args:
            grid_u, grid_v, grid_w: Grid coordinates.
            partials: Spatial partial derivatives.
            additional_params: Dictionary for tension, density, etc.

        Returns:
            Dictionary of tensors: 'padded_raw', 'weighted_padded', 'modulated_padded'.
        """
        # Cache check for precomputed metric tensors
        pos_hash = hash((grid_u.sum().item(), grid_v.sum().item(), grid_w.sum().item()))
        g_ij, g_inv, det_g = self.check_or_compute(pos_hash, grid_u, grid_v, grid_w, *partials)

        # Grid shape before the 3x3 matrices
        grid_shape = g_ij.shape[:-2]


        # Step 1: Initialize padded_raw
        padded_raw = AbstractTensor.zeros((*grid_shape, 3, 3, 3))

        # Handle tension and density
        if additional_params is None:
            additional_params = {}

        tension = additional_params.get('tension', AbstractTensor.ones(grid_shape))
        if callable(tension):
            tension = tension(grid_u, grid_v, grid_w)

        density = additional_params.get('density', AbstractTensor.ones(grid_shape))
        if callable(density):
            density = density(grid_u, grid_v, grid_w)

        # Place g_ij (metric tensor) and g_inv (inverse metric tensor)
        padded_raw[..., 0, :, :] = g_ij[...,:,:]
        padded_raw[..., 1, :, :] = g_inv[...,:,:]

        # Place det_g (determinant) and additional parameters in diagonal of third 3x3 slot
        padded_raw[..., 2, 0, 0] = det_g
        padded_raw[..., 2, 1, 1] = tension
        padded_raw[..., 2, 2, 2] = density
        # Assign stencil constant dynamically (defaulting to STANDARD_STENCIL)
        default_stencil = INT_STANDARD_STENCIL if additional_params is None else additional_params.get("default_stencil", INT_STANDARD_STENCIL)
        stencil_map = AbstractTensor.full(grid_u.shape, float(default_stencil))
        padded_raw[..., 2, 0, 1] = stencil_map

        
        
        # Step 2: Apply custom hooks for extra state data
        for (i, j), hook_fn in self.custom_hooks.items():
            padded_raw[..., i, j] = hook_fn(grid_u, grid_v, grid_w, partials, additional_params)

        # Step 3: Forward pass through the network
        weighted_padded, modulated_padded = self.forward(padded_raw)

        # Step 4: Return outputs
        return {
            'padded_raw': padded_raw,
            'weighted_padded': weighted_padded,
            'modulated_padded': modulated_padded
        }




# Example Switchboard Configuration
switchboard_config = {
    'padded_raw': [{'func': lambda raw: raw, 'args': ['padded_raw']}],
    'weighted_padded': [{'func': lambda weighted: weighted, 'args': ['weighted_padded']}],
    'modulated_padded': [{'func': lambda modulated: modulated, 'args': ['modulated_padded']}]
}
