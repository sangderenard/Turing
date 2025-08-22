import importlib.util
import numpy as np
from collections import defaultdict, deque
import threading
import types

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    import torch  # pragma: no cover - optional dependency
    import torch.nn as nn  # pragma: no cover - optional dependency
else:  # torch not available
    torch = None  # type: ignore
    class _DummyModule:  # pragma: no cover - minimal stub
        pass
    nn = types.SimpleNamespace(Module=_DummyModule)  # type: ignore
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
class LocalStateNetwork(nn.Module):
    def __init__(self, metric_tensor_func, grid_shape, switchboard_config, cache_ttl=50, custom_hooks=None):
        """
        A mini-network for local state management, caching, NN integration, and procedural switchboarding.

        Args:
            metric_tensor_func: Function for metric tensor computation.
            grid_shape: Shape of the local grid.
            switchboard_config: Dictionary defining procedural processing flows for desired outputs.
            cache_ttl: Time-to-live (TTL) for cached values (default: 5 iterations).
            custom_hooks: Dictionary of hooks for custom tensor metrics.
        """
        super().__init__()
        self.metric_tensor_func = metric_tensor_func
        self.grid_shape = grid_shape
        self.switchboard_config = switchboard_config
        self.cache_ttl = cache_ttl
        self.custom_hooks = custom_hooks or {}

        # Cache Manager
        self.state_cache = {}  # Key: hashed position, Value: (tensor, iteration_count)
        self.current_iteration = 0  # For cache freshness
        self.cache_lock = threading.Lock()
        num_parameters = 27
        # NN Integration Manager
        self.weight_layer = nn.Parameter(torch.ones(3, 3, 3))
        self.spatial_layer = nn.Conv3d(
            in_channels=num_parameters,  
            out_channels=num_parameters,  
            kernel_size=3,  
            padding=1,  
            stride=1,
            bias=False
        )

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

        B, D, H, W, _, _, _ = padded_raw.shape
        
        # Step 1: Compute weighted_padded
        weight_layer = self.weight_layer.view(1,1,1,1,3,3,3) # Align dimensions for element-wise multiplication
        weighted_padded = padded_raw * weight_layer  # Apply spatial weights

        # Step 2: Reshape 3x3x3 state into 27 channels
        padded_flat = padded_raw.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3)  # (B, 27, D, H, W)

        # Step 3: Apply 3D convolution
        modulated_flat = self.spatial_layer(padded_flat)  # Output shape: (B, 27, D, H, W)

        # Step 4: Reshape back to original (B, D, H, W, 3, 3) shape
        modulated_padded = modulated_flat.permute(0, 2, 3, 4, 1).reshape(B, D, H, W, 3, 3, 3)

        return weighted_padded, modulated_padded
    # --------- CACHE MANAGER --------- #
    def check_or_compute(self, pos_hash, *inputs):
        """
        Check the cache for existing data or compute it if stale/missing.
        """
        with self.cache_lock:
            if pos_hash in self.state_cache:
                value, timestamp = self.state_cache[pos_hash]
                if self.current_iteration - timestamp < self.cache_ttl:
                    return value
            value = self.metric_tensor_func(*inputs)
            self.state_cache[pos_hash] = (value, self.current_iteration)
            return value

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
        weight_output = self.weight_layer.unsqueeze(-1).unsqueeze(-1) * state_tensor
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
        padded_raw = torch.zeros((*grid_shape, 3,  3, 3), device=g_ij.device)
        # Handle tension and density
        if additional_params is None:
            additional_params = {}

        tension = additional_params.get('tension', torch.ones(grid_shape, device=g_ij.device))
        if callable(tension):
            tension = tension(grid_u, grid_v, grid_w)  # Call the function if it's a hook

        density = additional_params.get('density', torch.ones(grid_shape, device=g_ij.device))
        if callable(density):
            density = density(grid_u, grid_v, grid_w)  # Call the function if it's a hook

        # Place g_ij (metric tensor) and g_inv (inverse metric tensor)
        padded_raw[..., 0, :, :] = g_ij[...,:,:]  # First 3x3 slot for metric tensor
        padded_raw[..., 1, :, :] = g_inv[...,:,:]  # Second 3x3 slot for inverse metric tensor

        # Place det_g (determinant) and additional parameters in diagonal of third 3x3 slot
        padded_raw[..., 2, 0, 0] = det_g  # Determinant of metric tensor
        padded_raw[..., 2, 1, 1] = tension
        padded_raw[..., 2, 2, 2] = density
        # Assign stencil constant dynamically (defaulting to STANDARD_STENCIL)
        default_stencil = INT_STANDARD_STENCIL if additional_params is None else additional_params.get("default_stencil", INT_STANDARD_STENCIL)
        stencil_map = torch.full(grid_u.shape, float(default_stencil), dtype=grid_u.dtype, device=grid_u.device)
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

# Test Example
def mock_metric_tensor_func(grid_u, grid_v, grid_w, *partials):
    shape = grid_u.shape + (3, 3)
    g_ij = torch.eye(3).repeat(*grid_u.shape, 1, 1)
    g_inv = torch.eye(3).repeat(*grid_u.shape, 1, 1)
    det_g = torch.ones(grid_u.shape)
    return g_ij, g_inv, det_g

grid_shape = (4, 4, 4)
if torch is not None:
    local_state = LocalStateNetwork(mock_metric_tensor_func, grid_shape, switchboard_config)
    grid_u = torch.ones(grid_shape)
    grid_v = torch.ones(grid_shape)
    grid_w = torch.ones(grid_shape)
    partials = (grid_u, grid_v, grid_w, grid_u, grid_v, grid_w)
    outputs = local_state(grid_u, grid_v, grid_w, partials)
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
else:  # pragma: no cover - fallback when torch missing
    local_state = None
