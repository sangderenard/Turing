
import torch
import numpy as np
import math
import logging
from scipy.sparse import coo_matrix

# Configure the logger at the module level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of logs

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create formatter and add it to the handlers
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)

# Add the handlers to the logger
if not logger.handlers:
    logger.addHandler(ch)


import torch
import numpy as np
from scipy.sparse import coo_matrix
import math
import torch
import torch.nn as nn

from .local_state_network import LocalStateNetwork, DEFAULT_CONFIGURATION, INT_LAPLACEBELTRAMI_STENCIL
from src.common.index_composer.indexcomposer import GeneralIndexComposer

class BuildLaplace3D:
    def __init__(self, grid_domain, wave_speed=343, precision=torch.float64, resolution=68,
                 metric_tensor_func=None, density_func=None, tension_func=None,
                 singularity_conditions=None, singularity_dirichlet_func=None, singularity_neumann_func=None,
                 boundary_conditions=('neumann', 'neumann', 'neumann', 'neumann', 'neumann', 'neumann'),
                 artificial_stability=0, switchboard_config = {
                    'padded_raw': [{'func': lambda raw: raw, 'args': ['padded_raw']}],
                    'weighted_padded': [{'func': lambda weighted: weighted, 'args': ['weighted_padded']}],
                    'modulated_padded': [{'func': lambda modulated: modulated, 'args': ['modulated_padded']}]
                }, device=None):
        """
        Initialize BuildLaplace for 3D with additional dimension handling.
        
        Args:
            grid_domain: Object that handles the grid transformations (u, v, w) -> (x, y, z).
            wave_speed: Speed of wave propagation, used to compute local wave numbers.
            precision: Torch precision type for tensor creation (default: torch.float64).
            resolution: Maximum resolution for dense tensor calculations (default: 68).
            metric_tensor_func: Function to compute the metric tensor (default: None for Euclidean space).
            density_func: Function or tensor defining the density over the grid (default: None, assumes 1.0 everywhere).
            tension_func: Function or tensor defining the tension over the grid (default: None, assumes 1.0 everywhere).
            singularity_conditions: Conditions at singularities (default: None, assumes no singularities).
            singularity_dirichlet_func: Function to compute Dirichlet boundary values at singularities (default: None).
            singularity_neumann_func: Function to compute Neumann boundary values at singularities (default: None).
            boundary_conditions: Tuple specifying boundary conditions for u, v, w axes
                                 (default: ('dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet')).
            artificial_stability: Small stability term added to metrics (default: 0).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.general_index_composer = GeneralIndexComposer(device=self.device)
        
        self.grid_domain = grid_domain
        self.wave_speed = wave_speed
        self.precision = precision
        self.resolution = resolution
        self.metric_tensor_func = metric_tensor_func
        self.density_func = density_func
        self.tension_func = tension_func
        self.singularity_conditions = singularity_conditions
        self.singularity_dirichlet_func = singularity_dirichlet_func
        self.singularity_neumann_func = singularity_neumann_func
        self.boundary_conditions = boundary_conditions  # (u_min, u_max, v_min, v_max, w_min, w_max)
        self.artificial_stability = artificial_stability
        self.switchboard_config = switchboard_config

        # Generate Laplace labels dynamically based on boundary conditions
        self.labels = self._generate_laplace_labels(boundary_conditions)
        self.index_map_patterns = self.general_index_composer.generate_patterns(self.labels)
        self.index_map = self.general_index_composer.compose_indices(
            (self.grid_domain.resolution_u, self.grid_domain.resolution_v, self.grid_domain.resolution_w),
            self.index_map_patterns,
        )

    def _generate_laplace_labels(self, boundary_conditions):
        """
        Generates the complete label set for a 3D Laplace operator based on boundary conditions.

        Args:
            boundary_conditions (tuple): Boundary conditions for (u_min, u_max, v_min, v_max, w_min, w_max).

        Returns:
            list: Full set of labels (e.g., ['u+', 'U-', 'v+', 'V-', 'uv+', ...]).
        """
        # Identify periodicity: True if boundary condition is 'periodic'
        periodic_u = boundary_conditions[0] == 'periodic' or boundary_conditions[1] == 'periodic'
        periodic_v = boundary_conditions[2] == 'periodic' or boundary_conditions[3] == 'periodic'
        periodic_w = boundary_conditions[4] == 'periodic' or boundary_conditions[5] == 'periodic'

        # Case handling for axes
        u_case = 'U' if periodic_u else 'u'
        v_case = 'V' if periodic_v else 'v'
        w_case = 'W' if periodic_w else 'w'

        # Generate primary terms
        labels = [f"{u_case}+", f"{u_case}-", f"{v_case}+", f"{v_case}-", f"{w_case}+", f"{w_case}-"]

        # Generate cross terms
        labels += [f"{u_case}{v_case}+", f"{u_case}{v_case}-"]
        labels += [f"{u_case}{w_case}+", f"{u_case}{w_case}-"]
        labels += [f"{v_case}{w_case}+", f"{v_case}{w_case}-"]

        return labels

    def build_general_laplace(self, grid_u, grid_v, grid_w, boundary_conditions=None, singularity_conditions=None, 
                              singularity_dirichlet_func=None, singularity_neumann_func=None, k=0.0, 
                              metric_tensor_func=None, density_func=None, tension_func=None, 
                              device=None, grid_boundaries=(True, True, True, True, True, True), 
                              artificial_stability=None, f=0, normalize=False, deploy_mode="raw", dense=False):
        """
        Builds the Laplacian matrix for a 3D coordinate system using the provided u, v, w grids.
        Handles singularities using custom Dirichlet/Neumann conditions.
        """
        logger.debug("Starting build_general_laplace")
        logger.debug(f"Input grid shapes - grid_u: {grid_u.shape}, grid_v: {grid_v.shape}, grid_w: {grid_w.shape}")
        logger.debug(f"Boundary conditions: {boundary_conditions}")
        logger.debug(f"Singularity conditions: {singularity_conditions}")
        logger.debug(f"k: {k}, f: {f}")
        device = device or self.device
        if device != self.device:
            self.device = device
            self.general_index_composer.device = device
        logger.debug(f"Device: {device}, Grid boundaries: {grid_boundaries}")

        # Conditional reassignments, use method parameters if provided, otherwise use class attributes
        boundary_conditions = boundary_conditions if boundary_conditions is not None else self.boundary_conditions
        singularity_conditions = singularity_conditions if singularity_conditions is not None else self.singularity_conditions
        singularity_dirichlet_func = singularity_dirichlet_func if singularity_dirichlet_func is not None else self.singularity_dirichlet_func
        singularity_neumann_func = singularity_neumann_func if singularity_neumann_func is not None else self.singularity_neumann_func
        metric_tensor_func = metric_tensor_func if metric_tensor_func is not None else self.metric_tensor_func
        density_func = density_func if density_func is not None else self.density_func
        tension_func = tension_func if tension_func is not None else self.tension_func
        artificial_stability = artificial_stability if artificial_stability is not None else self.artificial_stability

        if grid_u.shape[0] != self.grid_domain.resolution_u or grid_v.shape[1] != self.grid_domain.resolution_v or grid_w.shape[2] != self.grid_domain.resolution_w:
            self.index_map = self.general_index_composer.compose_indices((grid_u.shape[0], grid_v.shape[0], grid_w.shape[0]), self.index_map_patterns)

        logger.debug("Parameters reassigned with either provided values or class attributes.")

        # Allow callers to pass either 1D coordinate vectors or full 3D meshes.
        def _to_mesh3d(u, v, w):
            if u.dim() == v.dim() == w.dim() == 1:
                U, V, W = torch.meshgrid(u, v, w, indexing='ij')
                return U, V, W
            if u.dim() == v.dim() == w.dim() == 3:
                return u, v, w
            raise ValueError("grid_u/v/w must all be 1D or all be 3D")

        grid_u, grid_v, grid_w = _to_mesh3d(grid_u, grid_v, grid_w)

        def default_metric_tensor(u, v, w, dxdu, dydu, dzdu, dxdv, dydv, dzdv, dxdw, dydw, dzdw):
            """
            Default metric tensor for a flat Euclidean 3D space.
            """
            logger.debug("Calculating default metric tensor.")
            logger.debug(f"dxdu shape: {dxdu.shape}, dxdv shape: {dxdv.shape}, dydu shape: {dydu.shape}, dydv shape: {dydv.shape}")

            g_uu = dxdu**2 + dydu**2 + dzdu**2
            g_vv = dxdv**2 + dydv**2 + dzdv**2
            g_ww = dxdw**2 + dydw**2 + dzdw**2
            g_uv = dxdu * dxdv + dydu * dydv + dzdu * dzdv
            g_uw = dxdu * dxdw + dydu * dydw + dzdu * dzdw
            g_vw = dxdv * dxdw + dydv * dydw + dzdv * dzdw

            logger.debug("Computed metric tensor components.")

            # Stack into a 3x3 matrix
            g_ij = torch.stack([
                torch.stack([g_uu, g_uv, g_uw], dim=-1),
                torch.stack([g_uv, g_vv, g_vw], dim=-1),
                torch.stack([g_uw, g_vw, g_ww], dim=-1)
            ], dim=-2)  # Shape: (N_u, N_v, N_w, 3, 3)

            logger.debug(f"g_ij shape: {g_ij.shape}")

            det_g = torch.det(g_ij)  # Shape: (N_u, N_v, N_w)
            logger.debug(f"det_g shape: {det_g.shape}")

            g_inv = torch.inverse(g_ij)  # Shape: (N_u, N_v, N_w, 3, 3)
            logger.debug("Computed inverse metric tensor.")

            return g_ij, g_inv, det_g

        if metric_tensor_func is None:
            metric_tensor_func = default_metric_tensor
            logger.debug("Using default metric tensor function.")

        # Apply the transformation function to the grid
        logger.debug("Applying transformation to the grid.")
        X, Y, Z = self.grid_domain.transform.transform(grid_u, grid_v, grid_w)[:3]  # Transform to physical space
        self.grid_u = grid_u
        self.grid_v = grid_v
        self.grid_w = grid_w
        N_u = grid_u.shape[0]
        N_v = grid_v.shape[1]
        N_w = grid_w.shape[2]

        logger.debug(f"Grid dimensions - N_u: {N_u}, N_v: {N_v}, N_w: {N_w}")

        # Compute partial derivatives
        logger.debug("Computing partial derivatives.")
        dXdu, dYdu, dZdu, dXdv, dYdv, dZdv, dXdw, dYdw, dZdw = self.grid_domain.transform.get_or_compute_partials(grid_u, grid_v, grid_w)
        logger.debug("Computed partial derivatives.")

        unique_u_values = grid_u[:, 0, 0]
        unique_v_values = grid_v[0, :, 0]
        unique_w_values = grid_w[0, 0, :]  # Assuming grid_w is along the third dimension

        logger.debug(f"Unique u values: {unique_u_values}")
        logger.debug(f"Unique v values: {unique_v_values}")
        logger.debug(f"Unique w values: {unique_w_values}")

        # Derive uniform step sizes for each axis and their squares.
        u_vals = unique_u_values
        v_vals = unique_v_values
        w_vals = unique_w_values
        du = (u_vals[1:] - u_vals[:-1]).mean()
        dv = (v_vals[1:] - v_vals[:-1]).mean()
        dw = (w_vals[1:] - w_vals[:-1]).mean()
        h2_u = du * du
        h2_v = dv * dv
        h2_w = dw * dw

        final_u_row = wrap_u_row = None
        final_v_row = wrap_v_row = None
        final_w_row = wrap_w_row = None

        # Handle the u-direction (dim=0, radial or x-direction)
        if boundary_conditions[0] == 'periodic' or boundary_conditions[1] == 'periodic':
            logger.debug("Handling periodic boundary conditions for u-direction.")
            sum_du = torch.sum(unique_u_values[1:] - unique_u_values[:-1])
            final_du = (2 * np.pi) - sum_du
            final_u_value = unique_u_values[-1]
            final_u_row = torch.full_like(unique_v_values.unsqueeze(1).repeat(1, N_w), final_u_value.item())
            wrap_u_row = torch.full_like(unique_v_values.unsqueeze(1).repeat(1, N_w), unique_u_values[0].item())

            logger.debug(f"sum_du: {sum_du}, final_du: {final_du}")
            logger.debug(f"final_u_row: {final_u_row}, wrap_u_row: {wrap_u_row}")

            # Calculate final dx, dy, dz using the transform function
            final_X, final_Y, final_Z = self.grid_domain.transform.transform_metric(final_u_row, unique_v_values.unsqueeze(1).repeat(1, N_w), unique_w_values.unsqueeze(0).repeat(N_u, 1))
            wrap_X, wrap_Y, wrap_Z = self.grid_domain.transform.transform_metric(wrap_u_row, unique_v_values.unsqueeze(1).repeat(1, N_w), unique_w_values.unsqueeze(0).repeat(N_u, 1))

            logger.debug("Calculated transformed metrics for u-direction boundaries.")

            # Calculate the final differential in x, y, z directions
            final_dXdu = (wrap_X - final_X)
            final_dYdu = (wrap_Y - final_Y)
            final_dZdu = (wrap_Z - final_Z)

            logger.debug("Calculated final differentials for u-direction.")

            # Append this final differential to dXdu, dYdu, dZdu
            if final_dXdu.dim() == 2:
                final_dXdu = final_dXdu.unsqueeze(0)
            if final_dYdu.dim() == 2:
                final_dYdu = final_dYdu.unsqueeze(0)
            if final_dZdu.dim() == 2:
                final_dZdu = final_dZdu.unsqueeze(0)

            dXdu = torch.cat([dXdu, final_dXdu], dim=0)
            dYdu = torch.cat([dYdu, final_dYdu], dim=0)
            dZdu = torch.cat([dZdu, final_dZdu], dim=0)
            logger.debug("Appended final differentials to u-direction derivatives.")
        else:
            logger.debug("Non-periodic boundary conditions for u-direction. No action taken.")

        # Handle the v-direction (dim=1, angular or y-direction)
        if boundary_conditions[2] == 'periodic' or boundary_conditions[3] == 'periodic':
            logger.debug("Handling periodic boundary conditions for v-direction.")
            sum_dv = torch.sum(unique_v_values[1:] - unique_v_values[:-1])
            final_dv = (2 * np.pi) - sum_dv
            final_v_value = unique_v_values[-1]
            final_v_row = torch.full_like(unique_u_values.unsqueeze(1).unsqueeze(2).repeat(1, 1, N_w), final_v_value.item())
            wrap_v_row = torch.full_like(unique_u_values.unsqueeze(1).unsqueeze(2).repeat(1, 1, N_w), unique_v_values[0].item())

            logger.debug(f"sum_dv: {sum_dv}, final_dv: {final_dv}")
            logger.debug(f"final_v_row: {final_v_row}, wrap_v_row: {wrap_v_row}")

            # Calculate final dx, dy, dz using the transform function
            final_X, final_Y, final_Z = self.grid_domain.transform.transform_metric(
                unique_u_values.unsqueeze(1).unsqueeze(2).repeat(1, 1, N_w),
                final_v_row,
                unique_w_values.unsqueeze(0).repeat(N_u, 1, 1)
            )
            wrap_X, wrap_Y, wrap_Z = self.grid_domain.transform.transform_metric(
                unique_u_values.unsqueeze(1).unsqueeze(2).repeat(1, 1, N_w),
                wrap_v_row,
                unique_w_values.unsqueeze(0).repeat(N_u, 1, 1)
            )

            logger.debug("Calculated transformed metrics for v-direction boundaries.")

            # Calculate the final differential in x, y, z directions
            final_dXdv = (wrap_X - final_X)
            final_dYdv = (wrap_Y - final_Y)
            final_dZdv = (wrap_Z - final_Z)

            logger.debug("Calculated final differentials for v-direction.")

            # Append this final differential to dXdv, dYdv, dZdv
            if final_dXdv.dim() == 2:
                final_dXdv = final_dXdv.unsqueeze(1)
            if final_dYdv.dim() == 2:
                final_dYdv = final_dYdv.unsqueeze(1)
            if final_dZdv.dim() == 2:
                final_dZdv = final_dZdv.unsqueeze(1)

            dXdv = torch.cat([dXdv, final_dXdv], dim=1)
            dYdv = torch.cat([dYdv, final_dYdv], dim=1)
            dZdv = torch.cat([dZdv, final_dZdv], dim=1)
            logger.debug("Appended final differentials to v-direction derivatives.")
        else:
            logger.debug("Non-periodic boundary conditions for v-direction. No action taken.")

        # Handle the w-direction (dim=2, z-direction)
        if boundary_conditions[4] == 'periodic' or boundary_conditions[5] == 'periodic':
            logger.debug("Handling periodic boundary conditions for w-direction.")
            sum_dw = torch.sum(unique_w_values[1:] - unique_w_values[:-1])
            final_dw = (2 * np.pi) - sum_dw
            final_w_value = unique_w_values[-1]
            final_w_row = torch.full_like(unique_u_values.unsqueeze(1).unsqueeze(2).repeat(1, N_v, 1), final_w_value.item())
            wrap_w_row = torch.full_like(unique_u_values.unsqueeze(1).unsqueeze(2).repeat(1, N_v, 1), unique_w_values[0].item())

            logger.debug(f"sum_dw: {sum_dw}, final_dw: {final_dw}")
            logger.debug(f"final_w_row: {final_w_row}, wrap_w_row: {wrap_w_row}")

            # Calculate final dx, dy, dz using the transform function
            final_X, final_Y, final_Z = self.grid_domain.transform.transform_metric(grid_u, grid_v, final_w_row)[:3]
            wrap_X, wrap_Y, wrap_Z = self.grid_domain.transform.transform_metric(grid_u, grid_v, wrap_w_row)[:3]

            logger.debug("Calculated transformed metrics for w-direction boundaries.")

            # Calculate the final differential in x, y, z directions
            final_dXdw = (wrap_X - final_X)
            final_dYdw = (wrap_Y - final_Y)
            final_dZdw = (wrap_Z - final_Z)

            logger.debug("Calculated final differentials for w-direction.")

            # Append this final differential to dXdw, dYdw, dZdw
            if final_dXdw.dim() == 2:
                final_dXdw = final_dXdw.unsqueeze(2)
            if final_dYdw.dim() == 2:
                final_dYdw = final_dYdw.unsqueeze(2)
            if final_dZdw.dim() == 2:
                final_dZdw = final_dZdw.unsqueeze(2)

            dXdw = torch.cat([dXdw, final_dXdw], dim=2)
            dYdw = torch.cat([dYdw, final_dYdw], dim=2)
            dZdw = torch.cat([dZdw, final_dZdw], dim=2)
            logger.debug("Appended final differentials to w-direction derivatives.")
        else:
            logger.debug("Non-periodic boundary conditions for w-direction. No action taken.")

        # Prepare for sparse matrix construction
        logger.debug("Preparing for sparse matrix construction.")
        row_indices = []
        col_indices = []
        values = []
        diagonal_entries = {}

        # Precompute the metric tensor (g_ij), its inverse (g_inv), and the determinant (det_g) for the entire grid
        if metric_tensor_func is not None:
            logger.debug("Computing metric tensor using the provided function.")
            # Apply the metric tensor function to the entire grid
            #g_ij, g_inv, det_g = metric_tensor_func(grid_u, grid_v, grid_w, dXdu, dYdu, dZdu, dXdv, dYdv, dZdv, dXdw, dYdw, dZdw)
            # Initialize LocalState with grid shape and metric function
            local_state = LocalStateNetwork(metric_tensor_func, (N_u, N_v, N_w), DEFAULT_CONFIGURATION)
            
            # Move to CPU and convert to numpy for processing
            logger.debug("Moving metric tensors to CPU and converting to numpy.")
            #g_ij = g_ij.clone().detach().cpu().numpy()  # Shape: (N_u, N_v, N_w, 3, 3)
            #g_inv = g_inv.clone().detach().cpu().numpy()  # Shape: (N_u, N_v, N_w, 3, 3)
            #det_g_all = det_g.clone().detach().cpu().numpy()  # Shape: (N_u, N_v, N_w)
            logger.debug("Metric tensors computed and converted.")
        else:
            logger.error("Metric tensor function must be provided for 3D Laplacian.")
            raise ValueError("Metric tensor function must be provided for 3D Laplacian.")

        # Iterate over the 3D grid to construct Laplacian
        logger.debug("Starting iteration over the 3D grid to construct Laplacian.")
        
        

        # 2. Obtain State Outputs
        state_outputs = local_state(
            grid_u, grid_v, grid_w,
            partials=(dXdu, dYdu, dZdu, dXdv, dYdv, dZdv, dXdw, dYdw, dZdw),
            additional_params={"default_stencil":INT_LAPLACEBELTRAMI_STENCIL}
        )

        raw_tensor = state_outputs['padded_raw']
        weighted_tensor = state_outputs['weighted_padded']
        modulated_tensor = state_outputs['modulated_padded']

        # 3. Select Appropriate Tensor Based on deploy_mode
        if deploy_mode == 'raw':
            selected_tensor = raw_tensor
        elif deploy_mode == 'weighted':
            selected_tensor = weighted_tensor
        elif deploy_mode == 'modulated':
            selected_tensor = modulated_tensor
        else:
            raise ValueError("Invalid deploy_mode. Use 'raw', 'weighted', or 'modulated'.")

        # 4. Extract Components from Selected Tensor
        g_ij = selected_tensor[..., 0, :, :]      # Metric tensor
        g_inv = selected_tensor[..., 1, :, :]     # Inverse metric tensor
        det_g = selected_tensor[..., 2, 0, 0]     # Determinant of the metric tensor
        tension = selected_tensor[..., 2, 1, 1]
        density = selected_tensor[..., 2, 2, 2]



        # 5. Compute grid metrics using step sizes with optional stabilizer.
        eps = artificial_stability
        metric_u = h2_u + eps
        metric_v = h2_v + eps
        metric_w = h2_w + eps

        # 6. Identify Singularities
        singularity_mask = torch.zeros_like(det_g, dtype=torch.bool)

        # 7. Extract Cross Terms
        inv_g_uv = g_inv[..., 0, 1]
        inv_g_uw = g_inv[..., 0, 2]
        inv_g_vw = g_inv[..., 1, 2]

        # 8. Initialize Diagonal Entries
        diagonal_entries = torch.zeros_like(det_g)

        if singularity_conditions == 'dirichlet':
            diagonal_entries[singularity_mask] = 1.0
        elif singularity_conditions == 'neumann':
            diagonal_entries[singularity_mask] = 0.0

        scale = tension / density
        laplacian_off_diag_u = -scale * g_inv[..., 0, 0] / metric_u
        laplacian_off_diag_v = -scale * g_inv[..., 1, 1] / metric_v
        laplacian_off_diag_w = -scale * g_inv[..., 2, 2] / metric_w

        laplacian_diag = -2.0 * (laplacian_off_diag_u + laplacian_off_diag_v + laplacian_off_diag_w)


        # 20. Assemble the Sparse Laplacian Matrix

        # 1. Prepare Laplacian Matrix Contributions
        row_indices = []
        col_indices = []
        values = []

        # Diagonal Entries Initialization
        total_size = torch.prod(torch.tensor(det_g.shape)).item()
        diagonal_entries = torch.zeros(total_size, device='cpu')

        # 2. Iterate Through the Index Map Patterns
        for label, row_indices_map in self.index_map['row_indices'].items():
            col_indices_map = self.index_map['col_indices'][label]

            # Identify Contributions Based on Label
            if label in ['u+', 'u-']:
                laplacian_contrib = laplacian_off_diag_u
            elif label in ['v+', 'v-']:
                laplacian_contrib = laplacian_off_diag_v
            elif label in ['w+', 'w-']:
                laplacian_contrib = laplacian_off_diag_w
            elif 'uv' in label:
                laplacian_contrib = inv_g_uv / density
            elif 'uw' in label:
                laplacian_contrib = inv_g_uw / density
            elif 'vw' in label:
                laplacian_contrib = inv_g_vw / density
            else:
                raise ValueError("unsupported terms")
                

            # Flatten Values and Indices
            row_indices.extend(row_indices_map.flatten().tolist())
            col_indices.extend(col_indices_map.flatten().tolist())
            values.extend(laplacian_contrib[self.index_map['masks'][label].cpu()].flatten().tolist())

            print(f"row_indices_map shape {row_indices_map.shape}" )
            print(f"col_indices_map shape {col_indices_map.shape}" )
            print(f"laplacian_contrib flattened shape {laplacian_contrib.flatten().shape}" )

        # 3. Add Diagonal Contributions
        flat_diag_indices = torch.arange(total_size, device='cpu')
        row_indices.extend(flat_diag_indices.tolist())
        col_indices.extend(flat_diag_indices.tolist())
        values.extend(laplacian_diag.flatten().tolist())

        # 4. Assemble Sparse Tensor
        laplacian = torch.sparse_coo_tensor(
            indices=torch.tensor([row_indices, col_indices], device='cpu'),
            values=torch.tensor(values, device='cpu'),
            size=(total_size, total_size)
        ).coalesce()

        laplacian = coo_matrix((laplacian.values().cpu().numpy(), laplacian.indices().cpu().numpy()), shape=laplacian.shape)
        # Now, use laplacian_matrix as needed in your application

        # Convert to dense tensor and move to specified device
        perturbation_mode = False  # Enable perturbation
        perturbation_seed = 42      # Use a fixed seed for reproducibility, or None for random
        perturbation_scale = 1e-3   # Small scale for the noise perturbation

        if self.resolution <= 50 and dense:
            logger.debug("Converting Laplacian to dense tensor.")
            laplacian_dense = laplacian.toarray()
            laplacian_tensor = torch.tensor(laplacian_dense, device=device, dtype=self.precision)
            logger.debug(f"Dense Laplacian tensor created with shape {laplacian_tensor.shape} on device {device}.")

            # Dense perturbation
            if perturbation_mode:
                logger.debug("Applying dense perturbation to Laplacian tensor.")
                if perturbation_seed is not None:
                    torch.manual_seed(perturbation_seed)  # Set seed for deterministic behavior
                    logger.debug(f"Set perturbation seed to {perturbation_seed}.")
                # Apply Gaussian noise to dense matrix
                noise_dense = torch.randn(laplacian_tensor.shape, dtype=self.precision, device=device) * perturbation_scale
                laplacian_tensor += noise_dense  # Add noise to the dense Laplacian
                logger.debug("Added Gaussian noise to dense Laplacian tensor.")

            # Validate perturbed dense Laplace tensor
            logger.debug("Validating dense Laplacian tensor.")
            self.validate_laplace_tensor(laplacian_tensor)
            logger.debug("Dense Laplacian tensor validated.")
        else:
            laplacian_tensor = None
            logger.debug("Resolution exceeds 128. Skipping dense Laplacian tensor creation.")

        # Sparse perturbation (always applied independently)
        if perturbation_mode:
            logger.debug("Applying sparse perturbation to Laplacian matrix.")
            if perturbation_seed is not None:
                np.random.seed(perturbation_seed)  # Use numpy's random seed for reproducibility
                logger.debug(f"Set sparse perturbation seed to {perturbation_seed}.")

            # Generate noise for the non-zero elements in numpy format
            noise_sparse = np.random.randn(laplacian.data.shape[0]) * perturbation_scale
            logger.debug(f"Generated noise for {laplacian.data.shape[0]} non-zero elements.")

            # Apply the noise by creating a new COO matrix with perturbed data
            laplacian = coo_matrix((laplacian.data + noise_sparse, (laplacian.row, laplacian.col)), 
                                   shape=laplacian.shape)
            logger.debug("Applied noise to sparse Laplacian matrix.")

        # Validate perturbed sparse Laplace tensor
        logger.debug("Validating sparse Laplacian matrix.")
        self.validate_laplace_tensor(laplacian)
        logger.debug("Sparse Laplacian matrix validated.")

        logger.debug("Completed build_general_laplace.")
        return laplacian_tensor, laplacian


    def validate_laplace_tensor(self, laplace_tensor, check_diagonal=True, check_off_diagonal=True, verbose=True):
        """
        Validates a given Laplace tensor for issues such as zero diagonal entries, NaN, Inf, or invalid values.
        
        Args:
            laplace_tensor: The Laplace tensor to validate, which can be sparse (COO) or dense.
            check_diagonal: Whether to check for invalid or zero diagonal entries (default: True).
            check_off_diagonal: Whether to check for invalid values in the off-diagonal elements (default: True).
            verbose: Whether to print detailed error messages (default: True).
        
        Returns:
            valid: Boolean indicating whether the Laplace tensor passed all checks.
        """
        valid = True

        # Handle both dense and sparse cases
        if isinstance(laplace_tensor, torch.Tensor):
            # Dense matrix case
            diagonal = torch.diag(laplace_tensor)

            # Check diagonal entries in dense matrix
            if check_diagonal:
                if torch.any(diagonal == 0):
                    valid = False
                    if verbose:
                        zero_indices = torch.where(diagonal == 0)[0].tolist()
                        print(f"Zero diagonal entries detected at indices: {zero_indices}")
                
                if torch.any(torch.isnan(diagonal)):
                    valid = False
                    if verbose:
                        nan_indices = torch.where(torch.isnan(diagonal))[0].tolist()
                        print(f"NaN detected in diagonal at indices: {nan_indices}")
                
                if torch.any(torch.isinf(diagonal)):
                    valid = False
                    if verbose:
                        inf_indices = torch.where(torch.isinf(diagonal))[0].tolist()
                        print(f"Inf detected in diagonal at indices: {inf_indices}")

            # Check off-diagonal entries in dense matrix
            if check_off_diagonal:
                off_diagonal = laplace_tensor - torch.diag(torch.diag(laplace_tensor))
                if torch.any(torch.isnan(off_diagonal)):
                    valid = False
                    if verbose:
                        nan_locations = torch.where(torch.isnan(off_diagonal))
                        print(f"NaN detected in off-diagonal at indices: {nan_locations}")
                
                if torch.any(torch.isinf(off_diagonal)):
                    valid = False
                    if verbose:
                        inf_locations = torch.where(torch.isinf(off_diagonal))
                        print(f"Inf detected in off-diagonal at indices: {inf_locations}")

        elif isinstance(laplace_tensor, coo_matrix):
            # Sparse matrix case (COO format)

            # Check diagonal entries in sparse matrix
            if check_diagonal:
                diagonal = laplace_tensor.diagonal()
                if (diagonal == 0).any():
                    valid = False
                    if verbose:
                        zero_indices = np.where(diagonal == 0)[0].tolist()
                        print(f"Zero diagonal entries detected at indices: {zero_indices}")
                if np.isnan(diagonal).any():
                    valid = False
                    if verbose:
                        nan_indices = np.where(np.isnan(diagonal))[0].tolist()
                        print(f"NaN detected in diagonal at indices: {nan_indices}")
                if np.isinf(diagonal).any():
                    valid = False
                    if verbose:
                        inf_indices = np.where(np.isinf(diagonal))[0].tolist()
                        print(f"Inf detected in diagonal at indices: {inf_indices}")

            # Check off-diagonal entries in sparse matrix
            if check_off_diagonal:
                row, col = laplace_tensor.row, laplace_tensor.col
                data = laplace_tensor.data
                for idx, (i, j, value) in enumerate(zip(row, col, data)):
                    if i != j:  # Only check off-diagonal elements
                        if np.isnan(value):
                            valid = False
                            if verbose:
                                print(f"NaN detected in off-diagonal at indices: ({i}, {j})")
                        if np.isinf(value):
                            valid = False
                            if verbose:
                                print(f"Inf detected in off-diagonal at indices: ({i}, {j})")

        else:
            raise TypeError("Unsupported matrix format. Please provide a torch.Tensor or scipy.sparse matrix.")

        if verbose and valid:
            print("Laplace tensor passed all validation checks.")

        return valid
# Assuming RectangularTransform and GridDomain are properly defined for 3D

import torch
import numpy as np
from scipy.sparse import coo_matrix
import math
import matplotlib.pyplot as plt


import torch
import random

class FaceMapGenerator:
    def __init__(self, vertices, edges, device="cpu"):
        """
        Args:
            vertices: (N, 3) tensor of vertex positions.
            edges: (E, 2) tensor of edge indices.
        """
        self.vertices = vertices.to(device)
        self.edges = edges.to(device)
        self.device = device
        self.graph = self.build_edge_graph()

    def build_edge_graph(self):
        from collections import defaultdict
        graph = defaultdict(list)
        for edge in self.edges:
            u, v = edge.tolist()
            graph[u].append(v)
            graph[v].append(u)
        return graph

    def find_edge_loops(self, max_face_size=6):
        visited_edges = set()
        faces = []

        def dfs(current, start, path):
            if len(path) > max_face_size:
                return
            for neighbor in self.graph[current]:
                edge = tuple(sorted((current, neighbor)))
                if neighbor == start and len(path) > 2:
                    faces.append(path + [start])
                    return
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    dfs(neighbor, start, path + [neighbor])
                    visited_edges.remove(edge)

        for start in range(self.vertices.shape[0]):
            dfs(start, start, [start])

        # Eliminate duplicates
        unique_faces = set(tuple(sorted(face)) for face in faces)
        return [list(face) for face in unique_faces]

    def is_planar(self, face):
        v0 = self.vertices[face[0]]
        normal = None
        for i in range(1, len(face)-1):
            edge1 = self.vertices[face[i]] - v0
            edge2 = self.vertices[face[i+1]] - v0
            cross_product = torch.cross(edge1, edge2)
            if normal is None:
                normal = cross_product
            else:
                # If not approximately parallel (same direction), not planar
                if not torch.allclose(normal, cross_product, atol=1e-6):
                    return False
        return True

    def generate_face_map(self, check_planarity=True):
        face_map = {}
        candidate_faces = self.find_edge_loops()
        face_index = 0
        for face in candidate_faces:
            if not check_planarity or self.is_planar(face):
                face_map[face_index] = face
                face_index += 1
        return face_map

class VolumeMapGenerator:
    def __init__(self, vertices, edges, faces, device="cpu"):
        self.vertices = vertices.to(device)
        self.edges = edges.to(device)
        self.faces = faces
        self.device = device

    def generate_volume_map(self):
        # Placeholder for future logic.
        # Could detect 3D volumes (tetrahedra) from faces.
        # For now, return an empty dict or a passthrough.
        print("Volume detection not implemented yet.")
        return {}
    
import torch
import hashlib

class HodgeStarBuilder:
    def __init__(self, device="cpu"):
        self.device = device
        self.hodge_cache = {}

    def _hash_topology(self, vertices, edges, faces=None):
        if faces is None:
            faces_data = b''
        else:
            # Convert face_map to a stable structure for hashing
            # face_map: {face_index: [v1, v2, v3, ...]}
            # Sort each face and then sort by face_index
            face_list = [(k, sorted(v)) for k,v in faces.items()]
            face_list.sort(key=lambda x: x[0])
            faces_data = str(face_list).encode()

        topology_data = (vertices.shape, edges.cpu().numpy().tobytes(), faces_data)
        return hashlib.sha256(b''.join([str(t).encode() for t in topology_data])).hexdigest()

    def build_basic_hodge_star(self, vertices, edges):
        """
        Basic hodge star: only vertex volumes and edge dual volumes if possible.
        Without face information, we compute simple vertex-based volumes and
        edge-based dual measures. This is a simplified approximation.
        """
        hash_key = self._hash_topology(vertices, edges)
        if hash_key in self.hodge_cache:
            return self.hodge_cache[hash_key]

        # Compute primal volumes for vertices as a trivial measure (e.g., unit)
        # Without faces, we cannot get a proper area-based measure easily.
        # Placeholder: assign uniform volumes
        num_vertices = vertices.shape[0]
        vertex_volumes = torch.ones(num_vertices, device=self.device)
        
        # For edges: length could be a primal measure, and dual might be inverse
        print(f"vertices dimensions: {vertices.shape} edges dimensions: {edges.shape}")
        edge_lengths = torch.sqrt(torch.sum((vertices[edges[:,0]] - vertices[edges[:,1]])**2, dim=1))
        # Dual area approx: 1 / length (very rough approximation)
        dual_edge = 1/(edge_lengths+1e-8)

        hodge_0 = torch.diag(vertex_volumes)
        hodge_1 = torch.diag(dual_edge)

        # No faces => no hodge_2. Mark availability so downstream
        # consumers know higher-order operators are absent for this
        # topology.
        availability = {"hodge_0": True, "hodge_1": True, "hodge_2": False}
        hodge_stars = {
            "hodge_0": hodge_0,
            "hodge_1": hodge_1,
            "hodge_2": None,
            "availability": availability,
        }

        self.hodge_cache[hash_key] = hodge_stars
        return hodge_stars

    def build_full_hodge_star(self, vertices, edges, faces):
        """
        Build full hodge star including faces. With face information, we can
        properly compute areas and thus refine the Hodge star operators.
        """
        hash_key = self._hash_topology(vertices, edges, faces)
        if hash_key in self.hodge_cache:
            return self.hodge_cache[hash_key]

        # Compute vertex volumes (0-forms)
        vertex_volumes = torch.zeros(vertices.shape[0], device=self.device)
        face_tensors = []
        for f_idx, face in faces.items():
            face_tensor = torch.tensor(face, device=self.device)
            face_tensors.append(face_tensor)
            v0, v1, v2 = vertices[face_tensor[0]], vertices[face_tensor[1]], vertices[face_tensor[2]]
            area = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0))
            # Distribute area equally among vertices for volume approximation
            for vert in face:
                vertex_volumes[vert] += area/3.0

        # Compute edge dual areas (1-forms)
        # Each edge belongs to some faces; sum area contributions
        edge_dual_areas = torch.zeros(edges.shape[0], device=self.device)
        if face_tensors:
            face_tensor = torch.stack(face_tensors)
        else:
            # No faces were detected; fall back to an empty tensor so downstream
            # operations become no-ops instead of crashing with a stack error.
            face_tensor = torch.empty((0, 3), dtype=torch.long, device=self.device)
        for i, edge in enumerate(edges):
            # Find faces containing this edge
            mask = (face_tensor == edge[0]).any(dim=1) & (face_tensor == edge[1]).any(dim=1)
            shared_faces = face_tensor[mask]
            dual_area = 0.0
            for f in shared_faces:
                v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
                area = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0))
                dual_area += area/3.0
            edge_dual_areas[i] = dual_area if dual_area > 0 else 1.0  # fallback

        # Compute face areas (2-forms)
        face_areas = []
        for f in faces.values():
            f_tensor = torch.tensor(f, device=self.device)
            v0, v1, v2 = vertices[f_tensor[0]], vertices[f_tensor[1]], vertices[f_tensor[2]]
            area = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0))
            face_areas.append(area)
        face_areas = torch.tensor(face_areas, device=self.device)
        has_faces = face_areas.numel() > 0

        hodge_0 = torch.diag(vertex_volumes)
        hodge_1 = torch.diag(edge_dual_areas)
        hodge_2 = torch.diag(face_areas) if has_faces else None

        availability = {"hodge_0": True, "hodge_1": True, "hodge_2": has_faces}
        hodge_stars = {
            "hodge_0": hodge_0,
            "hodge_1": hodge_1,
            "hodge_2": hodge_2,
            "availability": availability,
        }
        self.hodge_cache[hash_key] = hodge_stars
        return hodge_stars


class TransformHub:
    def __init__(self, uextent, vextent, grid_boundaries, device='cpu'): #need to add wextent badly, this is a huge oversight impacting all extending classes
        self.uextent = uextent
        self.vextent = vextent
        self.grid_boundaries = grid_boundaries
        self.device = device

    def calculate_geometry(self, U, V, W, edge_index=None, detect_faces=False, detect_volumes=False):
        """
        Compute geometry (coordinates, metric tensors, normals) and primary DEC operators.
        Optionally detect faces and volumes for advanced DEC computations.
        """
        # Compute coordinates and metric-related data
        X, Y, Z, dX_dU, dY_dU, dZ_dU, dX_dV, dY_dV, dZ_dV, dX_dW, dY_dW, dZ_dW, normals = \
            self.compute_partials_and_normals(U, V, W)
        g_ij, g_inv, det_g = self.metric_tensor_func(
            U, V, W, dX_dU, dY_dU, dZ_dU, dX_dV, dY_dV, dZ_dV, dX_dW, dY_dW, dZ_dW
        )

        geometry = {
            "coordinates": (X, Y, Z),
            "normals": normals,
            "metric": {"tensor": g_ij, "inverse": g_inv, "determinant": det_g}
        }

        if edge_index is not None:
            # Flatten the coordinate grids
            X_flat = X.flatten()  # shape: (N*N*N,)
            Y_flat = Y.flatten()  # shape: (N*N*N,)
            Z_flat = Z.flatten()  # shape: (N*N*N,)

            # Now create a (num_vertices, 3) vertex array
            vertex_reference = torch.stack([X_flat, Y_flat, Z_flat], dim=1)  # shape: (N*N*N, 3)
            
            network_profile = self.build_network_profile(edge_index, X, Y, Z)
            d_operators = self.build_d_operators(edge_index, network_profile)
            vertices = torch.stack([X, Y, Z], dim=1)  # (N, 3) vertex positions
            
            hodge_builder = HodgeStarBuilder(self.device)

            # Base Hodge star: without faces or volumes, can still do vertex and edge ops
            hodge_stars = hodge_builder.build_basic_hodge_star(vertex_reference, edge_index)

            geometry["DEC"] = {
                "network_profile": network_profile,
                "d_operators": d_operators,
                "hodge_stars": hodge_stars
            }

            # Optional: Enhanced detection of faces
            if detect_faces:
                face_generator = FaceMapGenerator(vertex_reference, edge_index, device=self.device)
                face_map = face_generator.generate_face_map()
                geometry["DEC"]["faces"] = face_map

                # Update Hodge stars now that we have faces
                hodge_stars = hodge_builder.build_full_hodge_star(vertex_reference, edge_index, face_map)
                geometry["DEC"]["hodge_stars"] = hodge_stars

            # Optional: Enhanced detection of volumes (3-simplices)
            if detect_volumes:
                volume_generator = VolumeMapGenerator(vertex_reference, edge_index, geometry["DEC"].get("faces", None), device=self.device)
                volume_map = volume_generator.generate_volume_map()
                geometry["DEC"]["volumes"] = volume_map
                # Extend Hodge stars further with volume information if needed
                # hodge_stars = self.hodge_builder.build_3d_hodge_star(vertices, edge_index, face_map, volume_map)
                # geometry["DEC"]["hodge_stars"] = hodge_stars

        return geometry

    def build_network_profile(self, edge_index, X, Y, Z):
        # Flatten coordinates
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()

        source, target = edge_index[:, 0], edge_index[:, 1]
        edge_lengths = torch.sqrt((X_flat[source] - X_flat[target])**2 +
                                (Y_flat[source] - Y_flat[target])**2 +
                                (Z_flat[source] - Z_flat[target])**2)
        num_vertices = X_flat.numel()
        return {
            "edge_index": edge_index,
            "edge_lengths": edge_lengths,
            "num_vertices": num_vertices,
            "num_edges": edge_index.shape[0]
        }


    def build_d_operators(self, edge_index, network_profile):
        num_vertices = network_profile["num_vertices"]
        num_edges = network_profile["num_edges"]

        d0 = torch.zeros((num_edges, num_vertices), device=self.device)
        for i, (src, tgt) in enumerate(edge_index):
            d0[i, src] = -1
            d0[i, tgt] = 1

        # d1 placeholder if we have faces defined
        # This can be constructed similarly if face maps are available
        d1 = None

        return {"d0": d0, "d1": d1}

    def compute_frobenius_norm(self, g_ij):
        """
        Compute the Frobenius norm of the metric tensor.

        Args:
            g_ij (torch.Tensor): Metric tensor with shape (..., 2, 2) for 2D surfaces.

        Returns:
            torch.Tensor: Frobenius norm of the metric tensor.
        """
        self.frobenius_norm = torch.sqrt(torch.sum(g_ij**2, dim=(-2, -1)))
        return self.frobenius_norm

    def compute_partials_and_normals(self, U, V, W, validate_normals=True, diagnostic_mode=False):
        # Ensure U, V, W require gradients for autograd
        U.requires_grad_(True)
        V.requires_grad_(True)
        W.requires_grad_(True)

        # Forward pass: get transformed coordinates
        X, Y, Z = self.transform_spatial(U, V, W)

        if diagnostic_mode:
            print("U Grid:")
            print(U)
            print("V Grid:")
            print(V)
            print("W Grid:")
            print(W)
            print("Transformed Coordinates (X, Y, Z):")
            print("X:", X)
            print("Y:", Y)
            print("Z:", Z)

        # Calculate partial derivatives with respect to U
        dXdu = torch.autograd.grad(X, U, grad_outputs=torch.ones_like(X), retain_graph=True, allow_unused=True)[0]
        dYdu = torch.autograd.grad(Y, U, grad_outputs=torch.ones_like(Y), retain_graph=True, allow_unused=True)[0]
        dZdu = torch.autograd.grad(Z, U, grad_outputs=torch.ones_like(Z), retain_graph=True, allow_unused=True)[0]

        # Calculate partial derivatives with respect to V
        dXdv = torch.autograd.grad(X, V, grad_outputs=torch.ones_like(X), retain_graph=True, allow_unused=True)[0]
        dYdv = torch.autograd.grad(Y, V, grad_outputs=torch.ones_like(Y), retain_graph=True, allow_unused=True)[0]
        dZdv = torch.autograd.grad(Z, V, grad_outputs=torch.ones_like(Z), retain_graph=True, allow_unused=True)[0]

        # Calculate partial derivatives with respect to W
        dXdw = torch.autograd.grad(X, W, grad_outputs=torch.ones_like(X), retain_graph=True, allow_unused=True)[0]
        dYdw = torch.autograd.grad(Y, W, grad_outputs=torch.ones_like(Y), retain_graph=True, allow_unused=True)[0]
        dZdw = torch.autograd.grad(Z, W, grad_outputs=torch.ones_like(Z), retain_graph=True, allow_unused=True)[0]

        target_shape = U.shape  # (N_u, N_v, N_w)

        # Handle None values from autograd
        dXdu = dXdu if dXdu is not None else torch.zeros(target_shape).to(U.device)
        dYdu = dYdu if dYdu is not None else torch.zeros(target_shape).to(U.device)
        dZdu = dZdu if dZdu is not None else torch.zeros(target_shape).to(U.device)
        dXdv = dXdv if dXdv is not None else torch.zeros(target_shape).to(V.device)
        dYdv = dYdv if dYdv is not None else torch.zeros(target_shape).to(V.device)
        dZdv = dZdv if dZdv is not None else torch.zeros(target_shape).to(V.device)
        dXdw = dXdw if dXdw is not None else torch.zeros(target_shape).to(W.device)
        dYdw = dYdw if dYdw is not None else torch.zeros(target_shape).to(W.device)
        dZdw = dZdw if dZdw is not None else torch.zeros(target_shape).to(W.device)

        if diagnostic_mode:
            print("Partial Derivatives:")
            print("dXdu:", dXdu)
            print("dYdu:", dYdu)
            print("dZdu:", dZdu)
            print("dXdv:", dXdv)
            print("dYdv:", dYdv)
            print("dZdv:", dZdv)
            print("dXdw:", dXdw)
            print("dYdw:", dYdw)
            print("dZdw:", dZdw)

        # Compute normals as cross-product of partial derivatives
        normals = torch.stack([
            torch.linalg.cross(torch.stack([dXdu, dYdu, dZdu], dim=-1), torch.stack([dXdv, dYdv, dZdv], dim=-1), dim=-1),
            torch.linalg.cross(torch.stack([dXdv, dYdv, dZdv], dim=-1), torch.stack([dXdw, dYdw, dZdw], dim=-1), dim=-1),
            torch.linalg.cross(torch.stack([dXdw, dYdw, dZdw], dim=-1), torch.stack([dXdu, dYdu, dZdu], dim=-1), dim=-1)
        ], dim=-1)

        # Compute distances from the origin
        distances = torch.sqrt(X**2 + Y**2 + Z**2)

        # Select the top 10% farthest points
        top_10_percent_threshold = max(1, int(0.1 * distances.numel()))
        top_10_percent_indices = torch.topk(distances.flatten(), top_10_percent_threshold).indices

        # Randomly sample 10% of the top 10% farthest points
        sample_size = max(1, int(0.1 * top_10_percent_threshold))
        sample_indices = random.sample(top_10_percent_indices.tolist(), sample_size)

        # Conduct majority check based on sampled normals
        outward_votes = 0
        inward_votes = 0
        grid_shape = distances.shape  # (N_u, N_v, N_w)
        for idx in sample_indices:
            i, j, k = np.unravel_index(idx, grid_shape)  # Convert flat index to 3D grid indices
            farthest_point = torch.tensor([X[i, j, k], Y[i, j, k], Z[i, j, k]], device=U.device, dtype=U.dtype)
            outward_reference_point = 1.01 * farthest_point  # 1% further outward

            # Directional check based on the sampled normal and reference point
            sample_normal = normals[i, j, k]
            direction_to_reference = outward_reference_point - farthest_point
            # Ensure sample_normal and direction_to_reference are broadcastable
            if sample_normal.dim() == 2 and direction_to_reference.dim() == 1:
                dot_product = torch.einsum('ij,j->i', sample_normal, direction_to_reference)
            else:
                dot_product = torch.dot(sample_normal, direction_to_reference)

            if (dot_product > 0).all():
                outward_votes += 1
            else:
                inward_votes += 1

        # Conditionally invert normals based on majority vote
        if inward_votes > outward_votes:
            normals = -normals

        # Continue with normalization and validation
        norm_magnitudes = torch.norm(normals, dim=-1, keepdim=True)

        # Normalize normals, avoid division by zero for zero-magnitude normals
        normals = torch.where(norm_magnitudes > 1e-16, normals / norm_magnitudes, normals)

        # Identify zero-magnitude normals
        zero_norm_mask = norm_magnitudes.squeeze() < 1e-16  # Boolean mask for zero-magnitude normals

        if torch.any(zero_norm_mask):
            count_zero_normals = torch.sum(zero_norm_mask).item()  # Number of zero-magnitude normals
            print(f"{count_zero_normals} out of {normals.numel()} zero-magnitude normals detected.")

            if diagnostic_mode:
                # Find the indices of the first zero-magnitude normal
                zero_indices = torch.nonzero(zero_norm_mask, as_tuple=True)
                first_zero_idx = (zero_indices[0][0].item(), zero_indices[1][0].item(), zero_indices[2][0].item())

                print(f"First zero-magnitude normal at index: {first_zero_idx}")

                # Extract the partials contributing to this normal
                i, j, k = first_zero_idx

                partials = {
                    'dXdu': dXdu[i, j, k],
                    'dYdu': dYdu[i, j, k],
                    'dZdu': dZdu[i, j, k],
                    'dXdv': dXdv[i, j, k],
                    'dYdv': dYdv[i, j, k],
                    'dZdv': dZdv[i, j, k],
                    'dXdw': dXdw[i, j, k],
                    'dYdw': dYdw[i, j, k],
                    'dZdw': dZdw[i, j, k],
                }

                print("Partials at the first zero-magnitude normal:")
                for name, value in partials.items():
                    print(f"{name}[{i}, {j}, {k}] = {value}")

                # Stop execution until the issue is resolved
                print("Diagnostics complete. Exiting due to zero-magnitude normal.")
                exit()
            else:
                # Proceed to repair zero-magnitude normals if not in diagnostic mode
                print("Repairing zero-magnitude normals.")

                # Repair zero-magnitude normals by averaging surrounding normals
                zero_indices = torch.nonzero(zero_norm_mask, as_tuple=True)
                for idx in zip(*zero_indices):
                    # Collect neighboring normals
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                if di == 0 and dj == 0 and dk == 0:
                                    continue  # Skip the center point
                                ni, nj, nk = idx[0] + di, idx[1] + dj, idx[2] + dk
                                if (0 <= ni < grid_shape[0] and 
                                    0 <= nj < grid_shape[1] and 
                                    0 <= nk < grid_shape[2]):
                                    neighbor_normal = normals[ni, nj, nk]
                                    neighbor_magnitude = norm_magnitudes[ni, nj, nk]
                                    if neighbor_magnitude > 1e-16:
                                        neighbors.append(neighbor_normal)
                    if neighbors:
                        avg_normal = torch.mean(torch.stack(neighbors), dim=0)
                        avg_normal_norm = torch.norm(avg_normal)
                        if avg_normal_norm > 1e-16:
                            normals[idx[0], idx[1], idx[2]] = avg_normal / avg_normal_norm  # Normalize average
                        else:
                            print(f"Unable to repair normal at index {idx} due to zero magnitude of averaged normal.")
                    else:
                        print(f"No valid neighbors to repair normal at index {idx}.")

        if validate_normals:
            # Validation checks for the final normals
            if torch.any(torch.isnan(normals)):
                print("Validation failed: NaN values detected in normals.")
                exit()
            if not torch.all(torch.isfinite(normals)):
                print("Validation failed: Non-finite values detected in normals.")
                exit()
            if not torch.allclose(torch.norm(normals, dim=-1), torch.ones_like(norm_magnitudes.squeeze()), atol=1e-5):
                print("Validation failed: Normals are not unit length within tolerance after normalization.")
                exit()

            print("Validation passed: Normals are ideal.")

        return X, Y, Z, dXdu, dYdu, dZdu, dXdv, dYdv, dZdv, dXdw, dYdw, dZdw, normals

                                            


    def get_or_compute_partials(self, U, V, W):
        """
        Helper to compute partials if they are not provided.
        
        Args:
            U, V (torch.Tensor): Parameter grids.
            dX_dU, dY_dU, dX_dV, dY_dV, dZ_dU, dZ_dV (torch.Tensor or None): Optional partials.
        
        Returns:
            Tuple[torch.Tensor]: Partial derivatives.
        """
        _, _, _, dX_dU, dY_dU, dZ_dU, dX_dV, dY_dV, dZ_dV, dXdw, dYdw, dZdw, _ = self.compute_partials_and_normals(U, V, W)
        return dX_dU, dY_dU, dZ_dU, dX_dV, dY_dV, dZ_dV, dXdw, dYdw, dZdw


    def metric_tensor_func(self, U, V, W, dX_dU=None, dY_dU=None, dZ_dU=None, 
                        dX_dV=None, dY_dV=None, dZ_dV=None,
                        dX_dW=None, dY_dW=None, dZ_dW=None):
        """
        Enhanced metric tensor function for 3D geometry, calculated adaptively using partial derivatives.
        
        Args:
            U, V, W (torch.Tensor): Grids for parameter space.
            Partial derivatives dX_dU, dY_dU, ..., dZ_dW (torch.Tensor): Optional precomputed partial derivatives.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Metric tensor (g_ij), its inverse (g_inv), and determinant (det_g).
        """
        # Compute partial derivatives if not provided
        dX_dU, dY_dU, dZ_dU, dX_dV, dY_dV, dZ_dV, dX_dW, dY_dW, dZ_dW = self.get_or_compute_partials(U, V, W)

        # Metric tensor components: g_ij = dot product of partials
        g_uu = dX_dU**2 + dY_dU**2 + dZ_dU**2
        g_vv = dX_dV**2 + dY_dV**2 + dZ_dV**2
        g_ww = dX_dW**2 + dY_dW**2 + dZ_dW**2
        g_uv = dX_dU * dX_dV + dY_dU * dY_dV + dZ_dU * dZ_dV
        g_uw = dX_dU * dX_dW + dY_dU * dY_dW + dZ_dU * dZ_dW
        g_vw = dX_dV * dX_dW + dY_dV * dY_dW + dZ_dV * dZ_dW

        # Stack into a symmetric 3x3 metric tensor
        g_ij = torch.stack([
            torch.stack([g_uu, g_uv, g_uw], dim=-1),
            torch.stack([g_uv, g_vv, g_vw], dim=-1),
            torch.stack([g_uw, g_vw, g_ww], dim=-1)
        ], dim=-2)

        # Determinant of the 3x3 metric tensor
        det_g = (g_uu * (g_vv * g_ww - g_vw**2) - 
                g_uv * (g_uv * g_ww - g_vw * g_uw) +
                g_uw * (g_uv * g_vw - g_vv * g_uw))
        det_g = torch.clamp(det_g, min=1e-6)  # Avoid singularities

        # Inverse metric tensor g_inv using explicit formula for 3x3 matrices
        g_inv = torch.zeros_like(g_ij)
        g_inv[..., 0, 0] = (g_vv * g_ww - g_vw**2) / det_g
        g_inv[..., 0, 1] = (g_uw * g_vw - g_uv * g_ww) / det_g
        g_inv[..., 0, 2] = (g_uv * g_vw - g_uw * g_vv) / det_g
        g_inv[..., 1, 0] = g_inv[..., 0, 1]
        g_inv[..., 1, 1] = (g_uu * g_ww - g_uw**2) / det_g
        g_inv[..., 1, 2] = (g_uw * g_uv - g_uu * g_vw) / det_g
        g_inv[..., 2, 0] = g_inv[..., 0, 2]
        g_inv[..., 2, 1] = g_inv[..., 1, 2]
        g_inv[..., 2, 2] = (g_uu * g_vv - g_uv**2) / det_g

        return g_ij, g_inv, det_g

    def transform_spatial(self, U, V, W):
        raise NotImplementedError("Subclasses must implement the transform_spatial method.")

def validate_transform_hub():
    # Simple validation: identity transform
    class IdentityTransform(TransformHub):
        def transform_spatial(self, U, V, W):
            return U, V, W

    # Create a small grid and edges for testing
    N = 3
    U_lin = torch.linspace(0, 1, N)
    V_lin = torch.linspace(0, 1, N)
    W_lin = torch.linspace(0, 1, N)
    U, V, W = torch.meshgrid(U_lin, V_lin, W_lin)
    # Flatten to list vertices
    num_vertices = U.numel()
    vertices = torch.arange(num_vertices)

    # Simple edge index: connect each vertex to next in a line for demonstration
    edges = []
    for i in range(num_vertices-1):
        edges.append([i, i+1])
    edge_index = torch.tensor(edges, dtype=torch.long)

    hub = IdentityTransform(1.0, 1.0, (True, True, True, True))
    geometry = hub.calculate_geometry(U, V, W, edge_index=edge_index, detect_faces=True)

    # Validate that each edge has zero net incidence.
    # In a proper incidence matrix every row should sum to zero
    # because edges connect exactly two vertices with opposite
    # orientation.  This property ensures that a constant 0-form
    # lies in the kernel of d0.
    d0 = geometry["DEC"]["d_operators"]["d0"]
    row_sum = d0.sum(dim=1)
    assert torch.allclose(
        row_sum, torch.zeros_like(row_sum), atol=1e-8
    ), "Edge incidences should sum to zero."

    print("Validation successful. Edge incidences sum to zero and faces detected (if any).")

if __name__ == "__main__":
    validate_transform_hub()
def unpack_values(returned_values, n_desired):
    """
    Unpack the returned values and ensure that the output has exactly n_desired elements.
    If fewer values are returned, fill the remaining with None.
    If more values are returned, truncate to n_desired elements.

    Args:
    returned_values: The tuple or list of returned values.
    n_desired: The number of desired return values.

    Returns:
    A tuple of length n_desired with values or None.
    """
    return (returned_values + (None,) * n_desired)[:n_desired]
import torch
import numpy as np

class PeriodicLinspace:
    def __init__(self, min_density=0.5, max_density=1.5, num_oscillations=1):
        self.min_density = min_density
        self.max_density = max_density
        self.num_oscillations = num_oscillations

    def sin(self, normalized_i):
        return self._oscillate(torch.sin, normalized_i)

    def cos(self, normalized_i):
        return self._oscillate(torch.cos, normalized_i)

    def tan(self, normalized_i):
        density = self._oscillate(torch.tan, normalized_i)
        return torch.clamp(density, min=self.min_density, max=self.max_density)

    def cot(self, normalized_i):
        density = self._oscillate(lambda x: 1 / torch.tan(x + 1e-6), normalized_i)
        return torch.clamp(density, min=self.min_density, max=self.max_density)

    def exp_sin(self, normalized_i):
        density = self._oscillate(lambda x: torch.exp(torch.sin(x)), normalized_i)
        return torch.clamp(density, min=self.min_density, max=self.max_density)

    def exp_cos(self, normalized_i):
        density = self._oscillate(lambda x: torch.exp(torch.cos(x)), normalized_i)
        return torch.clamp(density, min=self.min_density, max=self.max_density)

    def _oscillate(self, func, normalized_i):
        phase_shifted_i = 2 * np.pi * self.num_oscillations * normalized_i - np.pi / 2
        return self.min_density + (self.max_density - self.min_density) * 0.5 * (1 + func(phase_shifted_i))

    def get_density(self, normalized_i, oscillation_type):
        if not hasattr(self, oscillation_type):
            raise ValueError(f"Unknown oscillation_type: '{oscillation_type}'.")
        return getattr(self, oscillation_type)(normalized_i)
import torch
import numpy as np

class GridDomain:
    def __init__(self, U, V, W, u_mode=None, u_p=1, v_mode=None, v_p=1, w_mode=None, w_p=1,
                 Lx=1, Ly=1, Lz=1, grid_boundaries=(True, True, True, True, True, True),
                 transform=None, coordinate_system="rectangular"):
        """
        Initializes the GridDomain object for 3D grids.

        Args:
            U, V, W: Meshgrids representing the three axes.
            u_mode, v_mode, w_mode: Modes for grid generation.
            u_p, v_p, w_p: Parameters for grid generation.
            Lx, Ly, Lz: Physical extents in each direction.
            grid_boundaries: Tuple indicating boundary inclusions for each axis.
            transform: Transformation object.
            coordinate_system: Type of coordinate system.
        """
        # Store U, V, W as the meshgrids that represent the actual domain
        self.U = U
        self.V = V
        self.W = W
        self.transform = transform
        self.vertices = self.transform.transform(U, V, W)  # Pass W here

        self.u_mode = u_mode
        self.v_mode = v_mode
        self.w_mode = w_mode

        self.u_p = u_p
        self.v_p = v_p
        self.w_p = w_p

        # Store physical extents
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        # Store boundary conditions
        self.grid_boundaries = grid_boundaries
        self.coordinate_system = coordinate_system

        # Step 1: Calculate resolution (number of points in U, V, W directions)
        self.resolution_u = U.shape[0]  # Number of points along the U axis
        self.resolution_v = V.shape[1]  # Number of points along the V axis
        self.resolution_w = W.shape[2]  # Number of points along the W axis

        # Step 2: Calculate extents (total span of the domain along U, V, W)
        self.extent_u = U.max() - U.min()  # Extent along the U axis
        self.extent_v = V.max() - V.min()  # Extent along the V axis
        self.extent_w = W.max() - W.min()  # Extent along the W axis

        # Step 3: Compute normalized U, V, W for grid_sample interpolation (from -1 to 1)
        self.normalized_U = self.normalize_grid(self.U, self.extent_u)
        self.normalized_V = self.normalize_grid(self.V, self.extent_v)
        self.normalized_W = self.normalize_grid(self.W, self.extent_w)

        # Step 4: Create a combined normalized grid for interpolation
        self.normalized_grid = torch.stack([self.normalized_U, self.normalized_V, self.normalized_W], dim=-1).unsqueeze(0)

    @staticmethod
    def generate_grid_domain(coordinate_system, N_u, N_v, N_w, u_mode=None, v_mode=None, w_mode=None,
                            device='cpu', precision=torch.float64, **kwargs):
        """
        Generates a GridDomain object based on the coordinate system and its parameters.
        """
        # Set default modes if u_mode, v_mode, or w_mode not provided
        u_mode = u_mode or {'method': 'linear', 'p': 1}
        v_mode = v_mode or {'method': 'linear', 'p': 1}
        w_mode = w_mode or {'method': 'linear', 'p': 1}

        # Combine all the parameters that are common across different transforms
        transform_params = {
            'N_u': N_u,
            'N_v': N_v,
            'N_w': N_w,
            'device': device,
            'precision': precision,
            'Lx': kwargs.get('Lx', None),
            'Ly': kwargs.get('Ly', None),
            'Lz': kwargs.get('Lz', None)
        }

        # Update with any additional kwargs that may be provided
        transform_params.update(kwargs)

        # Create the appropriate Transform instance with all packed parameters
        transform = Transform.create_transform(coordinate_system, **transform_params)

        U, V, W = None, None, None
        # Get domain extents and grid boundaries from the Transform
        (uextent, vextent, wextent), grid_boundaries = transform.get_transform_parameters()

        if getattr(transform, 'autogrid', False):
            U, V, W = transform.obtain_autogrid()
        else:
            # Generate u_grid, v_grid, w_grid using the respective mode dictionaries
            u_grid, _ = generate_grid(
                N=N_u, L=uextent, device=device, dtype=precision,
                keep_end=grid_boundaries[1], **u_mode
            )
            v_grid, _ = generate_grid(
                N=N_v, L=vextent, device=device, dtype=precision,
                keep_end=grid_boundaries[3], **v_mode
            )
            w_grid, _ = generate_grid(
                N=N_w, L=wextent, device=device, dtype=precision,
                keep_end=grid_boundaries[5], **w_mode
            )

            # Create U, V, W meshgrids
            U, V, W = torch.meshgrid(u_grid, v_grid, w_grid, indexing='ij')

        # Create and return the GridDomain
        return GridDomain(U, V, W, transform=transform, coordinate_system=coordinate_system)


    def apply_transform(self):
        return self.transform(self.U, self.V)
    
    def get_vertices(self):
        return self.vertices
    def return_dense_copy(self, scaling_factor=(2,2), normalize=True):
        """
        Returns a higher resolution copy of the grid (U, V) by scaling the resolution.
        The dense grid will use the same distribution method as the original grid.

        Args:
            scaling_factor: The factor by which to increase the resolution.
            normalize: Whether to return the normalized grid.

        Returns:
            A dense version of U and V (or normalized versions if normalize=True).
        """
        print("DEPRECIATED")
        
        # New high-resolution by scaling
        high_res_u = int(self.resolution_u * scaling_factor[0])
        high_res_v = int(self.resolution_v * scaling_factor[1])

        # Create high-resolution U and V grids by regenerating the grid using the same method and parameters
        u_high_res, _ = generate_grid(
            N=high_res_u,
            L=self.U.max(),
            device=self.U.device,  # Keep the device same as the original grid
            dtype=self.U.dtype,
            **self.u_mode
        )

        v_high_res, _ = generate_grid(
            N=high_res_v,
            L=self.V.max(),
            device=self.V.device,
            dtype=self.V.dtype,
            **self.v_mode
        )

        # Create the high-resolution meshgrid
        U_high_res, V_high_res = torch.meshgrid(u_high_res, v_high_res, indexing='ij')

        # Normalize if required
        if normalize:
            U_norm_high = self.normalize_grid(U_high_res, self.extent_u)
            V_norm_high = self.normalize_grid(V_high_res, self.extent_v)
            return U_norm_high, V_norm_high
        else:
            return U_high_res, V_high_res



    def normalize_grid(self, grid, extent):
        """
        Normalizes the grid values into the range [-1, 1] for use in grid_sample.

        Args:
            grid: The original meshgrid (U or V).
            extent: The span (max - min) of the grid.
        
        Returns:
            The normalized grid in the range [-1, 1].
        """
        grid_min = grid.min()
        normalized_grid = (grid - grid_min) / extent.to(grid.device) * 2 - 1  # Normalize to [-1, 1]
        return normalized_grid

    def summary(self):
        """
        Returns a summary of the grid domain, including resolution and extents.
        
        Returns:
            dict: A dictionary containing resolution and extents of the grid.
        """
        return {
            "resolution_u": self.resolution_u,
            "resolution_v": self.resolution_v,
            "extent_u": self.extent_u.item(),  # Convert from tensor to Python float
            "extent_v": self.extent_v.item(),
            "normalized_U_range": (self.normalized_U.min().item(), self.normalized_U.max().item()),
            "normalized_V_range": (self.normalized_V.min().item(), self.normalized_V.max().item())
        }
    def __getstate__(self):
        """
        Prepares the state for serialization. Convert tensors to CPU (to avoid issues with GPU tensors) and
        store other attributes.
        """
        state = self.__dict__.copy()
        state['U'] = self.U.cpu()  # Ensure tensors are on the CPU for serialization
        state['V'] = self.V.cpu()
        state['normalized_U'] = self.normalized_U.cpu()
        state['normalized_V'] = self.normalized_V.cpu()
        state['normalized_grid'] = self.normalized_grid.cpu()
        state['coordinate_system']

        # Serialize any callable objects like transform if needed (could be done through another method or logic)
        state['transform'] = None  # Set transform to None or serialize it if required

        return state
    def __setstate__(self, state):
        """
        Restores the object state after deserialization. Move tensors back to the appropriate device and
        regenerate the transform.
        """
        # Restore the state dictionary
        self.__dict__.update(state)

        # Move tensors back to the appropriate device (you can modify the device as needed)
        self.U = self.U.to('cpu')
        self.V = self.V.to('cpu')
        self.normalized_U = self.normalized_U.to('cpu')
        self.normalized_V = self.normalized_V.to('cpu')
        self.normalized_grid = self.normalized_grid.to('cpu')

        # Regenerate the transform based on the coordinate system and stored parameters
        self.transform = self.regenerate_transform()

    def regenerate_transform(self):
        """
        Recreates the transform using stored parameters such as the coordinate system, resolution, boundary conditions, etc.
        This method is called during deserialization to restore the transform.
        """
        # Example of how you might regenerate the transform based on the stored parameters
        transform_params = {
            'N_u': self.resolution_u,
            'N_v': self.resolution_v,
            'u_mode': self.u_mode,
            'v_mode': self.v_mode,
            'device': 'cpu',  # Adjust device if needed
            'precision': self.U.dtype,  # Use the same dtype as the original grid
            'Lx': self.Lx,
            'Ly': self.Ly
        }

        # Call the same logic used during GridDomain initialization
        transform = Transform.create_transform(self.coordinate_system, **transform_params)
        return transform
    
def generate_grid(N, L, method='linear', p=2.0, min_density=0.5,
                  max_density=1.5, num_oscillations=1, keep_end=True, periodic=False,
                  oscillation_type='sin', device='cpu', dtype=torch.float64):
    """
    Generates a grid with various spacing methods and calculates infinitesimal values.
    
    Parameters are the same as before, with `oscillation_type` specifying
    the desired periodic pattern if `method` is 'periodic'.
    """
    if N < 2:
        raise ValueError("N must be at least 2.")

    # Generate indices and normalize
    i = torch.arange(0, N, device=device, dtype=dtype)
    normalized_i = i / (N - 1 if keep_end else N)

    if method == 'linear':
        grid = L * normalized_i
    elif method == 'non_uniform':
        grid = L * normalized_i ** p
    elif method == 'inverted':
        grid = L * (1 - (1 - normalized_i) ** p)
    elif method == 'periodic':
        # Use PeriodicLinspace for density modulation
        periodic_gen = PeriodicLinspace(min_density, max_density, num_oscillations)
        density = periodic_gen.get_density(normalized_i, oscillation_type)
        grid = torch.cumsum(density, dim=0)
        grid = grid / grid[-1] * L  # Normalize to fit within length L
    elif method == 'dense_extremes':
        grid = L * 0.5 * (normalized_i ** p + (1 - (1 - normalized_i) ** p))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear', 'non_uniform', 'inverted', 'periodic', or 'dense_extremes'.")

    # Compute infinitesimal values
    infinitesimal = torch.zeros(N, device=device, dtype=dtype)
    infinitesimal[:-1] = grid[1:] - grid[:-1]

    if not keep_end:
        infinitesimal[-1] = L - grid[-1] + grid[0]  # Wrap-around interval for periodic case

    return grid, infinitesimal

def generate_full_meshgrid(N_u, L_u, N_v, L_v, N_w, L_w, periodic_u=True, periodic_v=True, periodic_w=True,
                           umethod='dense_extremes', upow=2, vmethod="dense_extremes", vpow=2, wmethod='dense_extremes', wpow=2,
                           device='cpu', **kwargs):
    """
    Generate U, V, W meshgrids and their corresponding infinitesimal grids U', V', W'.
    """
    
    # Generate U, V, W grids
    U, U_prime = generate_grid(N_u, L_u, method=umethod, p=upow, periodic=periodic_u, keep_end=not periodic_u, device=device, **kwargs)
    V, V_prime = generate_grid(N_v, L_v, method=vmethod, p=vpow, periodic=periodic_v, keep_end=not periodic_v, device=device, **kwargs)
    W, W_prime = generate_grid(N_w, L_w, method=wmethod, p=wpow, periodic=periodic_w, keep_end=not periodic_w, device=device, **kwargs)
    
    # Create full 3D meshgrid for U, V, W
    U_mesh, V_mesh, W_mesh = torch.meshgrid(U, V, W, indexing='ij')
    
    # Create full 3D meshgrid for infinitesimal U', V', W'
    U_prime_mesh, V_prime_mesh, W_prime_mesh = torch.meshgrid(U_prime, V_prime, W_prime, indexing='ij')

    return U_mesh, V_mesh, W_mesh, U_prime_mesh, V_prime_mesh, W_prime_mesh

class Transform(TransformHub):
    def __init__(self, uextent, vextent, grid_boundaries):
        super().__init__(uextent, vextent, grid_boundaries)

    def get_transform_parameters(self):
        return (self.uextent, self.vextent), self.grid_boundaries

    def transform(self, U, V, W, use_metric=False):
        """
        Transform coordinates using either spatial or metric transformation.

        Args:
            U, V (torch.Tensor): Parameter grids.
            use_metric (bool): Whether to use the metric transformation.

        Returns:
            tuple: Transformed coordinates or metric data.
        """
        self.device = U.device
        geometry = self.calculate_geometry(U, V, W)
        return geometry["metric"] if use_metric else geometry["coordinates"]

    def convert_data_2d_to_3d(self, data_2d, use_metric=False):
        """
        Convert 2D parameter data to 3D coordinates and prepare for rendering.

        Args:
            data_2d (torch.Tensor): Stacked 2D data in U, V parameter space.
            use_metric (bool): Whether to use the metric transformation.

        Returns:
            Tuple of vertices, indices, normals, and data for OpenGL rendering.
        """
        resolution_u, resolution_v = data_2d.shape[-2], data_2d.shape[-1]
        # Generate U, V grid
        U, V = self.create_grid_mesh2(resolution_u, resolution_v)

        # Retrieve geometry (coordinates and optionally metric tensor)
        geometry_data = self.calculate_geometry(U, V)
        X, Y, Z = geometry_data["coordinates"]

        # Prepare vertices
        vertices = self.prepare_mesh_for_rendering(X, Y, Z)

        # Generate indices for triangulation (for OpenGL)
        indices = self.generate_triangle_indices(resolution_u, resolution_v)

        # Calculate normals for rendering
        normals = geometry_data["normals"]

        # Flatten data for rendering compatibility
        if data_2d.ndimension() == 2:
            data_3d = data_2d.flatten()
        else:
            data_3d = torch.stack([data_2d[i].flatten() for i in range(data_2d.shape[0])])

        return vertices, indices, normals, data_3d

    def create_grid_mesh2(self, resolution_u=100, resolution_v=100):
        if getattr(self, "autogrid", False):
            return self.obtain_autogrid()
        else:
            u_values = torch.linspace(0, self.uextent, resolution_u)
            v_values = torch.linspace(0, self.vextent, resolution_v)
            U, V = torch.meshgrid(u_values, v_values, indexing='ij')
            return U, V

    
    def create_grid_mesh(self, resolution_u, resolution_v, resolution_w):
        # Derive periodicity based on endpoint exclusion in grid boundaries
        periodic_u = not (self.grid_boundaries[0] and self.grid_boundaries[1])  # True if either endpoint is excluded for U
        periodic_v = not (self.grid_boundaries[2] and self.grid_boundaries[3])  # True if either endpoint is excluded for V
        periodic_w = not (self.grid_boundaries[4] and self.grid_boundaries[5])

        # Use generate_full_meshgrid with inferred periodicity
        U_mesh, V_mesh, W_mesh, U_prime_mesh, V_prime_mesh, W_prime_mesh= generate_full_meshgrid(
            N_u=resolution_u,
            L_u=self.uextent,
            N_v=resolution_v,
            L_v=self.vextent,
            N_w=resolution_w,
            L_w=self.wextent,
            periodic_u=periodic_u,
            periodic_v=periodic_v,
            periodic_w=periodic_w,
            device=self.device
        )
        return U_mesh, V_mesh, W_mesh
    def generate_triangle_indices(self, resolution_u, resolution_v):
        periodic_u = not (self.grid_boundaries[0] and self.grid_boundaries[1])  # True if either endpoint is excluded for U
        periodic_v = not (self.grid_boundaries[2] and self.grid_boundaries[3])  # True if either endpoint is excluded for V
        indices = []
        
        for u in range(resolution_u - 1):
            for v in range(resolution_v):
                # Add two vertices for each triangle strip
                indices.append(u * resolution_v + v)         # Current vertex
                indices.append((u + 1) * resolution_v + v)   # Vertex directly below in U direction
                
                # Connect the last vertex in v to the first if periodic_v
                if periodic_v and v == resolution_v - 1:
                    indices.append(u * resolution_v)              # Wrap to first column in current row
                    indices.append((u + 1) * resolution_v)        # Wrap to first column in next row

        # Connect the last row back to the first if periodic_u
        if periodic_u:
            for v in range(resolution_v):
                indices.append((resolution_u - 1) * resolution_v + v)   # Last row current column
                indices.append(v)                                       # First row current column
                
                # Handle periodicity in both dimensions at the corner
                if periodic_v and v == resolution_v - 1:
                    indices.append((resolution_u - 1) * resolution_v)   # Last row, first column
                    indices.append(0)                                   # First row, first column
                    
        return torch.tensor(indices, dtype=torch.int32)


    def prepare_mesh_for_rendering(self, X, Y, Z):
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        return torch.stack([X_flat, Y_flat, Z_flat], dim=-1)

    @classmethod
    def create_transform(cls, type_of_transform, **kwargs):
        #metric_transform = MetricTransform.create_metric_transform(type_of_transform, **kwargs)
        #if metric_transform is not None:
        #    return metric_transform

        transform_map = {
            "rectangular": RectangularTransform,

        }

        if type_of_transform in transform_map:
            return transform_map[type_of_transform](**kwargs)
        else:
            raise ValueError(f"Unsupported transform type '{type_of_transform}'")



class RectangularTransform(Transform):
    def __init__(self, Lx, Ly, Lz, device='cpu', N_u=None, N_v=None, N_w=None, u_mode=None, v_mode=None, w_mode=None, precision=None):
        """
        Initialize the RectangularTransform with extents and optional grid modes.
        """
        self.uextent = self.Lx = Lx 
        self.vextent = self.Ly = Ly 
        self.wextent = self.Lz = Lz 
        self.N_x = N_u
        self.N_y = N_v
        self.N_w = N_w
        self.device = device
        self.u_mode = u_mode
        self.v_mode = v_mode
        self.w_mode = w_mode
        self.grid_boundaries=(True, True, True, True, True, True)

    def get_transform_parameters(self):
        """
        Return extents for U, V, W and grid boundaries.
        """
        uextent = self.Lx
        vextent = self.Ly
        wextent = self.Lz
        grid_boundaries = (True, True, True, True, True, True)  # Example boundaries for 3D
        return (uextent, vextent, wextent), grid_boundaries

    def transform_spatial(self, grid_u, grid_v, grid_w):
        """
        Identity transform for simplicity.
        """
        return grid_u, grid_v, grid_w


def test_build_laplace3d():
    # Grid parameters
    Lx, Ly, Lz = 1.0, 1.0, 1.0  # Unit cube
    N_u, N_v, N_w = 20, 20, 20  # Grid resolution
    device = 'cpu'  # Change to 'cpu' if using GPU

    # Initialize transformation and grid domain
    transform = RectangularTransform(Lx=Lx, Ly=Ly, Lz=Lz, device=device)
    grid_u, grid_v, grid_w = transform.create_grid_mesh(N_u, N_v, N_w)
    print(f"Grid U shape: {grid_u.shape}")
    print(f"Grid V shape: {grid_v.shape}")
    print(f"Grid W shape: {grid_w.shape}")
    
    grid_domain = GridDomain.generate_grid_domain(
        coordinate_system='rectangular',
        N_u=N_u,
        N_v=N_v,
        N_w=N_w,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        device=device
    )

    # Initialize BuildLaplace3D with Dirichlet boundary conditions
    # All boundaries are Dirichlet
    boundary_conditions = ('dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet')
    # boundary_conditions = ('neumann', 'neumann', 'neumann', 'neumann', 'neumann', 'neumann')

    build_laplace = BuildLaplace3D(
        grid_domain=grid_domain,
        wave_speed=343,  # Arbitrary value
        precision=torch.float64,
        resolution=20,  # Should match N_u, N_v, N_w
        metric_tensor_func=None,  # Use default Euclidean metric
        density_func=None,        # Uniform density
        tension_func=None,        # Uniform tension
        singularity_conditions=None,
        boundary_conditions=boundary_conditions,
        artificial_stability=1e-10
    )

    # Build the Laplacian
    laplacian_tensor, laplacian_sparse = build_laplace.build_general_laplace(
        grid_u=grid_u,
        grid_v=grid_v,
        grid_w=grid_w,
        boundary_conditions=boundary_conditions,
        grid_boundaries=(True, True, True, True, True, True),
        device=device,
        f=0.0  # No wave number term for this test
    )

    # Define the test function f(x,y,z) = sin(pi x) sin(pi y) sin(pi z)
    X = grid_u.numpy()
    Y = grid_v.numpy()
    Z = grid_w.numpy()
    f = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    f_flat = f.flatten()

    # Convert f_flat to a Torch tensor
    f_tensor = torch.tensor(f_flat, dtype=torch.float64, device=device)

    # Apply the Laplacian matrix to f
    if laplacian_tensor is not None:
        # Dense Laplacian
        laplace_f_numerical = -laplacian_tensor @ f_tensor
    else:
        # Sparse Laplacian
        # Convert sparse matrix to Torch sparse tensor
        indices = torch.tensor([laplacian_sparse.row, laplacian_sparse.col], dtype=torch.long)
        values = torch.tensor(laplacian_sparse.data, dtype=torch.float64)
        laplacian_sparse_torch = torch.sparse_coo_tensor(indices, values, size=(N_u * N_v * N_w, N_u * N_v * N_w)).to(device)
        laplace_f_numerical = torch.sparse.mm(laplacian_sparse_torch, f_tensor.unsqueeze(1)).squeeze(1)

    # Compute the analytical Laplacian: -3 pi^2 f
    laplace_f_analytical = -3 * (math.pi ** 2) * f_flat
    # Note: Do NOT scale the analytical Laplacian

    # Convert laplace_f_numerical to numpy for comparison
    laplace_f_numerical_np = laplace_f_numerical.cpu().numpy()

    # Compute the error
    error = laplace_f_numerical_np/min(laplace_f_numerical_np) - laplace_f_analytical/min(laplace_f_analytical)
    max_error = np.max(np.abs(error))
    mean_error = np.mean(np.abs(error))

    print(f"Max Error: {max_error:.6e}")
    print(f"Mean Error: {mean_error:.6e}")

    # Visualization (optional)
    # Compare a central slice
    central_slice = N_w // 2
    laplace_f_numerical_reshaped = laplace_f_numerical_np.reshape(N_u, N_v, N_w)
    laplace_f_analytical_reshaped = laplace_f_analytical.reshape(N_u, N_v, N_w)
    error_reshaped = error.reshape(N_u, N_v, N_w)

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.title("Numerical Laplacian")
    plt.imshow(laplace_f_numerical_reshaped[:, :, central_slice], origin='lower', extent=[0, Lx, 0, Ly])
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Analytical Laplacian")
    plt.imshow(laplace_f_analytical_reshaped[:, :, central_slice], origin='lower', extent=[0, Lx, 0, Ly])
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Error")
    plt.imshow(error_reshaped[:, :, central_slice], origin='lower', extent=[0, Lx, 0, Ly], cmap='bwr')
    plt.colorbar()

    plt.show()

    # Assert that the error is within an acceptable tolerance
    # Adjust tolerance based on grid resolution and discretization
    # For N=20, the grid is relatively coarse, so a larger tolerance is acceptable
    tolerance = 1e-1  # Adjusted tolerance for coarser grid
    assert max_error < tolerance, f"Max error {max_error} exceeds tolerance {tolerance}"
    print("Test passed: Numerical Laplacian matches analytical Laplacian within tolerance.")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def heat_evolution_demo(laplacian_tensor, initial_temperature, alpha=0.01, dt=0.01, steps=500):
    """
    Simulates heat evolution on a 3D grid using a precomputed Laplacian matrix.
    
    Args:
        laplacian_tensor: A dense Laplacian matrix (torch.Tensor or numpy.ndarray).
        initial_temperature: Initial temperature distribution (numpy.ndarray).
        alpha: Thermal diffusivity coefficient (float).
        dt: Time step for simulation (float).
        steps: Number of simulation steps (int).
    """
    # Convert Laplacian tensor to NumPy if needed
    if isinstance(laplacian_tensor, torch.Tensor):
        laplacian_tensor = laplacian_tensor.cpu().numpy()

    # Flatten the initial temperature distribution
    initial_temperature = initial_temperature.flatten()

    # Time evolution of temperature
    temperature = initial_temperature.copy()
    
    # Reshape dimensions for visualization
    N_u = int(np.cbrt(len(temperature)))  # Assuming a cubic grid
    temperature_grid = temperature.reshape(N_u, N_u, N_u)

    # Prepare the plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    x = np.linspace(0, 1, N_u)
    y = np.linspace(0, 1, N_u)
    X, Y = np.meshgrid(x, y)
    Z = temperature_grid[:, :, N_u // 2]  # Middle slice for 3D visualization
    surface = ax.plot_surface(X, Y, Z, cmap='viridis')

    

    def update(frame):
        nonlocal temperature, temperature_grid, Z, surface
        # Compute temperature evolution: T_new = T + alpha * dt * Laplacian(T)
        temperature += -alpha * dt * laplacian_tensor @ temperature
        temperature_grid = temperature.reshape(N_u, N_u, N_u)
        Z = temperature_grid[:, :, N_u // 2]  # Update the middle slice
        
        # Remove the previous surface if it exists
        if surface is not None:
            surface.remove()
        
        # Replot the surface
        surface = ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title(f"Time Step: {frame}")
        ax.set_zlim(0, initial_temperature.max())  # Adjust limits dynamically
        return surface,


    # Animate
    ani = FuncAnimation(fig, update, frames=steps, blit=False, interval=50)

    # Show the animation
    plt.show()
# Store the surface plot reference globally
surface = None

# Example usage with a random initial condition
if __name__ == "__main__":
    # Parameters
    N_u, N_v, N_w = 20, 20, 20  # Grid resolution
    Lx, Ly, Lz = 1.0, 1.0, 1.0  # Domain size

    # Create a Laplacian matrix using the earlier code
    transform = RectangularTransform(Lx=Lx, Ly=Ly, Lz=Lz, device='cpu')
    grid_u, grid_v, grid_w = transform.create_grid_mesh(N_u, N_v, N_w)
    grid_domain = GridDomain.generate_grid_domain(
        coordinate_system='rectangular',
        N_u=N_u,
        N_v=N_v,
        N_w=N_w,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        device='cpu'
    )
    boundary_conditions = ('dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet')

    build_laplace = BuildLaplace3D(
        grid_domain=grid_domain,
        wave_speed=343,  # Arbitrary value
        precision=torch.float64,
        resolution=N_u,  # Should match N_u, N_v, N_w
        metric_tensor_func=None,  # Use default Euclidean metric
        density_func=None,        # Uniform density
        tension_func=None,        # Uniform tension
        singularity_conditions=None,
        boundary_conditions=boundary_conditions,
        artificial_stability=1e-10
    )
    laplacian_tensor, _ = build_laplace.build_general_laplace(
        grid_u=grid_u,
        grid_v=grid_v,
        grid_w=grid_w,
        boundary_conditions=boundary_conditions,
        grid_boundaries=(True, True, True, True, True, True),
        device='cpu',
        dense=True,
        f=0.0
    )

    # Initial temperature: Gaussian in the center
    x = np.linspace(0, 1, N_u)
    y = np.linspace(0, 1, N_v)
    z = np.linspace(0, 1, N_w)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    initial_temperature = np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2))

    # Run the demo
    heat_evolution_demo(laplacian_tensor, initial_temperature, alpha=0.01, dt=1.1, steps=200)
