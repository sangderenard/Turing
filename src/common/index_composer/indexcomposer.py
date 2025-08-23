from typing import Optional
import math

from ..tensors.abstraction import AbstractTensor

class GeneralIndexComposer:
    """
    A general-purpose index composer for sparse tensor assembly.
    Handles periodic and non-periodic neighbor relationships implicitly.

    Methods:
        interpret_label(label): Parses compact labels into offsets and periodicity flags.
        generate_patterns(labels): Converts compact labels into pattern dictionaries.
        compose_indices(grid_shape, patterns): Produces row/col indices and masks for all patterns.
        validate(grid_shape): Verifies the correctness on a small known example grid.
    """
    def __init__(self, device: Optional[str] = None):
        self.device = device or "cpu"
        self.dimension_map = {'u': 0, 'v': 1, 'w': 2}

    def interpret_label(self, label):
        """
        Interprets a label to generate offsets and periodicity flags.
        Applies the first sign encountered (from the right) to all 
        preceding dimensions.

        Args:
            label (str): Compact label ('u+', 'U-', 'uv+2', etc.).

        Returns:
            tuple: (offset, periodicity_flags)
        """
        ndim = len(self.dimension_map)
        offset = [0] * ndim
        periodicity_flags = [False] * ndim

        direction = 1  # Default direction
        current_stride = 1  # Default stride

        for char in reversed(label):  # Iterate from right to left
            if char in ('+', '-'):
                direction = 1 if char == '+' else -1
                current_stride = 1  # Reset stride after a sign
            elif char.isdigit():
                current_stride = int(char)  # Update stride
            else:
                dim_idx = self.dimension_map[char.lower()]
                offset[dim_idx] = direction * current_stride
                periodicity_flags[dim_idx] = char.isupper()

        return tuple(offset), tuple(periodicity_flags)

    def generate_patterns(self, labels):
        """
        Converts a list of compact labels into pattern dictionaries.

        Args:
            labels (list): List of compact labels (e.g., ['u+', 'U-', 'uv+']).

        Returns:
            list: List of pattern dictionaries with 'label', 'offset', and 'periodic'.
        """
        patterns = []
        for label in labels:
            offset, periodic = self.interpret_label(label)
            patterns.append({'label': label, 'offset': offset, 'periodic': periodic})
        return patterns

    def compose_indices(self, grid_shape, patterns, boundary_mask=None):
        device = self.device
        ndim = len(grid_shape)

        # float tensor just to pull backend + float dtype if needed later
        test_tensor = AbstractTensor.zeros(grid_shape, dtype=AbstractTensor.float_dtype_, device=device)
        ops = type(test_tensor)

        # Always use an *instance-derived* integer dtype (falls back to AbstractTensor)
        int_dtype = getattr(test_tensor, "long_dtype_", getattr(AbstractTensor, "long_dtype_", None))
        if int_dtype is None:
            raise RuntimeError("Could not resolve long/integer dtype for current backend.")

        # Build integer coordinate grids for safe advanced indexing
        grid_ranges = [ops.arange(s, dtype=int_dtype, device=device) for s in grid_shape]
        grid_indices = ops.meshgrid(*grid_ranges, indexing='ij')

        total_size = math.prod(grid_shape)
        flat_indices = ops.arange(total_size, dtype=int_dtype, device=device).reshape(grid_shape)

        results = {'row_indices': {}, 'col_indices': {}, 'masks': {}, 'boundary_masks': {}}

        for pattern in patterns:
            label = pattern['label']
            offset = pattern['offset']
            periodic = pattern['periodic']

            bm_accum = (
                ops.zeros(grid_shape, dtype=int_dtype, device=device)
                if boundary_mask is None
                else boundary_mask.astype(int_dtype)
            )

            neighbor_indices = []
            for dim in range(ndim):
                new_index = grid_indices[dim] + offset[dim]
                if periodic[dim]:
                    new_index = new_index % grid_shape[dim]
                else:
                    base = grid_indices[dim] + offset[dim]
                    if offset[dim] > 0:
                        cond = base >= grid_shape[dim]
                        bm_accum = bm_accum + cond.astype(int_dtype)
                    elif offset[dim] < 0:
                        cond = base < 0
                        bm_accum = bm_accum + cond.astype(int_dtype)
                    new_index = ops.clamp(new_index, min=0, max=grid_shape[dim] - 1)
                neighbor_indices.append(new_index)

            bm = bm_accum.greater(0)
            valid_mask = bm.logical_not()

            row_indices = flat_indices[valid_mask]
            col_indices = flat_indices[tuple(neighbor_indices)][valid_mask]

            results['row_indices'][label] = row_indices
            results['col_indices'][label] = col_indices
            results['masks'][label] = valid_mask
            results['boundary_masks'][label] = bm

        return results


    def validate(self, grid_shape):
        """
        Validates the index composer using a small known example grid.

        Args:
            grid_shape (tuple): Shape of the grid (e.g., (3, 3, 3)).
        """
        labels = ['u+', 'U-', 'v+', 'V-', 'w+', 'W-', 'uv+', 'UV-', 'uW+', 'Uw-']
        patterns = self.generate_patterns(labels)
        results = self.compose_indices(grid_shape, patterns)

        for label in labels:
            print(f"Pattern '{label}':")
            print("Row indices:", results['row_indices'][label])
            print("Col indices:", results['col_indices'][label])
            print("Mask:", results['masks'][label])
            print("------")


if __name__ == "__main__":
    validation = GeneralIndexComposer()
    validation.validate((3, 3, 3))
