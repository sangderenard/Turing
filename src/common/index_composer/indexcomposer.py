import torch

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
    def __init__(self):
        # Map dimension characters to axis indices
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
        """
        Generates row/column indices and masks for neighbor relationships, 
        returning pattern-specific boundary masks.

        Args:
            grid_shape (tuple): Shape of the grid.
            patterns (list): List of pattern dictionaries.

        Returns:
            dict: Contains row indices, column indices, masks for each pattern, 
                and boundary masks specific to each pattern.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ndim = len(grid_shape)

        # Grid indices and flat index grid
        grid_indices = torch.meshgrid(
            *[torch.arange(s, device=device) for s in grid_shape], indexing='ij'
        )
        flat_indices = torch.arange(torch.prod(torch.tensor(grid_shape))).reshape(grid_shape).to(device)

        results = {'row_indices': {}, 'col_indices': {}, 'masks': {}, 'boundary_masks': {}}

        for pattern in patterns:
            label = pattern['label']
            offset = pattern['offset']
            periodic = pattern['periodic']

            # Compute neighbor indices
            neighbor_indices = []
            if boundary_mask is None:
                boundary_mask = torch.zeros(grid_shape, dtype=torch.bool, device=device)

            for dim in range(ndim):
                new_index = grid_indices[dim] + offset[dim]

                if periodic[dim]:
                    new_index = new_index % grid_shape[dim]
                else:
                    # Identify boundary regions only on one side per offset direction
                    if offset[dim] > 0:
                        boundary_upper = (grid_indices[dim] + offset[dim]) >= grid_shape[dim]
                        boundary_mask |= boundary_upper
                    elif offset[dim] < 0:
                        boundary_lower = (grid_indices[dim] + offset[dim]) < 0
                        boundary_mask |= boundary_lower

                    new_index = torch.clamp(new_index, 0, grid_shape[dim] - 1)

                neighbor_indices.append(new_index)

            # Compute valid mask (excluding boundary contributions for this pattern)
            valid_mask = ~boundary_mask

            # Compute row and column indices
            row_indices = flat_indices[valid_mask]
            col_indices = flat_indices[tuple(neighbor_indices)][valid_mask]

            # Store results
            results['row_indices'][label] = row_indices
            results['col_indices'][label] = col_indices
            results['masks'][label] = valid_mask
            results['boundary_masks'][label] = boundary_mask  # Pattern-specific boundary mask

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
validation = GeneralIndexComposer()

validation.validate((3,3,3))