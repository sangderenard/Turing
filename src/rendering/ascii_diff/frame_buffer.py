import threading
from src.common.tensors import AbstractTensor
test_tensor = AbstractTensor.get_tensor([0])
integer_type = test_tensor.long_dtype_
class PixelFrameBuffer:
    """Manage pixel data frames for flicker-free terminal rendering.
    Each 'pixel' corresponds to a character cell in the terminal.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        diff_threshold: int = 0,
        *,
        tile_shape: tuple[int, int] = (1, 1),   # NEW: tile/cell size (rows, cols)
    ):
        """
        `diff_threshold`: Sum(|ΔRGB|) needed to mark a pixel as changed (0..765).
        `tile_shape`: (tile_rows, tile_cols). Use (>1,>1) to group diffs to tiles.
        """
        self.lock = threading.Lock()
        self.buffer_shape = (shape[0], shape[1], 3)  # rows, cols, RGB
        self.default_pixel = AbstractTensor.get_tensor([0, 0, 0], dtype=integer_type)  # Black
        self.buffer_render = AbstractTensor.full(self.buffer_shape, 0, dtype=integer_type)
        self.buffer_next = AbstractTensor.full(self.buffer_shape, 0, dtype=integer_type)
        self.buffer_display = AbstractTensor.full(self.buffer_shape, 0, dtype=integer_type)
        self.diff_threshold = max(0, diff_threshold)
        self.tile_shape = (max(1, int(tile_shape[0])), max(1, int(tile_shape[1])))
        self._force_full_diff_next_call = True

    def __repr__(self) -> str:
        return (
            f"PixelFrameBuffer(shape={self.buffer_shape[:2]}, "
            f"diff_threshold={self.diff_threshold}, tile_shape={self.tile_shape})"
        )

    def _resize(self, shape: tuple[int, int]) -> None:
        """Resize internal buffers to ``shape`` without acquiring the lock."""
        self.buffer_shape = (shape[0], shape[1], 3)
        self.buffer_render = AbstractTensor.full(self.buffer_shape, 0, dtype=integer_type)
        self.buffer_next = AbstractTensor.full(self.buffer_shape, 0, dtype=integer_type)
        self.buffer_display = AbstractTensor.full(self.buffer_shape, 0, dtype=integer_type)
        self._force_full_diff_next_call = True

    def resize(self, shape: tuple[int, int]) -> None:
        """Thread-safe wrapper around :meth:`_resize`."""
        with self.lock:
            self._resize(shape)

    def update_render(self, new_data: AbstractTensor) -> None:
        """Update the render buffer with ``new_data`` (rows, cols, 3)."""
        if new_data.shape != self.buffer_shape:
            self.resize((new_data.shape[0], new_data.shape[1]))
        with self.lock:
            AbstractTensor.copyto(self.buffer_render, new_data)

    def force_full_redraw_next_frame(self) -> None:
        """Signals that the next call to get_diff_and_promote should return all pixels/tiles."""
        self._force_full_diff_next_call = True

    # ---- NEW: core diff helpers ----
    def _compute_pixel_diff_mask(self) -> AbstractTensor:
        """Return a boolean mask (rows, cols) where pixels differ vs display."""
        if self.diff_threshold == 0:
            return AbstractTensor.any(self.buffer_next != self.buffer_display, dim=2)
        abs_diff = AbstractTensor.abs(
            self.buffer_next - self.buffer_display
        )
        sum_abs = AbstractTensor.sum(abs_diff, dim=2)
        return sum_abs > self.diff_threshold

    def _mask_to_tile_mask(self, diff_mask: AbstractTensor) -> AbstractTensor:
        """Reduce pixel diff mask to a tile-level mask via ANY within each tile."""
        th, tw = self.tile_shape
        rows, cols = diff_mask.shape
        # Pad to multiples of tile dims (without copying too much)
        pad_r = (th - (rows % th)) % th
        pad_c = (tw - (cols % tw)) % tw
        if pad_r or pad_c:
            diff_padded = AbstractTensor.pad(diff_mask, ((0, pad_r), (0, pad_c)), mode="constant", constant_values=False)
        else:
            diff_padded = diff_mask
        R, C = diff_padded.shape
        ty, tx = R // th, C // tw
        # Reshape to (ty, th, tx, tw) and reduce within each tile
        tile_mask = diff_padded.reshape(ty, th, tx, tw).any(dim=(1, 3))
        return tile_mask  # shape: (tiles_y, tiles_x)

    def get_diff_and_promote(
        self,
        *,
        mode: str = "tile",          # "pixel" or "tile"
        include_data: bool = False,   # only used in tile mode: return tile payloads
        max_updates: int | None = None,
    ) -> list:
        """
        Return changes since last promotion, then promote next->display.

        mode="pixel": list[(y, x, (r,g,b))]
        mode="tile":  list[(tile_y, tile_x)] or list[(tile_y, tile_x, ndarray)]
                      (the ndarray is the tile region if include_data=True)

        `max_updates`: if set, cap the number of returned updates (useful to avoid floods).
        """
        # 1) snapshot render -> next under lock
        with self.lock:
            AbstractTensor.copyto(self.buffer_next, self.buffer_render)

        rows, cols, _ = self.buffer_shape
        th, tw = self.tile_shape

        # 2) figure out what changed
        if self._force_full_diff_next_call:
            self._force_full_diff_next_call = False
            if mode == "pixel":
                ys, xs = AbstractTensor.meshgrid(
                    AbstractTensor.arange(rows),
                    AbstractTensor.arange(cols),
                    indexing="ij",
                )
                coords = AbstractTensor.stack(
                    [ys.view_flat(), xs.view_flat()], dim=1
                )
                updates = [
                    (int(y), int(x), tuple(int(v) for v in self.buffer_next[y, x]))
                    for y, x in coords
                ]
            elif mode == "tile":
                tiles_y = (rows + th - 1) // th
                tiles_x = (cols + tw - 1) // tw
                tcoords = AbstractTensor.stack(
                    AbstractTensor.meshgrid(
                        AbstractTensor.arange(tiles_y),
                        AbstractTensor.arange(tiles_x),
                        indexing="ij",
                    ),
                    dim=-1,
                ).reshape(-1, 2)
                updates = []
                for ty, tx in tcoords:
                    if include_data:
                        y0, x0 = ty * th, tx * tw
                        y1, x1 = min(y0 + th, rows), min(x0 + tw, cols)
                        
                        payload = self.buffer_next[y0:y1, x0:x1].copy()
                        updates.append((int(ty), int(tx), payload))
                    else:
                        updates.append((int(ty), int(tx)))
            else:
                raise ValueError("mode must be 'pixel' or 'tile'")
        else:
            diff_mask = self._compute_pixel_diff_mask()

            if mode == "pixel":
                coords = AbstractTensor.argwhere(diff_mask)
                if max_updates is not None and coords.shape[0] > max_updates:
                    coords = coords[:max_updates]
                updates = [
                    (int(y), int(x), tuple(int(v) for v in self.buffer_next[y, x]))
                    for (y, x) in coords
                ]
            elif mode == "tile":
                tile_mask = self._mask_to_tile_mask(diff_mask)
                tcoords = AbstractTensor.argwhere(tile_mask)  # (tile_y, tile_x)
                if max_updates is not None and tcoords.shape[0] > max_updates:
                    tcoords = tcoords[:max_updates]
                updates = []
                for ty, tx in tcoords:
                    if include_data:
                        y0, x0 = ty * th, tx * tw
                        y1, x1 = min(y0 + th, rows), min(x0 + tw, cols)
                        payload = self.buffer_next[y0:y1, x0:x1].copy()
                        updates.append((int(ty), int(tx), payload))
                    else:
                        updates.append((int(ty), int(tx)))
            else:
                raise ValueError("mode must be 'pixel' or 'tile'")

        # 3) promote next -> display
        AbstractTensor.copyto(self.buffer_display, self.buffer_next)

        # DO NOT print updates here (avoids log/fps collapse)
        return updates
