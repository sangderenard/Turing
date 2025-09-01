from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Tuple

import colorsys
import numpy as np
from PIL import Image


_VIGNETTE_CACHE: Dict[Tuple[int, int, float], np.ndarray] = {}

_GRADIENTS = {
    "grayscale": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
    "fire": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
}


def _vignette_mask(height: int, width: int, power: float = 4.0) -> np.ndarray:
    """Return a rounded‑square vignette mask in ``uint8``.

    The mask smoothly darkens towards the edges using a squircle profile.
    Results are cached by ``(height, width, power)``.
    """

    key = (height, width, power)
    mask = _VIGNETTE_CACHE.get(key)
    if mask is not None:
        return mask

    y = np.linspace(-1.0, 1.0, height)
    x = np.linspace(-1.0, 1.0, width)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    r = (np.abs(xx) ** power + np.abs(yy) ** power) ** (1.0 / power)
    mask = np.clip(r, 0.0, 1.0) ** 2
    mask = (mask * 255).astype(np.uint8)
    _VIGNETTE_CACHE[key] = mask
    return mask


def _pixel_vignette(tile: int, power: float = 4.0) -> np.ndarray:
    """Return a single‑pixel bubble mask of size ``tile×tile``."""

    return _vignette_mask(tile, tile, power) / 255.0


def add_vignette(frame: np.ndarray, tile: int = 8, power: float = 4.0) -> np.ndarray:
    """Upscale ``frame`` and tint each pixel with a rounded bubble mask.

    Parameters
    ----------
    frame:
        Grayscale or RGB image.  A ``(H,W)`` array is treated as grayscale.
    tile:
        Output size of each input pixel.  Defaults to ``8``.
    power:
        Squircle power controlling corner roundness.
    """

    arr = np.array(frame)
    if arr.ndim == 2:
        arr = arr[..., None]
    h, w, c = arr.shape
    mask = _pixel_vignette(tile, power)
    mask_tiled = np.tile(mask, (h, w))
    mask_tiled = mask_tiled.reshape(h, tile, w, tile).swapaxes(1, 2).reshape(h * tile, w * tile)

    up = arr.repeat(tile, axis=0).repeat(tile, axis=1).astype(np.float32)
    up *= (1.0 - mask_tiled)[..., None]
    out = up.clip(0, 255).astype(np.uint8)
    return out[..., 0] if c == 1 else out


def apply_colormap(frame: np.ndarray, cmap: str = "grayscale") -> np.ndarray:
    """Apply ``cmap`` to ``frame`` returning an RGB image."""

    arr = np.array(frame)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    if cmap == "hue":
        eps = 1.0 / 360.0
        hues = eps + (1.0 - 2.0 * eps) * arr
        flat = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues.ravel()]
        colour = np.array(flat, dtype=np.float32).reshape(arr.shape + (3,))
    else:
        grad = _GRADIENTS.get(cmap, _GRADIENTS["grayscale"])
        n = grad.shape[0]
        idx = arr * (n - 1)
        lo = np.floor(idx).astype(int)
        hi = np.clip(lo + 1, 0, n - 1)
        t = (idx - lo)[..., None]
        colour = grad[lo] * (1 - t) + grad[hi] * t
    return (colour * 255).astype(np.uint8)


@dataclass
class RenderItem:
    """Unit of work passed between training and GUI threads."""

    label: str
    frame: np.ndarray


class FrameCache:
    """Thread‑safe frame cache for demo visualisations.

    The training thread enqueues :class:`RenderItem` instances while the GUI
    thread drains the queue and stores the frames.  Images can later be saved
    as animations or combined via layout descriptors.  A target height/width
    can be supplied so composed layouts always match the display surface.
    """

    def __init__(self, target_height: Optional[int] = None, target_width: Optional[int] = None) -> None:
        self.queue: "Queue[RenderItem]" = Queue()
        self.cache: Dict[str, List[np.ndarray]] = {}
        # Cache of composed layouts keyed by a hash of their configuration
        # and selected frame indices. This avoids re-rendering identical
        # composites in the GUI.
        self.composite_cache: Dict[int, np.ndarray] = {}
        self.target_height = target_height
        self.target_width = target_width

    # ------------------------------------------------------------------
    # Queue helpers
    # ------------------------------------------------------------------
    def enqueue(
        self,
        label: str,
        frame: np.ndarray,
        cmap: Optional[str] = None,
        mask_first: bool = False,
    ) -> None:
        """Place a new frame on the queue applying optional colour and mask."""

        arr = np.array(frame)
        if mask_first:
            arr = add_vignette(arr)
            if cmap is not None:
                arr = apply_colormap(arr, cmap)
        else:
            if cmap is not None:
                arr = apply_colormap(arr, cmap)
            arr = add_vignette(arr)
        self.queue.put(RenderItem(label, arr))

    def process_queue(self) -> None:
        """Drain all pending frames into the cache.

        Any time new frames are processed the composite cache is cleared so
        that subsequent calls recompute layouts when necessary.
        """

        changed = False
        while not self.queue.empty():
            item = self.queue.get()
            self.cache.setdefault(item.label, []).append(item.frame)
            changed = True
        if changed:
            self.composite_cache.clear()

    def available_sources(self) -> List[str]:
        """Return sorted set of data sources derived from cached labels."""

        return sorted({label.split("_")[0] for label in self.cache})

    def available_types(self) -> List[str]:
        """Return sorted set of data types derived from cached labels."""

        return sorted({label.split("_")[1] for label in self.cache if "_" in label})

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    @staticmethod
    def nearest_neighbor_resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize ``img`` using nearest‑neighbour sampling."""

        pil = Image.fromarray(img)
        pil = pil.resize((size[1], size[0]), resample=Image.NEAREST)
        return np.array(pil)

    # ------------------------------------------------------------------
    # Format helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_alpha_map(img: np.ndarray) -> np.ndarray:
        """Ensure ``img`` is an RGB ``uint8`` array.

        Any extra channels are dropped; grayscale inputs are tiled to RGB.
        """

        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return np.repeat(arr[..., None], 3, axis=2)
        if arr.shape[2] == 1:
            return np.repeat(arr, 3, axis=2)
        if arr.shape[2] > 3:
            return arr[..., :3]
        return arr

    def compose_layout(self, layout: List[List[str]]) -> np.ndarray:
        """Compose a grid according to ``layout``.

        Parameters
        ----------
        layout:
            A nested list describing rows and their labels.  The most recent
            frame for each label is used.  Missing labels are skipped.
        """

        key = self._layout_hash(layout, None)
        cached = self.composite_cache.get(key)
        if cached is not None:
            return cached

        rows: List[np.ndarray] = []
        for row in layout:
            imgs: List[np.ndarray] = []
            max_h = 0
            for label in row:
                if label not in self.cache or not self.cache[label]:
                    continue
                img = self.cache[label][-1]
                max_h = max(max_h, img.shape[0])
                imgs.append(img)
            if not imgs:
                continue
            normed = [
                img if img.shape[0] == max_h else self.nearest_neighbor_resize(img, (max_h, img.shape[1]))
                for img in imgs
            ]
            rows.append(np.concatenate(normed, axis=1))
        if not rows:
            grid = np.zeros((1, 1), dtype=np.uint8)
        else:
            max_w = max(r.shape[1] for r in rows)
            padded = [
                r
                if r.shape[1] == max_w
                else np.concatenate([r, np.zeros((r.shape[0], max_w - r.shape[1], *r.shape[2:]), dtype=r.dtype)], axis=1)
                for r in rows
            ]
            grid = np.concatenate(padded, axis=0)
            if self.target_height and self.target_width:
                grid = self.nearest_neighbor_resize(grid, (self.target_height, self.target_width))
        grid = self._apply_alpha_map(grid)
        self.composite_cache[key] = grid
        return grid

    def _layout_hash(self, layout: List[List[str]], index: Optional[int]) -> int:
        """Return a hash describing ``layout`` and chosen frame indices."""

        rows = []
        for row in layout:
            row_key = []
            for label in row:
                frames = self.cache.get(label)
                if not frames:
                    frame_idx = -1
                else:
                    frame_idx = (len(frames) - 1) if index is None else index % len(frames)
                row_key.append((label, frame_idx))
            rows.append(tuple(row_key))
        return hash(tuple(rows))

    def compose_layout_at(self, layout: List[List[str]], index: int) -> np.ndarray:
        """Compose a grid using the ``index``-th frame for each label.

        ``index`` is wrapped by the length of each label's cache so the
        animation can loop seamlessly forward or backward.
        """

        key = self._layout_hash(layout, index)
        cached = self.composite_cache.get(key)
        if cached is not None:
            return cached

        rows: List[np.ndarray] = []
        for row in layout:
            imgs: List[np.ndarray] = []
            max_h = 0
            for label in row:
                frames = self.cache.get(label)
                if not frames:
                    continue
                img = frames[index % len(frames)]
                max_h = max(max_h, img.shape[0])
                imgs.append(img)
            if not imgs:
                continue
            normed = [
                img if img.shape[0] == max_h else self.nearest_neighbor_resize(img, (max_h, img.shape[1]))
                for img in imgs
            ]
            rows.append(np.concatenate(normed, axis=1))
        if not rows:
            grid = np.zeros((1, 1), dtype=np.uint8)
        else:
            max_w = max(r.shape[1] for r in rows)
            padded = [
                r
                if r.shape[1] == max_w
                else np.concatenate([r, np.zeros((r.shape[0], max_w - r.shape[1], *r.shape[2:]), dtype=r.dtype)], axis=1)
                for r in rows
            ]
            grid = np.concatenate(padded, axis=0)
            if self.target_height and self.target_width:
                grid = self.nearest_neighbor_resize(grid, (self.target_height, self.target_width))
        grid = self._apply_alpha_map(grid)
        self.composite_cache[key] = grid
        return grid

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_animation(self, label: str, path: str | Path, duration: int = 800) -> None:
        """Save the frames for ``label`` as a GIF animation."""

        frames = self.cache.get(label)
        if not frames:
            return
        images = [Image.fromarray(f) for f in frames]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            loop=0,
            duration=duration,
        )


__all__ = ["FrameCache", "RenderItem", "apply_colormap", "add_vignette"]
