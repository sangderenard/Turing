from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Tuple

import colorsys
import math
import numpy as np
from PIL import Image
import re


_VIGNETTE_CACHE: Dict[Tuple[int, int, float], np.ndarray] = {}

_GRADIENTS = {
    "grayscale": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
    "fire": np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=np.float32,
    ),
    "blue_fire": np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=np.float32,
    ),
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

    To emphasise the vignette, the source pixels are first oversampled using
    nearest‑neighbour expansion.  The vignette mask is then upscaled with a
    high quality filter before being applied.  Finally the image is reduced to
    the target tile size using a high quality downsampling filter.

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

    oversample = 4
    # Cap the working-set size by reducing oversample if necessary.
    # Estimate the intermediate working buffer as if each tile is rendered at
    # (tile * oversample)^2 across all tiles, using float32 during blending.
    # This is a conservative bound to avoid spikes on very large grids.
    max_bytes = 8 * 1024 * 1024  # 8 MiB default ceiling for intermediate data
    while oversample > 1:
        big_tile = tile * oversample
        est_bytes = (
            h * w * big_tile * big_tile * c * np.dtype(np.float32).itemsize
        )
        if est_bytes <= max_bytes:
            break
        oversample //= 2
    big_tile = tile * oversample

    # Nearest neighbour oversample of source pixels to the output tile size
    up = arr.repeat(tile, axis=0).repeat(tile, axis=1).astype(np.float32)

    # Oversample the vignette mask at high quality then downsample back
    base_mask = (_pixel_vignette(tile, power) * 255).astype(np.uint8)
    pil_mask = Image.fromarray(base_mask)
    pil_mask = pil_mask.resize((big_tile, big_tile), resample=Image.LANCZOS)
    pil_mask = pil_mask.resize((tile, tile), resample=Image.LANCZOS)
    mask = np.array(pil_mask, dtype=np.float32) / 255.0
    mask_tiled = np.tile(mask, (h, w))

    up *= (1.0 - mask_tiled)[..., None]
    out = up.clip(0, 255).astype(np.uint8)
    return out[..., 0] if c == 1 else out


def apply_colormap(frame: np.ndarray, cmap: str = "blue_fire") -> np.ndarray:
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
        grad = _GRADIENTS.get(cmap, _GRADIENTS["blue_fire"])
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
    can be supplied so composed layouts always match the display surface.  By
    default frames are stored at their original resolution and only upscaled
    with a vignette when rendered.  Set ``store_scaled=True`` to apply the
    vignette immediately and retain the upscaled result in memory.
    """

    def __init__(
        self,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
        *,
        store_scaled: bool = False,
    ) -> None:
        self.queue: "Queue[RenderItem]" = Queue()
        self.cache: Dict[str, List[np.ndarray]] = {}
        # Cache of composed layouts keyed by a hash of their configuration
        # and selected frame indices. This avoids re-rendering identical
        # composites in the GUI.
        self.composite_cache: Dict[int, np.ndarray] = {}
        self.target_height = target_height
        self.target_width = target_width
        self.store_scaled = store_scaled

    # ------------------------------------------------------------------
    # Queue helpers
    # ------------------------------------------------------------------
    def enqueue(self, label: str, frame: np.ndarray) -> None:
        """Place a new frame on the queue."""

        arr = np.array(frame)
        if self.store_scaled:
            arr = add_vignette(arr)
        self.queue.put(RenderItem(label, arr))

    def process_queue(self) -> bool:
        """Drain all pending frames into the cache.

        Any time new frames are processed the composite cache is cleared so
        that subsequent calls recompute layouts when necessary.

        Returns
        -------
        bool
            ``True`` if new frames were added to the cache.
        """

        changed = False
        while not self.queue.empty():
            item = self.queue.get()
            self.cache.setdefault(item.label, []).append(item.frame)
            changed = True
        if changed:
            self.composite_cache.clear()
        return changed

    def clear(self, *, preserve_composite: bool = True) -> None:
        """Wipe cached frames and queued items.

        Parameters
        ----------
        preserve_composite:
            When ``True`` (default) previously composed layouts remain in
            ``composite_cache`` so UIs can continue displaying the last
            rendered frame without retaining the per‑label intermediates.
            Set to ``False`` to drop everything.
        """

        self.cache.clear()
        while not self.queue.empty():
            self.queue.get()
        if not preserve_composite:
            self.composite_cache.clear()

    def available_sources(self) -> List[str]:
        """Return sorted set of data sources derived from cached labels."""

        return sorted({label.split("_")[0] for label in self.cache})

    def available_types(self) -> List[str]:
        """Return sorted set of data types derived from cached labels."""

        return sorted({label.split("_")[1] for label in self.cache if "_" in label})

    def _group_labels(self) -> Dict[str, List[str]]:
        """Map high level group names to cached labels.

        Groups are derived from naming conventions used in the demo:
        ``param{n}_param``/``param{n}_grad`` for parameters and gradients and
        ``*_input``/``*_prediction`` for inputs and predictions.
        """

        groups: Dict[str, List[str]] = {}
        param_re = re.compile(r"param\d+_param")
        grad_re = re.compile(r"param\d+_grad")
        input_re = re.compile(r".*_input")
        pred_re = re.compile(r".*_prediction")
        for label in self.cache:
            if param_re.fullmatch(label):
                groups.setdefault("params", []).append(label)
            if grad_re.fullmatch(label):
                groups.setdefault("grads", []).append(label)
            if input_re.fullmatch(label):
                groups.setdefault("inputs", []).append(label)
            if pred_re.fullmatch(label):
                groups.setdefault("predictions", []).append(label)
        return groups

    def available_options(self, stats: Optional[List[str]] = None) -> List[str]:
        """Return label/stat combinations for dropdown menus.

        Parameters
        ----------
        stats:
            Iterable of statistical reduction names.  Defaults to
            ``["sample", "mean", "std", "min", "max"]``.
        """

        if stats is None:
            stats = ["sample", "mean", "std", "min", "max"]
        group_stats = stats + ["grid"]
        options: List[str] = []
        labels = sorted(self.cache.keys())
        groups = self._group_labels()
        for lbl in labels:
            for stat in stats:
                options.append(f"{lbl}:{stat}")
        for grp in sorted(groups):
            for stat in group_stats:
                options.append(f"{grp}:{stat}")
        return options

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

    def compose_group(self, group: str, index: Optional[int] = None) -> np.ndarray:
        """Compose all frames in ``group`` into a near-square grid."""

        labels = self._group_labels().get(group)
        if not labels:
            return np.zeros((1, 1), dtype=np.uint8)
        if index is None:
            frames = [self.cache[l][-1] for l in labels if self.cache.get(l)]
        else:
            frames = [self.cache[l][index % len(self.cache[l])] for l in labels if self.cache.get(l)]
        if not frames:
            return np.zeros((1, 1), dtype=np.uint8)

        # Determine common tile size and resize so each parameter occupies
        # equal space regardless of its original dimensions.
        tile_h = max(img.shape[0] for img in frames)
        tile_w = max(img.shape[1] for img in frames)
        resized: List[np.ndarray] = [
            img
            if img.shape[:2] == (tile_h, tile_w)
            else self.nearest_neighbor_resize(img, (tile_h, tile_w))
            for img in frames
        ]

        cols = math.ceil(math.sqrt(len(resized)))
        rows = math.ceil(len(resized) / cols)
        ch = resized[0].shape[2:] if resized[0].ndim == 3 else ()
        canvas = np.zeros((rows * tile_h, cols * tile_w) + ch, dtype=resized[0].dtype)
        for i, img in enumerate(resized):
            r, c = divmod(i, cols)
            canvas[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w, ...] = img
        return canvas

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
            tile_h = 0
            tile_w = 0
            for label in row:
                if label not in self.cache or not self.cache[label]:
                    continue
                img = self.cache[label][-1]
                tile_h = max(tile_h, img.shape[0])
                tile_w = max(tile_w, img.shape[1])
                imgs.append(img)
            if not imgs:
                continue
            normed = [
                img
                if img.shape[:2] == (tile_h, tile_w)
                else self.nearest_neighbor_resize(img, (tile_h, tile_w))
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
            # Upscale only once the full grid is assembled to avoid keeping
            # large intermediates in memory.
            if not self.store_scaled:
                grid = add_vignette(grid)
            if self.target_height and self.target_width:
                grid = self.nearest_neighbor_resize(grid, (self.target_height, self.target_width))
        self.composite_cache[key] = grid
        return grid

    def _layout_hash(self, layout: List[List[str]], index: Optional[int]) -> int:
        """Return a hash describing ``layout`` and chosen frame indices."""

        rows = []
        for row in layout:
            row_key = []
            for label_stat in row:
                base, _, stat = label_stat.partition(":")
                if stat == "sample" and index is not None:
                    frame_idx = index
                else:
                    frame_idx = -1
                row_key.append((label_stat, frame_idx))
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

        groups = self._group_labels()
        rows: List[np.ndarray] = []
        for row in layout:
            imgs: List[np.ndarray] = []
            tile_h = 0
            tile_w = 0
            for label_stat in row:
                base, _, stat = label_stat.partition(":")
                frames_list: List[np.ndarray]
                if base in groups:
                    if stat == "sample":
                        frames_list = [self.cache[l][index % len(self.cache[l])] for l in groups[base] if self.cache.get(l)]
                    elif stat == "grid":
                        img = self.compose_group(base, index)
                        tile_h = max(tile_h, img.shape[0])
                        tile_w = max(tile_w, img.shape[1])
                        imgs.append(img)
                        continue
                    else:
                        frames_list = [f for l in groups[base] for f in self.cache.get(l, [])]
                else:
                    frames = self.cache.get(base)
                    if not frames:
                        continue
                    frames_list = frames if stat != "sample" else [frames[index % len(frames)]]
                if not frames_list:
                    continue
                if stat == "sample":
                    if len(frames_list) == 1:
                        stack = frames_list[0]
                    else:
                        stack = np.stack(frames_list, axis=0).mean(axis=0)
                else:
                    stack = np.stack(frames_list, axis=0).astype(np.float32)
                    if stat == "mean":
                        stack = stack.mean(axis=0)
                    elif stat == "std":
                        stack = stack.std(axis=0)
                    elif stat == "min":
                        stack = stack.min(axis=0)
                    elif stat == "max":
                        stack = stack.max(axis=0)
                    else:
                        stack = stack[0]
                img = stack if stack.dtype == np.uint8 else np.clip(stack, 0, 255).astype(np.uint8)
                tile_h = max(tile_h, img.shape[0])
                tile_w = max(tile_w, img.shape[1])
                imgs.append(img)
            if not imgs:
                continue
            normed = [
                img
                if img.shape[:2] == (tile_h, tile_w)
                else self.nearest_neighbor_resize(img, (tile_h, tile_w))
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
            # Apply vignette/upscaling only to the final composite.
            if not self.store_scaled:
                grid = add_vignette(grid)
            if self.target_height and self.target_width:
                grid = self.nearest_neighbor_resize(grid, (self.target_height, self.target_width))
        self.composite_cache[key] = grid
        return grid

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_animation(
        self, label: str, path: str | Path, duration: int = 800, cmap: Optional[str] = None
    ) -> None:
        """Save the frames for ``label`` as a GIF animation.

        Parameters
        ----------
        label:
            Key identifying which cached frames to save.
        path:
            Output file path. Parent directories are created automatically.
        duration:
            Duration of each frame in milliseconds. Defaults to ``800``.
        cmap:
            Optional colour map name applied to each frame prior to export.
        """

        frames = self.cache.get(label)
        if not frames:
            return
        # Upscale only when writing to disk so the cache retains compact
        # representations of each frame unless frames were pre-scaled.
        upscaled = frames if self.store_scaled else [add_vignette(f) for f in frames]
        if cmap is not None:
            images = [Image.fromarray(apply_colormap(f, cmap)) for f in upscaled]
        else:
            images = [Image.fromarray(f) for f in upscaled]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            loop=0,
            duration=duration,
        )


__all__ = ["FrameCache", "RenderItem", "apply_colormap", "add_vignette"]
