from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


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
    def enqueue(self, label: str, frame: np.ndarray) -> None:
        """Place a new frame on the queue."""

        self.queue.put(RenderItem(label, np.array(frame)))

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
        """Return ``img`` as RGB with an optional alpha map darkening overlay.

        Frames may include a fourth channel representing a depth mask.  The
        mask is a 1‑channel texture applied **after** upscaling to darken the
        colour channels and give a slight "bubble" appearance to tiles.  It is
        *not* treated as a transparency layer.
        """

        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Grayscale inputs -> RGB
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
            return arr

        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
            return arr

        if arr.shape[2] >= 4:
            colour = arr[..., :3].astype(np.float32)
            mask = arr[..., 3].astype(np.float32) / 255.0
            shaded = colour * (1.0 - mask[..., None])
            return shaded.clip(0, 255).astype(np.uint8)

        # Already RGB
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


__all__ = ["FrameCache", "RenderItem"]
