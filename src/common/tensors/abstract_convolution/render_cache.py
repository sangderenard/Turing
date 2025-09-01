from __future__ import annotations

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
        self.target_height = target_height
        self.target_width = target_width

    # ------------------------------------------------------------------
    # Queue helpers
    # ------------------------------------------------------------------
    def enqueue(self, label: str, frame: np.ndarray) -> None:
        """Place a new frame on the queue."""

        self.queue.put(RenderItem(label, np.array(frame)))

    def process_queue(self) -> None:
        """Drain all pending frames into the cache."""

        while not self.queue.empty():
            item = self.queue.get()
            self.cache.setdefault(item.label, []).append(item.frame)

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

    def compose_layout(self, layout: List[List[str]]) -> np.ndarray:
        """Compose a grid according to ``layout``.

        Parameters
        ----------
        layout:
            A nested list describing rows and their labels.  The most recent
            frame for each label is used.  Missing labels are skipped.
        """

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
            return np.zeros((1, 1), dtype=np.uint8)
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
