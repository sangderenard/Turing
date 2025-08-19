import numpy as np
from typing import Callable, List, Tuple, Optional, Dict, Any

class Texture:
    """
    A simple, composable texture object for 2D numpy array frames, supporting layering, tweening, and blend modes.
    """
    def __init__(
        self,
        frames: List[np.ndarray],
        positions: Optional[List[Tuple[int, int]]] = None,
        durations: Optional[List[float]] = None,
        blend_modes: Optional[List[str]] = None,
        tween: Optional[Callable[[np.ndarray, np.ndarray, float], np.ndarray]] = None,
        name: Optional[str] = None,
    ):
        """
        frames: list of 2D or 3D numpy arrays (H, W) or (H, W, C)
        positions: list of (x, y) tuples for each frame (default: (0,0) for all)
        durations: list of durations (seconds) for each frame (default: 1.0 for all)
        blend_modes: list of blend mode names (e.g. 'normal', 'add', 'multiply')
        tween: function for tweening between frames (optional)
        name: optional name for the texture
        """
        self.frames = frames
        self.positions = positions or [(0, 0)] * len(frames)
        self.durations = durations or [1.0] * len(frames)
        self.blend_modes = blend_modes or ['normal'] * len(frames)
        self.tween = tween
        self.name = name or "texture"
        self.num_frames = len(frames)

    def get_frame(self, t: float) -> Tuple[np.ndarray, Tuple[int, int], str]:
        """
        Get the frame, position, and blend mode for time t (seconds).
        Supports looping and tweening if a tween function is provided.
        """
        total_duration = sum(self.durations)
        t = t % total_duration
        acc = 0.0
        for i, dur in enumerate(self.durations):
            if acc + dur > t:
                if self.tween and i < self.num_frames - 1:
                    # Tween between frames i and i+1
                    alpha = (t - acc) / dur
                    frame = self.tween(self.frames[i], self.frames[i+1], alpha)
                else:
                    frame = self.frames[i]
                return frame, self.positions[i], self.blend_modes[i]
            acc += dur
        # Fallback to last frame
        return self.frames[-1], self.positions[-1], self.blend_modes[-1]

    @staticmethod
    def blend(base: np.ndarray, overlay: np.ndarray, mode: str = 'normal') -> np.ndarray:
        """
        Blend overlay onto base using the specified blend mode.
        """
        if mode == 'add':
            return np.clip(base + overlay, 0, 255)
        elif mode == 'multiply':
            return np.clip(base * overlay / 255, 0, 255)
        # Default: normal (overlay replaces base where not transparent)
        mask = (overlay > 0)
        result = base.copy()
        result[mask] = overlay[mask]
        return result

class TextureStack:
    """
    A stack of textures to be composed per frame.
    """
    def __init__(self, textures: List[Texture]):
        self.textures = textures

    def compose(self, t: float, base: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compose all textures at time t onto the base array (if provided).
        """
        if base is None:
            # Assume all textures are the same size as the first frame
            base = np.zeros_like(self.textures[0].frames[0])
        composed = base.copy()
        for tex in self.textures:
            frame, pos, mode = tex.get_frame(t)
            x, y = pos
            h, w = frame.shape[:2]
            # Blit with blend mode
            region = composed[y:y+h, x:x+w]
            blended = Texture.blend(region, frame, mode)
            composed[y:y+h, x:x+w] = blended
        return composed
