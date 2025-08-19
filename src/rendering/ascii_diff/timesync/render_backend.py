#!/usr/bin/env python3
"""Image rendering helper for theme effects and post-processing."""
from __future__ import annotations

from PIL import Image, ImageEnhance, ImageFilter
from typing import Optional
from .theme_manager import ThemeManager
# --- END HEADER ---

class RenderingBackend:
    """Apply theme effects and post-processing in a single pipeline."""

    def __init__(self, theme_manager: ThemeManager) -> None:
        self.theme_manager = theme_manager

    def apply_effects(self, image: Image.Image) -> Image.Image:
        """Apply visual effects defined in the current theme."""
        effects = self.theme_manager.current_theme.effects
        if not effects:
            return image

        if effects.get("glow_radius", 0) > 0:
            glow = image.filter(ImageFilter.GaussianBlur(effects["glow_radius"]))
            image = Image.blend(image, glow, effects.get("glow_intensity", 0.5))

        if effects.get("shadow_blur", 0) > 0:
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            offset_x, offset_y = effects.get("shadow_offset", (2, 2))
            shadow_color = tuple(self.theme_manager.current_theme.palette.get("shadow", (0, 0, 0, 150)))
            shadow = Image.new("RGBA", image.size, shadow_color)
            shadow = shadow.filter(ImageFilter.GaussianBlur(effects["shadow_blur"]))
            base = Image.new("RGBA", image.size, (0, 0, 0, 0))
            base.paste(shadow, (offset_x, offset_y), shadow)
            base.paste(image, (0, 0), image)
            image = base
        return image

    def apply_post_processing(self, image: Image.Image) -> Image.Image:
        """Apply post-processing adjustments from the current theme."""
        if self.theme_manager.current_theme.invert_clock:
            image = Image.eval(image, lambda x: 255 - x)

        pp = self.theme_manager.current_theme.post_processing
        if pp:
            brightness = pp.get("brightness", 1.0)
            contrast = pp.get("contrast", 1.0)
            saturation = pp.get("saturation", 1.0)

            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
            image = ImageEnhance.Color(image).enhance(saturation)
        return image

    def process(self, image: Image.Image) -> Image.Image:
        """Apply effects then post-processing to ``image``."""
        image = self.apply_effects(image)
        image = self.apply_post_processing(image)
        return image

    def list_available_operations(self) -> list[str]:
        """Return a curated list of extra Pillow operations."""
        # This implementation merely exposes a handful of common transforms.
        # Future versions may inspect PIL dynamically and provide richer
        # metadata for configuration files.
        return [
            "rotate",
            "transpose_left_right",
            "transpose_top_bottom",
            "blur",
            "sharpen",
            "posterize",
            "resize",
            "crop",
            "color_enhance",
        ]
