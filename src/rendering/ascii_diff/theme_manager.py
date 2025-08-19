#!/usr/bin/env python3
"""Manage color themes and post-processing for clock rendering."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageEnhance, ImageFilter
# --- END HEADER ---
from .time_units import TimeUnit # Import the new TimeUnit class

@dataclass
class ClockTheme:
    palette: dict
    effects: dict
    ascii_style: str
    post_processing: dict
    active_time_units: List[TimeUnit] = field(default_factory=list) # For analog/digital
    digital_format_key: str = "default_hms" # Key to lookup in presets.digital_format_strings
    invert_clock: bool = False
    current_backdrop_path: Optional[str] = None # Added for backdrop cycling
    invert_backdrop: bool = False

# Determine the directory of the current module to reliably locate presets
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRESETS_PATH = os.path.join(_MODULE_DIR, "presets", "default_themes.json")

class ThemeManager:
    def __init__(self, presets_path: str = DEFAULT_PRESETS_PATH):
        self.presets_path = presets_path
        self.current_theme = ClockTheme({}, {}, "block", {}, [])
        self._load_presets()
        if not self.current_theme.active_time_units: # Ensure a default set of units
            self.set_time_unit_set("default_analog") 
        if not self.current_theme.digital_format_key:
            self.set_digital_format_key("default_hms")

    def _load_presets(self) -> None:
        """Loads presets from the JSON file.
        Initializes self.presets to an empty structure if loading fails.
        """
        empty_presets = {"color_palettes": {}, "effects_presets": {},
                         "ascii_styles": {}, "post_processing": {}, "digital_format_strings": {},
                         "time_units_definitions": {}, "time_unit_sets": {}}
        try:
            if not os.path.exists(self.presets_path):
                print(f"Error loading presets: File not found at '{self.presets_path}'")
                self.presets = empty_presets
                return

            with open(self.presets_path, 'r', encoding='utf-8') as f:
                self.presets = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading presets: Could not decode JSON from '{self.presets_path}'. Error: {e}")
            self.presets = empty_presets
        except Exception as e:
            # Catch other potential errors like IO errors after file exists check
            print(f"Error loading presets: An unexpected error occurred with '{self.presets_path}'. Error: {e}")
            self.presets = empty_presets

    def apply_theme(self, image: Image.Image) -> Image.Image:
        """Apply current theme's post-processing to image"""
        # Ensure image is in a mode that supports enhancement (e.g., RGB)
        if self.current_theme.invert_clock:
            image = Image.eval(image, lambda x: 255 - x)
        
        pp = self.current_theme.post_processing
        if pp:
            brightness = pp.get("brightness", 1.0)
            contrast = pp.get("contrast", 1.0)
            saturation = pp.get("saturation", 1.0)

            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
            enhancer = ImageEnhance.Color(image) # For saturation
            image = ImageEnhance.Color(image).enhance(saturation)

        return image

    def apply_effects(self, image: Image.Image) -> Image.Image:
        """Apply current theme's effects to image"""
        effects = self.current_theme.effects
        if effects:
            if effects.get("glow_radius", 0) > 0:
                glow = image.filter(ImageFilter.GaussianBlur(effects["glow_radius"]))
                image = Image.blend(image, glow, effects.get("glow_intensity", 0.5))

            if effects.get("shadow_blur", 0) > 0:
                if image.mode != "RGBA":
                    image = image.convert("RGBA")
                offset_x, offset_y = effects.get("shadow_offset", (2, 2))
                shadow_color = tuple(self.current_theme.palette.get("shadow", (0, 0, 0, 150)))
                shadow = Image.new("RGBA", image.size, shadow_color)
                shadow = shadow.filter(ImageFilter.GaussianBlur(effects["shadow_blur"]))
                base = Image.new("RGBA", image.size, (0, 0, 0, 0))
                base.paste(shadow, (offset_x, offset_y), shadow)
                base.paste(image, (0, 0), image)
                image = base

        return image

    def get_current_ascii_ramp(self) -> str:
        """Get current ASCII style ramp.
        Falls back to "block" style if current is not found,
        then to a hardcoded default if "block" is also not found (e.g. due to loading error).
        """
        # Hardcoded default ramp, matching the "block" style in default_themes.json
        DEFAULT_FALLBACK_RAMP = " .:░▒▓█"

        ascii_styles_preset = self.presets.get("ascii_styles", {})

        # 1. Try to get the ramp for the current theme's ascii_style
        current_style_ramp = ascii_styles_preset.get(self.current_theme.ascii_style)
        if current_style_ramp is not None: # Check for None in case a style has an empty string ramp
            return current_style_ramp

        # 2. If current theme's style not found, try to get the "block" style ramp
        block_style_ramp = ascii_styles_preset.get("block")
        if block_style_ramp is not None:
            return block_style_ramp
        
        # 3. If "block" style is also not found, return the hardcoded default fallback ramp.
        return DEFAULT_FALLBACK_RAMP

    def cycle_ascii_style(self, step: int = 1) -> str:
        """Cycle the active ASCII style forward or backward by ``step``."""
        styles = list(self.presets.get("ascii_styles", {}).keys())
        if not styles:  # Fallback if no styles are loaded
            self.current_theme.ascii_style = "block"  # Default to "block" style name
            return self.current_theme.ascii_style

        try:
            idx = styles.index(self.current_theme.ascii_style)
        except ValueError:
            idx = 0

        new_idx = (idx + step) % len(styles)
        self.current_theme.ascii_style = styles[new_idx]
        return self.current_theme.ascii_style

    def toggle_clock_inversion(self) -> bool:
        self.current_theme.invert_clock = not self.current_theme.invert_clock
        return self.current_theme.invert_clock

    def toggle_backdrop_inversion(self) -> bool:
        self.current_theme.invert_backdrop = not self.current_theme.invert_backdrop
        return self.current_theme.invert_backdrop

    def set_palette(self, palette_name: str) -> None:
        """Set the color palette by name"""
        if palette_name in self.presets["color_palettes"]:
            palette = self.presets["color_palettes"][palette_name].copy()
            palette["name"] = palette_name
            self.current_theme.palette = palette
        else:
            print(f"Palette '{palette_name}' not found.")

    def set_effects(self, effects_name: str) -> None:
        """Set the effects preset by name"""
        if effects_name in self.presets["effects_presets"]:
            effects = self.presets["effects_presets"][effects_name].copy()
            effects["name"] = effects_name
            self.current_theme.effects = effects
        else:
            print(f"Effects preset '{effects_name}' not found.")

    def set_post_processing(self, pp_name: str) -> None:
        """Set the post-processing preset by name"""
        if pp_name in self.presets["post_processing"]:
            post_processing = self.presets["post_processing"][pp_name].copy()
            post_processing["name"] = pp_name
            self.current_theme.post_processing = post_processing
        else:
            print(f"Post-processing preset '{pp_name}' not found.")

    def get_palette_names(self) -> List[str]:
        """Return the available palette names."""
        return list(self.presets.get("color_palettes", {}).keys())

    def cycle_palette(self, step: int = 1) -> str:
        """Cycle the active palette forward or backward by ``step``."""
        names = self.get_palette_names()
        if not names:
            return ""
        current = self.current_theme.palette.get("name", names[0])
        try:
            idx = names.index(current)
        except ValueError:
            idx = 0
        new_idx = (idx + step) % len(names)
        self.set_palette(names[new_idx])
        return names[new_idx]

    def get_effects_preset_names(self) -> List[str]:
        return list(self.presets.get("effects_presets", {}).keys())

    def cycle_effects_preset(self, step: int = 1) -> str:
        names = self.get_effects_preset_names()
        if not names: return ""
        current = self.current_theme.effects.get("name", names[0])
        try: idx = names.index(current)
        except ValueError: idx = 0
        new_idx = (idx + step) % len(names)
        self.set_effects(names[new_idx])
        return names[new_idx]

    def get_post_processing_preset_names(self) -> List[str]:
        return list(self.presets.get("post_processing", {}).keys())

    def cycle_post_processing_preset(self, step: int = 1) -> str:
        names = self.get_post_processing_preset_names()
        if not names: return ""
        current = self.current_theme.post_processing.get("name", names[0])
        try: idx = names.index(current)
        except ValueError: idx = 0
        new_idx = (idx + step) % len(names)
        self.set_post_processing(names[new_idx])
        return names[new_idx]

    def get_time_unit_set_names(self) -> List[str]:
        return list(self.presets.get("time_unit_sets", {}).keys())

    def cycle_time_unit_set(self, current_set_name: str, step: int = 1) -> str:
        names = self.get_time_unit_set_names()
        if not names: return current_set_name
        try: idx = names.index(current_set_name)
        except ValueError: idx = 0
        new_idx = (idx + step) % len(names)
        # self.set_time_unit_set(names[new_idx]) # This would set for current_theme.active_time_units
        return names[new_idx] # Return the name, clock_demo will manage which clock uses it

    def get_digital_format_key_names(self) -> List[str]:
        return list(self.presets.get("digital_format_strings", {}).keys())

    def cycle_digital_format_key(self, current_format_key: str, step: int = 1) -> str:
        names = self.get_digital_format_key_names()
        if not names: return current_format_key
        try: idx = names.index(current_format_key)
        except ValueError: idx = 0
        new_idx = (idx + step) % len(names)
        return names[new_idx] # Return the key name

    def get_time_unit_definitions(self) -> Dict[str, TimeUnit]:
        """Returns all defined TimeUnit objects."""
        defs = self.presets.get("time_units_definitions", {})
        return {name: TimeUnit.from_dict(name, data) for name, data in defs.items()}

    def set_time_unit_set(self, set_name: str) -> None:
        """Sets the active time units based on a predefined set name."""
        unit_sets = self.presets.get("time_unit_sets", {})
        unit_names_in_set = unit_sets.get(set_name)

        if unit_names_in_set:
            all_defined_units = self.get_time_unit_definitions()
            active_units = []
            for unit_name in unit_names_in_set:
                if unit_name in all_defined_units:
                    active_units.append(all_defined_units[unit_name])
                else:
                    print(f"Warning: Time unit '{unit_name}' in set '{set_name}' not defined.")
            self.current_theme.active_time_units = active_units[:5] # Max 5 units
        else:
            print(f"Time unit set '{set_name}' not found. Using current or empty.")

    def set_digital_format_key(self, key_name: str) -> None:
        """Sets the key for the digital format string to be used from presets."""
        if key_name in self.presets.get("digital_format_strings", {}):
            self.current_theme.digital_format_key = key_name
        else:
            print(f"Digital format key '{key_name}' not found. Using current or default.")

    def get_current_digital_format_string(self) -> str:
        """Gets the actual format string based on the current_theme.digital_format_key."""
        formats = self.presets.get("digital_format_strings", {})
        # Fallback to a simple H:M:S if the key or "default_hms" is not found
        return formats.get(self.current_theme.digital_format_key, formats.get("default_hms", "{value:02.0f}:{value:02.0f}:{value:02.0f}"))

    def cycle_backdrop(self, available_backdrop_paths: List[str], step: int = 1) -> Optional[str]:
        if not available_backdrop_paths:
            self.current_theme.current_backdrop_path = None
            return None
        
        try:
            current_idx = available_backdrop_paths.index(self.current_theme.current_backdrop_path) if self.current_theme.current_backdrop_path else -1
        except ValueError:
            current_idx = -1 # If current path not in list, start from beginning
        
        new_idx = (current_idx + step) % len(available_backdrop_paths)
        self.current_theme.current_backdrop_path = available_backdrop_paths[new_idx]
        return self.current_theme.current_backdrop_path
