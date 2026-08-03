"""Publish the Python columnar state machine as a Wasm RGB-preview page."""

from __future__ import annotations

import inspect

from .columnar_multifluid_kernels import columnar_multifluid_rgb_step


def columnar_multifluid_present(red, green, blue):
    """Python-authored display expressions lowered to a packed WebGL color."""

    display_red = red / 255.0
    display_green = green / 255.0
    display_blue = blue / 255.0
    return display_red, display_green, display_blue


_PAGE_CONFIG = {
    "entrypoint": "columnar_multifluid_rgb_step",
    "presentation_entrypoint": "columnar_multifluid_present",
    "title": "Managed Columnar Multifluid World",
    "slug": "managed-columnar-multifluid-world",
    "width": 384,
    "height": 268,
    "probe_size": 16,
    "feeds": {
        "column_x": 0.5,
        "column_y": 0.5,
        "rest_surface": 1.0,
        "displacement": 0.0,
        "displacement_velocity": 0.0,
        "entity_x": 2.6,
        "entity_y": 2.0,
        "entity_velocity_x": 0.42,
        "entity_velocity_y": 0.16,
        "entity_b_x": 5.1,
        "entity_b_y": 4.8,
        "entity_b_velocity_x": -0.34,
        "entity_b_velocity_y": 0.27,
        "entity_c_x": 7.5,
        "entity_c_y": 2.8,
        "entity_c_velocity_x": 0.18,
        "entity_c_velocity_y": -0.38,
        "entity_cargo": 0.0,
        "entity_b_cargo": 0.0,
        "entity_c_cargo": 0.0,
        "food_store": 0.0,
        "nest_food": 0.0,
        "filter_reservoir": 0.0,
        "managed_time": 0.0,
        "dt": 0.025,
        "audio_low": 0.0,
        "audio_mid": 0.0,
        "audio_high": 0.0,
        "audio_level": 0.0,
        "ink_red": 0.0,
        "ink_yellow": 0.0,
        "ink_green": 0.0,
        "ink_cyan": 0.0,
        "ink_blue": 0.0,
        "ink_magenta": 0.0
    },
    "feed_expressions": {
        "column_x": "(x + 0.5) * 10.0 / w",
        "column_y": "(y + 0.5) * 7.0 / h",
        "rest_surface": "1.15 + 3.1 * Math.exp(-16.0 * (((x + 0.5) / w - 0.62) ** 2 + ((y + 0.5) / h - 0.52) ** 2))",
        "displacement": "0.0",
        "displacement_velocity": "0.0",
        "entity_x": "2.6",
        "entity_y": "2.0",
        "entity_velocity_x": "0.42",
        "entity_velocity_y": "0.16",
        "entity_b_x": "5.1",
        "entity_b_y": "4.8",
        "entity_b_velocity_x": "-0.34",
        "entity_b_velocity_y": "0.27",
        "entity_c_x": "7.5",
        "entity_c_y": "2.8",
        "entity_c_velocity_x": "0.18",
        "entity_c_velocity_y": "-0.38",
        "entity_cargo": "0.0",
        "entity_b_cargo": "0.0",
        "entity_c_cargo": "0.0",
        "food_store": "0.0",
        "nest_food": "0.0",
        "filter_reservoir": "0.0",
        "managed_time": "0.0",
        "dt": "0.025",
        "audio_low": "window.TuringAudioRuntime ? window.TuringAudioRuntime.feature('audio_low') : 0.0",
        "audio_mid": "window.TuringAudioRuntime ? window.TuringAudioRuntime.feature('audio_mid') : 0.0",
        "audio_high": "window.TuringAudioRuntime ? window.TuringAudioRuntime.feature('audio_high') : 0.0",
        "audio_level": "window.TuringAudioRuntime ? window.TuringAudioRuntime.feature('audio_level') : 0.0",
        "ink_red": "0.0",
        "ink_yellow": "0.0",
        "ink_green": "0.0",
        "ink_cyan": "0.0",
        "ink_blue": "0.0",
        "ink_magenta": "0.0"
    },
    "state_feedback": {
        "displacement": "next_displacement",
        "displacement_velocity": "next_velocity",
        "entity_x": "next_entity_x",
        "entity_y": "next_entity_y",
        "entity_velocity_x": "next_entity_velocity_x",
        "entity_velocity_y": "next_entity_velocity_y",
        "entity_b_x": "next_entity_b_x",
        "entity_b_y": "next_entity_b_y",
        "entity_b_velocity_x": "next_entity_b_velocity_x",
        "entity_b_velocity_y": "next_entity_b_velocity_y",
        "entity_c_x": "next_entity_c_x",
        "entity_c_y": "next_entity_c_y",
        "entity_c_velocity_x": "next_entity_c_velocity_x",
        "entity_c_velocity_y": "next_entity_c_velocity_y",
        "entity_cargo": "next_entity_cargo",
        "entity_b_cargo": "next_entity_b_cargo",
        "entity_c_cargo": "next_entity_c_cargo",
        "food_store": "next_food_store",
        "nest_food": "next_nest_food",
        "filter_reservoir": "next_filter_reservoir",
        "managed_time": "next_time",
        "ink_red": "next_ink_red",
        "ink_yellow": "next_ink_yellow",
        "ink_green": "next_ink_green",
        "ink_cyan": "next_ink_cyan",
        "ink_blue": "next_ink_blue",
        "ink_magenta": "next_ink_magenta"
    },
    "render_fps": 30.0,
    "autostart": True,
    "backend": "c",
    "remove_loops": True,
    "audio": {
        "generator": (
            "src.common.dt_system.fluid_mechanics.columnar_multifluid_audio:"
            "synthesize_columnar_audio"
        ),
        "arguments": {"duration": 8.0, "sample_rate": 24000, "feature_fps": 30},
        "managed_time_output": "next_time",
        "pan_output": "next_entity_x",
        "pan_range": [0.0, 10.0],
    },
}


_PAGE = "TURING_PAGE = " + repr(_PAGE_CONFIG)


SOURCE = "\n\n".join((
    _PAGE,
    inspect.getsource(columnar_multifluid_rgb_step),
    inspect.getsource(columnar_multifluid_present),
))


# Native/offline build of the identical authored tick.  Omitting the
# presentation entrypoint is intentional: this contract publishes the Python,
# SSA and generated Fortran plus its native fidelity proof, but no WebGL
# shader surface.  It is the backend-independent reference artifact while the
# browser presentation path is being qualified.
_FORTRAN_PAGE_CONFIG = {
    **_PAGE_CONFIG,
    "title": "Managed Columnar Multifluid World — Native Fortran",
    "slug": "managed-columnar-multifluid-world-fortran",
}
_FORTRAN_PAGE_CONFIG.pop("presentation_entrypoint", None)
_FORTRAN_PAGE = "TURING_PAGE = " + repr(_FORTRAN_PAGE_CONFIG)
FORTRAN_SOURCE = "\n\n".join((
    _FORTRAN_PAGE,
    inspect.getsource(columnar_multifluid_rgb_step),
))


__all__ = ["FORTRAN_SOURCE", "SOURCE", "columnar_multifluid_present"]
