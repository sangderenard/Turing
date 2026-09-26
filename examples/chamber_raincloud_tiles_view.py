"""The chamber as a grid of tiles, one per cell, in an OpenGL window.

A copy of chamber_raincloud_view.py with the mesh display taken out and
chamber_tile_shader.TileGrid put in its place.  The original is untouched
and still renders the room as lit geometry with the air as an additive
slice stack; this one has no camera, no meshes and no field volume,
because a tile grid needs none of them.

What each tile shows is the state local to ITS OWN CELL, and a cell is
rarely doing one thing -- it can be humid, hold cloud, and have rain
falling through it at once.  So the tile cycles: every indicator present
above a threshold takes the tile in turn, with a crossfade, and a row of
pips along the bottom edge says how many are in the rotation.  Watching
one tile for a couple of seconds tells you everything that cell is doing.

The stepping, the recording and the HUD are unchanged from the original.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_SPECTRAL_DIR = HERE.parent.parent / "spectral-analyzer"
if str(_SPECTRAL_DIR) not in sys.path:
    sys.path.insert(0, str(_SPECTRAL_DIR))

import base_gl_renderer as _base_gl_renderer          # noqa: E402
from OpenGL.GL import (                               # noqa: E402
    GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_STREAM_DRAW, GL_FLOAT, GL_INT, GL_FALSE, GL_ONE,
    GL_TEXTURE0, GL_TEXTURE_2D, GL_TEXTURE_3D, GL_RGB32F, GL_RGB, GL_LINEAR, GL_CLAMP_TO_EDGE,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R,
    GL_DEPTH_TEST, GL_BLEND, GL_LEQUAL, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_RGBA, GL_UNSIGNED_BYTE,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_TRIANGLE_STRIP, GL_UNPACK_ALIGNMENT,
    glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer, glBufferData,
    glDeleteVertexArrays, glDeleteBuffers,
    glEnableVertexAttribArray, glVertexAttribPointer, glVertexAttribIPointer,
    glGenTextures, glBindTexture, glTexImage2D, glTexImage3D, glTexSubImage3D, glTexParameteri,
    glEnable, glDisable, glDepthFunc, glClearColor, glClear, glViewport, glReadPixels,
    glUseProgram, glUniform1i, glGetUniformLocation, glDrawArrays, glPixelStorei, glBlendFunc,
    glActiveTexture,
)
from OpenGL.GL import shaders as gl_shaders                  # noqa: E402
from chamber_tile_shader import Autoscale, TileGrid, plane   # noqa: E402

# -- display optics: real numbers, so the gain is a gain and not a lie --------
RHO_W = 1000.0
R_EFF_CLOUD = 7.0e-6          # m; the droplet the recorder's m_unit is sized for
R_RAIN = 5.0e-4               # m; the recorder's m_unit_r
CLOUD_RGB = np.array([0.96, 0.96, 0.98])
ICE_RGB = np.array([0.86, 0.93, 1.00])
RAIN_RGB = np.array([0.62, 0.68, 0.78])
LIGHT_FROM_ABOVE = np.array([1.0, 0.97, 0.92])
# The ambient term stands in for the multiple scattering a single-scatter
# reduction cannot do -- which is what makes a real cloud white -- so it is
# near-neutral with only a faint blue bias, not a sky colour.  A blue fill
# outweighs the sideways-scattered sun (HG(g=0.62) at 90 deg is ~0.03) and
# tints the whole cloud blue.
SKY_FILL = np.array([0.30, 0.31, 0.34])
G_HG = 0.62

_HUD_VERT = """
#version 330 core
layout(location=0) in vec2 pos;
layout(location=1) in vec2 uv;
out vec2 v_uv;
void main() { gl_Position = vec4(pos, 0.0, 1.0); v_uv = uv; }
"""
_HUD_FRAG = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
out vec4 frag_color;
void main() { frag_color = texture(u_tex, v_uv); }
"""


class SpeciesHud:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.program = gl_shaders.compileProgram(
            gl_shaders.compileShader(_HUD_VERT, 0x8B31),
            gl_shaders.compileShader(_HUD_FRAG, 0x8B30),
        )
        quad = np.array([-1, -1, 0, 0, 1, -1, 1, 0,
                         -1, 1, 0, 1, 1, 1, 1, 1], np.float32)
        self.vao = glGenVertexArrays(1); glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STREAM_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, False, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 2, GL_FLOAT, False, 16, ctypes.c_void_p(8))
        glBindVertexArray(0)
        self.texture = glGenTextures(1); glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)
        self.font = None

    def draw(self, pygame, lines):
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        if self.font is None:
            self.font = pygame.font.SysFont("consolas", 14)
        font = self.font
        line_height = 17
        panel_height = 10 + line_height * len(lines)
        pygame.draw.rect(surface, (4, 8, 10, 178),
                         (5, 5, self.width - 10, panel_height), border_radius=4)
        y = 8
        for line in lines:
            surface.blit(font.render(line, True, (0, 0, 0)), (10, y + 1))
            surface.blit(font.render(line, True, (218, 240, 228)), (9, y))
            y += line_height
        data = pygame.image.tostring(surface, "RGBA", True)
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, self.texture); glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(self.program); glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, self.texture)
        glUniform1i(glGetUniformLocation(self.program, "u_tex"), 0)
        glBindVertexArray(self.vao); glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0); glBindTexture(GL_TEXTURE_2D, 0); glUseProgram(0)
        glEnable(GL_DEPTH_TEST)


def species_hud_lines(rec, frame, playing):
    def scalar(name, default=float("nan")):
        return float(np.asarray(rec.arrays.get(name, [[default]])[frame]).reshape(-1)[0])
    regime = {0: "closed", 1: "supply", -1: "vent", 2: "mixed"}.get(
        int(scalar("port_regime", 0.0)), "unknown")
    chemistry = getattr(rec, "meta", {}).get("chemistry", {})
    chemistry_lines = []
    if chemistry:
        phases = {}
        for state in chemistry["states"]:
            phase = state.rsplit("@", 1)[-1]
            phases[phase] = phases.get(phase, 0) + 1
        phase_text = " ".join(f"{name}:{count}" for name, count in phases.items())
        reactions = chemistry["reactions"]
        chemistry_lines = [
            f"CHEM ABI {chemistry['abi'][:12]}  species={len(chemistry['species'])}  "
            f"states={len(chemistry['states'])} [{phase_text}]  reactions={len(reactions)}",
            "CHEM species  " + "  ".join(chemistry["species"]),
            "CHEM rxn  " + "  ".join(reactions[:4]),
            "          " + "  ".join(reactions[4:]),
            f"CHEM pack  owners={chemistry['owners']}  interfaces={chemistry['boundaries']}  "
            f"Precision[{chemistry['precision_limbs']}]  inventory={chemistry['inventory_bytes']} B  "
            f"exchange={chemistry['exchange_bytes']} B",
            f"CHEM execution  {chemistry['mode']}",
        ]
    requested = scalar("machine_requested", 0.0)
    accepted = scalar("machine_accepted", 0.0)
    rejected = max(0.0, requested - accepted)
    return [
        f"{'LIVE' if playing else 'PAUSED'}  chamber t={float(rec.t[-1]):.5g}s  dt={scalar('dt'):.3e}s  "
        f"machine t={scalar('machine_elapsed', 0.0):.5g}s  [SPACE] pause  [H] HUD",
        f"COMMIT requested={requested:.5g}s  accepted={accepted:.5g}s  refused={rejected:.3g}s  "
        f"acceptance={(accepted / requested if requested > 0 else 0.0):.1%}",
        f"GAS T={scalar('T'):.3f}K  P={scalar('P'):.2f}Pa  saturation={scalar('S'):.6f}  "
        f"dry-air={scalar('m_a'):.6e}kg",
        f"H2O vapor={scalar('m_v'):.6e}  cloud={scalar('m_l'):.6e}  "
        f"ice={scalar('m_i'):.6e}  rain={scalar('m_r'):.6e} kg",
        f"HYDROMETEORS LWC={scalar('LWC'):.6e}  IWC={scalar('IWC'):.6e}  "
        f"RWC={scalar('RWC'):.6e} kg/m3",
        f"PORT {regime}  radius={scalar('port_radius', 0.0) * 1e3:.3f}mm  "
        f"dP={scalar('port_dp', 0.0):+.2f}Pa  flow={scalar('port_mass_flow', 0.0) * 1e3:+.6f}g/s  "
        f"ambient RH={scalar('ambient_rh'):.1%}",
        f"DAMAGE PORTS count={int(scalar('damage_port_count', 0.0))}  "
        f"area={scalar('damage_port_area', 0.0) * 1e6:.3f}mm2  "
        f"dP/dt={scalar('pressure_drop_rate', 0.0):.3f}Pa/s  "
        f"event={'SUDDEN DECOMPRESSION' if scalar('decompression', 0.0) > 0.5 else 'none'}",
        f"COLD HEAD tip={scalar('cold_tip_temperature', 0.0):.3f}K  "
        f"gas flow={scalar('working_gas_mass_flow', 0.0) * 1e3:.3f}g/s  "
        f"removed={scalar('cold_head_heat_removed', 0.0):.2f}W  "
        f"atmosphere load={scalar('cold_tip_heat', 0.0):+.4f}W",
        f"SURFACE film={scalar('surface0.h_film', 0.0) * 1e6:.3f}um  "
        f"frost={scalar('surface0.h_frost', 0.0) * 1e6:.3f}um  "
        f"pool depth={scalar('pool0.h_pool', 0.0) * 1e3:.4f}mm  wet area={scalar('pool0.A_wet', 0.0):.6f}m2",
        f"VACUUM P={scalar('vacuum_pressure', 0.0):.6g}Pa  "
        f"integrity={'OK' if scalar('vacuum_intact', 1.0) > 0.5 else 'LOST'}",
        f"COMPRESSOR case={scalar('compressor_temperature', 0.0):.3f}K  "
        f"fan={scalar('compressor_fan', 0.0):.2%}  "
        f"shaft={scalar('compressor_shaft_power', 0.0):.1f}W  "
        f"radiator rejection={scalar('radiator_rejection', 0.0):.1f}W  "
        f"expander shaft={scalar('expander_shaft_power', 0.0):.1f}W",
        f"PLANT battery={scalar('battery_soc', 0.0):.3%}  makeup={scalar('working_gas_fill', 0.0):.3%}  "
        f"drain tank={scalar('drain_tank_fill', 0.0):.3%}  drain blockage={scalar('drain_blocked', 0.0):.3%}  "
        f"events={int(scalar('maintenance_count', 0.0))}",
        f"ENGINE ENGINE t={scalar('engine_system_time', 0.0):.5g}s  "
        f"rpm={scalar('engine_rpm', 0.0):.1f}  shaft={scalar('engine_power', 0.0):.1f}W  "
        f"alternator={scalar('engine_alternator_power', 0.0):.1f}W  "
        f"bus={scalar('engine_battery_voltage', 0.0):.2f}V  "
        f"engine battery={scalar('engine_battery_soc', 0.0):.3%}",
        f"ELECTRICAL t={scalar('electrical_system_time', 0.0):.5g}s  "
        f"compressor={scalar('compressor_motor_current', 0.0):.2f}A / "
        f"{scalar('compressor_power', 0.0):.1f}W  "
        f"wire+junction+motor heat={scalar('electrical_heat', 0.0):.2f}W  "
        f"safety events={int(scalar('electrical_safety_events', 0.0))}",
        f"SLIPPING CLOCKS requested={float(rec.t[-1]):.5g}s  "
        f"engine={scalar('engine_system_time', 0.0):.5g}s  "
        f"fluid={scalar('fluid_system_time', 0.0):.5g}s  "
        f"electrical={scalar('electrical_system_time', 0.0):.5g}s  "
        f"thermal={scalar('thermal_system_time', 0.0):.5g}s",
        *chemistry_lines,
    ]


# -- the recording ---------------------------------------------------------------

class Recording:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["__meta__"]))
        self.arrays = {k: z[k] for k in z.files if k != "__meta__"}
        self.nx, self.ny, self.nz = self.meta["shape"]
        self.dx = float(self.meta["dx"])
        self.n_frames = int(self.meta["n_frames"])
        self.t = self.arrays["t"].reshape(-1)
        # semantic -> output name, from the laws' own declaration
        self.by_semantic = {v["semantic"]: k for k, v in self.meta["publications"].items()}

    def channel(self, semantic: str, frame: int, default=None):
        name = self.by_semantic.get(semantic)
        if name is None or name not in self.arrays:
            return default
        return self.arrays[name][frame]

    def field(self, name: str, frame: int):
        """A voxel column (flat x + nx*(y + ny*z)) as (nz, ny, nx)."""
        return self.arrays[name][frame].reshape(self.nz, self.ny, self.nx)

    def frame_at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t), 0, self.n_frames - 1))


# -- the mesh-based display was here ------------------------------------------
# radiance_field, RoomMaterials, Mesh, quad, slice_stack, disc, look_at,
# perspective, model, _AdditiveRenderer, FieldTexture and RainRepresentatives
# built a lit box of triangles with the air as an additive slice stack and the
# rain as representative streaks. A tile grid needs none of it: there is no
# camera, no geometry and no volume texture, so it is gone rather than left
# dead. chamber_raincloud_view.py still holds the original, unchanged.

# -- the window --------------------------------------------------------------------

def main(argv=None, *, recording=None, live_step=None,
         atmosphere_visible=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", nargs="?", default=str(HERE / "chamber_raincloud_run.npz"))
    ap.add_argument("--speed", type=float, default=2.0, help="seconds of weather per wall second")
    ap.add_argument("--gain", type=float, default=1.0, help="multiplier on the field's light")
    ap.add_argument("--cycle", type=float, default=1.6,
                    help="seconds each indicator holds its tile")
    ap.add_argument("--size", type=int, nargs=2, default=(960, 720))
    ap.add_argument("--snapshot", type=str, default=None)
    ap.add_argument("--snapshot-at", type=float, default=None, help="weather time to snapshot")
    ap.add_argument("--exit-after", type=float, default=None, help="wall seconds")
    ap.add_argument("--no-hud", action="store_true", help="start with the species HUD hidden")
    ap.add_argument("--hide-atmosphere", action="store_true",
                    help="run the simulation without uploading or drawing the atmosphere")
    args = ap.parse_args(argv)

    rec = recording if recording is not None else Recording(args.recording)
    nx, ny, nz, dx = rec.nx, rec.ny, rec.nz, rec.dx
    col_w, col_h, col_d = nx * dx, nz * dx, ny * dx          # room metres: x, y(up), z

    import pygame
    from pygame.locals import OPENGL, DOUBLEBUF, QUIT, KEYDOWN, K_SPACE, K_LEFT, K_RIGHT, K_r, K_h, K_ESCAPE, \
        MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, MOUSEWHEEL
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    W, H = args.size
    pygame.display.set_mode((W, H), OPENGL | DOUBLEBUF)
    pygame.display.set_caption("raincloud -- chamber_raincloud_demo through base_material")

    hud = SpeciesHud(W, H)
    show_hud = not args.no_hud
    # THE DISPLAY: one tile per cell, each cycling its own indicators.
    # The room is no longer a box of meshes lit by a Phong rig -- there is
    # no camera, no slice stack and no field volume, because a tile grid
    # needs none of them. What the chamber is doing is read straight off
    # the cells and handed to the tile shader as two float textures.
    tiles = TileGrid(nx, nz, cycle_s=args.cycle)
    scales = {name: Autoscale() for name in ("S", "LWC", "IWC", "RWC", "T")}
    t_weather, playing = 0.0, True
    snapshot_done = args.snapshot is None
    t_wall0 = time.time(); last = t_wall0
    clock = pygame.time.Clock()

    while True:
        for ev in pygame.event.get():
            if ev.type == QUIT or (ev.type == KEYDOWN and ev.key == K_ESCAPE):
                pygame.quit(); return
            if ev.type == KEYDOWN:
                if ev.key == K_SPACE: playing = not playing
                elif ev.key == K_h: show_hud = not show_hud
                elif ev.key == K_LEFT: t_weather = max(0.0, t_weather - 2.0)
                elif ev.key == K_RIGHT: t_weather = min(rec.t[-1], t_weather + 2.0)
                elif ev.key == K_r: t_weather = 0.0
        now = time.time(); dt_wall = now - last; last = now
        if live_step is not None and playing:
            live_step()
            t_weather = float(rec.t[-1])
        elif playing:
            t_weather += dt_wall * args.speed
            if t_weather > rec.t[-1]:
                t_weather = 0.0
        frame = rec.frame_at(t_weather)
        draw_atmosphere = (
            bool(atmosphere_visible()) if atmosphere_visible is not None
            else not args.hide_atmosphere
        )

        # -- what every cell is doing, as two float textures --------
        # One texel per cell. The condensable channels go in one, and
        # temperature with the surface films in the other; the shader
        # decides which of them are present in a cell and cycles those.
        def cell_plane(name):
            values = rec.arrays[name][frame] if name in rec.arrays else 0.0
            return plane(values, nx, ny, nz)

        state = np.zeros((nz, nx, 4), dtype=np.float32)
        for channel, name in enumerate(("S", "LWC", "IWC", "RWC")):
            state[:, :, channel] = scales[name](cell_plane(name))
        tiles.state.upload(state)

        # surface0 is the ceiling plate (build() lists it first); the
        # recorder keys surface channels by index, not by semantic, since
        # there are two of them.
        h_film = float(np.max(rec.arrays["surface0.h_film"][frame])) if "surface0.h_film" in rec.arrays else 0.0
        h_frost = float(np.max(rec.arrays["surface0.h_frost"][frame])) if "surface0.h_frost" in rec.arrays else 0.0
        h_pool = float(rec.arrays["pool0.h_pool"][frame][0]) if "pool0.h_pool" in rec.arrays else 0.0

        aux = np.zeros((nz, nx, 4), dtype=np.float32)
        aux[:, :, 0] = scales["T"](cell_plane("T"))
        # the films belong to the boundary rows: the plate is the top row
        # of tiles, the floor the bottom one, which is where a film or a
        # frost layer actually is
        aux[nz - 1, :, 1] = float(np.clip(h_film / 2e-4, 0.0, 1.0))
        aux[nz - 1, :, 2] = float(np.clip(h_frost / 5e-4, 0.0, 1.0))
        aux[0, :, 1] = float(np.clip(h_pool / 0.01, 0.0, 1.0))
        tiles.aux.upload(aux)

        glViewport(0, 0, W, H)
        glClearColor(0.03, 0.035, 0.05, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        tiles.draw(time.time() - t_wall0)

        if show_hud:
            hud.draw(pygame, species_hud_lines(rec, frame, playing))

        lwc_top = float(rec.arrays["LWC"][frame][-1]) if "LWC" in rec.arrays else float("nan")
        pygame.display.set_caption(
            f"raincloud  t={t_weather:6.1f} s  frame {frame}/{rec.n_frames}  LWC top={lwc_top:.2e} kg/m3  "
            f"film={h_film * 1e6:.0f} um  frost={h_frost * 1e6:.0f} um  pool={h_pool * 1000:.1f} mm  {'||' if not playing else '>'}")

        if not snapshot_done and (args.snapshot_at is None or t_weather >= args.snapshot_at):
            buf = glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE)
            surf = pygame.image.frombuffer(buf, (W, H), "RGBA")
            pygame.image.save(pygame.transform.flip(surf, False, True), args.snapshot)
            print(f"snapshot at t={t_weather:.1f} s -> {args.snapshot}", flush=True)
            snapshot_done = True
        pygame.display.flip()
        if args.exit_after is not None and time.time() - t_wall0 > args.exit_after:
            pygame.quit(); return
        clock.tick(60)


if __name__ == "__main__":
    main()
