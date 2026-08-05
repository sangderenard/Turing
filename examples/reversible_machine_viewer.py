"""Native shader display for a live or resumed reversible AMD64 machine."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE
from OpenGL import GL as gl
from OpenGL.GL.shaders import compileProgram, compileShader

from src.compiler.binary_machine_program import BinaryMachineProgram
from src.compiler.machine_state_buffer import (
    MachineRunDirection, SubjectOutputFormat, SubjectOutputKind,
)
from src.compiler.machine_system_ports import deterministic_windows_bootstrap_port
from src.compiler.shell_io import VirtualFileSystemContract, VirtualMount
from src.compiler.virtual_filesystem import VirtualFileSystemState


VERTEX_SHADER = """#version 330 core
const vec2 positions[3] = vec2[3](
  vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0)
);
out vec2 uv;
void main() {
  vec2 p = positions[gl_VertexID];
  uv = p * 0.5 + 0.5;
  gl_Position = vec4(p, 0.0, 1.0);
}
"""


FRAGMENT_SHADER = """#version 330 core
in vec2 uv;
out vec4 display_color;
uniform usampler2D register_words;
uniform sampler2D subject_output;
uniform vec2 machine_shape;
uniform float subject_available;
uniform float gpu_active;
void main() {
  vec3 chip = vec3(0.012, 0.018, 0.028);
  if (uv.y >= 0.50 && machine_shape.x > 0.0 && machine_shape.y > 0.0) {
    vec2 q = vec2(uv.x, (uv.y - 0.50) * 2.0);
    ivec2 cell = ivec2(clamp(floor(q * machine_shape), vec2(0.0), machine_shape - 1.0));
    uvec2 words = texelFetch(register_words, cell, 0).rg;
    float lo = log2(1.0 + float(words.x)) / 32.0;
    float hi = log2(1.0 + float(words.y)) / 32.0;
    float occupied = float((words.x | words.y) != 0u);
    vec2 grid = fract(q * machine_shape);
    float border = step(0.045, grid.x) * step(0.08, grid.y);
    chip = border * vec3(0.07 + lo, 0.13 + hi, 0.23 + 0.72 * occupied);
  } else if (uv.y < 0.50 && subject_available > 0.5) {
    chip = texture(subject_output, vec2(uv.x, uv.y * 2.0)).rgb;
  }
  vec3 lamp_color = mix(vec3(0.07, 0.08, 0.10), vec3(0.12, 1.0, 0.40), gpu_active);
  float lamp = 1.0 - smoothstep(0.022, 0.038, distance(uv, vec2(0.95, 0.055)));
  display_color = vec4(mix(chip, lamp_color, lamp), 1.0);
}
"""


class NativeMachineDisplay:
    def __init__(self, width: int = 1280, height: int = 800) -> None:
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("Turing reversible AMD64 machine")
        self.program = compileProgram(
            compileShader(VERTEX_SHADER, gl.GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, gl.GL_FRAGMENT_SHADER),
        )
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)
        self.register_texture = self._texture(integer=True)
        self.subject_texture = self._texture(integer=False)
        self.shape_location = gl.glGetUniformLocation(self.program, "machine_shape")
        self.subject_location = gl.glGetUniformLocation(self.program, "subject_available")
        self.active_location = gl.glGetUniformLocation(self.program, "gpu_active")
        gl.glUseProgram(self.program)
        gl.glUniform1i(gl.glGetUniformLocation(self.program, "register_words"), 0)
        gl.glUniform1i(gl.glGetUniformLocation(self.program, "subject_output"), 1)
        self.font = pygame.font.SysFont("Consolas", 20)
        self.last_generation = -1
        self.subject_available = 0.0

    @staticmethod
    def _texture(*, integer: bool) -> int:
        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        return texture

    def _upload_terminal(self, payload: bytes) -> None:
        surface = pygame.Surface((1024, 512))
        surface.fill((5, 8, 10))
        text = payload.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        rows: list[str] = []
        for line in text.split("\n"):
            if line:
                rows.extend(line[index:index + 96] for index in range(0, len(line), 96))
            else:
                rows.append("")
        for row, line in enumerate(rows[-24:]):
            surface.blit(self.font.render(line, True, (154, 255, 182)), (18, 14 + row * 20))
        pixels = pygame.image.tostring(surface, "RGBA", True)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.subject_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, 1024, 512, 0,
            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, pixels,
        )
        self.subject_available = 1.0

    def upload(self, snapshot) -> None:
        if snapshot.header.generation == self.last_generation:
            return
        self.last_generation = snapshot.header.generation
        width, height = snapshot.header.register_count, snapshot.header.core_count
        packed = bytearray(width * height * 8)
        for core in range(height):
            source = snapshot.header.register_offset + core * snapshot.header.register_stride_bytes
            begin = core * width * 8
            packed[begin:begin + width * 8] = snapshot.data[source:source + width * 8]
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.register_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RG32UI, width, height, 0,
            gl.GL_RG_INTEGER, gl.GL_UNSIGNED_INT, packed,
        )
        gl.glUseProgram(self.program)
        gl.glUniform2f(self.shape_location, width, height)
        self.subject_available = 0.0
        for index in range(snapshot.header.output_count):
            descriptor = snapshot.output_descriptor(index)
            payload = bytes(snapshot.output_bytes(index))
            if (
                descriptor.kind is SubjectOutputKind.TERMINAL
                and descriptor.format is SubjectOutputFormat.UTF8
            ):
                self._upload_terminal(payload)
                break
            if (
                descriptor.kind is SubjectOutputKind.FRAMEBUFFER
                and descriptor.format is SubjectOutputFormat.RGBA8
                and descriptor.width * descriptor.height * 4 <= len(payload)
            ):
                gl.glActiveTexture(gl.GL_TEXTURE1)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.subject_texture)
                gl.glTexImage2D(
                    gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8,
                    descriptor.width, descriptor.height, 0,
                    gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, payload,
                )
                self.subject_available = 1.0
                break

    def draw(self, *, gpu_active: bool) -> None:
        width, height = pygame.display.get_window_size()
        gl.glViewport(0, 0, width, height)
        gl.glClearColor(0.01, 0.015, 0.025, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glUseProgram(self.program)
        gl.glBindVertexArray(self.vao)
        gl.glUniform1f(self.subject_location, self.subject_available)
        gl.glUniform1f(self.active_location, float(gpu_active))
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        pygame.display.flip()


def _load(options) -> BinaryMachineProgram:
    if options.tape and options.tape.exists() and not options.new:
        return BinaryMachineProgram.load_system_tape(
            options.tape, maximum_file_size=128 * 1024 * 1024,
            machine_block_backend=(
                None if options.machine_backend == "translated" else options.machine_backend
            ),
        )
    subject = options.binary.read_bytes()
    filesystem = VirtualFileSystemState.create(
        VirtualFileSystemContract(
            current_directory="/c/work",
            mounts=(VirtualMount.create("/", "memory", access="read_write"),),
        ),
        files={"/c/windows/system32/cmd.exe": subject},
    )
    return BinaryMachineProgram.load_pe(
        subject, maximum_file_size=128 * 1024 * 1024,
        virtual_filesystem=filesystem,
        virtual_environment={
            "COMSPEC": r"C:\Windows\System32\cmd.exe",
            "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        },
        machine_block_backend=(
            None if options.machine_backend == "translated" else options.machine_backend
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path(os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe")))
    parser.add_argument("--tape", type=Path)
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--clocked", action="store_true", help="use shell ticks instead of maximum-speed free spin")
    parser.add_argument("--speed", type=float, default=60_000.0)
    parser.add_argument(
        "--machine-backend", choices=("translated", "node-wasm"),
        default="translated",
        help="select translated Python blocks or automatic journalled Wasm safe prefixes",
    )
    parser.add_argument("command", nargs="*", default=["/c", "echo hello"])
    options = parser.parse_args(argv)
    machine = _load(options)
    machine.set_speed(options.speed)
    port = deterministic_windows_bootstrap_port(
        arguments=("cmd.exe", *options.command),
        environment=(
            r"COMSPEC=C:\Windows\System32\cmd.exe",
            "PATHEXT=.COM;.EXE;.BAT;.CMD",
        ),
        module_virtual_path="/c/windows/system32/cmd.exe",
    )
    display = NativeMachineDisplay()
    clock = pygame.time.Clock()
    free_spin = not options.clocked
    if free_spin:
        machine.runner.start(MachineRunDirection.FORWARD)
    else:
        machine.set_direction(MachineRunDirection.FORWARD)
    running = True
    try:
        while running:
            elapsed = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    direction = (
                        MachineRunDirection.FORWARD
                        if machine.runner.direction is MachineRunDirection.PAUSED
                        else MachineRunDirection.PAUSED
                    )
                    machine.set_direction(direction)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_b:
                    machine.set_direction(MachineRunDirection.BACKWARD)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                    machine.set_direction(MachineRunDirection.FORWARD)
            if machine.pending_external_requests():
                machine.service_external_requests(port)
                machine.set_direction(MachineRunDirection.FORWARD)
            elif (
                machine.runner.direction is MachineRunDirection.PAUSED
                and machine.runner._last_results
                and machine.runner._last_results[0].status.name == "BLOCKED_CONTROL"
                and machine.service_dispatch_frontiers(core_index=0)
            ):
                machine.set_direction(MachineRunDirection.FORWARD)
            if not free_spin and machine.runner.direction is not MachineRunDirection.PAUSED:
                machine.tick(elapsed)
            with machine.snapshots.lease_latest() as snapshot:
                if snapshot is not None:
                    display.upload(snapshot)
            display.draw(gpu_active=machine.runner.direction is not MachineRunDirection.PAUSED)
            if machine.runner.failure is not None:
                raise machine.runner.failure
    finally:
        if machine.runner.running:
            machine.runner.stop()
        if options.tape:
            machine.save_system_tape(options.tape)
        machine.close()
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
