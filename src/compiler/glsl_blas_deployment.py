"""Standalone GLSL deployments for source and intrinsic BLAS variants.

Both variants use the same three-slot arena ABI and the same canonical
``blas.gemm`` role.  The source variant preserves the authored reduction loop;
the intrinsic variant uses the GLSL backend's cooperative tiled identity.
Each artifact includes a dependency-free C dispatch shell which obtains its
own hidden SDL2 OpenGL context, loads the compute entry points, and compiles
the adjacent deterministic shader source without a Python or Turing runtime.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from src.common.tensors.blas import BLASRole, blas_role
from src.common.tensors.accelerator_backends.glsl_backend import (
    emit_matmul_source,
    emit_source_matmul_source,
    glsl_blas_shader_identity,
)
from src.compiler.deployment_lowering import (
    ComputeDispatchLimits,
    select_deployment_strategy,
)
from src.compiler.tiling_strategy import select_gemm_shader_dispatch


SCHEMA = "turing.glsl-blas-deployment.v1"


@dataclass(frozen=True, slots=True)
class WrittenGLSLBLASDeployment:
    shader_path: Path
    shell_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class GLSLBLASDeployment:
    name: str
    role: str
    variant: str
    shader_source: str
    shell_source: str
    manifest: Mapping[str, Any]

    def write(self, directory: str | Path) -> WrittenGLSLBLASDeployment:
        output = Path(directory).resolve()
        output.mkdir(parents=True, exist_ok=True)
        shader_path = output / f"{self.name}.comp.glsl"
        shell_path = output / f"{self.name}_shell.c"
        manifest_path = output / f"{self.name}.manifest.json"
        shader_path.write_text(self.shader_source, encoding="utf-8")
        shell_path.write_text(self.shell_source, encoding="utf-8")
        manifest_path.write_text(
            json.dumps(dict(self.manifest), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return WrittenGLSLBLASDeployment(
            shader_path, shell_path, manifest_path,
        )


def _identifier(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", str(value))
    if not result or result[0].isdigit():
        result = "turing_" + result
    return result


def _sha(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _validate_gemm_role(role: BLASRole) -> None:
    """Pin the finite AST shape this direct shader lowering implements."""

    if role.name != "gemm" or role.parameter_order != (
        "A", "B", "C", "alpha", "beta", "m", "n", "k"
    ):
        raise ValueError("GLSL GEMM deployment requires the canonical gemm role")
    tree = ast.parse(role.source)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    )
    loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
    loop_roles = tuple(
        (
            node.target.id,
            node.iter.args[0].id,
        )
        for node in loops
        if (
            isinstance(node.target, ast.Name)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
            and len(node.iter.args) == 1
            and isinstance(node.iter.args[0], ast.Name)
        )
    )
    if loop_roles != (("i", "m"), ("j", "n"), ("p", "k")):
        raise ValueError(
            "canonical GEMM source loop roles changed; update the finite AST "
            f"lowering deliberately: {loop_roles!r}"
        )


def emit_source_gemm_shader(
    m: int,
    n: int,
    k: int,
    *,
    local_size: int,
) -> str:
    """Mechanically preserve the canonical GEMM reduction in one shader."""

    role = blas_role("gemm")
    _validate_gemm_role(role)
    return emit_source_matmul_source(
        (m, k),
        (k, n),
        local_size=int(local_size),
    )


def _dispatch_shell(
    name: str,
    *,
    variant: str,
    shader_file: str,
    shape: tuple[int, int, int],
    slot_offsets: tuple[int, int, int],
    arena_elements: int,
    groups: tuple[int, int, int],
    count: int,
    extent: int,
    warmup_dispatches: int,
    measured_dispatches: int,
) -> str:
    symbol = _identifier(name)
    gx, gy, gz = map(int, groups)
    m, n, k = map(int, shape)
    slot_a, slot_b, slot_c = map(int, slot_offsets)
    template = r'''/* Generated standalone OpenGL compute deployment.
 * Owns a hidden SDL2 OpenGL 4.3 context and dynamically loads every symbol.
 * Build on Windows: zig cc -O3 @SHELL_FILE@ -o @NAME@.exe
 * Run: @NAME@.exe [shader.comp.glsl] [SDL2.dll] [optional-C-output.bin]
 * No Python or Turing runtime is used.
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SDL_INIT_VIDEO 0x00000020u
#define SDL_WINDOWPOS_UNDEFINED 0x1FFF0000
#define SDL_WINDOW_OPENGL 0x00000002u
#define SDL_WINDOW_HIDDEN 0x00000008u
#define SDL_GL_CONTEXT_MAJOR_VERSION 17
#define SDL_GL_CONTEXT_MINOR_VERSION 18
#define SDL_GL_CONTEXT_PROFILE_MASK 21
#define SDL_GL_CONTEXT_PROFILE_CORE 1

#define GL_COMPUTE_SHADER 0x91B9u
#define GL_COMPILE_STATUS 0x8B81u
#define GL_LINK_STATUS 0x8B82u
#define GL_SHADER_STORAGE_BUFFER 0x90D2u
#define GL_DYNAMIC_DRAW 0x88E8u
#define GL_SHADER_STORAGE_BARRIER_BIT 0x2000u
#define GL_BUFFER_UPDATE_BARRIER_BIT 0x0200u
#define GL_VERSION 0x1F02u

typedef unsigned int GLuint;
typedef unsigned int GLenum;
typedef int GLint;
typedef int GLsizei;
typedef char GLchar;
typedef ptrdiff_t GLsizeiptr;
typedef ptrdiff_t GLintptr;
typedef unsigned char GLubyte;

typedef int (__cdecl *PFN_SDL_InitSubSystem)(uint32_t);
typedef void (__cdecl *PFN_SDL_QuitSubSystem)(uint32_t);
typedef int (__cdecl *PFN_SDL_GL_SetAttribute)(int, int);
typedef void *(__cdecl *PFN_SDL_CreateWindow)(
    const char *, int, int, int, int, uint32_t);
typedef void *(__cdecl *PFN_SDL_GL_CreateContext)(void *);
typedef int (__cdecl *PFN_SDL_GL_MakeCurrent)(void *, void *);
typedef void *(__cdecl *PFN_SDL_GL_GetProcAddress)(const char *);
typedef void (__cdecl *PFN_SDL_GL_DeleteContext)(void *);
typedef void (__cdecl *PFN_SDL_DestroyWindow)(void *);
typedef const char *(__cdecl *PFN_SDL_GetError)(void);

typedef GLuint (APIENTRY *PFN_glCreateShader)(GLenum);
typedef void (APIENTRY *PFN_glShaderSource)(GLuint, GLsizei, const GLchar *const *, const GLint *);
typedef void (APIENTRY *PFN_glCompileShader)(GLuint);
typedef void (APIENTRY *PFN_glGetShaderiv)(GLuint, GLenum, GLint *);
typedef void (APIENTRY *PFN_glGetShaderInfoLog)(GLuint, GLsizei, GLsizei *, GLchar *);
typedef GLuint (APIENTRY *PFN_glCreateProgram)(void);
typedef void (APIENTRY *PFN_glAttachShader)(GLuint, GLuint);
typedef void (APIENTRY *PFN_glLinkProgram)(GLuint);
typedef void (APIENTRY *PFN_glGetProgramiv)(GLuint, GLenum, GLint *);
typedef void (APIENTRY *PFN_glGetProgramInfoLog)(GLuint, GLsizei, GLsizei *, GLchar *);
typedef void (APIENTRY *PFN_glDeleteShader)(GLuint);
typedef void (APIENTRY *PFN_glDeleteProgram)(GLuint);
typedef void (APIENTRY *PFN_glUseProgram)(GLuint);
typedef void (APIENTRY *PFN_glGenBuffers)(GLsizei, GLuint *);
typedef void (APIENTRY *PFN_glBindBuffer)(GLenum, GLuint);
typedef void (APIENTRY *PFN_glBufferData)(GLenum, GLsizeiptr, const void *, GLenum);
typedef void (APIENTRY *PFN_glBindBufferBase)(GLenum, GLuint, GLuint);
typedef GLint (APIENTRY *PFN_glGetUniformLocation)(GLuint, const GLchar *);
typedef void (APIENTRY *PFN_glUniform1ui)(GLint, GLuint);
typedef void (APIENTRY *PFN_glUniform1uiv)(GLint, GLsizei, const GLuint *);
typedef void (APIENTRY *PFN_glDispatchCompute)(GLuint, GLuint, GLuint);
typedef void (APIENTRY *PFN_glMemoryBarrier)(GLenum);
typedef void (APIENTRY *PFN_glFinish)(void);
typedef void (APIENTRY *PFN_glGetBufferSubData)(GLenum, GLintptr, GLsizeiptr, void *);
typedef void (APIENTRY *PFN_glDeleteBuffers)(GLsizei, const GLuint *);
typedef const GLubyte *(APIENTRY *PFN_glGetString)(GLenum);

static char *read_file(const char *path, size_t *size_out) {
    FILE *stream = fopen(path, "rb");
    char *data;
    long size;
    if (!stream) return NULL;
    if (fseek(stream, 0, SEEK_END) || (size = ftell(stream)) < 0) {
        fclose(stream); return NULL;
    }
    rewind(stream);
    data = (char *)malloc((size_t)size + 1u);
    if (!data || fread(data, 1u, (size_t)size, stream) != (size_t)size) {
        free(data); fclose(stream); return NULL;
    }
    fclose(stream);
    data[size] = '\0';
    if (size_out) *size_out = (size_t)size;
    return data;
}

static double seconds_now(void) {
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}

#define LOAD_SDL(field, type, symbol) do { \
    field = (type)(void *)GetProcAddress(sdl, symbol); \
    if (!(field)) { fprintf(stderr, "missing SDL symbol %s\n", symbol); goto cleanup; } \
} while (0)

#define LOAD_GL(field, type, symbol) do { \
    field = (type)sdl_gl_get_proc_address(symbol); \
    if (!(field)) { fprintf(stderr, "missing OpenGL symbol %s\n", symbol); goto cleanup; } \
} while (0)

int main(int argc, char **argv) {
    const char *shader_path = argc > 1 ? argv[1] : "@SHADER_FILE@";
    const char *sdl_path = argc > 2 ? argv[2] : "SDL2.dll";
    const char *output_path = argc > 3 ? argv[3] : NULL;
    HMODULE sdl = NULL;
    void *window = NULL, *context = NULL;
    char *shader_source = NULL;
    float *arena = NULL;
    GLuint shader = 0u, program = 0u, arena_buffer = 0u;
    int result = 1;

    PFN_SDL_InitSubSystem sdl_init = NULL;
    PFN_SDL_QuitSubSystem sdl_quit = NULL;
    PFN_SDL_GL_SetAttribute sdl_gl_set_attribute = NULL;
    PFN_SDL_CreateWindow sdl_create_window = NULL;
    PFN_SDL_GL_CreateContext sdl_gl_create_context = NULL;
    PFN_SDL_GL_MakeCurrent sdl_gl_make_current = NULL;
    PFN_SDL_GL_GetProcAddress sdl_gl_get_proc_address = NULL;
    PFN_SDL_GL_DeleteContext sdl_gl_delete_context = NULL;
    PFN_SDL_DestroyWindow sdl_destroy_window = NULL;
    PFN_SDL_GetError sdl_get_error = NULL;

    PFN_glCreateShader glCreateShader = NULL;
    PFN_glShaderSource glShaderSource = NULL;
    PFN_glCompileShader glCompileShader = NULL;
    PFN_glGetShaderiv glGetShaderiv = NULL;
    PFN_glGetShaderInfoLog glGetShaderInfoLog = NULL;
    PFN_glCreateProgram glCreateProgram = NULL;
    PFN_glAttachShader glAttachShader = NULL;
    PFN_glLinkProgram glLinkProgram = NULL;
    PFN_glGetProgramiv glGetProgramiv = NULL;
    PFN_glGetProgramInfoLog glGetProgramInfoLog = NULL;
    PFN_glDeleteShader glDeleteShader = NULL;
    PFN_glDeleteProgram glDeleteProgram = NULL;
    PFN_glUseProgram glUseProgram = NULL;
    PFN_glGenBuffers glGenBuffers = NULL;
    PFN_glBindBuffer glBindBuffer = NULL;
    PFN_glBufferData glBufferData = NULL;
    PFN_glBindBufferBase glBindBufferBase = NULL;
    PFN_glGetUniformLocation glGetUniformLocation = NULL;
    PFN_glUniform1ui glUniform1ui = NULL;
    PFN_glUniform1uiv glUniform1uiv = NULL;
    PFN_glDispatchCompute glDispatchCompute = NULL;
    PFN_glMemoryBarrier glMemoryBarrier = NULL;
    PFN_glFinish glFinish = NULL;
    PFN_glGetBufferSubData glGetBufferSubData = NULL;
    PFN_glDeleteBuffers glDeleteBuffers = NULL;
    PFN_glGetString glGetString = NULL;

    sdl = LoadLibraryA(sdl_path);
    if (!sdl) { fprintf(stderr, "cannot load SDL2: %s\n", sdl_path); goto cleanup; }
    LOAD_SDL(sdl_init, PFN_SDL_InitSubSystem, "SDL_InitSubSystem");
    LOAD_SDL(sdl_quit, PFN_SDL_QuitSubSystem, "SDL_QuitSubSystem");
    LOAD_SDL(sdl_gl_set_attribute, PFN_SDL_GL_SetAttribute, "SDL_GL_SetAttribute");
    LOAD_SDL(sdl_create_window, PFN_SDL_CreateWindow, "SDL_CreateWindow");
    LOAD_SDL(sdl_gl_create_context, PFN_SDL_GL_CreateContext, "SDL_GL_CreateContext");
    LOAD_SDL(sdl_gl_make_current, PFN_SDL_GL_MakeCurrent, "SDL_GL_MakeCurrent");
    LOAD_SDL(sdl_gl_get_proc_address, PFN_SDL_GL_GetProcAddress, "SDL_GL_GetProcAddress");
    LOAD_SDL(sdl_gl_delete_context, PFN_SDL_GL_DeleteContext, "SDL_GL_DeleteContext");
    LOAD_SDL(sdl_destroy_window, PFN_SDL_DestroyWindow, "SDL_DestroyWindow");
    LOAD_SDL(sdl_get_error, PFN_SDL_GetError, "SDL_GetError");

    if (sdl_init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL video init failed: %s\n", sdl_get_error()); goto cleanup;
    }
    sdl_gl_set_attribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    sdl_gl_set_attribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    sdl_gl_set_attribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    window = sdl_create_window(
        "@NAME@", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        1, 1, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
    if (!window) { fprintf(stderr, "SDL window failed: %s\n", sdl_get_error()); goto cleanup; }
    context = sdl_gl_create_context(window);
    if (!context || sdl_gl_make_current(window, context) != 0) {
        fprintf(stderr, "SDL GL context failed: %s\n", sdl_get_error()); goto cleanup;
    }

    LOAD_GL(glCreateShader, PFN_glCreateShader, "glCreateShader");
    LOAD_GL(glShaderSource, PFN_glShaderSource, "glShaderSource");
    LOAD_GL(glCompileShader, PFN_glCompileShader, "glCompileShader");
    LOAD_GL(glGetShaderiv, PFN_glGetShaderiv, "glGetShaderiv");
    LOAD_GL(glGetShaderInfoLog, PFN_glGetShaderInfoLog, "glGetShaderInfoLog");
    LOAD_GL(glCreateProgram, PFN_glCreateProgram, "glCreateProgram");
    LOAD_GL(glAttachShader, PFN_glAttachShader, "glAttachShader");
    LOAD_GL(glLinkProgram, PFN_glLinkProgram, "glLinkProgram");
    LOAD_GL(glGetProgramiv, PFN_glGetProgramiv, "glGetProgramiv");
    LOAD_GL(glGetProgramInfoLog, PFN_glGetProgramInfoLog, "glGetProgramInfoLog");
    LOAD_GL(glDeleteShader, PFN_glDeleteShader, "glDeleteShader");
    LOAD_GL(glDeleteProgram, PFN_glDeleteProgram, "glDeleteProgram");
    LOAD_GL(glUseProgram, PFN_glUseProgram, "glUseProgram");
    LOAD_GL(glGenBuffers, PFN_glGenBuffers, "glGenBuffers");
    LOAD_GL(glBindBuffer, PFN_glBindBuffer, "glBindBuffer");
    LOAD_GL(glBufferData, PFN_glBufferData, "glBufferData");
    LOAD_GL(glBindBufferBase, PFN_glBindBufferBase, "glBindBufferBase");
    LOAD_GL(glGetUniformLocation, PFN_glGetUniformLocation, "glGetUniformLocation");
    LOAD_GL(glUniform1ui, PFN_glUniform1ui, "glUniform1ui");
    LOAD_GL(glUniform1uiv, PFN_glUniform1uiv, "glUniform1uiv");
    LOAD_GL(glDispatchCompute, PFN_glDispatchCompute, "glDispatchCompute");
    LOAD_GL(glMemoryBarrier, PFN_glMemoryBarrier, "glMemoryBarrier");
    LOAD_GL(glFinish, PFN_glFinish, "glFinish");
    LOAD_GL(glGetBufferSubData, PFN_glGetBufferSubData, "glGetBufferSubData");
    LOAD_GL(glDeleteBuffers, PFN_glDeleteBuffers, "glDeleteBuffers");
    LOAD_GL(glGetString, PFN_glGetString, "glGetString");

    shader_source = read_file(shader_path, NULL);
    if (!shader_source) { fprintf(stderr, "cannot read shader: %s\n", shader_path); goto cleanup; }
    shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, (const GLchar *const *)&shader_source, NULL);
    glCompileShader(shader);
    {
        GLint ok = 0; GLchar log[8192]; GLsizei written = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok) { glGetShaderInfoLog(shader, 8191, &written, log); log[written] = 0;
            fprintf(stderr, "shader compile failed:\n%s\n", log); goto cleanup; }
    }
    program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    {
        GLint ok = 0; GLchar log[8192]; GLsizei written = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) { glGetProgramInfoLog(program, 8191, &written, log); log[written] = 0;
            fprintf(stderr, "program link failed:\n%s\n", log); goto cleanup; }
    }

    arena = (float *)calloc((size_t)@ARENA_ELEMENTS@, sizeof(float));
    if (!arena) { fprintf(stderr, "arena allocation failed\n"); goto cleanup; }
    for (uint32_t i = 0; i < @A_ELEMENTS@u; ++i)
        arena[@SLOT_A@u + i] = (float)((int)(i % 29u) - 14) / 17.0f;
    for (uint32_t i = 0; i < @B_ELEMENTS@u; ++i)
        arena[@SLOT_B@u + i] = (float)((int)(i % 31u) - 15) / 19.0f;

    glGenBuffers(1, &arena_buffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, arena_buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
        (GLsizeiptr)(@ARENA_ELEMENTS@u * sizeof(float)), arena, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0u, arena_buffer);
    glUseProgram(program);
    {
        const GLuint extents[1] = {@EXTENT@u};
        const GLuint slots[3] = {@SLOT_A@u, @SLOT_B@u, @SLOT_C@u};
        glUniform1ui(glGetUniformLocation(program, "u_count"), @COUNT@u);
        glUniform1uiv(glGetUniformLocation(program, "u_extent"), 1, extents);
        glUniform1uiv(glGetUniformLocation(program, "u_slot"), 3, slots);
    }
    glFinish();
    for (uint32_t iteration = 0; iteration < @WARMUPS@u; ++iteration) {
        glDispatchCompute(@GX@u, @GY@u, @GZ@u);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    }
    glFinish();
    {
        double started = seconds_now(), elapsed, checksum = 0.0;
        for (uint32_t iteration = 0; iteration < @ITERATIONS@u; ++iteration) {
            glDispatchCompute(@GX@u, @GY@u, @GZ@u);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
        }
        glFinish();
        elapsed = (seconds_now() - started) / (double)@ITERATIONS@u;
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,
            (GLintptr)(@SLOT_C@u * sizeof(float)),
            (GLsizeiptr)(@C_ELEMENTS@u * sizeof(float)), arena + @SLOT_C@u);
        for (uint32_t i = 0; i < @C_ELEMENTS@u; ++i)
            checksum += (double)arena[@SLOT_C@u + i];
        printf("{\"variant\":\"@VARIANT@\",\"m\":@M@,\"n\":@N@,\"k\":@K@,"
               "\"warmups\":@WARMUPS@,\"iterations\":@ITERATIONS@,"
               "\"groups\":[@GX@,@GY@,@GZ@],\"elapsed_ms\":%.9g,"
               "\"gflops\":%.9g,\"checksum\":%.17g,\"opengl\":\"%s\"}\n",
               elapsed * 1000.0,
               (2.0 * (double)@M@ * (double)@N@ * (double)@K@) / elapsed / 1.0e9,
               checksum, (const char *)glGetString(GL_VERSION));
    }
    if (output_path) {
        FILE *output = fopen(output_path, "wb");
        if (!output || fwrite(arena + @SLOT_C@u, sizeof(float), @C_ELEMENTS@u, output)
                != @C_ELEMENTS@u) {
            fprintf(stderr, "cannot write C output: %s\n", output_path);
            if (output) fclose(output); goto cleanup;
        }
        fclose(output);
    }
    result = 0;

cleanup:
    if (glUseProgram) glUseProgram(0u);
    if (arena_buffer && glDeleteBuffers) glDeleteBuffers(1, &arena_buffer);
    if (program && glDeleteProgram) glDeleteProgram(program);
    if (shader && glDeleteShader) glDeleteShader(shader);
    free(arena); free(shader_source);
    if (context && sdl_gl_delete_context) sdl_gl_delete_context(context);
    if (window && sdl_destroy_window) sdl_destroy_window(window);
    if (sdl_quit) sdl_quit(SDL_INIT_VIDEO);
    if (sdl) FreeLibrary(sdl);
    return result;
}
'''
    replacements = {
        "@NAME@": symbol,
        "@SHELL_FILE@": f"{name}_shell.c",
        "@SHADER_FILE@": shader_file,
        "@VARIANT@": str(variant),
        "@M@": str(m), "@N@": str(n), "@K@": str(k),
        "@GX@": str(gx), "@GY@": str(gy), "@GZ@": str(gz),
        "@COUNT@": str(int(count)), "@EXTENT@": str(int(extent)),
        "@WARMUPS@": str(int(warmup_dispatches)),
        "@ITERATIONS@": str(int(measured_dispatches)),
        "@SLOT_A@": str(slot_a), "@SLOT_B@": str(slot_b),
        "@SLOT_C@": str(slot_c),
        "@A_ELEMENTS@": str(m * k), "@B_ELEMENTS@": str(k * n),
        "@C_ELEMENTS@": str(m * n),
        "@ARENA_ELEMENTS@": str(int(arena_elements)),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def _deployment(
    *,
    role: BLASRole,
    variant: str,
    name: str,
    shader_source: str,
    shape: tuple[int, int, int],
    work: int,
    preferred_local_size: int,
    limits: ComputeDispatchLimits,
    warmup_dispatches: int,
    measured_dispatches: int,
) -> GLSLBLASDeployment:
    m, n, k = shape
    choice = select_deployment_strategy(
        backend="glsl",
        execution_class="shader-compute",
        work=int(work),
        preferred_local_size=int(preferred_local_size),
        compute_limits=limits,
    )
    if choice.compute is None:
        raise RuntimeError("GLSL BLAS deployment produced no compute geometry")
    groups = tuple(map(int, choice.compute.groups))
    local_size = int(choice.compute.workgroup_size[0])
    offsets = {"A": 0, "B": m * k, "C": m * k + k * n}
    arena_elements = m * k + k * n + m * n
    shader_file = f"{name}.comp.glsl"
    shell = _dispatch_shell(
        name,
        variant=variant,
        shader_file=shader_file,
        shape=shape,
        slot_offsets=(offsets["A"], offsets["B"], offsets["C"]),
        arena_elements=arena_elements,
        groups=groups,
        count=int(work),
        extent=int(m * n),
        warmup_dispatches=warmup_dispatches,
        measured_dispatches=measured_dispatches,
    )
    manifest = {
        "schema": SCHEMA,
        "name": name,
        "role": role.identity,
        "role_source_sha256": _sha(role.source),
        "role_parameter_order": list(role.parameter_order),
        "semantic_parameters": {"alpha": 1.0, "beta": 0.0},
        "variant": variant,
        "lowering": (
            {
                "kind": "finite_role_ast",
                "outer_loop_mapping": ["i,j", "global_invocation_id"],
                "reduction_loop": "p preserved in source order",
                "backend_identity": None,
            }
            if variant == "source_algorithm"
            else {
                "kind": "backend_identity",
                "identity": "glslblas_gemm",
                "algorithm": "cooperative shared-memory tiled GEMM",
            }
        ),
        "comparison": {
            "baseline": "driver optimization of source-order GEMM",
            "specialized": "compiler-selected GLSL cooperative tiling",
            "same_inputs_and_arena_abi": True,
        },
        "problem_shape": {"m": m, "n": n, "k": k},
        "shader_sha256": _sha(shader_source),
        "shader_plan_identity": glsl_blas_shader_identity(
            role_source=role.source,
            variant=variant,
            left_shape=(m, k),
            right_shape=(k, n),
            left_dtype=np.float32,
            right_dtype=np.float32,
            output_dtype=np.float32,
            local_size=local_size,
            shader_source=shader_source,
        ),
        "shell_sha256": _sha(shell),
        "shader_file": shader_file,
        "shell_file": f"{name}_shell.c",
        "arena_abi": {
            "binding": 0,
            "dtype": "float32",
            "slot_order": ["A", "B", "C"],
            "slot_offsets_elements": offsets,
            "arena_elements": arena_elements,
        },
        "recommended_dispatch": {
            "logical_work_items": int(work),
            "local_size": [local_size, 1, 1],
            "groups": list(groups),
            "reasons": list(choice.reasons),
        },
        "measurement": {
            "warmup_dispatches": int(warmup_dispatches),
            "measured_dispatches": int(measured_dispatches),
            "reported_time": "mean completed GPU dispatch wall time",
        },
        "standalone": {
            "python_runtime_dependency": False,
            "turing_runtime_dependency": False,
            "owns_hidden_opengl_context": True,
            "compiles_and_links_shader": True,
            "context_provider": "dynamically loaded SDL2",
            "platform": "windows",
            "minimum_opengl": "4.3",
        },
    }
    return GLSLBLASDeployment(
        name, role.identity, variant, shader_source, shell, manifest,
    )


def build_gemm_deployment_pair(
    m: int,
    n: int,
    k: int,
    *,
    limits: ComputeDispatchLimits,
    warmup_dispatches: int = 3,
    measured_dispatches: int = 20,
) -> tuple[GLSLBLASDeployment, GLSLBLASDeployment]:
    """Build comparable source-order and optimized standalone deployments."""

    m, n, k = int(m), int(n), int(k)
    if min(m, n, k) <= 0:
        raise ValueError("GEMM deployment dimensions must be positive")
    if warmup_dispatches < 0 or measured_dispatches < 1:
        raise ValueError("dispatch measurements require warmups >= 0 and iterations >= 1")
    role = blas_role("gemm")

    baseline_dispatch = select_gemm_shader_dispatch(
        m, n, k, backend="glsl", limits=limits,
        variant="source_algorithm",
    )
    baseline_local = baseline_dispatch.choice.compute.workgroup_size[0]
    baseline_source = emit_source_gemm_shader(
        m, n, k, local_size=baseline_local,
    )
    baseline = _deployment(
        role=role,
        variant="source_algorithm",
        name=f"blas_gemm_source_{m}_{n}_{k}",
        shader_source=baseline_source,
        shape=(m, n, k),
        work=baseline_dispatch.logical_work,
        preferred_local_size=baseline_local,
        limits=limits,
        warmup_dispatches=warmup_dispatches,
        measured_dispatches=measured_dispatches,
    )

    intrinsic_dispatch = select_gemm_shader_dispatch(
        m, n, k, backend="glsl", limits=limits,
        variant="glslblas_gemm",
    )
    intrinsic_local = intrinsic_dispatch.choice.compute.workgroup_size[0]
    intrinsic_source = emit_matmul_source(
        (m, k), (k, n), local_size=intrinsic_local,
    )
    intrinsic = _deployment(
        role=role,
        variant="glslblas_gemm",
        name=f"blas_gemm_glslblas_{m}_{n}_{k}",
        shader_source=intrinsic_source,
        shape=(m, n, k),
        work=intrinsic_dispatch.logical_work,
        preferred_local_size=intrinsic_local,
        limits=limits,
        warmup_dispatches=warmup_dispatches,
        measured_dispatches=measured_dispatches,
    )
    return baseline, intrinsic


__all__ = [
    "GLSLBLASDeployment",
    "SCHEMA",
    "WrittenGLSLBLASDeployment",
    "build_gemm_deployment_pair",
    "emit_source_gemm_shader",
]
