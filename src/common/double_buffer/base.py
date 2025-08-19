import time
import threading
from collections import defaultdict, deque
import numpy as np
import random
import os
import queue  # <-- Add this for LockManagerThread and elsewhere
import importlib
import subprocess
import sys

try:  # Optional heavy deps
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None  # type: ignore

def ensure_package(pkg_name, import_name=None):
    """Best-effort import helper that skips installation if missing."""
    module_name = import_name or pkg_name
    try:
        return importlib.import_module(module_name)
    except Exception:  # pragma: no cover
        return None

# ---- install & import PyOpenGL and PyCUDA if needed ----
try:
    gl = ensure_package("PyOpenGL", "OpenGL.GL")
    _gl_all = gl  # for `from OpenGL.GL import *`
    shaders = ensure_package("PyOpenGL", "OpenGL.GL.shaders")
    from OpenGL.GL import *  # type: ignore
    from OpenGL.GL.shaders import compileShader, compileProgram  # type: ignore
except Exception:  # pragma: no cover
    gl = _gl_all = shaders = None  # type: ignore
    def compileShader(*args, **kwargs):  # type: ignore
        raise RuntimeError("OpenGL not available")
    def compileProgram(*args, **kwargs):  # type: ignore
        raise RuntimeError("OpenGL not available")

# attempt PyCUDA-GL interop
try:  # pragma: no cover
    cuda_gl = ensure_package("pycuda", "pycuda.gl")
    cuda = ensure_package("pycuda", "pycuda.driver")
except Exception:  # pragma: no cover
    cuda_gl = None
    cuda = None


VERBOSE = False
VERBOSE_LOGFILE = os.path.join(os.path.dirname(__file__), "double_buffer_verbose.log")
physics_keys = [
    'positions', 'velocities', 'accelerations', 'lorentz',
    'edges', 'net', 'active_edges', 'active_lengths',
    'node_lvl', 'node_typ', 'node_role', 'glow_alpha',
    'glow_radius', 'fixed_mask', 'colors',
    'kinetic_energy', 'potential_energy', 'pca_1',
    'pca_2', 'pca_1_rank'
]
video_keys = [
    'vertex_positions', 'vertex_normals', 'vertex_colors', 'vertex_uvs',
    'vertex_indices', 'vertex_weights', 'vertex_bone_ids',
    'instance_transforms', 'instance_colors', 'instance_ids',
    'draw_commands', 'indirect_args', 'shader_uniforms',
    'compute_inputs', 'compute_outputs', 'framebuffer_targets',
    'texture_coords', 'texture_indices', 'material_ids',
    'light_positions', 'light_colors', 'camera_params',
    'viewport', 'depth_buffer', 'stencil_buffer',
    'ssbo_data', 'ubo_data', 'vao_data', 'ebo_data',
    'vbo_data', 'pbo_data', 'fbo_data'
]
def set_verbose(val=True):
    global VERBOSE
    VERBOSE = val

def verbose_log(msg):
    if VERBOSE:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        tid = threading.get_ident()
        full = f"[{ts}][TID:{tid}] {msg}"
        print(full)
        with open(VERBOSE_LOGFILE, "a") as f:
            f.write(full + "\n")

# ---- Simulated ThreadSafeBuffer, agents, and helpers ----
# (The real class must support the interface shown below, see comments for strict expectations)

class DeviceMismatchError(Exception): pass

def random_tensor(dtype, shape, device):
    if device == "cpu":
        arr = np.random.randn(*shape).astype(dtype)
        return arr
    elif device == "cuda":
        t = torch.randn(*shape, dtype=getattr(torch, np.dtype(dtype).name))
        return t.cuda()
    raise ValueError("Device must be 'cpu' or 'cuda'")

# ==== Simulated Agent Definition ====
class AgentSpec:
    def __init__(self, agent_id, backend, device):
        self.agent_id = agent_id
        self.backend = backend   # "numpy" or "torch"
        self.device = device     # "cpu" or "cuda"

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader

def create_program(vertex_src, fragment_src):
    program = glCreateProgram()
    vs = compile_shader(vertex_src, GL_VERTEX_SHADER)
    fs = compile_shader(fragment_src, GL_FRAGMENT_SHADER)
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(program).decode())
    return program

def setup_vbo():
    vbo = glGenBuffers(1)
    return vbo

def update_vbo(vbo, data):
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

