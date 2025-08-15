import time
import torch
import threading
from collections import defaultdict, deque
import threading, time
import numpy as np
from collections import deque
import threading
import time
import numpy as np
import torch
import random
import os
import networkx as nx
import queue  # <-- Add this for LockManagerThread and elsewhere

# Optional: OpenGL and CUDA interop for Tribuffer video buffer support
import importlib
import subprocess
import sys

def ensure_package(pkg_name, import_name=None):
    """
    Ensure that pkg_name is installed (via pip) and importable.
    pkg_name: the name to pass to pip
    import_name: the module name to import (if different)
    """
    module_name = import_name or pkg_name
    try:
        importlib.import_module(module_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        importlib.invalidate_caches()
    return importlib.import_module(module_name)

# ---- install & import PyOpenGL and PyCUDA if needed ----
gl     = ensure_package("PyOpenGL",      "OpenGL.GL")
_gl_all = ensure_package("PyOpenGL",      "OpenGL.GL")  # for `from OpenGL.GL import *`
shaders = ensure_package("PyOpenGL",      "OpenGL.GL.shaders")

# you’ll now have:
#   gl      → the OpenGL.GL module
#   _gl_all → same, so `from _gl_all import *` works
#   shaders → OpenGL.GL.shaders

# attempt PyCUDA-GL interop
try:
    cuda_gl = ensure_package("pycuda", "pycuda.gl")
    cuda    = ensure_package("pycuda", "pycuda.driver")
except Exception:
    cuda_gl = None
    cuda    = None

# finally, bring in all the GL symbols you need
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram


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

class AsyncGPUSyncWorker(threading.Thread):
    """
    Handles async, prioritized sync from pinned-CPU tensors to CUDA tensors.
    Uses lock commands via the manager to respect the lock graph.
    Now does non-blocking, region-wise sync with binary search fallback.
    """
    def __init__(self, buffer_sync, manager, poll_interval=0.001, kernel_size=16, stride=8):
        verbose_log(f"AsyncGPUSyncWorker.__init__()")
        super().__init__(daemon=True)
        self.buffer_sync = buffer_sync
        self.manager = manager
        self.queue = deque()       # fresh dirty keys
        self.lockwait_queue = deque()
        self.shutdown_flag = threading.Event()
        self.poll_interval = poll_interval
        self.kernel_size = kernel_size
        self.stride = stride

    def enqueue(self, key):
        verbose_log(f"AsyncGPUSyncWorker.enqueue(key={key})")
        self.queue.append(key)

    def run(self):
        verbose_log("AsyncGPUSyncWorker.run() started")
        while not self.shutdown_flag.is_set():
            if self.queue:
                key = self.queue.popleft()
            elif self.lockwait_queue:
                key = self.lockwait_queue.popleft()
            else:
                time.sleep(self.poll_interval)
                continue

            # Assume key is the buffer key (e.g. "positions"), get shape info
            arr = self.buffer_sync.host.get(key)
            if arr is None:
                continue
            N = arr.shape[0]
            page = 0  # If you have pages, set this accordingly

            # Start with full range, then binary search down
            stack = [(0, N)]
            while stack:
                idx_start, idx_end = stack.pop()
                if idx_end <= idx_start:
                    continue
                # Find all kernel regions covering this range
                regions = self.manager.lock_graph.find_covering_regions(
                    region_prefix="buf", page=page, idx_start=idx_start, idx_end=idx_end,
                    kernel_size=self.kernel_size, stride=self.stride
                )
                did_any = False
                for region in regions:
                    cmd = LockCommand('acquire', region, blocking=False)
                    self.manager.submit(cmd)
                    cmd.reply_event.wait()
                    if cmd.error:
                        # Could not acquire, subdivide further if possible
                        if idx_end - idx_start > 1:
                            mid = (idx_start + idx_end) // 2
                            stack.append((idx_start, mid))
                            stack.append((mid, idx_end))
                        continue
                    did_any = True
                    token = cmd.result
                    try:
                        # Only sync if dirty
                        if key in self.buffer_sync.host_dirty:
                            verbose_log(f"AsyncGPUSyncWorker: host->cpu sync for key {key} [{idx_start}:{idx_end}]")
                            arr = self.buffer_sync.host[key]
                            cpu_t = torch.from_numpy(arr[idx_start:idx_end]).pin_memory()
                            self.buffer_sync.cpu[key][idx_start:idx_end] = cpu_t
                            # Don't clear dirty flag unless all regions are done
                        cpu_t = self.buffer_sync.cpu.get(key)
                        if cpu_t is not None:
                            verbose_log(f"AsyncGPUSyncWorker: cpu->gpu sync for key {key} [{idx_start}:{idx_end}]")
                            if key not in self.buffer_sync.gpu or self.buffer_sync.gpu[key].device.type != 'cuda':
                                self.buffer_sync.gpu[key] = cpu_t.to('cuda', non_blocking=True)
                            else:
                                self.buffer_sync.gpu[key][idx_start:idx_end].copy_(cpu_t[idx_start:idx_end], non_blocking=True)
                    finally:
                        rel = LockCommand('release', region)
                        rel.token = token
                        self.manager.submit(rel)
                        rel.reply_event.wait()
                # If we did any region, mark as not dirty for this range
                if did_any and key in self.buffer_sync.host_dirty:
                    # Optionally, track which regions are clean
                    pass
            # After all, if all regions are clean, clear dirty flag
            # (You may want to track per-region dirty state for true correctness)

    def stop(self):
        verbose_log("AsyncGPUSyncWorker.stop()")
        self.shutdown_flag.set()


class AsyncCPUSyncWorker(threading.Thread):
    """
    Handles async sync between NumPy host and pinned-CPU torch tensors.
    Now does non-blocking, region-wise sync with binary search fallback.
    """
    def __init__(self, buffer_sync, manager, poll_interval=0.001, kernel_size=16, stride=8):
        verbose_log(f"AsyncCPUSyncWorker.__init__()")
        super().__init__(daemon=True)
        self.buffer_sync = buffer_sync
        self.manager = manager
        self.queue = deque()
        self.lockwait_queue = deque()
        self.shutdown_flag = threading.Event()
        self.poll_interval = poll_interval
        self.kernel_size = kernel_size
        self.stride = stride

    def enqueue(self, key):
        verbose_log(f"AsyncCPUSyncWorker.enqueue(key={key})")
        self.queue.append(key)

    def run(self):
        verbose_log("AsyncCPUSyncWorker.run() started")
        while not self.shutdown_flag.is_set():
            if self.queue:
                key = self.queue.popleft()
            elif self.lockwait_queue:
                key = self.lockwait_queue.popleft()
            else:
                time.sleep(self.poll_interval)
                continue

            arr = self.buffer_sync.host.get(key)
            if arr is None:
                continue
            N = arr.shape[0]
            page = 0  # If you have pages, set this accordingly

            stack = [(0, N)]
            while stack:
                idx_start, idx_end = stack.pop()
                if idx_end <= idx_start:
                    continue
                regions = self.manager.lock_graph.find_covering_regions(
                    region_prefix="buf", page=page, idx_start=idx_start, idx_end=idx_end,
                    kernel_size=self.kernel_size, stride=self.stride
                )
                did_any = False
                for region in regions:
                    cmd = LockCommand('acquire', region, blocking=False)
                    self.manager.submit(cmd)
                    cmd.reply_event.wait()
                    if cmd.error:
                        if idx_end - idx_start > 1:
                            mid = (idx_start + idx_end) // 2
                            stack.append((idx_start, mid))
                            stack.append((mid, idx_end))
                        continue
                    did_any = True
                    token = cmd.result
                    try:
                        if key in self.buffer_sync.host_dirty:
                            verbose_log(f"AsyncCPUSyncWorker: host->cpu sync for key {key} [{idx_start}:{idx_end}]")
                            arr = self.buffer_sync.host[key]
                            self.buffer_sync.cpu[key][idx_start:idx_end] = torch.from_numpy(arr[idx_start:idx_end]).pin_memory()
                        if key in self.buffer_sync.cpu_dirty:
                            verbose_log(f"AsyncCPUSyncWorker: cpu->host sync for key {key} [{idx_start}:{idx_end}]")
                            cpu_t = self.buffer_sync.cpu[key]
                            self.buffer_sync.host[key][idx_start:idx_end] = cpu_t[idx_start:idx_end].cpu().numpy()
                    finally:
                        rel = LockCommand('release', region)
                        rel.token = token
                        self.manager.submit(rel)
                        rel.reply_event.wait()
                if did_any and key in self.buffer_sync.host_dirty:
                    pass

    def stop(self):
        verbose_log("AsyncCPUSyncWorker.stop()")
        self.shutdown_flag.set()
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

class Tribuffer:
    """
    For code using three concurrent views, this holds the actual data
    but does not manage it, it is merely a shell for three objects and
    some conveniences
    """
    def __init__(self, keys, shapes, type, depth, init='zeroes', manager=None, kernel_config=None):
        verbose_log(f"Tribuffer.__init__(keys={keys}, shapes={shapes}, type={type}, depth={depth}, init={init})")
        npinittypes = {'zeroes': np.zeros, 'ones': np.ones, 'random': np.random.rand}
        tinittypes = {'zeroes': torch.zeros, 'ones': torch.ones, 'random': torch.rand}
        npfdepths = {'16': np.float16, '32': np.float32, '64': np.float64}
        npidepths = {'8': np.int8, '16': np.int16, '32': np.int32, '64': np.int64}
        tfdepths = {'16': torch.float16, '32': torch.float32, '64': torch.float64}
        tidepths = {'8': torch.int8, '16': torch.int16, '32': torch.int32, '64': torch.int64}
        npdtype = npfdepths[depth] if type == 'float' else npidepths[depth]
        tdtype = tfdepths[depth] if type == 'float' else tidepths[depth]
        self.type = type
        self.depth = depth


        self.keys = keys
        self.shapes = shapes
        self.npdtype = npdtype
        self.tdtype = tdtype
        self.data = {k: np.zeros(s, dtype=npdtype) for k, s in zip(keys, shapes)}
        self.cpu = {k: torch.zeros(s, dtype=tdtype).pin_memory() for k, s in zip(keys, shapes)}
        if torch.cuda.is_available():
            self.gpu = {k: torch.zeros(s, dtype=tdtype).cuda() for k, s in zip(keys, shapes)}
        else:
            self.gpu = {}
        self.video_buffers = {k: None for k in keys}  # OpenGL buffer handles or wrappers
        # If manager is a LockManagerThread, pass kernel info to its LockGraph
        if manager and hasattr(manager, 'lock_graph'):
            num_pages = shapes[0] if isinstance(shapes, (list, tuple)) else None
            key_shapes = {k: s for k, s in zip(keys, shapes)}
            kernels = kernel_config if kernel_config else None
            # Patch: only set up if not already done
            if not getattr(manager.lock_graph, "_kernel_initialized", False):
                manager.lock_graph.__init__(num_pages=num_pages, key_shapes=key_shapes, kernels=kernels)
                manager.lock_graph._kernel_initialized = True
        self.sync_manager = GeneralBufferSync(keys, self, manager)
    def prepare_video_buffers(self):
        """
        Prepare or wrap GPU tensors as OpenGL buffers.
        This uses PyOpenGL and optional PyCUDA interop to register buffers.
        """
        for k, tensor in self.gpu.items():
            if self.video_buffers[k] is None:
                verbose_log(f"Preparing OpenGL buffer for key '{k}'")
                # Generate and bind an OpenGL buffer
                buf_id = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, buf_id)
                size = tensor.numel() * tensor.element_size()
                # Allocate buffer storage
                glBufferData(GL_ARRAY_BUFFER, size, None, GL_DYNAMIC_DRAW)

                if cuda_gl and cuda:
                    # Register CUDA-GL buffer for zero-copy interop
                    reg_buf = cuda_gl.RegisteredBuffer(int(buf_id))
                    self.video_buffers[k] = reg_buf
                    verbose_log(f"Registered CUDA-GL buffer for key '{k}'")
                else:
                    self.video_buffers[k] = buf_id
                # Unbind buffer
                glBindBuffer(GL_ARRAY_BUFFER, 0)
            else:
                verbose_log(f"OpenGL buffer for key '{k}' already exists")

    def sync_to_video_buffers(self):
        """
        Sync GPU tensor data to OpenGL buffers.
        Uses CUDA-GL mapping if available, else falls back to glBufferSubData.
        """
        for k, tensor in self.gpu.items():
            ogl_buf = self.video_buffers.get(k)
            if ogl_buf is None:
                verbose_log(f"No OpenGL buffer for key '{k}', skipping")
                continue

            # Bind buffer
            if cuda_gl and isinstance(ogl_buf, cuda_gl.RegisteredBuffer):
                verbose_log(f"Mapping CUDA-GL buffer for key '{k}'")
                # Map the buffer for CUDA access
                mapped_ptr, size = ogl_buf.map()
                # Copy device memory directly
                cuda.memcpy_dtod(mapped_ptr, tensor.data_ptr(), tensor.numel() * tensor.element_size())
                ogl_buf.unmap()
                verbose_log(f"Unmapped CUDA-GL buffer for key '{k}'")
            else:
                # Fallback: OpenGL buffer sub-data update
                buf_id = int(ogl_buf)
                glBindBuffer(GL_ARRAY_BUFFER, buf_id)
                size = tensor.numel() * tensor.element_size()
                # Copy from GPU to CPU pinned memory synchronously
                cpu_view = tensor.detach().cpu().numpy()
                # Upload to GL buffer
                glBufferSubData(GL_ARRAY_BUFFER, 0, size, cpu_view)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                verbose_log(f"Updated GL buffer {buf_id} for key '{k}' using glBufferSubData")
    def compile_compute_shader(self, name: str, source: str):
        """
        Compile and link a GLSL compute shader program.
        name: identifier
        source: GLSL compute shader code
        """
        shader = compileShader(source, GL_COMPUTE_SHADER)
        program = compileProgram(shader)
        self.compute_programs[name] = program
        verbose_log(f"Compiled compute shader '{name}' with program ID {program}")
        return program

    def setup_ssbo(self, key: str):
        """
        Create a Shader Storage Buffer Object for a given key's GPU tensor.
        """
        buf = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf)
        size = self.gpu[key].numel() * self.gpu[key].element_size()
        glBufferData(GL_SHADER_STORAGE_BUFFER, size, None, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf)
        self.storage_buffers[key] = buf
        verbose_log(f"Created SSBO for '{key}' as buffer {buf}")
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        return buf

    def run_compute(self, name: str, key: str, mask=None):
        """
        Dispatch the compute shader 'name', bind SSBO for 'key', optionally set a mask uniform,
        and copy back results into torch GPU tensor.
        """
        program = self.compute_programs[name]
        glUseProgram(program)
        if mask is not None:
            loc = glGetUniformLocation(program, 'u_mask')
            glUniform1i(loc, int(mask))
        buf = self.storage_buffers.get(key)
        if buf is None:
            buf = self.setup_ssbo(key)
        # bind SSBO
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf)
        # dispatch: assume 1D
        count = int(self.gpu[key].numel())
        glDispatchCompute(count // 128 + 1, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        # map and copy back
        ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY)
        size = self.gpu[key].numel() * self.gpu[key].element_size()
        if ptr:
            # Create a numpy view from pointer then torch.from_numpy as needed
            arr = np.frombuffer((ctypes.c_byte * size).from_address(ptr), dtype=self.npdtype)
            self.gpu[key].copy_(torch.from_numpy(arr).cuda(), non_blocking=True)
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        glUseProgram(0)
        verbose_log(f"Ran compute shader '{name}' on key '{key}'")
        return self.gpu[key]

class GeneralBufferSync:
    """
    Manages three views: NumPy host, pinned-CPU torch, and CUDA torch.
    Accepts external data; marks buffers dirty and enqueues sync workers.
    """
    def __init__(self, keys, tribuffer, manager=None, kernel_config=None):
        verbose_log(f"GeneralBufferSync.__init__(keys={keys})")
        self.tribuffer = tribuffer
        self.keys = list(keys)
        # start lock manager
        if manager is None:
            # Pass kernel info to LockGraph
            num_pages = tribuffer.shapes[0] if isinstance(tribuffer.shapes, (list, tuple)) else None
            key_shapes = {k: s for k, s in zip(keys, tribuffer.shapes)}
            kernels = kernel_config if kernel_config else None
            self.manager = LockManagerThread(LockGraph(num_pages=num_pages, key_shapes=key_shapes, kernels=kernels))
        
        else:
            self.manager = manager
        if not self.manager.running:
            self.manager.start()
        # storage
        self.host = tribuffer.data
        self.cpu = tribuffer.cpu    # pinned CPU tensors
        self.gpu = tribuffer.gpu        # CUDA tensors
        # dirty flags
        self.host_dirty = set()
        self.cpu_dirty = set()
        # workers
        self.cpu_worker = AsyncCPUSyncWorker(self, self.manager)
        self.gpu_worker = AsyncGPUSyncWorker(self, self.manager)
        self.cpu_worker.start()
        self.gpu_worker.start()

    def set_data(self, key, data):
        verbose_log(f"GeneralBufferSync.set_data(key={key}, data_type={type(data)})")
        """
        Accept a NumPy array or torch.Tensor (CPU or CUDA).
        Flags appropriate buffers dirty and enqueues sync work.
        """
        if key not in self.host:
            raise KeyError(f"Unknown key: {key}")
        if isinstance(data, np.ndarray):
            self.host[key] = data
            self.host_dirty.add(key)
            # schedule both CPU and GPU sync
            self.cpu_worker.enqueue(key)
            self.gpu_worker.enqueue(key)
        elif isinstance(data, torch.Tensor):
            if data.device.type == 'cpu':
                pinned = data.pin_memory()
                self.cpu[key] = pinned
                self.cpu_dirty.add(key)
                self.gpu_worker.enqueue(key)
            elif data.device.type == 'cuda':
                self.gpu[key] = data
                # assume CPU and host stale
                self.cpu_dirty.add(key)
                self.host_dirty.add(key)
                self.cpu_worker.enqueue(key)
            else:
                raise TypeError(f"Unsupported tensor device: {data.device}")
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def get_data(self, key, target='host'):
        verbose_log(f"GeneralBufferSync.get_data(key={key}, target={target})")
        """
        Retrieve the latest view. Note: sync is async; ensure workers have finished.
        """
        if target == 'host':
            return self.host[key]
        elif target == 'cpu':
            return self.cpu.get(key)
        elif target == 'gpu':
            return self.gpu.get(key)
        else:
            raise ValueError(f"Invalid target: {target}")

    def shutdown(self):
        verbose_log("GeneralBufferSync.shutdown()")
        self.cpu_worker.stop()
        self.gpu_worker.stop()
        self.manager.shutdown()

class LockCommand:
    def __init__(self, op, name, blocking=True, callback=None, reply_event=None, timeout=None, *args, **kwargs):
        verbose_log(f"LockCommand.__init__(op={op}, name={name}, blocking={blocking})")
        self.op = op              # "acquire", "release", etc
        self.name = name
        self.blocking = blocking
        self.callback = callback  # Called under exclusive lock, if supplied
        self.reply_event = reply_event or threading.Event()
        self.timeout = timeout
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.error = None
        self.thread = threading.current_thread().name
class RegionToken:
    """
    A unique handle given to agents for persistent region ownership.
    Agent must present this token to continue or release the region.
    """
    _counter = 0
    _lock = threading.Lock()
    def __init__(self, region_name):
        verbose_log(f"RegionToken.__init__(region_name={region_name})")
        with RegionToken._lock:
            RegionToken._counter += 1
            self.id = RegionToken._counter
        self.region_name = region_name
        self.created = time.time()
    def __repr__(self):
        return f"<RegionToken #{self.id} for {self.region_name}>"
class LockManagerThread(threading.Thread):
    """
    The core of the abstraction. Owns all lock state, runs callbacks for
    small ops, issues tokens for persistent/complex access, manages queue.
    """
    def __init__(self, lock_graph):
        verbose_log("LockManagerThread.__init__()")
        super().__init__(daemon=True)
        self.lock_graph = lock_graph
        self.cmd_queue = queue.Queue()
        self.running = True
        self.region_tokens = {} # region -> token (if locked persistently)
        self.thread_safe_buffer_response_queue = queue.Queue()  # for async buffer responses
    def get_response_queue(self):
        verbose_log("LockManagerThread.get_response_queue()")
        return self.thread_safe_buffer_response_queue
    def submit(self, cmd: LockCommand):
        verbose_log(f"LockManagerThread.submit(cmd={cmd.op}, name={cmd.name})")
        self.cmd_queue.put(cmd)
        return cmd.reply_event
    def register_buffer_sync(self, buffer_sync):
        verbose_log("LockManagerThread.register_buffer_sync()")
        """
        Register a buffer sync manager to handle async sync operations.
        """
        if buffer_sync.manager is not None and buffer_sync.manager != self:
            need_to_start = False
            if not self.running and self.buffer_sync.manager.running:
                need_to_start = True
            self.buffer_sync.manager.shutdown()
            self.buffer_sync.manager = self
            if need_to_start:
                self.start()
        self.buffer_sync = buffer_sync

    def start(self):
        verbose_log("LockManagerThread.start()")
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
        else:
            raise RuntimeError("LockManagerThread is already running")

    def run(self):
        verbose_log("LockManagerThread.run() started")
        while self.running:
            cmd = self.cmd_queue.get()
            if cmd is None:
                verbose_log("LockManagerThread.run() received shutdown signal")
                break
            def deploy_runner():
                try:
                    verbose_log(f"LockManagerThread.deploy_runner() op={cmd.op} name={cmd.name} [START]")
                    # Print lock graph state before handling command
                    extra_verbose_node_print(self.lock_graph)
                    # ----------- SMALL OP: INLINE READ -----------
                    if cmd.op == "read":
                        verbose_log(f"LockManagerThread.deploy_runner() [READ] Trying to acquire lock for {cmd.name}")
                        if self.lock_graph.try_acquire(cmd.name):
                            try:
                                verbose_log(f"LockManagerThread.deploy_runner() [READ] Lock acquired for {cmd.name}")
                                if cmd.callback is not None:
                                    verbose_log(f"LockManagerThread.deploy_runner() [READ] Running callback for {cmd.name}")
                                    cmd.result = self.lock_graph.run_callback(cmd.callback, *cmd.args, **cmd.kwargs)
                            finally:
                                verbose_log(f"LockManagerThread.deploy_runner() [READ] Releasing lock for {cmd.name}")
                                self.lock_graph.release(cmd.name)
                            cmd.reply_event.set()
                            verbose_log(f"LockManagerThread.deploy_runner() [READ] Done for {cmd.name}")
                        else:
                            verbose_log(f"LockManagerThread.deploy_runner() [READ] Region busy for {cmd.name}")
                            cmd.error = RuntimeError("Region busy")
                            cmd.reply_event.set()

                    # ----------- PERSISTENT/COMPLEX OP: TOKEN GRANT -----------
                    elif cmd.op == "acquire":
                        verbose_log(f"LockManagerThread.deploy_runner() [ACQUIRE] Trying to acquire lock for {cmd.name}")
                        if self.lock_graph.try_acquire(cmd.name):
                            token = RegionToken(cmd.name)
                            self.region_tokens[cmd.name] = token
                            cmd.result = token
                            cmd.reply_event.set()
                            verbose_log(f"LockManagerThread.deploy_runner() [ACQUIRE] Token granted for {cmd.name}")
                        else:
                            verbose_log(f"LockManagerThread.deploy_runner() [ACQUIRE] Region busy for {cmd.name}")
                            cmd.error = RuntimeError("Region busy")
                            cmd.reply_event.set()

                    # ----------- RELEASE (MUST PRESENT TOKEN) -----------
                    elif cmd.op == "release":
                        verbose_log(f"LockManagerThread.deploy_runner() [RELEASE] Attempting release for {cmd.name}")
                        token = cmd.token
                        if not token:
                            verbose_log(f"LockManagerThread.deploy_runner() [RELEASE] No token provided for {cmd.name}")
                            cmd.error = RuntimeError("Token required for release")
                            cmd.reply_event.set()
                        else:
                            self.lock_graph.release(cmd.name, token)
                            self.region_tokens.pop(cmd.name, None)
                            cmd.result = True
                            cmd.reply_event.set()
                            verbose_log(f"LockManagerThread.deploy_runner() [RELEASE] Released {cmd.name}")

                    # ----------- EXTENSION: BATCH, QUERY, ETC. -----------
                    # (Handle more ops as needed)

                except Exception as e:
                    verbose_log(f"LockManagerThread.deploy_runner() [ERROR] Exception: {e}")
                    cmd.error = e
                    cmd.reply_event.set()

                # Print lock graph state after handling command
                extra_verbose_node_print(self.lock_graph)
                verbose_log(f"LockManagerThread.deploy_runner() op={cmd.op} name={cmd.name} [END]")
                return cmd.result
            # Run the command in a separate thread to avoid blocking
            runner_thread = threading.Thread(target=deploy_runner, daemon=True)
            runner_thread.start()
            # if the op was read, wait and deliver the result directly
            return_message = None
            if cmd.op == "read":
                runner_thread.join()
                return_message = cmd.result

            if return_message is not None:
                self.thread_safe_buffer_response_queue.put(return_message)
    def shutdown(self):
        verbose_log("LockManagerThread.shutdown()")
        self.running = False
        self.cmd_queue.put(None)

class LockNode:
    def __init__(self, name):
        verbose_log(f"LockNode.__init__(name={name})")
        self.name = name
        self.lock = threading.Lock()
        self.holder = None  # Which thread owns the lock
        self.authority_edges = set()   # children: has authority over
        self.submission_edges = set()  # parents: submits to
        self.waiting_queue = deque()

class LockGraph:
    def __init__(self, *,
                 num_pages=None,
                 key_shapes=None,
                 kernels=None,
                 default_kernel_size=4,
                 default_stride=2,
                 boundary='clip'):
        """
        LockGraph: Dense region lock graph for ND buffer concurrency.

        Node/edge structure (all-to-all, not sparse):

            [Page Layer]      [Key Layer]      [Kernel Layer]      [Vertex Layer]
            ┌───────────┐     ┌──────────┐     ┌────────────┐      ┌────────────┐
            │ page:0    │────▶│ key:pos  │────▶│ kernel:... │─────▶│ vtx:...    │
            │ page:1    │────▶│ key:vel  │────▶│ ...        │─────▶│ ...        │
            │ ...       │────▶│ ...      │────▶│            │─────▶│            │
            └───────────┘     └──────────┘     └────────────┘      └────────────┘

        - Every page is connected to every key, every key to every kernel region, every kernel region to every vertex it covers.
        - All nodes are LockNode instances, all possible edges are present for immediate access and analysis.
        - All nodes and edges are also present in self.nx_graph (networkx.DiGraph) for full graph ops.

        Example node names:
            page:0
            buf:positions
            buf:positions:p0_d0_0_16_d1_0_16   # kernel region for page 0, positions, 2D window
            buf:positions:p0_d0_0_16_d1_0_16:vtx_5_7  # vertex region (optional)

        """
        verbose_log("LockGraph.__init__()")
        self.master_lock = threading.Lock()
        self.nodes = {}   # name -> LockNode
        self.nx_graph = nx.DiGraph()
        self.interval_index = {}  # prefix -> IntervalTree
        # --- for fast edge correlation lookups ---
        self.edge_table = None        # pandas DataFrame of edges + metadata
        self.crash_buffer = None      # circular buffer of verbose results
        self.verbose_mode = False
        # --- Build all nodes and all possible edges ---
        # 1. Pages
        page_nodes = []
        if num_pages is not None:
            for p in range(num_pages):
                pname = f"page:{p}"
                self.add_node(pname)
                page_nodes.append(pname)
        # 2. Keys
        key_nodes = []
        if key_shapes is not None:
            for k in key_shapes:
                kname = f"buf:{k}"
                self.add_node(kname)
                key_nodes.append(kname)
        # 3. Kernel regions
        kernel_nodes = []
        if num_pages is not None and key_shapes is not None:
            for k, shape in key_shapes.items():
                if isinstance(shape, int):
                    dims = (shape,)
                elif isinstance(shape, (tuple, list)):
                    dims = tuple(shape)
                else:
                    raise ValueError(f"Unsupported shape type for key '{k}': {shape}")
                if kernels and k in kernels:
                    kernel = kernels[k]
                    if isinstance(kernel, int):
                        kernel_shape = tuple([kernel] * len(dims))
                        stride = tuple([kernel] * len(dims))
                    elif isinstance(kernel, (tuple, list)):
                        kernel_shape = tuple(kernel)
                        stride = tuple(kernel)
                    elif isinstance(kernel, dict):
                        kernel_shape = tuple(kernel.get('shape', [default_kernel_size]*len(dims)))
                        stride = tuple(kernel.get('stride', kernel_shape))
                    else:
                        raise ValueError(f"Unsupported kernel type for key '{k}': {kernel}")
                else:
                    kernel_shape = tuple([default_kernel_size] * len(dims))
                    stride = tuple([default_stride] * len(dims))
                for page in range(num_pages):
                    # ND sliding window
                    from itertools import product
                    ranges = [range(0, dims[d] - kernel_shape[d] + 1, stride[d]) for d in range(len(dims))]
                    for idxs in product(*ranges):
                        idx_str = "_".join(f"d{d}{idxs[d]}_{idxs[d]+kernel_shape[d]}" for d in range(len(dims)))
                        region_name = f"buf:{k}:p{page}_{idx_str}"
                        self.add_node(region_name)
                        kernel_nodes.append(region_name)

                        # — now add only axis-aligned extreme vertices —
                        dcount = len(dims)
                        # build list of vertex offsets
                        if dcount <= 4:
                            # all corners: 2^d
                            corner_bits = product(*([[0,1]] * dcount))
                            offsets = [
                                tuple(
                                    idxs[d] + bit * (kernel_shape[d] - 1)
                                    for d, bit in enumerate(bits)
                                )
                                for bits in corner_bits
                            ]
                        else:
                            # only low/high on each axis: 2*d
                            center = [
                                idxs[d] + kernel_shape[d] // 2
                                for d in range(dcount)
                            ]
                            offsets = []
                            for d in range(dcount):
                                low  = center.copy(); high = center.copy()
                                low[d]  = idxs[d]
                                high[d] = idxs[d] + kernel_shape[d] - 1
                                offsets.extend([tuple(low), tuple(high)])

                        # add vertex nodes & edges
                        for off in offsets:
                            coord_str = "_".join(str(c) for c in off)
                            vname = f"{region_name}:vtx_{coord_str}"
                            self.add_node(vname)
                            # hierarchical edge region → vertex
                            self.add_authority(region_name, vname)
        # 4. Vertices (optional, for full density)
        # For each kernel region, add all possible vertex nodes it covers
        # (This can be omitted if not needed for your use case.)

        # --- Add all possible edges ---
        # page -> key (all-to-all)
        for p in page_nodes:
            for k in key_nodes:
                self.add_authority(p, k)
        # key -> kernel (all-to-all)
        for k in key_nodes:
            for kr in kernel_nodes:
                # Only connect keys to their own kernel regions
                if kr.startswith(k):
                    self.add_authority(k, kr)


        self.monitor_event = threading.Event()
        # once nx_graph is fully built, build the edge correlation table
        self._build_edge_table()

        # start monitor thread (unchanged)
        self.monitor_event = threading.Event()
        self.monitor_thread = threading.Thread(
            target=self._monitor_queues, daemon=True
        )
        self.monitor_thread.start()
    # ——————————— Correlation table machinery ———————————
    def _build_edge_table(self):
        """Construct a pandas DataFrame with one row per edge, plus parsed metadata."""
        import pandas as pd, re
        rows = []
        for src, dst in self.nx_graph.edges():
            m_src = self._parse_node_meta(src)
            m_dst = self._parse_node_meta(dst)
            row = {'src': src, 'dst': dst}
            # prefix each src-meta with “src_” and dst-meta with “dst_”
            for k, v in m_src.items():
                row[f"src_{k}"] = v
            for k, v in m_dst.items():
                row[f"dst_{k}"] = v
            rows.append(row)
        self.edge_table = pd.DataFrame(rows)

    def _parse_node_meta(self, name):
        """Extract page/key/region/vertex info from a node name."""
        # defaults
        info = {'page': None, 'key': None, 'region': None, 'vertex': None}
        # page:<n>
        if name.startswith("page:"):
            info['page'] = int(name.split(":",1)[1])
            return info
        # buf:<key> or buf:<key>:p<page>_<region>...
        if name.startswith("buf:"):
            parts = name.split(":p",1)
            if len(parts) < 2:
                print(parts)
                raise ValueError(f"Invalid node name format: {name}")
            
            prospective_key = parts[0].split("buf:",1)
            if len(prospective_key) < 2:
                print(prospective_key)
                raise ValueError(f"Invalid key format in node name: {name}")    
            info['key'] = prospective_key[1]
            if len(parts)==2:
                pg, rest = parts[1].split("_",1)
                info['page'] = int(pg)
                # region e.g. d0_0_16_d1_0_16…
                region = tuple(tuple(map(int,x.split("_")[-2:]))
                               for x in rest.split("_d") if x)
                info['region'] = region
        # vertex appended
        if ":vtx_" in name:
            base, coords = name.split(":vtx_",1)
            info_v = tuple(map(int, coords.split("_")))
            info['vertex'] = info_v
            # also inherit page/key/region from the base part
            base_meta = self._parse_node_meta(base)
            info.update({k: base_meta[k] for k in ['page','key','region']})
        return info

    def query_edges(self, **filters):
        """
        Slice the edge_table by arbitrary filters:
          e.g. page=2, src_key='positions',
                dst_region=lambda r: any(start<10 for start,end in r)
        Filters map column names (src_page, dst_key, src_region, dst_vertex, etc.)
        to either:
          - a literal to ==-compare
          - a callable f(col_series)→bool mask
        Returns a DataFrame of matching rows.
        """
        df = self.edge_table
        for col, cond in filters.items():
            if callable(cond):
                mask = cond(df[col])
                df = df[mask]
            else:
                df = df[df[col] == cond]
        return df.copy()

    def enable_verbose(self, interval=60, buffer_size=1000):
        """Turn on verbose crash‐buffering, dump every `interval` seconds."""
        import threading
        from collections import deque
        self.verbose_mode = True
        self.crash_buffer = deque(maxlen=buffer_size)
        self._crash_interval = interval

        def _dumper():
            from time import strftime, localtime
            if self.crash_buffer:
                with open("lockgraph_crash.log","a") as f:
                    for ts, res in list(self.crash_buffer):
                        tstr = strftime("%Y-%m-%d %H:%M:%S", localtime(ts))
                        f.write(f"[{tstr}] {res}\n")
                self.crash_buffer.clear()
            threading.Timer(self._crash_interval, _dumper).start()

        # schedule first dump
        threading.Timer(self._crash_interval, _dumper).start()
    def add_node(self, name):
        verbose_log(f"LockGraph.add_node(name={name})")
        with self.master_lock:
            if name not in self.nodes:
                self.nodes[name] = LockNode(name)
                self.nx_graph.add_node(name)

    def add_authority(self, parent, child):
        verbose_log(f"LockGraph.add_authority(parent={parent}, child={child})")
        with self.master_lock:
            self.nodes[parent].authority_edges.add(child)
            self.nodes[child].submission_edges.add(parent)
            self.nx_graph.add_edge(parent, child)

    def try_acquire(self, name, blocking=True):
        verbose_log(f"LockGraph.try_acquire(name={name}, blocking={blocking})")
        current_thread = threading.current_thread()
        with self.master_lock:
            if self._can_acquire(name, current_thread):
                node = self.nodes[name]
                acquired = node.lock.acquire(blocking)
                if acquired:
                    node.holder = current_thread
                return acquired
            else:
                if blocking:
                    node = self.nodes[name]
                    node.waiting_queue.append(current_thread)
                    self.monitor_event.set()
                return False

    def release(self, name):
        verbose_log(f"LockGraph.release(name={name})")
        with self.master_lock:
            node = self.nodes[name]
            if node.holder != threading.current_thread():
                raise RuntimeError("Cannot release a lock not held by this thread")
            node.holder = None
            node.lock.release()
            self.monitor_event.set()  # Wake the monitor thread

    def _can_acquire(self, name, thread):
        verbose_log(f"LockGraph._can_acquire(name={name}, thread={thread})")
        """Return True if no ancestor or descendant lock is held by any other thread."""
        # Breadth-first search up and down
        visited = set()
        queue = deque([name])
        while queue:
            n = queue.popleft()
            if n in visited:
                continue
            visited.add(n)
            node = self.nodes[n]
            # If the node is locked by someone else, can't acquire
            if node.holder and node.holder != thread:
                return False
            # Recurse to parents (submission_edges) and children (authority_edges)
            queue.extend(node.submission_edges)
            queue.extend(node.authority_edges)
        return True

    def _monitor_queues(self):
        verbose_log("LockGraph._monitor_queues() started")
        while True:
            self.monitor_event.wait()
            with self.master_lock:
                for node in self.nodes.values():
                    if node.waiting_queue and self._can_acquire(node.name, node.waiting_queue[0]):
                        t = node.waiting_queue.popleft()
                        # You would need a more robust notification system here.
                        # For now, just acquire for them (dangerous: demo only)
                        node.lock.acquire()
                        node.holder = t
                        # (In practice, notify thread via event/condition, not this)
            self.monitor_event.clear()

    def read_only_traverse(self, name, fn, direction='both'):
        verbose_log(f"LockGraph.read_only_traverse(name={name}, direction={direction})")
        """
        Traverse the graph from `name`, calling `fn(node)` for every
        node that is NOT currently locked. Skips locked nodes entirely.
        direction: 'authority', 'submission', or 'both'
        """
        with self.master_lock:
            visited = set()
            queue = deque([name])
            while queue:
                n = queue.popleft()
                if n in visited:
                    continue
                visited.add(n)
                node = self.nodes[n]
                # If locked, skip this node (do not visit or descend)
                if node.lock.locked():
                    continue
                fn(node)
                if direction in ('authority', 'both'):
                    queue.extend(node.authority_edges)
                if direction in ('submission', 'both'):
                    queue.extend(node.submission_edges)

class DoubleBuffer:
    """
    Thin agent-index/cursor tracker for hyperlocal concurrent buffer access.
    All data operations are callback-driven; this class holds *no* locks.
    """
    def __init__(self, roll_length=2, num_agents=2, reference=None):
        verbose_log(f"DoubleBuffer.__init__(roll_length={roll_length}, num_agents={num_agents})")
        self.roll_length = roll_length
        self.num_agents = num_agents
        self.reference = reference or physics_keys
        self.read_idx = [0] * num_agents
        self.write_idx = [1] * num_agents
        phase_distance = roll_length // num_agents
        for i in range(num_agents):
            self.read_idx[i] = self.read_idx[i-1]+1 if i > 0 else 0
            self.write_idx[i] = (self.read_idx[i] + phase_distance) % roll_length

    def get_read_page(self, agent_idx=0):
        verbose_log(f"DoubleBuffer.get_read_page(agent_idx={agent_idx})")
        return self.read_idx[agent_idx]

    def get_write_page(self, agent_idx=0):
        verbose_log(f"DoubleBuffer.get_write_page(agent_idx={agent_idx})")
        return self.write_idx[agent_idx]

    def advance(self, agent_idx=0):
        verbose_log(f"DoubleBuffer.advance(agent_idx={agent_idx})")
        self.read_idx[agent_idx] = (self.read_idx[agent_idx] + 1) % self.roll_length
        self.write_idx[agent_idx] = (self.write_idx[agent_idx] + 1) % self.roll_length

    def for_read(self, agent_idx=0, keys=None, callback=None):
        verbose_log(f"DoubleBuffer.for_read(agent_idx={agent_idx}, keys={keys})")
        """Invoke `callback(page_idx, keys, agent_idx)` for agent's current read page."""
        page_idx = self.get_read_page(agent_idx)
        if callback is not None:
            return callback(page_idx, keys, agent_idx)

    def for_write(self, agent_idx=0, keys=None, callback=None):
        verbose_log(f"DoubleBuffer.for_write(agent_idx={agent_idx}, keys={keys})")
        """Invoke `callback(page_idx, keys, agent_idx)` for agent's current write page."""
        page_idx = self.get_write_page(agent_idx)
        if callback is not None:
            return callback(page_idx, keys, agent_idx)

    # Optional: a universal accessor if you want to specify r/w or have more metadata
    def access(self, agent_idx=0, mode='read', keys=None, callback=None):
        verbose_log(f"DoubleBuffer.access(agent_idx={agent_idx}, mode={mode}, keys={keys})")
        idx = self.get_read_page(agent_idx) if mode == 'read' else self.get_write_page(agent_idx)
        if callback:
            return callback(idx, keys, agent_idx)

class NumpyActionHistory:
    def __init__(self, num_agents, num_pages, num_keys, window_size=256):
        verbose_log(f"NumpyActionHistory.__init__(num_agents={num_agents}, num_pages={num_pages}, num_keys={num_keys}, window_size={window_size})")
        self.window = window_size
        self.na = num_agents
        self.np = num_pages
        self.nk = num_keys
        self.ptr = 0
        self.actions = np.zeros((window_size, num_agents, num_pages, num_keys, 2), dtype=np.uint8)
        self.timestamps = np.zeros(window_size, dtype=np.float64)  # optional

    def record(self, agent, page, key, action):
        verbose_log(f"NumpyActionHistory.record(agent={agent}, page={page}, key={key}, action={action})")
        # action: 0=read, 1=write
        self.actions[self.ptr, agent, page, key, action] = 1
        self.timestamps[self.ptr] = time.time()
        self.ptr = (self.ptr + 1) % self.window

    def get_recent(self, agent=None, page=None, key=None, action=None, kernel=None):
        verbose_log(f"NumpyActionHistory.get_recent(agent={agent}, page={page}, key={key}, action={action}, kernel={kernel})")
        """
        Returns a view or sum over the history window for the given indices.
        Use `kernel=(start, end)` for time slices.
        """
        idx = slice(None) if kernel is None else slice(*kernel)
        sl = [
            idx,
            agent if agent is not None else slice(None),
            page if page is not None else slice(None),
            key if key is not None else slice(None),
            action if action is not None else slice(None),
        ]
        return self.actions[tuple(sl)]

    def last_write_idx(self, agent, page, key):
        verbose_log(f"NumpyActionHistory.last_write_idx(agent={agent}, page={page}, key={key})")
        """Returns the latest window idx (or -1) where a write occurred."""
        # Get all indices where a write occurred for given (a,p,k)
        writes = np.nonzero(self.actions[:, agent, page, key, 1])[0]
        return writes[-1] if writes.size else -1

    def reads_since_last_write(self, agent, page, key):
        verbose_log(f"NumpyActionHistory.reads_since_last_write(agent={agent}, page={page}, key={key})")
        last_write = self.last_write_idx(agent, page, key)
        # All reads since last write (could broadcast across agents if needed)
        reads = self.actions[last_write+1:, agent, page, key, 0]
        return np.sum(reads)

    def unique_agents_since_last_write(self, page, key):
        verbose_log(f"NumpyActionHistory.unique_agents_since_last_write(page={page}, key={key})")
        """Which agents have read (page,key) since the last write by anyone?"""
        # Find last write index for any agent
        writes = np.nonzero(self.actions[:, :, page, key, 1])
        if writes[0].size == 0:
            start = 0
        else:
            start = writes[0].max() + 1
        reads = self.actions[start:, :, page, key, 0]
        return np.where(reads.sum(axis=0) > 0)[0]  # agent indices

    # Add fast reductions, kernel/stride ops, etc. as needed


import time
import threading
import random
import numpy as np
import torch

# ----------- Gold Standard Stress Test ------------

class EnginePerfTracker:
    """Tracks running averages and break-even points for each engine."""
    def __init__(self):
        verbose_log("EnginePerfTracker.__init__()")
        self.stats = {'numpy': [], 'torch': []}
    def record(self, engine, batch_size, duration):
        verbose_log(f"EnginePerfTracker.record(engine={engine}, batch_size={batch_size}, duration={duration})")
        self.stats[engine].append((batch_size, duration))
    def avg_time(self, engine):
        verbose_log(f"EnginePerfTracker.avg_time(engine={engine})")
        d = [x[1] for x in self.stats[engine]] if self.stats[engine] else [0.0]
        return sum(d) / len(d)
    def avg_batch(self, engine):
        verbose_log(f"EnginePerfTracker.avg_batch(engine={engine})")
        b = [x[0] for x in self.stats[engine]] if self.stats[engine] else [0]
        return sum(b) / max(1, len(b))
    def report(self):
        verbose_log("EnginePerfTracker.report()")
        print("Engine Perf:")
        for eng in self.stats:
            print(f"  {eng} - avg batch: {self.avg_batch(eng):.1f}, avg time: {self.avg_time(eng):.4f}s")

class MultiAgentEngineSplitter:
    """
    Routes each job to numpy or torch engine based on delta mask, age, perf table.
    """
    def __init__(self, perf_tracker, torch_threshold=32, old_age=10):
        verbose_log(f"MultiAgentEngineSplitter.__init__(torch_threshold={torch_threshold}, old_age={old_age})")
        self.perf = perf_tracker
        self.torch_threshold = torch_threshold
        self.old_age = old_age  # in frames

    def choose_engine(self, batch_size, age):
        verbose_log(f"MultiAgentEngineSplitter.choose_engine(batch_size={batch_size}, age={age})")
        # If large/young, torch. If small/old, numpy.
        if batch_size >= self.torch_threshold and age < self.old_age:
            return 'torch'
        else:
            return 'numpy'

    def profile(self, engine, batch_size, fn):
        verbose_log(f"MultiAgentEngineSplitter.profile(engine={engine}, batch_size={batch_size})")
        t0 = time.time()
        result = fn()
        t1 = time.time()
        self.perf.record(engine, batch_size, t1-t0)
        return result

    def split_and_run(self, mask, ages, do_numpy, do_torch):
        verbose_log("MultiAgentEngineSplitter.split_and_run()")
        """
        mask: bool array, which indices to process
        ages: int array, age of each particle/region
        """
        idxs = np.where(mask)[0]
        jobs_torch = [i for i in idxs if ages[i] < self.old_age]
        jobs_numpy = [i for i in idxs if ages[i] >= self.old_age]
        res_numpy = []
        res_torch = []
        if jobs_numpy:
            res_numpy = self.profile('numpy', len(jobs_numpy), lambda: do_numpy(jobs_numpy))
        if jobs_torch:
            res_torch = self.profile('torch', len(jobs_torch), lambda: do_torch(jobs_torch))
        return res_numpy, res_torch

class ThreadSafeBuffer:
    def __init__(self, shape, dtype, agent_specs, manager, buffer_size=2, key_library=physics_keys, late_join=True):
        verbose_log(f"ThreadSafeBuffer.__init__(shape={shape}, dtype={dtype}, buffer_size={buffer_size}, late_join={late_join})")
        self.shape = tuple([buffer_size] + list(shape))
        self.dtype = dtype
        self.manager = manager
        if manager is None:
            self.manager = LockManagerThread(LockGraph())
            if not self.manager.running:
                self.manager.start()
        self.responses = self.manager.get_response_queue()
        self.blacklist = None  # Optional: set of agent IDs that cannot join
        self.whitelist = None
        self.agents_on_auto_advance = set()  # Agents that auto-advance
        self.auto_advance = False  # If True, auto-advance agents on read/write
        self.late_join = late_join
        self.readstates = DoubleBuffer(roll_length=buffer_size, num_agents=len(agent_specs), reference=physics_keys)
        self.host = Tribuffer(physics_keys, self.shape, dtype, '32', init='zeroes', manager=self.manager)
        self.mailboxes = {spec.agent_id: [] for spec in agent_specs}
        self.agent_specs = {spec.agent_id: spec for spec in agent_specs}
        self.mailbox_locks = {spec.agent_id: threading.Lock() for spec in agent_specs}
        self.mailbox_tokens = {spec.agent_id: None for spec in agent_specs}
        
    def late_join_agent(self, agent_specs):
        verbose_log(f"ThreadSafeBuffer.late_join_agent(agent_specs={agent_specs})")
        """Allow an agent to join late, initializing its mailbox and state."""
        rejected_agents = []
        accepted_agents = []
        for i, agent_spec in enumerate(agent_specs):
            agent_id = agent_spec.agent_id
            if agent_id in self.agent_specs:
                print(f"Agent {agent_id} already registered.")
                rejected_agents.append(i)
            elif self.blacklist and agent_id not in self.blacklist:
                print(f"Agent {agent_id} is blacklisted and cannot join.")
                rejected_agents.append(i)
            elif self.whitelist and agent_id in self.whitelist:
                print(f"Agent {agent_id} is not whitelisted and cannot join whitelisted buffer.")
                rejected_agents.append(i)
            elif self.late_join and agent_id not in self.agent_specs:
                accepted_agents.append(i)
            elif not self.late_join:
                print(f"Late joining not allowed for agent {agent_id}.")
                rejected_agents.append(i)
        if rejected_agents:
            print(f"Agents {', '.join(str(agent_specs[i].agent_id) for i in rejected_agents)} were rejected due to existing registration or policy.")
        for i in accepted_agents:
            agent_spec = agent_specs[i]
            agent_id = agent_spec.agent_id
            self.agent_specs[agent_id] = agent_spec
            self.mailboxes[agent_id] = []
            self.mailbox_locks[agent_id] = threading.Lock()
            self.mailbox_tokens[agent_id] = None
            self.readstates = self.readstates.insert_agents(agent_specs)

    def __getitem__(self, idx, agent=None, backend=None, device=None, blocking=True, readonly=False, callback=lambda x: x, reply_event=None, timeout=None, *args, **kwargs):
        verbose_log(f"ThreadSafeBuffer.__getitem__(idx={idx}, agent={agent}, backend={backend}, device={device}, blocking={blocking})")
        self.manager.submit(LockCommand('read', f"buf:{idx}", blocking=blocking, callback=callback, reply_event=reply_event, timeout=timeout, *args, **kwargs))
        cmd = self.manager.get_response_queue().get()
        if self.auto_advance and agent is not None:
            # Auto-advance the agent if it has auto-advance enabled
            if agent in self.agents_on_auto_advance:
                self.advance_agent(agent)
        return cmd.result
        
    def __setitem__(self, idx, value, agent=None, backend=None, device=None, blocking=True):
        verbose_log(f"ThreadSafeBuffer.__setitem__(idx={idx}, value_type={type(value)}, agent={agent}, backend={backend}, device={device}, blocking={blocking})")
        self.manager.submit(LockCommand('write', f"buf:{idx}", value=value, blocking=blocking))
        cmd = self.manager.get_response_queue().get()
        if self.auto_advance and agent is not None:
            # Auto-advance the agent if it has auto-advance enabled
            if agent in self.agents_on_auto_advance:
                self.advance_agent(agent)
        return cmd.result

    

    def sync(self):
        verbose_log("ThreadSafeBuffer.sync()")
        # In real version: must union/copy states so both numpy and torch arrays have the freshest view
        pass

    def register_agent(self, agent_id, allow_clipping=False):
        verbose_log(f"ThreadSafeBuffer.register_agent(agent_id={agent_id}, allow_clipping={allow_clipping})")
        # Register agent as mailbox owner; maybe allow special privilege
        pass

    def advance_agent(self, agent_id):
        verbose_log(f"ThreadSafeBuffer.advance_agent(agent_id={agent_id})")
        # Move this agent’s “moment” forward
        pass

    def query(self, region, agent=None, query={}):
        verbose_log(f"ThreadSafeBuffer.query(region={region}, agent={agent}, query={query})")
        # Query region state, token, who’s got it, queue length, etc.
        pass

    # ...additional helper methods as needed



# ----------- Smoke Test Scenario ------------

def smoke_test():
    verbose_log("smoke_test() started")
    print("\n=== ADVANCED MONTE CARLO CONCURRENCY STRESS TEST ===")

    N = 512      # number of particles, big enough for races
    SHAPE = (N, 3)
    AGENTS = 16
    STEPS = 200
    np.random.seed(1337)

    # ----- Agent registration: half numpy, half torch, all mixed devices -----
    agent_specs = []
    for agent_id in range(AGENTS):
        if agent_id % 2 == 0:
            agent_specs.append(AgentSpec(agent_id, backend="numpy", device="cpu"))
        else:
            agent_specs.append(AgentSpec(agent_id, backend="torch", device="cuda"))

    # --- Our supreme, mailbox-driven, type-hiding buffer ---
    buf = ThreadSafeBuffer(shape=SHAPE, dtype=np.float32, agent_specs=agent_specs, manager=None)

    errors = []
    def agent_thread(spec: AgentSpec):
        for step in range(STEPS):
            # Monte Carlo: random region, random operation, random type request
            idx_start = random.randint(0, N-16)
            idx_end = idx_start + random.randint(1, 16)
            idx = slice(idx_start, idx_end)
            mode = random.choice(["read", "write"])
            want_backend = random.choice(["numpy", "torch"])
            # All agents sometimes (10%) ask for wrong type on purpose
            if random.random() < 0.1:
                want_backend = "torch" if spec.backend == "numpy" else "numpy"
            try:
                if mode == "read":
                    out = buf[idx, spec.agent_id, want_backend, spec.device]
                    # Validate type contract
                    if spec.backend == "numpy" and not isinstance(out, np.ndarray):
                        raise DeviceMismatchError(f"Agent {spec.agent_id} expected numpy, got {type(out)}")
                    elif spec.backend == "torch" and not isinstance(out, torch.Tensor):
                        raise DeviceMismatchError(f"Agent {spec.agent_id} expected torch, got {type(out)}")
                else:  # write
                    value = random_tensor(np.float32, (idx_end - idx_start, 3), spec.device if want_backend == spec.backend else ("cpu" if spec.backend == "numpy" else "cuda"))
                    try:
                        buf[idx, spec.agent_id, want_backend, spec.device] = value
                    except DeviceMismatchError as e:
                        if want_backend == spec.backend:
                            raise   # Should not happen!
                        # else: expected, since we purposely mismatch sometimes
                # Random sleep, chaos
                time.sleep(random.uniform(0.0001, 0.002))
            except Exception as e:
                # Always print
                print(f"[Agent {spec.agent_id} step {step}] ERROR: {e}")
                errors.append(e)

    threads = [threading.Thread(target=agent_thread, args=(spec,)) for spec in agent_specs]
    for t in threads: t.start()
    for t in threads: t.join()

    buf.sync()  # Make sure all device mirrors are consistent at the end
    print("\nErrors encountered:", len(errors))
    print("=== STRESS TEST COMPLETE ===\n")

# ----------- RUN THE TEST ------------

if __name__ == "__main__":
    set_verbose(True)
    smoke_test()

def extra_verbose_node_print(lock_graph):
    """
    Print an ASCII visualization of the lock graph.
    o = open, - = waiting, x = locked.
    Nodes are grouped by their level (distance from root).
    """
    # Find roots (nodes with no submission_edges)
    with lock_graph.master_lock:
        nodes = lock_graph.nodes
        roots = [n for n in nodes if not nodes[n].submission_edges]
        # BFS to assign levels
        level_map = {}
        visited = set()
        queue = [(r, 0) for r in roots]
        while queue:
            n, lvl = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)
            level_map.setdefault(lvl, []).append(n)
            for c in nodes[n].authority_edges:
                queue.append((c, lvl + 1))
        # Print by level
        print("\n[LockGraph State]")
        for lvl in sorted(level_map):
            line = []
            for n in sorted(level_map[lvl]):
                node = nodes[n]
                if node.lock.locked():
                    ch = "x"
                elif node.waiting_queue:
                    ch = "-"
                else:
                    ch = "o"
                line.append(f"{n}:{ch}")
            print(f"Level {lvl}: " + "  ".join(line))
        print("[End LockGraph State]\n")

