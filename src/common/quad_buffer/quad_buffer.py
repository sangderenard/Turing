from ..double_buffer.base import *
from ..double_buffer.workers import AsyncCPUSyncWorker, AsyncGPUSyncWorker
from ..double_buffer.lock import LockManagerThread, LockGraph
from src.common.tensors.abstraction import AbstractTensor
import numpy as np
import torch


class QuadBuffer:
    """
    Four-way buffer for any mix of numpy, torch, and abstract tensor backings (plus optional video/OpenGL buffer).
    Each buffer is optional and can be enabled/disabled at init.
    Includes all advanced features from Tribuffer: OpenGL/compute shader integration, video buffer management, and more.
    """
    def __init__(self, keys, shapes, type, depth, use_numpy=True, use_torch=True, use_abstract=True, use_video=True, init='zeroes', manager=None, kernel_config=None):
        verbose_log(f"QuadBuffer.__init__(keys={keys}, shapes={shapes}, type={type}, depth={depth}, init={init})")
        self.type = type
        self.depth = depth
        self.keys = keys
        self.shapes = shapes
        self.use_numpy = use_numpy
        self.use_torch = use_torch
        self.use_abstract = use_abstract
        self.use_video = use_video
        npinittypes = {'zeroes': np.zeros, 'ones': np.ones, 'random': np.random.rand}
        tinittypes = {'zeroes': torch.zeros, 'ones': torch.ones, 'random': torch.rand}
        npfdepths = {'16': np.float16, '32': np.float32, '64': np.float64}
        npidepths = {'8': np.int8, '16': np.int16, '32': np.int32, '64': np.int64}
        tfdepths = {'16': torch.float16, '32': torch.float32, '64': torch.float64}
        tidepths = {'8': torch.int8, '16': torch.int16, '32': torch.int32, '64': torch.int64}
        npdtype = npfdepths[depth] if type == 'float' else npidepths[depth]
        tdtype = tfdepths[depth] if type == 'float' else tidepths[depth]
        self.npdtype = npdtype
        self.tdtype = tdtype
        self.data = {k: np.zeros(s, dtype=npdtype) for k, s in zip(keys, shapes)} if use_numpy else None
        self.cpu = {k: torch.zeros(s, dtype=tdtype).pin_memory() for k, s in zip(keys, shapes)} if use_torch else None
        if use_torch and torch.cuda.is_available():
            self.gpu = {k: torch.zeros(s, dtype=tdtype).cuda() for k, s in zip(keys, shapes)}
        else:
            self.gpu = {} if use_torch else None
        self.abstract = {k: AbstractTensor.tensor_from_list(np.zeros(s, dtype=npdtype).tolist()) for k, s in zip(keys, shapes)} if use_abstract else None
        self.video_buffers = {k: None for k in keys} if use_video else None
        self.compute_programs = {}  # For compute shaders
        self.storage_buffers = {}   # For SSBOs
        # Lock manager and sync
        if manager and hasattr(manager, 'lock_graph'):
            num_pages = shapes[0] if isinstance(shapes, (list, tuple)) else None
            key_shapes = {k: s for k, s in zip(keys, shapes)}
            kernels = kernel_config if kernel_config else None
            if not getattr(manager.lock_graph, "_kernel_initialized", False):
                manager.lock_graph.__init__(num_pages=num_pages, key_shapes=key_shapes, kernels=kernels)
                manager.lock_graph._kernel_initialized = True
        self.sync_manager = GeneralQuadBufferSync(keys, self, manager)

    def _ensure_gl_context(self):
        """
        Ensure an OpenGL context exists. If not, create a minimal one (headless or hidden).
        This is a no-op if a context is already current.
        """
        try:
            from OpenGL.GL import glGetString, GL_VERSION
            version = glGetString(GL_VERSION)
            if version is not None:
                return  # Context exists
        except Exception:
            pass
        # Try to create a context (platform-specific, fallback to pyglet or glfw)
        try:
            import pyglet
            config = pyglet.gl.Config(double_buffer=True)
            window = pyglet.window.Window(visible=False, config=config)
            verbose_log("Created hidden OpenGL context using pyglet.")
            self._hidden_gl_window = window
        except Exception as e:
            verbose_log(f"Failed to create OpenGL context automatically: {e}")
            raise RuntimeError("No OpenGL context found and could not create one automatically.")

    def prepare_video_buffers(self):
        """
        Prepare or wrap GPU tensors as OpenGL buffers.
        This uses PyOpenGL and optional PyCUDA interop to register buffers.
        Ensures an OpenGL context exists before proceeding.
        """
        if not self.use_video or not self.gpu:
            return
        self._ensure_gl_context()
        for k, tensor in self.gpu.items():
            if self.video_buffers[k] is None:
                verbose_log(f"Preparing OpenGL buffer for key '{k}'")
                buf_id = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, buf_id)
                size = tensor.numel() * tensor.element_size()
                glBufferData(GL_ARRAY_BUFFER, size, None, GL_DYNAMIC_DRAW)
                if cuda_gl and cuda:
                    reg_buf = cuda_gl.RegisteredBuffer(int(buf_id))
                    self.video_buffers[k] = reg_buf
                    verbose_log(f"Registered CUDA-GL buffer for key '{k}'")
                else:
                    self.video_buffers[k] = buf_id
                glBindBuffer(GL_ARRAY_BUFFER, 0)
            else:
                verbose_log(f"OpenGL buffer for key '{k}' already exists")

    def sync_to_video_buffers(self):
        """
        Sync GPU tensor data to OpenGL buffers.
        Uses CUDA-GL mapping if available, else falls back to glBufferSubData.
        """
        if not self.use_video or not self.gpu:
            return
        for k, tensor in self.gpu.items():
            ogl_buf = self.video_buffers.get(k)
            if ogl_buf is None:
                verbose_log(f"No OpenGL buffer for key '{k}', skipping")
                continue
            if cuda_gl and isinstance(ogl_buf, cuda_gl.RegisteredBuffer):
                verbose_log(f"Mapping CUDA-GL buffer for key '{k}'")
                mapped_ptr, size = ogl_buf.map()
                cuda.memcpy_dtod(mapped_ptr, tensor.data_ptr(), tensor.numel() * tensor.element_size())
                ogl_buf.unmap()
                verbose_log(f"Unmapped CUDA-GL buffer for key '{k}'")
            else:
                buf_id = int(ogl_buf)
                glBindBuffer(GL_ARRAY_BUFFER, buf_id)
                size = tensor.numel() * tensor.element_size()
                cpu_view = tensor.detach().cpu().numpy()
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
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf)
        count = int(self.gpu[key].numel())
        glDispatchCompute(count // 128 + 1, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY)
        size = self.gpu[key].numel() * self.gpu[key].element_size()
        if ptr:
            arr = np.frombuffer((ctypes.c_byte * size).from_address(ptr), dtype=self.npdtype)
            self.gpu[key].copy_(torch.from_numpy(arr).cuda(), non_blocking=True)
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        glUseProgram(0)
        verbose_log(f"Ran compute shader '{name}' on key '{key}'")
        return self.gpu[key]

class GeneralQuadBufferSync:
    def __init__(self, keys, quadbuffer, manager=None, kernel_config=None):
        verbose_log(f"GeneralQuadBufferSync.__init__(keys={keys})")
        self.quadbuffer = quadbuffer
        self.keys = list(keys)
        if manager is None:
            num_pages = quadbuffer.shapes[0] if isinstance(quadbuffer.shapes, (list, tuple)) else None
            key_shapes = {k: s for k, s in zip(keys, quadbuffer.shapes)}
            kernels = kernel_config if kernel_config else None
            self.manager = LockManagerThread(LockGraph(num_pages=num_pages, key_shapes=key_shapes, kernels=kernels))
        else:
            self.manager = manager
        if not self.manager.running:
            self.manager.start()
        self.host = quadbuffer.data if quadbuffer.use_numpy else None
        self.cpu = quadbuffer.cpu if quadbuffer.use_torch else None
        self.gpu = quadbuffer.gpu if quadbuffer.use_torch else None
        self.abstract = quadbuffer.abstract if quadbuffer.use_abstract else None
        self.video = quadbuffer.video_buffers if quadbuffer.use_video else None
        self.host_dirty = set()
        self.cpu_dirty = set()
        self.abstract_dirty = set()
        self.cpu_worker = AsyncCPUSyncWorker(self, self.manager) if quadbuffer.use_torch else None
        self.gpu_worker = AsyncGPUSyncWorker(self, self.manager) if quadbuffer.use_torch else None
        if self.cpu_worker: self.cpu_worker.start()
        if self.gpu_worker: self.gpu_worker.start()

    def set_data(self, key, data):
        verbose_log(f"GeneralQuadBufferSync.set_data(key={key}, data_type={type(data)})")
        if self.host is not None and isinstance(data, np.ndarray):
            self.host[key] = data
            self.host_dirty.add(key)
            if self.cpu_worker: self.cpu_worker.enqueue(key)
            if self.gpu_worker: self.gpu_worker.enqueue(key)
        elif self.cpu is not None and isinstance(data, torch.Tensor) and data.device.type == 'cpu':
            pinned = data.pin_memory()
            self.cpu[key] = pinned
            self.cpu_dirty.add(key)
            if self.gpu_worker: self.gpu_worker.enqueue(key)
        elif self.gpu is not None and isinstance(data, torch.Tensor) and data.device.type == 'cuda':
            self.gpu[key] = data
            self.cpu_dirty.add(key)
            self.host_dirty.add(key)
            if self.cpu_worker: self.cpu_worker.enqueue(key)
        elif self.abstract is not None and isinstance(data, AbstractTensor):
            self.abstract[key] = data
            self.abstract_dirty.add(key)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def get_data(self, key, target='host'):
        verbose_log(f"GeneralQuadBufferSync.get_data(key={key}, target={target})")
        if target == 'host':
            return self.host[key] if self.host is not None else None
        elif target == 'cpu':
            return self.cpu.get(key) if self.cpu is not None else None
        elif target == 'gpu':
            return self.gpu.get(key) if self.gpu is not None else None
        elif target == 'abstract':
            return self.abstract.get(key) if self.abstract is not None else None
        elif target == 'video':
            return self.video.get(key) if self.video is not None else None
        else:
            raise ValueError(f"Invalid target: {target}")

    def shutdown(self):
        verbose_log("GeneralQuadBufferSync.shutdown()")
        if self.cpu_worker: self.cpu_worker.stop()
        if self.gpu_worker: self.gpu_worker.stop()
        self.manager.shutdown()
