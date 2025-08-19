from ..double_buffer.base import *
from ..double_buffer.workers import AsyncCPUSyncWorker, AsyncGPUSyncWorker
from ..double_buffer.lock import LockManagerThread, LockGraph


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

