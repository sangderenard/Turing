"""Small CUDA/OpenGL interop bridge using Torch's bundled CUDA runtime.

The project already depends on Torch for GPU physics, so importing another
CUDA binding just to move resident positions into an OpenGL buffer would add
the wrong dependency.  This module binds only the five graphics-resource
calls needed for a device-to-device copy.
"""

from __future__ import annotations

import ctypes
import glob
import os


class CudaInteropError(RuntimeError):
    pass


class CudaGLBuffer:
    """A CUDA registration for one OpenGL buffer object."""

    _WRITE_DISCARD = 1
    _DEVICE_TO_DEVICE = 3

    def __init__(self, gl_buffer: int, *, device_index: int = 0) -> None:
        import torch

        candidates = glob.glob(
            os.path.join(os.path.dirname(torch.__file__), "lib", "cudart64_*.dll")
        )
        if not candidates:
            raise CudaInteropError("Torch's CUDA runtime DLL was not found")
        self._runtime = ctypes.CDLL(candidates[0])
        self._resource = ctypes.c_void_p()
        self._closed = False
        self._check(self._runtime.cudaSetDevice(int(device_index)), "cudaSetDevice")
        self._check(
            self._runtime.cudaGraphicsGLRegisterBuffer(
                ctypes.byref(self._resource),
                ctypes.c_uint(int(gl_buffer)),
                ctypes.c_uint(self._WRITE_DISCARD),
            ),
            "cudaGraphicsGLRegisterBuffer",
        )

    def _check(self, result: int, operation: str) -> None:
        if int(result) == 0:
            return
        self._runtime.cudaGetErrorString.restype = ctypes.c_char_p
        message = self._runtime.cudaGetErrorString(int(result))
        detail = message.decode("utf-8", "replace") if message else "unknown error"
        raise CudaInteropError(f"{operation} failed ({result}): {detail}")

    def copy_from_tensor(self, tensor: object, byte_count: int) -> None:
        """Copy a contiguous CUDA tensor into the registered GL allocation."""

        if self._closed:
            raise CudaInteropError("CUDA/OpenGL resource is closed")
        if not getattr(tensor, "is_cuda", False):
            raise TypeError("CUDA/OpenGL interop requires a CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError("CUDA/OpenGL interop requires a contiguous tensor")
        resources = (ctypes.c_void_p * 1)(self._resource)
        self._check(
            self._runtime.cudaGraphicsMapResources(1, resources, ctypes.c_void_p()),
            "cudaGraphicsMapResources",
        )
        try:
            pointer = ctypes.c_void_p()
            capacity = ctypes.c_size_t()
            self._check(
                self._runtime.cudaGraphicsResourceGetMappedPointer(
                    ctypes.byref(pointer), ctypes.byref(capacity), self._resource
                ),
                "cudaGraphicsResourceGetMappedPointer",
            )
            if int(byte_count) > int(capacity.value):
                raise CudaInteropError(
                    f"CUDA copy needs {byte_count} bytes; GL buffer has {capacity.value}"
                )
            self._check(
                self._runtime.cudaMemcpy(
                    pointer,
                    ctypes.c_void_p(int(tensor.data_ptr())),
                    ctypes.c_size_t(int(byte_count)),
                    self._DEVICE_TO_DEVICE,
                ),
                "cudaMemcpy(device-to-device)",
            )
        finally:
            self._check(
                self._runtime.cudaGraphicsUnmapResources(
                    1, resources, ctypes.c_void_p()
                ),
                "cudaGraphicsUnmapResources",
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._check(
            self._runtime.cudaGraphicsUnregisterResource(self._resource),
            "cudaGraphicsUnregisterResource",
        )


__all__ = ["CudaGLBuffer", "CudaInteropError"]
