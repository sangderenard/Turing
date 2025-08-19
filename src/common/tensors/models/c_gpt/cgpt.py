"""Python bindings for the minimal C GPT implementation."""

from __future__ import annotations

import os
from pathlib import Path
from cffi import FFI


ffi = FFI()
ffi.cdef(
    """
    typedef struct {
        size_t n_layers;
    } CGPTModel;

    void cgpt_forward(const CGPTModel* model,
                      const double* input,
                      size_t n_tokens,
                      double* output);
    """
)

SOURCE_PATH = Path(__file__).with_name("cgpt.c")
C_SOURCE = SOURCE_PATH.read_text()

_lib = ffi.verify(C_SOURCE, include_dirs=[str(Path(__file__).parent)])

class CGPT:
    """Thin wrapper around the C implementation."""

    def __init__(self, n_layers: int = 1) -> None:
        self.model = ffi.new("CGPTModel*", dict(n_layers=n_layers))

    def forward(self, inputs: list[float]) -> list[float]:
        buf = ffi.new("double[]", len(inputs))
        inp = ffi.new("double[]", [float(x) for x in inputs])
        _lib.cgpt_forward(self.model, inp, len(inputs), buf)
        return [buf[i] for i in range(len(inputs))]
