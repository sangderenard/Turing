# src/common/tensors/pyopengl_handler.py
from OpenGL.arrays import arraydatatype, numpymodule
import numpy as np

# Delegate to NumPy's handler, but force a NumPy view first (via __array__)
class _AbstractTensorHandler(numpymodule.NumpyHandler):
    @classmethod
    def asArray(cls, value, typeCode=None):
        arr = np.asarray(value, order="C")      # triggers AbstractTensor.__array__
        print(f"[DBG] PyOpenGL converting AbstractTensor to array: {arr}, typeCode={typeCode}")
        return super().asArray(arr, typeCode)   # dtype/contiguity handled here

def install_pyopengl_handlers():
    # Import here to avoid import cycles
    from .abstraction import AbstractTensor
    from .numpy_backend import NumPyTensorOperations

    reg = arraydatatype.ArrayDatatype.getRegistry()
    handler = _AbstractTensorHandler()

    # Register for both wrapper + backend instances (your traceback shows both)
    reg.register(handler, types=(AbstractTensor, NumPyTensorOperations))
