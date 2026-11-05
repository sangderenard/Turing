"""A bool span into an unannotated helper that converts it lowers.

The helper's ``to_dtype`` result returns through the call's result binding,
which turns the transported tensor contract dynamic after earlier static
posts of the same transport edge.  The edge's revision must cite that
binding (TENSOR_METADATA_RESULT_BINDING) as its cause; before it did, the
book refused "REVISE without a changed source".
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.tensors import AbstractTensor  # noqa: E402
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (  # noqa: E402
    c_backend_repository_ssa_reference,
)
from src.compiler.called_inventory import TensorClass  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.native_law_kernels import batch_contract  # noqa: E402

SOURCE = '''
def helper(m):
    return m.to_dtype("float64")

def law(x, dt):
    y = helper(x > 0.5)
    return y, x * 0.0 + 1.0
'''


def test_bool_span_into_converting_helper_lowers():
    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCE, "law", python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        name="helper_result_cause", runtime_closure_only=True,
        extraction_contract=batch_contract("law", ("x", "dt"), 4),
        root_calls=[{"x": TensorClass("float64"), "dt": TensorClass("float64")}],
        progress_bars=False,
    )
    assert exports and exports[0] in module.functions
