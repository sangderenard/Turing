"""``view_identity_concordance --probe solve_dt | solve_loop``.

The two sources of probe_solve_in_loop.py (plain static loop) and
probe_solve_in_dt_loop.py (outer static driver around a dt-controlled
substep ``while``), lowered exactly as those probes lower them: the public
entry ``lower_ast_source_to_ssa``, the ``program_extraction`` contract plus a
declared ABI for the three tensor parameters, the repository tensor reference.
The sources are the same text as in the probes (those are scripts that run
on import, so they cannot be imported); keep them in step.

``lower_for_viewer(sink, which)`` returns (module, root symbol).
"""

from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

A0 = np.asarray([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)
B0 = np.asarray([1.0, 2.0], dtype=np.float64)
DIAG = np.eye(2, dtype=np.float64)

SOURCES = {
    "solve_loop": ("solve_loop", """
import torch

def solve_loop(a0: torch.Tensor, b0: torch.Tensor, diag: torch.Tensor):
    x = b0 * 0.0
    for step in range(4):
        A = a0 + diag * (0.25 * step)
        b = b0 + 0.5 * step
        x = x + torch.linalg.solve(A, b)
    return x
"""),
    "solve_dt": ("solve_dt_loop", """
import torch

def solve_dt_loop(a0: torch.Tensor, b0: torch.Tensor, diag: torch.Tensor):
    x = b0 * 0.0
    dt_cap = 0.5
    for outer in range(3):
        total = 0.0
        while 1.0 - total > 1e-15:
            remainder = 1.0 - total
            dt_try = min(dt_cap, remainder)
            A = a0 + diag * (dt_try + 0.25 * outer)
            b = b0 + dt_try
            x = x + torch.linalg.solve(A, b) * dt_try
            total = total + dt_try
            dt_cap = max(dt_try * 0.5, 0.1)
    return x
"""),
}


def lower_for_viewer(process_graph_sink, which="solve_dt"):
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    entry, source = SOURCES[which]
    contract = ExtractionContract(
        root / "extraction_contracts" / "program_extraction.yaml"
    ).with_program_abi({
        "records": {},
        "bindings": [],
        "values": [
            {
                "function": entry, "parameter": name,
                "storage": "span", "dtype": "float64", "rank": value.ndim,
                "shape": list(value.shape), "python_type": "AbstractTensor",
            }
            for name, value in (("a0", A0), ("b0", B0), ("diag", DIAG))
        ],
    })
    tag = f"abstract_{which}_viewer"
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, entry, name=tag, extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        resolved_process_graph_sink=process_graph_sink,
    )
    return module, f"{tag}__{entry}"
