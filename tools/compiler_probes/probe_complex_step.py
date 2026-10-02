"""Lane FD probe: does complex-step differentiation survive each stage?

    python tools/compiler_probes/probe_complex_step.py eager
    python tools/compiler_probes/probe_complex_step.py native

f(x) evaluated at x + i*h (h = 1e-30); Im f / h vs the analytic f'(x).
Results are recorded in docs/DIFFERENTIATION_FEASIBILITY_2026-10-02.md.
Probe only: edits nothing, commits nothing.
"""
from __future__ import annotations

import math
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

H = 1.0e-30
X = 0.7


def stamp(msg: str, t0=[time.perf_counter()]) -> None:
    print(f"[{time.perf_counter() - t0[0]:8.1f}s] {msg}", flush=True)


# name -> (f on an AbstractTensor-like, analytic derivative on float)
CASES = {
    "sin*exp": (lambda z: z.sin() * z.exp(),
                lambda x: math.exp(x) * (math.sin(x) + math.cos(x))),
    "x**3+log": (lambda z: z ** 3 + z.log(),
                 lambda x: 3 * x * x + 1 / x),
    "sqrt/(1+x^2)": (lambda z: z.sqrt() / (z * z + 1.0),
                     lambda x: (0.5 / math.sqrt(x)) / (x * x + 1)
                     - math.sqrt(x) * 2 * x / (x * x + 1) ** 2),
    "squire-trapp": (lambda z: z.exp() / (z.cos() ** 3 + z.sin() ** 3).sqrt(),
                     None),
}


def squire_trapp_prime(x: float) -> float:
    s, c = math.sin(x), math.cos(x)
    g = c ** 3 + s ** 3
    return math.exp(x) / math.sqrt(g) - 0.5 * math.exp(x) * g ** -1.5 * (
        3 * s * s * c - 3 * c * c * s)


def measure_eager() -> int:
    stamp("importing AbstractTensor")
    from src.common.tensors.abstraction import AbstractTensor
    import numpy as np

    z = AbstractTensor.tensor([X + 1j * H])
    stamp(f"input: backend {type(z).__name__}, data dtype "
          f"{getattr(getattr(z, 'data', None), 'dtype', None)}, data "
          f"{getattr(z, 'data', None)!r}")
    for name, (f, fprime) in CASES.items():
        exact = squire_trapp_prime(X) if fprime is None else fprime(X)
        try:
            w = f(z)
            data = np.asarray(w.data if hasattr(w, "data") else w)
            print(f"   {name}: result dtype {data.dtype}, value {data!r}")
            if np.iscomplexobj(data):
                est = float(data.imag.reshape(-1)[0]) / H
                print(f"      Im f/h = {est:.17g}  analytic {exact:.17g}  "
                      f"rel err {abs(est - exact) / abs(exact):.3e}")
            else:
                print("      complex DROPPED (real result)")
        except Exception as exc:  # noqa: BLE001 -- report the stage verbatim
            print(f"   {name}: FAILED {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=4)
    return 0


# Each form is a batch AbstractTensor function lowered the way
# native_law_kernels._lower_law lowers a law (batch_contract, x a span).
SOURCES = {
    # complex built INSIDE the program from a complex literal
    "literal": ("float64", (
        "def tick(x):\n"
        "    z = x + 1e-30j\n"
        "    w = z.sin() * z.exp()\n"
        "    return w.imag() / 1e-30\n")),
    # complex built inside by the declared tensor op AbstractTensor.complex
    "complex_op": ("float64", (
        "def tick(x):\n"
        "    z = AbstractTensor.complex(x, x * 0.0 + 1e-30)\n"
        "    w = z.sin() * z.exp()\n"
        "    return w.imag() / 1e-30\n")),
    # the real program, its column declared complex128 and fed complex
    "fed": ("complex128", (
        "def tick(x):\n"
        "    return x.sin() * x.exp()\n")),
}


def _contract(dtype: str):
    from src.compiler.extraction_contract import ExtractionContract
    contracts = REPO / "extraction_contracts"
    return ExtractionContract(contracts / "program_extraction.yaml").with_program_abi({
        "records": {}, "bindings": [], "values": [{
            "function": "tick", "parameter": "x", "storage": "span",
            "dtype": dtype, "rank": 1, "shape": [1],
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }]}).with_execution_file(contracts / "vehicle_full_native_execution.yaml")


def measure_native() -> int:
    import tempfile
    import numpy as np
    stamp("importing compiler")
    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference)
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c

    exact = CASES["sin*exp"][1](X)
    for label in sys.argv[2:] or list(SOURCES):
        dtype, source = SOURCES[label]
        print(f"=== {label} (x declared {dtype})", flush=True)
        stage = "lower_ast_source_to_ssa"
        try:
            lowered = lower_ast_source_to_ssa(
                source, "tick", python_bindings={"AbstractTensor": AbstractTensor},
                tensor_ssa_reference=c_backend_repository_ssa_reference(),
                name=f"cs_{label}", runtime_closure_only=True,
                extraction_contract=_contract(dtype))
            module = lowered[0] if isinstance(lowered, tuple) else lowered.module
            entry = next(n for n in module.functions if n.endswith("__tick"))
            fn = module.functions[entry]
            stamp(f"{label}: lowered; entry {entry}; arg dtypes "
                  f"{[getattr(a, 'dtype', None) for a in fn.args]}")
            stage = "emit_ssa_module_to_c"
            artifact = emit_ssa_module_to_c(module, entry)
            if not artifact.complete:
                print("   STOPPED at C emission (incomplete): " + "; ".join(
                    f"{s.operation}: {s.reason}" for s in artifact.shortfalls[:6]))
                continue
            stage = "compile"
            artifact.compile(Path(tempfile.mkdtemp(prefix=f"fd_cs_{label}_")))
            stamp(f"{label}: compiled")
            stage = "run"
            ids = dict(fn.metadata["parameter_names"])
            npdtype = np.complex128 if dtype == "complex128" else np.float64
            x = np.array([X + 1j * H] if dtype == "complex128" else [X], dtype=npdtype)
            feeds = {int(ids["x"]): x}
            execution = artifact.prepare_execution(feeds)
            execution.run()
            outs = {k: np.asarray(v) for k, v in execution.buffers.items()
                    if int(k) != int(ids["x"])}
            print(f"   buffers after run: {[(k, v.dtype, v.reshape(-1)[:2]) for k, v in outs.items()]}")
            for k, y in outs.items():
                val = y.reshape(-1)[0]
                est = (float(np.imag(val)) / H if dtype == "complex128"
                       else float(np.real(val)))
                print(f"   buffer {k}: Im f/h = {est!r} analytic {exact!r} "
                      f"rel err {abs(est - exact) / abs(exact):.3e}")
        except Exception as exc:  # noqa: BLE001 -- the stage that stops
            print(f"   STOPPED at {stage}: {type(exc).__name__}: {exc}"[:2000])
            traceback.print_exc(limit=-8)
    return 0


if __name__ == "__main__":
    sys.exit({"eager": measure_eager, "native": measure_native}[
        sys.argv[1] if len(sys.argv) > 1 else "eager"]())
