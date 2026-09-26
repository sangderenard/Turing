"""Run the four-stage sanctioned symbolic compile chain on a target file.

    1. compile_sympy_equations()        SymPy            -> ProcessGraph/SSA
    2. symbolic_abstract_tensor_source() SSA              -> AbstractTensor Python
    3. lower_ast_source_to_ssa()         AbstractTensor Py -> repository SSA
    4. emit_ssa_function_to_llvm()       repository SSA   -> LLVM

The target file is imported and must define either

  ``EQUATIONS``  one law: a sequence of ``sympy.Eq(Symbol(name), expr,
                 evaluate=False)``; optional ``NAME``, ``FUNCTION_NAME``,
                 ``PUBLICATIONS``.
  ``LAWS``       several laws: a mapping law name -> such a sequence, each
                 compiled as its own function; optional ``LAW_PUBLICATIONS``
                 mapping law name -> publications.

Optional in both forms: ``SCHEDULE``, ``DTYPE``, ``BATCH``.

Usage:
    python tools/compile_symbolic_source.py path/to/target.py [law ...]
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.symbolic_equation_compiler import compile_sympy_equations
from src.compiler.vehicle_python_compilation import symbolic_abstract_tensor_source
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
from src.compiler.native_law_kernels import batch_contract


def _load_target(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_law(name, equations, *, publications, schedule, dtype, batch):
    compilation = compile_sympy_equations(
        equations,
        name=name,
        schedule=schedule,
        publications=publications,
        dtype=dtype,
    )
    print(f"[{name}] STAGE1 compile_sympy_equations OK "
          f"inputs={len(compilation.input_ids)} outputs={len(compilation.output_ids)} "
          f"cache_hit={compilation.cache_hit}")

    source = symbolic_abstract_tensor_source(compilation, name)
    print(f"[{name}] STAGE2 symbolic_abstract_tensor_source OK chars={len(source)}")

    argument_names = compilation.function.metadata["argument_names"]
    extraction_contract = batch_contract(name, argument_names, batch)
    module, _outputs, exports = lower_ast_source_to_ssa(
        source,
        name,
        name=name,
        extraction_contract=extraction_contract,
    )
    print(f"[{name}] STAGE3 lower_ast_source_to_ssa OK exports={exports}")

    artifact = emit_ssa_function_to_llvm(module, exports[0])
    print(f"[{name}] STAGE4 emit_ssa_function_to_llvm OK name={artifact.name} "
          f"llvm_ir_chars={len(artifact.llvm_ir)}")
    return artifact


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1

    path = Path(argv[0])
    target = _load_target(path)
    schedule = getattr(target, "SCHEDULE", "asap")
    dtype = getattr(target, "DTYPE", "float64")
    batch = getattr(target, "BATCH", 1)

    laws = getattr(target, "LAWS", None)
    if laws is None:
        name = getattr(target, "NAME", path.stem)
        laws = {getattr(target, "FUNCTION_NAME", name): target.EQUATIONS}
        law_publications = {next(iter(laws)): getattr(target, "PUBLICATIONS", ())}
    else:
        law_publications = getattr(target, "LAW_PUBLICATIONS", {})

    selected = argv[1:] or list(laws)
    for law in selected:
        compile_law(
            law, laws[law],
            publications=law_publications.get(law, ()),
            schedule=schedule, dtype=dtype, batch=batch,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
