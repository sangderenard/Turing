"""Auto-port foreign source into ``common.tensors.absorbed`` as real modules.

This is the machine that fills that package: it compiles authored source
through this tree's own pipeline and writes the emitted AbstractTensor Python
out with its provenance attached.  It never edits the emitted code -- if the
translation is wrong or ugly, the compiler is what changes, and the module is
regenerated.

Used as a library::

    from tools.absorb_source import absorb
    result = absorb(source_text, ["cqt_frequencies"], name="spectral_cqt")
    print(result.code)          # the emitted AbstractTensor Python
    print(result.skipped)       # functions the materializer refused, and why

or from the command line, which additionally writes the module::

    python tools/absorb_source.py <file.py> <entrypoint> [more entrypoints]
"""
from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_python_materializer import materialize_ir_module


@dataclass(frozen=True)
class AbsorptionResult:
    """What one translation produced, including what it would not translate."""

    name: str
    entrypoint: str
    code: str
    skipped: Mapping[str, str]
    emitted_names: tuple[str, ...]
    authored_parameters: tuple[str, ...]
    emitted_parameters: tuple[str, ...]

    @property
    def complete(self) -> bool:
        """True when the materializer refused nothing."""

        return not self.skipped

    @property
    def widened_parameters(self) -> tuple[str, ...]:
        """Formals the translation has that the authored function did not.

        A slice or an unresolved bound becomes a real parameter of the
        emitted function. Naming them is what keeps a widened signature a
        stated fact rather than a surprise at the call site.
        """

        return tuple(self.emitted_parameters[len(self.authored_parameters):])


def absorb(
    source: str,
    entrypoint: str,
    *,
    name: str,
    tensor_vocabulary: bool = True,
) -> AbsorptionResult:
    """Compile ``source`` and return its AbstractTensor translation.

    ``tensor_vocabulary`` defaults to True here, unlike the materializer's own
    default: this tool exists to produce AbstractTensor code, so the scalar
    reading would be the surprising one.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, outputs, _exports = lower_ast_source_to_ssa(
            source, entrypoint, name=name
        )

    qualified = f"{name}__{entrypoint}"
    if not outputs.get(qualified):
        raise RuntimeError(
            f"{entrypoint} lowered to no outputs at all. The statement was "
            "discarded rather than compiled -- absorbing this would deposit a "
            "module that computes nothing while every stage reported success."
        )

    emitted, skipped = materialize_ir_module(
        module, tensor_vocabulary=tensor_vocabulary
    )
    code = ast.unparse(emitted)

    function = module.functions[qualified]
    authored = tuple(dict(function.metadata.get("parameter_names") or ()))
    emitted_names = tuple(
        node.name for node in emitted.body if isinstance(node, ast.FunctionDef)
    )
    emitted_parameters: tuple[str, ...] = ()
    for node in emitted.body:
        if isinstance(node, ast.FunctionDef) and node.name == qualified:
            emitted_parameters = tuple(a.arg for a in node.args.args)

    return AbsorptionResult(
        name=name,
        entrypoint=entrypoint,
        code=code,
        skipped=dict(skipped),
        emitted_names=emitted_names,
        authored_parameters=authored,
        emitted_parameters=emitted_parameters,
    )


def render_module(
    results: Sequence[AbsorptionResult],
    *,
    absorption_repr: str,
    docstring: str,
) -> str:
    """One importable module holding several translations, with provenance."""

    header = [
        '"""' + docstring.rstrip() + '\n"""',
        "from __future__ import annotations",
        "",
        "from src.common.tensors import AbstractTensor",
        "from src.common.tensors.absorbed.provenance import Absorption",
        "",
        f"ABSORPTION = {absorption_repr}",
        "",
    ]
    bodies = [result.code for result in results]
    return "\n".join(header) + "\n" + "\n\n".join(bodies) + "\n"


def _main(argv: Sequence[str]) -> int:
    import pathlib

    if len(argv) < 3:
        print(__doc__)
        return 2
    path = pathlib.Path(argv[1])
    source = path.read_text(encoding="utf-8")
    for entrypoint in argv[2:]:
        result = absorb(source, entrypoint, name=path.stem)
        status = "complete" if result.complete else "PARTIAL"
        print(f"{entrypoint}: {status}")
        if result.widened_parameters:
            print(f"  widened by: {list(result.widened_parameters)}")
        for function_name, reason in result.skipped.items():
            print(f"  refused {function_name}: {reason}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
