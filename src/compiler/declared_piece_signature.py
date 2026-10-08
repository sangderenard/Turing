"""Declared-piece signature definitions (relocated from fortran_c_shell)."""

from __future__ import annotations

import ast
from typing import Any


_PIECE_FLOAT64_SPELLINGS = frozenset({"double", "float64", "f64", "float"})


def _declared_piece_signature_definition(name: str, piece: Any) -> ast.FunctionDef:
    """The def a linked piece enters the source compiler as: its DECLARED
    signature, and nothing of its authored body.

    A piece bound at a call site (``LLVMPiece`` or a declared external
    ``ExternalFunction``) is already compiled.  The link supplies its SSA
    (``linked_repository_ssa`` -> ``host_ssa_module`` on the function table)
    and the planned shell of this def is dropped for that module
    (``host_repository_ssa_complete``).  So the call site needs exactly three
    facts from this def, and the piece declares all three:

    * the formal names, in call order -- ``argument_names``;
    * the result arity and order -- ``output_names`` (one unpacked target
      per name at every call: ``o_a, o_b, = step_i(...)``);
    * each result's shape and dtype -- the artifact's ``buffer_shapes`` /
      ``buffer_dtypes`` at ``output_ids[name]``, the same ``piece_abi`` the
      extraction contract decides the call on
      (``ExtractionContract._decide_llvm_piece``), so the caller's projections
      acquire their tensor descriptors in the graph phase and the ABI check
      at the SSA link meets the same shape the host root returns.

    Parsing ``piece.source`` for these facts put the whole authored law body
    back into the ProcessGraph -- 131,700 of 224,889 nodes on the 35-piece
    orbital dt system, every one reduced, planned and specialized, then
    dropped at the planned shell.  Nothing read that body: inline mode
    (``llvm_dt_system.lowered_system(piece_mode="inline")``) emits the
    linked SSA by popping the ``llvm_piece`` receipt on the piece root; it
    never looks at the def.

    The body is the declaration form ``external_functions.declare_external``
    already uses for a piece-shaped leaf: NaN of the result's shape, never
    emitted (the link supplies the SSA), never a plausible number if a lane
    ran it.  A rank-0 result is ``float('nan')``; a ranked result is
    ``<first formal declared at that shape> * float('nan')``; a
    ``constant_outputs`` entry is its declared literal, exactly as the
    authored source spelled it.  A result at a shape no formal is declared
    at, or at a non-float64 dtype, has no declaration form here and is
    refused -- as ``declare_external`` refuses -- rather than declared
    approximately.
    """

    artifact = piece.artifact
    buffer_order = tuple(int(v) for v in artifact.buffer_order)
    shapes = {
        value_id: tuple(int(n) for n in (shape or ()))
        for value_id, shape in zip(buffer_order, artifact.buffer_shapes or ())
    }
    dtypes = {
        value_id: str(dtype)
        for value_id, dtype in zip(buffer_order, artifact.buffer_dtypes or ())
    }
    argument_names = tuple(str(a) for a in piece.argument_names)
    argument_ids = tuple(int(v) for v in piece.argument_ids)
    constant_outputs = dict(piece.constant_outputs or {})
    output_ids = {str(k): int(v) for k, v in dict(piece.output_ids).items()}
    symbol = str(getattr(artifact, "name", piece.entry))

    def nan() -> ast.expr:
        return ast.Call(
            func=ast.Name(id="float", ctx=ast.Load()),
            args=[ast.Constant("nan")], keywords=[],
        )

    body: list[ast.stmt] = []
    returned: list[ast.expr] = []
    for output in map(str, piece.output_names):
        if output in constant_outputs:
            value: ast.expr = ast.Constant(float(constant_outputs[output]))
        else:
            value_id = output_ids.get(output)
            if value_id is None or value_id not in shapes:
                raise ValueError(
                    f"LLVM piece {name!r} ({symbol}): output {output!r} is "
                    "declared by neither output_ids nor constant_outputs; the "
                    "link cannot state its signature")
            dtype = dtypes.get(value_id)
            if dtype is not None and dtype not in _PIECE_FLOAT64_SPELLINGS:
                raise ValueError(
                    f"LLVM piece {name!r} ({symbol}): output {output!r} is "
                    f"declared {dtype!r}; the signature declaration form "
                    "states float64 results only")
            shape = shapes[value_id]
            if shape == ():
                value = nan()
            else:
                formal = next((
                    formal_name
                    for formal_name, formal_id in zip(argument_names, argument_ids)
                    if shapes.get(formal_id) == shape
                ), None)
                if formal is None:
                    raise ValueError(
                        f"LLVM piece {name!r} ({symbol}): output {output!r} is "
                        f"declared at shape {shape} and no formal is declared "
                        "at that shape; the signature declaration form has no "
                        "spelling for it")
                value = ast.BinOp(
                    left=ast.Name(id=formal, ctx=ast.Load()), op=ast.Mult(),
                    right=nan(),
                )
        body.append(ast.Assign(
            targets=[ast.Name(id=output, ctx=ast.Store())], value=value,
        ))
        returned.append(ast.Name(id=output, ctx=ast.Load()))
    if not returned:
        raise ValueError(
            f"LLVM piece {name!r} ({symbol}) declares no outputs; the link "
            "cannot state its signature")
    body.append(ast.Return(
        value=returned[0] if len(returned) == 1
        else ast.Tuple(elts=returned, ctx=ast.Load())
    ))
    return ast.FunctionDef(
        name=str(name),
        args=ast.arguments(
            posonlyargs=[], vararg=None, kwonlyargs=[], kw_defaults=[],
            kwarg=None, defaults=[],
            args=[ast.arg(arg=formal) for formal in argument_names]),
        body=body, decorator_list=[], returns=None,
    )
