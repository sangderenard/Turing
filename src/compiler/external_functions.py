"""External functions: declared callees the host supplies at runtime.

An applied undefined SymPy Function (``r1(s)``, ``F1(s)``) names a function
the program calls but does not define.  It is compiled as a DECLARED
external: one ``ExternalDeclaration`` per undefined Function class (its name
and arity), posted on the book as a NOVEL external-call identity, with every
authored callsite DERIVED from it (``post_external_functions``).

At lowering it is the same piece-shaped leaf an ``LLVMPiece`` is -- the
buffer ABI ``void (void **buffers, int32_t *extents)`` -- tagged
``llvm_piece`` with ``binding: runtime-slot`` instead of ``static-link``
(``ExternalFunction``).  The C and LLVM lanes call it through a function
pointer slot table the program exports; the host fills the table at load
(``bind_external_slots``) with a C-callable piece: an ``LLVMPiece``'s own
entry, a native function, or a Python callable wrapped once over the buffer
ABI.  A slot left unfilled at call time records a fault the execution raises
on; it is never a silent zero.

The external's signature (argument and result dtype and shape) is declared
on the external and specialized at the callsite's shape.  The declaration
carries an optional ``derivative`` external (orbital work item 4).
"""

from __future__ import annotations

import copy
import ctypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_CONTRACTS = Path(__file__).resolve().parents[2] / "extraction_contracts"

#: The slot ABI of every external: the piece buffer ABI.
EXTERNAL_SLOT_FUNCTION = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int32))
RUNTIME_SLOT = "runtime-slot"


# -- declaration at the symbolic ingestion ------------------------------------

@dataclass(frozen=True)
class ExternalDeclaration:
    """One undefined SymPy Function class an equation set applies."""

    name: str
    arity: int


def declared_external_functions(expressions: Sequence[Any]) -> tuple[ExternalDeclaration, ...]:
    """The undefined Functions applied in ``expressions``, by their class.

    Declared by what SymPy says the node IS (an ``AppliedUndef``), never by
    its name.  One Function applied with two arities is refused."""

    import sympy
    from sympy.core.function import AppliedUndef

    arities: dict[str, int] = {}
    for expression in expressions:
        for node in sympy.preorder_traversal(expression):
            if not isinstance(node, AppliedUndef):
                continue
            name = str(node.func.__name__)
            arity = len(node.args)
            if arities.setdefault(name, arity) != arity:
                raise ValueError(
                    f"external function {name!r} is applied with {arities[name]} "
                    f"and {arity} arguments")
    return tuple(ExternalDeclaration(name, arities[name]) for name in sorted(arities))


class UndeclaredExternalDerivative(TypeError):
    """A Derivative of an external whose derivative the host did not declare."""


def resolve_external_derivatives(expression: Any, derivatives: Mapping[Any, Any],
                                 resolved: list | None = None) -> Any:
    """Replace each derivative of an external by its declared derivative external.

    ``derivatives`` maps an undefined Function class to the undefined
    Function class the host declares as its derivative (``r1 -> v1``,
    ``v1 -> a1``): ``Derivative(r1(s), (s, 2))`` is ``a1(s)``, and SymPy's
    chain-rule form ``Subs(Derivative(r1(x), x), x, g)`` is ``v1(g)``.  A
    derivative of an external with no declared derivative is refused naming
    the external; there is no finite-difference fallback.  Only unary
    externals carry a declared derivative (a partial derivative of a
    multi-argument external is refused)."""

    import sympy
    from sympy.core.function import AppliedUndef

    def nth(function, order: int, argument):
        current = function
        for _ in range(int(order)):
            following = derivatives.get(current)
            if following is None:
                raise UndeclaredExternalDerivative(
                    f"external {current.__name__!r} has no declared derivative "
                    f"(needed for {function.__name__!r} order {order}); declare it "
                    "with compile_sympy_equations(external_derivatives=...)")
            if resolved is not None:
                # every declaration the chain read (r1 -> v1, v1 -> a1)
                resolved.append(str(current.__name__))
            current = following
        return current(argument)

    def derivative_of(node):
        if not isinstance(node.expr, AppliedUndef):
            return None
        function = node.expr.func
        if len(node.expr.args) != 1:
            raise UndeclaredExternalDerivative(
                f"external {function.__name__!r} takes {len(node.expr.args)} "
                "arguments; a partial derivative of it has no declaration")
        (argument,) = node.expr.args
        orders = dict(node.variable_count)
        if set(orders) != {argument}:
            return sympy.S.Zero if argument not in orders and all(
                variable not in argument.free_symbols for variable in orders) else None
        return nth(function, orders[argument], argument)

    def resolve(node):
        if isinstance(node, sympy.Subs) and isinstance(node.expr, sympy.Derivative):
            inner = derivative_of(node.expr)
            if inner is not None:
                return inner.xreplace(dict(zip(node.variables, node.point)))
        if isinstance(node, sympy.Derivative):
            inner = derivative_of(node)
            if inner is not None:
                return inner
        return None

    return expression.replace(lambda node: resolve(node) is not None, resolve)


def derivatives_of_externals(expression: Any) -> tuple[str, ...]:
    """Names of the externals still under a Derivative in ``expression``."""

    import sympy
    from sympy.core.function import AppliedUndef

    return tuple(sorted({
        str(node.expr.func.__name__)
        for node in sympy.preorder_traversal(expression)
        if isinstance(node, sympy.Derivative) and isinstance(node.expr, AppliedUndef)}))


def resolved_derivative_form(expression: Any, derivatives: Mapping[Any, Any]) -> Any:
    """``expression`` with every Derivative differentiated by SymPy and each
    derivative of an external replaced by its declared derivative external:
    the externals the ingestion will call.  Undeclared ones stay Derivatives."""

    import sympy

    def differentiate(node):
        try:
            return resolve_external_derivatives(
                sympy.diff(node.expr, *node.variables), derivatives)
        except UndeclaredExternalDerivative:
            return node

    return expression.replace(lambda node: isinstance(node, sympy.Derivative), differentiate)


def post_external_functions(program: Any, equations: Sequence[Any],
                            equation_cells: Mapping[int, Any],
                            outputs: Sequence[tuple[str, int, Any]],
                            output_cells: Mapping[str, Any],
                            derivatives: Mapping[Any, Any] | None = None) -> dict[str, Any]:
    """Post each declared external (NOVEL, minted) and each callsite (DERIVED).

    ``outputs`` is ``(output name, equation index, expression)`` per declared
    output.  The externals are those the outputs call once each Derivative
    of an external is its declared derivative external (``derivatives``,
    ``r1 -> v1``).  Each is NOVEL(DECLARE_EXTERNAL_FUNCTION) from the cell of
    the first equation that calls it; its name row records the minted id
    (DERIVED from the external cell), so posting the same program again
    reuses the identity instead of minting a second one.  A declared
    derivative is an ``external_derivative`` row DERIVED from the external's
    and the derivative external's cells.  Each distinct call in an output is
    an ``external_callsite`` row DERIVED from the external cell and the
    output cell.  Returns the external cells by name.
    """

    import sympy
    from sympy.core.function import AppliedUndef

    from .concordance_declarations import (
        DECLARE_EXTERNAL_FUNCTION, EXTERNAL_CALLSITE, EXTERNAL_DERIVATIVE,
        EXTERNAL_FUNCTION, EXTERNAL_FUNCTION_NAME, INGESTION,
        ExternalCallsiteFact, ExternalDerivativeFact, ExternalFunctionFact,
    )
    from .identity_concordance import NEW, Derived, Mode, Novel, current_identity_book

    derivatives = dict(derivatives or {})
    book = current_identity_book()
    resolved_equations = [
        resolved_derivative_form(sympy.Tuple(eq.lhs, eq.rhs), derivatives)
        for eq in equations]
    declarations = declared_external_functions(resolved_equations)
    cells: dict[str, Any] = {}
    for declaration in declarations:
        named = book.latest_ref(EXTERNAL_FUNCTION_NAME, (program, declaration.name))
        if named is not None:
            minted = book.page(EXTERNAL_FUNCTION_NAME).latest(
                (program, declaration.name)).external_id
            cells[declaration.name] = book.latest_ref(EXTERNAL_FUNCTION, (program, minted))
            continue
        first = next(
            index for index, equation in enumerate(resolved_equations)
            if any(isinstance(node, AppliedUndef) and node.func.__name__ == declaration.name
                   for node in sympy.preorder_traversal(equation)))
        cell = book.post(
            EXTERNAL_FUNCTION, (program, NEW),
            ExternalFunctionFact(declaration.name, declaration.arity),
            stage=INGESTION,
            provenance=Novel(DECLARE_EXTERNAL_FUNCTION, (equation_cells[first],)),
            mode=Mode.CONCORD,
        )
        book.post(
            EXTERNAL_FUNCTION_NAME, (program, declaration.name),
            _name_fact(cell.row[1]),
            stage=INGESTION, provenance=Derived((cell,)), mode=Mode.CONCORD,
        )
        cells[declaration.name] = cell
    for function, derivative in derivatives.items():
        name, derived = str(function.__name__), str(derivative.__name__)
        if name in cells and derived in cells:
            book.post(
                EXTERNAL_DERIVATIVE, (program, name), ExternalDerivativeFact(derived),
                stage=INGESTION, provenance=Derived((cells[name], cells[derived])),
                mode=Mode.CONCORD,
            )
    for output, _equation_index, expression in outputs:
        seen = set()
        for node in sympy.preorder_traversal(resolved_derivative_form(expression, derivatives)):
            if not isinstance(node, AppliedUndef) or node in seen:
                continue
            seen.add(node)
            name = str(node.func.__name__)
            book.post(
                EXTERNAL_CALLSITE, (program, str(output), sympy.srepr(node)),
                ExternalCallsiteFact(name, len(node.args)),
                stage=INGESTION,
                provenance=Derived((cells[name], output_cells[str(output)])),
                mode=Mode.CONCORD,
            )
    return cells


def _name_fact(minted: int):
    from .concordance_declarations import ExternalFunctionNameFact

    return ExternalFunctionNameFact(int(minted))


# -- the piece-shaped leaf ----------------------------------------------------

@dataclass(frozen=True)
class ExternalSignature:
    """Arguments and result of one external at one callsite shape."""

    argument_dtypes: tuple[str, ...]
    argument_shapes: tuple[tuple[int, ...], ...]
    result_dtype: str = "float64"
    result_shape: tuple[int, ...] = ()


@dataclass
class ExternalSlotABI:
    """The declared buffer ABI of an external's slot.

    Shaped as an ``LLVMFunctionArtifact`` exposes its ABI (``name``,
    ``buffer_order``, ``buffer_dtypes``, ``buffer_shapes``, ``extent_order``)
    so the piece seam reads either one; there is no LLVM text and no library:
    the symbol is bound at load."""

    name: str
    buffer_order: tuple[int, ...]
    buffer_dtypes: tuple[str, ...]
    buffer_shapes: tuple[tuple[int, ...], ...]
    extent_order: tuple[tuple[int, str, int | None], ...]
    llvm_ir: str = ""
    library_path: Any = None


@dataclass(eq=False)
class ExternalFunction:
    """A declared external, specialized at one callsite shape.

    A piece-shaped leaf: the source compiler links ``module`` for the call's
    signature, and the C / LLVM lanes call slot ``name`` through the
    program's slot table.  ``implementation`` (optional) is what the host
    binds: a Python callable, an ``LLVMPiece`` or a native address.  Called
    eagerly it runs the implementation; with none it refuses."""

    name: str
    signature: ExternalSignature
    artifact: ExternalSlotABI
    argument_names: tuple[str, ...]
    argument_ids: tuple[int, ...]
    output_names: tuple[str, ...]
    output_ids: dict[str, int]
    module: Any
    entry: str
    outputs: Any
    source: str
    implementation: Any = None
    derivative: "ExternalFunction | None" = None
    identity: Any = None
    constant_outputs: dict = field(default_factory=dict)
    batch: int = 1
    binding: str = RUNTIME_SLOT
    #: The declared external this leaf specializes (``r1`` for the leaf
    #: ``r1__s4``); the host binds by it.  Defaults to ``name``.
    external: str = ""

    def __post_init__(self):
        if not self.external:
            self.external = self.name

    def piece_record(self) -> dict[str, Any]:
        """The ``llvm_piece`` metadata fields that make the leaf a slot call."""

        return {
            "binding": RUNTIME_SLOT,
            "external": self.external,
            "leaf": self.name,
            "external_identity": None if self.identity is None else repr(self.identity),
            "argument_slots": tuple(
                self.artifact.buffer_order.index(int(v)) for v in self.argument_ids),
            "output_slots": tuple(
                self.artifact.buffer_order.index(int(self.output_ids[n]))
                for n in self.output_names),
            "buffer_shapes": tuple(tuple(s) for s in self.artifact.buffer_shapes),
            "derivative": None if self.derivative is None else self.derivative.name,
        }

    def __call__(self, *arguments):
        if self.implementation is None:
            raise RuntimeError(f"external {self.name!r} has no implementation bound")
        return self.implementation(*arguments)


def _signature_contract(entry: str, names: Sequence[str], shapes: Sequence[tuple]):
    """The extraction contract declaring each span argument at its shape."""

    from .extraction_contract import ExtractionContract

    values = [{
        "function": entry, "parameter": name, "storage": "span",
        "dtype": "float64", "rank": len(shape), "shape": [int(n) for n in shape],
        "python_type": "src.common.tensors.abstraction.AbstractTensor",
    } for name, shape in zip(names, shapes) if shape]
    return ExtractionContract(
        _CONTRACTS / "program_extraction.yaml"
    ).with_program_abi(
        {"records": {}, "bindings": [], "values": values}
    ).with_execution_file(_CONTRACTS / "vehicle_full_native_execution.yaml")


def declare_external(name: str, signature: ExternalSignature, *,
                     implementation: Any = None,
                     derivative: ExternalFunction | None = None,
                     identity: Any = None, external: str = "") -> ExternalFunction:
    """The piece-shaped leaf of external ``name`` at ``signature``.

    With an ``LLVMPiece`` implementation the leaf IS that piece's ABI and
    linked SSA (its module, entry, buffer and extent order), so the host can
    bind the piece's own entry to the slot.  Otherwise the signature is
    lowered through the sanctioned entry from a declaration def whose body is
    NaN of the result's shape: the body is never emitted (the leaf is a slot
    call in every lane that honours ``llvm_piece``), and a lane that did run
    it would produce NaN, never a plausible number.  Buffers are the
    arguments then the result; one extent, the result's element count."""

    from .extraction_contract import llvm_piece_of

    piece = llvm_piece_of(implementation)
    if piece is not None and piece is implementation and not isinstance(
            implementation, ExternalFunction):
        artifact = implementation.artifact
        abi = ExternalSlotABI(
            name=f"turing_external_{name}",
            buffer_order=tuple(int(v) for v in artifact.buffer_order),
            buffer_dtypes=tuple(str(d) for d in (artifact.buffer_dtypes or ())),
            buffer_shapes=tuple(tuple(s) for s in artifact.buffer_shapes),
            extent_order=tuple(artifact.extent_order),
        )
        return ExternalFunction(
            name=name, signature=signature, artifact=abi,
            argument_names=tuple(implementation.argument_names),
            argument_ids=tuple(int(v) for v in implementation.argument_ids),
            output_names=tuple(implementation.output_names),
            output_ids={str(k): int(v) for k, v in implementation.output_ids.items()},
            # The leaf is tagged runtime-slot on its own copy: the host
            # piece's module stays the static piece it is.
            module=copy.deepcopy(implementation.module), entry=str(implementation.entry),
            outputs=implementation.outputs, source=implementation.source,
            implementation=implementation, derivative=derivative,
            identity=identity, batch=int(getattr(implementation, "batch", 1)),
            external=external or name,
        )
    if any(tuple(shape) != tuple(signature.result_shape)
           for shape in signature.argument_shapes):
        raise ValueError(
            f"external {name!r}: elementwise signature needs every argument at the "
            f"result shape {signature.result_shape}, got {signature.argument_shapes}")
    if not signature.argument_shapes:
        raise ValueError(f"external {name!r}: a zero-argument external is not declared")
    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from .fortran_c_shell import lower_ast_source_to_ssa

    names = tuple(f"a{index}" for index in range(len(signature.argument_shapes)))
    poison = " + ".join(f"{argument} * float('nan')" for argument in names)
    source = f"def {name}({', '.join(names)}):\n    return {poison}\n"
    module, outputs, exports = lower_ast_source_to_ssa(
        source, name,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        name=f"external_{name}", runtime_closure_only=True,
        extraction_contract=_signature_contract(name, names, signature.argument_shapes),
    )
    entry = exports[0]
    root = module.functions[entry]
    parameter_ids = {str(k): int(v) for k, v in dict(root.metadata.get("parameter_names") or ()).items()}
    named = dict(root.metadata.get("named_outputs") or ())
    (output_id,) = (int(v) for v in named.values())
    argument_ids = tuple(parameter_ids[argument] for argument in names)
    order = (*argument_ids, output_id)
    abi = ExternalSlotABI(
        name=f"turing_external_{name}",
        buffer_order=order,
        buffer_dtypes=("double",) * len(order),
        buffer_shapes=(*signature.argument_shapes, tuple(signature.result_shape)),
        extent_order=((output_id, "numel", None),),
    )
    return ExternalFunction(
        name=name, signature=signature, artifact=abi,
        argument_names=names, argument_ids=argument_ids,
        output_names=("result",), output_ids={"result": output_id},
        module=module, entry=entry, outputs=outputs, source=source,
        implementation=implementation, derivative=derivative, identity=identity,
        external=external or name,
    )


def _shape_tag(shape: tuple[int, ...]) -> str:
    return "s" + "x".join(str(int(n)) for n in shape)


def external_callsite_shapes(source: str, entry: str, argument_names: Sequence[str],
                             batch: int, callsites: Mapping[str, str]) -> dict[str, tuple]:
    """Each external callsite's (argument shapes, result shape).

    ``callsites`` maps a callsite's callee spelling (one per call
    instruction) to its external.  Read by running the law's own
    AbstractTensor stage once on batch columns, each callsite recording the
    shapes it is called at and answering ones of its argument's shape."""

    from src.common.tensors import AbstractTensor

    seen: dict[str, tuple] = {}

    def recorder(spelling):
        def record(*arguments):
            shapes = tuple(tuple(int(n) for n in getattr(a, "shape", ()) or ())
                           for a in arguments)
            result = shapes[0] if shapes else ()
            seen[spelling] = (shapes, result)
            if not result:
                return 1.0
            return AbstractTensor.get_tensor(np.ones(result))
        return record

    namespace: dict[str, Any] = {"AbstractTensor": AbstractTensor}
    namespace.update({spelling: recorder(spelling) for spelling in callsites})
    exec(compile(source, f"<{entry}>", "exec"), namespace)
    columns = [AbstractTensor.get_tensor(np.full(batch, 0.75)) for _ in argument_names]
    namespace[entry](*columns)
    return seen


def _respelled(compilation: Any, callees: Mapping[int, str]) -> Any:
    """``compilation`` with each external Call (by result id) calling
    ``callees[id]``; the cached compilation itself is not touched."""

    import dataclasses

    from ..transmogrifier.ssa import BasicBlock, Instr

    function = compilation.function
    blocks = {}
    for label, block in function.blocks.items():
        instrs = []
        for instruction in block.instrs:
            if (instruction.op in {"Call", "call"} and instruction.res is not None
                    and int(instruction.res.id) in callees):
                instruction = Instr(
                    instruction.op, list(instruction.args), instruction.res,
                    arg_roles=list(instruction.arg_roles),
                    attributes={**dict(instruction.attributes),
                                "callee": callees[int(instruction.res.id)]},
                    source_span=instruction.source_span,
                )
            instrs.append(instruction)
        blocks[label] = BasicBlock(label, instrs)
    specialized = copy.copy(function)
    specialized.blocks = blocks
    return dataclasses.replace(compilation, function=specialized)


def externals_for_law(compilation: Any, law: str, batch: int,
                      supplied: Mapping[str, Any] | None = None):
    """The law's declared externals, one leaf per (external, callsite shape).

    Returns ``(compilation, source, bindings)``: the compilation whose
    external calls are spelled with their specialized leaf names (an
    external called at one shape keeps its own name; one called at several,
    ``r1`` at ``s`` and at ``0``, gets ``r1__s4`` and ``r1__s``), the
    law's AbstractTensor source, and the leaf per name.  Each leaf is its own
    slot; the host fills every slot of an external with its one
    implementation (``bind_external_slots``).  ``supplied`` maps an external
    to its implementation (an ``LLVMPiece`` the leaf takes its ABI from, a
    host callable, or an ``ExternalFunction`` used as the leaf as is)."""

    from .vehicle_python_compilation import symbolic_abstract_tensor_source

    metadata = compilation.function.metadata
    declared = dict((str(name), int(arity))
                    for name, arity in metadata.get("external_functions") or ())
    if not declared:
        return compilation, symbolic_abstract_tensor_source(compilation, law), {}
    supplied = dict(supplied or {})
    calls = [
        instruction
        for block in compilation.function.blocks.values()
        for instruction in block.instrs
        if instruction.op in {"Call", "call"} and instruction.res is not None
        and str(instruction.attributes.get("callee") or "") in declared]
    probe_spellings = {
        int(call.res.id): f"{call.attributes['callee']}__callsite_{index}"
        for index, call in enumerate(calls)}
    external_of = {
        probe_spellings[int(call.res.id)]: str(call.attributes["callee"]) for call in calls}
    probe = _respelled(compilation, probe_spellings)
    shapes = external_callsite_shapes(
        symbolic_abstract_tensor_source(probe, law), law,
        tuple(metadata["argument_names"]), batch, external_of)
    by_external: dict[str, dict[tuple, list[int]]] = {}
    for call in calls:
        spelling = probe_spellings[int(call.res.id)]
        if spelling not in shapes:
            raise ValueError(f"{law}: external callsite {spelling!r} never ran")
        by_external.setdefault(external_of[spelling], {}).setdefault(
            shapes[spelling], []).append(int(call.res.id))
    callees: dict[int, str] = {}
    bindings: dict[str, ExternalFunction] = {}
    for name, specializations in sorted(by_external.items()):
        given = supplied.get(name)
        for (argument_shapes, result_shape), call_ids in sorted(specializations.items()):
            leaf = name if len(specializations) == 1 else f"{name}__{_shape_tag(result_shape)}"
            if len(argument_shapes) != declared[name]:
                raise ValueError(f"{law}: external {name!r} arity {declared[name]} != "
                                 f"callsite {len(argument_shapes)}")
            if isinstance(given, ExternalFunction):
                if len(specializations) != 1:
                    raise ValueError(
                        f"{law}: external {name!r} is called at "
                        f"{len(specializations)} shapes; one ExternalFunction leaf "
                        "cannot serve them")
                bindings[leaf] = given
            else:
                bindings[leaf] = declare_external(
                    leaf, ExternalSignature(("float64",) * len(argument_shapes),
                                            argument_shapes, "float64", result_shape),
                    implementation=given, external=name)
            for call_id in call_ids:
                callees[call_id] = leaf
    # Each external's declared derivative external (orbital item 4): the
    # leaf carries the derivative's leaf at the same shape, and its
    # ``llvm_piece`` record names it.
    leaves_of: dict[str, dict[tuple, ExternalFunction]] = {}
    for leaf in bindings.values():
        leaves_of.setdefault(leaf.external, {})[leaf.signature.result_shape] = leaf
    for function, derivative in metadata.get("external_derivatives") or ():
        for shape, leaf in leaves_of.get(function, {}).items():
            leaf.derivative = leaves_of.get(derivative, {}).get(shape)
    specialized = _respelled(compilation, callees)
    return specialized, symbolic_abstract_tensor_source(specialized, law), bindings


# -- the slot table in the C and LLVM lanes -----------------------------------

def runtime_slot_functions(module: Any, names: Sequence[str]) -> tuple[str, ...]:
    """The runtime-slot leaves among ``names``, in slot order (by external)."""

    rows = []
    for name in names:
        function = module.functions.get(name)
        record = None if function is None else function.metadata.get("llvm_piece")
        if record and record.get("binding") == RUNTIME_SLOT:
            rows.append((str(record["external"]), str(name)))
    return tuple(name for _external, name in sorted(rows))


def external_slot_rows(module: Any, slot_functions: Sequence[str]) -> tuple[dict, ...]:
    """The artifact's declared slots: slot index, external name and ABI."""

    rows = []
    for slot, name in enumerate(slot_functions):
        record = module.functions[name].metadata["llvm_piece"]
        rows.append({
            "slot": slot, "external": str(record["external"]),
            "identity": record.get("external_identity"),
            "buffer_order": tuple(record["buffer_order"]),
            "extent_order": tuple(record["extent_order"]),
            "argument_slots": tuple(record["argument_slots"]),
            "output_slots": tuple(record["output_slots"]),
            "buffer_shapes": tuple(record["buffer_shapes"]),
            "derivative": record.get("derivative"),
        })
    return tuple(rows)


def c_slot_runtime(entry: str, count: int) -> list[str]:
    """The C module's slot table, its exported binder and fault reader."""

    return [
        "typedef void (*turing_external_fn)(void **buffers, int32_t *extents);",
        f"static turing_external_fn {entry}__external_slots[{max(count, 1)}];",
        f"static int32_t {entry}__external_fault = 0;",
        f"TURING_EXPORT int32_t {entry}__bind_external(int32_t slot, void *fn) {{",
        f"    if (slot < 0 || slot >= {count}) return -1;",
        f"    {entry}__external_slots[slot] = (turing_external_fn)fn;",
        "    return 0;",
        "}",
        f"TURING_EXPORT int32_t {entry}__external_fault_take(void) {{",
        f"    int32_t fault = {entry}__external_fault;",
        f"    {entry}__external_fault = 0;",
        "    return fault;",
        "}",
    ]


def c_slot_call(entry: str, slot: int) -> list[str]:
    """The shim's call through slot ``slot``; an empty slot records the fault."""

    return [
        f"    turing_external_fn external_fn = {entry}__external_slots[{slot}];",
        f"    if (external_fn == NULL) {{ if ({entry}__external_fault == 0) "
        f"{entry}__external_fault = {slot + 1}; return; }}",
        "    external_fn(piece_buffers, piece_extents);",
    ]


def llvm_slot_runtime(entry: str, count: int) -> dict[str, str]:
    """The LLVM module's slot table, binder and fault reader, by symbol."""

    table = f"@{entry}.external_slots"
    fault = f"@{entry}.external_fault"
    count = max(int(count), 1)
    return {
        f"{entry}.external_slots": f"{table} = internal global [{count} x ptr] zeroinitializer, align 8",
        f"{entry}.external_fault": f"{fault} = internal global i32 0, align 4",
        f"{entry}__bind_external": "\n".join((
            f"define i32 @{entry}__bind_external(i32 %slot, ptr %fn) {{",
            "entry:",
            "  %low = icmp slt i32 %slot, 0",
            f"  %high = icmp sge i32 %slot, {count}",
            "  %bad = or i1 %low, %high",
            "  br i1 %bad, label %refuse, label %store",
            "refuse:",
            "  ret i32 -1",
            "store:",
            "  %index = sext i32 %slot to i64",
            f"  %cell = getelementptr [{count} x ptr], ptr {table}, i64 0, i64 %index",
            "  store ptr %fn, ptr %cell, align 8",
            "  ret i32 0",
            "}",
        )),
        **{
            f"{entry}.external_unbound.{slot}": "\n".join((
                f"define internal void @{entry}.external_unbound.{slot}(ptr %buffers, ptr %extents) {{",
                "entry:",
                f"  %prior = load i32, ptr {fault}, align 4",
                "  %first = icmp eq i32 %prior, 0",
                f"  %code = select i1 %first, i32 {slot + 1}, i32 %prior",
                f"  store i32 %code, ptr {fault}, align 4",
                "  ret void",
                "}",
            ))
            for slot in range(count)
        },
        f"{entry}__external_fault_take": "\n".join((
            f"define i32 @{entry}__external_fault_take() {{",
            "entry:",
            f"  %fault = load i32, ptr {fault}, align 4",
            f"  store i32 0, ptr {fault}, align 4",
            "  ret i32 %fault",
            "}",
        )),
    }


def llvm_slot_call(entry: str, slot: int, count: int, tag: str,
                   table: str, extents: str) -> list[str]:
    """Call slot ``slot``.  An empty slot calls that slot's unbound stub,
    which records the fault.  No block is split, so the caller's phis keep
    their predecessors."""

    count = max(int(count), 1)
    return [
        f"  %external.cell.{tag} = getelementptr [{count} x ptr], "
        f"ptr @{entry}.external_slots, i64 0, i64 {slot}",
        f"  %external.fn.{tag} = load ptr, ptr %external.cell.{tag}, align 8",
        f"  %external.unbound.{tag} = icmp eq ptr %external.fn.{tag}, null",
        f"  %external.target.{tag} = select i1 %external.unbound.{tag}, "
        f"ptr @{entry}.external_unbound.{slot}, ptr %external.fn.{tag}",
        f"  call void %external.target.{tag}(ptr {table}, ptr {extents})",
    ]


# -- binding at load ----------------------------------------------------------

class ExternalSlotBinding:
    """The host's slot fills for one loaded artifact; keeps the callables alive."""

    def __init__(self, artifact: Any, library: Any, entry: str, keep: list,
                 errors: list | None = None):
        self.artifact = artifact
        self.errors = errors if errors is not None else []
        self.library = library
        self.entry = entry
        self.keep = keep
        take = getattr(library, f"{entry}__external_fault_take")
        take.restype = ctypes.c_int32
        take.argtypes = []
        self._take = take

    def check(self) -> None:
        """Raise if a call met an unfilled slot since the last check."""

        fault = int(self._take())
        if self.errors:
            error = self.errors[0]
            self.errors.clear()
            raise RuntimeError(f"{self.entry}: a host external raised") from error
        if fault:
            row = self.artifact.external_slots[fault - 1]
            raise RuntimeError(
                f"{self.entry}: external {row['external']!r} (slot {fault - 1}) was "
                "called with its slot unfilled; its result was not computed")


def _python_slot(row: Mapping[str, Any], function: Any, errors: list):
    """Wrap a Python callable once as a C-callable piece over the buffer ABI."""

    shapes = tuple(tuple(int(n) for n in s) for s in row["buffer_shapes"])
    argument_slots = tuple(row["argument_slots"])
    (output_slot,) = tuple(row["output_slots"])
    out_shape = shapes[output_slot]
    count = int(np.prod(out_shape)) if out_shape else 1

    def call(buffers, extents):
        try:
            body(buffers, extents)
        except BaseException as error:  # noqa: BLE001 -- a callback cannot raise
            # into native code; the result is NaN and the binding's check()
            # raises this error after the run.
            errors.append(error)
            target = np.ctypeslib.as_array(
                ctypes.cast(buffers[output_slot], ctypes.POINTER(ctypes.c_double)), (count,))
            target[:] = np.nan

    def body(buffers, extents):
        if int(extents[0]) != count:
            raise RuntimeError(
                f"external {row['external']!r}: extent {int(extents[0])} != declared {count}")
        arguments = [
            np.ctypeslib.as_array(
                ctypes.cast(buffers[slot], ctypes.POINTER(ctypes.c_double)), (count,)
            ).reshape(shapes[slot]).copy()
            for slot in argument_slots]
        if not out_shape:
            arguments = [float(a) for a in arguments]
        result = np.asarray(function(*arguments), dtype=np.float64)
        target = np.ctypeslib.as_array(
            ctypes.cast(buffers[output_slot], ctypes.POINTER(ctypes.c_double)), (count,))
        target[:] = np.broadcast_to(result, out_shape).reshape(count)

    return EXTERNAL_SLOT_FUNCTION(call)


def _slot_address(row: Mapping[str, Any], implementation: Any, keep: list,
                  errors: list) -> int:
    from .extraction_contract import llvm_piece_of

    if isinstance(implementation, ExternalFunction):
        implementation = implementation.implementation
    if isinstance(implementation, int):
        return implementation
    piece = llvm_piece_of(implementation)
    if piece is not None and getattr(piece.artifact, "library_path", None):
        if tuple(int(v) for v in piece.artifact.buffer_order) != tuple(row["buffer_order"]):
            raise ValueError(
                f"external {row['external']!r}: LLVM piece buffer order "
                f"{piece.artifact.buffer_order} != declared {row['buffer_order']}")
        entry = piece.artifact.entry()
        keep.append(piece.artifact)
        return int(ctypes.cast(entry, ctypes.c_void_p).value)
    if callable(implementation):
        wrapped = _python_slot(row, implementation, errors)
        keep.append(wrapped)
        return int(ctypes.cast(wrapped, ctypes.c_void_p).value)
    raise TypeError(f"external {row['external']!r}: cannot bind {implementation!r}")


def bind_external_slots(artifact: Any, implementations: Mapping[str, Any]) -> ExternalSlotBinding:
    """Fill every declared slot of a compiled C or LLVM artifact.

    Each declared external must be supplied (a slot left empty at load is
    refused here; one emptied later faults at call time).  Returns the
    binding, whose ``check()`` raises if a call met an unfilled slot."""

    rows = tuple(getattr(artifact, "external_slots", ()) or ())
    missing = sorted({row["external"] for row in rows} - set(implementations))
    if missing:
        raise ValueError(f"{artifact.name}: externals not supplied: {missing}")
    artifact.entry()
    library = ctypes.CDLL(str(artifact.library_path))
    binder = getattr(library, f"{artifact.name}__bind_external")
    binder.restype = ctypes.c_int32
    binder.argtypes = [ctypes.c_int32, ctypes.c_void_p]
    keep: list = []
    errors: list = []
    for row in rows:
        address = _slot_address(row, implementations[row["external"]], keep, errors)
        if int(binder(int(row["slot"]), ctypes.c_void_p(address))) != 0:
            raise RuntimeError(f"{artifact.name}: slot {row['slot']} refused")
    binding = ExternalSlotBinding(artifact, library, artifact.name, keep, errors)
    _BINDINGS[(str(artifact.library_path), artifact.name)] = binding
    return binding


#: Live bindings by (library, entry): the wrapped callables must outlive every
#: call the native program makes through them.
_BINDINGS: dict[tuple[str, str], ExternalSlotBinding] = {}


def check_external_faults(artifact: Any) -> None:
    """Raise if the artifact's last run met an unfilled slot."""

    rows = getattr(artifact, "external_slots", ()) or ()
    if not rows:
        return
    binding = _BINDINGS.get((str(artifact.library_path), artifact.name))
    if binding is None:
        library = ctypes.CDLL(str(artifact.library_path))
        binding = ExternalSlotBinding(artifact, library, artifact.name, [])
    binding.check()
