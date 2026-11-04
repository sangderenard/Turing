"""Selected native tensor families, without strengthening authored ``Any``.

The Huffman selector uses topk's raw indices through ensure_tensor. A span
layout does not prove that Python branch. Inspect the selected backend's
actual raw-type declaration/return and the public wrapper before reboxing
that exact aggregate output as a storage view.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import textwrap

from .concordance_declarations import (
    INGESTION, INGEST_SOURCE, NATIVE_TENSOR_FAMILY_SOURCE,
    NATIVE_TENSOR_RESULT_FAMILY, NATIVE_TENSOR_WRAP, SOURCE_AGGREGATE_LAYOUT,
    SSA_VALUE, TENSOR_SSA_LOWERING,
    SOURCE_VALUE_CLASS, VALUE_TYPE_ABI,
)
from .identity_concordance import Derived, Mode, Novel
from .ssa_record_return_state import function_scope_of, ssa_value_identity_cell


def _assignments(tree):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]


def _single_assignment(tree, name):
    values = [node.value for node in _assignments(tree)
              if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id == name]
    return values[0] if len(values) == 1 else None


def raw_result_slots(fact):
    """Recheck the stored source fragments; unknown shapes/types prove nothing."""
    if not isinstance(fact, tuple) or len(fact) != 7:
        return ()
    module, backend, raw_type, declaration, hook_text, public_text, wrap_text = fact
    try:
        declaration = ast.parse(declaration).body[0]
        hook = ast.parse(hook_text).body[0]
        public = ast.parse(public_text).body[0]
        wrap = ast.parse(wrap_text).body[0]
    except (SyntaxError, TypeError, IndexError):
        return ()
    if (not module or not backend or not raw_type
            or not isinstance(declaration, ast.Assign)
            or len(declaration.targets) != 1
            or not isinstance(declaration.targets[0], ast.Name)
            or declaration.targets[0].id != "tensor_type_"
            or not isinstance(declaration.value, ast.Name)
            or declaration.value.id != raw_type):
        return ()
    if not all(isinstance(node, ast.FunctionDef) for node in (hook, public, wrap)):
        return ()
    # The public operation binds the exact backend hook's ordered results.
    bindings = [node for node in _assignments(public)
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == hook.name]
    returns = [node.value for node in ast.walk(hook)
               if isinstance(node, ast.Return)]
    public_returns = [node.value for node in ast.walk(public)
                      if isinstance(node, ast.Return)]
    if (len(bindings) != 1 or len(returns) != 1
            or not isinstance(returns[0], ast.Tuple)
            or len(public_returns) != 1
            or not isinstance(public_returns[0], ast.Call)):
        return ()
    targets = bindings[0].targets[0].elts
    outputs = returns[0].elts
    if len(targets) != len(outputs) or any(not isinstance(n, ast.Name) for n in targets):
        return ()
    # The raw-type branch must attach the input as .data and return that
    # same wrapper. A conversion, clone, or different return is not an alias.
    wrap_ok = False
    backend_assignment = _single_assignment(wrap, "backend_cls")
    if (backend_assignment is None or ast.dump(backend_assignment)
            != ast.dump(ast.parse("self.__class__", mode="eval").body)):
        return ()
    for branch in ast.walk(wrap):
        if not isinstance(branch, ast.If):
            continue
        expected = ast.parse("isinstance(tensor, self.tensor_type)", mode="eval").body
        if ast.dump(branch.test) != ast.dump(expected):
            continue
        if len(branch.body) != 3:
            continue
        create, attach, result = branch.body
        wrap_ok = (
            isinstance(create, ast.Assign) and len(create.targets) == 1
            and isinstance(create.targets[0], ast.Name)
            and isinstance(create.value, ast.Call)
            and isinstance(create.value.func, ast.Name)
            and create.value.func.id == "backend_cls"
            and isinstance(attach, ast.Assign) and len(attach.targets) == 1
            and ast.dump(attach.targets[0]) == ast.dump(ast.Attribute(
                value=ast.Name(id=create.targets[0].id, ctx=ast.Load()),
                attr="data", ctx=ast.Store()))
            and isinstance(attach.value, ast.Name) and attach.value.id == "tensor"
            and isinstance(result, ast.Return) and isinstance(result.value, ast.Name)
            and result.value.id == create.targets[0].id
        )
    if not wrap_ok:
        return ()
    raw_bindings = set()
    for position, value in enumerate(outputs):
        if not isinstance(value, ast.Name):
            continue
        constructor = _single_assignment(hook, value.id)
        if (isinstance(constructor, ast.Call)
                and isinstance(constructor.func, ast.Name)
                and constructor.func.id == raw_type):
            raw_bindings.add(targets[position].id)
    return tuple(position for position, value in enumerate(public_returns[0].args)
                 if isinstance(value, ast.Name) and value.id in raw_bindings)


def selected_family_source(reference, operation, slot, *, book):
    source = getattr(reference, "backend_source", None)
    if source is None:
        return None
    from ..common.tensors.abstraction import AbstractTensor

    module, filename, backend_name = source
    tree = ast.parse(Path(filename).read_text(encoding="utf-8"))
    backend = next((node for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == backend_name), None)
    public_callable = getattr(AbstractTensor, str(operation), None)
    if backend is None or not inspect.isfunction(public_callable):
        return None
    public_text = textwrap.dedent(inspect.getsource(public_callable))
    public = ast.parse(public_text)
    hooks = [node for node in backend.body if isinstance(node, ast.FunctionDef)
             and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                     and call.func.attr == node.name for call in ast.walk(public))]
    declarations = [node for node in backend.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "tensor_type_"
                            for target in node.targets)]
    if len(hooks) != 1 or len(declarations) != 1:
        return None
    raw_type = declarations[0].value
    if not isinstance(raw_type, ast.Name) or not any(
            isinstance(node, ast.ClassDef) and node.name == raw_type.id for node in tree.body):
        return None
    fact = (module, backend_name, raw_type.id, ast.unparse(declarations[0]),
            ast.unparse(hooks[0]), public_text,
            textwrap.dedent(inspect.getsource(AbstractTensor.ensure_tensor)))
    if slot not in raw_result_slots(fact):
        return None
    row = (reference.source_identity, str(operation), int(slot))
    incumbent = book.latest_ref(NATIVE_TENSOR_FAMILY_SOURCE, row)
    if incumbent is not None:
        if book.page(incumbent.page).cell(incumbent.row, incumbent.column) != fact:
            raise ValueError("selected native tensor family source changed")
        return incumbent
    return book.post(NATIVE_TENSOR_FAMILY_SOURCE, row, fact, stage=INGESTION,
                     provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)


def post_result_family(reference, operation, function, kernel, receiver,
                       layout, member, slot, *, book):
    declaration = selected_family_source(reference, operation, slot, book=book)
    if declaration is None:
        return None
    layout_fact = book.page(SOURCE_AGGREGATE_LAYOUT).cell(layout.row, layout.column)
    value_id = int(layout_fact.members[slot].value_id)
    output_name, formal_id = tuple(kernel.metadata["named_outputs"])[slot]
    formal = ssa_value_identity_cell(kernel, int(formal_id), book=book)
    result = ssa_value_identity_cell(function, value_id, book=book)
    receiver_cell = ssa_value_identity_cell(function, int(receiver.id), book=book)
    if any(cell is None for cell in (formal, result, receiver_cell)):
        raise ValueError("native tensor family lacks exact output or receiver identity")
    fact = (declaration, str(kernel.name), str(output_name), formal,
            layout, member, int(slot), receiver_cell)
    return book.post(
        NATIVE_TENSOR_RESULT_FAMILY, (function_scope_of(function), value_id), fact,
        stage=TENSOR_SSA_LOWERING,
        provenance=Derived((declaration, formal, result, layout, member, receiver_cell)),
        mode=Mode.CONCORD,
    )


def _receiver_class_proofs(book, receiver_cell):
    proofs = []
    pending, seen = [receiver_cell], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for parent, _stage in book.edges_into(current):
            if parent.page == SSA_VALUE and parent.row == receiver_cell.row:
                pending.append(parent)
            elif parent.page == SOURCE_VALUE_CLASS:
                if book.page(parent.page).cell(parent.row, parent.column)[0] == (
                        "src.common.tensors.abstraction.AbstractTensor"):
                    if parent not in proofs:
                        proofs.append(parent)
            elif parent.page == VALUE_TYPE_ABI:
                abi = book.page(parent.page).cell(parent.row, parent.column)
                if abi.alternatives == ("src.common.tensors.abstraction.AbstractTensor",):
                    if parent not in proofs:
                        proofs.append(parent)
    return proofs


def wrap_family(function, receiver, raw, *, book):
    cell = book.latest_ref(NATIVE_TENSOR_RESULT_FAMILY,
                           (function_scope_of(function), int(raw.id)))
    if cell is None:
        return None
    fact = book.page(cell.page).cell(cell.row, cell.column)
    receiver_cell = ssa_value_identity_cell(function, int(receiver.id), book=book)
    if receiver_cell is None:
        return None
    # Huffman's topk(-priority) is wrapped by weights.ensure_tensor. Those
    # are distinct tensors. The exact receiver's AbstractTensor class/ABI
    # selects the same native backend reference; do not demand data identity.
    proofs = _receiver_class_proofs(book, receiver_cell)
    receiver_proof = (receiver_cell if fact[7].row == receiver_cell.row
                      else next(iter(proofs), None))
    if receiver_proof is None:
        return None
    declaration = fact[0]
    if fact[6] not in raw_result_slots(book.page(declaration.page).cell(
            declaration.row, declaration.column)):
        raise ValueError("native tensor family no longer proves raw wrapping")
    return cell, receiver_proof


def family_findings(module):
    """Audit exact native member/formal and wrapper edges, including stale slots."""
    from .identity_concordance import Finding, Ref
    book = (getattr(module, "metadata", {}) or {}).get("identity_book")
    if book is None:
        return []
    found = []
    for row in book.page(NATIVE_TENSOR_RESULT_FAMILY).rows():
        cell = book.latest_ref(NATIVE_TENSOR_RESULT_FAMILY, row)
        declaration, kernel_name, name, formal, layout, member, slot, receiver = book.page(
            cell.page).cell(cell.row, cell.column)
        source_fact = book.page(declaration.page).cell(declaration.row, declaration.column)
        layout_fact = book.page(layout.page).cell(layout.row, layout.column)
        kernel = module.functions.get(kernel_name)
        sources = {source for source, _stage in book.edges_into(cell)}
        valid = (declaration.page == NATIVE_TENSOR_FAMILY_SOURCE
                 and slot == declaration.row[2] and slot in raw_result_slots(source_fact)
                 and 0 <= slot < len(layout_fact.members)
                 and int(layout_fact.members[slot].value_id) == row[1]
                 and member in layout_fact.members[slot].source_cells
                 and kernel is not None
                 and tuple(kernel.metadata.get("named_outputs", ()))[slot] == (name, formal.row[1])
                 and formal.row[0] == function_scope_of(kernel)
                 and {declaration, formal, layout, member, receiver} <= sources)
        if not valid:
            found.append(Finding("native-tensor-family", str(row[0]), row[1],
                                 "raw backend family lost its source, exact member or output formal"))
    for row in book.page(NATIVE_TENSOR_WRAP).rows():
        cell = book.latest_ref(NATIVE_TENSOR_WRAP, row)
        wrapping = book.page(cell.page).cell(cell.row, cell.column)
        if (not isinstance(wrapping, tuple) or len(wrapping) != 5
                or not all(isinstance(value, Ref) for value in wrapping)
                or wrapping[0].page != NATIVE_TENSOR_RESULT_FAMILY
                or any(value.page != SSA_VALUE for value in wrapping[1:4])
                or wrapping[4].page not in {SSA_VALUE, SOURCE_VALUE_CLASS, VALUE_TYPE_ABI}):
            found.append(Finding("native-tensor-wrap", str(row[0]), row[1],
                                 "wrapper lacks its typed source identity references"))
            continue
        family, raw, receiver, result, receiver_proof = wrapping
        try:
            family_fact = book.page(family.page).cell(family.row, family.column)
        except (KeyError, IndexError, TypeError):
            family_fact = None
        if (not isinstance(family_fact, tuple) or len(family_fact) != 8
                or not isinstance(family_fact[7], Ref)):
            found.append(Finding("native-tensor-wrap", str(row[0]), row[1],
                                 "wrapper lacks its exact result family receiver"))
            continue
        sources = {source for source, _stage in book.edges_into(cell)}
        receiver_valid = family_fact[7].row == receiver.row and receiver_proof == receiver
        if receiver_proof.page == SOURCE_VALUE_CLASS:
            receiver_valid = book.page(receiver_proof.page).cell(
                receiver_proof.row, receiver_proof.column)[0] == (
                    "src.common.tensors.abstraction.AbstractTensor")
        elif receiver_proof.page == VALUE_TYPE_ABI:
            receiver_valid = book.page(receiver_proof.page).cell(
                receiver_proof.row, receiver_proof.column).alternatives == (
                    "src.common.tensors.abstraction.AbstractTensor",)
        if receiver_proof != receiver:
            receiver_valid = receiver_valid and receiver_proof in _receiver_class_proofs(
                book, receiver)
        if (family.row != raw.row or not receiver_valid or result.row != row
                or not {family, raw, receiver, result, receiver_proof} <= sources):
            found.append(Finding("native-tensor-wrap", str(row[0]), row[1],
                                 "wrapper no longer names its exact raw value and receiver"))
        # The Huffman wrapper's Ret was correctly rewritten to raw19, but
        # cached descriptor20 still claimed separately owned data20. The
        # aggregate legalizer reads that descriptor, so the source wrapping
        # proof must agree with its exact region output storage as well.
        for name, function in module.functions.items():
            region = (function.metadata or {}).get("source_region_integral") or {}
            if (function_scope_of(function) != row[0]
                    or row[1] not in tuple(map(int, region.get("output_value_ids", ())))
                    or int(raw.row[-1]) not in {int(value.id) for value in function.args}):
                continue
            table = module.tensor_tables.get(name)
            owner = None if table is None else table.by_id(int(raw.row[-1]))
            wrapped = None if table is None else table.by_id(int(row[1]))
            if (owner is None or wrapped is None
                    or wrapped.storage != "view"
                    or wrapped.data_value_id != owner.data_value_id
                    or wrapped.alias_of != owner.tensor_id
                    or wrapped.owns_allocation
                    or wrapped.arena_id != owner.arena_id
                    or wrapped.allocation_owner != owner.allocation_owner):
                found.append(Finding(
                    "native-tensor-wrap-storage", str(name), int(row[1]),
                    "native wrapping receipt disagrees with its descriptor storage alias",
                ))
    return found
