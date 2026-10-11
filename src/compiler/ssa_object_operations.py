"""Object operations over opaque references, using ordinary native SSA calls.

The existing module function is the implementation and interface authority.
There is no object registry, handle-to-pointer cast, Python callback, or new
opcode here. A helper declares exact formal identities for its object, storage
context and additional operands. Its body owns the actual storage semantics.
"""
from __future__ import annotations

from enum import Enum
from typing import Mapping

from ..transmogrifier.ssa import Instr, SSAOpaqueObject, SSAValue
from .monotonic_ids import GLOBAL_MONOTONIC_IDS


class ObjectOperation(str, Enum):
    KIND = "kind"
    TYPE_MASK = "type_mask"
    LENGTH = "length"
    GET = "get"
    SET = "set"
    CONSTRUCT = "construct"
    MATERIALIZE = "materialize"
    MATERIALIZE_FROZEN = "materialize_frozen"
    COLLECT = "collect"
    UPDATE = "update"
    EXTEND = "extend"
    IDENTITY_ALL = "identity_all"
    RELEASE = "release"


_RESULT_DTYPES = {
    ObjectOperation.KIND: "int64", ObjectOperation.LENGTH: "int64",
    ObjectOperation.TYPE_MASK: "int64",
    ObjectOperation.GET: "opaque_ref", ObjectOperation.SET: "bool",
    ObjectOperation.CONSTRUCT: "opaque_ref", ObjectOperation.RELEASE: None,
    ObjectOperation.MATERIALIZE: "opaque_ref",
    ObjectOperation.MATERIALIZE_FROZEN: "opaque_ref",
    ObjectOperation.COLLECT: "opaque_ref",
    ObjectOperation.UPDATE: "bool",
    ObjectOperation.EXTEND: "bool",
    ObjectOperation.IDENTITY_ALL: "bool",
}


def _result_layout(function):
    returns = [i.args for b in function.blocks.values() for i in b.instrs
               if i.op == "Ret"]
    if not returns or any(len(values) > 1 for values in returns):
        raise ValueError("object helper requires a defined single-result or void body")
    layouts = {(values[0].dtype, tuple(values[0].shape or ())) if values else (None, ())
               for values in returns}
    if len(layouts) != 1:
        raise ValueError("object helper return paths disagree about result layout")
    return next(iter(layouts))


def _helper_receipt(module, function_name: str, operation: ObjectOperation | str,
                          *, storage_identity: str, reference_formal_id: int | None,
                          context_formal_ids=(), operand_formal_ids=(), kind_vocabulary=()):
    """Read an object interface from an incumbent, compiled SSA function.

    All roles use exact callee value IDs, never signature positions or guessed
    names. This API cannot bless an absent external symbol or unknown layout.
    Construct has no input object; release has no result. Ownership and effects
    stay in the helper's real instructions and existing interprocedural analysis.
    """
    operation = ObjectOperation(operation)
    if not isinstance(storage_identity, str) or not storage_identity:
        raise ValueError("object helper requires an admitted storage identity")
    function = module.functions[function_name]
    from .ssa_object_context import validate_context_function
    validate_context_function(module, function_name)
    if function.metadata.get("native_interface"):
        raise ValueError("object helper declaration currently requires its native SSA body")
    context = tuple(map(int, context_formal_ids))
    operands = tuple(map(int, operand_formal_ids))
    reference = None if reference_formal_id is None else int(reference_formal_id)
    if (reference is None) != (operation is ObjectOperation.CONSTRUCT):
        raise ValueError("only object construction has no reference formal")
    roles = (*(() if reference is None else (reference,)), *context, *operands)
    formals = {int(value.id): value for value in function.args}
    if len(formals) != len(function.args) or len(set(roles)) != len(roles) or set(roles) != set(formals):
        raise ValueError("object helper roles must cover each exact formal once")
    if reference is not None:
        SSAOpaqueObject(formals[reference], storage_identity)
    if any(value.dtype in {None, "unknown"} for value in function.args):
        raise ValueError("object helper has an unresolved formal layout")
    expected_result = (_RESULT_DTYPES[operation], ())
    if _result_layout(function) != expected_result:
        raise ValueError(f"{operation.value} helper requires result {expected_result}")
    vocabulary = tuple(kind_vocabulary)
    if vocabulary and (operation is not ObjectOperation.KIND or
                       any(not isinstance(token, str) or not token for token in vocabulary)
                       or len(set(vocabulary)) != len(vocabulary)):
        raise ValueError("kind vocabulary requires unique declared tokens on a kind helper")
    receipt = {
        "operation": operation.value, "storage_identity": storage_identity,
        "reference_formal_id": reference, "context_formal_ids": context,
        "operand_formal_ids": operands, "result_layout": expected_result,
        "kind_vocabulary": vocabulary,
        "result_value_ids": tuple(sorted({int(i.args[0].id) for b in function.blocks.values()
                                           for i in b.instrs if i.op == "Ret" and i.args})),
    }
    if operation is ObjectOperation.TYPE_MASK:
        vocabulary = tuple(function.metadata.get('object_context', {}).get('type_vocabulary', ()))
        if not vocabulary:
            raise ValueError('type evidence helper has no declared vocabulary')
        receipt['type_vocabulary'] = vocabulary
    return receipt


def declare_object_helper(module, function_name: str, operation: ObjectOperation | str,
                          *, storage_identity: str, reference_formal_id: int | None,
                          context_formal_ids=(), operand_formal_ids=(), kind_vocabulary=()):
    """Publish one checked interface on the existing native function."""
    receipt = _helper_receipt(module, function_name, operation,
        storage_identity=storage_identity, reference_formal_id=reference_formal_id,
        context_formal_ids=context_formal_ids, operand_formal_ids=operand_formal_ids,
        kind_vocabulary=kind_vocabulary)
    function = module.functions[function_name]
    incumbent = function.metadata.get("opaque_object_helper")
    if incumbent is not None and incumbent != receipt:
        raise ValueError("object helper already has a different interface")
    function.metadata["opaque_object_helper"] = receipt
    # The native API's result is the helper's own Ret value, not a caller's
    # projection ID. Structural source getters can leave named_outputs empty;
    # without this established return_value contract Fortran saw kind's
    # six-input call as an outputless procedure and refused its scalar result.
    returned = {int(i.args[0].id): i.args[0] for b in function.blocks.values()
                for i in b.instrs if i.op == "Ret" and i.args}
    if len(returned) == 1 and function.metadata.get("return_value") is None:
        function.metadata["return_value"] = next(iter(returned.values()))
    return receipt


def object_call(module, function_name: str, object_: SSAOpaqueObject | None,
                *, context: Mapping[int, SSAValue], operands: Mapping[int, SSAValue],
                result: SSAValue | None, source_span=None) -> Instr:
    """Lower an object operation to the same Call consumed by all backends."""
    function = module.functions[function_name]
    receipt = function.metadata["opaque_object_helper"]
    reference = receipt["reference_formal_id"]
    if reference is None:
        if object_ is not None:
            raise ValueError("construct does not consume an existing object")
    elif object_ is None or object_.storage_identity != receipt["storage_identity"]:
        raise ValueError("object and helper disagree about storage identity")
    if set(context) != set(receipt["context_formal_ids"]) or set(operands) != set(receipt["operand_formal_ids"]):
        raise ValueError("object call must bind the exact declared context and operand identities")
    bindings = {**context, **operands}
    if reference is not None:
        bindings[reference] = object_.reference
    args = [bindings[int(formal.id)] for formal in function.args]
    instruction = Instr("Call", args, result, source_span=source_span, attributes={
        "callee": function_name, "callee_input_ids": tuple(int(v.id) for v in function.args),
        "source_linked": True, "opaque_object_operation": dict(receipt),
        "opaque_object_bindings": tuple((int(formal.id), int(actual.id))
                                        for formal, actual in zip(function.args, args)),
        "native_result_contract": (
            ((receipt["result_value_ids"][0], *receipt["result_layout"]),)
            if len(receipt["result_value_ids"]) == 1 else ()
        ),
    })
    validate_object_call(module, instruction)
    return instruction


def validate_object_call(module, instruction: Instr, *, _helpers=None) -> None:
    """Check the operation against its live implementation, not a stale copy."""
    receipt = instruction.attributes.get("opaque_object_operation")
    if receipt is None:
        return
    if instruction.op != "Call":
        raise ValueError("opaque object operation must use the existing Call opcode")
    function = module.functions.get(instruction.attributes.get("callee"))
    if function is None or function.metadata.get("opaque_object_helper") != receipt:
        raise ValueError("opaque object call has no matching live helper interface")
    # Revalidate the live body: changing its signature/results cannot silently
    # leave an old call receipt apparently valid.
    live = None if _helpers is None else _helpers.get(function.name)
    if live is None:
        live = _helper_receipt(module, function.name, receipt["operation"],
            storage_identity=receipt["storage_identity"],
            reference_formal_id=receipt["reference_formal_id"],
            context_formal_ids=receipt["context_formal_ids"],
            operand_formal_ids=receipt["operand_formal_ids"], kind_vocabulary=receipt["kind_vocabulary"])
        if _helpers is not None:
            _helpers[function.name] = live
    if live != receipt:
        raise ValueError("opaque object call disagrees with its live helper layout")
    expected_returns = (((receipt["result_value_ids"][0], *receipt["result_layout"]),)
                        if len(receipt["result_value_ids"]) == 1 else ())
    if instruction.attributes.get("native_result_contract") != expected_returns:
        raise ValueError("opaque object call has stale native result identities")
    expected_ids = tuple(int(v.id) for v in function.args)
    if instruction.attributes.get("callee_input_ids") != expected_ids or len(instruction.args) != len(function.args):
        raise ValueError("opaque object call disagrees with exact callee input identities")
    if instruction.attributes.get("opaque_object_bindings") != tuple(
            (int(formal.id), int(actual.id)) for formal, actual in zip(function.args, instruction.args)):
        raise ValueError("opaque object call has stale context/reference binding identities")
    for formal, actual in zip(function.args, instruction.args):
        if (formal.dtype, tuple(formal.shape or ())) != (actual.dtype, tuple(actual.shape or ())):
            raise ValueError(f"opaque object argument {formal.id} has incompatible physical layout")
    actual_result = ((instruction.res.dtype, tuple(instruction.res.shape or ()))
                     if instruction.res is not None else (None, ()))
    if actual_result != receipt["result_layout"]:
        raise ValueError("opaque object call has incompatible result layout")


def validate_object_operations(module, function_name: str | None = None) -> None:
    # Validate only the emitted call closure. Each live helper body is checked
    # once per read-only emission boundary, not once per caller occurrence.
    # This avoids calls x helper-body work and unrelated module-body scans.
    pending = list(module.functions) if function_name is None else [function_name]
    visited = set()
    helpers = {}
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        function = module.functions[name]
        from .ssa_object_context import (validate_context_function,
            validate_source_object_instruction, source_object_definitions, validate_native_object_requirements)
        validate_context_function(module, name)
        validate_native_object_requirements(module, name)
        source_definitions = (source_object_definitions(function)
            if function.metadata.get('native_object_context') is not None else {})
        from .ssa_heap_allocation import validate_heap_context_projection
        for formal in function.args:
            validate_heap_context_projection(module, name, formal)
        for block in function.blocks.values():
            for instruction in block.instrs:
                validate_object_call(module, instruction, _helpers=helpers)
                validate_source_object_instruction(module, name, instruction,
                    _definitions=source_definitions, _helpers=helpers)
                callee = instruction.attributes.get("callee") if instruction.op == "Call" else None
                if callee in module.functions and callee not in visited:
                    pending.append(callee)


def object_identity(object_: SSAOpaqueObject, result: SSAValue) -> Instr:
    if result.dtype != "int64" or result.shape:
        raise ValueError("object identity requires a scalar int64 result")
    return Instr("Cast", [object_.reference], result, attributes={
        "binding": "opaque_object_identity", "storage_identity": object_.storage_identity,
    })


def object_kind_matches(module, helper_name: str, object_: SSAOpaqueObject, *,
                        context: Mapping[int, SSAValue], accepted: tuple[str, ...],
                        kind_result: SSAValue, result: SSAValue) -> list[Instr]:
    """A native test against the owner's declared one-based kind vocabulary.

    This is closed-domain kind membership, not an implementation of arbitrary
    Python metaclasses/subclass hooks. A future isinstance lowering must admit
    that semantics explicitly instead of silently treating kind as full MRO.
    """
    receipt = module.functions[helper_name].metadata["opaque_object_helper"]
    vocabulary = receipt["kind_vocabulary"]
    if not vocabulary or not accepted or any(token not in vocabulary for token in accepted):
        raise ValueError("kind test requires tokens from the helper's declared vocabulary")
    if result.dtype != "bool" or result.shape:
        raise ValueError("kind test requires a scalar bool result")
    instructions = [object_call(module, helper_name, object_, context=context,
                                operands={}, result=kind_result)]
    predicates = []
    for token in dict.fromkeys(accepted):
        constant = SSAValue(GLOBAL_MONOTONIC_IDS.mint(), 'int64')
        compared = result if len(set(accepted)) == 1 else SSAValue(GLOBAL_MONOTONIC_IDS.mint(), 'bool')
        instructions.extend((Instr('Const', [], constant, attributes={
            'value': vocabulary.index(token) + 1, 'program_abi_vocabulary_token': token,
        }), Instr('Eq', [kind_result, constant], compared)))
        predicates.append(compared)
    combined = predicates[0]
    for index, compared in enumerate(predicates[1:], 1):
        merged = result if index == len(predicates) - 1 else SSAValue(GLOBAL_MONOTONIC_IDS.mint(), 'bool')
        instructions.append(Instr('Or', [combined, compared], merged))
        combined = merged
    return instructions


def object_same(left: SSAOpaqueObject, right: SSAOpaqueObject, result: SSAValue) -> Instr:
    if left.storage_identity != right.storage_identity:
        raise ValueError("identity comparison requires the same admitted storage domain")
    if result.dtype != "bool" or result.shape:
        raise ValueError("object identity comparison requires a scalar bool result")
    return Instr("Eq", [left.reference, right.reference], result, attributes={
        "binding": "opaque_object_same", "storage_identity": left.storage_identity,
    })
