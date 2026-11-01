"""C11 synchronization selection from authoritative source/SSA receipts.

The initial host mapping accepts only host-local scopes and order-only masks.
It does not reinterpret a device memory class or visibility bit as a host fence.
"""

from .concordance_declarations import (
    C_SYNCHRONIZATION_POLICY, SOURCE_SYNCHRONIZATION, SSA_SYNCHRONIZATION, EMISSION_C,
)
from .identity_concordance import Derived, Mode


NATIVE_PRIMITIVES = {
    "WorkerToken": "turing_sync_worker_token", "MutexCreate": "turing_source_mutex_create",
    "WaitQueueCreate": "turing_source_wait_queue_create", "WaitQueueMutex": "turing_sync_wait_queue_mutex",
    "MutexAcquire": "turing_sync_mutex_acquire", "MutexAcquired": "token_status",
    "MutexToken": "turing_sync_mutex_token", "MutexRelease": "turing_sync_mutex_release",
    "MutexSuspend": "turing_sync_mutex_suspend", "MutexRestore": "turing_sync_mutex_restore",
    "MutexLocked": "turing_sync_mutex_locked", "LatchCreate": "turing_source_latch_create",
    "LatchSignal": "turing_sync_latch_signal", "LatchWait": "turing_sync_latch_wait",
    "LatchWaitTimeout": "turing_sync_latch_wait", "LatchDestroy": "turing_source_latch_destroy",
    "WaiterRegister": "turing_sync_waiter_register", "WaiterSelect": "turing_sync_waiter_select",
    "WaiterRetire": "turing_sync_waiter_retire",
    "LatchArrive": "turing_sync_latch_arrive", "LatchReset": "turing_sync_latch_reset",
    "ControlBarrier": "turing_sync_control_barrier", "CompletionWait": "turing_source_thread_join",
    "BarrierCreate": "turing_source_barrier_create", "BarrierWait": "turing_sync_barrier_wait",
    "BarrierAbort": "turing_sync_barrier_abort", "BarrierReset": "turing_sync_barrier_reset",
}


def native_policy(module, owner, instruction):
    """Select the shared host primitives from exact source/SSA projections."""
    book = module.metadata["identity_book"]
    attributes = instruction.attributes
    contract, receipt = attributes.get("synchronization_contract"), attributes.get("synchronization_receipt")
    if contract is None or receipt is None:
        raise ValueError("native synchronization lacks its source/SSA contract")
    fact = book.page(SSA_SYNCHRONIZATION).cell(receipt.row, receipt.column)
    if fact != (contract, instruction.op, attributes.get("synchronization_phase"),
                tuple(instruction.arg_roles), tuple(value.id for value in instruction.args), instruction.res.id):
        raise ValueError("native synchronization operands disagree with concordance")
    source = book.page(SOURCE_SYNCHRONIZATION).cell(contract.row, contract.column)
    from ..transmogrifier.dtype_layout import layout_for
    for role, value in zip(instruction.arg_roles, instruction.args):
        if role == "after":
            continue
        layout = layout_for(value.dtype)
        if role == "timeout":
            if (layout.kind not in {"int", "uint", "float", "bool"}
                    or layout.kind == "float" and layout.byte_size > 8):
                raise ValueError("native synchronization timeout requires a basic numeric scalar")
        elif layout.kind not in {"int", "uint", "bool"} or layout.byte_size > 8:
            raise ValueError(f"native synchronization {role} requires an integer/handle ABI of at most 64 bits")
    if (instruction.op, attributes["synchronization_phase"]) not in {(op, phase) for op, phase, _ in source[2]}:
        raise ValueError("native synchronization is not a declared source projection")
    constants = dict(source[3])
    scope = constants.get("memory_scope", constants.get("execution_scope", 257))
    if scope not in {4, 256, 257}:
        raise ValueError(f"native host synchronization does not support scope {scope}")
    if instruction.op == 'ControlBarrier' and scope == 4 and constants.get('participants', 1) != 1:
        raise ValueError('C invocation-scope control barrier requires exactly one participant')
    semantics = constants.get("memory_semantics", 0) if "memory_semantics" in instruction.arg_roles else 0
    projection = next(semantics for op, phase, semantics in source[2]
                      if op == instruction.op and phase == attributes["synchronization_phase"])
    if projection is not None:
        semantics = projection
    supported = {"MutexAcquire": 2, "MutexRelease": 4, "MutexSuspend": 4, "MutexRestore": 2}.get(instruction.op, 0)
    if semantics != supported:
        raise ValueError(f"native {instruction.op} does not support memory semantics {semantics:#x}; expected {supported:#x}")
    causes = [contract, receipt]
    lifetime = "borrowed_or_operation_local"
    if instruction.op in {"MutexCreate", "WaitQueueCreate", "LatchCreate", "BarrierCreate"}:
        from .ssa_record_return_state import ssa_value_identity_cell
        function = module.functions[owner]
        identities = {instruction.res.id}
        changed = True
        while changed:
            changed = False
            for block in function.blocks.values():
                for item in block.instrs:
                    if (item.op in {"Phi", "Cast", "Select"} and item.res is not None
                            and item.res.dtype in {"opaque_ref", "ssa.sync_token"}
                            and any(value.id in identities for value in item.args)
                            and item.res.id not in identities):
                        identities.add(item.res.id)
                        causes.append(ssa_value_identity_cell(function, item.res.id, book=book))
                        changed = True
        for block in function.blocks.values():
            for item in block.instrs:
                if item.op == "Ret":
                    if any(value.id in identities for value in item.args):
                        raise ValueError("escaping synchronization object requires an explicit ownership-transfer ABI")
                    causes.extend(ssa_value_identity_cell(function, value.id, book=book) for value in item.args)
        lifetime = "activation_owned_join_before_release"
    mechanism = {"WorkerToken": "native_context_identity", "WaitQueueMutex": "native_row_reference",
                 "MutexAcquired": "token_status"}.get(instruction.op, "native_monitor")
    policy = (NATIVE_PRIMITIVES[instruction.op], mechanism, scope, semantics, "host_synchronization", lifetime)
    cell = book.post(C_SYNCHRONIZATION_POLICY, (receipt,), policy, stage=EMISSION_C,
                     provenance=Derived(tuple(dict.fromkeys((*(cell for cell in causes if cell is not None),
                        *(() if attributes.get("lock_box_entry") is None else (attributes["lock_box_entry"],)))))), mode=Mode.CONCORD)
    return cell, policy


def fence_policy(module, instruction):
    """Return a concorded C intrinsic/order pair, or refuse this exact policy."""
    book = module.metadata.get("identity_book")
    attributes = instruction.attributes or {}
    contract = attributes.get("synchronization_contract")
    receipt = attributes.get("synchronization_receipt")
    if book is None or contract is None or receipt is None:
        raise ValueError("C fence requires source and SSA synchronization receipts")
    fact = book.page(SSA_SYNCHRONIZATION).cell(receipt.row, receipt.column)
    if fact != (contract, instruction.op, attributes.get("synchronization_phase"),
                tuple(instruction.arg_roles), tuple(value.id for value in instruction.args),
                instruction.res.id):
        raise ValueError("C fence disagrees with its SSA synchronization receipt")
    kind, _roles, projections, constants, _operands, _source = book.page(
        SOURCE_SYNCHRONIZATION
    ).cell(contract.row, contract.column)
    constants = dict(constants)
    scope = constants["memory_scope"]
    phase = attributes["synchronization_phase"]
    candidates = [semantics for operator, candidate_phase, semantics in projections
                  if operator == instruction.op and candidate_phase == phase]
    if len(candidates) != 1:
        raise ValueError("C fence has no unique source projection")
    semantics = candidates[0]
    if scope not in (4, 256, 257):
        raise ValueError(f"C host fence does not support memory scope {scope}")
    orders = {2: "memory_order_acquire", 4: "memory_order_release",
              8: "memory_order_acq_rel", 16: "memory_order_seq_cst"}
    if semantics not in orders:
        raise ValueError(f"C host fence does not support memory semantics mask {semantics:#x}")
    domain = "compiler" if kind == "compiler_fence" else "memory"
    from .synchronization import SYNCHRONIZATION_SPECS
    if attributes.get("synchronization_domain") != SYNCHRONIZATION_SPECS[kind].domain:
        raise ValueError("C fence domain disagrees with source synchronization contract")
    intrinsic = "atomic_signal_fence" if domain == "compiler" else "atomic_thread_fence"
    policy = (intrinsic, orders[semantics], scope, semantics, domain)
    book.post(C_SYNCHRONIZATION_POLICY, (receipt,), policy, stage=EMISSION_C,
              provenance=Derived((contract, receipt)), mode=Mode.CONCORD)
    return policy
