"""Condition procedure bodies composed into the incumbent control SSA builder.

No syntax, runtime interpreter, or separate receipt writer lives here. The
source synchronization contract owns every primitive, value and CFG edge.
"""
from .concordance_declarations import SOURCE_SYNCHRONIZATION, SYNCHRONIZATION_TOKEN, BindingKind
from .concordance_declarations import SSA_SYNCHRONIZATION_PROCEDURE, SSA_BLOCK, SSA_CONTROL_EDGE
from ..transmogrifier.ssa import SSAValue
from ..transmogrifier.ssa_registry import Handler


def emit_condition_procedure(builder, block, arguments, lock_entry):
    from .synchronization import emit_ssa_projection
    contract = block.synchronization_cell
    source = builder._book().page(SOURCE_SYNCHRONIZATION).cell(contract.row, contract.column)
    kind, roles, constants = source[0], source[1], dict(source[3])
    operands = dict(zip(roles, arguments))
    wait_queue = operands["mutex"]
    tails = builder.__dict__.setdefault("_synchronization_tails", {})
    book = builder._book()
    entry = book.latest_ref(SSA_BLOCK, (builder._scope(), builder.function_name, builder.current.name))
    previous_edges = set(book.page(SSA_CONTROL_EDGE).scope_rows(builder._scope()))
    projections = []

    def fresh(dtype, *causes):
        return builder.fresh_value(dtype=dtype, transform=SYNCHRONIZATION_TOKEN,
                                   operands=(contract, *causes))

    def literal(value, dtype="int64"):
        result = fresh(dtype)
        builder.emit(Handler.Const, [], result, attributes={"value": value})
        return result

    def project(operator, phase, bindings, dtype="ssa.sync_token", *, after=None):
        selected_roles = tuple(role for role, _value in bindings)
        values = [value for _role, value in bindings]
        previous = tails.get(builder.current.name, after)
        if previous is not None:
            selected_roles = (*selected_roles, "after")
            values.append(previous)
        result = fresh(dtype, *values)
        emit_ssa_projection(builder, block, contract, operator, phase,
                            selected_roles, values, result, lock_entry=lock_entry)
        projections.append(builder.current.instrs[-1].attributes["synchronization_receipt"])
        tails[builder.current.name] = result
        return result

    def compare(operator, left, right):
        result = fresh("bool", left, right)
        builder.emit(Handler(operator), [left, right], result)
        return result

    def new(stem):
        return builder.new_block(stem, also=(contract,))

    def phi(dtype, incoming, result=None):
        result = result or fresh(dtype, *(value for _predecessor, value in incoming))
        builder.emit(Handler.Phi, [value for _pred, value in incoming], result,
                     attributes={"incoming_blocks": tuple(pred.name for pred, _value in incoming)})
        return result

    def source_result():
        return (SSAValue(int(block.result_value_id), dtype=block.result_dtype)
                if block.result_value_id is not None else fresh(block.result_dtype))

    zero, one = literal(0), literal(1)
    generation = zero
    worker = project("WorkerToken", "context", (), "opaque_ref")
    mutex = project("WaitQueueMutex", "mutex", (("wait_queue", wait_queue),), "opaque_ref")
    token = project("MutexToken", "lookup", (("mutex", mutex), ("worker", worker)), "ssa.lock_token")
    valid = compare("Gt", token, zero)
    merge = new("sync_merge")

    if kind == "condition_wait":
        false = literal(False, "bool")
        unavailable = literal(-1)
        scope = literal(constants["memory_scope"])
        release, acquire = literal(4), literal(2)
        finite = literal(constants["has_timeout"], "bool")
        timeout = operands["timeout"] if constants["has_timeout"] else zero
        failed, allocate, register = new("sync_failed"), new("sync_ready"), new("sync_ready")
        suspend, park = new("sync_ready"), new("sync_ready")
        retire, discard = new("sync_cleanup"), new("sync_cleanup")
        builder.conditional_branch(valid, allocate, failed)

        builder.current = allocate
        latch = project("LatchCreate", "allocate", (("participants", one), ("generation", generation),
                          ("execution_scope", scope)), "opaque_ref", after=token)
        builder.conditional_branch(compare("Ne", latch, zero), register, failed)

        builder.current = register
        registered = project("WaiterRegister", "register", (("wait_queue", wait_queue), ("worker", worker),
                             ("token", token), ("latch", latch)), after=latch)
        builder.conditional_branch(compare("Gt", registered, zero), suspend, discard)

        builder.current = suspend
        depth = project("MutexSuspend", "suspend", (("mutex", mutex), ("worker", worker),
                        ("token", token), ("memory_scope", scope), ("memory_semantics", release)),
                        "int64", after=registered)
        builder.conditional_branch(compare("Gt", depth, zero), park, retire)

        builder.current = park
        status = project("LatchWaitTimeout", "wait", (("latch", latch), ("generation", generation),
                         ("timeout", timeout), ("finite", finite)), "int64", after=depth)
        restored = project("MutexRestore", "restore", (("mutex", mutex), ("worker", worker),
                           ("depth", depth), ("memory_scope", scope), ("memory_semantics", acquire)),
                           "ssa.lock_token")
        builder.branch(retire)

        builder.current = retire
        outcome = phi("int64", ((suspend, unavailable), (park, status)))
        ownership = phi("ssa.lock_token", ((suspend, token), (park, restored)))
        retired = project("WaiterRetire", "retire", (("wait_queue", wait_queue), ("worker", worker),
                          ("latch", latch)), after=ownership)
        builder.branch(discard)

        builder.current = discard
        outcome = phi("int64", ((register, unavailable), (retire, outcome)))
        retirement = phi("ssa.sync_token", ((register, registered), (retire, retired)))
        project("LatchDestroy", "destroy", (("latch", latch),), after=retirement)
        success = compare("Gt", outcome, zero)
        builder.branch(merge)

        builder.current = failed
        builder.branch(merge)
        builder.current = merge
        result = phi("bool", ((failed, false), (discard, success)), source_result())
    else:
        # notify(n) and notify_all are the same actual SSA loop. Only the
        # count predicate differs; no opaque native ConditionNotify survives.
        entry_block = builder.current
        header, select, body = new("sync_notify_header"), new("sync_notify_select"), new("sync_notify_body")
        count = operands.get("n")
        if kind == "condition_notify" and count is None:
            count = literal(constants["n"])
        decrement = fresh(count.dtype, count) if count is not None else None
        builder.conditional_branch(valid, header, merge)
        builder.current = header
        if count is not None:
            remaining = phi(count.dtype, ((entry_block, count), (body, decrement)))
            builder.conditional_branch(compare("Gt", remaining, zero), select, merge)
        else:
            builder.branch(select)
        builder.current = select
        latch = project("WaiterSelect", "select", (("wait_queue", wait_queue), ("worker", worker),
                        ("token", token)), "opaque_ref", after=token)
        builder.conditional_branch(compare("Ne", latch, zero), body, merge)
        builder.current = body
        project("LatchSignal", "signal", (("latch", latch), ("generation", generation)), after=latch)
        if count is not None:
            builder.emit(Handler.Sub, [remaining, one], decrement)
        builder.branch(header)
        builder.current = merge
        result = source_result()
        builder.emit(Handler.NoneValue, [], result)

    builder._value_cell(result, contract, *arguments)
    tails[builder.current.name] = result
    if block.result_value_id is not None:
        builder._bind(int(block.result_value_id), result, BindingKind.CALL_RESULT, contract, *arguments)
    from .identity_concordance import Derived, Mode
    exit_cell = book.latest_ref(SSA_BLOCK, (builder._scope(), builder.function_name, builder.current.name))
    edges = tuple(book.latest_ref(SSA_CONTROL_EDGE, row)
                  for row in book.page(SSA_CONTROL_EDGE).scope_rows(builder._scope())
                  if row not in previous_edges and row[1] == builder.function_name)
    book.post(SSA_SYNCHRONIZATION_PROCEDURE, (contract, builder.function_name, entry),
              (kind, result.id, exit_cell, tuple(projections), edges),
              stage=builder._stage(), provenance=Derived((contract, entry, exit_cell,
                  *projections, *edges, builder._value_cell(result))), mode=Mode.CONCORD)
