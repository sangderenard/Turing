"""Source declarations and SSA contracts for synchronization.

These intrinsics are consumed by source ingestion. They do not run a Python
scheduler or emulate native synchronization. All scopes are explicit; a latch
is control state, and memory ordering is requested separately.
"""

from dataclasses import dataclass
from enum import IntEnum


class Scope(IntEnum):
    CROSS_DEVICE = 0
    DEVICE = 1
    WORKGROUP = 2
    SUBGROUP = 3
    INVOCATION = 4
    QUEUE_FAMILY = 5
    SHADER_CALL = 6
    THREAD_REGION = 256
    OBJECT = 257


class MemorySemantics:
    # Plain integer masks use Python's ordinary bitwise graph operators;
    # no enum __or__ implementation or runtime flag object is required.
    NONE = 0
    ACQUIRE = 0x2
    RELEASE = 0x4
    ACQUIRE_RELEASE = 0x8
    SEQUENTIALLY_CONSISTENT = 0x10
    UNIFORM_MEMORY = 0x40
    SUBGROUP_MEMORY = 0x80
    WORKGROUP_MEMORY = 0x100
    CROSS_WORKGROUP_MEMORY = 0x200
    ATOMIC_COUNTER_MEMORY = 0x400
    IMAGE_MEMORY = 0x800
    OUTPUT_MEMORY = 0x1000
    MAKE_AVAILABLE = 0x2000
    MAKE_VISIBLE = 0x4000
    VOLATILE = 0x8000


def concord_host_barrier_policy():
    """One declared host source policy shared by graph and SSA projections."""
    from .concordance_declarations import SOURCE_THREAD_BARRIER_POLICY, POLICY_DECLARATION, CONTROL_PROGRAM_BUILD
    from .identity_concordance import current_identity_book, Novel, Mode
    return current_identity_book().post(SOURCE_THREAD_BARRIER_POLICY, ('threading.Barrier.wait',),
        ('barrier_participants', 'barrier_participant_effects', MemorySemantics.ACQUIRE_RELEASE),
        stage=CONTROL_PROGRAM_BUILD, provenance=Novel(POLICY_DECLARATION, ()), mode=Mode.CONCORD)


def _requires_compilation():
    raise RuntimeError("synchronization intrinsic requires Turing source compilation")


def memory_barrier(memory_scope, memory_semantics, storage):
    """Order selected storage effects without making workers rendezvous."""
    _requires_compilation()


memory_fence = memory_barrier


def compiler_fence(memory_semantics, storage):
    """Constrain compiler memory reordering in the invocation scope."""
    _requires_compilation()


def control_barrier(latch, execution_scope, generation, participants):
    """Rendezvous in this latch generation, with no implicit memory fence."""
    _requires_compilation()


def control_memory_barrier(latch, execution_scope, memory_scope, memory_semantics,
                           generation, participants, storage):
    """Release before rendezvous and acquire after it, as explicitly requested."""
    _requires_compilation()


def latch_create(participants, generation, execution_scope):
    """Create generation-scoped control state; zero participants is ready."""
    _requires_compilation()


def latch_arrive(latch, generation, count):
    """Subtract count arrivals; zero remaining releases this generation."""
    _requires_compilation()


def latch_wait(latch, generation):
    """Wait for this generation's release without requesting a memory fence."""
    _requires_compilation()


def latch_signal(latch, generation):
    """Release this generation explicitly, independently of its arrival count."""
    _requires_compilation()


def latch_reset(latch, previous_generation, generation, participants):
    """Rearm a completed generation; old-generation waiters retain their release."""
    _requires_compilation()


def completion_wait(handle, memory_scope, memory_semantics, storage):
    """Wait for one handle's completion, then apply its explicit memory fence."""
    _requires_compilation()


def mutex_create(execution_scope, reentrant):
    """Create a mutex handle serviced by the SSA lock box."""
    _requires_compilation()


def mutex_acquire(mutex, blocking, timeout):
    """Acquire a lock token; report success without inventing shared-data locks."""
    _requires_compilation()


def mutex_release(mutex):
    """Return the held token; enforce the entry's ownership/reentrancy policy."""
    _requires_compilation()


def mutex_locked(mutex):
    """Read the mutex's held state from its lock-box row."""
    _requires_compilation()


@dataclass(frozen=True)
class SynchronizationSpec:
    roles: tuple
    operator: str
    progress: bool = False
    domain: str = "control"


SYNCHRONIZATION_SPECS = {
    # Python's default Condition owns an RLock. Its source identity is retained
    # while allocation uses the incumbent lock-box primitive and row layout.
    "barrier_create": SynchronizationSpec(("participants", "action", "timeout"), "BarrierCreate"),
    "barrier_wait": SynchronizationSpec(("barrier", "timeout"), "BarrierWait", True, 'combined'),
    "barrier_abort": SynchronizationSpec(("barrier",), "BarrierAbort"),
    "barrier_reset": SynchronizationSpec(("barrier",), "BarrierReset"),
    "condition_create": SynchronizationSpec(("lock",), "MutexCreate"),
    "condition_acquire": SynchronizationSpec(("mutex", "blocking", "timeout"), "MutexAcquire", True),
    "condition_release": SynchronizationSpec(("mutex",), "MutexRelease"),
    "condition_wait": SynchronizationSpec(("mutex", "timeout"), "LatchWaitTimeout", True),
    "condition_notify": SynchronizationSpec(("mutex", "n"), "LatchSignal"),
    "condition_notify_all": SynchronizationSpec(("mutex",), "LatchSignal"),
    "memory_barrier": SynchronizationSpec(("memory_scope", "memory_semantics", "storage"), "MemoryBarrier", domain="memory"),
    "compiler_fence": SynchronizationSpec(("memory_semantics", "storage"), "MemoryBarrier", domain="compiler"),
    "control_barrier": SynchronizationSpec(("latch", "execution_scope", "generation", "participants"), "ControlBarrier", True),
    "control_memory_barrier": SynchronizationSpec(("latch", "execution_scope", "memory_scope", "memory_semantics", "generation", "participants", "storage"), "ControlBarrier", True, "combined"),
    "latch_create": SynchronizationSpec(("participants", "generation", "execution_scope"), "LatchCreate"),
    "latch_arrive": SynchronizationSpec(("latch", "generation", "count"), "LatchArrive"),
    "latch_wait": SynchronizationSpec(("latch", "generation"), "LatchWait", True),
    "latch_signal": SynchronizationSpec(("latch", "generation"), "LatchSignal"),
    "latch_reset": SynchronizationSpec(("latch", "previous_generation", "generation", "participants"), "LatchReset"),
    "completion_wait": SynchronizationSpec(("handle", "memory_scope", "memory_semantics", "storage"), "CompletionWait", True, "completion"),
    "mutex_create": SynchronizationSpec(("execution_scope", "reentrant"), "MutexCreate"),
    "mutex_acquire": SynchronizationSpec(("mutex", "blocking", "timeout"), "MutexAcquire", True),
    "mutex_release": SynchronizationSpec(("mutex",), "MutexRelease"),
    "mutex_locked": SynchronizationSpec(("mutex",), "MutexLocked"),
}


def synchronization_identity(identity):
    prefix = __name__ + "."
    return identity[len(prefix):] if isinstance(identity, str) and identity.startswith(prefix) and identity[len(prefix):] in SYNCHRONIZATION_SPECS else None


def concord_source_synchronization(graph, node, operation, parent_ids, keyword_names):
    """Bind actual operands and validate scope/order before SSA emission."""
    from .concordance_declarations import SOURCE_SYNCHRONIZATION, CONTROL_PROGRAM_BUILD
    from .identity_concordance import Derived, Mode, current_identity_book, _canonical_identity_base
    from .concorded_literals import literal, MISSING as _STATIC_READ_MISSING
    from ..common.tensors.topological_reducer import node_identity_cell
    from .process_graph_facts import operation as source_operation

    process_fact, process_cell = source_operation(graph, int(node))
    if process_fact is None or process_fact[0] != operation:
        raise ValueError("synchronization projection disagrees with its source process operation")

    kind = operation.removeprefix("sync_")
    spec = SYNCHRONIZATION_SPECS[kind]
    positional = len(parent_ids) - len(keyword_names)
    roles = (*spec.roles[:positional], *keyword_names)
    if positional < 0 or positional > len(spec.roles) or len(roles) != len(parent_ids):
        raise ValueError(f"{kind}: unsupported operands; expected {spec.roles}")
    defaulted_acquire = (kind in {"condition_acquire", "condition_wait", "condition_notify", "condition_create"}
                         and ("mutex" in roles or kind == "condition_create")
                         and len(set(roles)) == len(roles)
                         and set(roles) <= set(spec.roles))
    defaulted_barrier = (kind in {'barrier_create', 'barrier_wait'} and len(set(roles)) == len(roles)
                         and set(roles) <= set(spec.roles)
                         and ('participants' if kind == 'barrier_create' else 'barrier') in roles)
    if not defaulted_acquire and not defaulted_barrier and (len(roles) != len(spec.roles) or set(roles) != set(spec.roles)):
        raise ValueError(f"{kind}: explicit operands required: {spec.roles}")
    constants = {}
    policy_causes = ()
    if kind.startswith('barrier_'):
        policy_causes = (concord_host_barrier_policy(),)
        constants['execution_scope'] = int(Scope.OBJECT)
    if kind in {'barrier_create', 'barrier_wait'}:
        constants.update(execution_scope=int(Scope.OBJECT), has_timeout='timeout' in roles)
        if kind == 'barrier_wait':
            constants.update(memory_scope=int(Scope.OBJECT), memory_semantics=MemorySemantics.ACQUIRE_RELEASE)
    if kind == "condition_create":
        constants.update(execution_scope=int(Scope.OBJECT), reentrant=True, owns_mutex="lock" not in roles)
    if defaulted_acquire and kind == "condition_acquire":
        constants.update((role, value) for role, value in (("blocking", True), ("timeout", -1))
                         if role not in roles)
    if kind == "condition_notify" and "n" not in roles:
        constants["n"] = 1
    if kind == "condition_wait":
        constants["has_timeout"] = "timeout" in roles
    literal_causes = []
    def static_value(parent):
        value, cells = literal(graph, int(parent))
        literal_causes.extend(cells)
        return value

    for role, parent in zip(roles, parent_ids):
        value = static_value(parent)
        if kind.startswith('barrier_') and role == 'action' and value is not None:
            raise ValueError('Barrier action requires a compiled callback contract')
        if kind in {'condition_wait', 'barrier_create', 'barrier_wait'} and role == "timeout" and value is None:
            constants["has_timeout"] = False
        if kind == "condition_create" and role == "lock" and value is None:
            constants["owns_mutex"] = True
        if role in {"execution_scope", "memory_scope", "memory_semantics"}:
            if value is _STATIC_READ_MISSING or not isinstance(value, int):
                raise ValueError(f"{kind}: {role} must be an explicit integer constant")
            value = int(value)
            constants[role] = value
            if role.endswith("scope") and value not in set(Scope):
                raise ValueError(f"{kind}: unknown {role} {value}")
        elif role in {"generation", "previous_generation", "participants", "count"} and value is not _STATIC_READ_MISSING:
            if not isinstance(value, int):
                raise ValueError(f"{kind}: {role} must be an integer")
            constants[role] = int(value)
            if value < 0 or role == "count" and value == 0 or role == "participants" and kind in {"control_barrier", "control_memory_barrier"} and value == 0:
                raise ValueError(f"{kind}: invalid {role} {value}")
        elif role in {"blocking", "reentrant"}:
            if role == "blocking" and value is _STATIC_READ_MISSING:
                continue
            if value is _STATIC_READ_MISSING or value not in (False, True, 0, 1):
                raise ValueError(f"{kind}: {role} must be an explicit boolean")
            constants[role] = bool(value)
    if "previous_generation" in constants and "generation" in constants and constants["generation"] <= constants["previous_generation"]:
        raise ValueError("latch_reset: generation must advance")
    if kind in {"mutex_acquire", "mutex_release", "condition_acquire", "condition_release"}:
        constants["memory_scope"] = int(Scope.OBJECT)
        constants["memory_semantics"] = 2 if kind.endswith("acquire") else 4
    semantics = constants.get("memory_semantics")
    if semantics is not None:
        order = semantics & 0x1e
        allowed = sum(value for value in vars(MemorySemantics).values() if type(value) is int)
        if semantics < 0 or semantics & ~allowed or order not in {2, 4, 8, 16}:
            raise ValueError(f"{kind}: memory semantics require one valid acquire/release/acq_rel/seq_cst ordering")
        if semantics & 0x2000 and order not in {4, 8, 16}:
            raise ValueError(f"{kind}: MakeAvailable requires release semantics")
        if semantics & 0x4000 and order not in {2, 8, 16}:
            raise ValueError(f"{kind}: MakeVisible requires acquire semantics")
    projections = []
    if kind == 'barrier_wait':
        projections.extend((('MemoryBarrier', 'release', 4), ('BarrierWait', 'rendezvous', None), ('MemoryBarrier', 'acquire', 2)))
    elif kind == "condition_create":
        if not constants["owns_mutex"]:
            constants.pop("reentrant")
        projections.append(("MutexCreate" if constants["owns_mutex"] else "WaitQueueCreate", "control", None))
    elif kind == "condition_wait":
        constants.update(memory_scope=int(Scope.OBJECT), memory_semantics=MemorySemantics.ACQUIRE_RELEASE)
        projections.extend((operator, phase, 4 if operator == "MutexSuspend" else 2 if operator == "MutexRestore" else None) for operator, phase in (
            ("WorkerToken", "context"), ("WaitQueueMutex", "mutex"), ("MutexToken", "lookup"),
            ("LatchCreate", "allocate"), ("WaiterRegister", "register"),
            ("MutexSuspend", "suspend"), ("LatchWaitTimeout", "wait"),
            ("MutexRestore", "restore"), ("WaiterRetire", "retire"), ("LatchDestroy", "destroy")))
    elif kind in {"condition_notify", "condition_notify_all"}:
        projections.extend((operator, phase, None) for operator, phase in (
            ("WorkerToken", "context"), ("WaitQueueMutex", "mutex"), ("MutexToken", "lookup"),
            ("WaiterSelect", "select"), ("LatchSignal", "signal")))
    elif kind in {"mutex_acquire", "condition_acquire"}:
        projections.extend((("WorkerToken", "context", None), ("MutexAcquire", "acquire", None), ("MutexAcquired", "status", None)))
        if kind == "condition_acquire":
            projections.insert(1, ("WaitQueueMutex", "mutex", None))
    elif kind in {"mutex_release", "condition_release"}:
        projections.extend((("WorkerToken", "context", None), ("MutexToken", "lookup", None), ("MutexRelease", "release", None)))
        if kind == "condition_release":
            projections.insert(1, ("WaitQueueMutex", "mutex", None))
    elif spec.domain in {"combined", "completion"}:
        order, flags = semantics & 0x1e, semantics & ~0x1e
        if order in {4, 8, 16}:
            projections.append(("MemoryBarrier", "release", (flags & ~0x4000) | (16 if order == 16 else 4)))
        projections.append(("ControlBarrier", "rendezvous", None) if spec.domain == "combined"
                           else ("CompletionWait", "completion", None))
        if order in {2, 8, 16}:
            projections.append(("MemoryBarrier", "acquire", (flags & ~0x2000) | (16 if order == 16 else 2)))
    else:
        projections.append((spec.operator, spec.domain, semantics))
    if spec.domain == "compiler":
        constants["memory_scope"] = int(Scope.INVOCATION)
    book = current_identity_book()
    actual_cells = tuple(node_identity_cell(graph, parent) for parent in parent_ids)
    actual_source = node_identity_cell(graph, node)
    # Specialization forks are views of these source identities. Do not
    # redeclare a different contract merely because the view's scope changed.
    cells = tuple(_canonical_identity_base(book, cell) for cell in actual_cells)
    source = _canonical_identity_base(book, actual_source)
    scope = graph.G.graph["lexical_read_scope"]
    return book.post(
        SOURCE_SYNCHRONIZATION, (scope, node),
        (kind, tuple(roles), tuple(projections), tuple(sorted(constants.items())), cells, source),
        stage=CONTROL_PROGRAM_BUILD, provenance=Derived((process_cell, source, *cells, actual_source, *actual_cells, *literal_causes, *policy_causes)), mode=Mode.CONCORD,
    )


def emit_ssa_synchronization(builder, block, arguments):
    """Emit a source contract through the incumbent ControlProgram SSA builder."""
    from .concordance_declarations import (
        SOURCE_SYNCHRONIZATION, SSA_SYNCHRONIZATION, SYNCHRONIZATION_TOKEN,
        BindingKind,
    )
    from .identity_concordance import Derived, Mode
    from ..transmogrifier.ssa import SSAValue
    from ..transmogrifier.ssa_registry import Handler

    book = builder._book()
    contract = block.synchronization_cell
    kind, roles, projections, constants, _cells, _source = book.page(
        SOURCE_SYNCHRONIZATION
    ).cell(contract.row, contract.column)
    constants = dict(constants)
    operands = dict(zip(roles, arguments))
    lock_entry = None
    if kind.startswith("mutex_") or kind.startswith("condition_"):
        from .concordance_declarations import SOURCE_MUTEX_BINDING
        source_cell = _source if kind in {"mutex_create", "condition_create"} else _cells[roles.index("mutex")]
        lock_entry = book.page(SOURCE_MUTEX_BINDING).latest((contract.row[0], int(source_cell.row[1])))
        if lock_entry is None:
            raise ValueError("mutex operation has no concorded lock-box binding")
    if kind in {"condition_wait", "condition_notify", "condition_notify_all"}:
        from .condition_ssa import emit_condition_procedure
        return emit_condition_procedure(builder, block, arguments, lock_entry)
    tails = builder.__dict__.setdefault("_synchronization_tails", {})
    previous = tails.get(builder.current.name)
    worker_context = None
    mutex_context = None
    source_result = None

    def literal(role, value):
        if role in operands and constants.get(role) == value:
            return operands[role]
        result = builder.fresh_value(dtype="bool" if type(value) is bool else "int64", transform=SYNCHRONIZATION_TOKEN,
                                     operands=(contract,))
        builder.emit(Handler.Const, [], result, attributes={"value": value, "synchronization_contract": contract})
        return result

    for index, (operator, phase, semantics) in enumerate(projections):
        if operator in {'BarrierCreate', 'BarrierWait'}:
            selected_roles = (('participants', 'timeout', 'finite', 'execution_scope') if operator == 'BarrierCreate'
                              else ('barrier', 'timeout', 'finite'))
            selected = [operands['participants' if operator == 'BarrierCreate' else 'barrier'],
                        operands['timeout'] if constants['has_timeout'] else literal('timeout_default', 0),
                        literal('finite', constants['has_timeout'])]
            if operator == 'BarrierCreate':
                selected.append(literal('execution_scope', constants['execution_scope']))
        elif operator == "MutexCreate" and kind == "condition_create":
            selected_roles = ("execution_scope", "reentrant")
            selected = [literal(role, constants[role]) for role in selected_roles]
        elif operator == "WaitQueueCreate":
            selected_roles = ("mutex", "execution_scope")
            selected = [operands["lock"], literal("execution_scope", constants["execution_scope"])]
        elif operator == "WaitQueueMutex":
            selected_roles, selected = ("wait_queue",), [operands["mutex"]]
        elif operator == "WorkerToken":
            selected_roles, selected = (), []
        elif operator == "MutexAcquired":
            selected_roles, selected = ("acquisition",), [previous]
        elif operator == "MutexToken":
            selected_roles, selected = ("mutex", "worker"), [mutex_context or operands["mutex"], worker_context]
        elif operator == "MutexRelease":
            selected_roles = ("mutex", "token", "worker", "memory_scope", "memory_semantics")
            selected = [mutex_context or operands["mutex"], previous, worker_context, literal("memory_scope", constants["memory_scope"]),
                        literal("memory_semantics", constants["memory_semantics"])]
        elif operator == "MutexAcquire":
            selected_roles = ("mutex", "worker", "blocking", "timeout", "memory_scope", "memory_semantics")
            selected = [mutex_context or operands["mutex"], worker_context,
                        *(operands[role] if role in operands else literal(role, constants[role])
                          for role in ("blocking", "timeout"))]
            selected.extend((literal("memory_scope", constants["memory_scope"]),
                             literal("memory_semantics", constants["memory_semantics"])))
        elif operator == "MemoryBarrier":
            selected_roles = ("memory_scope", "memory_semantics", "storage")
            selected = [literal("memory_scope", constants["memory_scope"]),
                        literal("memory_semantics", semantics), operands['barrier'] if kind == 'barrier_wait' else operands["storage"]]
        elif operator == "ControlBarrier":
            selected_roles = ("latch", "execution_scope", "generation", "participants")
            selected = [operands[role] for role in selected_roles]
        elif operator == "CompletionWait":
            selected_roles = ("handle",)
            selected = [operands["handle"]]
        else:
            selected_roles = roles
            selected = list(arguments)
        if previous is not None:
            selected.append(previous)
            selected_roles = (*selected_roles, "after")
        last = index == len(projections) - 1
        source_projection = (operator == 'BarrierWait') if kind == 'barrier_wait' else last
        result = (SSAValue(int(block.result_value_id), dtype=block.result_dtype)
                  if source_projection and block.result_value_id is not None else
                  builder.fresh_value(dtype=block.result_dtype if source_projection and kind.startswith(("condition_", 'barrier_')) else "opaque_ref" if operator in {"WorkerToken", "WaitQueueMutex"} else "ssa.lock_token" if operator in {"MutexAcquire", "MutexToken"} else "ssa.sync_token", transform=SYNCHRONIZATION_TOKEN,
                                      operands=(contract, *selected)))
        emit_ssa_projection(builder, block, contract, operator, phase,
                            selected_roles, selected, result, lock_entry=lock_entry)
        if operator == "WorkerToken":
            worker_context = result
        if operator == "WaitQueueMutex":
            mutex_context = result
        if source_projection:
            source_result = result
        previous = result
    tails[builder.current.name] = previous
    if block.result_value_id is not None:
        builder._bind(int(block.result_value_id), source_result, BindingKind.CALL_RESULT, contract, *arguments)


def emit_ssa_projection(builder, block, contract, operator, phase,
                        selected_roles, selected, result, *, lock_entry=None):
    """One incumbent receipt writer for simple and compound sync procedures."""
    from .concordance_declarations import SOURCE_SYNCHRONIZATION, SSA_SYNCHRONIZATION
    from .identity_concordance import Derived, Mode
    from ..transmogrifier.ssa_registry import Handler
    book = builder._book()
    source = book.page(SOURCE_SYNCHRONIZATION).cell(contract.row, contract.column)
    kind, constants = source[0], dict(source[3])
    attributes = {
        "synchronization_contract": contract,
        "synchronization_kind": kind,
        "synchronization_phase": phase,
        "synchronization_domain": SYNCHRONIZATION_SPECS[kind].domain,
        "effectful": True, "ordered_effect": True,
        "requires_concurrent_progress": operator in {"ControlBarrier", "BarrierWait", "LatchWait", "LatchWaitTimeout", "MutexRestore", "CompletionWait"}
            or operator == "MutexAcquire" and constants.get("blocking", True),
        "source_callsite_id": int(block.callsite_id),
        "source_effect_node_id": int(block.callsite_id),
        "deployment_owner": "dispatcher",
        "required_capability": (
            "compiler_ordering" if SYNCHRONIZATION_SPECS[kind].domain == "compiler" else
            "memory_ordering" if operator == "MemoryBarrier" else "communicating_tasks"
        ),
        "atomic_control_state": operator != "MemoryBarrier",
        **({"lock_box_entry": lock_entry, "token_lock": True,
            "runtime_mutex_key": "mutex_handle"} if lock_entry is not None else {}),
    }
    builder.emit(Handler(operator), selected, result, attributes=attributes)
    instruction = builder.current.instrs[-1]
    instruction.arg_roles = list(selected_roles)
    site = len(builder.current.instrs) - 1
    receipt = book.post(
        SSA_SYNCHRONIZATION, (builder._scope(), builder.current.name, site),
        (contract, operator, phase, tuple(selected_roles), tuple(value.id for value in selected), result.id),
        stage=builder._stage(),
        provenance=Derived((contract, *builder._cells(*selected, result))), mode=Mode.CONCORD,
    )
    instruction.attributes["synchronization_receipt"] = receipt
