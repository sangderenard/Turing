"""C task ABI decisions read from the SSA thread-management table."""
from .concordance_declarations import C_THREAD_ACTIVATION, EMISSION_C, NAME_BINDING, SOURCE_THREAD_ACTIVATION
from .identity_concordance import Derived, Mode


def worker_targets(module, owner):
    table = getattr(module, "thread_table", None)
    if table is None:
        return ()
    return tuple(dict.fromkeys(target.row[1] for cell, fact in table.ssa_activations.items()
                               if cell.row[0] == owner for target in fact[3]))


def activation_plan(module, owner, instruction):
    from .ssa_record_return_state import ssa_value_identity_cell
    from .concordance_declarations import SOURCE_AGGREGATE_LAYOUT, SOURCE_FORMAL_ORDER
    table = module.thread_table
    if table is None:
        raise ValueError("C ThreadCreate requires the SSA thread table")
    book = table.book
    from .concordance_declarations import SSA_PROCESS_CALL
    result_cell = ssa_value_identity_cell(module.functions[owner], instruction.res.id, book=book)
    call_cell = book.latest_ref(SSA_PROCESS_CALL, (result_cell,))
    if call_cell is None:
        raise ValueError("C ThreadCreate lacks its concorded SSA operand contract")
    call = book.page(SSA_PROCESS_CALL).cell(call_cell.row, call_cell.column)
    if call.argument_ids != tuple(value.id for value in instruction.args) or call.semantic != instruction.op:
        raise ValueError("C ThreadCreate operands disagree with concordance")
    matching = [(cell, fact) for cell, fact in table.ssa_activations.items()
                if cell.row == (owner, call.callsite)]
    if len(matching) != 1:
        raise ValueError("C ThreadCreate has no unique table activation")
    activation, (source, handle, _arguments, targets) = matching[0]
    if handle != instruction.res.id or len(targets) != 1:
        raise ValueError("C thread target or handle is ambiguous")
    worker_name = targets[0].row[1]
    worker = module.functions[worker_name]
    if call.keyword_names not in {(), ("args",)}:
        raise ValueError("C thread constructor currently requires target and optional positional args")
    lifetime_cells = _prove_joined_lifetime(module.functions[owner], handle, book)
    bindings = {}
    causes = [activation, call_cell, *targets, *lifetime_cells]
    for cell, fact in table.contexts.items():
        if cell.row[0] != source or cell.row[2:] != (owner, worker_name):
            continue
        actual, formal, actual_members, formal_members, status = fact
        causes.append(cell)
        if status == "unused":
            continue
        parent_sequences = module.sequence_tables.get(owner)
        descriptor = None if parent_sequences is None else parent_sequences.sequences.get(actual)
        if descriptor is not None and str(descriptor.capacity_policy.value) != "fixed":
            raise ValueError("C worker capture requires a retained fixed sequence; dynamic owner transfer is not lowered")
        actual_roles = {role[1]: member for member, roles in actual_members for role in roles
                        if len(role) == 2 and isinstance(role[1], tuple)}
        for formal_value in worker.args:
            member = int(formal_value.id)
            roles = next((roles for candidate, roles in formal_members if candidate == member), None)
            if roles is None:
                continue
            if not roles and member == formal:
                bindings[member] = ("copy", int(actual), None)
                continue
            supported = [(role[1], actual_roles[role[1]]) for role in roles
                         if len(role) == 2 and role[1] in actual_roles]
            if not supported:
                raise ValueError("C capture member has no exact parent descriptor role")
            role, actual_member = supported[0]
            kind = "copy" if role == ("capacity_value_id",) else "alias"
            bindings[member] = (kind, int(actual_member), None)
    # Constructor args are a source argument pack, not an ordinary caller.
    # Bind their positions to the worker's retained formal identities.
    source_fact = book.page(SOURCE_THREAD_ACTIVATION).cell(source.row, source.column)
    body = source_fact[0]
    graph = table.source_body(body)
    from ..common.tensors.topological_reducer import _source_numeric_scope
    signature = book.latest_ref(SOURCE_FORMAL_ORDER, (_source_numeric_scope(graph),))
    if signature is None:
        raise ValueError("C worker requires an ingested formal order")
    parameters = book.page(SOURCE_FORMAL_ORDER).cell(signature.row, signature.column)
    causes.append(signature)
    runtime_args = instruction.args[:len(instruction.args) - len(call.capture_value_ids)]
    if call.worker_argument_pack_expanded:
        expanded = call.worker_arguments
        if len(expanded) != len(runtime_args):
            raise ValueError("C worker argument operands disagree with their source bindings")
        for (formal_id, _value_id, source_cell), actual in zip(expanded, runtime_args, strict=True):
            if any(value.id == formal_id for value in worker.args):
                bindings[int(formal_id)] = ("copy", int(actual.id), None)
                causes.append(source_cell)
    elif runtime_args:
        if call.keyword_names != ("args",) or len(runtime_args) != 1:
            raise ValueError("C worker positional argument pack is not explicit")
        parent = module.function_table.entry(source_fact[1]).graph
        packs = [cell for cell, role in source_fact[2] if role == "kw:args"]
        if len(packs) != 1:
            raise ValueError("C worker argument pack has no exact source operand")
        # The source operand cell retains its canonical value id.
        pack_id = int(packs[0].row[1])
        layout_row = (parent.G.graph["lexical_read_scope"], pack_id)
        layout_cell = book.latest_ref(SOURCE_AGGREGATE_LAYOUT, layout_row)
        layout = None if layout_cell is None else book.page(SOURCE_AGGREGATE_LAYOUT).cell(layout_cell.row, layout_cell.column)
        if layout is None or layout.aggregate_kind != "tuple":
            raise ValueError("C worker args require their exact authored tuple layout")
        causes.append(book.latest_ref(SOURCE_AGGREGATE_LAYOUT, tuple(layout_row)))
        leaves = tuple(layout.members)
        if len(leaves) != len(parameters):
            raise ValueError("C worker args arity does not match the source signature")
        scope = graph.G.graph["lexical_read_scope"]
        for index, parameter in enumerate(parameters):
            binding = book.page(NAME_BINDING).latest((scope, parameter, 0))
            if binding is None:
                raise ValueError("C worker argument has no exact formal binding")
            formal_id = int(binding.value_id)
            if any(value.id == formal_id for value in worker.args):
                bindings[formal_id] = ("pack", int(runtime_args[0].id), int(index))
    elif parameters:
        raise ValueError("C worker defaults/keyword argument packs are not lowered")
    if set(bindings) != {int(value.id) for value in worker.args}:
        raise ValueError("C thread context does not account for every worker formal")
    # Scalar closure values can be copied only when the authored cell is stable.
    # A rebound cell needs shared closure-cell storage, never an initializer copy.
    from .concordance_declarations import SOURCE_CLOSURE_CAPTURE
    for context, fact in table.contexts.items():
        if context.row[0] != source or fact[-1] != "live":
            continue
        if any(kind == "copy" and actual == fact[0] for kind, actual, _index in bindings.values()):
            capture = context.row[1]
            parent_binding = book.page(SOURCE_CLOSURE_CAPTURE).cell(capture.row, capture.column)[1]
            if any(row[:2] == parent_binding.row[:2] and row[2] != parent_binding.row[2]
                   for row in book.page(NAME_BINDING).scope_rows(parent_binding.row[0])):
                raise ValueError("C worker needs retained storage for a rebound closure cell")
    ordered = tuple((int(formal.id), *bindings[int(formal.id)]) for formal in worker.args)
    causes.extend(ssa_value_identity_cell(worker, formal.id, book=book) for formal in worker.args)
    policy = (worker_name, ordered, "join_before_owner_return")
    cell = book.post(C_THREAD_ACTIVATION, (activation,), policy, stage=EMISSION_C,
                     provenance=Derived(tuple(causes)), mode=Mode.CONCORD)
    return cell, policy


def _prove_joined_lifetime(function, handle, book):
    """Require a completion dependency before normal owner return.

    Cleanup may wait while unwinding a fault, but cannot silently add a normal
    join to an authored program. Escaping task/context ownership is a separate
    lifetime ABI and is explicitly refused until represented.
    """
    import networkx as nx
    graph = nx.DiGraph()
    starts, joins, returns = [], [], []
    from .concordance_declarations import SSA_BLOCK_GEOMETRY
    from .ssa_record_return_state import function_scope_of
    cells = []
    for name, block in function.blocks.items():
        row = (function_scope_of(function), str(function.name), str(name))
        cell = book.latest_ref(SSA_BLOCK_GEOMETRY, row)
        if cell is None:
            raise ValueError('C thread lifetime requires concorded SSA block geometry')
        instructions, successors = book.page(SSA_BLOCK_GEOMETRY).cell(cell.row, cell.column)
        actual = (tuple((str(instruction.op), tuple(int(value.id) for value in instruction.args),
                         None if instruction.res is None else int(instruction.res.id),
                         tuple(instruction.attributes.get('incoming_blocks') or ()))
                        for instruction in block.instrs), tuple(block.successors))
        if actual != (instructions, successors):
            raise ValueError('C thread lifetime geometry disagrees with actual SSA edges')
        cells.append(cell)
        graph.add_node(name)
        for target in successors:
            graph.add_edge(name, target)
        for index, (operation, arguments, _result, _incoming) in enumerate(instructions):
            if operation in {'ThreadStart', 'ThreadJoin', 'CompletionWait'} and arguments and arguments[0] == handle:
                roles = tuple(function.blocks[name].instrs[index].arg_roles)
                if len(arguments) != 1 and not (operation == 'CompletionWait' and roles == ('handle', 'after')):
                    raise ValueError('C thread timeout/keyword lifecycle arguments are not lowered')
                (starts if operation == 'ThreadStart' else joins).append((name, index))
            # A fault-specific unwind exemption needs its own control fact.
            # Until then every return must prove completion of borrowed state.
            if operation == 'Ret':
                returns.append(name)
    for start_block, start_index in starts:
        if any(block == start_block and index > start_index for block, index in joins):
            continue
        remainder = graph.copy()
        remainder.remove_nodes_from(block for block, _index in joins if block != start_block)
        if any(target in remainder and start_block in remainder and nx.has_path(remainder, start_block, target)
               for target in returns):
            raise ValueError("C thread capture can outlive its owner; a completing join is required")

    return tuple(cells)
