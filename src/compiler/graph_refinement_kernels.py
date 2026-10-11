"""Authored graph mechanics, compiled by the public source compiler.

These routines operate on local integer tables. They neither mint identities
nor write the book: the calling graph pass owns those decisions. Scratch and
output spans belong to one invocation, so independent invocations share only
the compiled code, never a work arena.
"""


def indegree_counts(edge_count, targets, indegree):
    """Accumulate one admitted integer edge table into its owned counts."""
    edge = edge_count - edge_count
    while edge < edge_count:
        node = targets[edge]
        indegree[node] = indegree[node] + 1
        edge = edge + 1
    return indegree


def topological_order(n, edge_count, offsets, targets, indegree, heap, output, status):
    """Kahn's algorithm with a lexical min-heap on normalized identities.

    ``offsets``/``targets`` are CSR outgoing edges, including multiplicity.
    Equal lexical keys are normalized by graph insertion order by the caller.
    The caller supplies zeroed indegrees and node-sized scratch/output spans.
    Status is the number emitted; fewer than n means a cycle.
    """
    edge = n - n
    while edge < edge_count:
        target = targets[edge]
        indegree[target] = indegree[target] + 1
        edge = edge + 1
    size = n - n
    node = n - n
    while node < n:
        if indegree[node] == 0:
            # Nodes enter in lexical order, already a valid heap.
            heap[size] = node
            size = size + 1
        node = node + 1
    emitted = n - n
    while size > 0:
        node = heap[0]
        size = size - 1
        replacement = heap[size]
        position = n - n
        child = n - n + 1
        while child < size:
            if child + 1 < size:
                if heap[child + 1] < heap[child]:
                    child = child + 1
            if replacement <= heap[child]:
                break
            heap[position] = heap[child]
            position = child
            child = position * 2 + 1
        heap[position] = replacement
        output[emitted] = node
        emitted = emitted + 1
        edge = offsets[node]
        while edge < offsets[node + 1]:
            target = targets[edge]
            indegree[target] = indegree[target] - 1
            if indegree[target] == 0:
                position = size
                parent = (position - 1) // 2
                while position > 0:
                    if heap[parent] <= target:
                        break
                    heap[position] = heap[parent]
                    position = parent
                    parent = (position - 1) // 2
                heap[position] = target
                size = size + 1
            edge = edge + 1
    status[0] = emitted
    return status


def dependency_levels(n, preference, order, parent_offsets, parents, child_offsets, children, asap, levels):
    """The existing ILPScheduler equations, on an admitted DAG's integer CSR.

    ASAP(v) = 0 for no parents, else 1 + max ASAP(parent).
    ALAP(v) = ASAP(v) for no children, else min(ALAP(child) - 1).
    In particular disconnected outputs keep THEIR ASAP level, not a global
    makespan. The coordinator validates both relations against the order.
    Outputs are the caller-owned levels span; the return is the processed count.
    """
    position = n - n
    while position < n:
        node = order[position]
        level = n - n
        edge = parent_offsets[node]
        while edge < parent_offsets[node + 1]:
            candidate = asap[parents[edge]] + 1
            if candidate > level:
                level = candidate
            edge = edge + 1
        asap[node] = level
        levels[node] = level
        position = position + 1
    if preference != 0:
        position = n
        while position > 0:
            position = position - 1
            node = order[position]
            edge = child_offsets[node]
            if edge < child_offsets[node + 1]:
                level = levels[children[edge]] - 1
                edge = edge + 1
                while edge < child_offsets[node + 1]:
                    candidate = levels[children[edge]] - 1
                    if candidate < level:
                        level = candidate
                    edge = edge + 1
                levels[node] = level
    return n


def broadcast_shape_extents(rank, start, left, right, output):
    """The descriptor's exact broadcast law, on already aligned extents.

    Equal extents survive; an extent of one yields to the other operand.
    In particular (0, 1) yields zero, NOT max(0, 1). Incompatibility makes
    the entire output row unusable; it does not propose a partial shape.
    The graph owner admits proven shapes and retains dtype/book authority.
    """
    axis = rank - rank
    compatible = rank - rank + 1
    while axis < rank:
        position = start + axis
        left_extent = left[position]
        right_extent = right[position]
        if left_extent == right_extent:
            output[position] = left_extent
        elif left_extent == 1:
            output[position] = right_extent
        elif right_extent == 1:
            output[position] = left_extent
        else:
            compatible = rank - rank
            break
        axis = axis + 1
    return compatible


def broadcast_shapes(n, offsets, left, right, output, status):
    """Independent descriptor rows, with outputs indexed by the SAME rows.

    offsets delimit aligned integer extents; an empty interval is rank zero.
    This batch calls the same law used by the eager single-query path. No
    graph identity, provenance record, dtype or runtime tensor crosses ABI.
    """
    row = n - n
    while row < n:
        start = offsets[row]
        rank = offsets[row + 1] - start
        status[row] = broadcast_shape_extents(rank, start, left, right, output)
        row = row + 1
    return status


def dead_metadata_order(n, edge_count, offsets, parents, eligible, remaining, heap, output, status):
    """The ENTIRE ordered dead-metadata worklist, not a per-node predicate.

    Reverse edges turn each original out-degree into Kahn's remaining count.
    An ineligible node has one extra permanent count, so it and every producer
    it still consumes survive. A cycle also survives. The existing lexical
    min-heap emits exactly the first dead node in original insertion order,
    then updates all affected predecessors until exhaustion. One invocation
    handles the whole admitted graph; Python only publishes the removals.
    """
    node = n - n
    while node < n:
        remaining[node] = n - n
        if eligible[node] == 0:
            remaining[node] = n - n + 1
        node = node + 1
    topological_order(n, edge_count, offsets, parents, remaining, heap, output, status)
    return status


def read_only_storage_effects(n, instruction_count, use_count, blocked, kinds, results,
                             argument_offsets, operands, call_formals, writes,
                             active, memory_addresses, memory_dtypes,
                             edge_sources, edge_targets, offsets, cursor, targets,
                             unsafe, queue, read_only, storage_types):
    """Fused complete effect AND pointer-storage analyses in ONE invocation.

    Every existing (function, value ID) has a query state and an address state.
    Definitions veto the QUERY, not every alias of its storage. Address states
    follow only GEP/View/PtrCast's base operand; Store's value operand is read,
    its destination is written. An admitted call follows the exact callee
    formal's QUERY. Imported headers contribute their accepted effect facts.

    Build the dependency relation, reverse-index it, seed actual writes and
    definitions, then reach the least fixed point across aliases AND recursive
    calls. No Python callback, per-value native call, graph mutation proposal
    or Python worklist: all edge construction, CSR construction, queue updates,
    convergence and final Boolean-table writes occur in this authored program.
    The same body executes eagerly; the public compiler compiles it unchanged.
    """
    state_count = n * 2
    state = n - n
    while state < state_count:
        unsafe[state] = n - n
        offsets[state] = n - n
        state = state + 1
    offsets[state_count] = n - n

    # Query(v) depends on Address(v). Definitions veto only Query(v), which
    # is why a GEP result definition does not make its BASE storage writable.
    edge_count = n - n
    value = n - n
    while value < n:
        unsafe[value] = blocked[value]
        edge_sources[edge_count] = value + n
        edge_targets[edge_count] = value
        edge_count = edge_count + 1
        value = value + 1

    instruction = n - n
    while instruction < instruction_count:
        result = results[instruction]
        if active[instruction] != 0:
            if result >= 0:
                unsafe[result] = n - n + 1
            if kinds[instruction] != 0:
                start = argument_offsets[instruction]
                if start < argument_offsets[instruction + 1]:
                    if result >= 0:
                        base = operands[start]
                        edge_sources[edge_count] = result + n
                        edge_targets[edge_count] = base + n
                        edge_count = edge_count + 1
        instruction = instruction + 1

    use = n - n
    while use < use_count:
        actual = operands[use]
        if writes[use] != 0:
            unsafe[actual + n] = n - n + 1
        formal = call_formals[use]
        if formal >= 0:
            edge_sources[edge_count] = formal
            edge_targets[edge_count] = actual + n
            edge_count = edge_count + 1
        use = use + 1

    # Reverse CSR: each known bad state wakes exactly its dependants.
    edge = n - n
    while edge < edge_count:
        source = edge_sources[edge]
        offsets[source + 1] = offsets[source + 1] + 1
        edge = edge + 1
    state = n - n
    while state < state_count:
        offsets[state + 1] = offsets[state + 1] + offsets[state]
        cursor[state] = offsets[state]
        state = state + 1
    edge = n - n
    while edge < edge_count:
        source = edge_sources[edge]
        targets[cursor[source]] = edge_targets[edge]
        cursor[source] = cursor[source] + 1
        edge = edge + 1

    head = n - n
    tail = n - n
    state = n - n
    while state < state_count:
        if unsafe[state] != 0:
            queue[tail] = state
            tail = tail + 1
        state = state + 1
    while head < tail:
        state = queue[head]
        head = head + 1
        edge = offsets[state]
        while edge < offsets[state + 1]:
            dependant = targets[edge]
            if unsafe[dependant] == 0:
                unsafe[dependant] = n - n + 1
                queue[tail] = dependant
                tail = tail + 1
            edge = edge + 1

    value = n - n
    while value < n:
        read_only[value] = 1 - unsafe[value]
        value = value + 1

    # Reuse the SAME edge/CSR/queue scratch after effects have settled.
    # Storage inference is LOCAL, has no call edges, and excludes View.
    # Lattice: 0=no evidence, positive=one exact dtype, -1=conflict. 'ptr'
    # is a normal evidence token here; the original publisher vetoes it.
    value = n - n
    while value < n:
        storage_types[value] = n - n
        offsets[value] = n - n
        unsafe[value] = n - n
        value = value + 1
    offsets[n] = n - n
    edge_count = n - n
    instruction = n - n
    while instruction < instruction_count:
        result = results[instruction]
        if kinds[instruction] == 1:
            start = argument_offsets[instruction]
            if start < argument_offsets[instruction + 1]:
                if result >= 0:
                    base = operands[start]
                    edge_sources[edge_count] = result
                    edge_targets[edge_count] = base
                    edge_count = edge_count + 1
        address = memory_addresses[instruction]
        dtype = memory_dtypes[instruction]
        if address >= 0:
            if dtype != 0:
                incumbent = storage_types[address]
                if incumbent == 0:
                    storage_types[address] = dtype
                elif incumbent != dtype:
                    storage_types[address] = n - n - 1
        instruction = instruction + 1

    edge = n - n
    while edge < edge_count:
        source = edge_sources[edge]
        offsets[source + 1] = offsets[source + 1] + 1
        edge = edge + 1
    value = n - n
    while value < n:
        offsets[value + 1] = offsets[value + 1] + offsets[value]
        cursor[value] = offsets[value]
        value = value + 1
    edge = n - n
    while edge < edge_count:
        source = edge_sources[edge]
        targets[cursor[source]] = edge_targets[edge]
        cursor[source] = cursor[source] + 1
        edge = edge + 1

    # Each value can rise at most twice: no evidence -> dtype -> conflict.
    # Thus the existing 2*n queue is sufficient even with alias cycles.
    head = n - n
    tail = n - n
    value = n - n
    while value < n:
        if storage_types[value] != 0:
            queue[tail] = value
            tail = tail + 1
            unsafe[value] = n - n + 1
        value = value + 1
    while head < tail:
        source = queue[head]
        head = head + 1
        unsafe[source] = n - n
        incoming = storage_types[source]
        edge = offsets[source]
        while edge < offsets[source + 1]:
            base = targets[edge]
            incumbent = storage_types[base]
            proposed = incumbent
            if incumbent == 0:
                proposed = incoming
            elif incumbent != incoming:
                proposed = n - n - 1
            if proposed != incumbent:
                storage_types[base] = proposed
                if unsafe[base] == 0:
                    queue[tail] = base
                    tail = tail + 1
                    unsafe[base] = n - n + 1
            edge = edge + 1
    return read_only
