"""Read an authored induction binding only at a use inside its native loop."""
from .concordance_declarations import (
    CANONICAL_VALUE, CELL_SET, CONTROL_PROGRAM, CONTROL_VALUE_BINDING,
    LOOP_SCOPE, PHI_LOOP_HEADER, SSA_BLOCK_GEOMETRY, SSA_VALUE,
    BindingKind, ControlBinding, ControlProgramFact,
)
from .identity_concordance import Ref, CorrelationTable


def loop_induction_sources(module, function_name, actual, source, use_block):
    """Return exact scoped evidence, never reactivate a withdrawn name globally."""
    from .ssa_record_return_state import function_scope_of
    name = str(function_name)
    function = module.functions.get(name)
    book = dict(getattr(module, 'metadata', {}) or {}).get('identity_book')
    if function is None or book is None or actual.page != SSA_VALUE or source.page != CANONICAL_VALUE:
        return None
    def fact(cell):
        return book.page(cell.page).cell(cell.row, cell.column)
    try:
        if actual.row[0] != function_scope_of(function) or book.latest_ref(SSA_VALUE, actual.row) != actual:
            return None
        mint = book.mint_of(actual)
        if mint is None or mint[0] != PHI_LOOP_HEADER or len(mint[1]) != 1:
            return None
        bundle = mint[1][0]
        if bundle.page != CELL_SET:
            return None
        parents = tuple(p for p, _ in book.edges_into(bundle))
        if set(fact(bundle)) != {p.key for p in parents}:
            return None
        loops = [p for p in parents if p.page == CANONICAL_VALUE]
        if len(loops) != 1:
            return None
        loop, = loops
        if loop.row[0] != source.row[0]:
            return None
        programs = []
        for row in book.page(CONTROL_PROGRAM).rows():
            if row[0] != source.row[0]:
                continue
            cell = book.latest_ref(CONTROL_PROGRAM, row)
            value = fact(cell)
            if not isinstance(value, ControlProgramFact):
                continue
            matches = [entry for entry in value.projected_iterable_bindings
                       if entry[0] == loop and entry[1] == source and entry[3] == 'induction']
            if len(matches) > 1:
                return None
            if matches and {loop, source} <= {p for p, _ in book.edges_into(cell)}:
                programs.append(cell)
        if len(programs) != 1:
            return None
        # History is evidence only for this exact SSA value and loop scope.
        # The latest name binding may correctly have been withdrawn on exit.
        bindings = []
        row = (actual.row[0], int(source.row[-1]))
        for (candidate_row, column), value in book.page(CONTROL_VALUE_BINDING).cells.items():
            if candidate_row != row or not isinstance(value, ControlBinding):
                continue
            if value.kind != BindingKind.LOOP_CONTROL or value.ssa_value != actual:
                continue
            cell = Ref(CONTROL_VALUE_BINDING, row, column)
            if {loop, source, actual} <= {p for p, _ in book.edges_into(cell)}:
                bindings.append(cell)
        if len(bindings) != 1:
            return None
        boundary = book.latest_ref(LOOP_SCOPE, (name, int(loop.row[-1]), 'boundary'))
        if boundary is None or loop not in {p for p, _ in book.edges_into(boundary)}:
            return None
        header, latch, exit_block = fact(boundary)
        if len({header, latch, exit_block}) != 3 or any(label not in function.blocks for label in (header,latch,exit_block)):
            return None
        definitions = [(label,op) for label,block in function.blocks.items() for op in block.instrs
                       if op.res is not None and int(op.res.id) == int(actual.row[-1])]
        if len(definitions) != 1 or definitions[0][0] != header or definitions[0][1].op != 'Phi':
            return None
        phi = definitions[0][1]
        incoming = tuple(phi.attributes.get('incoming_blocks', ()))
        predecessors = {label for label,block in function.blocks.items() if header in block.successors}
        if (len(incoming) != 2 or len(set(incoming)) != 2 or header in incoming
                or set(incoming) != predecessors or incoming[1] != latch or len(phi.args) != 2):
            return None
        # Traverse the declared loop body, not arbitrary SSA ancestry. Never
        # admit header, exit, preheader or a block after the scope has ended.
        region, pending = set(), list(function.blocks[header].successors)
        while pending:
            label = pending.pop()
            if label in (header,exit_block) or label in region:
                continue
            if label not in function.blocks:
                return None
            region.add(label)
            pending.extend(function.blocks[label].successors)
        dominators = CorrelationTable._dominators(function)
        if (str(use_block) not in region or latch not in region
                or header not in dominators.get(str(use_block), set())
                or exit_block in dominators.get(str(use_block), set())
                or header not in dominators.get(latch, set())):
            return None
        proof = {actual, source, loop, bundle, *parents, *programs, *bindings, boundary}
        for label in region | {header}:
            block = function.blocks[label]
            cell = book.latest_ref(SSA_BLOCK_GEOMETRY, (actual.row[0], name, str(label)))
            geometry = (tuple((str(op.op),tuple(int(v.id) for v in op.args),
                None if op.res is None else int(op.res.id),tuple(op.attributes.get('incoming_blocks') or ()))
                for op in block.instrs),tuple(block.successors))
            if cell is None or fact(cell) != geometry:
                return None
            proof.add(cell)
        return tuple(sorted(proof,key=repr))
    except (KeyError,TypeError,ValueError,AttributeError,IndexError):
        return None
