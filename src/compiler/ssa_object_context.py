"""Executable native object context using the incumbent sequence/table ABI.

No Python object registry, callback, handle-to-address cast or new opcode.
These are SSA ABI helpers, NOT automatic lowering of arbitrary Python objects.
The context's exact sequence tables remain the storage authority. Zero handles
are explicit failure/absence. All allocation lives until context release.
"""
from __future__ import annotations

from ..transmogrifier.ssa import Function, IRModule, SSAObjectContext, SSASequenceTable, SSAValue
from .ir_sequence_tables import (
    _Builder, _storage_values, lower_sequence_allocate, lower_sequence_insert,
    lower_table_lookup, lower_table_store,
    lower_sequence_reserve,
)
from .ssa_heap_allocation import heap_release
from .ssa_object_operations import ObjectOperation, declare_object_helper


def _publish_generated_helper(function):
    """Publish the SSA identities at the writer that installs a helper.

    Object-context helpers are assembled after ordinary control lowering, so
    they do not pass through ``_ControlSSABuilder._register_sequence_lowering``.
    They are nevertheless ordinary repository SSA: the helper's function root
    owns its blocks and formals, and each generated result is caused by its
    already-published operands (or by that root for a literal/cyclic seed).
    Record that chain here, before any later geometry-sensitive rewrite sees
    the function.
    """
    from .concordance_declarations import (
        CELL_SET,
        CONTROL_CONST,
        CONTROL_EXPRESSION,
        CONTROL_SSA_FINISH,
        SSA_BLOCK,
        SSA_VALUE,
        SSABlockKind,
        SSAValueFact,
        SSAValueOrigin,
    )
    from .identity_concordance import Derived, Mode, Novel, current_identity_book
    from .precompile_to_ssa import _function_root_cell, _post_ssa_block_geometry
    from .ssa_record_return_state import function_scope_of

    book = current_identity_book()
    scope = function_scope_of(function)
    root = _function_root_cell(book, scope)
    block_cells = {}
    for label in function.blocks:
        row = (scope, str(function.name), str(label))
        cell = book.latest_ref(SSA_BLOCK, row)
        if cell is None:
            cell = book.post(
                SSA_BLOCK,
                row,
                SSABlockKind.SEQUENCE_HELPER,
                stage=CONTROL_SSA_FINISH,
                provenance=Derived((root,)),
                mode=Mode.CONCORD,
            )
        block_cells[str(label)] = cell

    for argument in function.args:
        row = (scope, int(argument.id))
        if book.latest_ref(SSA_VALUE, row) is None:
            book.post(
                SSA_VALUE,
                row,
                SSAValueFact(
                    argument.dtype,
                    tuple(argument.shape or ()),
                    SSAValueOrigin.ADOPTED_GRAPH_ID,
                ),
                stage=CONTROL_SSA_FINISH,
                provenance=Derived((root,)),
                mode=Mode.CONCORD,
            )

    pending = [
        (str(label), instruction)
        for label, block in function.blocks.items()
        for instruction in block.instrs
        if instruction.res is not None
        and book.latest_ref(
            SSA_VALUE, (scope, int(instruction.res.id)),
        ) is None
    ]

    def post_result(label, instruction, *, require_all_operands):
        result = instruction.res
        operand_cells = tuple(dict.fromkeys(
            cell
            for value in instruction.args
            if (cell := book.latest_ref(
                SSA_VALUE, (scope, int(value.id)),
            )) is not None
        ))
        if require_all_operands and len(operand_cells) != len({
            int(value.id) for value in instruction.args
        }):
            return False
        sources = tuple(dict.fromkeys((
            *operand_cells,
            block_cells[label],
        )))
        if len(sources) == 1:
            operand = sources[0]
        else:
            ordinal = book.page(CELL_SET).scope_row_count(scope) + 1
            operand = book.post(
                CELL_SET,
                (scope, ordinal),
                tuple(cell.key for cell in sources),
                stage=CONTROL_SSA_FINISH,
                provenance=Derived(sources),
                mode=Mode.CONCORD,
            )
        book.post(
            SSA_VALUE,
            (scope, int(result.id)),
            SSAValueFact(
                result.dtype,
                tuple(result.shape or ()),
                SSAValueOrigin.MINTED,
            ),
            stage=CONTROL_SSA_FINISH,
            provenance=Novel(
                CONTROL_CONST
                if str(instruction.op).casefold() in {"const", "constant"}
                else CONTROL_EXPRESSION,
                (operand,),
            ),
            mode=Mode.CONCORD,
        )
        return True

    while pending:
        retained = []
        for label, instruction in pending:
            if not post_result(
                label, instruction, require_all_operands=True,
            ):
                retained.append((label, instruction))
        if len(retained) == len(pending):
            # Loop Phi/backedge pairs are mutually causal in SSA. Their exact
            # helper block is the common producer boundary; publish against
            # that owner plus every operand already established outside the
            # cycle, then let the ordinary operand graph retain the cycle.
            for label, instruction in retained:
                post_result(
                    label, instruction, require_all_operands=False,
                )
            break
        pending = retained

    _post_ssa_block_geometry(book, scope, function)


def publish_linked_object_context_module(module: IRModule) -> None:
    """Admit marked object-context helpers into the active compilation book.

    ``lower_object_context`` may run before a source compilation opens its
    book.  The resulting module is still the exact linked SSA input, but facts
    written while it was built belong to the earlier book.  Publish those same
    helper identities at the host-module adoption boundary; later geometry
    passes then read the active concordance instead of reconstructing an owner.
    """

    for function in module.functions.values():
        if function.metadata.get("generated_helper_provenance") == (
            "object-context"
        ):
            _publish_generated_helper(function)


def object_type_classes(vocabulary):
    """Resolve only admitted builtin guards, never arbitrary MRO hooks."""
    import builtins
    supported = {"builtins." + name: getattr(builtins, name) for name in
                 ('dict', 'list', 'tuple', 'set', 'frozenset', 'object',
                  'bool', 'int', 'float', 'str', 'bytes')}
    supported['builtins.NoneType'] = type(None)
    # Closed compiler-owned set view, not arbitrary ABC/user class admission.
    # A builtin set subclass with empty C storage cannot page: set(view) and
    # frozenset(view) would bypass iteration and silently produce empties.
    from .live_tiles import TiledSet
    supported['src.compiler.live_tiles.TiledSet'] = TiledSet
    if any(token not in supported for token in vocabulary):
        raise ValueError('object type evidence supports declared builtin guards only')
    return tuple(supported[token] for token in vocabulary)


def _object_source_contract(graph, value_id, role, fact, nodes):
    from .concordance_declarations import NATIVE_OBJECT_SOURCE, PLANNER_HIERARCHY
    from .identity_concordance import current_identity_book, Derived, Mode
    from ..common.tensors.topological_reducer import node_identity_cell
    book = current_identity_book()
    cells = tuple(dict.fromkeys(node_identity_cell(graph, int(node)) for node in nodes))
    row = (cells[0].row[0], int(value_id), role)
    contents = tuple(sorted(fact.items()))
    previous = book.latest_ref(NATIVE_OBJECT_SOURCE, row)
    if previous is not None and book.page(NATIVE_OBJECT_SOURCE).latest(row) == contents:
        return previous
    return book.post(NATIVE_OBJECT_SOURCE, row, contents, stage=PLANNER_HIERARCHY,
                     provenance=Derived(cells), mode=Mode.REVISE)


def object_source_contract(cell, *, book=None):
    from .concordance_declarations import NATIVE_OBJECT_SOURCE
    from .identity_concordance import current_identity_book
    book = current_identity_book() if book is None else book
    if cell is None or cell.page != NATIVE_OBJECT_SOURCE:
        raise ValueError('native object source has no accepted concordance cell')
    if book.latest_ref(NATIVE_OBJECT_SOURCE, cell.row) != cell:
        raise ValueError('native object source contract was superseded')
    return dict(book.page(NATIVE_OBJECT_SOURCE).latest(cell.row))


def _mapping_items_source(graph, node_id):
    """The reducer's items operation and exact receiver edge, not a name lookup."""
    node = graph.nodes.get(node_id, {})
    if node.get('op') != 'items':
        return None
    sources = [parent for parent, role in node.get('parents') or () if role == 'operand']
    return sources[0] if len(sources) == 1 and sources[0] in graph else None


def source_object_type_tests(graph, admitted, context):
    """Consume the intrinsic's incumbent static type objects and operand edge."""
    if not context.type_vocabulary:
        return {}
    tests = {}
    classes = dict(zip(object_type_classes(context.type_vocabulary), context.type_vocabulary))
    for node_id, data in graph.nodes(data=True):
        attributes = data.get('attributes') or {}
        if attributes.get('extraction_identity') != 'builtins.isinstance':
            continue
        targets = (attributes.get('static_call_values') or {}).get('arg:1')
        targets = targets if type(targets) is tuple else (targets,)
        if not targets or any(not isinstance(target, type) or target not in classes for target in targets):
            continue
        operands = [int(graph.nodes[parent].get('value_id', parent))
                    for parent, role in data.get('parents') or () if role == 'arg:0']
        if len(operands) != 1 or operands[0] not in admitted:
            continue
        value_id = int(data.get('value_id', node_id))
        tests[value_id] = {'source_value_id': value_id, 'reference_value_id': operands[0],
                          'accepted_types': tuple(classes[target] for target in targets),
                          'intrinsic': 'builtins.isinstance'}
    return tests


def source_native_type_tests(graph):
    """Source type predicates from exact definition/concordance cells.

    ``isinstance`` records subtype admission. ``type(value) is T`` records
    exact identity.  The two never share inference rules.  Callable and
    target identities are already concordance cells produced by source
    reduction; ``static_call_values`` is deliberately ignored because a live
    class object is neither a native ABI nor durable compiler evidence.
    """

    from .concordance_declarations import (
        SOURCE_PYTHON_IDENTITY,
        STATIC_REFERENCE_USE,
        canonical_python_subtype_identities,
        canonical_python_type_identity,
    )
    from .identity_concordance import current_identity_book

    book = current_identity_book()
    identity_page = book.page("source_python_identity_concordance")
    tests = {}
    for node_id, data in graph.nodes(data=True):
        attributes = data.get("attributes") or {}
        identity_row = attributes.get("source_python_identity_row")
        if identity_row is None:
            continue
        identity = identity_page.latest(tuple(identity_row))
        if not identity or str(identity[0]) != "builtins.isinstance":
            continue
        definition_cells = dict(
            attributes.get("static_call_definition_cells") or {}
        )
        target_cell = definition_cells.get("arg:1")
        if target_cell is None:
            continue
        definition_page = book.pages.get(target_cell.page.name)
        if definition_page is None:
            continue
        definition = definition_page.cells.get(
            (target_cell.row, target_cell.column)
        )
        if getattr(definition, "kind", None) is None or str(
            getattr(definition.kind, "value", definition.kind)
        ) != "class":
            continue
        module, qualname, _receiver = target_cell.row
        accepted = canonical_python_type_identity(f"{module}.{qualname}")
        operands = [
            int(graph.nodes[parent].get("value_id", parent))
            for parent, role in data.get("parents") or ()
            if str(role) == "arg:0" and parent in graph
        ]
        if len(operands) != 1:
            continue
        value_id = int(data.get("value_id", node_id))
        tests[value_id] = {
            "source_value_id": value_id,
            "subject_value_id": operands[0],
            "accepted_types": canonical_python_subtype_identities(
                accepted, book=book,
            ),
            "intrinsic": "builtins.isinstance",
            "intrinsic_cell": book.latest_ref(
                SOURCE_PYTHON_IDENTITY, tuple(identity_row),
            ),
            "target_definition_cell": target_cell,
            "predicate_kind": "subtype",
        }

    # Exact ``type(value) is T`` uses the StaticReference use edge of T.  The
    # target's spelling is not consulted: the definition cell is the same
    # canonical identity used by VALUE_TYPE_ABI.
    import ast

    static_use_page = book.pages.get("static_reference_use")

    def definition_for_reference(reference_node_id):
        if static_use_page is None:
            return None
        cells = []
        for row in static_use_page.rows():
            if (
                isinstance(row, tuple) and len(row) == 3
                and int(row[2]) == int(reference_node_id)
            ):
                use_cell = book.latest_ref(STATIC_REFERENCE_USE, row)
                cells.extend(
                    source for source, _stage in book.edges_into(use_cell)
                    if source.page.name == "static_symbol_definition"
                )
        cells = tuple(dict.fromkeys(cells))
        return cells[0] if len(cells) == 1 else None

    for node_id, data in graph.nodes(data=True):
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Compare)
            and len(expression.ops) == 1
            and isinstance(expression.ops[0], ast.Is)
        ):
            continue
        parents = [
            int(parent) for parent, _role in data.get("parents") or ()
            if parent in graph
        ]
        type_calls = []
        for parent in parents:
            call_attributes = graph.nodes[parent].get("attributes") or {}
            row = call_attributes.get("source_python_identity_row")
            identity = (
                None if row is None else identity_page.latest(tuple(row))
            )
            if identity and str(identity[0]) == "builtins.type":
                type_calls.append((parent, tuple(row)))
        if len(type_calls) != 1:
            continue
        type_call, intrinsic_row = type_calls[0]
        subjects = [
            int(graph.nodes[parent].get("value_id", parent))
            for parent, role in graph.nodes[type_call].get("parents") or ()
            if str(role) == "arg:0" and parent in graph
        ]
        targets = []
        for parent in parents:
            if parent == type_call:
                continue
            definition_cell = definition_for_reference(parent)
            if definition_cell is not None:
                targets.append(definition_cell)
        if len(subjects) != 1 or len(targets) != 1:
            continue
        target_cell = targets[0]
        module, qualname, _receiver = target_cell.row
        value_id = int(data.get("value_id", node_id))
        tests[value_id] = {
            "source_value_id": value_id,
            "subject_value_id": subjects[0],
            "accepted_types": (
                canonical_python_type_identity(f"{module}.{qualname}"),
            ),
            "intrinsic": "builtins.type",
            "intrinsic_cell": book.latest_ref(
                SOURCE_PYTHON_IDENTITY, intrinsic_row,
            ),
            "target_definition_cell": target_cell,
            "predicate_kind": "exact",
        }
    return tests


def source_object_identity_reductions(graph, admitted):
    """Fuse the exact pure all(map(is, zip(a,b))) causal island.

    Consume incumbent producer/ownership edges. An extra use, filter, async
    generator or effectful body is not this reduction and is never discarded.
    """
    import ast
    from .dependency_cone import parent_record_dependents
    users = parent_record_dependents(graph)
    reductions = {}
    for node, data in graph.nodes(data=True):
        if (data.get('attributes') or {}).get('extraction_identity') != 'builtins.all':
            continue
        args = [parent for parent, role in data.get('parents') or () if role == 'arg:0']
        if len(args) != 1 or args[0] not in graph:
            continue
        collection = args[0]
        collected = graph.nodes[collection]
        attrs = collected.get('attributes') or {}
        edges = dict((role, parent) for parent, role in collected.get('parents') or ())
        if (attrs.get('result_kind') != 'collection' or set(edges) != {'value', 'control'}
                or any(user != node for user in users.get(collection, ()))):
            continue
        predicate, loop_id = edges['value'], edges['control']
        materializer = attrs.get('materializer_node_id')
        iterable = attrs.get('collection_iterable_value_id')
        if any(value not in graph for value in (predicate, loop_id, materializer, iterable)):
            continue
        test, loop, zipper = (graph.nodes[value] for value in (predicate, loop_id, iterable))
        expression = test.get('expr_obj')
        generator = graph.nodes[materializer].get('expr_obj')
        targets = tuple(((loop.get('attributes') or {}).get('loop_target_bindings') or {}).values())
        components = dict((role, parent) for parent, role in test.get('parents') or ())
        sources = tuple(parent for parent, role in sorted(zipper.get('parents') or (), key=lambda p: p[1])
                        if str(role).startswith('arg:'))
        if (not isinstance(expression, ast.Compare) or len(expression.ops) != 1
                or not isinstance(expression.ops[0], ast.Is)
                or not isinstance(generator, ast.GeneratorExp) or len(generator.generators) != 1
                or generator.generators[0].ifs or generator.generators[0].is_async
                or len(targets) != 2 or components != {'lhs': targets[0], 'rhs': targets[1]}
                or (zipper.get('attributes') or {}).get('extraction_identity') != 'builtins.zip'
                or len(sources) != 2 or any(str(role).startswith('kw:') for _, role in zipper.get('parents') or ())
                or any(int(graph.nodes[source].get('value_id', source)) not in admitted for source in sources)):
            continue
        result_id = int(data.get('value_id', node))
        fact = {'result_value_id': result_id, 'source_node_id': int(node),
                'source_value_ids': tuple(int(graph.nodes[source].get('value_id', source)) for source in sources),
                'loop_node_id': int(loop_id), 'owned_value_ids': (int(collection), int(materializer),
                    int(iterable), int(predicate), *map(int, targets))}
        cell = _object_source_contract(graph, result_id, 'identity_reduction', fact,
                                       (node, collection, materializer, iterable, predicate, loop_id, *sources, *targets))
        reductions[result_id] = {**fact, 'contract_cell': cell}
    return reductions


def install_object_identity_reductions(control, graph, dispatch_subgraphs, reductions):
    """Replace accepted pure generator control with its native scalar fold."""
    from dataclasses import replace
    import re
    from .control_source import (ControlSequenceMutation, SequenceMutationBlock, SequenceBlock,
        StatementBlock, ConditionalBlock, LoopBlock, WhileBlock, CallBlock)
    if not reductions:
        return control
    by_loop = {receipt['loop_node_id']: receipt for receipt in reductions.values()}
    covered = {value for receipt in reductions.values()
               for value in (receipt['source_node_id'], *receipt['owned_value_ids'])}
    retired_regions = {index for index, region in enumerate(dispatch_subgraphs)
                       if (nodes := set(region.G.graph.get('deployment_nodes', ()))) and nodes <= covered}
    removed_inductions, installed = set(), set()
    def rebuild(block):
        if isinstance(block, LoopBlock) and block.source_loop_node_id in by_loop:
            receipt = by_loop[block.source_loop_node_id]
            accepted = object_source_contract(receipt['contract_cell'])
            if block.terminal_controls:
                raise ValueError('pure identity reduction acquired terminal effects')
            removed_inductions.add(block.induction)
            installed.add(accepted['result_value_id'])
            return SequenceMutationBlock(ControlSequenceMutation(accepted['result_value_id'],
                'object_identity_reduce', accepted['source_value_ids'], accepted['source_node_id'],
                object_contract_cell=receipt['contract_cell']))
        if isinstance(block, StatementBlock) and len(block.lines) == 1:
            match = re.fullmatch(r'__scheduled_region_(\d+)__', block.lines[0])
            if match and int(match[1]) in retired_regions:
                return SequenceBlock(())
        if isinstance(block, SequenceBlock):
            return replace(block, blocks=tuple(rebuild(child) for child in block.blocks))
        if isinstance(block, ConditionalBlock):
            return replace(block, body=rebuild(block.body), orelse=None if block.orelse is None else rebuild(block.orelse))
        if isinstance(block, WhileBlock):
            return replace(block, condition=rebuild(block.condition), body=rebuild(block.body))
        if isinstance(block, LoopBlock):
            return replace(block, body=rebuild(block.body))
        if isinstance(block, CallBlock):
            return replace(block, callee=rebuild(block.callee))
        return block
    root = rebuild(control.root)
    if installed != set(reductions):
        raise ValueError('native identity reduction lost its exact owning loop')
    return replace(control, root=root,
        region_indices=tuple(index for index in control.region_indices if index not in retired_regions),
        iterable_bindings=tuple(binding for binding in control.iterable_bindings if binding[2] not in removed_inductions),
        projected_iterable_bindings=tuple(binding for binding in control.projected_iterable_bindings if binding[2] not in removed_inductions))


def source_constructor_mutations(graph, context):
    """Read native object births from exact authored literal nodes.

    The explicit context admits a representation, not a schema for future
    contents. Empty dict/list literals have known kind and zero initial extent.
    Their source identities become handles; they are not private column arenas.
    Other construction forms are not inferred from a callable's spelling.
    """
    import ast
    from .control_source import ControlSequenceMutation
    from .dependency_cone import iter_dependency_cone
    aggregate_users = {}
    for user, data in graph.nodes(data=True):
        for parent, role in data.get('parents') or ():
            if role in {'elts', 'elt'}:
                aggregate_users.setdefault(parent, []).append(user)
    def loop_owner(node_id):
        owners = set()
        for user in iter_dependency_cone((node_id,), lambda node: aggregate_users.get(node, ())):
            data = graph.nodes[user]
            if str(data.get('type', '')).casefold() == 'generatorexp':
                owners.update(int(parent) for parent, role in data.get('parents') or ()
                              if role == 'generators')
        if len(owners) > 1:
            raise ValueError('native object birth has ambiguous generator ownership')
        return next(iter(owners), None)
    def constructor_contract(node_id, value_id, kind, members):
        owner = loop_owner(node_id)
        return _object_source_contract(graph, value_id, 'constructor',
            {'source_value_id': value_id, 'source_node_id': int(node_id), 'kind': kind,
             'member_value_ids': members, 'owner_loop_node_id': owner},
            (node_id, *members, *((owner,) if owner is not None else ())))
    mutations = []
    for node_id, data in graph.nodes(data=True):
        expression = data.get('expr_obj')
        kind = ('builtins.dict' if isinstance(expression, ast.Dict) and not expression.keys
                else 'builtins.list' if isinstance(expression, ast.List) and not expression.elts
                else None)
        if kind is None:
            attributes = data.get('attributes') or {}
            literal = attributes.get('value', data.get('constant'))
            # Structural reduction of [] retains its exact aggregate birth
            # receipt but removes the AST object. Read that incumbent record,
            # not an inferred variable name or source-coordinate match.
            if (attributes.get('producer_kind') == 'aggregate'
                    and attributes.get('aggregate_kind') in {'dict', 'list', 'set', 'frozenset'}
                    and not attributes.get('aggregate_leaf_value_ids')
                    and type(literal) in {dict, list, set, frozenset} and not literal
                    and type(literal).__name__ == attributes['aggregate_kind']):
                kind = 'builtins.' + attributes['aggregate_kind']
        if kind is None:
            continue
        if kind not in context.kind_vocabulary:
            raise ValueError(f'native object context has no admitted constructor kind {kind}')
        value_id = int(data.get('value_id', node_id))
        cell = constructor_contract(node_id, value_id, kind, ())
        mutations.append(ControlSequenceMutation(value_id,
            'object_construct', (), int(node_id), argument_kind=kind, object_contract_cell=cell))
    reference_inputs = {int(value_id) for value_id, abi in
        (graph.graph.get('linked_value_abi') or {}).items()
        if abi.get('storage') == 'reference' and abi.get('dtype') == 'opaque_ref'}
    admitted = source_object_value_ids(graph, reference_inputs | {
        mutation.sequence_value_id for mutation in mutations})
    for node_id, data in graph.nodes(data=True):
        attrs = data.get('attributes') or {}
        leaves = tuple(attrs.get('aggregate_leaf_value_ids', ()))
        # Pair producer3 retains leaves1/0, while the collector's inherited
        # ledger on LoopResult11 is NOT a new tuple literal. Admit only the
        # actual tuple producer with all components already admitted handles.
        if (str(data.get('type', '')).casefold() != 'tuple' or not leaves
                or attrs.get('producer_kind') != 'aggregate'
                or not all(leaf in graph and int(graph.nodes[leaf].get('value_id', leaf)) in admitted for leaf in leaves)):
            continue
        value_id = int(data.get('value_id', node_id))
        members = tuple(int(graph.nodes[leaf].get('value_id', leaf)) for leaf in leaves)
        kind = 'builtins.tuple'
        if kind not in context.kind_vocabulary:
            raise ValueError('native tuple producer has no admitted constructor kind')
        cell = constructor_contract(node_id, value_id, kind, members)
        mutations.append(ControlSequenceMutation(value_id, 'object_construct', members,
            int(node_id), argument_kind=kind, object_contract_cell=cell))
    for node_id, data in graph.nodes(data=True):
        attributes = data.get('attributes') or {}
        source = native_materialization_source(graph, data)
        if source is None or source not in admitted:
            continue
        kind = 'builtins.' + attributes['aggregate_kind']
        if kind not in context.kind_vocabulary:
            raise ValueError(f'native object context has no admitted materialization kind {kind}')
        mutations.append(ControlSequenceMutation(int(data.get('value_id', node_id)),
            'object_materialize', (source,), int(node_id), argument_kind=kind))
    collections = source_object_collections(graph, admitted)
    admitted = source_object_value_ids(graph, admitted | set(collections))
    for node_id, data in graph.nodes(data=True):
        if data.get('op') not in {'update', 'extend'}:
            continue
        roles = dict((role, parent) for parent, role in data.get('parents') or ())
        destination, source = roles.get('operand'), roles.get('arg:0')
        if (destination not in graph or source not in graph
                or int(graph.nodes[destination].get('value_id', destination)) not in admitted
                or int(graph.nodes[source].get('value_id', source)) not in collections):
            continue
        effect_id = int(data.get('value_id', node_id))
        target_id = int(graph.nodes[destination].get('value_id', destination))
        source_id = int(graph.nodes[source].get('value_id', source))
        cell = _object_source_contract(graph, effect_id, 'update',
            {'destination_value_id': target_id, 'source_value_id': source_id,
             'effect_value_id': effect_id, 'operation': data['op']}, (node_id, destination, source))
        mutations.append(ControlSequenceMutation(target_id, 'object_update', (source_id,),
            effect_id, object_contract_cell=cell))
    return tuple(mutations)


def native_materialization_source(graph, data):
    """Use the admitted builtin identity and exact argument/producer ledger."""
    attributes = data.get('attributes') or {}
    kind = attributes.get('aggregate_kind')
    if (kind not in {'list', 'tuple', 'frozenset'}
            or attributes.get('producer_kind') != 'aggregate_materialization'
            or attributes.get('extraction_identity') != 'builtins.' + kind):
        return None
    sources = tuple(int(graph.nodes[parent].get('value_id', parent))
        for parent, role in data.get('parents') or ()
        if str(role).startswith('arg:') and parent in graph)
    if len(sources) != 1 or sources != tuple(attributes.get('materialized_source_value_ids', ())):
        return None
    return sources[0]


def source_object_collections(graph, admitted):
    """Read collection ownership from the actual loop/result/materializer edges.

    This admits storage, not an element dtype. The body must still produce
    opaque_ref members; no scalar or unresolved result is cast into a handle.
    """
    collections = {}
    from .dependency_cone import parent_record_dependents
    consumers_by_value = parent_record_dependents(graph)
    for node_id, data in graph.nodes(data=True):
        attributes = data.get('attributes') or {}
        parents = dict((str(role), int(parent)) for parent, role in data.get('parents') or ())
        if (str(data.get('type', '')).casefold() != 'loopresult'
                or attributes.get('result_kind') != 'collection'
                or set(parents) != {'value', 'control'}):
            continue
        materializer_id = attributes.get('materializer_node_id')
        iterable_id = attributes.get('collection_iterable_value_id')
        if materializer_id not in graph or iterable_id not in graph:
            continue
        materializer = graph.nodes[materializer_id].get('attributes') or {}
        kind = attributes.get('aggregate_kind')
        value_id = int(data.get('value_id', node_id))
        source_id = int(graph.nodes[iterable_id].get('value_id', iterable_id))
        control = graph.nodes[parents['control']]
        targets = (control.get('attributes') or {}).get('loop_target_bindings') or {}
        mapping_source = _mapping_items_source(graph, iterable_id)
        physical_source = (source_id if mapping_source is None else
                           int(graph.nodes[mapping_source].get('value_id', mapping_source)))
        paired = mapping_source is not None and len(targets) == 2
        generator = str(graph.nodes[materializer_id].get('type', '')).casefold() == 'generatorexp'
        consumers = [graph.nodes[user] for user in consumers_by_value.get(node_id, ())
                     if graph.nodes[user].get('op') in {'update', 'extend'}
                     and (node_id, 'arg:0') in (graph.nodes[user].get('parents') or ())]
        finite_update = generator and len(consumers) == 1
        if kind is None and finite_update:
            # A finite generator of scalar handles has no tuple literal whose
            # aggregate_kind could be inherited. Its admitted transport is
            # the same resident tuple collection used by paired generators.
            kind = 'tuple'
        if (physical_source not in admitted or kind not in {'list', 'tuple'}
                or not (materializer.get('extraction_identity') == 'builtins.' + kind or finite_update)
                or materializer.get('collection_owner_id') != node_id
                or tuple(materializer.get('materialized_source_value_ids', ())) != (node_id,)
                or attributes.get('loop_id') != parents['control'] or (len(targets) != 1 and not paired)):
            continue
        target_ids = tuple(targets.values())
        target = target_ids[0]
        if any(target not in graph for target in target_ids):
            continue
        element = parents['value']
        seen = set()
        while element in graph and element not in seen:
            seen.add(element)
            actual = (graph.nodes[element].get('attributes') or {}).get('structural_identity_actual_value_id')
            if actual is None or actual not in graph:
                break
            element = actual
        receipt = {'collection_value_id': value_id,
            'iterable_value_id': source_id, 'source_value_id': int(graph.nodes[parents['value']].get('value_id', parents['value'])),
            'member_value_id': int(graph.nodes[element].get('value_id', element)),
            'target_value_id': int(graph.nodes[target].get('value_id', target)),
            'loop_node_id': parents['control'], 'materializer_node_id': materializer_id,
            'kind': 'builtins.' + kind}
        if paired:
            receipt.update(source_iterable_value_id=physical_source,
                target_projection_ids=tuple(int(graph.nodes[target].get('value_id', target)) for target in target_ids))
        cells = (node_id, parents['control'], materializer_id, iterable_id,
                 parents['value'], *target_ids, *((mapping_source,) if paired else ()))
        cell = _object_source_contract(graph, value_id, 'collection', receipt, cells)
        collections[value_id] = {**object_source_contract(cell), 'contract_cell': cell}
    return collections


def source_object_value_ids(graph, births):
    """Propagate admitted handles through exact control-owned graph edges.

    A Phi of handles is a handle, not a new container arena. LoopResult also
    requires its initial value: its update alone says nothing about zero trips.
    Each rule is visited only as its incoming admitted identities arrive.
    """
    admitted = set(map(int, births))
    if not admitted:
        return frozenset()
    rules, dependents = {}, {}
    for node_id, data in graph.nodes(data=True):
        parents = dict((str(role), int(parent)) for parent, role in data.get('parents') or ())
        kind = str(data.get('type') or data.get('op') or '').casefold()
        dependencies = None
        materialized_source = native_materialization_source(graph, data)
        attributes = data.get('attributes') or {}
        owner = attributes.get('collection_owner_id')
        if (attributes.get('producer_kind') == 'sequence_materialization'
                and owner in graph
                and (graph.nodes[owner].get('attributes') or {}).get('materializer_node_id') == node_id
                and tuple(attributes.get('materialized_source_value_ids', ())) == (owner,)):
            dependencies = (int(owner),)
        identity_source = attributes.get('structural_identity_actual_value_id')
        if identity_source is not None:
            dependencies = (int(identity_source),)
        elif kind in {'comprehension', 'for'}:
            # The loop target is a member handle of its admitted iterable.
            targets = (data.get('attributes') or {}).get('loop_target_bindings') or {}
            iterable = parents.get('iter')
            mapping_source = _mapping_items_source(graph, iterable)
            source = mapping_source if mapping_source is not None and len(targets) == 2 else iterable
            if source is not None and (len(targets) == 1 or mapping_source is not None and len(targets) == 2):
                for target in targets.values():
                    if target not in graph or source not in graph:
                        continue
                    result = int(graph.nodes[target].get('value_id', target))
                    needed = int(graph.nodes[source].get('value_id', source))
                    rules[result] = {needed}
                    dependents.setdefault(needed, []).append(result)
            continue
        elif kind == 'tuple' and attributes.get('producer_kind') == 'aggregate':
            dependencies = tuple(attributes.get('aggregate_leaf_value_ids', ())) or None
        if materialized_source is not None:
            # This source is already a VALUE identity, unlike parent node keys.
            result = int(data.get('value_id', node_id))
            rules[result] = {materialized_source}
            dependents.setdefault(materialized_source, []).append(result)
            continue
        if kind == 'phi' and 'body' in parents and 'orelse' in parents:
            dependencies = (parents['body'], parents['orelse'])
        elif kind == 'loopresult' and 'control' in parents and 'value' in parents:
            control = graph.nodes[parents['control']]
            attributes = control.get('attributes') or {}
            pairs = [tuple(map(int, pair)) for pair in
                     (attributes.get('loop_carried_bindings') or {}).values()
                     if len(pair) == 2 and int(pair[1]) == parents['value']]
            if len(pairs) == 1 and not attributes.get('loop_break_bindings'):
                dependencies = pairs[0]
        if dependencies is None:
            continue
        if any(parent not in graph for parent in dependencies):
            # A planned unit can retain an authored identity receipt while
            # its producer belongs to the enclosing graph. No local edge,
            # no local inference; the whole graph/call ABI owns that fact.
            continue
        result = int(data.get('value_id', node_id))
        needed = {int(graph.nodes[p].get('value_id', p)) for p in dependencies}
        rules[result] = needed
        for dependency in needed:
            dependents.setdefault(dependency, []).append(result)
    pending = list(admitted)
    while pending:
        value_id = pending.pop()
        for result in dependents.get(value_id, ()):
            rules[result].discard(value_id)
            if not rules[result] and result not in admitted:
                admitted.add(result)
                pending.append(result)
    return frozenset(admitted)


def context_values(context: SSAObjectContext) -> tuple[SSAValue, ...]:
    """One exact native ABI; duplicate owner references share one cell."""
    values = {}
    for descriptor in (context.objects, context.members):
        for value in _storage_values(descriptor):
            values[value.id] = value
        status = SSAValue(descriptor.status_address_id, 'int', shape=(1,))
        values[status.id] = status
    owner = SSAValue(context.objects.heap_owner_value_id, 'ptr', shape=(1,),
                     accounting={'heap_owner': True})
    counter = SSAValue(context.next_handle_value_id, 'int64', shape=(1,))
    values[owner.id], values[counter.id] = owner, counter
    return tuple(values.values())


def validate_context_function(module, function_name: str) -> None:
    """A context receipt must still agree with its actual tables and ABI."""
    function = module.functions[function_name]
    receipt = function.metadata.get('object_context')
    source_context = function.metadata.get('native_object_context')
    if source_context is not None:
        if receipt is not None and receipt != source_context:
            raise ValueError('source and helper object contexts disagree')
        receipt = source_context
        if module.metadata.get('native_object_context') != receipt:
            raise ValueError('source object context differs from accepted module context')
    if receipt is None:
        return
    table = module.sequence_tables.get(function_name)
    if table is None:
        raise ValueError('object context has no live sequence table')
    objects = table.by_id(int(receipt['objects']['sequence_id']))
    members = table.by_id(int(receipt['members']['sequence_id']))
    if objects is None or members is None:
        raise ValueError('object context has a missing storage descriptor')
    context = SSAObjectContext(receipt['storage_identity'], objects, members,
        receipt['next_handle_value_id'], tuple(receipt['kind_vocabulary']),
        tuple(receipt.get('type_vocabulary', ())))
    if context.to_mapping() != receipt:
        raise ValueError('object context disagrees with its live storage descriptors')
    formals = {value.id: value for value in function.args}
    for value in context_values(context):
        actual = formals.get(value.id)
        if actual is None or (actual.dtype, actual.shape) != (value.dtype, value.shape):
            raise ValueError(f'object context has incompatible storage formal {value.id}')
        for key in ('heap_owner', 'heap_owner_value_id', 'heap_element_dtype'):
            if (actual.accounting or {}).get(key) != (value.accounting or {}).get(key):
                raise ValueError(f'object context has incompatible ownership formal {value.id}')
        if source_context is not None and (
                actual.accounting.get('native_object_context_value_id') != value.id
                or actual.accounting.get('native_object_context_storage_identity') != context.storage_identity):
            raise ValueError(f'source object context has stale role formal {value.id}')


def source_object_definitions(function):
    definitions = {}
    for block in function.blocks.values():
        for instruction in block.instrs:
            if instruction.res is not None:
                definitions.setdefault(instruction.res.id, []).append(instruction)
    return definitions


def native_object_requirement_sources(module, function_name):
    """Existing call-graph reachability determines the lease's payload needs."""
    operations = {'length', 'get', 'set', 'materialize', 'materialize_frozen', 'collect', 'update', 'extend', 'identity_all'}
    return tuple(name for name in module.reachable_functions(function_name)
        if module.functions[name].metadata.get('opaque_object_helper', {}).get('operation') in operations
        or any((i.attributes.get('source_object_operation') or (None,))[0] in operations
               for block in module.functions[name].blocks.values() for i in block.instrs))


def publish_native_object_requirements(module):
    if module is None:
        return
    for name, function in module.functions.items():
        if (function.metadata.get('native_object_context') is not None
                or function.metadata.get('object_context') is not None):
            sources = native_object_requirement_sources(module, name)
            function.metadata['native_object_requirement_sources'] = sources
            function.metadata['native_object_requires_member_storage'] = bool(sources)


def validate_native_object_requirements(module, function_name):
    function = module.functions[function_name]
    if 'native_object_requirement_sources' in function.metadata:
        sources = native_object_requirement_sources(module, function_name)
        if (function.metadata['native_object_requirement_sources'] != sources
                or function.metadata.get('native_object_requires_member_storage') != bool(sources)):
            raise ValueError('native object payload requirements disagree with live call graph')


def validate_source_object_instruction(module, function_name, instruction, *, _definitions=None, _helpers=None):
    """Check authored object receipts against the live helper and context."""
    validate_native_collection_instruction(module, function_name, instruction, _definitions=_definitions)
    test = instruction.attributes.get('source_object_type_test')
    if test is not None:
        function = module.functions[function_name]
        context = function.metadata.get('native_object_context')
        definitions = source_object_definitions(function) if _definitions is None else _definitions
        if (test not in function.metadata.get('native_object_type_tests', ())
                or test['context'] != context or test['intrinsic'] != 'builtins.isinstance'
                or instruction.op != 'Ne' or instruction.res is None
                or instruction.res.id != test['source_value_id'] or instruction.res.dtype != 'bool'):
            raise ValueError('source object type test has stale source/context receipt')
        vocabulary = tuple(context.get('type_vocabulary', ()))
        accepted = tuple(test['accepted_types'])
        if not accepted or any(token not in vocabulary for token in accepted):
            raise ValueError('source object type test has unadmitted targets')
        bits = sum(1 << vocabulary.index(token) for token in set(accepted))
        query, = definitions.get(test['mask_value_id'], ())
        token, = definitions.get(test['token_value_id'], ())
        selected, = definitions.get(test['selected_value_id'], ())
        zero, = definitions.get(instruction.args[1].id, ())
        if (test['mask'] != bits or query.attributes.get('source_object_operation') !=
                ('type_mask', test['reference_value_id'], (), None)
                or token.op != 'Const' or token.attributes.get('value') != bits
                or token.res.dtype != 'int64' or selected.op != 'And'
                or query.res.dtype != 'int64' or selected.res.dtype != 'int64'
                or query.res.shape or selected.res.shape
                or tuple(value.id for value in selected.args) != (query.res.id, token.res.id)
                or instruction.args[0].id != selected.res.id or zero.op != 'Const'
                or zero.attributes.get('value') != 0 or zero.res.dtype != 'int64' or instruction.res.shape):
            raise ValueError('source object type test differs from live evidence/mask/predicate')
        return
    birth = instruction.attributes.get('source_object_constructor')
    operation = instruction.attributes.get('source_object_operation')
    if birth is None and operation is None:
        return
    function = module.functions[function_name]
    context = function.metadata.get('native_object_context')
    interface = instruction.attributes.get('opaque_object_operation')
    if context is None or interface is None:
        raise ValueError('source object operation has no accepted context/helper')
    from .ssa_object_operations import validate_object_call
    validate_object_call(module, instruction, _helpers=_helpers)
    if interface['storage_identity'] != context['storage_identity']:
        raise ValueError('source object operation uses a different storage domain')
    bindings = dict(instruction.attributes['opaque_object_bindings'])
    actuals = {value.id: value for value in instruction.args}
    for role in interface['context_formal_ids']:
        actual = actuals[bindings[role]]
        if (actual.accounting.get('native_object_context_value_id') != role
                or actual.accounting.get('native_object_context_storage_identity') != context['storage_identity']):
            raise ValueError('source object operation has stale context bindings')
    if birth is not None:
        from .identity_concordance import identity_book
        accepted = object_source_contract(birth.get('contract_cell'), book=identity_book(module))
        if any(birth.get(key) != value for key, value in accepted.items()):
            raise ValueError('source constructor differs from its authoritative concordance cell')
        if (operation is not None or interface['operation'] != 'construct'
                or birth['context'] != context or birth['initial_extent'] != 0
                or birth['kind'] not in context['kind_vocabulary']):
            raise ValueError('source constructor has stale kind/storage receipt')
        operands = interface['operand_formal_ids']
        if len(operands) != 1:
            raise ValueError('source constructor requires its one declared kind token')
        token_id = bindings[operands[0]]
        definitions = source_object_definitions(function) if _definitions is None else _definitions
        tokens = definitions.get(token_id, ())
        if (len(tokens) != 1 or tokens[0].op != 'Const'
                or tokens[0].attributes.get('value') != tuple(context['kind_vocabulary']).index(birth['kind']) + 1):
            raise ValueError('source constructor kind token differs from its live producer')
        if accepted['member_value_ids']:
            from .identity_concordance import resolved_concordant_alias_bindings
            from .concordance_declarations import PLANNING_VALUE
            aliases = resolved_concordant_alias_bindings(str(function.name),
                function.metadata.get('value_aliases', {}),
                page=identity_book(module).page(PLANNING_VALUE))
            stores = [i for b in function.blocks.values() for i in b.instrs
                if (i.attributes.get('source_object_operation') or (None, None))[:2]
                == ('set', accepted['source_value_id'])]
            if len(stores) != len(accepted['member_value_ids']):
                raise ValueError('native tuple fields differ from their concordance arity')
            for ordinal, (store, member) in enumerate(zip(stores, accepted['member_value_ids'], strict=True), 1):
                key_id, stored_member = store.attributes['source_object_operation'][2]
                casts = definitions.get(key_id, ())
                if (stored_member != aliases.get(member, member) or len(casts) != 1 or casts[0].op != 'Cast'
                        or casts[0].attributes.get('binding') != 'object_context_position'):
                    raise ValueError('native tuple field differs from its concordance component')
                literal, = definitions.get(casts[0].args[0].id, ())
                if literal.op != 'Const' or literal.attributes.get('value') != ordinal:
                    raise ValueError('native tuple field has a different declared position')
    elif operation[0] != interface['operation']:
        raise ValueError('source object operation disagrees with its live helper')
    materialization = instruction.attributes.get('source_object_materialization')
    if materialization is not None:
        expected_operation = 'materialize_frozen' if materialization['kind'] == 'builtins.frozenset' else 'materialize'
        if (interface['operation'] != expected_operation or materialization['context'] != context
                or materialization['kind'] not in {'builtins.list', 'builtins.tuple', 'builtins.frozenset'}
                or instruction.res.accounting.get('native_object_source_kind') != materialization['kind']):
            raise ValueError('source materialization has stale kind/storage receipt')
        operand, = interface['operand_formal_ids']
        definitions = source_object_definitions(function) if _definitions is None else _definitions
        tokens = definitions.get(bindings[operand], ())
        if (len(tokens) != 1 or tokens[0].op != 'Const'
                or tokens[0].attributes.get('value') != tuple(context['kind_vocabulary']).index(materialization['kind']) + 1):
            raise ValueError('source materialization kind token differs from its live producer')
    collection = instruction.attributes.get('source_object_collection')
    if collection is not None:
        from .identity_concordance import identity_book
        accepted = object_source_contract(collection.get('contract_cell'), book=identity_book(module))
        if {key: value for key, value in collection.items() if key != 'contract_cell'} != accepted:
            raise ValueError('source collection differs from its authoritative concordance cell')
        if (interface['operation'] != 'collect' or collection not in function.metadata.get('native_object_collections', ())
                or instruction.res.id != collection['collection_value_id']
                or instruction.res.accounting.get('native_object_source_kind') != collection['kind']):
            raise ValueError('source collection has stale ownership/helper receipt')
        reference = actuals[bindings[interface['reference_formal_id']]]
        if reference.id != collection.get('source_iterable_value_id', collection['iterable_value_id']):
            raise ValueError('source collection has stale iterable binding')
        operand, = interface['operand_formal_ids']
        definitions = source_object_definitions(function) if _definitions is None else _definitions
        tokens = definitions.get(bindings[operand], ())
        if (len(tokens) != 1 or tokens[0].op != 'Const'
                or tokens[0].attributes.get('value') != tuple(context['kind_vocabulary']).index(collection['kind']) + 1):
            raise ValueError('source collection kind differs from its live token')
    update = instruction.attributes.get('source_object_update')
    if update is not None:
        from .identity_concordance import identity_book
        accepted = object_source_contract(update['contract_cell'], book=identity_book(module))
        if (interface['operation'] != accepted['operation']
                or operation != (accepted['operation'], accepted['destination_value_id'], (accepted['source_value_id'],), None)):
            raise ValueError('source update differs from its authoritative concordance cell')
    projection = instruction.attributes.get('native_pair_projection')
    reduction = instruction.attributes.get('source_object_identity_reduction')
    if reduction is not None:
        from .identity_concordance import identity_book
        accepted = object_source_contract(reduction['contract_cell'], book=identity_book(module))
        left, right = accepted['source_value_ids']
        if operation != ('identity_all', left, (right,), accepted['result_value_id']):
            raise ValueError('native identity reduction differs from its concordance edges')
    if projection is not None:
        from .identity_concordance import identity_book
        accepted = object_source_contract(projection['contract_cell'], book=identity_book(module))
        key, value = accepted['target_projection_ids']
        if operation != ('get', accepted['source_iterable_value_id'], (key,), value):
            raise ValueError('native pair projection differs from its concordance binding')


def validate_native_collection_instruction(module, function_name, instruction, *, _definitions=None):
    row = instruction.attributes.get('native_collection_row')
    if row is None:
        return
    function = module.functions[function_name]
    context = function.metadata.get('native_object_context')
    source = {key: value for key, value in row.items() if key not in {'action', 'base_value_id', 'index_value_id'}}
    if context is None or source not in function.metadata.get('native_object_collections', ()):
        raise ValueError('native collection row has stale source ownership')
    definitions = source_object_definitions(function) if _definitions is None else _definitions
    def definition(value_id, operation):
        found = definitions.get(value_id, ())
        if len(found) != 1 or found[0].op != operation:
            raise ValueError(f'native collection row has no exact {operation} producer for {value_id}')
        return found[0]
    birth = definition(source['collection_value_id'], 'Call')
    if birth.attributes.get('source_object_collection') != source or birth.attributes.get('opaque_object_operation', {}).get('operation') != 'collect':
        raise ValueError('native collection row is not owned by a live collect call')
    base = definition(row['base_value_id'], 'Load')
    cell = definition(base.args[0].id, 'GetElementPtr')
    if cell.args[0].id != context['members']['length_address_id']:
        raise ValueError('native collection row base does not read its context extent')
    index = definition(row['index_value_id'], 'Phi')
    def constant_or_cast(value):
        producers = definitions.get(value.id, ())
        if len(producers) != 1 or producers[0].op not in {'Const', 'Cast'}:
            raise ValueError('native collection induction has no exact constant/cast producer')
        return producers[0]
    start = constant_or_cast(index.args[0])
    initial = definition(start.args[0].id, 'Const') if start.op == 'Cast' else start
    if initial.attributes.get('value') != 0 or index.res.dtype != 'int64':
        raise ValueError('native collection row does not use its dense zero-based induction')
    increment = definition(index.args[1].id, 'Add')
    if increment.args[0].id != index.res.id:
        raise ValueError('native collection row has stale induction recurrence')
    unit = constant_or_cast(increment.args[1])
    unit_constant = definition(unit.args[0].id, 'Const') if unit.op == 'Cast' else unit
    if unit_constant.attributes.get('value') != 1:
        raise ValueError('native collection row does not have unit-step recurrence')
    if index.attributes.get('source_loop_node_id') != source['loop_node_id']:
        raise ValueError('native collection row induction belongs to a different loop')
    predicate = definition(index.res.accounting.get('native_collection_predicate_value_id'), 'Lt')
    if predicate.args[0].id != index.res.id:
        raise ValueError('native collection predicate uses a different induction')
    extent = definition(predicate.args[1].id, 'Sub')
    if extent.args[1].id != base.res.id:
        raise ValueError('native collection extent differs from its retained row base')
    end = definition(extent.args[0].id, 'Load')
    if end.args[0].id != base.args[0].id:
        raise ValueError('native collection extent reads a different context cell')
    if row['action'] == 'address':
        if instruction.op != 'GetElementPtr' or instruction.args[0].id != context['members']['column_value_ids'][2]:
            raise ValueError('native collection address uses a different storage column')
        offset = definition(instruction.args[1].id, 'Add')
        if tuple(value.id for value in offset.args) != (row['base_value_id'], row['index_value_id']):
            raise ValueError('native collection address differs from its exact base/index')
    elif row['action'] in {'iterate', 'publish'}:
        address = definition(instruction.args[0 if row['action'] == 'iterate' else 1].id, 'GetElementPtr')
        expected = {**row, 'action': 'address'}
        if address.attributes.get('native_collection_row') != expected:
            raise ValueError('native collection member uses a different row address')
        value = instruction.res if row['action'] == 'iterate' else instruction.args[0]
        if (instruction.op != ('Load' if row['action'] == 'iterate' else 'Store')
                or value.dtype != 'opaque_ref' or value.shape
                or (row['action'] == 'iterate' and value.id != source['target_value_id'])
                or (row['action'] == 'publish' and value.id != source['member_value_id'])):
            raise ValueError('native collection member has unresolved/non-reference identity')
    else:
        raise ValueError('native collection row has an undeclared action')


def lower_object_context(context: SSAObjectContext, *, prefix: str = 'native_object') -> IRModule:
    """Create initialize/construct/kind/get/set/length/release native helpers.

    Initialization requires zero pointer cells/counters and a null owner supplied
    by the caller; it is not a reset of a live context. Tables begin empty at their
    declared capacities. Contexts are invocation-owned, not safe for concurrent
    mutation; distinct contexts may execute concurrently. Allocation failures
    use the existing allocator's fail-fast policy. Construct returns zero for
    invalid kinds, exhausted handles, collision or invalid storage extents.
    """
    values = context_values(context)
    by_id = {value.id: value for value in values}
    module = IRModule(functions={})

    def add(lowering):
        if not lowering.complete:
            raise ValueError(f'object context storage lowering refused: {lowering.shortfalls}')
        for function in lowering.functions:
            if function.name in module.functions:
                raise ValueError(f'duplicate object context helper {function.name}')
            function.metadata['generated_helper_provenance'] = 'object-context'
            _publish_generated_helper(function)
            module.functions[function.name] = function
        return lowering.functions[0]

    allocate = [add(lower_sequence_allocate(descriptor,
        function_name=f'{prefix}_{label}_allocate'))
        for label, descriptor in (('objects', context.objects), ('members', context.members))]
    insert = add(lower_sequence_insert(context.objects, function_name=f'{prefix}_object_insert'))
    from dataclasses import replace
    def object_projection(column):
        return replace(context.objects,
            column_value_ids=(context.objects.column_value_ids[0], context.objects.column_value_ids[column]),
            column_dtypes=('opaque_ref', 'int64'), column_shapes=((), ()))
    kind_lookup = add(lower_table_lookup(object_projection(1), function_name=f'{prefix}_kind_lookup'))
    type_lookup = (add(lower_table_lookup(object_projection(2), function_name=f'{prefix}_type_lookup'))
                   if context.type_vocabulary else None)
    member_lookup = add(lower_table_lookup(context.members, function_name=f'{prefix}_member_lookup'))
    store = add(lower_table_store(context.members, function_name=f'{prefix}_member_store'))

    def call(builder, block, callee, extra, result=None):
        bindings = {**by_id, **extra}
        builder.emit(block, 'Call', [bindings[value.id] for value in callee.args], result,
            attributes={'callee': callee.name,
                        'callee_input_ids': tuple(value.id for value in callee.args)})

    def arguments(callee, descriptor):
        storage = {value.id for value in _storage_values(descriptor)}
        storage.update((descriptor.status_address_id, descriptor.heap_owner_value_id))
        return [value for value in callee.args if value.id not in storage]

    def load_cell(builder, block, value):
        zero = builder.const(block, 0, dtype='int64')
        address, loaded = builder.fresh('ptr'), builder.fresh(value.dtype)
        builder.emit(block, 'GetElementPtr', [value, zero], address)
        builder.emit(block, 'Load', [address], loaded)
        return address, loaded

    def finish(label, builder, operands=(), reference=None, operation=None):
        name = f'{prefix}_{label}'
        # Exact IDs, not reconstructed signature positions, name the ABI.
        # Each function owns its SSA occurrences. Sharing mutable SSAValue
        # objects between functions lets one frame's accounting overwrite the
        # other's history even though their immutable context view agrees.
        local_values = context_values(context)
        local_by_id = {value.id: value for value in local_values}
        for block in builder.blocks.values():
            for instruction in block.instrs:
                instruction.args = [local_by_id.get(value.id, value) for value in instruction.args]
        function = Function(name, [*local_values, *operands], builder.blocks,
            metadata={'object_context': context.to_mapping(),
                      'generated_helper_provenance': 'object-context',
                      'parameter_names': tuple((f'cell_{value.id}', value.id)
                                               for value in (*values, *operands)),
                      'sequence_array_argument_ids': tuple(value.id for value in values
                          if value.dtype == 'ptr' or value.shape)})
        _publish_generated_helper(function)
        module.functions[name] = function
        from .frame_identity_book import _frame_register_sequence

        table = SSASequenceTable(owner=name)
        _frame_register_sequence(function, table, context.objects)
        _frame_register_sequence(function, table, context.members)
        module.sequence_tables[name] = table
        returned = [instruction.args[0] for block in function.blocks.values()
                    for instruction in block.instrs if instruction.op == 'Ret' and instruction.args]
        if returned:
            function.metadata['return_value'] = returned[0]
            function.metadata['named_outputs'] = (('result', returned[0].id),)
        if operation is not None:
            declare_object_helper(module, name, operation, storage_identity=context.storage_identity,
                reference_formal_id=None if reference is None else reference.id,
                context_formal_ids=tuple(value.id for value in values),
                operand_formal_ids=tuple(value.id for value in operands if value is not reference),
                kind_vocabulary=context.kind_vocabulary if operation is ObjectOperation.KIND else ())
        validate_context_function(module, name)
        return function

    builder = _Builder(0)
    entry = builder.block('entry')
    for helper in allocate:
        call(builder, entry, helper, {})
    counter_slot, _ = load_cell(builder, entry, by_id[context.next_handle_value_id])
    builder.emit(entry, 'Store', [builder.const(entry, 1, dtype='int64'), counter_slot])
    builder.emit(entry, 'Ret', [])
    finish('initialize', builder)

    builder = _Builder(0)
    kind = builder.fresh('int64')
    entry, insert_block = builder.block('entry'), builder.block('insert')
    publish, reject, complete = builder.block('publish'), builder.block('reject'), builder.block('complete')
    counter_slot, candidate = load_cell(builder, entry, by_id[context.next_handle_value_id])
    zero, one = builder.const(entry, 0, dtype='int64'), builder.const(entry, 1, dtype='int64')
    valid = []
    for op, left, right in (
        ('Gt', candidate, zero),
        ('Lt', candidate, builder.const(entry, 2**63 - 1, dtype='int64')),
        ('Ge', kind, one),
        ('Le', kind, builder.const(entry, len(context.kind_vocabulary), dtype='int64'))):
        result = builder.fresh('bool')
        builder.emit(entry, op, [left, right], result)
        valid.append(result)
    accepted = valid[0]
    for condition in valid[1:]:
        result = builder.fresh('bool')
        builder.emit(entry, 'LAnd', [accepted, condition], result)
        accepted = result
    builder.cond(entry, accepted, insert_block, reject)
    handle = builder.fresh('opaque_ref')
    builder.emit(insert_block, 'Cast', [candidate], handle,
        attributes={'binding': 'object_context_mint', 'storage_identity': context.storage_identity})
    row_ids = insert.metadata['sequence_row_argument_ids']
    status = builder.fresh('int')
    row = {row_ids[0]: handle, row_ids[1]: kind}
    if context.type_vocabulary:
        mask = builder.const(insert_block, 0, dtype='int64')
        classes = object_type_classes(context.type_vocabulary)
        for index, label in enumerate(context.kind_vocabulary, 1):
            try:
                actual_class, = object_type_classes((label,))
            except ValueError:
                raise ValueError('native births with type evidence require builtin kind identities')
            bits = sum(1 << bit for bit, target in enumerate(classes) if issubclass(actual_class, target))
            matched, wide, weighted, combined = (builder.fresh(dtype) for dtype in ('bool', 'int64', 'int64', 'int64'))
            builder.emit(insert_block, 'Eq', [kind, builder.const(insert_block, index, dtype='int64')], matched)
            builder.emit(insert_block, 'Cast', [matched], wide)
            builder.emit(insert_block, 'Mul', [wide, builder.const(insert_block, bits, dtype='int64')], weighted)
            builder.emit(insert_block, 'Or', [mask, weighted], combined)
            mask = combined
        row[row_ids[2]] = mask
    call(builder, insert_block, insert, row, status)
    success = builder.fresh('bool')
    builder.emit(insert_block, 'Eq', [status, builder.const(insert_block, 1)], success)
    builder.cond(insert_block, success, publish, reject)
    next_handle = builder.fresh('int64')
    builder.emit(publish, 'Add', [candidate, one], next_handle)
    builder.emit(publish, 'Store', [next_handle, counter_slot])
    builder.branch(publish, complete)
    failed = builder.const(reject, 0, dtype='opaque_ref')
    builder.branch(reject, complete)
    result = builder.fresh('opaque_ref')
    builder.emit(complete, 'Phi', [handle, failed], result,
        attributes={'incoming_blocks': (publish.name, reject.name)})
    builder.emit(complete, 'Ret', [result])
    finish('construct', builder, (kind,), operation=ObjectOperation.CONSTRUCT)

    queries = [('kind', kind_lookup, ObjectOperation.KIND), ('get', member_lookup, ObjectOperation.GET)]
    if type_lookup is not None:
        queries.append(('type_mask', type_lookup, ObjectOperation.TYPE_MASK))
    for label, helper, operation in queries:
        builder = _Builder(0)
        entry = builder.block('entry')
        operands = tuple(builder.fresh(value.dtype) for value in arguments(helper,
            object_projection(2 if label == 'type_mask' else 1) if label != 'get' else context.members))
        returned = builder.fresh('opaque_ref' if label == 'get' else 'int64')
        call(builder, entry, helper, dict(zip((v.id for v in arguments(helper,
            object_projection(2 if label == 'type_mask' else 1) if label != 'get' else context.members)), operands)), returned)
        builder.emit(entry, 'Ret', [returned])
        finish(label, builder, operands, reference=operands[0], operation=operation)

    builder = _Builder(0)
    entry, write = builder.block('entry'), builder.block('write')
    reject, complete = builder.block('reject'), builder.block('complete')
    queries = arguments(store, context.members)
    operands = tuple(builder.fresh(value.dtype) for value in queries)
    kind = builder.fresh('int64')
    kind_query = arguments(kind_lookup, context.objects)[0]
    call(builder, entry, kind_lookup, {kind_query.id: operands[0]}, kind)
    live = builder.fresh('bool')
    builder.emit(entry, 'Gt', [kind, builder.const(entry, 0, dtype='int64')], live)
    builder.cond(entry, live, write, reject)
    call(builder, write, store, dict(zip((value.id for value in queries), operands)))
    _, status = load_cell(builder, write, by_id[context.members.status_address_id])
    inserted, updated, succeeded = builder.fresh('bool'), builder.fresh('bool'), builder.fresh('bool')
    builder.emit(write, 'Eq', [status, builder.const(write, 1)], inserted)
    builder.emit(write, 'Eq', [status, builder.const(write, 3)], updated)
    builder.emit(write, 'LOr', [inserted, updated], succeeded)
    builder.branch(write, complete)
    failed = builder.const(reject, 0, dtype='bool')
    builder.branch(reject, complete)
    result = builder.fresh('bool')
    builder.emit(complete, 'Phi', [succeeded, failed], result,
        attributes={'incoming_blocks': (write.name, reject.name)})
    builder.emit(complete, 'Ret', [result])
    finish('set', builder, operands, reference=operands[0], operation=ObjectOperation.SET)

    builder = _Builder(0)
    handle = builder.fresh('opaque_ref')
    entry, header, body = builder.block('entry'), builder.block('header'), builder.block('body')
    latch, complete = builder.block('latch'), builder.block('complete')
    zero, one = builder.const(entry, 0, dtype='int64'), builder.const(entry, 1, dtype='int64')
    _, length = load_cell(builder, entry, by_id[context.members.length_address_id])
    builder.branch(entry, header)
    index, count, next_index, next_count = (builder.fresh('int64') for _ in range(4))
    for value, update in ((index, next_index), (count, next_count)):
        builder.emit(header, 'Phi', [zero, update], value,
            attributes={'incoming_blocks': (entry.name, latch.name)})
    more = builder.fresh('bool')
    builder.emit(header, 'Lt', [index, length], more)
    builder.cond(header, more, body, complete)
    address, row_owner, matches, increment = (builder.fresh(dtype)
        for dtype in ('ptr', 'opaque_ref', 'bool', 'int64'))
    builder.emit(body, 'GetElementPtr', [by_id[context.members.column_value_ids[0]], index], address)
    builder.emit(body, 'Load', [address], row_owner)
    builder.emit(body, 'Eq', [row_owner, handle], matches)
    builder.emit(body, 'Cast', [matches], increment)
    builder.emit(body, 'Add', [count, increment], next_count)
    builder.branch(body, latch)
    builder.emit(latch, 'Add', [index, one], next_index)
    builder.branch(latch, header)
    builder.emit(complete, 'Ret', [count])
    finish('length', builder, (handle,), reference=handle, operation=ObjectOperation.LENGTH)

    # tuple(value)/list(value) stamp a whole shallow container in ONE native
    # invocation. Do not dispatch get/set/construct through Python per item,
    # or run the unique member-table scan for every copied edge. Reserve once,
    # then write the new owner's disjoint rows directly. Stable handles, not
    # payload addresses, retain aliases/backedges through reserve relocation.
    materialize_kinds = {'builtins.list', 'builtins.tuple', 'builtins.frozenset'} & set(context.kind_vocabulary)
    if materialize_kinds:
        reserve = add(lower_sequence_reserve(context.members,
            function_name=f'{prefix}_materialize_reserve'))
        for label, operation in (('materialize', ObjectOperation.MATERIALIZE),
                                 ('collect', ObjectOperation.COLLECT),
                                 ('materialize_frozen', ObjectOperation.MATERIALIZE_FROZEN)):
            builder = _Builder(0)
            source, requested = builder.fresh('opaque_ref'), builder.fresh('int64')
            blocks = {label: builder.block(label) for label in (
                'entry', 'accepted', 'same', 'count_entry', 'count_header', 'count_body',
                'count_latch', 'reserve', 'construct', 'copy_entry', 'copy_header',
                'copy_body', 'copy_write', 'copy_latch', 'publish', 'reject', 'complete')}
            blocks['copy_stamp'] = builder.block('copy_stamp')
            entry, reject = blocks['entry'], blocks['reject']
            zero, one = builder.const(entry, 0, dtype='int64'), builder.const(entry, 1, dtype='int64')
            kind = builder.fresh('int64')
            call(builder, entry, module.functions[f'{prefix}_kind'],
                {module.functions[f'{prefix}_kind'].metadata['opaque_object_helper']['reference_formal_id']: source}, kind)
            def kind_is(block, value, labels):
                predicates = []
                for label in labels:
                    if label in context.kind_vocabulary:
                        predicate = builder.fresh('bool')
                        builder.emit(block, 'Eq', [value, builder.const(block,
                            context.kind_vocabulary.index(label) + 1, dtype='int64')], predicate)
                        predicates.append(predicate)
                if not predicates:
                    return builder.const(block, 0, dtype='bool')
                combined = predicates[0]
                for predicate in predicates[1:]:
                    merged = builder.fresh('bool')
                    builder.emit(block, 'LOr', [combined, predicate], merged)
                    combined = merged
                return combined
            source_valid = kind_is(entry, kind, ('builtins.dict', 'builtins.list',
                'builtins.tuple', 'builtins.set', 'builtins.frozenset'))
            target_valid = kind_is(entry, requested, ('builtins.frozenset',)
                if operation is ObjectOperation.MATERIALIZE_FROZEN else ('builtins.list', 'builtins.tuple'))
            valid = builder.fresh('bool')
            builder.emit(entry, 'LAnd', [source_valid, target_valid], valid)
            positional = kind_is(entry, kind, ('builtins.list', 'builtins.tuple'))
            dictionary = kind_is(entry, kind, ('builtins.dict',))
            builder.cond(entry, valid, blocks['accepted'], reject)
            source_tuple = kind_is(blocks['accepted'], kind, ('builtins.tuple',))
            target_tuple = kind_is(blocks['accepted'], requested, ('builtins.tuple',))
            same = builder.fresh('bool')
            builder.emit(blocks['accepted'], 'LAnd', [source_tuple, target_tuple], same)
            if operation is ObjectOperation.COLLECT:
                same = builder.const(blocks['accepted'], 0, dtype='bool')
            builder.cond(blocks['accepted'], same, blocks['same'], blocks['count_entry'])
            builder.branch(blocks['same'], blocks['complete'])
            start = blocks['count_entry']
            length_slot, original_length = load_cell(builder, start, by_id[context.members.length_address_id])
            nonnegative, fits, bounded, extent_valid, all_valid = (builder.fresh('bool') for _ in range(5))
            builder.emit(start, 'Ge', [original_length, zero], nonnegative)
            builder.emit(start, 'Le', [original_length, by_id[context.members.capacity_value_id]], fits)
            builder.emit(start, 'Le', [by_id[context.members.capacity_value_id],
                builder.const(start, (2**63 - 1 - 8) // 8, dtype='int64')], bounded)
            builder.emit(start, 'LAnd', [nonnegative, fits], extent_valid)
            builder.emit(start, 'LAnd', [extent_valid, bounded], all_valid)
            builder.cond(start, all_valid, blocks['count_header'], reject)
            index, count, next_index, next_count = (builder.fresh('int64') for _ in range(4))
            header, body, latch = blocks['count_header'], blocks['count_body'], blocks['count_latch']
            for value, update in ((index, next_index), (count, next_count)):
                builder.emit(header, 'Phi', [zero, update], value,
                    attributes={'incoming_blocks': (start.name, latch.name)})
            more = builder.fresh('bool')
            builder.emit(header, 'Lt', [index, original_length], more)
            builder.cond(header, more, body, blocks['reserve'])
            def row_value(block, column, row, dtype='opaque_ref'):
                address, value = builder.fresh('ptr'), builder.fresh(dtype)
                builder.emit(block, 'GetElementPtr', [by_id[column], row], address)
                builder.emit(block, 'Load', [address], value)
                return value
            owner = row_value(body, context.members.column_value_ids[0], index)
            matches, increment = builder.fresh('bool'), builder.fresh('int64')
            builder.emit(body, 'Eq', [owner, source], matches)
            builder.emit(body, 'Cast', [matches], increment)
            builder.emit(body, 'Add', [count, increment], next_count)
            builder.branch(body, latch)
            builder.emit(latch, 'Add', [index, one], next_index)
            builder.branch(latch, header)
            required, reserved = builder.fresh('int64'), builder.fresh('int')
            builder.emit(blocks['reserve'], 'Add', [original_length, count], required)
            call(builder, blocks['reserve'], reserve,
                {reserve.metadata['required_capacity_value_id']: required}, reserved)
            allocated = builder.fresh('bool')
            builder.emit(blocks['reserve'], 'Lt', [reserved,
                builder.const(blocks['reserve'], 2, dtype='int')], allocated)
            builder.cond(blocks['reserve'], allocated, blocks['construct'], reject)
            constructor = module.functions[f'{prefix}_construct']
            destination = builder.fresh('opaque_ref')
            call(builder, blocks['construct'], constructor,
                {constructor.metadata['opaque_object_helper']['operand_formal_ids'][0]: requested}, destination)
            live = builder.fresh('bool')
            builder.emit(blocks['construct'], 'Gt', [destination,
                builder.const(blocks['construct'], 0, dtype='opaque_ref')], live)
            frozen_complete = None
            if operation is ObjectOperation.MATERIALIZE_FROZEN:
                # Keep the positional bulk helper free of per-member stores.
                # Keyed immutable construction has a separate accepted ABI.
                frozen_entry, frozen_header, frozen_body, frozen_write, frozen_latch, frozen_complete = (
                    builder.block(name) for name in ('frozen_entry', 'frozen_header', 'frozen_body',
                                                      'frozen_write', 'frozen_latch', 'frozen_complete'))
                builder.cond(blocks['construct'], live, frozen_entry, reject)
                builder.branch(frozen_entry, frozen_header)
                frozen_index, frozen_next = builder.fresh('int64'), builder.fresh('int64')
                builder.emit(frozen_header, 'Phi', [zero, frozen_next], frozen_index,
                             attributes={'incoming_blocks': (frozen_entry.name, frozen_latch.name)})
                frozen_more = builder.fresh('bool')
                builder.emit(frozen_header, 'Lt', [frozen_index, original_length], frozen_more)
                builder.cond(frozen_header, frozen_more, frozen_body, frozen_complete)
                def frozen_row(block, column):
                    address, value = builder.fresh('ptr'), builder.fresh('opaque_ref')
                    builder.emit(block, 'GetElementPtr', [by_id[column], frozen_index], address)
                    builder.emit(block, 'Load', [address], value)
                    return value
                frozen_owner = frozen_row(frozen_body, context.members.column_value_ids[0])
                frozen_matches = builder.fresh('bool')
                builder.emit(frozen_body, 'Eq', [frozen_owner, source], frozen_matches)
                builder.cond(frozen_body, frozen_matches, frozen_write, frozen_latch)
                frozen_key = frozen_row(frozen_write, context.members.column_value_ids[1])
                frozen_item = frozen_row(frozen_write, context.members.column_value_ids[2])
                frozen_payload, frozen_status = builder.fresh('opaque_ref'), builder.fresh('bool')
                builder.emit(frozen_write, 'Select', [dictionary, frozen_key, frozen_item], frozen_payload)
                setter = module.functions[f'{prefix}_set']
                setter_roles = setter.metadata['opaque_object_helper']
                call(builder, frozen_write, setter,
                     {setter_roles['reference_formal_id']: destination,
                      setter_roles['operand_formal_ids'][0]: frozen_payload,
                      setter_roles['operand_formal_ids'][1]: frozen_payload}, frozen_status)
                builder.cond(frozen_write, frozen_status, frozen_latch, reject)
                builder.emit(frozen_latch, 'Add', [frozen_index, one], frozen_next)
                builder.branch(frozen_latch, frozen_header)
                builder.branch(frozen_complete, blocks['complete'])
            else:
                builder.cond(blocks['construct'], live, blocks['copy_entry'], reject)
            start, header, body, write, latch = (blocks[label] for label in
                ('copy_entry', 'copy_header', 'copy_body', 'copy_write', 'copy_latch'))
            builder.branch(start, header)
            index, copied, next_index, next_copied = (builder.fresh('int64') for _ in range(4))
            for value, update in ((index, next_index), (copied, next_copied)):
                builder.emit(header, 'Phi', [zero, update], value,
                    attributes={'incoming_blocks': (start.name, latch.name)})
            more = builder.fresh('bool')
            builder.emit(header, 'Lt', [index, original_length], more)
            builder.cond(header, more, body, blocks['publish'])
            owner = row_value(body, context.members.column_value_ids[0], index)
            matches = builder.fresh('bool')
            builder.emit(body, 'Eq', [owner, source], matches)
            builder.cond(body, matches, write, latch)
            key = row_value(write, context.members.column_value_ids[1], index)
            item = row_value(write, context.members.column_value_ids[2], index)
            payload = builder.fresh('opaque_ref')
            builder.emit(write, 'Select', [dictionary, key, item], payload)
            numbered, source_position, position = (builder.fresh('int64') for _ in range(3))
            builder.emit(write, 'Add', [copied, one], numbered)
            builder.emit(write, 'Cast', [key], source_position,
                attributes={'binding': 'object_context_position'})
            builder.emit(write, 'Select', [positional, source_position, numbered], position)
            lower, upper, position_valid = (builder.fresh('bool') for _ in range(3))
            builder.emit(write, 'Ge', [position, one], lower)
            builder.emit(write, 'Le', [position, count], upper)
            builder.emit(write, 'LAnd', [lower, upper], position_valid)
            stamp = blocks['copy_stamp']
            builder.cond(write, position_valid, stamp, reject)
            offset, destination_row, position_handle = (builder.fresh(dtype)
                for dtype in ('int64', 'int64', 'opaque_ref'))
            builder.emit(stamp, 'Sub', [position, one], offset)
            builder.emit(stamp, 'Add', [original_length, offset], destination_row)
            builder.emit(stamp, 'Cast', [position], position_handle,
                attributes={'binding': 'object_context_position'})
            for column, value in zip(context.members.column_value_ids,
                    (destination, position_handle, payload), strict=True):
                address = builder.fresh('ptr')
                builder.emit(stamp, 'GetElementPtr', [by_id[column], destination_row], address)
                builder.emit(stamp, 'Store', [value, address])
            builder.branch(stamp, latch)
            builder.emit(latch, 'Phi', [copied, numbered], next_copied,
                attributes={'incoming_blocks': (body.name, stamp.name)})
            builder.emit(latch, 'Add', [index, one], next_index)
            builder.branch(latch, header)
            builder.emit(blocks['publish'], 'Store', [required, length_slot])
            builder.branch(blocks['publish'], blocks['complete'])
            failed = builder.const(reject, 0, dtype='opaque_ref')
            builder.branch(reject, blocks['complete'])
            result = builder.fresh('opaque_ref')
            incoming = [source, destination, failed]
            incoming_blocks = [blocks['same'].name, blocks['publish'].name, reject.name]
            if frozen_complete is not None:
                incoming.insert(2, destination)
                incoming_blocks.insert(2, frozen_complete.name)
            builder.emit(blocks['complete'], 'Phi', incoming, result,
                attributes={'incoming_blocks': tuple(incoming_blocks)})
            builder.emit(blocks['complete'], 'Ret', [result])
            finish(label, builder, (source, requested), reference=source,
                operation=operation)

    # One resident sequence fold: dictionaries consume pairs, sets consume
    # identity members, lists consume ordered members. No host item dispatch.
    # Handles survive reserve; the initial extent excludes newly inserted rows.
    for label, operation, kinds in (
            ('update', ObjectOperation.UPDATE, ('builtins.dict', 'builtins.set')),
            ('extend', ObjectOperation.EXTEND, ('builtins.list',))):
        builder = _Builder(0)
        destination, source = builder.fresh('opaque_ref'), builder.fresh('opaque_ref')
        entry, header, body, consume, pair_block, member_block, store_block, written, latch, complete, reject = (
            builder.block(name) for name in ('entry', 'header', 'body', 'consume', 'pair',
                'member', 'store', 'written', 'latch', 'complete', 'reject'))
        zero, one = builder.const(entry, 0, dtype='int64'), builder.const(entry, 1, dtype='int64')
        def invoke(block, helper, reference, operands, result):
            interface = helper.metadata['opaque_object_helper']
            call(builder, block, helper, {interface['reference_formal_id']: reference,
                **dict(zip(interface['operand_formal_ids'], operands, strict=True))}, result)
        kind_helper, get_helper, set_helper, length_helper = (
            module.functions[f'{prefix}_{name}'] for name in ('kind', 'get', 'set', 'length'))
        destination_kind, source_kind = builder.fresh('int64'), builder.fresh('int64')
        invoke(entry, kind_helper, destination, (), destination_kind)
        invoke(entry, kind_helper, source, (), source_kind)
        def matches_kind(value, kind):
            predicate = builder.fresh('bool')
            builder.emit(entry, 'Eq', [value, builder.const(entry,
                context.kind_vocabulary.index(kind) + 1 if kind in context.kind_vocabulary else -1,
                dtype='int64')], predicate)
            return predicate
        def accepts(value, kinds):
            result = builder.const(entry, 0, dtype='bool')
            for kind in kinds:
                merged = builder.fresh('bool')
                builder.emit(entry, 'LOr', [result, matches_kind(value, kind)], merged)
                result = merged
            return result
        valid = builder.fresh('bool')
        builder.emit(entry, 'LAnd', [accepts(destination_kind, kinds),
                     accepts(source_kind, ('builtins.tuple', 'builtins.list'))], valid)
        dictionary = matches_kind(destination_kind, 'builtins.dict')
        _, extent = load_cell(builder, entry, by_id[context.members.length_address_id])
        initial_count = builder.fresh('int64')
        invoke(entry, length_helper, destination, (), initial_count)
        builder.cond(entry, valid, header, reject)
        index, next_index, count, next_count = (builder.fresh('int64') for _ in range(4))
        builder.emit(header, 'Phi', [zero, next_index], index,
                     attributes={'incoming_blocks': (entry.name, latch.name)})
        builder.emit(header, 'Phi', [initial_count, next_count], count,
                     attributes={'incoming_blocks': (entry.name, latch.name)})
        more = builder.fresh('bool')
        builder.emit(header, 'Lt', [index, extent], more)
        builder.cond(header, more, body, complete)
        def row(block, column):
            address, value = builder.fresh('ptr'), builder.fresh('opaque_ref')
            builder.emit(block, 'GetElementPtr', [by_id[column], index], address)
            builder.emit(block, 'Load', [address], value)
            return value
        owner = row(body, context.members.column_value_ids[0])
        matches = builder.fresh('bool')
        builder.emit(body, 'Eq', [owner, source], matches)
        builder.cond(body, matches, consume, latch)
        member = row(consume, context.members.column_value_ids[2])
        builder.cond(consume, dictionary, pair_block, member_block)
        pair_key, pair_item = builder.fresh('opaque_ref'), builder.fresh('opaque_ref')
        for ordinal, result in ((1, pair_key), (2, pair_item)):
            position = builder.fresh('opaque_ref')
            builder.emit(pair_block, 'Cast', [builder.const(pair_block, ordinal, dtype='int64')], position,
                         attributes={'binding': 'object_context_position'})
            invoke(pair_block, get_helper, member, (position,), result)
        builder.branch(pair_block, store_block)
        numbered = builder.fresh('int64')
        builder.emit(member_block, 'Add', [count, one], numbered)
        member_key = member
        if operation is ObjectOperation.EXTEND:
            member_key = builder.fresh('opaque_ref')
            builder.emit(member_block, 'Cast', [numbered], member_key,
                         attributes={'binding': 'object_context_position'})
        builder.branch(member_block, store_block)
        key, item, status = builder.fresh('opaque_ref'), builder.fresh('opaque_ref'), builder.fresh('bool')
        for result, incoming_values in ((key, [pair_key, member_key]), (item, [pair_item, member])):
            builder.emit(store_block, 'Phi', incoming_values, result,
                         attributes={'incoming_blocks': (pair_block.name, member_block.name)})
        invoke(store_block, set_helper, destination, (key, item), status)
        builder.cond(store_block, status, written, reject)
        updated_count = builder.fresh('int64')
        builder.emit(written, 'Add', [count, one], updated_count)
        builder.branch(written, latch)
        builder.emit(latch, 'Phi', [count, updated_count], next_count,
                     attributes={'incoming_blocks': (body.name, written.name)})
        builder.emit(latch, 'Add', [index, one], next_index)
        builder.branch(latch, header)
        builder.emit(complete, 'Ret', [builder.const(complete, 1, dtype='bool')])
        builder.emit(reject, 'Ret', [builder.const(reject, 0, dtype='bool')])
        finish(label, builder, (destination, source), reference=destination, operation=operation)

    builder = _Builder(0)
    entry = builder.block('entry')
    # all(map(is, zip(a,b))) is scalar control, not collection storage.
    # Positions are resolved through the incumbent native member ABI, never
    # Python iteration or value equality; false short-circuits and empty is true.
    reduction_builder = _Builder(0)
    left, right = reduction_builder.fresh('opaque_ref'), reduction_builder.fresh('opaque_ref')
    reduction_blocks = {name: reduction_builder.block(name) for name in
                        ('entry', 'header', 'body', 'latch', 'complete', 'reject')}
    re = reduction_blocks['entry']
    rz = reduction_builder.const(re, 0, dtype='int64')
    ro = reduction_builder.const(re, 1, dtype='int64')
    lengths = []
    length_helper = module.functions[f'{prefix}_length']
    for source in (left, right):
        length = reduction_builder.fresh('int64')
        call(reduction_builder, re, length_helper,
             {length_helper.metadata['opaque_object_helper']['reference_formal_id']: source}, length)
        lengths.append(length)
    shorter, extent = reduction_builder.fresh('bool'), reduction_builder.fresh('int64')
    reduction_builder.emit(re, 'Lt', lengths, shorter)
    reduction_builder.emit(re, 'Select', [shorter, *lengths], extent)
    reduction_builder.branch(re, reduction_blocks['header'])
    rh, rb, rl = (reduction_blocks[name] for name in ('header', 'body', 'latch'))
    index, next_index = reduction_builder.fresh('int64'), reduction_builder.fresh('int64')
    reduction_builder.emit(rh, 'Phi', [rz, next_index], index,
                           attributes={'incoming_blocks': (re.name, rl.name)})
    more = reduction_builder.fresh('bool')
    reduction_builder.emit(rh, 'Lt', [index, extent], more)
    reduction_builder.cond(rh, more, rb, reduction_blocks['complete'])
    ordinal, position = reduction_builder.fresh('int64'), reduction_builder.fresh('opaque_ref')
    reduction_builder.emit(rb, 'Add', [index, ro], ordinal)
    reduction_builder.emit(rb, 'Cast', [ordinal], position, attributes={'binding': 'object_context_position'})
    getter = module.functions[f'{prefix}_get']
    roles = getter.metadata['opaque_object_helper']
    members = []
    for source in (left, right):
        member = reduction_builder.fresh('opaque_ref')
        call(reduction_builder, rb, getter, {roles['reference_formal_id']: source,
             roles['operand_formal_ids'][0]: position}, member)
        members.append(member)
    same = reduction_builder.fresh('bool')
    reduction_builder.emit(rb, 'Eq', members, same)
    reduction_builder.cond(rb, same, rl, reduction_blocks['reject'])
    reduction_builder.emit(rl, 'Add', [index, ro], next_index)
    reduction_builder.branch(rl, rh)
    for name, truth in (('complete', 1), ('reject', 0)):
        block = reduction_blocks[name]
        reduction_builder.emit(block, 'Ret', [reduction_builder.const(block, truth, dtype='bool')])
    finish('identity_all', reduction_builder, (left, right), reference=left, operation=ObjectOperation.IDENTITY_ALL)

    entry.instrs.append(heap_release(by_id[context.objects.heap_owner_value_id]))
    # Release invalidates addresses and counters together, never leaves a stale
    # descriptor pointing at freed payload. Reinitialize to start a new domain.
    for value in values:
        if (value.accounting or {}).get('heap_owner_value_id') is not None:
            # This is the pointer CELL itself, not element zero of its now
            # freed payload. GEP would follow the arena and write freed bytes.
            slot = value
        else:
            slot, _ = load_cell(builder, entry, value)
        builder.emit(entry, 'Store', [builder.const(entry, 0, dtype=value.dtype), slot])
    builder.emit(entry, 'Ret', [])
    finish('release', builder)
    publish_native_object_requirements(module)
    return module
