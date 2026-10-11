"""Observe actual graph/formal identities in tiny copier source lowerings."""
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.test_native_object_contract_batch import policy, typed_context
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

SOURCES = {
    'recursive_dict': ('def root(value):\n    def visit(item):\n'
        '        if isinstance(item,dict):\n            result={}\n'
        '            result.update((key,visit(child)) for key,child in item.items())\n'
        '            return result\n        return item\n    return visit(value)\n'),
    'identity_all': ('def root(value):\n    copied=tuple(item for item in value)\n'
        '    return all(new is old for new,old in zip(copied,value))\n'),
    'extend': ('def root(value):\n    result=[]\n'
        '    result.extend(item for item in value)\n    return result\n'),
    'set': ('def root(value):\n    result=set()\n'
        '    result.update(item for item in value)\n    return result\n'),
    'frozenset': 'def root(value):\n    return frozenset(value)\n',
    'get': 'def root(value,memo):\n    return memo.get(id(value))\n',
}
observed_functions = {}
previous_line = {}

def trace(frame, event, arg):
    if event == 'return' and frame.f_code.co_name == '_attach_graph_control_expressions':
        graph = frame.f_locals['graph_obj']
        if graph.graph.get('function_name') == 'isolate_metadata':
            from src.compiler.glsl_deployment_strategy import _source_control_records
            print('AUTHORED_PATHS', tuple((node, getattr(record['expression'], '_turing_source_span_row', None))
                  for node, record in _source_control_records(graph).items()), flush=True)
    if event == 'return' and frame.f_code.co_name in {
        '_place_plan_callsites_lexically', '_install_lexical_sequence_mutations',
        '_install_lexical_sequence_queries', '_attach_graph_control_expressions',
    } and hasattr(arg, 'root'):
        from src.compiler.control_source import SequenceBlock, ConditionalBlock
        from src.compiler.fortran_c_shell import _authored_node_position
        graph = frame.f_locals.get('graph')
        if graph is not None and graph.G.graph.get('function_name') == 'isolate_metadata':
            blocks = arg.root.blocks if isinstance(arg.root, SequenceBlock) else (arg.root,)
            print('ROOT_ORDER', frame.f_code.co_name, tuple(
                (type(block).__name__, getattr(block, 'source_node_id', None),
                 _authored_node_position(graph.G, block.source_node_id)
                 if isinstance(block, ConditionalBlock) else None)
                for block in blocks), flush=True)
    if event == 'return' and frame.f_code.co_name in {'project_control_regions', 'overlay_scheduled_control'}:
        from src.compiler.control_source import SequenceBlock, LoopBlock, WhileBlock, ConditionalBlock
        def loops(block):
            found = set()
            if isinstance(block, SequenceBlock):
                for child in block.blocks:
                    found.update(loops(child))
            if isinstance(block, (LoopBlock,WhileBlock)):
                found.add(block.source_loop_node_id)
                found.update(loops(block.body))
            if isinstance(block, ConditionalBlock):
                found.update(loops(block.body))
                if block.orelse is not None:
                    found.update(loops(block.orelse))
            return found
        before = set()
        for program in frame.f_locals.get('controls', (frame.f_locals.get('program'),)):
            if program is not None:
                before.update(loops(program.root))
        after = set() if arg is None else loops(arg.root)
        if before != after:
            print('CONTROL_LOSS', frame.f_code.co_name, 'BEFORE',before,'AFTER',after,
                  'PARENT', frame.f_back.f_code.co_name, frame.f_back.f_lineno,
                  'NESTING', frame.f_locals.get('known_nesting'),flush=True)
    if event == 'return' and frame.f_code.co_name == 'analyze_shader_loop_reductions':
        print('LOOP_REDUCTIONS', tuple((r.loop_node_id,r.blockers) for r in arg), flush=True)
    if event == 'return' and frame.f_code.co_name == '_install_lexical_sequence_mutations':
        if frame.f_locals['graph'].G.graph.get('function_name') == 'isolate_metadata':
            print('INSTALLED_EFFECTS', tuple((m.effect_node_id,getattr(m,'operator',None))
                for m in frame.f_locals.get('mutations', ())),
                'EXISTING', frame.f_locals.get('existing_effect_ids'), flush=True)
    if event in {'call','return'} and frame.f_code.co_name == '_place_plan_callsites_lexically':
        from src.compiler.control_source import SequenceBlock, LoopBlock, WhileBlock, ConditionalBlock, StatementBlock
        def paths(block, owners=()):
            if isinstance(block, StatementBlock) and block.lines and '__plan_callsite_' in block.lines[0]:
                print('PLACED_CALL', block.lines, owners, flush=True)
            if isinstance(block, SequenceBlock):
                for child in block.blocks:
                    paths(child, owners)
            if isinstance(block, (LoopBlock, WhileBlock)):
                print('PLACED_LOOP', event, block.source_loop_node_id, owners, flush=True)
                paths(block.body, (*owners, ('loop', block.source_loop_node_id)))
            if isinstance(block, ConditionalBlock):
                paths(block.body, (*owners, ('if', block.source_node_id, 'body')))
                if block.orelse is not None:
                    paths(block.orelse, (*owners, ('if', block.source_node_id, 'orelse')))
        if frame.f_locals['graph'].G.graph.get('function_name') == 'isolate_metadata':
            from src.compiler.hierarchical_plan import PlanCall
            print('PLANNED_CALLS', event, [(c.callsite_id,c.enclosing_loop_ids) for c in frame.f_locals['hierarchy_plan'].items
                                         if isinstance(c,PlanCall)], flush=True)
            paths((frame.f_locals['control'] if event == 'call' else arg).root)
    if event == 'line' and frame.f_code.co_name == '_class_surface_ssa_program' and '--writes' in sys.argv:
        for label, function in frame.f_locals.get('all_functions', {}).items():
            if label == 'root__visit':
                signature = tuple((block.name, tuple((ins.op,ins.attributes.get('callee'))
                    for ins in block.instrs)) for block in function.blocks.values())
                if observed_functions.get(label) != signature:
                    print('FUNCTION_WRITE', previous_line.get(id(frame)), '->', frame.f_lineno, signature, flush=True)
                    observed_functions[label] = signature
        previous_line[id(frame)] = frame.f_lineno
    if event == 'return' and frame.f_code.co_name == 'source_constructor_mutations':
        graph = frame.f_locals['graph']
        if graph.graph.get('function_name') in {'visit', 'root'}:
            print('BIRTHS', tuple((m.sequence_value_id,m.operator) for m in arg),
                  'ABI', graph.graph.get('linked_value_abi'), flush=True)
            if graph.graph.get('function_name') == 'visit':
                from src.compiler.glsl_deployment_strategy import _source_control_records
                print('CONTROLS', _source_control_records(graph), flush=True)
            if any((data.get('attributes') or {}).get('extraction_identity') == 'builtins.all'
                   for _, data in graph.nodes(data=True)):
                print('REDUCTION_GRAPH', tuple((node, data.get('type'), data.get('op'), data.get('parents'),
                    {key: value for key, value in (data.get('attributes') or {}).items()
                     if key in {'aggregate_kind','producer_kind','extraction_identity','loop_target_bindings',
                        'loop_id','materializer_node_id','result_kind','collection_iterable_value_id'}})
                    for node, data in graph.nodes(data=True)), flush=True)
    if event == 'exception' and frame.f_code.co_name == 'recursive_frame_lease':
        local = frame.f_locals
        source, callee = local['source'], local['callee']
        print('CALLEE_FORMALS', callee.name, callee.metadata.get('parameter_names'),
              tuple((v.id,v.dtype,v.accounting) for v in callee.args), flush=True)
        print('LEASE', repr(source), 'ACCOUNTING', source.accounting, flush=True)
        if '--brief' in sys.argv:
            return trace
        for block in callee.blocks.values():
            for instruction in block.instrs:
                if (instruction.res is not None and instruction.res.id == source.id
                        or any(value.id == source.id for value in instruction.args)):
                    print('CONSUMER', block.name, repr(instruction), flush=True)
        graph = local['caller_graph']
        print('GRAPH_ABI', graph.graph.get('linked_value_abi'), flush=True)
        for node, data in graph.nodes(data=True):
            print('GRAPH', node, data.get('type'), data.get('op'), data.get('parents'),
                {key: value for key, value in (data.get('attributes') or {}).items()
                 if key in {'aggregate_kind', 'producer_kind', 'extraction_identity',
                    'loop_target_bindings', 'callee_ref', 'structural_identity_actual_value_id',
                    'collection_owner_id', 'materialized_source_value_ids'}}, flush=True)
    return trace

if __name__ == '__main__':
    for name in sys.argv[1:] or SOURCES:
        print('CASE', name, flush=True)
        sys.settrace(trace)
        try:
            module, _, exports = lower_ast_source_to_ssa(SOURCES[name], 'root',
                extraction_contract=policy(name == 'get'), native_object_context=typed_context(),
                progress=lambda _: None, progress_bars=False,
                runtime_closure_only=True)
            print('LOWERED', exports, flush=True)
            if name == 'recursive_dict':
                for function in module.functions.values():
                    if function.metadata.get('native_object_collections'):
                        print('FUNCTION', function.name, 'FORMALS', [(v.id,v.dtype) for v in function.args], flush=True)
                        for block in function.blocks.values():
                            print('BLOCK', block.name, block.successors, flush=True)
                            for ins in block.instrs:
                                print('SSA', ins.op, tuple(v.id for v in ins.args),
                                      None if ins.res is None else ins.res.id, ins.attributes, flush=True)
            from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
            artifact = emit_ssa_function_to_llvm(module, exports[0])
            print('EMITTED', artifact.complete, artifact.shortfalls, flush=True)
        except Exception:
            traceback.print_exc(file=sys.stdout)
        finally:
            sys.settrace(None)
