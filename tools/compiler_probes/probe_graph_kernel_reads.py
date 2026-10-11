"""Print the authoritative operands at the graph-kernel lowering refusal."""
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.compiler.precompile_to_ssa import _ControlSSABuilder
from src.compiler import glsl_deployment_strategy as strategy
from src.compiler.identity_concordance import current_identity_book
from src.transmogrifier.ssa import Instr
import traceback

instruction_setattr = Instr.__setattr__


class ObservedArguments(list):
    def __setitem__(self, key, value):
        print("PUSH_ARG_WRITE", [item.id for item in self], key,
              getattr(value, "id", value), traceback.format_stack(limit=6), flush=True)
        return super().__setitem__(key, value)


def observed_instruction_setattr(self, key, value):
    if "push" in sys.argv and key == "args" and getattr(self, "attributes", {}).get("region_index") == 40:
        before = getattr(self, "args", ())
        if [item.id for item in before] != [item.id for item in value]:
            print("PUSH_ARGS_REPLACED", [item.id for item in before],
                  [item.id for item in value], traceback.format_stack(limit=6), flush=True)
        value = ObservedArguments(value)
    return instruction_setattr(self, key, value)


Instr.__setattr__ = observed_instruction_setattr
emit_instruction = _ControlSSABuilder.emit


def observed_instruction_emit(self, op, args, result=None, *, attributes=None):
    emit_instruction(self, op, args, result, attributes=attributes)
    if "push" in sys.argv and (attributes or {}).get("region_index") == 40:
        self.current.instrs[-1].args = ObservedArguments(self.current.instrs[-1].args)


_ControlSSABuilder.emit = observed_instruction_emit
bind_value = _ControlSSABuilder._bind


def observed_bind(self, value_id, value, kind, *cells):
    if "push" in sys.argv and value_id == 84:
        print("PUSH_BIND", self.current.name, value_id, value.id, kind,
              traceback.format_stack(limit=3), flush=True)
    return bind_value(self, value_id, value, kind, *cells)


_ControlSSABuilder._bind = observed_bind

indexer = strategy._dispatch_graph_index


def observe_index(graph):
    if len(graph.G) > 60:
        for node, data in graph.G.nodes(data=True):
            attributes = data.get("attributes") or {}
            if data.get("type") in {"While", "LoopResult", "If"} or node in {84, 90, 91, 92, 93, 94, 116}:
                print("SOURCE_NODE", node, data.get("type"),
                      "line", getattr(data.get("expr_obj"), "lineno", None),
                      "parents", data.get("parents"), "attributes", attributes, flush=True)
        scope = graph.G.graph.get("lexical_read_scope")
        page = current_identity_book().page("name_binding")
        for row in page.scope_rows(scope):
            if row[1] in {"position", "child", "node", "size"}:
                print("SOURCE_BINDING", row, page.latest(row), flush=True)
    return indexer(graph)


strategy._dispatch_graph_index = observe_index
build_plan = strategy._build_shell_hierarchy_plan


def observed_plan(shell, _memo=None):
    if "push" in sys.argv and len(shell.process_graph.G) > 60:
        for value in (18, 27, 81, 123, 84, 95, 103):
            data = shell.process_graph.G.nodes.get(value, {})
            print("PLAN_INPUT", value, data.get("type"), data.get("parents"),
                  data.get("tensor"), strategy._tensor_descriptor(shell.process_graph, value), flush=True)
    plan = build_plan(shell, _memo)
    if "push" in sys.argv and len(shell.process_graph.G) > 60:
        def walk(item):
            shapes = getattr(item, "value_shapes", ())
            if shapes:
                print("PLAN_TYPES", item.name, shapes, flush=True)
            for child in getattr(item, "items", ()):
                walk(child)
        walk(plan)
    return plan


strategy._build_shell_hierarchy_plan = observed_plan

dispatch = strategy._dispatch_subgraph


def observe_dispatch(graph, node_ids, **kwargs):
    selected = dispatch(graph, node_ids, **kwargs)
    for node in (15, 20):
        if node in selected.G:
            data = graph.G.nodes.get(node, {})
            print("CUT_NODE", node, {key: data.get(key) for key in
                  ("type", "label", "parents", "value", "attributes")}, flush=True)
    return selected


strategy._dispatch_subgraph = observe_dispatch
post_region = strategy._post_deployment_region


def observed_region(graph, subgraph, region_index):
    if region_index == 26:
        print("REGION_MEMBERS", region_index, [(node,
              getattr(graph.G.nodes[node].get("expr_obj"), "lineno", None),
              graph.G.nodes[node].get("source_span"), graph.G.nodes[node].get("type"))
              for node in subgraph.G.graph.get("deployment_nodes", ())], flush=True)
    return post_region(graph, subgraph, region_index)


strategy._post_deployment_region = observed_region

original = _ControlSSABuilder._region_feed
lower = _ControlSSABuilder.lower


def observed_control(self, block, *, path="root"):
    if path == "root":
        def walk(item, prefix):
            print("CONTROL_TREE", prefix, type(item).__name__,
                  getattr(item, "source_loop_node_id", None),
                  getattr(item, "source_node_id", None),
                  getattr(item, "lines", None), flush=True)
            for field in ("blocks", "condition", "body", "orelse", "terminal_controls"):
                child = getattr(item, field, None)
                if child is None:
                    continue
                for index, part in enumerate(child if isinstance(child, tuple) else (child,)):
                    walk(part, prefix + "." + field + str(index))
        walk(block, "root")
    return lower(self, block, path=path)


_ControlSSABuilder.lower = observed_control


def observed(self, region_index, value_id):
    try:
        result = original(self, region_index, value_id)
        if "push" in sys.argv and region_index in {39, 40, 41, 42}:
            book = self._book()
            row = (str(self.tensor_shape_concordance_scope), int(region_index), int(value_id))
            consumers = book.page("region_feed_consumer").latest(row) or ()
            reads = []
            for consumer in consumers:
                positions = book.page("consumer_operand").latest((self.lexical_read_scope, consumer, value_id)) or ()
                for role, ordinal in positions:
                    key = (self.lexical_read_scope, consumer, role, ordinal)
                    reads.append((key, book.page("lexical_read_binding").latest(key)))
            print("PUSH_FEED", self.current.name, region_index, value_id,
                  "reads", reads, "result", result.id, result.dtype, flush=True)
        return result
    except ValueError:
        book = self._book()
        row = (str(self.tensor_shape_concordance_scope), int(region_index), int(value_id))
        consumers = book.page("region_feed_consumer").latest(row) or ()
        print("REFUSED_FEED", row, "lexical_scope", self.lexical_read_scope, flush=True)
        for consumer in consumers:
            positions = book.page("consumer_operand").latest((self.lexical_read_scope, consumer, value_id))
            print("CONSUMER", consumer, "positions", positions, flush=True)
            for role, ordinal in positions or ():
                key = (self.lexical_read_scope, consumer, role, ordinal)
                print("READ", key, book.page("lexical_read_binding").history(key),
                      "TRANSITION", book.page("identity_transition").history(key), flush=True)
        for page_name in ("name_binding", "loop_carried_binding"):
            page = book.page(page_name)
            for key in page.scope_rows(self.lexical_read_scope):
                fact = page.latest(key)
                if str(value_id) in repr(fact):
                    print(page_name, key, fact, flush=True)
        raise


_ControlSSABuilder._region_feed = observed
from src.compiler import ssa_call_input_adapters as adapters
concord = adapters._concord_exact_region_feed_dtypes


def observed_types(functions, changes=None):
    for function in functions.values():
        residents = {item.res.id: item.res for block in function.blocks.values()
                     for item in block.instrs if item.res is not None}
        for block in function.blocks.values():
            for item in block.instrs:
                if item.attributes.get("region_index") == 13:
                    print("SETTLE_FEED", [(arg.id, arg.dtype,
                           None if arg.id not in residents else residents[arg.id].dtype)
                          for arg in item.args], item.attributes, flush=True)
    return concord(functions, changes)


adapters._concord_exact_region_feed_dtypes = observed_types
__file__ = str(REPO / "tests/test_graph_refinement_kernels.py")
exec(compile(Path(__file__).read_text(), __file__, "exec"))
if "push" in sys.argv:
    emit_kernel = emit_ssa_function_to_llvm

    def observed_emit(module, name):
        root = module.functions[name]
        print("ROOT_ALIASES", root.metadata.get("value_aliases"), flush=True)
        for block in root.blocks.values():
            for instruction in block.instrs:
                if instruction.op == "Phi":
                    print("ROOT_PHI", block.name, instruction.res,
                          instruction.args, instruction.attributes, flush=True)
                if instruction.attributes.get("region_index") in {39, 40, 41, 42, 53}:
                    print("ROOT_PUSH_CALL", block.name, instruction.op,
                          instruction.args, instruction.attributes, flush=True)
        book = module.metadata["identity_book"]
        for page_name in ("loop_carried_binding", "loop_state_port"):
            page = book.page(page_name)
            for row in page.rows():
                print("LOOP_FACT", page_name, row, page.latest(row), flush=True)
        return emit_kernel(module, name)

    emit_ssa_function_to_llvm = observed_emit
    compile_kernel(Path(tempfile.mkdtemp(prefix="graph-kernel-push-")))
elif "heap" in sys.argv:
    test_graph_kernel_first_heap_pop_keeps_the_selected_value(Path(tempfile.mkdtemp(prefix="graph-kernel-read-")))
else:
    test_graph_kernel_initial_frontier_preserves_conditional_carried_count(Path(tempfile.mkdtemp(prefix="graph-kernel-read-")))
