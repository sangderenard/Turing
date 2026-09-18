"""Semantic Python-source lowering into the existing ProcessGraph vocabulary.

Structural ``ast.AST`` ingestion remains available in ``ProcessGraph``.  This
module is the source-language specialization: it translates Python spellings
to operators ProcessGraph/SSA already know and leaves unknown syntax explicit
as ``opaque_python`` nodes for the existing diagnostics.
"""

from __future__ import annotations

import ast

from ..common.tensors.fused_ir import ELEMENTWISE_BINARY, ELEMENTWISE_UNARY
from ..transmogrifier.solver_types import DomainNode


_BINARY = {
    ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "truediv",
    ast.FloorDiv: "floordiv", ast.Mod: "mod", ast.Pow: "pow",
    ast.MatMult: "matmul", ast.BitAnd: "bitand", ast.BitOr: "bitor",
    ast.BitXor: "bitxor", ast.LShift: "shl", ast.RShift: "shr",
}
_COMPARE = {
    ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le",
    ast.Gt: "gt", ast.GtE: "ge",
}
_CALLS = {
    **{name: name for name in ELEMENTWISE_UNARY | ELEMENTWISE_BINARY},
    "absolute": "abs", "divide": "truediv", "subtract": "sub",
    "multiply": "mul", "power": "pow", "concat": "cat",
    "concatenate": "cat", "arange": "arange", "broadcast_to": "broadcast_to",
    "cat": "cat", "clone": "clone", "cumsum": "cumsum",
    "flatten": "flatten", "full": "full", "gather": "gather",
    "log_softmax": "log_softmax", "matmul": "matmul", "max": "max",
    "mean": "mean", "pad": "pad", "reshape": "reshape", "stack": "stack",
    "sum": "sum", "topk": "topk", "transpose": "transpose",
}

_RANDOM_REQUESTS = {
    "getrandbits": ("bits", "integer"),
    "integers": ("integer", "tensor"),
    "normal": ("normal", "tensor"),
    "rand": ("uniform", "tensor"),
    "rand_like": ("uniform", "tensor_like"),
    "randint": ("integer", "tensor"),
    "randint_like": ("integer", "tensor_like"),
    "randn": ("normal", "tensor"),
    "random": ("uniform", "scalar_or_tensor"),
    "random_sample": ("uniform", "scalar_or_tensor"),
    "random_source": ("uniform", "scalar"),
    "random_tensor": ("uniform", "tensor"),
    "randoms": ("uniform", "tensor"),
    "randrange": ("integer", "scalar"),
    "sample": ("uniform", "scalar_or_tensor"),
    "standard_normal": ("normal", "tensor"),
    "uniform": ("uniform_range", "scalar_or_tensor"),
}


def _call_spelling(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _call_spelling(node.value)
        return node.attr if owner is None else f"{owner}.{node.attr}"
    return None


#: Expression constructs that carry their own nested body rather than a
#: block list. Handled alongside the block fields below.
_EXPRESSION_SCOPES = (
    ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
)


def _scope_label(node: ast.AST, field: str = "") -> str:
    name = getattr(node, "name", None)
    kind = type(node).__name__.casefold()
    where = name or getattr(node, "lineno", 0)
    return f"{kind}:{field}:{where}" if field else f"{kind}:{where}"


#: Width of a colour flag. 48 bits divided by 2**48 is exact in binary, so
#: a flag is an exactly-representable float and two flags compare equal or
#: they do not. No tolerance, no near-match, no ordering implied.
_FLAG_BITS = 48


def colour_flag(*parts: object) -> float:
    """Collapse an identity into one float frequency.

    The identity is a real, structured thing at the moment it is computed
    -- a scope path, a region index, a pass name. Once computed it stops
    being structure and becomes a flag: one float, carried, compared by
    equality, never parsed back apart. Anything that needs the original
    text asks the side table; nothing in the pipeline does.

    Deterministic across processes: built from a digest, not from Python's
    randomised ``hash``.
    """
    import hashlib

    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=_FLAG_BITS // 8).digest()
    return int.from_bytes(digest, "big") / float(1 << _FLAG_BITS)


def add_layer(layers: tuple, flag: float) -> tuple:
    """Append one translation's flag, building OUTWARD from the ingested form.

    Layer 0 is whatever form the program was ingested as. Each translation
    that touches a value appends its own flag, so the tuple read left to
    right is the order the value was transformed in, and two values sharing
    a prefix shared that much history exactly.

    Never inserts, never rewrites an existing layer, never reorders: an
    inner layer is a fact about a translation that already happened.
    """
    return tuple(layers) + (float(flag),)


def layers_by_node(graph) -> dict[int, tuple]:
    """{graph node id: its layers}, the join surface for the next pass.

    A translation reading this inherits a value's layers from the node it
    is translating FROM and appends its own flag with ``add_layer``. That
    is the whole contract: inherit at the exchange, append outward, never
    recompute. Downstream identifies nodes by the same id it already uses
    (``value_id`` defaults to the node id), so no new correspondence is
    introduced -- this exposes what is already there.
    """
    found: dict[int, tuple] = {}
    for node_id, data in graph.G.nodes(data=True):
        layers = tuple((data.get("source_span") or {}).get("layers") or ())
        if layers:
            found[int(node_id)] = layers
    return found


def scope_paths(root: ast.AST, base: tuple = ()) -> dict[int, tuple]:
    """Map every AST node to the scope path enclosing it.

    Taken ONCE, here, at ingestion, where the boundaries are still written
    down in the syntax. Every later representation carries the path it was
    given rather than recovering it: a scope recomputed downstream is a
    guess about a fact that was known exactly at the point of entry, and a
    guess is what this must never be.

    Keyed by ``id(node)``, so the caller must hold the tree alive while the
    map is in use -- which ``build_semantic_ast`` does.
    """
    paths: dict[int, tuple] = {}

    def walk(node: ast.AST, path: tuple) -> None:
        paths[id(node)] = path
        for field, value in ast.iter_fields(node):
            # A block a construct opens is a scope. `if/else` opens two and
            # they are distinct, so the FIELD is part of the label -- a loop
            # body and the else that runs when it completes are different
            # places, and a `try` body is not its handler.
            if isinstance(value, list) and any(
                isinstance(item, ast.stmt) for item in value
            ):
                inner = path + (_scope_label(node, field),)
                for item in value:
                    walk(item, inner)
                continue
            # A parameter list is a BOUNDARY, not an interior and not the
            # exterior. Which of those a given parameter actually is depends
            # on ref/copy: storage passed by reference stays the caller's and
            # is merely aliased here, while a copied value is genuinely
            # local. Ingestion cannot know which -- that is settled later, by
            # the record ABI. So it is marked as what it verifiably is, a
            # binding at the boundary of this function, and the pass that
            # resolves ref/copy adds the layer that says which.
            #
            # Lumping parameters into the enclosing scope would assert they
            # are the caller's; lumping them into the body would assert they
            # are local. Both are claims ingestion has no grounds for.
            if field == "args" and isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            ):
                for child in (value if isinstance(value, list) else [value]):
                    if isinstance(child, ast.AST):
                        walk(child, path + (_scope_label(node, "args"),))
                continue
            for child in (
                value if isinstance(value, list) else [value]
            ):
                if not isinstance(child, ast.AST):
                    continue
                walk(
                    child,
                    path + (_scope_label(child),)
                    if isinstance(child, _EXPRESSION_SCOPES) else path,
                )

    walk(root, base)
    return paths


def build_semantic_ast(
    graph, tree: ast.AST, *, filename: str | None = None,
    helpers: dict | None = None, self_constants: dict | None = None,
):
    """Lower one Python function/module into ``graph`` using existing ops.

    ``helpers`` maps names to ``ast.FunctionDef`` bodies that calls may be
    lowered INTO (module-local helper functions): the call site's arguments
    bind to the helper's parameters and the helper's dataflow continues in
    place, so composition pursues down to the canonical vocabulary instead of
    stopping at an unimplementable ``call`` node. ``self_constants`` maps
    attribute names to literal values for ``self.NAME`` reads (class-level
    constants), which otherwise have no dataflow meaning inside one function.
    """

    env: dict[str, int] = {}
    next_id = 0
    while next_id in graph.G:
        next_id += 1

    # The scope division is taken here, once, from the syntax. It is a real
    # structured thing for exactly as long as it takes to compute; then it
    # becomes layer 0, one float, and the structure is not carried.
    #
    # Every later translation appends ITS flag to these layers by direct
    # provenance -- the value it produces inherits the layers of the value
    # it was produced FROM, at the point of exchange, plus one. A layer is
    # never recovered by matching or by looking at the finished form.
    enclosing = scope_paths(tree)
    ingest_flags = {
        node_id: colour_flag("scope", path)
        for node_id, path in enclosing.items()
    }
    # The side table: flag -> what it was, kept for readers only. Nothing in
    # the pipeline consults it, exactly as the runtime never reads
    # trace_manifest.
    graph.G.graph.setdefault("layer_names", {}).update({
        colour_flag("scope", path): ".".join(path) or "<module>"
        for path in set(enclosing.values())
    })

    def span(node):
        flag = ingest_flags.get(id(node))
        return {
            "filename": filename,
            "line": getattr(node, "lineno", None),
            "column": getattr(node, "col_offset", None),
            "end_line": getattr(node, "end_lineno", None),
            "end_column": getattr(node, "end_col_offset", None),
            "layers": () if flag is None else (flag,),
        }

    def add_node(
        op, inputs=(), *, label=None, attributes=None, constant=None,
        tensor=None, control=None, source=None, output_roles=("result",),
    ):
        nonlocal next_id
        while next_id in graph.G:
            next_id += 1
        nid = next_id
        next_id += 1
        parents = list(inputs)
        domain = DomainNode(shape=(1, 1, 1), unit_size=1)
        domain.id = id(domain)
        attrs = dict(attributes or {})
        mutation = getattr(graph, "graph_mutation", None)
        if mutation is None:
            from contextlib import nullcontext
            mutation = nullcontext
        with mutation():
            graph.G.add_node(
                nid, label=label or op, type=op, op=op, expr_obj=None,
                extra_args=attrs, attributes=attrs, constant=constant,
                tensor=dict(tensor or {}), bit_quanta=None,
                control=dict(control or {}), source_span=source,
                input_roles=tuple(role for _, role in parents),
                output_roles=tuple(output_roles), schema_version=1,
                domain_node=domain, store_id=None, parents=parents, children=[],
            )
            for parent, role in parents:
                graph.G.add_edge(parent, nid, role=role)
                graph.G.nodes[parent]["children"].append((nid, role))
            observer = getattr(graph, "observe_evolution_node", None)
            if observer is not None:
                observer(nid, graph.G.nodes[nid])
            edge_observer = getattr(graph, "observe_evolution_edge", None)
            if edge_observer is not None:
                for parent, role in parents:
                    edge_observer(parent, nid, role)
        return nid

    def opaque(node):
        return add_node(
            "opaque_python",
            attributes={"ast_type": type(node).__name__, "dump": ast.dump(node)},
            source=span(node),
        )

    def inline_helper(function, call_node):
        """Continue lowering inside a helper: arguments bind to parameters.

        Returns the node id of the helper's returned value, or None when the
        call shape is not bindable (defaults, keywords, arity mismatch) --
        the caller then falls back to an explicit ``call`` node.
        """

        parameters = [parameter.arg for parameter in function.args.args]
        if call_node.keywords or len(call_node.args) != len(parameters):
            return None
        values = [expression(argument) for argument in call_node.args]
        saved = dict(env)
        env.clear()
        env.update(dict(zip(parameters, values)))
        result = None
        for child in function.body:
            if isinstance(child, ast.Return):
                result = (expression(child.value)
                          if child.value is not None else None)
                break
            statement(child)
        env.clear()
        env.update(saved)
        return result

    def expression(node):
        if isinstance(node, ast.Name):
            if node.id not in env:
                env[node.id] = add_node(
                    "input", label=node.id, attributes={"name": node.id},
                    tensor={}, source=span(node), output_roles=("value",),
                )
            return env[node.id]
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and self_constants is not None
            and node.attr in self_constants
        ):
            value = self_constants[node.attr]
            return add_node(
                "const", label=repr(value), constant=value,
                source=span(node), output_roles=("value",),
            )
        if isinstance(node, ast.Constant):
            return add_node(
                "const", label=repr(node.value), constant=node.value,
                source=span(node), output_roles=("value",),
            )
        if isinstance(node, ast.BinOp):
            op = _BINARY.get(type(node.op))
            if op is None:
                return opaque(node)
            return add_node(
                op, ((expression(node.left), "lhs"), (expression(node.right), "rhs")),
                source=span(node),
            )
        if isinstance(node, ast.UnaryOp):
            op = {ast.USub: "neg", ast.UAdd: "identity", ast.Not: "logical_not",
                  ast.Invert: "invert"}.get(type(node.op))
            return (add_node(op, ((expression(node.operand), "operand"),), source=span(node))
                    if op is not None else opaque(node))
        if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
            op = _COMPARE.get(type(node.ops[0]))
            return (add_node(
                op,
                ((expression(node.left), "lhs"),
                 (expression(node.comparators[0]), "rhs")),
                source=span(node),
            ) if op is not None else opaque(node))
        if isinstance(node, ast.Call):
            qualified = _call_spelling(node.func)
            spelling = (node.func.id if isinstance(node.func, ast.Name)
                        else node.func.attr if isinstance(node.func, ast.Attribute)
                        else None)
            random_request = _RANDOM_REQUESTS.get(spelling)
            if random_request is not None:
                distribution, result_kind = random_request
                arguments = [expression(arg) for arg in node.args]
                attrs = {
                    "request": qualified or spelling,
                    "distribution": distribution,
                    "result_kind": result_kind,
                    "arguments": tuple(ast.unparse(arg) for arg in node.args),
                }
                for keyword in node.keywords:
                    key = keyword.arg or "**"
                    try:
                        attrs[key] = ast.literal_eval(keyword.value)
                    except (ValueError, TypeError):
                        attrs[key] = ast.unparse(keyword.value)
                return add_node(
                    "random_source",
                    zip(arguments, (f"arg{i}" for i in range(len(arguments)))),
                    label="random_source",
                    attributes=attrs,
                    source=span(node),
                )
            canonical = _CALLS.get(spelling)
            if (
                canonical is None
                and helpers
                and isinstance(node.func, ast.Name)
                and node.func.id in helpers
            ):
                inlined = inline_helper(helpers[node.func.id], node)
                if inlined is not None:
                    return inlined
            arguments = []
            if canonical is not None and isinstance(node.func, ast.Attribute):
                arguments.append(expression(node.func.value))
            arguments.extend(expression(arg) for arg in node.args)
            if canonical is not None:
                roles = (("operand",) if canonical in ELEMENTWISE_UNARY
                         else ("lhs", "rhs") if canonical in ELEMENTWISE_BINARY
                         else tuple(f"arg{i}" for i in range(len(arguments))))
                if len(roles) != len(arguments):
                    return opaque(node)
                attrs = {}
                for keyword in node.keywords:
                    key = keyword.arg or "**"
                    try:
                        attrs[key] = ast.literal_eval(keyword.value)
                    except (ValueError, TypeError):
                        attrs[key] = ast.unparse(keyword.value)
                return add_node(
                    canonical, zip(arguments, roles), label=canonical,
                    attributes=attrs, source=span(node),
                )
            function = (node.func.id if isinstance(node.func, ast.Name)
                        else ast.unparse(node.func))
            return add_node(
                "call", zip(arguments, (f"arg{i}" for i in range(len(arguments)))),
                label=function,
                attributes={"function": function, "keywords": {
                    keyword.arg or "**": ast.unparse(keyword.value)
                    for keyword in node.keywords
                }}, source=span(node),
            )
        return opaque(node)

    def statement(node):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = (node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1
                      else node.target if isinstance(node, ast.AnnAssign) else None)
            value_node = getattr(node, "value", None)
            if value_node is None:
                return None
            value = expression(value_node)
            if isinstance(target, ast.Name):
                env[target.id] = value
                return value
            return opaque(node)
        if isinstance(node, ast.Expr):
            return expression(node.value)
        if isinstance(node, ast.Return):
            value = expression(node.value) if node.value is not None else None
            result = add_node(
                "return", () if value is None else ((value, "value"),),
                source=span(node), output_roles=(),
            )
            graph.roots.append(result)
            return result
        if isinstance(node, ast.If):
            condition = expression(node.test)
            before = dict(env)
            for child in node.body:
                statement(child)
            then_env = dict(env)
            env.clear(); env.update(before)
            for child in node.orelse:
                statement(child)
            else_env = dict(env)
            merged = dict(before)
            for name in sorted(set(then_env) | set(else_env)):
                yes = then_env.get(name, before.get(name))
                no = else_env.get(name, before.get(name))
                if yes is None or no is None:
                    continue
                merged[name] = yes if yes == no else add_node(
                    "select", ((condition, "condition"), (yes, "if_true"),
                               (no, "if_false")),
                    attributes={"variable": name}, source=span(node),
                )
            env.clear(); env.update(merged)
            return condition
        if isinstance(node, (ast.With, ast.AsyncWith)):
            # A context manager guards execution; it carries no dataflow. The
            # body lowers in the enclosing environment, and the context
            # expression (autograd.no_grad, device scopes) is machinery the
            # destination does not re-run.
            for child in node.body:
                statement(child)
            return None
        return opaque(node)

    if isinstance(tree, ast.Module):
        body = tree.body
    elif isinstance(tree, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for argument in tree.args.args:
            env[argument.arg] = add_node(
                "input", label=argument.arg, attributes={"name": argument.arg},
                tensor={}, source=span(argument), output_roles=("value",),
            )
        body = tree.body
    else:
        body = [ast.Expr(value=tree)]

    for child in body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if graph.G.number_of_nodes():
                raise ValueError("semantic AST import accepts one top-level function")
            return build_semantic_ast(
                graph, child, filename=filename,
                helpers=helpers, self_constants=self_constants,
            )
        statement(child)
    if not graph.roots:
        graph.roots = [nid for nid in graph.G if graph.G.out_degree(nid) == 0]
    graph.domain_shape = (1,)
    return graph
