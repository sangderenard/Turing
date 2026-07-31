# deep_graph_compiler.py
"""Turn a fully‑levelled ProcessGraph into a single Python function.

The emitted function is *pure* (no internal state) and therefore fast in
serial execution loops.  All operator kernels are looked‑up from the
provided `op_table` so the same compiler works for torch, numpy, jax, ….

Example
-------
>>> pg.compute_levels("asap")
>>> compile_pg = GraphDeepCompiler(pg, op_table)
>>> f = compile_pg.build_function()
>>> out1, out2 = f(x=np.ones(3), y=np.arange(3))
"""
from __future__ import annotations

import enum, textwrap, inspect, hashlib, types
from typing import Any, Callable, Dict, List, Tuple

import networkx as nx


def _graph_input_name(label: str) -> str:
    """Return the established keyword name for one graph input."""

    import re

    name = str(label).strip()
    lowered = name.lower()
    match = re.match(r"([a-zA-Z]+)[_\d]*$", lowered)
    root = match.group(1) if match else lowered
    if root in ("i", "j", "k", "l", "m", "n"):
        prefix = "int"
    elif (
        lowered.startswith("num")
        or lowered.endswith("idx")
        or lowered.isdigit()
    ):
        prefix = "int"
    elif lowered.startswith("is_") or lowered.startswith("has_"):
        prefix = "bool"
    else:
        prefix = "float"
    return f"{prefix}{name}"


def _is_emittable_literal(value: Any) -> bool:
    """True when ``repr(value)`` is a closed literal with no runtime symbols.

    ProcessGraph constants may contain immutable structural tuples (for
    example, JPEG quantization tables).  They are data owned by the compiled
    program, not Python callables or names to resolve at runtime.  Validate
    every leaf before embedding the representation so accepting a tuple
    cannot become a route for smuggling arbitrary Python objects into a
    compiled shell.
    """

    if value is Ellipsis or isinstance(
        value, (str, bytes, int, float, complex, bool, type(None))
    ):
        return True
    if isinstance(value, (tuple, list)):
        return all(_is_emittable_literal(item) for item in value)
    if isinstance(value, dict):
        return all(
            _is_emittable_literal(key) and _is_emittable_literal(item)
            for key, item in value.items()
        )
    return False


class GraphDeepCompiler:
    """Compile a *level‑sorted* ProcessGraph into one Python function."""

    #: attribute names we expect on ProcessGraph nodes
    _REQ = ("type", "label", "parents")

    def __init__(
        self,
        pg: "ProcessGraph",
        op_table: Dict[str, Callable],
        signatures: Dict[str, Dict[str, Any]],
        *,
        node_observer: Callable[
            [int, tuple[int, ...], tuple[Any, ...], Any], Any
        ] | None = None,
    ):
        self.pg        = pg
        self.op_table  = op_table
        self.op_table["Store"] = lambda a: a  # Store just returns its input
        from ..function_table import FunctionTable
        self.function_table = getattr(pg, "function_table", None)
        if self.function_table is None:
            self.function_table = FunctionTable()
            pg.function_table = self.function_table
        self._code     = None          # str
        self._fn       = None          # compiled callable
        self.signatures = signatures
        # This optional compilation observer receives the already-authoritative
        # ProcessGraph node identity and that node's immediate result.  It is
        # deliberately not a graph builder: it may correlate a backend's
        # primitive implementation occurrence with the planned node while a
        # one-shot forward capture is active, but it cannot add dependencies,
        # infer aliases, inspect values, or change the result.
        self.node_observer = node_observer
    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    
    def build_function(self, device=None) -> Callable:
        """Return (and cache) a function `f(**inputs) -> tuple(outputs)`."""
        if self._fn is not None:
            return self._fn

        src, env, output_names = self._emit_source(device=device)
        code = compile(src, filename="<graph_fn>", mode="exec")
        ns: Dict[str, Any] = {}
        ns.update(env)
        exec(code, ns)
        self._fn = ns["graph_fn"]
        self._code = src
        self._outs = output_names
        return self._fn

    def print_source(self):
        """Print the generated source for the compiled graph."""
        print(self._code)

    @classmethod
    def assemble_function_table(
        cls,
        function_table,
        op_table: Dict[str, Callable],
        signatures: Dict[str, Dict[str, Any]],
        *,
        target: str = "python",
        device=None,
    ):
        """Compile function graphs against predeclared callable slots."""

        from ..function_table import StaticFunctionSlot

        entries = sorted(
            (
                entry
                for entry in function_table
                if entry.graph is not None
            ),
            key=lambda entry: entry.reference.address,
        )
        slots = {
            entry.reference: StaticFunctionSlot(entry.reference)
            for entry in entries
        }
        for reference, slot in slots.items():
            function_table.install_implementation(reference, target, slot)

        definitions = {}
        for entry in entries:
            reference = entry.reference
            if (
                not entry.graph.levels
                or set(entry.graph.levels) != set(entry.graph.G)
            ):
                if not nx.is_directed_acyclic_graph(entry.graph.G):
                    raise RuntimeError(
                        "a function body ProcessGraph must remain acyclic; "
                        "recursion belongs in its Call reference"
                    )
                levels = {}
                for node_id in nx.topological_sort(entry.graph.G):
                    levels[node_id] = max(
                        (
                            levels[parent] + 1
                            for parent in entry.graph.G.predecessors(node_id)
                        ),
                        default=0,
                    )
                entry.graph.levels = levels

            compiler = cls(
                entry.graph,
                dict(op_table),
                signatures,
            )
            raw_definition = compiler.build_function(device=device)
            positional = tuple(
                entry.graph.G.graph.get("positional_parameters", ())
            )
            keyword_only = tuple(
                entry.graph.G.graph.get("keyword_only_parameters", ())
            )

            def bind_definition(
                raw=raw_definition,
                positional_names=positional,
                keyword_only_names=keyword_only,
            ):
                def definition(*args, **kwargs):
                    if len(args) > len(positional_names):
                        raise TypeError(
                            "too many positional arguments for compiled "
                            "ProcessGraph function"
                        )
                    values = dict(zip(positional_names, args))
                    valid_names = {
                        *positional_names,
                        *keyword_only_names,
                    }
                    for name, value in kwargs.items():
                        if name not in valid_names:
                            raise TypeError(
                                f"unexpected keyword argument {name!r}"
                            )
                        if name in values:
                            raise TypeError(
                                f"multiple values for argument {name!r}"
                            )
                        values[name] = value
                    missing = valid_names.difference(values)
                    if missing:
                        names = ", ".join(sorted(missing))
                        raise TypeError(
                            "missing arguments for compiled ProcessGraph "
                            f"function: {names}"
                        )
                    outputs = raw(
                        **{
                            _graph_input_name(name): values[name]
                            for name in valid_names
                        }
                    )
                    return outputs[0] if len(outputs) == 1 else outputs

                return definition

            definition = bind_definition()
            slots[reference].bind(definition)
            entry.metadata[f"{target}_source"] = compiler._code
            definitions[reference] = slots[reference]

        return definitions

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _emit_source(self, *, device):
        """Generate python source for the graph and a globals‑env dict."""
        G        = self.pg.G
        levels   = self.pg.levels               # nid -> level idx
        max_lvl  = max(levels.values())
        run_order: List[int] = []
        for lv in range(max_lvl+1):
            # deterministic order inside a level -> sorted nids
            run_order.extend(sorted(n for n,l in levels.items() if l==lv))

        lines: List[str] = ["def graph_fn(**inputs):"]
        env: Dict[str, Any] = {}
        indent = " " * 4
        if self.node_observer is not None:
            env["_observe_process_graph_node"] = self.node_observer

        def observe(
            nid: int,
            parent_ids: tuple[int, ...],
            expression: str,
        ) -> str:
            if self.node_observer is None:
                return expression
            def operand_leaves(parent_id: int) -> tuple[int, ...]:
                parent = G.nodes[int(parent_id)]
                if parent.get("type") not in {"List", "Tuple"}:
                    return (int(parent_id),)
                leaves = []
                for child, role in parent.get("parents", ()):
                    if str(role) in {"elts", "element", "item"}:
                        leaves.extend(operand_leaves(int(child)))
                return tuple(leaves)

            parent_ids = tuple(
                leaf
                for parent_id in parent_ids
                for leaf in operand_leaves(int(parent_id))
            )
            parent_values = (
                "("
                + ", ".join(f"v{parent}" for parent in parent_ids)
                + ("," if len(parent_ids) == 1 else "")
                + ")"
            )
            return (
                f"_observe_process_graph_node("
                f"{int(nid)}, {tuple(map(int, parent_ids))!r}, "
                f"{parent_values}, "
                f"{expression})"
            )


        for nid in run_order:
            node = G.nodes[nid]
            for k in self._REQ:
                if k not in node:
                    raise KeyError(f"ProcessGraph node missing '{k}' field")

            ntype = node["type"]
            sig = self.signatures.get(ntype, {})
            role_parents = list(node["parents"])

            lhs   = f"v{nid}"           # unique local name

            if ntype in ("Symbol", "Input", "Var", "IndexedBase", "Integer", "NegativeOne", "One", "Zero"):
                # pure argument
                label = node["label"]
                if ntype in ("Symbol", "Input", "Var"):
                    lines.append(
                        f"{indent}{lhs} = inputs[{_graph_input_name(label)!r}]"
                    )
                elif ntype in ("IndexedBase"):
                    lines.append(f"{indent}{lhs} = inputs['domain{''.join(self.pg.G.nodes[nid]['domain_shape'])}{label}']")
                else:
                    lines.append(f"{indent}{lhs} = {label}")
                continue

            if ntype in ("Const", "Constant"):
                if "constant" in node:
                    literal = node["constant"]
                else:
                    expression = node.get("expr_obj")
                    if not hasattr(expression, "value"):
                        raise KeyError(
                            f"{ntype} node {nid} has no literal payload"
                        )
                    literal = expression.value
                # IntEnum reprs name their defining Python type and are not
                # valid standalone source expressions.  The graph constant is
                # the integer value; do not retain or resolve the enum class
                # as a runtime symbol in the compiled shell.
                if isinstance(literal, enum.IntEnum):
                    literal = int(literal)
                if isinstance(literal, str):
                    literal = literal.encode("utf-8")
                if not _is_emittable_literal(literal):
                    raise TypeError(
                        f"{ntype} node {nid} has unsupported literal type "
                        f"{type(literal).__name__}"
                    )
                lines.append(f"{indent}{lhs} = {literal!r}")
                continue
                

            if ntype in ("Add", "Mul", "Sub", "Div", "Pow"):
                op_map = {"Add": "+", "Mul": "*", "Sub": "-", "Div": "/", "Pow": "**"}[ntype]
                # these are simple operators we can directly code them
                lhs = f"v{nid}"
                rhs = f" {op_map} ".join(f"v{pid}" for pid, _ in node["parents"])
                lines.append(
                    f"{indent}{lhs} = {observe(nid, tuple(parent for parent, _role in role_parents), rhs)}"
                )
                continue
            elif ntype in {"List", "Tuple"}:
                elements = ", ".join(
                    f"v{parent}"
                    for parent, role in role_parents
                    if str(role) in {"elts", "element", "item"}
                )
                if ntype == "List":
                    expression = f"[{elements}]"
                else:
                    expression = (
                        f"({elements},)"
                        if len([
                            parent
                            for parent, role in role_parents
                            if str(role) in {"elts", "element", "item"}
                        ]) == 1
                        else f"({elements})"
                    )
                lines.append(
                    f"{indent}{lhs} = {observe(nid, tuple(parent for parent, _role in role_parents), expression)}"
                )
                continue
            elif ntype == "Call":
                attributes = dict(node.get("attributes") or {})
                callee_ref = attributes.get("callee_ref")
                target = None
                if callee_ref is not None:
                    entry = self.function_table.entry(callee_ref)
                    target = self.function_table.implementation(
                        callee_ref,
                        "python",
                    )
                    if target is None:
                        target = entry.python_callable

                if target is None:
                    callee = next(
                        (
                            parent
                            for parent, role in role_parents
                            if role in {"func", "callee"}
                        ),
                        None,
                    )
                    if callee is None:
                        raise KeyError(
                            f"Call node {nid} has no resolved function "
                            "reference or callee input"
                        )
                    callee_expression = f"v{callee}"
                else:
                    callee_name = f"function_{callee_ref}"
                    env[callee_name] = target
                    callee_expression = callee_name

                positional = [
                    f"v{parent}"
                    for parent, role in role_parents
                    if role in {"args", "arg"}
                    or str(role).startswith("arg:")
                ]
                keywords = []
                unresolved_keywords = []
                for parent, role in role_parents:
                    role = str(role)
                    if role.startswith("kw:"):
                        keywords.append(
                            f"{role[3:]}=v{parent}"
                        )
                    elif role in {"keywords", "keyword"}:
                        unresolved_keywords.append(parent)
                if unresolved_keywords:
                    raise KeyError(
                        f"Call node {nid} has unresolved AST keyword nodes "
                        f"{unresolved_keywords}"
                    )
                arguments = ", ".join((*positional, *keywords))
                lines.append(
                    f"{indent}{lhs} = "
                    f"{observe(nid, tuple(parent for parent, role in role_parents if role not in {'func', 'callee'}), f'{callee_expression}({arguments})')}"
                )
                continue
            else:
                # operator
                fn = self.op_table.get(ntype)
                if fn is None:
                    raise KeyError(f"No operator impl for '{ntype}'")
                fn_name = f"op_{nid}"
                env[fn_name] = fn

                # parents come in topo order already
                keyword_parents = [
                    (parent, str(role)[3:])
                    for parent, role in role_parents
                    if str(role).startswith("kw:")
                ]
                positional_parents = [
                    parent
                    for parent, role in role_parents
                    if not str(role).startswith("kw:")
                ]
                if keyword_parents:
                    args = ", ".join((
                        *(f"v{parent}" for parent in positional_parents),
                        *(
                            f"{name}=v{parent}"
                            for parent, name in keyword_parents
                        ),
                    ))
                elif sig.get("min_inputs",None) is None and sig.get("max_inputs",None) is None and sig.get("min_outputs",None) is None and sig.get("max_outputs",None) is None:
                    args = f"[{', '.join(f'v{pid}' for pid,_ in node['parents'])}]"
                else:
                    args = ", ".join(f"v{pid}" for pid, _ in node["parents"])
                lines.append(
                    f"{indent}{lhs} = "
                    f"{observe(nid, tuple((*positional_parents, *(parent for parent, _name in keyword_parents))), f'{fn_name}({args})')}"
                )

        #  final return – collect nodes marked as outputs / Store
        outputs = [n for n, data in G.nodes(data=True)
                    if data.get("type") in ("Store", "Output")]
        if not outputs:
            # fallback: last node in topo order
            outputs = [run_order[-1]]
        out_expr = ", ".join(f"v{n}" for n in outputs)
        lines.append(f"{indent}return ({out_expr},)\n")
        print("\n".join(lines))
        return textwrap.dedent("\n".join(lines)), env, outputs
    def emit_cffi_source(self):
        """
        Generate C source + CFFI bindings for the current graph.

        Returns:
        c_source: str       # full C code for graph_fn
        cdef_text: str      # CFFI cdef declarations
        py_loader: str      # Python snippet to verify and load via CFFI
        output_indices: List[int]  # indices of outputs in enumeration
        """
        G = self.pg.G
        levels = self.pg.levels
        # Topological order
        max_lvl = max(levels.values())
        topo = []
        for lv in range(max_lvl + 1):
            topo.extend(sorted(n for n, l in levels.items() if l == lv))

        # Collect distinct input labels
        inputs = []
        for nid in topo:
            node = G.nodes[nid]
            if node['type'] in ('Input', 'Symbol', 'Var'):
                lab = node['label']
                if lab not in inputs:
                    inputs.append(lab)

        # Build C enum for inputs
        enum_lines = ['typedef enum {']
        for idx, lab in enumerate(inputs):
            enum_lines.append(f'    IDX_{lab.upper()} = {idx},')
        enum_lines.append(f'    N_INPUTS = {len(inputs)}')
        enum_lines.append('} input_idx_t;')

        # Begin C source
        c_lines = []
        c_lines.append('#include <stddef.h>')
        c_lines.append('#include "ctensor_ops.h"   // user-provided op implementations')
        c_lines.append('\n'.join(enum_lines))
        c_lines.append('')
        c_lines.append('void graph_fn(const double *inputs, double *outputs, size_t n) {')
        c_lines.append('    // per-index pointers')
        c_lines.append('    const double *inp[N_INPUTS];')
        c_lines.append('    for (size_t i = 0; i < N_INPUTS; ++i) inp[i] = inputs + i*n;')

        # Emit node computations
        for nid in topo:
            node = G.nodes[nid]
            lhs = f'double *v{nid} = NULL;'
            if node['type'] in ('Input', 'Symbol', 'Var'):
                lab = node['label']
                idx = inputs.index(lab)
                c_lines.append(f'    // input {lab}')
                c_lines.append(f'    v{nid} = (double *)inp[IDX_{lab.upper()}];')
            else:
                # operator case
                fn = self.op_table[node['type']].__name__
                parents = node['parents']
                args = ', '.join(f'v{pid}[i]' for pid, _ in parents)
                c_lines.append(f'    // node {nid}: {node["type"]}')
                c_lines.append(f'    v{nid} = malloc(n * sizeof(double));  // temp buffer')
                c_lines.append(f'    for (size_t i = 0; i < n; ++i)')
                c_lines.append(f'        v{nid}[i] = {fn}({args});')

        # Write outputs
        out_nodes = [n for n, d in G.nodes(data=True) if d['type'] in ('Store','Output')]
        if not out_nodes:
            out_nodes = [topo[-1]]
        c_lines.append('    // write outputs')
        for idx, nid in enumerate(out_nodes):
            c_lines.append(f'    for (size_t i = 0; i < n; ++i) outputs[{idx}*n + i] = v{nid}[i];')
        c_lines.append('}')

        c_source = '\n'.join(c_lines)

        # CFFI cdef
        cdef_lines = [
            'typedef enum {',
        ]
        for idx, lab in enumerate(inputs):
            cdef_lines.append(f'    IDX_{lab.upper()} = {idx},')
        cdef_lines.append(f'    N_INPUTS = {len(inputs)}')
        cdef_lines.append('} input_idx_t;')
        cdef_lines.append('void graph_fn(const double *inputs, double *outputs, size_t n);')
        cdef_text = '\n'.join(cdef_lines)

        # Python loader snippet
        py_loader = f"""
    from cffi import FFI
    ffi = FFI()
    ffi.cdef(r'''{cdef_text}''')
    C = ffi.verify(r'''
    {c_source}
    ''',
        extra_compile_args=['-O2'],
    )
    # C.graph_fn now available
    """

        return c_source, cdef_text, py_loader, out_nodes

    # ------------------------------------------------------------------
    # misc helpers / diagnostics
    # ------------------------------------------------------------------
    def code(self) -> str:
        """Return generated source as text (compiles lazily)."""
        if self._code is None:
            self.build_function()
        return self._code

    def hash(self) -> str:
        """Return a stable hash of the generated source (after build)."""
        src = self.code().encode()
        return hashlib.sha1(src).hexdigest()

# ────────────────────────────────────────────────────────────────────────
# quick self‑test  (run as `python deep_graph_compiler.py`)
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sympy as sp, numpy as np
    from graph_express2 import ProcessGraph
    # toy graph x + y * z
    pg = ProcessGraph()
    x,y,z = sp.symbols("x y z")
    expr = x + y*z
    pg.build_from_expression(expr)
    pg.compute_levels("asap")

    # ops for numpy
    op_table = {
        "Mul": lambda a,b: a*b,
        "Add": lambda a,b: a+b,
    }

    compiler = GraphDeepCompiler(pg, op_table)
    f = compiler.build_function()
    # data
    X = np.array([1,2,3])
    Y = np.array([10,20,30])
    Z = np.array([2,2,2])
    out, = f(x=X, y=Y, z=Z)
    print("result", out)
