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

import textwrap, inspect, hashlib, types
from typing import Any, Callable, Dict, List, Tuple

class GraphDeepCompiler:
    """Compile a *level‑sorted* ProcessGraph into one Python function."""

    #: attribute names we expect on ProcessGraph nodes
    _REQ = ("type", "label", "parents")

    def __init__(self, pg: "ProcessGraph", op_table: Dict[str, Callable], signatures: Dict[str, Dict[str, Any]]):
        self.pg        = pg
        self.op_table  = op_table
        self.op_table["Store"] = lambda a: a  # Store just returns its input
        self._code     = None          # str
        self._fn       = None          # compiled callable
        self.signatures = signatures
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


        for nid in run_order:
            node = G.nodes[nid]
            for k in self._REQ:
                if k not in node:
                    raise KeyError(f"ProcessGraph node missing '{k}' field")

            ntype = node["type"]
            sig = self.signatures.get(ntype, {})

            lhs   = f"v{nid}"           # unique local name

            if ntype in ("Symbol", "Input", "Var", "IndexedBase", "Integer", "NegativeOne", "One", "Zero"):
                # pure argument
                label = node["label"]
                def parse_guess_type_from_string_content(s):
                    try:
                        int(s)
                        return "int"
                    except Exception:
                        try:
                            float(s)
                            return "float"
                        except Exception:
                            # Variable name heuristics (single-letter or known schemes)
                            import re

                            def parse_common_variable_schemes(name):
                                name = name.strip()
                                name_lower = name.lower()
                                
                                # Regex: root letters, then digits or _digits (subscript/numeral)
                                m = re.match(r"([a-zA-Z]+)[_\d]*$", name_lower)
                                root = m.group(1) if m else name_lower

                                # Single-letter/symbolic float variables
                                if root in ("x", "y", "z", "t", "a", "b", "c"):
                                    return "float"
                                # Single-letter int/index variables
                                if root in ("i", "j", "k", "l", "m", "n"):
                                    return "int"
                                # Explicit int hints (common for indexing, etc.)
                                if (
                                    name_lower.startswith("num") or
                                    name_lower.endswith("idx") or
                                    root in ("n", "k") or
                                    name_lower.isdigit()
                                ):
                                    return "int"
                                # Explicit float hints
                                if (
                                    name_lower.startswith("float") or
                                    name_lower.endswith("val") or
                                    name_lower.endswith("amp") or
                                    name_lower.endswith("pos")
                                ):
                                    return "float"
                                # Boolean flags
                                if name_lower.startswith("is_") or name_lower.startswith("has_"):
                                    return "bool"
                                # Fallback
                                return "float"
                            return parse_common_variable_schemes(s)
                if ntype in ("Symbol", "Input", "Var"):
                    lines.append(f"{indent}{lhs} = inputs['{parse_guess_type_from_string_content(label)}{label}']")
                elif ntype in ("IndexedBase"):
                    lines.append(f"{indent}{lhs} = inputs['domain{''.join(self.pg.G.nodes[nid]['domain_shape'])}{label}']")
                else:
                    lines.append(f"{indent}{lhs} = {label}")
                continue
                

            if ntype in ("Add", "Mul", "Sub", "Div", "Pow"):
                op_map = {"Add": "+", "Mul": "*", "Sub": "-", "Div": "/", "Pow": "**"}[ntype]
                # these are simple operators we can directly code them
                lhs = f"v{nid}"
                rhs = f" {op_map} ".join(f"v{pid}" for pid, _ in node["parents"])
                lines.append(f"{indent}{lhs} = {rhs}")
                continue
            else:
                # operator
                fn = self.op_table.get(ntype)
                if fn is None:
                    raise KeyError(f"No operator impl for '{ntype}'")
                fn_name = f"op_{nid}"
                env[fn_name] = fn

                # parents come in topo order already
                if sig.get("min_inputs",None) is None and sig.get("max_inputs",None) is None and sig.get("min_outputs",None) is None and sig.get("max_outputs",None) is None:
                    args = f"[{', '.join(f'v{pid}' for pid,_ in node['parents'])}]"
                else:
                    args = ", ".join(f"v{pid}" for pid, _ in node["parents"])
                lines.append(f"{indent}{lhs} = {fn_name}({args})")

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
