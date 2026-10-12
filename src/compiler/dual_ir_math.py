"""Composable exact mathematics over the compiler's numeric/control IR.

``MathProgram`` is a boundary view, not another instruction language: its
numeric body is a ProcessGraph, its schedule is a ControlProgram, and an
optional source DualIRShell retains the map/ownership/hierarchy description.
Transformations produce the same pair and post their causes to concordance.

The symbolic component lane currently admits pure, straight-line schedules
and explicit scalar components. It never treats a captured loop/branch/effect
as its whole-program mathematical meaning. ``reverse`` uses the established
controlled adjoint for those programs instead. Coordinate Hessians are not
covariant Hessians. PDE methods construct residual functions, not PDE solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import networkx as nx
import sympy as sp
from sympy.core.function import AppliedUndef

from .control_source import ControlProgram, SequenceBlock, StatementBlock
from .process_graph_autograd import (
    _graph_node_cell, differentiate_process_program,
)
from .symbolic_process_graph import (
    ingest_sympy_expressions, process_graph_to_sympy_expressions,
)
from .concordance_declarations import (
    GRAPH_MATH, GRAPH_MATH_OUTPUT, GRAPH_MATH_SOURCE, MathOutputFact,
)
from .identity_concordance import (
    Derived, Mode, Novel, current_identity_book,
)
from ..transmogrifier.graph.graph_express2 import ProcessGraph


class MathOperatorError(ValueError):
    """A requested transformation has no established exact contract."""


def _scheduled_regions(block):
    if isinstance(block, SequenceBlock):
        return tuple(region for child in block.blocks for region in _scheduled_regions(child))
    if isinstance(block, StatementBlock):
        regions = []
        for line in block.lines:
            text = str(line)
            prefix, suffix = "__scheduled_region_", "__"
            if not text.startswith(prefix) or not text.endswith(suffix):
                raise MathOperatorError("symbolic mathematics cannot discard executable statements")
            middle = text[len(prefix):-len(suffix)]
            if not middle.isdecimal():
                raise MathOperatorError("invalid scheduled region identity")
            regions.append(int(middle))
        return tuple(regions)
    raise MathOperatorError(
        f"symbolic component transformation cannot discard {type(block).__name__}; "
        "use the controlled reverse contract or select an explicit pure region"
    )


@dataclass(frozen=True)
class MathProgram:
    """An encapsulated function using the existing dual IR and port identities.

    ``shape`` is the mathematical result layout, in row-major component order;
    it does not assert a new native array ABI. All original scalar parameters
    remain declared even when differentiation makes their numerical use zero.
    ``source_shell`` is retained by reference, never reconstructed from names.
    """

    numeric: ProcessGraph
    control: ControlProgram
    region_nodes: Mapping[int, tuple[int, ...]]
    input_ids: Mapping[str, int]
    shape: tuple[int, ...]
    source_shell: Any = None
    sources: tuple["MathProgram", ...] = ()
    identity_book: Any = None

    def __post_init__(self):
        shape = tuple(self.shape)
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in shape):
            raise MathOperatorError("result dimensions must be positive integers")
        if prod(shape) != len(self.numeric.roots):
            raise MathOperatorError("result layout disagrees with numeric output ports")
        ownership = [int(node) for nodes in self.region_nodes.values() for node in nodes]
        if len(ownership) != len(set(ownership)) or set(ownership) != set(self.numeric.G):
            raise MathOperatorError("numeric nodes must have exactly one control-region owner")
        inputs = {str(name): int(node) for name, node in self.input_ids.items()}
        if len(set(inputs.values())) != len(inputs):
            raise MathOperatorError("input ports cannot alias distinct declarations")
        for name, node in inputs.items():
            if node not in self.numeric.G or str(self.numeric.G.nodes[node].get("op")).lower() != "input":
                raise MathOperatorError(f"input {name!r} has no numeric input identity")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "input_ids", MappingProxyType(inputs))
        object.__setattr__(self, "region_nodes", MappingProxyType({
            int(region): tuple(map(int, nodes)) for region, nodes in self.region_nodes.items()
        }))

    @classmethod
    def bind(cls, numeric, control, *, region_nodes, input_ids, shape=(), source_shell=None):
        """View an existing dual-IR function without copying its bodies or maps."""
        return cls(numeric, control, region_nodes, input_ids, tuple(shape), source_shell,
                   identity_book=current_identity_book())

    @classmethod
    def from_compilation(cls, compilation, *, region_nodes, input_ids, shape=()):
        """Use the existing AOT shell, map, ownership and function graph as-is.

        Region/port IDs must be the planner's explicit node identities, not
        reconstructed from observed values or hierarchical SSA positions.
        """
        return cls.bind(compilation.deployment.process_graph, compilation.shell_control_program,
                        region_nodes=region_nodes, input_ids=input_ids, shape=shape,
                        source_shell=compilation.shell)

    @classmethod
    def symbolic(cls, expressions, *, inputs, shape=None, source_shell=None):
        """Author a pure function through the existing symbolic graph importer."""
        values = tuple(sp.sympify(e) for e in expressions)
        coordinates = tuple(sp.sympify(x) for x in inputs)
        return _make_program(values, coordinates, shape, "authored", (), source_shell)

    def symbols(self):
        result = {}
        for name, node in self.input_ids.items():
            data = self.numeric.G.nodes[node]
            authored = data.get("expr_obj")
            result[name] = authored if isinstance(authored, sp.Symbol) else sp.Symbol(name)
        return MappingProxyType(result)

    def expressions(self):
        """A checked derived mathematical view; never a second authority."""
        scheduled = _scheduled_regions(self.control.root)
        if (set(scheduled) != set(self.region_nodes) or len(scheduled) != len(set(scheduled))
                or set(self.control.region_indices) != set(self.region_nodes)):
            raise MathOperatorError("symbolic view requires exactly-once execution of every numeric region")
        from dataclasses import fields
        if any(getattr(self.control, field.name) for field in fields(self.control)
               if field.name not in {"root", "region_indices"}):
            raise MathOperatorError("control aliases, uniforms or deployment effects need an explicit mathematical contract")
        if not nx.is_directed_acyclic_graph(self.numeric.G):
            raise MathOperatorError("a recurrence is not an unconditional symbolic expression")
        order = {node: i for i, region in enumerate(scheduled) for node in self.region_nodes[region]}
        if any(order[u] > order[v] for u, v in self.numeric.G.edges()):
            raise MathOperatorError("control schedule reverses a numerical dependency")
        declared = set(self.input_ids.values())
        actual = {n for n, d in self.numeric.G.nodes(data=True) if str(d.get("op")).lower() == "input"}
        if actual != declared:
            raise MathOperatorError("all numerical inputs need explicit boundary identities")
        for node, data in self.numeric.G.nodes(data=True):
            tensor = data.get("tensor") or {}
            if tuple(tensor.get("shape", (None,))) != ():
                raise MathOperatorError(f"node {node} needs an explicit scalar-component contract")
            if (data.get("control") or (data.get("attributes") or {}).get("mutable")
                    or str(data.get("op")).lower() in {"store", "setattr", "indexedstore", "return"}):
                raise MathOperatorError(f"node {node} is not a pure mathematical value")
        result = process_graph_to_sympy_expressions(self.numeric)
        replacements = {sp.Symbol(name): symbol for name, symbol in self.symbols().items()}
        result = tuple(e.xreplace(replacements) for e in result)
        if any(e.has(AppliedUndef, sp.Derivative, sp.Integral, sp.Sum) for e in result):
            raise MathOperatorError("opaque calls or unresolved symbolic operations need a declared mathematical rule")
        if set().union(*(e.free_symbols for e in result)) - set(self.symbols().values()):
            raise MathOperatorError("rendered mathematics introduced undeclared boundary inputs")
        return result

    def _axes(self, names=None, *, repeated=False):
        selected = tuple(self.input_ids) if names is None else tuple(names)
        if not selected or (not repeated and len(set(selected)) != len(selected)):
            raise MathOperatorError("derivative coordinates must be nonempty and unique")
        unknown = set(selected) - set(self.input_ids)
        if unknown:
            raise MathOperatorError(f"unknown coordinate ports: {sorted(unknown)}")
        return tuple(self.symbols()[name] for name in selected)

    def _result(self, expressions, *, shape=None, operation, sources=(), inputs=None):
        parents = (self, *sources)
        coordinates = tuple(self.symbols().values()) if inputs is None else tuple(inputs)
        if any(parent.identity_book is not self.identity_book for parent in parents):
            raise MathOperatorError("composition across identity books requires an explicit identity import")
        from .identity_concordance import begin_identity_book, end_identity_book
        _, token = begin_identity_book(self.identity_book)
        try:
            return _make_program(tuple(expressions), coordinates, shape, operation, parents, self.source_shell)
        finally:
            end_identity_book(token=token)

    def reverse(self, *, outputs=None, wrt=None, **contracts):
        """Keep the existing logical adjoint, region ownership and binding ledger."""
        return differentiate_process_program(
            self.numeric, self.control, region_nodes=self.region_nodes,
            outputs=outputs, wrt=wrt, **contracts,
        )

    def partial(self, *coordinates):
        axes = self._axes(coordinates, repeated=True)
        return self._result((sp.diff(e, *axes) for e in self.expressions()),
                            shape=self.shape, operation="partial")

    def jacobian(self, *, wrt=None):
        axes = self._axes(wrt)
        entries = tuple(sp.diff(e, x) for e in self.expressions() for x in axes)
        return self._result(entries, shape=(*self.shape, len(axes)), operation="jacobian")

    def gradient(self, *, wrt=None):
        if self.shape:
            raise MathOperatorError("gradient requires a scalar function")
        return self.jacobian(wrt=wrt)

    def hessian(self, *, wrt=None):
        if self.shape:
            raise MathOperatorError("coordinate Hessian requires a scalar function")
        return self.gradient(wrt=wrt).jacobian(wrt=wrt)

    def divergence(self, *, wrt=None):
        axes = self._axes(wrt)
        if self.shape != (len(axes),):
            raise MathOperatorError("divergence needs one vector component per coordinate")
        value = sum(sp.diff(e, x) for e, x in zip(self.expressions(), axes))
        return self._result((value,), shape=(), operation="divergence")

    def curl(self, *, wrt):
        axes = self._axes(wrt)
        if len(axes) != 3 or self.shape != (3,):
            raise MathOperatorError("coordinate curl requires three coordinates and components")
        a, b, c = self.expressions(); x, y, z = axes
        return self._result((sp.diff(c, y)-sp.diff(b, z), sp.diff(a, z)-sp.diff(c, x),
                             sp.diff(b, x)-sp.diff(a, y)), shape=(3,), operation="curl")

    def laplacian(self, *, wrt=None):
        axes = self._axes(wrt)
        return self._result((sum(sp.diff(e, x, 2) for x in axes) for e in self.expressions()),
                            shape=self.shape, operation="coordinate_laplacian")

    def biharmonic(self, *, wrt=None):
        return self.laplacian(wrt=wrt).laplacian(wrt=wrt)

    def directional_derivative(self, direction, *, wrt=None):
        axes = self._axes(wrt)
        vector = tuple(sp.sympify(v) for v in direction)
        if len(vector) != len(axes):
            raise MathOperatorError("direction must match derivative coordinates")
        return self._result((sum(v*sp.diff(e, x) for v, x in zip(vector, axes))
                             for e in self.expressions()), shape=self.shape, operation="directional_derivative")

    def linear_operator(self, *, seeds):
        """Extract exact symbolic coefficients, never evaluate numeric seeds.

        For B(x,v)=J(x)^T v this returns J(x)^T. Unregistered call bodies
        remain refused by expressions(); no derivative meaning is invented.
        """
        axes = self._axes(seeds)
        entries = []
        for expression in self.expressions():
            try:
                polynomial = sp.Poly(expression, *axes, domain="EX")
            except sp.PolynomialError as error:
                raise MathOperatorError("seed dependence is not an established linear polynomial") from error
            if polynomial.total_degree() > 1 or polynomial.coeff_monomial((0,)*len(axes)) != 0:
                raise MathOperatorError("seed dependence must be homogeneous and linear")
            entries.extend(polynomial.coeff_monomial(tuple(int(i == j) for i in range(len(axes))))
                           for j in range(len(axes)))
        inputs = tuple(x for x in self.symbols().values() if x not in axes)
        return self._result(entries, shape=(*self.shape, len(axes)), inputs=inputs,
                            operation="linear_seed_coefficients")

    def transpose(self):
        if len(self.shape) != 2:
            raise MathOperatorError("transpose requires a two-axis result layout")
        rows, columns = self.shape; values = self.expressions()
        return self._result((values[i*columns+j] for j in range(columns) for i in range(rows)),
                            shape=(columns, rows), operation="transpose")

    def substitute(self, bindings: Mapping[str, "MathProgram"]):
        replacements = {}
        retained = [symbol for name, symbol in self.symbols().items() if name not in bindings]
        for name, program in bindings.items():
            if name not in self.input_ids or program.shape:
                raise MathOperatorError("substitution requires a declared input and scalar function")
            replacements[self.symbols()[name]] = program.expressions()[0]
            retained.extend(program.symbols().values())
        return self._result((e.xreplace(replacements) for e in self.expressions()), shape=self.shape,
                            operation="substitute", sources=tuple(bindings.values()),
                            inputs=tuple(dict.fromkeys(retained)))

    def compose(self, inner: "MathProgram"):
        """self(inner(...)), with ordered boundary ports and shared expressions."""
        if len(inner.numeric.roots) != len(self.input_ids):
            raise MathOperatorError("composition output/input arities differ")
        replacements = dict(zip(self.symbols().values(), inner.expressions()))
        return self._result((e.xreplace(replacements) for e in self.expressions()),
                            shape=self.shape, operation="compose", sources=(inner,),
                            inputs=tuple(inner.symbols().values()))

    def heat(self, *, time, space, diffusivity=1):
        """u_t - k Delta u; scalar constant k, no time integrator."""
        return self._pde(time=time, space=space, coefficient=diffusivity, order=1)

    def wave(self, *, time, space, speed_squared=1):
        """u_tt - c^2 Delta u; scalar constant c^2, no solver."""
        return self._pde(time=time, space=space, coefficient=speed_squared, order=2)

    def poisson(self, forcing=0, *, space):
        """Delta u - f: coordinate-space sign, not DEC's positive Laplacian."""
        if self.shape:
            raise MathOperatorError("Poisson residual requires a scalar field")
        u, = self.expressions()
        axes = self._axes(space)
        return self._result((sum(sp.diff(u, x, 2) for x in axes)-sp.sympify(forcing),),
                            shape=(), operation="poisson")

    def helmholtz(self, *, space, wave_number_squared=1):
        if self.shape:
            raise MathOperatorError("Helmholtz residual requires a scalar field")
        u, = self.expressions(); axes = self._axes(space)
        value = sum(sp.diff(u, x, 2) for x in axes)+sp.sympify(wave_number_squared)*u
        return self._result((value,), shape=(), operation="helmholtz")

    def _pde(self, *, time, space, coefficient, order):
        if self.shape or time in space:
            raise MathOperatorError("PDE residual needs a scalar field and distinct time/space ports")
        t, = self._axes((time,)); axes = self._axes(space)
        u, = self.expressions(); coefficient = sp.sympify(coefficient)
        if coefficient.free_symbols - set(self.symbols().values()):
            raise MathOperatorError("coefficient requires an existing input port")
        value = sp.diff(u, t, order)-coefficient*sum(sp.diff(u, x, 2) for x in axes)
        return self._result((value,), shape=(), operation="heat" if order == 1 else "wave")

    def lower_source(self, *, name="math_operator", **compiler_options):
        """Link the graph into a real source wrapper at the sanctioned entry."""
        from .fortran_c_shell import lower_ast_source_to_ssa

        _identifier(name)
        names = tuple(self.input_ids)
        for parameter in names:
            _identifier(parameter)
        linked = self.linkable_graph(name=f"{name}_body")
        call = f"{name}_body({', '.join(names)})"
        if len(self.numeric.roots) == 1:
            body = f"    return {call}\n"
        else:
            body = f"    values = {call}\n    return " + ", ".join(
                f"values[{i}]" for i in range(len(self.numeric.roots))) + "\n"
        source = f"def {name}({', '.join(names)}):\n{body}"
        return lower_ast_source_to_ssa(source, name, name=name,
                    linked_process_graphs={f"{name}_body": linked}, **compiler_options)

    def linkable_graph(self, *, name="math_body"):
        """Realize the declared function boundary using existing symbolic typing.

        Input echoes ensure constant derivative functions retain their original
        parameter boundary. Only the selected result roots are public; no SSA
        instruction is edited. The source compiler plans the linked graph anew.
        """
        from .symbolic_equation_compiler import compile_sympy_equations
        _identifier(name)
        values = self.expressions()
        symbols = tuple(self.symbols().values())
        output_names = tuple(f"math_result_{i}" for i in range(len(values)))
        echoes = tuple(f"math_parameter_echo_{i}" for i in range(len(symbols)))
        if set((*output_names, *echoes)) & set(self.input_ids):
            raise MathOperatorError("internal result names collide with an input port")
        equations = tuple(sp.Eq(sp.Symbol(n), v, evaluate=False)
                          for n, v in zip((*output_names, *echoes), (*values, *symbols)))
        linked = compile_sympy_equations(equations, name=name).process_graph
        roots = tuple(linked.G.graph["identity_table"][n][-1] for n in output_names)
        linked.roots = list(roots)
        linked.G.graph.update(function_parameters=tuple(self.input_ids),
                              positional_parameters=tuple(self.input_ids),
                              function_outputs=output_names, deployment_outputs=roots)
        linked.G.graph["identity_table"] = {
            n: ids for n, ids in linked.G.graph["identity_table"].items() if n not in echoes
        }
        return linked


def _identifier(name):
    import keyword
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise MathOperatorError(f"not a source parameter/function identifier: {name!r}")


def _make_program(values, inputs, shape, operation, sources, shell):
    if not values:
        raise MathOperatorError("a mathematical function needs an output")
    if any(not isinstance(x, sp.Symbol) for x in inputs) or len(set(inputs)) != len(inputs):
        raise MathOperatorError("inputs must be unique declared scalar symbols")
    if len({str(x) for x in inputs}) != len(inputs):
        raise MathOperatorError("distinct symbols cannot share a boundary spelling")
    if set().union(*(e.free_symbols for e in values)) - set(inputs):
        raise MathOperatorError("mathematical output uses an undeclared input")
    shape = (() if len(values) == 1 else (len(values),)) if shape is None else tuple(shape)
    if prod(shape) != len(values):
        raise MathOperatorError("output component count disagrees with result layout")
    book = current_identity_book()
    scope = book.mint_scope("graph_math", GRAPH_MATH)
    causes = tuple(dict.fromkeys(cell for source in sources
        for node in (*source.numeric.roots, *source.input_ids.values())
        for cell in (_graph_node_cell(source.numeric, int(node)),) if cell is not None))
    if sources and not causes:
        raise MathOperatorError("source program has no concordance identities for this transformation")
    cells = tuple(book.post(
        GRAPH_MATH_OUTPUT, (scope, i), MathOutputFact(operation, shape, i, sp.srepr(value)),
        stage=GRAPH_MATH, provenance=Derived(causes) if sources else Novel(GRAPH_MATH_SOURCE, ()),
        mode=Mode.CONCORD,
    ) for i, value in enumerate(values))
    graph = ProcessGraph(materialize_memory=False)
    # Append input declarations to the import envelope so zero derivatives
    # retain their ABI and can be differentiated again. They are not outputs.
    names = tuple(f"result_{i}" for i in range(len(values)))
    roots = ingest_sympy_expressions(graph, (*values, *inputs),
        output_names=(*names, *(f"__parameter_{i}" for i in range(len(inputs)))), strict=True,
        expression_sources=(*cells, *(cells[0] for _ in inputs)))[:len(values)]
    graph.roots = list(roots)
    parameters = {str(d["attributes"]["binding_name"]): int(n)
                  for n, d in graph.G.nodes(data=True) if d.get("op") == "input"}
    parameters = {str(x): parameters[str(x)] for x in inputs}
    for _, data in graph.G.nodes(data=True):
        data["tensor"] = {"shape": (), "dtype": "float64"}
    graph.G.graph.update(deployment_outputs=roots, function_outputs=names,
        function_parameters=tuple(parameters), positional_parameters=tuple(parameters),
        identity_table={**{n:(i,) for n,i in parameters.items()}, **dict(zip(names, ((i,) for i in roots)))},
        math_result_shape=shape, math_operator_cells=cells,
        math_operator_scope=scope, semantic_authority="ProcessGraph")
    control = ControlProgram(StatementBlock(("__scheduled_region_0__",)), region_indices=(0,))
    return MathProgram(graph, control, {0:tuple(graph.G)}, parameters, shape, shell, tuple(sources), book)


@dataclass(frozen=True)
class DECGraphDiagram:
    """Edge constitutive functions + joint functions on an existing DECSystem.

    This is a wiring view of CellComplex.edges, not a new topology or DEC
    implementation. Source lowering retains each local law as a linked
    ProcessGraph function. Symbolic inspection can expand the same wiring.
    Hodge weights are runtime inputs, not baked numerical geometry.
    """

    system: Any
    edge_laws: Mapping[int, MathProgram]
    joint_laws: Mapping[int, MathProgram]

    @classmethod
    def laplace0(cls, system, *, edge_laws=None, joint_laws=None):
        tail, head, weight = sp.symbols("tail head weight")
        accumulation, measure = sp.symbols("accumulation measure")
        edge = MathProgram.symbolic((weight*(head-tail),), inputs=(tail, head, weight))
        joint = MathProgram.symbolic((accumulation/measure,), inputs=(accumulation, measure))
        edges = {i:edge for i in range(system.complex.edge_count)}
        joints = {i:joint for i in range(system.complex.vertex_count)}
        for supplied, target in ((edge_laws or {}, edges), (joint_laws or {}, joints)):
            if set(supplied)-set(target):
                raise MathOperatorError("DEC law names a cell not present in the complex")
            target.update(supplied)
        for law in edges.values():
            if law.shape or tuple(law.input_ids) != ("tail", "head", "weight"):
                raise MathOperatorError("edge law ports must be (tail, head, weight) -> scalar")
        for law in joints.values():
            if law.shape or tuple(law.input_ids) != ("accumulation", "measure"):
                raise MathOperatorError("joint law ports must be (accumulation, measure) -> scalar")
        return cls(system, MappingProxyType(edges), MappingProxyType(joints))

    def _ports(self):
        vertices = self.system.complex.vertex_count
        return (tuple(sp.Symbol(f"u_{i}") for i in range(vertices)),
                tuple(sp.Symbol(f"star0_{i}") for i in range(vertices)),
                tuple(sp.Symbol(f"star1_{i}") for i in range(self.system.complex.edge_count)))

    def symbolic(self):
        """Compose local function graphs by the complex's exact oriented incidence."""
        field, star0, star1 = self._ports()
        accumulation = [sp.Integer(0) for _ in field]
        causes = tuple(dict.fromkeys(id(law) for law in (*self.edge_laws.values(), *self.joint_laws.values())))
        laws = {id(law):law for law in (*self.edge_laws.values(), *self.joint_laws.values())}
        for edge, (tail, head) in enumerate(self.system.complex.edges):
            law = self.edge_laws[edge]
            flux = law.expressions()[0].xreplace(dict(zip(law.symbols().values(),
                    (field[int(tail)], field[int(head)], star1[edge]))))
            accumulation[int(head)] += flux
            accumulation[int(tail)] -= flux
        outputs = []
        for vertex, law in self.joint_laws.items():
            outputs.append(law.expressions()[0].xreplace(dict(zip(law.symbols().values(),
                           (accumulation[vertex], star0[vertex])))))
        source = next(iter(laws.values()))
        return source._result(outputs, inputs=(*field, *star0, *star1), shape=(len(field),),
                              operation="dec_edge_joint_composition",
                              sources=tuple(laws[k] for k in causes if k != id(source)))

    def lower_source(self, *, name="dec_diagram", **compiler_options):
        """Compile wired edge/joint graph calls through the public source entry."""
        from .fortran_c_shell import lower_ast_source_to_ssa
        _identifier(name)
        field, star0, star1 = self._ports()
        parameters = tuple(map(str, (*field, *star0, *star1)))
        functions = {}; names = {}; lines = []
        def function(law):
            key = id(law)
            if key not in names:
                local = f"{name}_law_{len(names)}"
                names[key] = local; functions[local] = law.linkable_graph(name=local)
            return names[key]
        contributions = [[] for _ in field]
        for edge, (tail, head) in enumerate(self.system.complex.edges):
            tail, head = int(tail), int(head)
            lines.append(f"    flux_{edge} = {function(self.edge_laws[edge])}(u_{tail}, u_{head}, star1_{edge})")
            contributions[head].append(f"flux_{edge}")
            contributions[tail].append(f"(-flux_{edge})")
        outputs = []
        for vertex, law in self.joint_laws.items():
            accumulation = " + ".join(contributions[vertex]) or "0.0"
            lines.append(f"    result_{vertex} = {function(law)}({accumulation}, star0_{vertex})")
            outputs.append(f"result_{vertex}")
        lines.append(f"    return {', '.join(outputs)}")
        source = "\n".join((f"def {name}({', '.join(parameters)}):", *lines, ""))
        return lower_ast_source_to_ssa(source, name, name=name,
                    linked_process_graphs=functions, **compiler_options)


__all__ = ["MathProgram", "MathOperatorError", "DECGraphDiagram"]
