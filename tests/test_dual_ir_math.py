"""Exact mathematics and real-source native execution, never finite differences."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp
from pathlib import Path

from src.compiler.dual_ir_math import MathProgram, MathOperatorError
from src.compiler.control_source import ConditionalBlock, ControlProgram, StatementBlock
from src.compiler.identity_concordance import current_identity_book, concordance_report
from src.compiler.concordance_declarations import GRAPH_MATH_OUTPUT
from src.compiler.extraction_contract import ExtractionContract

CONTRACT = ExtractionContract(Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml")
TENSOR_TYPE = "src.common.tensors.abstraction.AbstractTensor"


def _same_expressions(actual, expected):
    assert len(actual) == len(expected)
    assert all(sp.simplify(left-right) == 0 for left, right in zip(actual, expected))


def _tensor_contract(function, parameters, shape):
    """Give a scalar-authored graph an exact elementwise tensor boundary."""
    return CONTRACT.with_program_abi({
        "records": {},
        "bindings": [],
        "values": [{
            "function": function,
            "parameter": parameter,
            "storage": "span",
            "dtype": "float64",
            "rank": len(shape),
            "shape": list(shape),
            "python_type": TENSOR_TYPE,
        } for parameter in parameters],
    })


def _mixed_tensor_scalar_contract(function, tensor, scalar, shape):
    return CONTRACT.with_program_abi({
        "records": {},
        "bindings": [],
        "values": [
            {
                "function": function,
                "parameter": tensor,
                "storage": "span",
                "dtype": "float64",
                "rank": len(shape),
                "shape": list(shape),
                "python_type": TENSOR_TYPE,
            },
            {
                "function": function,
                "parameter": scalar,
                "storage": "scalar",
                "dtype": "float64",
                "rank": 0,
                "python_type": "builtins.float",
            },
        ],
    })


def test_symbolic_jacobian_hessian_and_seed_operator_are_graphs():
    x, y, a, b = sp.symbols("x y a b")
    f = MathProgram.symbolic((x*x, x*y), inputs=(x,y), shape=(2,))
    j = f.jacobian()
    assert j.shape == (2,2)
    assert j.expressions() == (2*x, 0, y, x)
    vjp = MathProgram.symbolic((2*x*a+y*b, x*b), inputs=(x,y,a,b), shape=(2,))
    extracted = vjp.linear_operator(seeds=("a","b")).transpose()
    assert extracted.expressions() == j.expressions()
    assert tuple(extracted.input_ids) == ("x","y")
    scalar = MathProgram.symbolic((x*x*y+sp.sin(x),), inputs=(x,y))
    assert scalar.hessian().expressions() == (2*y-sp.sin(x),2*x,2*x,0)
    # Zero derivatives keep the parameter contract and admit higher orders.
    assert f.partial("y","y").partial("x").expressions() == (0,0)
    assert tuple(f.partial("y","y").input_ids) == ("x","y")
    assert isinstance(j.control, ControlProgram)
    assert set(j.region_nodes[0]) == set(j.numeric.G)


def test_compounded_differentials_and_second_order_pde_residuals():
    x,y,z,t = sp.symbols("x y z t")
    vector = MathProgram.symbolic((x*x, x*y, z*z), inputs=(x,y,z), shape=(3,))
    assert vector.divergence().expressions() == (3*x+2*z,)
    assert vector.curl(wrt=("x","y","z")).expressions() == (0,0,y)
    scalar = MathProgram.symbolic((x**4+y**4,), inputs=(x,y))
    assert scalar.laplacian().expressions() == (12*x*x+12*y*y,)
    assert scalar.biharmonic().expressions() == (48,)
    assert scalar.directional_derivative((1,2)).expressions() == (4*x**3+8*y**3,)
    field = MathProgram.symbolic((t*t+x**4+y*y,), inputs=(t,x,y))
    assert field.wave(time="t", space=("x","y")).expressions() == (-12*x*x,)
    assert field.heat(time="t", space=("x","y"), diffusivity=2).expressions() == (2*t-24*x*x-4,)
    assert field.poisson(2,space=("x","y")).expressions() == (12*x*x,)


def test_complete_exact_operator_surface_produces_owned_graphs():
    """Every promised exact operator returns another numeric/control program."""
    x,y,z,t,a,b=sp.symbols("x y z t a b")
    scalar=MathProgram.symbolic((x**3*y+sp.sin(x),),inputs=(x,y))
    vector=MathProgram.symbolic((x*y,y*z,z*x),inputs=(x,y,z),shape=(3,))
    quartic=MathProgram.symbolic((x**4+y**4,),inputs=(x,y))
    field=MathProgram.symbolic((t*t+x**4+y*y,),inputs=(t,x,y))
    seeded=MathProgram.symbolic((a*(x+y)+b*x, a*y-b*z),
                                inputs=(x,y,z,a,b),shape=(2,))
    matrix=MathProgram.symbolic((x,y,z,t),inputs=(x,y,z,t),shape=(2,2))
    replacement=MathProgram.symbolic((z*z,),inputs=(z,))
    inner=MathProgram.symbolic((x+y,x-y),inputs=(x,y),shape=(2,))
    outer=MathProgram.symbolic((x*y,),inputs=(x,y))

    programs=(
        (scalar.partial("x","y"),(3*x*x,)),
        (scalar.gradient(),(3*x*x*y+sp.cos(x),x**3)),
        (scalar.jacobian(),(3*x*x*y+sp.cos(x),x**3)),
        (scalar.hessian(),(6*x*y-sp.sin(x),3*x*x,3*x*x,0)),
        (vector.divergence(),(x+y+z,)),
        (vector.curl(wrt=("x","y","z")),(-y,-z,-x)),
        (vector.laplacian(),(0,0,0)),
        (quartic.biharmonic(),(48,)),
        (scalar.directional_derivative((2,-1)),
         (2*(3*x*x*y+sp.cos(x))-x**3,)),
        (seeded.linear_operator(seeds=("a","b")),(x+y,x,y,-z)),
        (matrix.transpose(),(x,z,y,t)),
        (MathProgram.symbolic((x+y,),inputs=(x,y)).substitute({"x":replacement}),
         (z*z+y,)),
        (outer.compose(inner),(x*x-y*y,)),
        (field.heat(time="t",space=("x","y"),diffusivity=2),
         (2*t-24*x*x-4,)),
        (field.wave(time="t",space=("x","y")),(-12*x*x,)),
        (field.poisson(2,space=("x","y")),(12*x*x,)),
        (field.helmholtz(space=("x","y"),wave_number_squared=3),
         (12*x*x+2+3*(t*t+x**4+y*y),)),
    )
    for program,expected in programs:
        _same_expressions(program.expressions(),expected)
        assert set(program.region_nodes) == set(program.control.region_indices)
        assert {node for nodes in program.region_nodes.values() for node in nodes} == set(program.numeric.G)
        assert program.numeric.G.graph["semantic_authority"] == "ProcessGraph"
        assert len(program.numeric.G.graph["math_operator_cells"]) == len(program.numeric.roots)

    # Reverse-mode is the existing controlled adjoint, reached through this
    # boundary rather than reimplemented as another symbolic convention.
    reverse=scalar.reverse()
    assert reverse.control.forward is scalar.control
    assert reverse.control.forward_to_backward_regions == {0:1}
    assert set(reverse.forward_region_nodes[0]) == set(scalar.numeric.G)
    assert set(reverse.backward_region_nodes) == {1}


def test_composition_preserves_shared_functions_and_concordance():
    x,y = sp.symbols("x y")
    outer = MathProgram.symbolic((x*y,), inputs=(x,y))
    inner = MathProgram.symbolic((x+y,x-y), inputs=(x,y),shape=(2,))
    composed = outer.compose(inner)
    assert sp.expand(composed.expressions()[0]) == x*x-y*y
    assert composed.gradient().expressions() == (2*x,-2*y)
    cells = composed.numeric.G.graph["math_operator_cells"]
    book = composed.identity_book
    for cell in cells:
        assert book.pages[GRAPH_MATH_OUTPUT.name].latest(cell.row).operation == "compose"
        assert tuple(book.edges_into(cell))
    assert composed.sources == (outer,inner)


@pytest.mark.parametrize("expression", (lambda x,v: x+v, lambda x,v: v*v, lambda x,v: sp.sin(v)))
def test_seed_coefficient_extraction_refuses_affine_or_nonlinear_seeds(expression):
    x,v=sp.symbols("x v")
    p=MathProgram.symbolic((expression(x,v),),inputs=(x,v))
    with pytest.raises(MathOperatorError):
        p.linear_operator(seeds=("v",))


def test_dual_control_and_shell_context_are_not_silently_flattened():
    x=sp.Symbol("x")
    p=MathProgram.symbolic((x*x,),inputs=(x,))
    shell=SimpleNamespace(map_ir={"permissions":"retained"}, hierarchy_plan=object())
    p=replace(p,source_shell=shell)
    assert p.partial("x").source_shell is shell
    branch=ControlProgram(ConditionalBlock(predicate_value_id=p.input_ids["x"],
                         body=p.control.root,orelse=p.control.root),region_indices=(0,))
    with pytest.raises(MathOperatorError,match="ConditionalBlock"):
        replace(p,control=branch).jacobian()
    with pytest.raises(MathOperatorError,match="executable statements"):
        replace(p,control=ControlProgram(StatementBlock(("side_effect()",)),region_indices=(0,))).jacobian()
    with pytest.raises(MathOperatorError,match="exactly one"):
        MathProgram.bind(p.numeric,p.control,region_nodes={},input_ids=p.input_ids)


def _dec():
    from src.common.tensors.abstract_convolution.dec_system import CellComplex, DECSystem
    empty=np.empty(0,dtype=np.int64)
    complex=CellComplex(3,np.array([[0,1],[1,2]]),0,empty,empty,np.empty(0))
    return DECSystem(complex,[1.,2.,4.],[2.,3.])


def test_dec_diagram_composes_oriented_edges_and_joints_without_matrix():
    system=_dec()
    system.matrix=lambda *a: pytest.fail("dense DEC materialization is forbidden")
    diagram=system.laplace0_diagram()
    p=diagram.symbolic()
    assert p.shape == (3,)
    field=[1.,3.,8.]
    feeds={**{f"u_{i}":v for i,v in enumerate(field)},
           **{f"star0_{i}":v for i,v in enumerate([1.,2.,4.])},
           **{f"star1_{i}":v for i,v in enumerate([2.,3.])}}
    values=[float(e.subs({p.symbols()[k]:v for k,v in feeds.items()})) for e in p.expressions()]
    expected=np.asarray(system.laplace0(field).data,dtype=float)
    np.testing.assert_array_equal(values,expected)
    assert values == [-4.,-5.5,3.75]
    # Constant field: exact incidence annihilation, no sampled proof.
    u=sp.Symbol("constant")
    assert tuple(sp.simplify(e.subs({p.symbols()[f"u_{i}"]:u for i in range(3)}))
                 for e in p.expressions()) == (0,0,0)


def _execute(lowering, bindings, tmp_path):
    from src.compiler.ssa_llvm_backend import compile_artifact,emit_ssa_function_to_llvm,prepare_artifact_execution
    from src.compiler.vehicle_python_compilation import _managed_native_feeds_by_id
    module,outputs,exports=lowering
    root=module.functions[exports[0]]
    report=concordance_report(module)
    assert report.splitlines()[0].endswith("0 finding(s)"),report
    artifact=emit_ssa_function_to_llvm(module,root.name)
    assert artifact.complete,artifact.shortfalls
    artifact=compile_artifact(artifact,directory=tmp_path,optimization="O3")
    feeds=_managed_native_feeds_by_id(SimpleNamespace(module=module,root_name=root.name),bindings)
    execution=prepare_artifact_execution(artifact,feeds); execution.run()
    return tuple(float(execution.buffers[v.id].reshape(-1)[0]) for v in outputs[root.name])


def _execute_arrays(lowering, bindings, tmp_path):
    from src.compiler.ssa_llvm_backend import compile_artifact,emit_ssa_function_to_llvm,prepare_artifact_execution
    from src.compiler.vehicle_python_compilation import _managed_native_feeds_by_id
    module,outputs,exports=lowering
    root=module.functions[exports[0]]
    report=concordance_report(module)
    assert report.splitlines()[0].endswith("0 finding(s)"),report
    artifact=emit_ssa_function_to_llvm(module,root.name)
    assert artifact.complete,artifact.shortfalls
    artifact=compile_artifact(artifact,directory=tmp_path,optimization="O3")
    feeds=_managed_native_feeds_by_id(SimpleNamespace(module=module,root_name=root.name),bindings)
    execution=prepare_artifact_execution(artifact,feeds); execution.run()
    return tuple(np.array(execution.buffers[v.id],copy=True) for v in outputs[root.name])


def test_native_jacobian_real_source_matches_authored_python(tmp_path):
    x,y=sp.symbols("x y")
    p=MathProgram.symbolic((x*x,x*y),inputs=(x,y),shape=(2,)).jacobian()
    result=_execute(p.lower_source(name="jacobian_native",extraction_contract=CONTRACT,progress=lambda _:None),{"x":2.,"y":3.},tmp_path)
    # Authored analytic Python oracle; no compiled-to-compiled comparison.
    def jacobian(x,y): return (2*x,0.,y,x)
    assert result == jacobian(2.,3.)


def test_scalar_authored_jacobian_materializes_elementwise_over_tensor_feeds(tmp_path):
    """Shaped feeds batch the scalar law; they do not invent cross-item coupling."""
    x,y=sp.symbols("x y")
    program=MathProgram.symbolic((x*x+sp.sin(y),x*y),inputs=(x,y),shape=(2,)).jacobian()
    shape=(5,)
    contract=_tensor_contract("tensor_jacobian",("x","y"),shape)
    x_value=np.linspace(-2.,2.,shape[0])
    y_value=np.linspace(.25,1.25,shape[0])
    result=_execute_arrays(program.lower_source(
        name="tensor_jacobian",extraction_contract=contract,progress=lambda _:None),
        {"x":x_value,"y":y_value},tmp_path)
    expected=(2*x_value,np.cos(y_value),y_value,x_value)
    assert all(value.shape == shape for value in result)
    for value,wanted in zip(result,expected):
        np.testing.assert_allclose(value,wanted,rtol=0.,atol=1e-14)


def test_tensor_materialization_keeps_scalar_coefficient_scalar(tmp_path):
    """A shaped operand lifts the result, not every parameter in the graph."""
    x,a=sp.symbols("x a")
    program=MathProgram.symbolic((a*x+a*a,),inputs=(x,a)).jacobian(wrt=("x","a"))
    shape=(5,)
    contract=_mixed_tensor_scalar_contract("mixed_jacobian","x","a",shape)
    lowering=program.lower_source(
        name="mixed_jacobian",extraction_contract=contract,progress=lambda _:None)
    module,_outputs,exports=lowering
    root=module.functions[exports[0]]
    names={int(value_id):str(name) for name,value_id in root.metadata["parameter_names"]}
    parameter_shapes={names[int(argument.id)]:tuple(argument.shape)
                      for argument in root.args if int(argument.id) in names}
    assert parameter_shapes["x"] == shape
    assert parameter_shapes["a"] == ()

    x_value=np.asarray([-2.,-1.,0.,1.,2.])
    a_value=3.
    result=_execute_arrays(lowering,{"x":x_value,"a":a_value},tmp_path)
    # d(a*x+a^2)/dx == a is still the scalar parameter itself. Expanding it
    # into a five-element allocation would be an improper tensorization; a
    # later elementwise consumer may broadcast it. The derivative with
    # respect to a genuinely depends on x and therefore has x's extent.
    assert result[0].shape == ()
    assert result[1].shape == shape
    assert float(result[0]) == a_value
    np.testing.assert_array_equal(result[1],x_value+2*a_value)


def test_native_dec_graph_functions_match_existing_eager_dec(tmp_path):
    system=_dec(); diagram=system.laplace0_diagram()
    values=[1.,3.,8.]
    feeds={**{f"u_{i}":v for i,v in enumerate(values)},
           **{f"star0_{i}":v for i,v in enumerate([1.,2.,4.])},
           **{f"star1_{i}":v for i,v in enumerate([2.,3.])}}
    result=_execute(diagram.lower_source(name="dec_native",extraction_contract=CONTRACT,progress=lambda _:None),feeds,tmp_path)
    np.testing.assert_array_equal(result,np.asarray(system.laplace0(values).data,dtype=float))


def test_scalar_authored_dec_laws_materialize_over_tensor_batches(tmp_path):
    """One topology/law graph evaluates independent DEC systems elementwise."""
    from src.common.tensors.abstract_convolution.dec_system import DECSystem

    system=_dec(); diagram=system.laplace0_diagram()
    batch=4
    field=np.asarray([[1.,2.,3.,4.],[3.,3.,5.,7.],[8.,6.,4.,2.]])
    # Powers-of-two measures keep both evaluation orders binary-exact; this
    # assertion is about the graph contract, not an unexplained tolerance.
    star0=np.asarray([[1.,1.,2.,2.],[2.,4.,2.,4.],[4.,4.,8.,8.]])
    star1=np.asarray([[2.,1.,3.,2.],[3.,4.,2.,5.]])
    bindings={**{f"u_{i}":field[i] for i in range(3)},
              **{f"star0_{i}":star0[i] for i in range(3)},
              **{f"star1_{i}":star1[i] for i in range(2)}}
    parameters=tuple(bindings)
    contract=_tensor_contract("tensor_dec",parameters,(batch,))
    result=_execute_arrays(diagram.lower_source(
        name="tensor_dec",extraction_contract=contract,progress=lambda _:None),
        bindings,tmp_path)
    expected=np.column_stack([
        np.asarray(DECSystem(system.complex,star0[:,i],star1[:,i]).laplace0(field[:,i]).data)
        for i in range(batch)
    ])
    assert all(value.shape == (batch,) for value in result)
    np.testing.assert_allclose(np.stack(result),expected,rtol=0.,atol=0.)


def test_native_source_jacobian_with_forced_disk_tier(tmp_path, monkeypatch):
    from src.compiler.identity_compression import disk_tile_stats

    monkeypatch.setenv("TURING_COMPRESS_COLD_PAGES", "1")
    monkeypatch.setenv("TURING_SPILL_COLD_PAGES", "0")
    # A one-byte budget deliberately forces the tier, NOT a plausible RSS cap.
    monkeypatch.setenv("TURING_MEMORY_BUDGET_BYTES", "1")
    x,y = sp.symbols("x y")
    program = MathProgram.symbolic((x*x,x*y), inputs=(x,y), shape=(2,)).jacobian()
    lowering = program.lower_source(name="disk_jacobian", extraction_contract=CONTRACT, progress=lambda _:None)
    book = lowering[0].metadata["identity_book"]
    # Live mode now also tiles writable pages, rather than snapshotting those
    # same tables into a second immutable cold-book representation.
    live = book.__dict__.get("_live_tile_report")
    if live is not None:
        assert live["disk_tiles"] > 0 and live["policy_sources"]
    else:
        assert disk_tile_stats(book)[0] > 0
        assert book.__dict__["_tile_offload_report"].policy_sources
    result = _execute(lowering, {"x":2.,"y":3.}, tmp_path)
    assert result == (4.,0.,3.,2.)
