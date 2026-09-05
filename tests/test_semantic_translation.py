from types import SimpleNamespace

from src.compiler.semantic_translation import (
    SemanticRepresentation,
    SemanticTranslationProof,
    SemanticTranslationResidual,
    prove_exact_translation,
    semantic_identity,
)
from src.compiler.ssa_builder import process_graph_to_ssa_instrs
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def test_full_process_graph_arithmetic_identity_survives_repository_ssa_edge():
    graph = ProcessGraph(materialize_memory=False)
    graph.G.add_node(1, op="Input", label="left", parents=[], tensor={})
    graph.G.add_node(2, op="Input", label="right", parents=[], tensor={})
    process_identity = semantic_identity(
        "Add", SemanticRepresentation.PROCESS_GRAPH,
        facets={
            "tensor_shape": (4,), "tensor_dtype": "float32",
            "tensor_device": "cpu",
        },
    )
    graph.G.add_node(
        3, op="Add", label="Add", parents=[(1, "lhs"), (2, "rhs")],
        tensor={"shape": (4,), "dtype": "float32", "device": "cpu"},
        attributes=process_identity.attributes(),
    )
    graph.compute_levels = lambda **_kwargs: {
        1: 0, 2: 0, 3: 1,
    }
    instructions = process_graph_to_ssa_instrs(graph)
    repository = next(item for item in instructions if item.res.id == 3)
    assert repository.attributes["semantic_family"] == "arithmetic.add"
    assert repository.attributes["semantic_representation"] == "repository-ssa"
    assert repository.attributes["semantic_source_representation"] == "process-graph"
    assert repository.attributes["semantic_facets"] == dict(
        process_identity.facets,
    )


def test_machine_and_tensor_add_share_family_without_erasing_machine_facets():
    machine = semantic_identity(
        "INTEGER_ADD", SemanticRepresentation.MACHINE_SSA,
        facets={"width": 64, "flags": "written", "memory": "possible"},
    )
    repository = semantic_identity(
        "Add", SemanticRepresentation.REPOSITORY_SSA,
        facets={"width": 64},
    )

    assert machine.family == repository.family == "arithmetic.add"
    residual = prove_exact_translation(machine, repository)
    assert isinstance(residual, SemanticTranslationResidual)
    assert residual.missing_facets == ("flags", "memory")

    exact_repository = semantic_identity(
        "Add", SemanticRepresentation.REPOSITORY_SSA,
        facets=dict(machine.facets),
    )
    proof = prove_exact_translation(machine, exact_repository)
    assert isinstance(proof, SemanticTranslationProof)
    assert proof.preserved_facets == ("width", "flags", "memory")


def test_semantic_family_is_not_an_exactness_claim():
    source = semantic_identity(
        "reshape", SemanticRepresentation.DUAL_IR,
        facets={"shape": (2, 2), "layout": "dense-row-major"},
    )
    target = semantic_identity(
        "reshape", SemanticRepresentation.PROCESS_GRAPH,
        facets={"shape": (2, 2)},
    )
    residual = prove_exact_translation(source, target)
    assert isinstance(residual, SemanticTranslationResidual)
    assert residual.family == "tensor.reshape"
    assert residual.missing_facets == ("layout",)
