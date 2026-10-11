from pathlib import Path

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


def test_public_lowering_catalogue_shares_declarative_program_abi():
    resolved = []
    result = lower_ast_source_to_ssa(
        "def helper(x):\n    return x + 1.0\ndef root(x):\n    return helper(x)\n",
        "root", name="shared_abi_catalogue", progress=lambda message: None,
        extraction_contract=Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml",
        resolved_process_graph_sink=resolved.append,
        stop_after_compilation_unit_plan=True,
    )
    assert result == (None, {}, ())
    graph = resolved[0]
    receipt = graph.G.graph["program_abi"]
    functions = [entry.graph for entry in graph.function_table if entry.graph is not None]
    assert len(functions) >= 2
    assert all(function.G.graph["program_abi"] is receipt for function in functions)


def test_region_shares_abi_without_walking_schema_but_specialization_isolates_it():
    from src.compiler.process_graph_fusion import extract_clean_process_subgraph
    from src.transmogrifier.graph.graph_express2 import ProcessGraph

    class CountedABI(dict):
        scans = 0

        def items(self):
            self.scans += 1
            return super().items()

    graph = ProcessGraph(materialize_memory=False)
    graph.G.add_nodes_from((node, {"parents": [], "children": []}) for node in range(32))
    graph.levels = {node: 0 for node in graph.G}
    receipt = CountedABI(records={str(field): {"shape": [88]} for field in range(128)})
    graph.G.graph["program_abi"] = receipt
    for node in graph.G:
        region = extract_clean_process_subgraph(graph, (node,))
        assert region.G.graph["program_abi"] is receipt
    assert receipt.scans == 0
    specialization = extract_clean_process_subgraph(graph, graph.G)
    isolated = specialization.G.graph["program_abi"]
    assert isolated is not receipt and isolated == receipt
    isolated["records"]["0"]["shape"].append(2)
    assert receipt["records"]["0"]["shape"] == [88]
