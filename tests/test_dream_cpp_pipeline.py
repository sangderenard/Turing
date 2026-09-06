"""Push a real, non-trivial C++-like dream block through the actual
dream_document pipeline end to end -- load_dream_document ->
compile_sections() -- no bypassing to call desugar_cpp_shell directly.
This is the same file exercised in examples/cpp_shell_torture.dream.
"""

from src.compiler.dream_document import load_dream_document


def test_inheritance_loops_and_conditionals_reach_a_real_process_graph():
    doc = load_dream_document("examples/cpp_shell_torture.dream")
    sections = doc.compile_sections()
    assert len(sections) == 1

    section = sections[0]
    assert section.block == "cpp-torture"
    assert section.route == "cpp-shell"

    graph = section.graph
    nodes = graph["nodes"]
    types = [str(node.get("type")) for node in nodes]

    # Two classes (four Struct nodes: two definitions, two typedef
    # references), the inheritance-driven While/If control flow inside
    # method bodies, and both constructors + both methods + the free
    # function all reaching FuncDef.
    assert types.count("FuncDef") == 5  # Animal__new, Animal__lifespan_estimate,
    #                                     Dog__new, Dog__total_score, run_report
    assert types.count("While") == 2
    assert types.count("If") == 1
    assert types.count("Struct") == 4
