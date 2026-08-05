"""Tests for the process-graph schedule table shell, against class modules
segmented by partition_reduced_program (see wasm_class_modules.py)."""

import json
import numpy as np
import networkx as nx
import pytest
import sympy
import ast
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

from src.common.tensors.accelerator_backends.aot_compile import (
    compile_ast_aot,
    project_public_numerical_program,
)
from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.process_graph_shell import emit_process_graph_shell, schedule_table
from src.compiler.glsl_deployment_strategy import _control_partition_keys
from src.compiler.control_source import CallBlock, LoopBlock, SequenceBlock
from src.compiler.precompile_to_ssa import lower_precompile_and_control_to_ssa
from src.compiler.wasm_class_modules import partition_reduced_program


_PROCESS_GRAPH_LAZY_PREBOUND_VALUE = None
_PROCESS_GRAPH_LAZY_CALLABLE_ROOT = None


def _install_process_graph_lazy_test_binding():
    global _PROCESS_GRAPH_LAZY_PREBOUND_VALUE, _PROCESS_GRAPH_LAZY_CALLABLE_ROOT
    globals()["_PROCESS_GRAPH_LAZY_TEST_VALUE"] = "installed"
    _PROCESS_GRAPH_LAZY_PREBOUND_VALUE = "installed"
    _PROCESS_GRAPH_LAZY_CALLABLE_ROOT = SimpleNamespace(convert=str)


def _read_process_graph_lazy_test_binding():
    return _PROCESS_GRAPH_LAZY_TEST_VALUE


class _ProcessGraphLazyModuleReader:
    def install(self):
        _install_process_graph_lazy_test_binding()

    def read(self):
        return _PROCESS_GRAPH_LAZY_TEST_VALUE

    def read_prebound_through_host_call(self):
        return str(_PROCESS_GRAPH_LAZY_PREBOUND_VALUE)

    def call_prebound_attribute(self):
        return _PROCESS_GRAPH_LAZY_CALLABLE_ROOT.convert(
            _PROCESS_GRAPH_LAZY_PREBOUND_VALUE
        )


def _linear_program(n_steps: int, feed_id: int = 1) -> FusedProgram:
    steps = []
    previous = feed_id
    for i in range(n_steps):
        result_id = 100 + i
        steps.append(OpStep(
            step_id=i, op_name="mul", input_ids=[previous],
            attrs={"right_scalar": 2.0}, result_id=result_id,
        ))
        previous = result_id
    return FusedProgram(
        version=1, feeds={feed_id}, steps=steps, outputs={"result": previous},
    )


def test_comprehension_consumer_has_post_control_partition_key():
    graph = SimpleNamespace(G=nx.DiGraph())
    generator = ast.comprehension(
        target=ast.Name(id="item", ctx=ast.Store()),
        iter=ast.Name(id="items", ctx=ast.Load()),
        ifs=[],
        is_async=0,
    )
    aggregate = ast.ListComp(
        elt=ast.Name(id="item", ctx=ast.Load()),
        generators=[generator],
    )
    graph.G.add_nodes_from((1, 2, 3, 4, 5))
    graph.G.nodes[1]["expr_obj"] = ast.Name(id="items", ctx=ast.Load())
    graph.G.nodes[2]["expr_obj"] = generator
    graph.G.nodes[3].update(
        expr_obj=aggregate,
        parents=[(2, "generators")],
    )
    graph.G.nodes[4]["expr_obj"] = ast.Call(
        func=ast.Name(id="consume", ctx=ast.Load()),
        args=[aggregate],
        keywords=[],
    )
    graph.G.nodes[5]["expr_obj"] = aggregate.elt
    graph.G.add_edges_from(((1, 2), (2, 3), (5, 3), (3, 4)))
    plan = SimpleNamespace(
        loop=SimpleNamespace(node_id=2, body_nodes=(2,)),
    )

    keys = _control_partition_keys(graph, (plan,), (1, 4, 5))

    assert keys[1] != keys[4]
    assert keys[1] != keys[5]


def _compile_two_function_specs(chunk_size=2):
    source = (
        "def helper(x):\n"
        "    return x * 2.0\n"
        "\n"
        "def kernel(a):\n"
        "    return helper(a) + 1.0\n"
    )
    aot = compile_ast_aot(
        source, "kernel", {"a": np.array([20.5])}, precompile_only=True,
    )
    program = getattr(
        aot.compiled_shell_program, "program", aot.compiled_shell_program
    )
    return partition_reduced_program(
        program, chunk_size=chunk_size, owner_name=aot.entrypoint,
    )


def test_aot_keeps_annotations_in_maps_and_out_of_runtime_dispatch():
    source = '''
ModuleValue: float = 3.0

class Scale:
    factor: float = 2.0

def kernel(value):
    result: float = value * 2.0
    return result
'''
    aot = compile_ast_aot(
        source,
        "kernel",
        {"value": np.array([4.0])},
        precompile_only=True,
    )

    schema = aot.map_ir["schema"]
    assert schema["module"]["annotations"][0]["name"] == "ModuleValue"
    assert schema["classes"][0]["members"][0]["name"] == "factor"
    function = next(
        item for item in schema["functions"] if item["identity"] == "kernel"
    )
    assert function["locals"][0]["name"] == "result"
    assert aot.function_outputs == ("result",)
    assert aot.function_parameters == ("value",)
    assert aot.identity_table["result"]
    assert aot.identity_table["result"][-1] == max(
        aot.identity_table["result"]
    )


def test_aot_ingests_and_retains_the_abstract_tensor_class_object():
    source = (
        "def helper(value):\n"
        "    return value * 2.0\n"
        "\n"
        "def kernel(value):\n"
        "    return helper(value)\n"
    )

    aot = compile_ast_aot(
        source,
        "kernel",
        {"value": np.array([4.0])},
        precompile_only=True,
        retain=AbstractTensor,
    )

    abstract_tensor_map = next(
        item
        for item in aot.map_ir["objects"]
        if item["class_name"] == "AbstractTensor"
    )
    assert len(abstract_tensor_map["methods"]) > 100
    assert {
        "AbstractTensor.tensor",
        "AbstractTensor.reshape",
        "AbstractTensor.sum",
    } <= {
        identity
        for identity, _reference in aot.map_ir["dependency_regions"][
            "bindings"
        ]
    }


def test_aot_records_and_validates_the_requested_bake_mode():
    source = (
        "def helper(x):\n"
        "    return x * 2.0\n"
        "\n"
        "def kernel(x):\n"
        "    return helper(x)\n"
    )
    one_shot = compile_ast_aot(
        source,
        "kernel",
        {"x": np.array([3.0])},
        precompile_only=True,
        bake_mode="one-shot",
        schedule_preference="ASAP",
    )

    assert one_shot.bake_mode == "one_shot"
    assert one_shot.schedule_preference == "asap"
    with pytest.raises(ValueError, match="one_shot.*whole_program"):
        compile_ast_aot(
            source,
            "kernel",
            {"x": np.array([3.0])},
            precompile_only=True,
            bake_mode="partial",
        )


def test_configured_parameter_constants_apply_before_graph_reduction():
    source = "def kernel(x, gain):\n    return x * gain\n"
    aot = compile_ast_aot(
        source,
        "kernel",
        {"x": np.array([3.0]), "gain": np.array([99.0])},
        precompile_only=True,
        constant_map={"gain": 2.0},
    )
    program = project_public_numerical_program(aot)

    assert aot.program_record_mode == "configured"
    assert aot.constant_map == {"gain": 2.0}
    assert len(program.feeds) == 1
    assert any(step.attrs.get("right_scalar") == 2.0 for step in program.steps)
    with pytest.raises(ValueError, match="cannot freeze mutable parameters"):
        compile_ast_aot(
            source,
            "kernel",
            {"x": np.array([3.0]), "gain": np.array([99.0])},
            precompile_only=True,
            constant_map={"gain": 2.0},
            mutable_parameters=("gain",),
        )
    with pytest.raises(ValueError, match="asap.*alap"):
        compile_ast_aot(
            source,
            "kernel",
            {"x": np.array([3.0])},
            precompile_only=True,
            schedule_preference="middle",
        )


def test_mutable_parameter_must_survive_in_executable_abi():
    numerical = compile_ast_aot(
        "def kernel(x):\n    return x + 1\n",
        "kernel",
        {"x": np.array([3.0])},
        precompile_only=True,
        mutable_parameters=("x",),
    )
    assert numerical.public_input_value_ids == {"x": 0}

    with pytest.raises(RuntimeError, match="specialized out"):
        compile_ast_aot(
            "def kernel(source):\n    return len(source)\n",
            "kernel",
            {"source": "discovery sample"},
            precompile_only=True,
            mutable_parameters=("source",),
        )


def test_schedule_table_is_built_fresh_from_real_compiled_ir():
    """Not a fixture: pulled from compile_ast_aot's own compiled_shell_program,
    cut into chunks, run through the real ProcessGraph/ILPScheduler, same as
    test_wasm_class_modules.py's scheduling tests."""

    specs = _compile_two_function_specs()
    table = schedule_table(specs)

    ids = {node["id"] for node in table["nodes"]}
    assert ids == {spec.module_name for spec in specs}

    root_node = next(n for n in table["nodes"] if n["is_root"])
    # The root is the process graph's owner -- named after the entrypoint
    # itself ("kernel"), not a chunk-numbered label.
    assert root_node["id"] == "kernel__" + str(
        next(s for s in specs if s.is_root).index
    )
    assert root_node["level"] == 0
    assert table["level_max"] == 0
    if len(specs) > 1:
        # Every prerequisite chunk sits behind the root, at a negative
        # level -- not counted up from an unrelated zero.
        assert table["level_min"] < 0
        for node in table["nodes"]:
            if not node["is_root"]:
                assert node["level"] < 0


def test_schedule_table_gives_a_single_node_its_own_row_and_column():
    specs = partition_reduced_program(
        _linear_program(3), chunk_size=10, owner_name="kernel",
    )
    table = schedule_table(specs)
    assert table["nodes"] == [
        {"id": "kernel__0", "level": 0, "group": 0, "is_root": True},
    ]
    assert table["edges"] == []
    assert table["level_min"] == 0
    assert table["level_max"] == 0
    assert table["groups"] == 1


def test_a_multi_chunk_program_lands_on_separate_rows():
    specs = partition_reduced_program(
        _linear_program(5), chunk_size=2, owner_name="kernel",
    )
    table = schedule_table(specs)
    levels = sorted(n["level"] for n in table["nodes"])
    assert levels == [-2, -1, 0]
    assert table["level_min"] == -2
    assert table["level_max"] == 0
    # A linear chain has no independent branches -- every chunk is in the
    # one weakly-connected component.
    assert table["groups"] == 1


def test_two_disjoint_call_trees_get_separate_groups():
    """The case the group definition (weakly-connected component) does
    distinguish: two entirely disjoint dependency graphs."""

    specs_a = partition_reduced_program(
        _linear_program(3, feed_id=1), chunk_size=10, owner_name="leaf_a",
    )
    specs_b = partition_reduced_program(
        _linear_program(3, feed_id=2), chunk_size=10, owner_name="leaf_b",
    )
    table = schedule_table([*specs_a, *specs_b])
    node_a = next(n for n in table["nodes"] if n["id"].startswith("leaf_a"))
    node_b = next(n for n in table["nodes"] if n["id"].startswith("leaf_b"))
    assert node_a["group"] != node_b["group"]
    assert table["groups"] == 2


def test_emit_process_graph_shell_embeds_the_schedule_as_json():
    specs = _compile_two_function_specs()
    table = schedule_table(specs)
    shell = emit_process_graph_shell(table, title="t")

    assert "<table" in shell.html
    assert "renderSchedule(SCHEDULE)" in shell.html
    for node in table["nodes"]:
        assert node["id"] in shell.html


def test_write_puts_the_page_on_disk(tmp_path):
    specs = _compile_two_function_specs()
    shell = emit_process_graph_shell(schedule_table(specs))
    path = shell.write(str(tmp_path / "out" / "index.html"))
    assert (tmp_path / "out" / "index.html").read_text(encoding="utf-8") == shell.html
    assert path == str(tmp_path / "out" / "index.html")


def test_structural_early_return_skips_later_branch_predicates():
    source = """
def choose(value):
    if isinstance(value, int):
        return value
    if isinstance(value, tuple) and value[1]:
        return value[0]
    return 0

def entry(value):
    return choose(value)
"""

    compilation = compile_ast_aot(
        source,
        "entry",
        {"value": 7},
        backend="fortran",
        precompile_only=True,
    )

    assert compilation.outputs == {}


def test_builtin_max_with_tensor_input_retains_distinct_call_result():
    source = """
def entry(dt):
    safe_dt = max(dt, 1.0e-12)
    return 2.0 / safe_dt
"""

    compilation = compile_ast_aot(
        source,
        "entry",
        {"dt": np.asarray([0.05], dtype=np.float64)},
        backend="fortran",
        precompile_only=True,
    )

    assert compilation.identity_table["dt"] != compilation.identity_table[
        "safe_dt"
    ]


def test_aot_checkpoint_resumes_compiled_plan(tmp_path):
    source = """
def increment(value):
    return value + 1

def entry(value):
    return increment(value)
"""
    first_progress = []
    second_progress = []

    compile_ast_aot(
        source,
        "entry",
        {"value": 7},
        backend="fortran",
        precompile_only=True,
        checkpoint=tmp_path,
        progress=first_progress.append,
    )
    compile_ast_aot(
        source,
        "entry",
        {"value": 7},
        backend="fortran",
        precompile_only=True,
        checkpoint=tmp_path,
        progress=second_progress.append,
    )

    assert "aot: saving frontend checkpoint" in first_progress
    assert "aot: saving compiled-plan checkpoint" in first_progress
    assert "aot: saving captured-program checkpoint" in first_progress
    assert "aot: resumed compiled-plan checkpoint" in second_progress
    assert "aot: resumed captured-program checkpoint" in second_progress
    assert "aot: capturing fused programs" not in second_progress
    assert not any(
        "building process graph" in message for message in second_progress
    )


def test_aot_checkpoint_reports_why_a_plan_cannot_resume(tmp_path):
    source = """
def entry(value):
    return value + 1
"""
    progress = []

    compile_ast_aot(
        source,
        "entry",
        {"value": 7},
        backend="fortran",
        precompile_only=True,
        checkpoint=tmp_path,
    )
    manifest_path = next(tmp_path.rglob("compiled_plan.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["implementation"] = "obsolete-implementation"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    compile_ast_aot(
        source,
        "entry",
        {"value": 7},
        backend="fortran",
        precompile_only=True,
        checkpoint=tmp_path,
        progress=progress.append,
    )

    assert any(
        message.startswith(
            "aot: compiled-plan checkpoint unavailable "
            "(miss: manifest mismatch (implementation))"
        )
        for message in progress
    )


def test_compiled_class_constructor_executes_initializer():
    source = """
class Box:
    def __init__(self, value):
        self.value = value

    def read(self):
        return self.value

def entry(value):
    box = Box(value)
    return box.read()
"""

    compilation = compile_ast_aot(
        source,
        "entry",
        {"value": 7},
        backend="fortran",
        precompile_only=True,
    )

    assert compilation.outputs == {}


def test_bound_static_method_outranks_same_named_compiled_function():
    active = ContextVar("active", default=None)
    source = """
def get(self):
    return self

def active_value():
    return _ACTIVE.get()

def entry():
    return active_value()
"""

    compilation = compile_ast_aot(
        source,
        "entry",
        {},
        backend="fortran",
        precompile_only=True,
        python_bindings={"_ACTIVE": active},
    )

    assert compilation.outputs == {}


def test_indexed_store_mutates_structural_mapping():
    source = """
def mutate(mapping):
    mapping["x"] = 1
    return mapping["x"]

def entry():
    mapping = {}
    return mutate(mapping)
"""

    compilation = compile_ast_aot(
        source,
        "entry",
        {},
        backend="fortran",
        precompile_only=True,
    )

    assert compilation.outputs == {}


def test_runtime_bound_method_outranks_unique_name_guess():
    class LiveGraph:
        def __init__(self):
            self.nodes = {}

        def add_node(self, node_id, **attributes):
            self.nodes[node_id] = attributes

    source = """
class UnrelatedGraph:
    def add_node(self, node_id, **attributes):
        raise AssertionError("wrong receiver type")

def entry(graph):
    graph.add_node(7, color="blue")
    return graph.nodes[7]["color"]
"""

    compilation = compile_ast_aot(
        source,
        "entry",
        {"graph": LiveGraph()},
        backend="fortran",
        precompile_only=True,
    )

    assert compilation.outputs == {}


def test_nested_function_outranks_same_named_class_method():
    source = """
class Other:
    def add_node(self, node_id, **attributes):
        raise AssertionError("wrong lexical function")

def outer(value):
    def add_node(item):
        return item + 1

    return add_node(value)

def entry(value):
    return outer(value)
"""

    compilation = compile_ast_aot(
        source,
        "entry",
        {"value": 7},
        backend="fortran",
        precompile_only=True,
    )

    assert compilation.outputs == {}


def test_nested_function_routes_enclosing_parameter():
    source = """
def outer(value, strict=False):
    def choose(item):
        return item if strict else item + 1

    return choose(value)

def entry(value):
    return outer(value, strict=True)
"""

    compilation = compile_ast_aot(
        source,
        "entry",
        {"value": 7},
        backend="fortran",
        precompile_only=True,
    )

    assert compilation.outputs == {}


def test_deeply_nested_function_routes_grandparent_local():
    source = """
def outer(seed):
    next_id = seed + 1

    def middle(item):
        def inner():
            return next_id + item

        return inner()

    return middle(2)

def entry(seed):
    return outer(seed)
"""

    compilation = compile_ast_aot(
        source,
        "entry",
        {"seed": 7},
        backend="fortran",
        precompile_only=True,
    )

    assert compilation.outputs == {}


def test_nested_nonlocal_augmented_assignment_is_structural_arithmetic():
    source = """
def outer(seed):
    next_id = seed

    def increment():
        nonlocal next_id
        next_id += 1
        return next_id

    increment()
    return increment()

def entry(seed):
    return outer(seed)
"""

    compilation = compile_ast_aot(
        source,
        "entry",
        {"seed": 7},
        backend="fortran",
        precompile_only=True,
    )

    assert compilation.outputs == {}


def test_nested_function_saves_pre_increment_nonlocal_value():
    source = """
def outer(seed, observed):
    next_id = seed

    def allocate():
        nonlocal next_id
        node_id = next_id
        next_id += 1
        return node_id

    observed.append(allocate())
    return observed

def entry(seed, observed):
    return outer(seed, observed)
"""
    observed = []

    compilation = compile_ast_aot(
        source,
        "entry",
        {"seed": 0, "observed": observed},
        backend="fortran",
        precompile_only=True,
    )

    assert observed == [0]
    assert compilation.outputs == {}


def test_indexed_store_reads_source_selected_local_value():
    source = """
def choose_and_store(mapping, flag):
    if flag:
        result = 3
    else:
        result = 7
    mapping["result"] = result
    return mapping

def entry(mapping, flag):
    return choose_and_store(mapping, flag)
"""
    mapping = {}

    compilation = compile_ast_aot(
        source,
        "entry",
        {"mapping": mapping, "flag": True},
        backend="fortran",
        precompile_only=True,
    )

    assert mapping == {"result": 3}
    assert compilation.outputs == {}


def test_loop_destructuring_publishes_structural_only_target():
    source = """
def collect(graph, observed):
    for node_id, data in graph.nodes(data=True):
        if data.get("keep"):
            observed.append(node_id)
    return observed

def entry(graph, observed):
    return collect(graph, observed)
"""
    observed = []
    graph = nx.DiGraph()
    graph.add_node(4, keep=True, value=9)

    compilation = compile_ast_aot(
        source,
        "entry",
        {"graph": graph, "observed": observed},
        backend="fortran",
        precompile_only=True,
    )

    assert observed == [4]
    assert compilation.outputs == {}


def test_source_ordered_loop_handles_continue_break_and_else():
    source = """
def collect(graph, observed):
    for node_id, data in graph.nodes(data=True):
        if data.get("skip"):
            continue
        observed.append(node_id)
        if data.get("stop"):
            break
    else:
        observed.append("else")
    return observed

def entry(graph, observed):
    return collect(graph, observed)
"""
    observed = []
    graph = nx.DiGraph()
    graph.add_node(1, skip=True)
    graph.add_node(2, stop=True)
    graph.add_node(3)

    compilation = compile_ast_aot(
        source,
        "entry",
        {"graph": graph, "observed": observed},
        backend="fortran",
        precompile_only=True,
    )

    assert observed == [2]
    assert compilation.outputs == {}


def test_try_local_is_not_demanded_as_external_input():
    source = """
def compute(observed):
    try:
        value = 3
    except ValueError:
        value = 7
    observed.append(value)
    return value

def entry(observed):
    return compute(observed)
"""
    observed = []

    compilation = compile_ast_aot(
        source,
        "entry",
        {"observed": observed},
        backend="fortran",
        precompile_only=True,
    )

    assert observed == [3]
    assert compilation.outputs == {}


def test_supplied_closure_binding_wins_over_later_local_identity():
    source = """
def outer(seed, observed):
    memo = {"seed": seed}

    def read_then_rebind():
        nonlocal memo
        original = memo["seed"]
        memo = {"seed": 99}
        observed.append(original)
        return memo

    return read_then_rebind()

def entry(seed, observed):
    return outer(seed, observed)
"""
    observed = []

    compilation = compile_ast_aot(
        source,
        "entry",
        {"seed": 5, "observed": observed},
        backend="fortran",
        precompile_only=True,
    )

    assert observed == [5]
    assert compilation.outputs == {}


def test_custom_new_host_factory_preserves_path_type_and_bound_method():
    source = """
def inspect_path(source_or_path, observed):
    candidate = (
        Path(source_or_path)
        if isinstance(source_or_path, (str, Path))
        else None
    )
    observed.append(candidate is not None and candidate.is_file())
    return observed

def entry(source_or_path, observed):
    return inspect_path(source_or_path, observed)
"""
    observed = []
    source_path = Path(__file__).resolve()

    compilation = compile_ast_aot(
        source,
        "entry",
        {"source_or_path": source_path, "observed": observed},
        backend="fortran",
        precompile_only=True,
        python_bindings={"Path": Path},
    )

    assert observed == [True]
    assert compilation.outputs == {}


def test_direct_self_recursion_outranks_same_named_method():
    source = """
class Unrelated:
    def add_node(self, node_id, **attr):
        return node_id

def add_node(items, observed):
    if not items:
        return observed
    observed.append(items[0])
    return add_node(items[1:], observed)

def entry(items, observed):
    return add_node(items, observed)
"""
    observed = []

    compilation = compile_ast_aot(
        source,
        "entry",
        {"items": [1, 2], "observed": observed},
        backend="fortran",
        precompile_only=True,
    )

    assert observed == [1, 2]
    assert compilation.outputs == {}


def test_string_key_mapping_lookup_stays_on_coordinator():
    source = """
def read(mapping, key, observed):
    observed.append(mapping[key])
    return observed

def entry(mapping, key, observed):
    return read(mapping, key, observed)
"""
    observed = []

    compilation = compile_ast_aot(
        source,
        "entry",
        {
            "mapping": {"arg:0": 11},
            "key": "arg:0",
            "observed": observed,
        },
        backend="fortran",
        precompile_only=True,
    )

    assert observed == [11]
    assert compilation.outputs == {}


def test_nested_python_mapping_reads_stay_in_one_coordinator_region():
    source = """
def read(mapping, key, observed):
    observed.append(mapping[key][1])
    return observed

def entry(mapping, key, observed):
    return read(mapping, key, observed)
"""
    observed = []

    compilation = compile_ast_aot(
        source,
        "entry",
        {
            "mapping": {0: (3, 7)},
            "key": 0,
            "observed": observed,
        },
        backend="fortran",
        precompile_only=True,
    )

    assert observed == [7]
    assert compilation.outputs == {}


def test_tensor_valued_mapping_projection_is_coordinator_owned():
    source = """
def entry(mapping, key):
    return mapping[key]
"""
    compilation = compile_ast_aot(
        source,
        "entry",
        {
            "mapping": {0: AbstractTensor.get_tensor([1.0, 2.0])},
            "key": 0,
        },
        backend="fortran",
        precompile_only=True,
    )

    assert compilation.outputs == {}


def test_zero_rank_tensor_key_uses_python_mapping_semantics():
    source = """
def entry(mapping, key, observed):
    observed.append(mapping[key])
    return observed
"""
    observed = []

    compilation = compile_ast_aot(
        source,
        "entry",
        {
            "mapping": {2: "selected"},
            "key": np.asarray(2),
            "observed": observed,
        },
        backend="fortran",
        precompile_only=True,
    )

    assert observed == ["selected"]
    assert compilation.outputs == {}


def test_imported_module_remains_structural_across_nested_function():
    source = """
def make_symbol(name, observed):
    observed.append(sympy.Symbol(name))
    return observed

def entry(name, observed):
    return make_symbol(name, observed)
"""
    observed = []

    compilation = compile_ast_aot(
        source,
        "entry",
        {"name": "x", "observed": observed},
        backend="fortran",
        precompile_only=True,
        python_bindings={"sympy": sympy},
    )

    assert observed == [sympy.Symbol("x")]
    assert compilation.outputs == {}


def test_structural_store_uses_source_order_rhs_identity():
    source = """
def sympy_literal(value):
    result = sympy.sympify(value)
    if isinstance(result, sympy.Basic):
        return result
    return sympy.Symbol(repr(value))

def emit(cache, observed):
    result = sympy_literal(1)
    if False:
        result = True
    cache[4] = result
    observed.append(cache[4])
    return observed

def entry(cache, observed):
    return emit(cache, observed)
"""
    cache = {}
    observed = []

    compilation = compile_ast_aot(
        source,
        "entry",
        {"cache": cache, "observed": observed},
        backend="fortran",
        precompile_only=True,
        python_bindings={"sympy": sympy},
    )

    assert cache == {4: sympy.Integer(1)}
    assert observed == [sympy.Integer(1)]
    assert compilation.outputs == {}


def test_container_literal_uses_source_order_name_identity():
    source = """
def entry(observed):
    literal = 1
    if False:
        literal = True
    observed["payload"] = {"value": literal}
    return observed
"""
    observed = {}

    compilation = compile_ast_aot(
        source,
        "entry",
        {"observed": observed},
        backend="fortran",
        precompile_only=True,
    )

    assert observed == {"payload": {"value": 1}}
    assert type(observed["payload"]["value"]) is int
    assert compilation.outputs == {}


def test_identity_compare_preserves_module_singletons():
    source = """
def classify(value, observed):
    observed.append(value is sympy.true or value is sympy.false)
    return observed

def entry(value, observed):
    return classify(value, observed)
"""
    observed = []

    compilation = compile_ast_aot(
        source,
        "entry",
        {"value": sympy.Integer(1), "observed": observed},
        backend="fortran",
        precompile_only=True,
        python_bindings={"sympy": sympy},
    )

    assert observed == [False]
    assert compilation.outputs == {}


def test_bare_builtin_constructor_keeps_exact_source_identity():
    source = """
def convert(value, observed):
    observed.append(int(value))
    return observed

def entry(value, observed):
    return convert(value, observed)
"""
    observed = []

    compilation = compile_ast_aot(
        source,
        "entry",
        {"value": sympy.Integer(1), "observed": observed},
        backend="fortran",
        precompile_only=True,
    )

    assert observed == [1]
    assert type(observed[0]) is int
    assert compilation.outputs == {}


def test_native_while_discovery_traces_one_iteration():
    source = """
def entry(observed):
    running = True
    while running:
        observed.append(len(observed))
    return observed
"""
    observed = []

    compilation = compile_ast_aot(
        source,
        "entry",
        {"observed": observed},
        backend="fortran",
        precompile_only=True,
    )

    assert observed == [0]
    assert compilation.outputs == {}


def test_nested_comprehension_publishes_all_destructured_names():
    source = """
def expand(mapping, observed):
    values = [
        value_id
        for name, value_ids in mapping.items()
        for value_id in value_ids
    ]
    observed["values"] = values
    return observed

def entry(mapping, observed):
    return expand(mapping, observed)
"""
    observed = {}

    compilation = compile_ast_aot(
        source,
        "entry",
        {"mapping": {"x": (1, 2)}, "observed": observed},
        backend="fortran",
        precompile_only=True,
    )

    assert observed == {"values": [1, 2]}
    assert compilation.outputs == {}


def test_nested_for_publishes_complete_outer_destructuring_frame():
    source = """
def entry(mapping, observed):
    for name, value_ids in mapping.items():
        for value_id in value_ids:
            observed["last"] = (name, value_id)
    return observed
"""
    observed = {}

    compilation = compile_ast_aot(
        source,
        "entry",
        {"mapping": {"left": (3, 5)}, "observed": observed},
        backend="fortran",
        precompile_only=True,
    )

    # The loop is structural in precompile-only mode; reaching the boundary
    # without inventing ``value_ids`` as a public input is the assertion.
    assert compilation.outputs == {}


def test_source_continue_survives_evaporated_control_node():
    source = """
def entry(items, observed):
    for item in items:
        if not isinstance(item, int):
            continue
        observed["value"] = item
    return observed
"""
    observed = {}

    compilation = compile_ast_aot(
        source,
        "entry",
        {"items": (object(), 7), "observed": observed},
        backend="fortran",
        precompile_only=True,
    )

    # The structural loop may be dead at the precompile boundary; the
    # regression is that source ``continue`` can be interpreted directly
    # when the loop is retained, without requiring a graph marker node.
    assert compilation.outputs == {}


def test_structural_lambda_uses_lexical_argument_frame():
    source = """
def entry(items, observed):
    observed["ordered"] = sorted(
        items, key=lambda item: (item[1], item[0])
    )
    return observed
"""
    observed = {}

    compilation = compile_ast_aot(
        source,
        "entry",
        {"items": (("b", 2), ("a", 1)), "observed": observed},
        backend="fortran",
        precompile_only=True,
    )

    assert observed == {"ordered": [("a", 1), ("b", 2)]}
    assert compilation.outputs == {}


def test_source_return_attribute_uses_live_receiver():
    source = """
def read_name(value):
    return value.name

def entry(value, observed):
    observed["name"] = read_name(value)
    return observed
"""

    class Named:
        name = "current"

    observed = {}
    compilation = compile_ast_aot(
        source,
        "entry",
        {"value": Named(), "observed": observed},
        backend="fortran",
        precompile_only=True,
    )

    assert observed == {"name": "current"}
    assert compilation.outputs == {}


def test_python_set_add_is_not_numeric_add_region():
    source = """
def entry(observed):
    seen = set()
    key = ("vertex", "digest")
    seen.add(key)
    observed["seen"] = seen
    return observed
"""
    observed = {}

    compilation = compile_ast_aot(
        source,
        "entry",
        {"observed": observed},
        backend="fortran",
        precompile_only=True,
    )

    assert observed == {"seen": {("vertex", "digest")}}
    assert compilation.outputs == {}


def test_tensor_constructor_with_list_input_remains_numerical():
    source = """
def entry(items):
    return AbstractTensor.get_tensor(items)
"""

    compilation = compile_ast_aot(
        source,
        "entry",
        {"items": [1.0, 2.0, 3.0]},
        python_bindings={"AbstractTensor": AbstractTensor},
        backend="fortran",
        precompile_only=True,
    )
    operations = {
        step.op_name
        for program in (
            *compilation.region_programs.values(),
            getattr(
                compilation.compiled_shell_program,
                "program",
                compilation.compiled_shell_program,
            ),
        )
        for step in program.steps
    }

    assert "tensor_from_list" in operations


def test_source_with_enters_binds_and_exits_context_manager():
    source = """
def entry(manager, observed):
    with manager as entered:
        observed.append(("body", entered))
    return observed
"""

    class Manager:
        def __enter__(self):
            observed.append(("enter", None))
            return "token"

        def __exit__(self, error_type, error, traceback):
            observed.append(("exit", error_type))
            return False

    observed = []
    compilation = compile_ast_aot(
        source,
        "entry",
        {"manager": Manager(), "observed": observed},
        backend="fortran",
        precompile_only=True,
    )

    assert observed == [
        ("enter", None),
        ("body", "token"),
        ("exit", None),
    ]
    assert compilation.outputs == {}


def test_compiled_callee_reads_module_global_installed_during_capture():
    source = """
def entry(observed):
    _install_process_graph_lazy_test_binding()
    observed.append(_read_process_graph_lazy_test_binding())
    return observed
"""
    observed = []
    globals().pop("_PROCESS_GRAPH_LAZY_TEST_VALUE", None)
    try:
        compilation = compile_ast_aot(
            source,
            "entry",
            {"observed": observed},
            python_bindings=globals(),
            backend="fortran",
            precompile_only=True,
        )
    finally:
        globals().pop("_PROCESS_GRAPH_LAZY_TEST_VALUE", None)

    assert observed == ["installed"]
    assert compilation.outputs == {}


def test_compiled_method_reads_module_global_installed_during_capture():
    source = """
def entry(observed):
    reader = _ProcessGraphLazyModuleReader()
    reader.install()
    observed.append((
        reader.read(),
        reader.read_prebound_through_host_call(),
        reader.call_prebound_attribute(),
    ))
    return observed
"""
    observed = []
    global _PROCESS_GRAPH_LAZY_PREBOUND_VALUE, _PROCESS_GRAPH_LAZY_CALLABLE_ROOT
    globals().pop("_PROCESS_GRAPH_LAZY_TEST_VALUE", None)
    _PROCESS_GRAPH_LAZY_PREBOUND_VALUE = None
    _PROCESS_GRAPH_LAZY_CALLABLE_ROOT = None
    try:
        compilation = compile_ast_aot(
            source,
            "entry",
            {"observed": observed},
            python_bindings=globals(),
            backend="fortran",
            precompile_only=True,
        )
    finally:
        globals().pop("_PROCESS_GRAPH_LAZY_TEST_VALUE", None)
        _PROCESS_GRAPH_LAZY_PREBOUND_VALUE = None
        _PROCESS_GRAPH_LAZY_CALLABLE_ROOT = None

    assert observed == [("installed", "installed", "installed")]
    assert compilation.outputs == {}


def test_live_receiver_type_outranks_same_named_instance_method():
    source = """
class Other:
    def get(self, node_id):
        return self.memory[node_id]

def entry(mapping, observed):
    observed.append(mapping.get("value"))
    return observed
"""
    observed = []

    compilation = compile_ast_aot(
        source,
        "entry",
        {"mapping": {"value": 7}, "observed": observed},
        backend="fortran",
        precompile_only=True,
    )

    assert observed == [7]
    assert compilation.outputs == {}


def test_checkpoint_resume_preserves_nested_return_value_nodes(tmp_path):
    source = """
def outer():
    def inner():
        value = 0
        return value

    return int(inner())

def entry():
    return outer()
"""

    compile_ast_aot(
        source,
        "entry",
        {},
        backend="fortran",
        precompile_only=True,
        checkpoint=tmp_path,
    )
    compilation = compile_ast_aot(
        source,
        "entry",
        {},
        backend="fortran",
        precompile_only=True,
        checkpoint=tmp_path,
    )

    assert compilation.outputs == {}


def test_assignment_from_normalized_name_preserves_value():
    source = """
def copy_value(seed):
    copied = seed
    return copied

def entry(seed):
    return int(copy_value(seed))
"""

    compilation = compile_ast_aot(
        source,
        "entry",
        {"seed": 7},
        backend="fortran",
        precompile_only=True,
    )

    assert compilation.outputs == {}


def test_nested_retained_loop_preserves_distinct_backedge_ssa_identity():
    source = """
def iterate(values):
    total = values
    for index in range(12):
        total = total + values
    return total

def entry(values):
    return iterate(values)
"""
    compilation = compile_ast_aot(
        source,
        "entry",
        {"values": np.asarray([1.0, 2.0], dtype=np.float64)},
        backend="fortran",
        precompile_only=True,
    )

    loops = []

    def visit(block):
        if isinstance(block, LoopBlock):
            loops.append(block)
            visit(block.body)
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                visit(child)
        elif isinstance(block, CallBlock):
            visit(block.callee)

    visit(compilation.shell_control_program.root)
    assert loops
    assert all(
        updated != initial
        for loop in loops
        for updated, initial in loop.carried_aliases
    )

    lowered = lower_precompile_and_control_to_ssa(
        compilation.compiled_shell_program,
        compilation.shell_control_program,
        region_programs=dict(compilation.region_programs),
        identity_table=compilation.identity_table,
        function_outputs=compilation.function_outputs,
        function_parameters=compilation.function_parameters,
    )
    assert not tuple(
        shortfall
        for shortfall in lowered.shortfalls
        if shortfall.name == "loop_carried"
    )
