import contextlib
import io

import numpy as np

from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot
from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.autograd import GradTape, autograd
from src.compiler.glsl_deployment_strategy import _walk_planned_shells
from src.compiler.precompile_to_ssa import lower_fused_program_to_ssa
from src.compiler.ssa_fortran_backend import emit_module


def _compile(source, entrypoint, feeds, *, python_bindings=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return compile_ast_aot(
            source,
            entrypoint,
            feeds,
            precompile_only=True,
            python_bindings=python_bindings,
        )


def _assert_fortran_complete(compilation):
    for region_index, program in compilation.region_programs.items():
        function, shortfalls = lower_fused_program_to_ssa(
            program, function_name=f"region_{region_index}"
        )
        assert shortfalls == ()
        assert emit_module({function.name: function}).shortfalls == ()


def _structural_optional_kernel(field, attempt_log=None):
    if attempt_log is not None:
        return field + 1.0
    return field - 1.0


class _SnapshotState:
    def __init__(self):
        self.snapshot_calls = 0

    def copy_shallow(self):
        self.snapshot_calls += 1
        return {"snapshot": self.snapshot_calls}

    def next_value(self):
        self.snapshot_calls += 1
        return self.snapshot_calls


class _ArrayState:
    def __init__(self):
        self.values = np.zeros((2, 2))

    def advance(self):
        self.values = self.values + 1.0
        return self.values


def test_ast_boolean_masked_augmented_store_stays_arena_shaped():
    source = """
def kernel(field, rhs, mask):
    field[mask] += 2.0 * rhs[mask]
    return field
"""
    compilation = _compile(
        source,
        "kernel",
        {
            "field": np.arange(8.0).reshape(2, 2, 2),
            "rhs": np.ones((2, 2, 2)),
            "mask": np.asarray(
                [[[True, False], [False, True]], [[False, False], [True, False]]]
            ),
        },
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
    assert "where" in operations
    _assert_fortran_complete(compilation)


def test_enumerate_counter_augmented_store_is_not_a_boolean_mask():
    source = """
def kernel(field):
    for idx, _value in enumerate(field):
        field[idx] *= 2.0
    return field
"""
    compilation = _compile(
        source,
        "kernel",
        {"field": np.arange(4.0)},
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
    assert "where" not in operations
    _assert_fortran_complete(compilation)


def test_scalar_index_records_zero_rank_slice_during_compiler_capture():
    field = AbstractTensor.get_tensor(np.arange(4.0))
    tape = GradTape()

    with autograd.forward_capture(tape):
        field._tape = tape
        selected = field[2]
    with autograd.forward_observation():
        repeated = [field[0], field[1]]
    packed = AbstractTensor.get_tensor([selected, *repeated])
    runtime_scalar = field[3]

    assert selected.shape == ()
    assert [value.shape for value in repeated] == [(), ()]
    assert packed.shape == (3,)
    assert not isinstance(runtime_scalar, AbstractTensor)
    assert [node.op for node in tape._nodes.values()] == ["slice"]


def test_existing_tensor_normalizer_becomes_process_graph_alias():
    source = """
def kernel(field):
    resident = AbstractTensor.get_tensor(field)
    field[0] = resident[0] + 1.0
    return field
"""
    compilation = _compile(
        source,
        "kernel",
        {"field": AbstractTensor.get_tensor(np.arange(4.0))},
        python_bindings={"AbstractTensor": AbstractTensor},
    )

    aliases = {
        int(result): int(parent)
        for shell in _walk_planned_shells(compilation.deployment)
        for result, parent in shell.compiled_process_graph_aliases.items()
    }
    assert aliases
    _assert_fortran_complete(compilation)


def test_percentile_accepts_class_style_raw_sequence_operand():
    result = AbstractTensor.percentile([1.0, 2.0, 3.0], 50)

    assert result == 2.0


def test_ast_basic_slice_store_lowers_to_fortran_array_section():
    compilation = _compile(
        "def kernel(field):\n    field[:, 1] -= 0.5\n    return field\n",
        "kernel",
        {"field": np.ones((6, 3))},
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
    assert "index_set" in operations
    _assert_fortran_complete(compilation)


def test_ast_advanced_indices_become_runtime_flat_gather():
    compilation = _compile(
        "def kernel(field, i, j, k):\n    return field[i, j, k]\n",
        "kernel",
        {
            "field": np.arange(24.0).reshape(4, 3, 2),
            "i": np.asarray([0, 2]),
            "j": np.asarray([1, 2]),
            "k": np.asarray([1, 0]),
        },
    )

    operations = {
        step.op_name
        for program in compilation.region_programs.values()
        for step in program.steps
    }
    assert {"reshape", "gather"} <= operations
    _assert_fortran_complete(compilation)


def test_ast_scalar_edge_pad_and_multi_axis_spans_lower_to_fortran():
    source = """
def kernel(field):
    padded = np.pad(field, 1, mode="edge")
    return padded[2:, 1:-1, 1:-1]
"""
    compilation = _compile(
        source,
        "kernel",
        {"field": np.ones((2, 2, 1))},
        python_bindings={"np": np},
    )

    operations = {
        step.op_name
        for program in compilation.region_programs.values()
        for step in program.steps
    }
    assert {"pad", "slice"} <= operations
    _assert_fortran_complete(compilation)


def test_loop_break_guard_prevents_speculative_division_by_zero():
    source = """
def kernel(field):
    for iteration in range(1):
        denominator = float(np.sum(field * field))
        if abs(denominator) < 1.0e-30:
            break
        field = field / denominator
    return field
"""
    compilation = _compile(
        source,
        "kernel",
        {"field": np.zeros((3, 2, 1))},
        python_bindings={"np": np},
    )

    assert compilation.control_shortfalls == ()
    _assert_fortran_complete(compilation)


def test_linked_structural_optional_branch_survives_reduced_test():
    source = """
def page(field, attempt_log):
    return structural_optional_kernel(field, attempt_log)
"""
    compilation = _compile(
        source,
        "page",
        {"field": np.zeros((2, 2)), "attempt_log": []},
        python_bindings={
            "structural_optional_kernel": _structural_optional_kernel,
        },
    )

    assert compilation.function_outputs == ("result_0",)
    _assert_fortran_complete(compilation)


def test_reduced_named_branch_uses_its_assignment_value():
    source = """
def kernel(field):
    rejected = float(np.sum(field)) > 100.0
    if rejected:
        return field * 0.0
    return field + 1.0
"""
    compilation = _compile(
        source,
        "kernel",
        {"field": np.zeros((2, 2))},
        python_bindings={"np": np},
    )

    assert compilation.function_outputs == ("result_0",)
    _assert_fortran_complete(compilation)


def test_while_recomputes_augmented_counter_and_predicate():
    source = """
def kernel(field, max_iters):
    iters = 0
    while iters < max_iters:
        iters += 1
        field = field + 1.0
    return field
"""
    compilation = _compile(
        source,
        "kernel",
        {"field": np.zeros((2, 2)), "max_iters": 1},
    )

    assert compilation.function_outputs == ("field",)
    assert compilation.control_shortfalls == ()
    _assert_fortran_complete(compilation)


def test_reduced_structural_method_call_uses_live_object_binding():
    source = """
def kernel(field, state):
    saved = state.copy_shallow()
    if saved is None:
        return field - 1.0
    return field + 1.0
"""
    state = _SnapshotState()
    compilation = _compile(
        source,
        "kernel",
        {"field": np.zeros((2, 2)), "state": state},
    )

    assert state.snapshot_calls == 1
    assert compilation.function_outputs == ("result_0",)
    assert compilation.control_shortfalls == ()
    _assert_fortran_complete(compilation)


def test_reduced_generator_expression_scopes_destructured_targets():
    source = """
def kernel(field, limits):
    failed = any(
        value > limit
        for value, limit in limits.items()
    )
    if failed:
        return field - 1.0
    return field + 1.0
"""
    compilation = _compile(
        source,
        "kernel",
        {"field": np.zeros((2, 2)), "limits": {}},
    )

    assert compilation.function_outputs == ("result_0",)
    assert compilation.control_shortfalls == ()
    _assert_fortran_complete(compilation)


def test_reduced_unary_not_drives_rejection_predicate():
    source = """
def kernel(field, ok):
    rejected = (not ok) or False
    if rejected:
        return field - 1.0
    return field + 1.0
"""
    compilation = _compile(
        source,
        "kernel",
        {"field": np.zeros((2, 2)), "ok": True},
    )

    assert compilation.function_outputs == ("result_0",)
    assert compilation.control_shortfalls == ()
    _assert_fortran_complete(compilation)


def test_reduced_tuple_assignment_publishes_each_lexical_value():
    source = """
def pair():
    return True, 3

def kernel(field):
    ok, count = pair()
    if (not ok) or count != 3:
        return field - 1.0
    return field + 1.0
"""
    compilation = _compile(
        source,
        "kernel",
        {"field": np.zeros((2, 2))},
    )

    assert compilation.function_outputs == ("result_0",)
    assert compilation.control_shortfalls == ()
    _assert_fortran_complete(compilation)


def test_reduced_scalar_binary_expression_drives_comparison():
    source = """
def kernel(field, threshold):
    rejected = 0.0 > threshold * 10.0
    if rejected:
        return field - 1.0
    return field + 1.0
"""
    compilation = _compile(
        source,
        "kernel",
        {"field": np.zeros((2, 2)), "threshold": 1.0},
    )

    assert compilation.function_outputs == ("result_0",)
    assert compilation.control_shortfalls == ()
    _assert_fortran_complete(compilation)


def test_pruned_loop_initializer_uses_published_carried_value():
    source = """
def kernel(field):
    carried = field
    for _index in range(1):
        carried = carried + 1.0
    return carried
"""
    compilation = _compile(
        source,
        "kernel",
        {"field": np.zeros((2, 2))},
    )

    assert compilation.function_outputs == ("carried",)
    assert compilation.control_shortfalls == ()
    _assert_fortran_complete(compilation)


def test_comprehension_does_not_replay_invariant_effectful_call():
    source = """
def kernel(field, state, items):
    base = state.next_value()
    values = tuple(base + item for item in items)
    return field + values[0]
"""
    state = _SnapshotState()
    compilation = _compile(
        source,
        "kernel",
        {
            "field": np.zeros((2, 2)),
            "state": state,
            "items": (2,),
        },
    )

    assert state.snapshot_calls == 1
    assert compilation.control_shortfalls == ()
    _assert_fortran_complete(compilation)


def test_numpy_any_accepts_reduction_keyword_protocol():
    source = """
def kernel(field):
    occupied = np.any(field > 0.0)
    if occupied:
        return field + 1.0
    return field - 1.0
"""
    compilation = _compile(
        source,
        "kernel",
        {"field": np.zeros((2, 2))},
        python_bindings={"np": np},
    )

    assert compilation.control_shortfalls == ()
    _assert_fortran_complete(compilation)


def test_shader_local_name_augassign_stays_in_numerical_region():
    source = """
def kernel(field):
    workspace = field * 0.0
    workspace += field
    return workspace
"""
    compilation = _compile(
        source,
        "kernel",
        {"field": np.ones((2, 2))},
        python_bindings={"np": np},
    )

    assert compilation.control_shortfalls == ()
    _assert_fortran_complete(compilation)


def test_shader_local_plain_masked_store_stays_numerical():
    source = """
def kernel(field, mask):
    workspace = field + 1.0
    workspace[mask] = 0.0
    return workspace
"""
    compilation = _compile(
        source,
        "kernel",
        {
            "field": np.ones((2, 2)),
            "mask": np.asarray([[True, False], [False, True]]),
        },
    )

    assert compilation.control_shortfalls == ()
    _assert_fortran_complete(compilation)


def test_object_array_field_keeps_its_dotted_public_feed_origin():
    compilation = _compile(
        "def kernel(state):\n    return state.values + 1.0\n",
        "kernel",
        {"state": _ArrayState()},
    )
    program = getattr(
        compilation.compiled_shell_program,
        "program",
        compilation.compiled_shell_program,
    )

    origins = (program.extras or {}).get("capture_feed_origins", {})

    assert {
        record.get("binding_name") for record in origins.values()
    } == {"state.values"}
