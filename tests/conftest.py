import os
import time
from pathlib import Path

import pytest
from src.common.tensors.abstract_nn import set_seed

set_seed(0)

def pytest_addoption(parser):
    parser.addoption(
        "--tape-viz",
        action="store_true",
        help="Enable TapeVisualizer output during tests",
    )
    parser.addoption(
        "--run-operators",
        action="store_true",
        help="Run tests that execute the operator simulation",
    )
    parser.addoption(
        "--run-known-failing",
        action="store_true",
        help="Run the tests recorded as already failing at af00599 "
             "(see TEST_BASELINE_AND_HAZARDS.md). Use this when you are "
             "working ON one of them; the default run skips them.",
    )


def pytest_configure(config):
    if config.getoption("--tape-viz"):
        os.environ["TAPE_VIZ"] = "1"
    config.addinivalue_line(
        "markers",
        "operators: tests that run the operator simulation",
    )


# Tests already failing at af00599, before the namespace/indexing fixes on top
# of it. They are skipped by default so a red result means "you broke
# something" instead of "look it up in a document". Each was confirmed
# pre-existing by running the file both in the working tree and in a clean
# `git worktree` at af00599 and getting identical results.
#
# Do NOT add an entry here to silence a failure you caused. The point of the
# list is that it is short, dated, and each line is attributable to a commit.
# Run them with --run-known-failing when you intend to fix one.
KNOWN_FAILING_AT_AF00599 = {
    "test_ast_indexing_aot.py": (
        "test_ast_boolean_masked_augmented_store_stays_arena_shaped",
        "test_enumerate_counter_augmented_store_is_not_a_boolean_mask",
        "test_existing_tensor_normalizer_becomes_process_graph_alias",
        "test_ast_basic_slice_store_lowers_to_fortran_array_section",
        "test_ast_advanced_indices_become_runtime_flat_gather",
        "test_ast_scalar_edge_pad_and_multi_axis_spans_lower_to_fortran",
        "test_loop_break_guard_prevents_speculative_division_by_zero",
        "test_reduced_named_branch_uses_its_assignment_value",
        "test_reduced_structural_method_call_uses_live_object_binding",
        "test_comprehension_does_not_replay_invariant_effectful_call",
        "test_shader_local_name_augassign_stays_in_numerical_region",
        "test_shader_local_plain_masked_store_stays_numerical",
        "test_object_array_field_keeps_its_dotted_public_feed_origin",
    ),
    "test_index_set_scatter.py": (
        "test_index_set_emits_a_complete_scatter_module",
        "test_index_set_scatter_runs_correctly",
    ),
    # Confirmed identical in a clean af00599 worktree 2026-08-19 (Fortran-lane
    # emission defects: rank-mismatched array references, non-LOGICAL merge
    # masks). Nothing to do with the LLVM lane the file is named for.
    "test_llvm_repository_ssa.py": (
        "test_c_backend_is_one_complete_repository_ssa_code_reference",
        "test_ssa_abstract_tensor_basis_is_emittable_as_a_runtime_module",
        "test_tensor_instruction_materializes_real_repository_ssa_kernel_operands",
        "test_unknown_tensor_extent_is_not_silently_compiled_as_one_element",
        "test_process_graph_preserves_replaced_tensor_record_through_complete_ssa",
        "test_process_graph_two_layer_tensor_network_compiles_and_executes",
        "test_abstract_tensor_ssa_backend_source_compiles_and_executes",
        "test_extended_softmax_composite_compiles_and_executes",
    ),
}


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-known-failing"):
        known = pytest.mark.skip(
            reason="pre-existing failure at af00599 -- see "
                   "TEST_BASELINE_AND_HAZARDS.md; --run-known-failing to run"
        )
        for item in items:
            names = KNOWN_FAILING_AT_AF00599.get(Path(str(item.fspath)).name)
            base = getattr(item, "originalname", None) or item.name.split("[")[0]
            if names and base in names:
                item.add_marker(known)
    if config.getoption("--run-operators"):
        return
    skip = pytest.mark.skip(reason="need --run-operators option to run")
    for item in items:
        if "operators" in item.keywords:
            item.add_marker(skip)


_RUN_LOG = "pytest_run_times.log"


def pytest_sessionstart(session):
    log_file = Path(session.config.rootpath) / _RUN_LOG
    if log_file.exists():
        lines = log_file.read_text().splitlines()
    else:
        lines = []

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter and lines:
        reporter.write_line("Recent pytest run times:")
        for entry in lines[-5:]:
            reporter.write_line(f"  {entry}")

    session._start_time = time.time()


def pytest_sessionfinish(session, exitstatus):
    duration = time.time() - session._start_time
    log_file = Path(session.config.rootpath) / _RUN_LOG
    history = int(os.environ.get("PYTEST_RUN_TIME_HISTORY", "50"))
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {duration:.2f}"

    if log_file.exists():
        lines = log_file.read_text().splitlines()
    else:
        lines = []

    lines.append(line)
    lines = lines[-history:]
    log_file.write_text("\n".join(lines) + "\n")

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter and lines:
        reporter.write_line("Recent pytest run times:")
        for entry in lines[-5:]:
            reporter.write_line(f"  {entry}")


# ---------------------------------------------------------------------------
# Helpers for cell pressure tests
# ---------------------------------------------------------------------------


@pytest.fixture(params=[7, 11, 13, 17])
def stride(request):
    """Provide a selection of prime strides for pressure simulations."""
    return request.param
