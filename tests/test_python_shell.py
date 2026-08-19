"""The instrumented Python shell: observation without perturbation.

The decision tree forbids source-level probes because they shift value ids
and have twice produced wrong conclusions. The shell records AFTER lowering,
so the program observed is exactly the program compiled, and the emitted
local names ARE the SSA value ids -- a trace is addressable against the IR
with no correlation table.
"""
from __future__ import annotations

import warnings

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.python_shell import compile_python_shell
from src.compiler.ssa_self_check import (
    check_formal_parity,
    check_id_scale,
    check_output_contract_agreement,
    run_all,
    suspicious_loop_invariant_formals,
)

_LOOP = """
def helper(a):
    return a * 1.0

def train(w, n):
    total = helper(w)
    for _ in range(n):
        next_w = w - 0.05 * w
        w = next_w
        total = w
    return total
"""


def _lower(source, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return lower_ast_source_to_ssa(source, "train", name=name)[0]


@pytest.fixture(scope="module")
def loop_shell():
    module = _lower(_LOOP, "ps")
    return module, compile_python_shell(module, "ps__train")


def test_the_shell_computes_the_authored_answer(loop_shell):
    _module, shell = loop_shell
    run = shell.run(w=2.0, n=3)
    expected = 2.0
    for _ in range(3):
        expected -= 0.05 * expected
    assert run.result == pytest.approx(expected, abs=1e-12)


def test_the_trace_shows_every_iteration_of_a_carried_value(loop_shell):
    """Not just the final value: the whole history, one entry per commit."""

    _module, shell = loop_shell
    run = shell.run(w=2.0, n=3)
    carried = [
        entry.value
        for entry in run.trace
        if entry.function == "ps__train" and entry.value_id == 7
    ]
    # Three loop iterations plus the post-loop port rebinding.
    assert carried[:3] == pytest.approx([1.9, 1.805, 1.71475], abs=1e-12)


def test_identical_runs_produce_identical_traces(loop_shell):
    _module, shell = loop_shell
    assert shell.run(w=2.0, n=3).first_divergence(shell.run(w=2.0, n=3)) is None


def test_divergence_is_localized_to_the_first_differing_value(loop_shell):
    """The routing instrument: WHERE two runs part, not just whether."""

    _module, shell = loop_shell
    baseline = shell.run(w=2.0, n=3)
    perturbed = shell.run(w=2.0001, n=3)
    divergence = baseline.first_divergence(perturbed)
    assert divergence is not None
    _index, mine, theirs = divergence
    assert mine.name == theirs.name
    assert mine.value != theirs.value


def test_an_unobserved_value_refuses_rather_than_defaulting(loop_shell):
    """'A measurement you cannot take is not a zero.'"""

    _module, shell = loop_shell
    run = shell.run(w=2.0, n=1)
    with pytest.raises(KeyError, match="never assigned"):
        run.last_value("ps__train", 999)


def test_the_shell_agrees_with_the_reference_evaluator(loop_shell):
    """Two independent executions of the same SSA must join.

    The shell runs materialized Python; the evaluator walks the SSA directly.
    Agreement here means the round trip preserved the meaning of a program
    with control flow, checked by a second interpreter rather than by the
    authored source alone.
    """

    from src.compiler.ssa_reference_evaluator import SSAReferenceEvaluator

    module, shell = loop_shell
    function = module.functions["ps__train"]
    produced = shell.run(w=2.0, n=3).result
    evaluated = SSAReferenceEvaluator(module).run(
        "ps__train",
        {int(a.id): v for a, v in zip(function.args, [3, 2.0])},
    )
    assert float(produced) == pytest.approx(float(evaluated.returned[0]), abs=0)


def test_a_missing_entry_refuses_at_compile_time(loop_shell):
    module, _shell = loop_shell
    with pytest.raises(ValueError, match="not among the emitted functions"):
        compile_python_shell(module, "no_such_function")


# -- the static self-checks over the same product -------------------------


def test_a_clean_program_raises_no_findings(loop_shell):
    module, _shell = loop_shell
    assert run_all(module) == []


def test_the_formal_collision_is_caught_statically():
    """The level-3 scorecard journey, flagged without executing anything."""

    module = _lower(
        """
def helper(a):
    return a * 1.0

def train(x):
    return helper(x) / 4.0 + x ** 2.0
""",
        "sc",
    )
    findings = check_formal_parity(module)
    assert findings, "the unnamed extra formal should be flagged"
    assert "no caller can know" in findings[0].detail


def test_id_scale_findings_name_the_poisoned_allocator():
    from src.transmogrifier.ssa import BasicBlock, Function, SSAValue

    class _Module:
        functions = {
            "f": Function(
                "f", [SSAValue(2266697258353)], {"entry": BasicBlock("entry", [])}
            )
        }

    findings = check_id_scale(_Module())
    assert findings and "memory-address scale" in findings[0].detail


def test_disagreeing_output_contracts_are_a_finding():
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue

    def call(outputs, result_id):
        return Instr(
            "Call", [SSAValue(0)], SSAValue(result_id),
            attributes={"callee": "r", "output_ids": outputs},
        )

    class _Module:
        functions = {
            "a": Function("a", [], {"entry": BasicBlock("entry", [call((1, 2), 5)])}),
            "b": Function("b", [], {"entry": BasicBlock("entry", [call((2, 1), 6)])}),
        }

    findings = check_output_contract_agreement(_Module())
    assert findings and "another call site projects" in findings[0].detail


def test_loop_invariant_formals_are_candidates_not_convictions(loop_shell):
    """The recorded gap: the SSA cannot say what the author meant to carry."""

    module = _lower(
        """
def helper(a):
    return a * 1.0

def train(w, m, n):
    total = helper(w)
    for _ in range(n):
        next_m = m * 0.5 + w
        next_w = w - 0.1 * next_m
        m = next_m
        w = next_w
        total = w
    return total
""",
        "cand",
    )
    candidates = suspicious_loop_invariant_formals(module)
    assert candidates, "the frozen m should surface as a candidate"
    assert "cannot say which" in candidates[0].detail
    # And deliberately NOT in run_all: candidates are not violations.
    clean_module, _shell = loop_shell
    assert all(f.check != "loop_invariant_formal" for f in run_all(module))


# -- hosted Python callables ----------------------------------------------


def test_a_binding_supplies_a_retained_callable():
    """The shell's second role: compiled and retained Python, one namespace.

    The compiled function calls a name the module does not define; the shell
    hosts it as an ordinary Python callable, and the compiled values around
    the call are still recorded.
    """

    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue

    argument = SSAValue(0, dtype="float64")
    fetched = SSAValue(1, dtype="float64")
    doubled = SSAValue(2, dtype="float64")
    function = Function(
        "mixed",
        [argument],
        {"entry": BasicBlock("entry", [
            Instr("Call", [argument], fetched, attributes={"callee": "retained_lookup"}),
            Instr("Mul", [fetched, fetched], doubled),
            Instr("Ret", [doubled], None),
        ])},
        metadata={"argument_names": ("x",)},
    )

    class _Module:
        functions = {"mixed": function}

    shell = compile_python_shell(
        _Module(), "mixed", bindings={"retained_lookup": lambda x: x + 10.0}
    )
    run = shell.run(x=2.0)

    assert run.result == pytest.approx(144.0)  # (2 + 10) ** 2
    # The retained call's RESULT was recorded like any compiled value.
    assert run.last_value("mixed", 1) == pytest.approx(12.0)
    assert shell.unresolved_callees() == ()


def test_a_binding_may_not_shadow_a_compiled_function(loop_shell):
    module, _shell = loop_shell
    with pytest.raises(ValueError, match="collide"):
        compile_python_shell(
            module, "ps__train", bindings={"ps__helper": lambda a: a}
        )


def test_unresolved_callees_are_reported_before_anything_runs():
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue

    argument = SSAValue(0, dtype="float64")
    fetched = SSAValue(1, dtype="float64")
    function = Function(
        "needy",
        [argument],
        {"entry": BasicBlock("entry", [
            Instr("Call", [argument], fetched, attributes={"callee": "missing_helper"}),
            Instr("Ret", [fetched], None),
        ])},
        metadata={"argument_names": ("x",)},
    )

    class _Module:
        functions = {"needy": function}

    shell = compile_python_shell(_Module(), "needy")
    assert "missing_helper" in shell.unresolved_callees()


def test_the_shell_writes_a_standalone_executable_file(tmp_path):
    """The fully realized form: the compiled program living AS a .py file."""

    import subprocess
    import sys

    module = _lower(_LOOP, "wf")
    shell = compile_python_shell(module, "wf__train")
    target = tmp_path / "emitted_shell.py"
    shell.write(target)

    probe = (
        f"import sys; sys.path.insert(0, {str(tmp_path)!r}); "
        "import emitted_shell as m; "
        "print(m.wf__train(n=3, w=2.0)); print(len(m.TRACE))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    result_line, trace_line = completed.stdout.strip().splitlines()[-2:]
    assert float(result_line) == pytest.approx(1.71475)
    assert int(trace_line) > 0
