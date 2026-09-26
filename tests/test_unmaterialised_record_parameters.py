"""A bound record that never reaches the ABI has to say so.

Declare a class in `program_abi.records`, bind it to a parameter in
`program_abi.bindings`, and `records_for_function` resolves it correctly
-- while `parameter_names` never gains the parameter. Everything reached
through it is then genuinely dead, the body empties, and emission reports
ZERO SHORTFALLS on a function whose whole content is `ret void`.

Undeclared, the same program refuses loudly and usefully:

    CompilationSubdivisionRequired ... blockers=('opaque-state-effect',)
        batch.step(dt)

So declaring the record turned a good refusal into a silent no-op. These
pin the complaint that was added to that gap, and -- just as important --
pin that it stays quiet about the wildcard bindings the repository sheet
carries, because a complaint that cries five times for every real case is
one people learn to filter out.
"""
from pathlib import Path
import sys
import warnings

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

SHEET = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

SOURCE = """
def probe(thing, out):
    thing.step(0.001)
    out[0] = 1.0
    return out
"""


def _contract(extra_bindings):
    policy = ExtractionContract(SHEET)
    program_abi = policy.program_abi.receipt()
    program_abi.setdefault("bindings", []).extend(extra_bindings)
    return policy.with_program_abi(program_abi)


def _lower(policy):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module, _outputs, _exports = lower_ast_source_to_ssa(
            SOURCE, "probe", name="probe_module", extraction_contract=policy)
    complaints = [str(w.message) for w in caught
                  if issubclass(w.category, RuntimeWarning)
                  and "never reached the ABI" in str(w.message)]
    return module, complaints


def test_it_complains_when_a_targeted_binding_does_not_materialise():
    policy = _contract([
        {"function": "*probe", "parameter": "thing", "record": "Metrics"},
    ])
    module, complaints = _lower(policy)
    function = module.functions["probe_module__probe"]
    assert "thing" not in dict(function.metadata.get("parameter_names", {}))
    assert complaints, "a bound record vanished and nothing said so"
    assert any("thing" in message for message in complaints)


def test_the_complaint_leaves_a_receipt_on_the_function():
    """Greppable afterwards, not only visible to whoever watched stderr."""
    policy = _contract([
        {"function": "*probe", "parameter": "thing", "record": "Metrics"},
    ])
    module, _complaints = _lower(policy)
    function = module.functions["probe_module__probe"]
    assert "thing" in function.metadata.get(
        "unmaterialised_record_parameters", ())


def test_it_stays_quiet_about_wildcard_bindings():
    """The sheet binds `state`, `targets`, `metrics`, `ctrl` and
    `controller` with a bare `"*"`, so they match every function compiled
    and their absence from any particular one means nothing.

    The first version of this check reported all five as failures on every
    lowering, which is five false positives against one real finding.
    """
    module, complaints = _lower(ExtractionContract(SHEET))
    assert complaints == []
    function = module.functions["probe_module__probe"]
    assert not function.metadata.get("unmaterialised_record_parameters")


def test_lowering_refuses_a_missing_contract():
    """The guard that stops the diagnosis being about the wrong thing.

    Without a contract, `dependency_search: reachable` has no declared
    boundary and pursues the program's whole reachable set. Measured on a
    module that imports the engine sim and the dt system: 589 s of CPU and
    16.6 GB of resident memory, still climbing at a gigabyte every 25
    seconds, with no output at all. With the right contract the same entry
    lowers in seconds.

    The refusal also says the useful half out loud -- that every receiver
    then reports as `opaque-state-effect`, which is "a diagnosis of the
    missing contract, not of the program". That sentence is the difference
    between an afternoon and a day.
    """
    with pytest.raises(ValueError, match="requires an extraction_contract"):
        lower_ast_source_to_ssa(SOURCE, "probe", name="probe_module",
                                extraction_contract=None)
