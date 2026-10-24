"""A record binding names a method by its class, and linked methods keep it.

``SSAReferenceEvaluator._value(self, values, value: SSAValue)`` reads
``value.id``.  The extraction contract already declares ``CompilerSSAValue``,
but a binding spelled ``SSAReferenceEvaluator._value`` never matched: bindings
were compared with the bare method name only, so ``value`` stayed an opaque
input and the unit reported it as an unrepresented used parameter.  Once the
binding matched, two further gaps showed up:

* the receiver lost its class record, because the class view was filled in
  only when no other parameter had a record yet;
* a caller that links the completed ``_value`` unit re-posted that body's
  facts in its own identity book, which does not hold them.
"""

from __future__ import annotations

from src.compiler.extraction_contract import (
    ExtractionContract, binding_names_function,
)
from src.compiler.project_compilation_product import compile_project_call


CONTRACT = "extraction_contracts/program_extraction.yaml"


def test_binding_names_a_method_bare_or_by_its_class():
    assert binding_names_function("_value", "_value", "Ev")
    assert binding_names_function("Ev._value", "_value", "Ev")
    assert not binding_names_function("Ev._value", "_value", "Other")
    assert not binding_names_function("Ev._value", "_value", None)


def test_class_qualified_contract_binding_selects_its_record():
    abi = ExtractionContract(CONTRACT).program_abi
    records = abi.records_for_function(
        "_value", method_owner="SSAReferenceEvaluator",
        parameters=("self", "values", "value"),
    )
    assert records["value"].identity == "src.transmogrifier.ssa.SSAValue"
    assert "value" not in abi.records_for_function(
        "_value", method_owner="SomeOtherClass",
        parameters=("self", "values", "value"),
    )


def test_ssa_value_parameter_is_its_contract_record(tmp_path):
    receipt = compile_project_call(
        "src/compiler/ssa_reference_evaluator.py",
        "SSAReferenceEvaluator._value",
        tmp_path / "value",
        progress_bars=False,
    )
    assert receipt["root_parameter_accounting"]["missing_used_parameters"] == []
    assert receipt["repository_ssa_complete"]


def test_receiver_keeps_its_class_record_beside_a_contract_record(tmp_path):
    # ``_operand`` stops at ``isinstance(stored, _Pointer)`` and
    # ``stored.read()`` on the untyped stored object: an honest frontier.
    # Neither ``self`` nor ``value`` may be the reason.
    receipt = compile_project_call(
        "src/compiler/ssa_reference_evaluator.py",
        "SSAReferenceEvaluator._operand",
        tmp_path / "operand",
        progress_bars=False,
    )
    assert receipt["root_parameter_accounting"]["missing_used_parameters"] == []


LINKED = '''
class Ev:
    def __init__(self) -> None:
        self.steps = 0

    def _value(self, values: dict[int, int], key: int) -> int:
        if key in values:
            return values[key]
        raise KeyError(f"value {key} was read before it was defined")

    def _operand(self, values: dict[int, int], key: int) -> int:
        return self._value(values, key) + 1
'''


def test_caller_links_a_completed_method_with_its_own_book(tmp_path):
    source = tmp_path / "linked_method.py"
    source.write_text(LINKED, encoding="utf-8")
    callee_root = tmp_path / "value"
    callee = compile_project_call(
        source, "Ev._value", callee_root, progress_bars=False,
    )
    assert callee["repository_ssa_complete"]
    caller = compile_project_call(
        source, "Ev._operand", tmp_path / "operand",
        linked_units={
            "Ev._value": (
                callee_root / "repository-ssa.pkl", callee["exports"][0],
            ),
        },
        progress_bars=False,
    )
    assert caller["repository_ssa_complete"]
