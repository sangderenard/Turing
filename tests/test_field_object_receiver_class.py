"""A field that holds a source-local object resolves the methods called on it.

``self.ordnance = OrdnanceField()`` in ``__init__`` and later, in a loop,
``self.ordnance.step(dt)``.  The read ``self.ordnance`` carried no class,
so the method call had no ``method_ref``, so the loop classifier recorded
it as a state effect with no model -- ``opaque`` -- and the loop refused:

    blockers=('opaque-state-effect',) ... 'receiver_class': None

Measured on the same loop with the method on ``self`` (whose class IS
known): the call resolves, becomes a source-linked call, the classifier's
own rule skips it ("the callee owns its effects"), and the loop lowers.
The only missing fact was the class the field holds.

The reducer now records ``(owner, field) -> class`` from constructor
assignments in the owner's methods and from class-level annotations, and
stamps ``result_class_ref`` on the field read -- the same fact a call
returning a class instance carries, and NOT ``class_ref``, whose presence
the deployment side reads as a construction.  A field assigned two
different classes is contested and stays unresolved: the table states
only what the source states once.
"""
from pathlib import Path
import sys
import warnings

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.glsl_deployment_strategy import CompilationSubdivisionRequired

SHEET = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

FIELD = """
class Field:
    def __init__(self):
        self.x = 0.0

    def step(self, dt):
        self.x = self.x + dt
"""

DIRECT = FIELD + """

class Sim:
    def __init__(self):
        self.field = Field()
        self.total = 0.0

    def step(self, dt):
        for i in range(4):
            self.field.step(dt)
            self.total = self.total + self.field.x
        return self.total
"""

# the engine's own spelling: ``turb = self._turbine; turb.step(...)``
ALIASED = FIELD + """

class Sim:
    def __init__(self):
        self._field = Field()
        self.total = 0.0

    def step(self, dt):
        f = self._field
        for i in range(4):
            f.step(dt)
            self.total = self.total + f.x
        return self.total
"""

CONTESTED = FIELD + """

class Other:
    def __init__(self):
        self.x = 0.0

    def step(self, dt):
        self.x = self.x - dt


class Sim:
    def __init__(self, flag):
        if flag:
            self.field = Field()
        else:
            self.field = Other()
        self.total = 0.0

    def step(self, dt):
        for i in range(4):
            self.field.step(dt)
            self.total = self.total + self.field.x
        return self.total
"""


def _lower(source, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lower_ast_source_to_ssa(
            source, "Sim.step", name=name,
            extraction_contract=ExtractionContract(SHEET),
        )


@pytest.mark.parametrize("label, source", [("direct", DIRECT), ("aliased", ALIASED)])
def test_a_method_on_a_field_held_object_is_a_linked_call(label, source):
    module, _outputs, _exports = _lower(source, label)
    assert f"{label}__Field__step" in module.functions, sorted(module.functions)
    entry = module.functions[f"{label}__Sim__step"]
    linked = [
        instruction
        for block in entry.blocks.values()
        for instruction in block.instrs
        if instruction.op in {"Call", "call"}
        and instruction.attributes.get("source_linked")
    ]
    assert [str(call.attributes["callee"]) for call in linked] == [
        f"{label}__Field__step"
    ]
    assert "loop_body" in entry.blocks, list(entry.blocks)


def test_a_field_holding_two_classes_stays_unresolved():
    with pytest.raises(CompilationSubdivisionRequired) as refusal:
        _lower(CONTESTED, "contested")
    message = str(refusal.value)
    assert "opaque-state-effect" in message
    assert "'receiver_class': None" in message
