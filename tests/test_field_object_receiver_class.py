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
from src.transmogrifier.graph import graph_express2

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


# ---------------------------------------------------------------------------
# The class the field holds is IMPORTED.  Two facts, both measured 2026-09-16:
#
# 1. Source pursuit resolves ``self.field.step`` only if ``self`` resolves.
#    For a class that exists only as submitted text there is no Python
#    object, so ``self`` resolved to nothing and every such call was reported
#    ``dynamic_or_primitive``.  Methods of a source-text class now bind
#    ``self``/``cls`` to the ``ast.ClassDef`` and a field read on it resolves
#    through the class body's own ``self.x = Cls()``.
# 2. Once the call resolves, the extraction contract decides whether the
#    method's origin may be ingested.  A file under no declared root is
#    ``unknown`` and rejected ``provenance_not_declared``; the refusal now
#    says so, with the origin, instead of "class unresolved".

IMPORTED_MODULE = """
class ImportedField:
    def __init__(self):
        self.x = 0.0

    def step(self, dt):
        self.x = self.x + dt
"""

IMPORTS = """
from {module} import ImportedField


class Sim:
    def __init__(self):
        self.field = ImportedField()
        self.total = 0.0

    def step(self, dt):
        for i in range(4):
            self.field.step(dt)
            self.total = self.total + self.field.x
        return self.total
"""


def _imported_module(tmp_path, monkeypatch, name):
    (tmp_path / f"{name}.py").write_text(IMPORTED_MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return IMPORTS.format(module=name)


def test_self_of_a_source_text_class_resolves_its_field_to_the_imported_class(
    tmp_path, monkeypatch,
):
    import ast
    import inspect

    source = _imported_module(tmp_path, monkeypatch, "held_module_a")
    tree = ast.parse(source)
    _tree, _links, unresolved, _bindings = (
        graph_express2._expand_unresolved_ast_parents(
            tree, {}, include=ExtractionContract(SHEET),
            pursuit_roots=("Sim.step",),
        )
    )
    step_calls = [
        record for record in unresolved
        if record["name"] == "step" and record["owner_name"] == "step"
    ]
    assert len(step_calls) == 1, unresolved
    # resolved -- the target is known -- and then refused by the contract
    assert step_calls[0]["reason"] == "declared_boundary"
    assert step_calls[0]["target_qualname"] == "ImportedField.step"
    assert step_calls[0]["extraction_contract"]["classification"] == "unknown"
    assert step_calls[0]["extraction_contract"]["parameters"]["reason"] == (
        "provenance_not_declared"
    )
    assert inspect.isclass(
        graph_express2._resolve_ast_parent_reference(
            ast.parse("self.field").body[0].value,
            {"self": next(
                node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "Sim"
            )},
        )
    )


def test_a_declared_root_lets_the_imported_field_class_be_pursued(
    tmp_path, monkeypatch,
):
    source = _imported_module(tmp_path, monkeypatch, "held_module_b")
    policy = ExtractionContract(SHEET).with_roots(authored=[tmp_path])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _exports = lower_ast_source_to_ssa(
            source, "Sim.step", name="held", extraction_contract=policy,
        )
    entry = module.functions["held__Sim__step"]
    linked = [
        str(instruction.attributes["callee"])
        for block in entry.blocks.values()
        for instruction in block.instrs
        if instruction.op in {"Call", "call"}
        and instruction.attributes.get("source_linked")
    ]
    assert len(linked) == 1 and linked[0].endswith("__step"), linked
    assert linked[0] != "held__Sim__step"
    assert "loop_body" in entry.blocks, list(entry.blocks)


def test_an_undeclared_root_is_named_by_the_refusal(tmp_path, monkeypatch):
    source = _imported_module(tmp_path, monkeypatch, "held_module_c")
    with pytest.raises(CompilationSubdivisionRequired) as refusal:
        _lower(source, "undeclared")
    message = str(refusal.value)
    assert "opaque-state-effect" in message
    assert "provenance_not_declared" in message
    assert "held_module_c.ImportedField.step" in message
    assert "held_module_c.py" in message
    assert "with_roots" in message
