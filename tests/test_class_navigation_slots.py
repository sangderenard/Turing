"""Real class instances get an addressed field-slot layout at the nexus.

ClassNavigationMember.slot gives every instance-storage attribute a
monotonic, per-class index -- the same shape ClassFieldSlot
(wasm_class_coordinator.py) already proves out for WebAssembly deployment,
generated here instead during the frontend phase every backend already
goes through (build_class_navigation_table), not re-derived separately per
backend. See GRAPH_DESCRIPTION_LAYER_SURVEY.md for why this exists.
"""

from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot


def test_instance_attributes_get_monotonic_slots_methods_do_not():
    source = """
class Counter:
    def __init__(self, start):
        self.value = start
        self.step = 1

    def bump(self):
        self.value = self.value + self.step
        return self.value

def run(start):
    c = Counter(start)
    return c.bump()
"""
    compilation = compile_ast_aot(source, "run", {"start": 10}, precompile_only=True)
    nav = compilation.class_navigation
    assert nav is not None
    record = nav.class_record("Counter")

    slots_by_name = {
        member.name: member.slot for member in record.members
        if member.kind == "attribute"
    }
    assert slots_by_name == {"value": 0, "step": 1}

    for member in record.members:
        if member.kind == "method":
            assert member.slot is None
            assert member.function_reference is not None


def test_a_program_with_no_classes_has_no_class_records():
    compilation = compile_ast_aot(
        "def add_one(x):\n    return x + 1\n", "add_one", {"x": 1},
        precompile_only=True,
    )
    assert compilation.class_navigation.classes == ()
