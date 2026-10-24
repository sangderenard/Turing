"""The whole-class trial plans every method and reports what the surface lost.

``compile_class_to_dll`` seeds its closure from every method.  Before the
seeds were activation roots, the deployment planned ``__init__`` alone and
the other methods vanished without a refusal; the trial is what made that
visible, so its reading of the planned set and the surface is pinned here.
"""

from __future__ import annotations

from types import SimpleNamespace

from tools.trial_whole_class_compile import method_coverage, run_class_route


TOY = '''
class Toy:
    def __init__(self, n: int) -> None:
        self.n = n
        self.steps = 0

    def bump(self, k: int) -> int:
        self.steps += k
        return self.steps * self.n

    def twice(self, k: int) -> int:
        return self.bump(k) + self.bump(k)
'''


def test_class_route_plans_every_seeded_method(tmp_path):
    source = tmp_path / "toy.py"
    source.write_text(TOY, encoding="utf-8")
    route = run_class_route(source, "Toy", tmp_path / "out")
    assert route["planned_methods"] == ["__init__", "bump", "twice"]
    coverage = route["method_coverage"]
    assert set(coverage) == {"__init__", "bump", "twice"}
    # Every method the module lacks is named, never silently passed over.
    assert route["false_acceptance_candidates"] == sorted(
        method for method, state in coverage.items() if state == "missing"
    )


def test_method_coverage_reads_source_qualified_names():
    def function(name):
        return SimpleNamespace(metadata={"source_qualified_name": name})

    module = SimpleNamespace(functions={
        "Toy____init__": function("Toy.__init__"),
        "Toy____init____planned_region_0": SimpleNamespace(metadata={}),
    })
    assert method_coverage("Toy", ["__init__", "bump"], module) == {
        "__init__": "present", "bump": "missing",
    }
    assert method_coverage("Toy", ["bump"], None) == {"bump": "unreached"}


ACC = '''
class Acc:
    def __init__(self) -> None:
        pass

    def add(self, a: int, b: int) -> int:
        return a + b
'''


def test_class_route_never_ships_a_library_without_a_planned_method(tmp_path):
    source = tmp_path / "acc.py"
    source.write_text(ACC, encoding="utf-8")
    route = run_class_route(source, "Acc", tmp_path / "out")
    assert route["planned_methods"] == ["__init__", "add"]
    missing = route["false_acceptance_candidates"]
    if route["refusal"] is None:
        # A library was produced: it must hold every method.
        assert missing == []
    else:
        # A dropped method is refused by name, never passed over.
        for method in missing:
            assert repr(method) in route["refusal"]["message"]
