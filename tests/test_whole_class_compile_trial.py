"""The whole-class trial plans every method and reports what the surface lost.

``compile_class_to_dll`` seeds its closure from every method.  Before the
seeds were activation roots, the deployment planned ``__init__`` alone and
the other methods vanished without a refusal; the trial is what made that
visible, so its reading of the planned set and the surface is pinned here.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.compiler.ssa_fortran_backend import fortran_compiler

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
    assert coverage == {
        "__init__": "present", "bump": "present", "twice": "present",
    }


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
    # Every seeded activation root is prepared, so the whole-object module
    # holds every method (whether or not a Fortran compiler is installed).
    assert route["method_coverage"] == {"__init__": "present", "add": "present"}
    assert route["false_acceptance_candidates"] == []


@pytest.mark.skipif(
    fortran_compiler() is None, reason="the class route emits through Fortran",
)
def test_seeded_method_runs_natively_and_matches_python(tmp_path):
    source = tmp_path / "acc.py"
    source.write_text(ACC, encoding="utf-8")
    route = run_class_route(source, "Acc", tmp_path / "out")
    assert route["refusal"] is None, route["refusal"]
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(Path(fortran_compiler()).resolve().parent))
    library = ctypes.CDLL(route["library"])
    add = library.Acc__add
    add.restype = None
    add.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64),
    ]
    for a, b in ((2, 3), (-7, 4), (2 ** 40, 5)):
        result = ctypes.c_int64()
        add(a, b, ctypes.byref(result))
        assert result.value == a + b


STORED_CALLABLE = '''
from typing import Callable


class Walker:
    def __init__(self, {name}: Callable[[int], None]) -> None:
        self.{name} = {name}
        self.steps = 0

    def poke(self, n: int) -> int:
        self.{name}(n)
        return n + 1
'''


@pytest.mark.parametrize("name", ["notify", "trace"])
def test_call_through_a_stored_callable_field_calls_the_host(tmp_path, name):
    # ``trace`` also spells a tensor operator: the unconsumed call used to be
    # pruned as a dead pure tensor call and the library shipped without it.
    # It is now a call through the host's function pointer
    # (tests/test_host_callback_fields.py), and it must happen.
    import yaml

    source = tmp_path / "walker.py"
    source.write_text(STORED_CALLABLE.format(name=name), encoding="utf-8")
    route = run_class_route(source, "Walker", tmp_path / "out")
    assert route["refusal"] is None
    api = yaml.safe_load(
        (tmp_path / "out" / "class_route" / "Walker.api.yaml").read_text()
    )
    entry = next(
        item for item in api["entry_points"] if item["name"] == "Walker__poke"
    )
    calls = []
    callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int64)(
        lambda _context, value: calls.append(value)
    )
    values = {
        "n": 41,
        f"{name}.__callback_function": ctypes.cast(
            callback, ctypes.c_void_p,
        ).value,
        f"{name}.__callback_context": 0,
    }
    result = ctypes.c_int64()
    arguments, types = [], []
    for parameter in entry["parameters"]:
        if parameter["role"] == "output":
            arguments.append(ctypes.byref(result))
            types.append(ctypes.POINTER(ctypes.c_int64))
        else:
            arguments.append(values[parameter["source_name"]])
            types.append(ctypes.c_int64)
    library = ctypes.CDLL(
        str(tmp_path / "out" / "class_route" / "Walker.so")
    )
    function = getattr(library, entry["symbol"])
    function.argtypes = types
    function(*arguments)
    assert (result.value, calls) == (42, [41])


SLOTS = '''
class Walker:
    def __init__(self, k: int) -> None:
        self.k = k
        self.steps = 5

    def poke(self, n: int) -> int:
        return n * self.k + 1

    def count(self, n: int) -> int:
        return self.steps * 10 + n
'''


@pytest.mark.skipif(
    fortran_compiler() is None, reason="the class route emits through Fortran",
)
def test_every_method_reads_the_class_table_slots(tmp_path):
    # ``__init__`` stored ``k`` in slot 1 while ``poke`` read slot 0, and a
    # method touching only ``steps`` renumbered it to slot 0: one object
    # arena passes through every method, laid out as the class table says.
    source = tmp_path / "walker.py"
    source.write_text(SLOTS, encoding="utf-8")
    route = run_class_route(source, "Walker", tmp_path / "out")
    assert route["refusal"] is None, route["refusal"]
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(Path(fortran_compiler()).resolve().parent))
    library = ctypes.CDLL(route["library"])
    state = (ctypes.c_double * 2)()
    init = library.Walker____init__
    init.restype = None
    init.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
    ]
    init(2, state, 7)
    assert list(state) == [7.0, 5.0]
    for method, expected in (("poke", 3 * 7 + 1), ("count", 5 * 10 + 3)):
        function = getattr(library, f"Walker__{method}")
        function.restype = None
        function.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
            ctypes.POINTER(ctypes.c_double),
        ]
        result = ctypes.c_double()
        function(2, state, 3, ctypes.byref(result))
        assert result.value == expected
