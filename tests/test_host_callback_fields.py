"""A stored callable field is a native call through a host-supplied pointer.

``self.log(i)`` where ``log`` is a ``Callable`` field calls code the host
owns.  The compiled method takes, per callable field, a C function pointer
and an opaque context handle (``turing-host-callback-v1``); the constructor's
store of the callable is the host's, so its parameter is no native input.
The program API names every such formal and its signature.  Results and the
order of the host's calls must match Python, with and without an optional
callback.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest
import yaml

from src.compiler.compile_section_to_dll import compile_class_to_dll
from src.compiler.fortran_c_shell import host_callback_signature
from src.compiler.runtime_abi import HOST_CALLBACK_SCHEMA


WALKER = '''
from typing import Callable, Optional


class Walker:
    def __init__(
        self,
        log: Callable[[int], None],
        score: Optional[Callable[[int, float], float]] = None,
    ) -> None:
        self.log = log
        self.score = score
        self.steps = 0

    def run(self, n: int) -> float:
        total = 0.0
        for i in range(n):
            self.log(i)
            if self.score is not None:
                total += self.score(i, total)
            self.log(-i)
        self.log(100)
        return total
'''

LOG = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int64)
SCORE = ctypes.CFUNCTYPE(
    ctypes.c_double, ctypes.c_void_p, ctypes.c_int64, ctypes.c_double,
)


@pytest.fixture(scope="module")
def walker(tmp_path_factory):
    output = tmp_path_factory.mktemp("walker")
    source = output / "walker.py"
    source.write_text(WALKER, encoding="utf-8")
    compile_class_to_dll(
        WALKER, "Walker", str(output / "class_route"),
        progress=lambda _message: None,
    )
    route = output / "class_route"
    api = yaml.safe_load((route / "Walker.api.yaml").read_text())
    library = ctypes.CDLL(str(next(route.glob("Walker.so"))))
    namespace: dict = {}
    exec(compile(WALKER, str(source), "exec"), namespace)
    return api, library, namespace["Walker"]


def _entry(api, symbol):
    return next(entry for entry in api["entry_points"] if entry["name"] == symbol)


def test_api_names_each_callback_formal_and_signature(walker):
    api, _library, _python = walker
    metadata = api["metadata"]
    assert metadata["host_callback_schema"] == HOST_CALLBACK_SCHEMA
    run = {item["field"]: item for item in metadata["host_callbacks"]["Walker__run"]}
    assert run["log"]["signature"]["arguments"] == ["int64"]
    assert run["log"]["signature"]["result"] is None
    assert run["score"]["signature"]["arguments"] == ["int64", "float64"]
    assert run["score"]["signature"]["result"] == "float64"
    assert run["score"]["signature"]["optional"] is True
    named = {
        parameter.get("source_name")
        for parameter in _entry(api, "Walker__run")["parameters"]
    }
    for item in run.values():
        assert item["function_parameter"] in named
        assert item["context_parameter"] in named
    # The constructor binds the callables for the host; it takes neither.
    bindings = metadata["host_callback_bindings"]["Walker____init__"]
    assert {item["field"]: item["parameter"] for item in bindings} == {
        "log": "log", "score": "score",
    }
    init_names = {
        parameter.get("source_name")
        for parameter in _entry(api, "Walker____init__")["parameters"]
    }
    assert not init_names & {"log", "score"}


def _run_native(api, library, n, log, score):
    entry = _entry(api, "Walker__run")
    callbacks = {
        "log.__callback_function": ctypes.cast(log, ctypes.c_void_p).value,
        "log.__callback_context": 7,
        "score.__callback_function": (
            0 if score is None else ctypes.cast(score, ctypes.c_void_p).value
        ),
        "score.__callback_context": 9,
    }
    arguments, types, result = [], [], ctypes.c_double()
    for parameter in entry["parameters"]:
        name = parameter.get("source_name")
        if parameter["role"] == "output":
            arguments.append(ctypes.byref(result))
            types.append(ctypes.POINTER(ctypes.c_double))
        elif name == "n":
            arguments.append(n)
            types.append(ctypes.c_int64)
        else:
            arguments.append(callbacks[name])
            types.append(ctypes.c_int64)
    function = getattr(library, entry["symbol"])
    function.argtypes = types
    function(*arguments)
    return result.value


@pytest.mark.parametrize("n", [0, 1, 4])
@pytest.mark.parametrize("with_score", [True, False])
def test_results_and_call_order_match_python(walker, n, with_score):
    api, library, python_walker = walker
    events: list = []
    contexts: set = set()

    def log(value):
        events.append(("log", value))

    def score(index, total):
        events.append(("score", index, total))
        return index * 1.5 + 0.25

    def native_log(context, value):
        contexts.add(("log", context))
        log(value)

    def native_score(context, index, total):
        contexts.add(("score", context))
        return score(index, total)

    native_callbacks = (LOG(native_log), SCORE(native_score))
    native = _run_native(
        api, library, n, native_callbacks[0],
        native_callbacks[1] if with_score else None,
    )
    native_events, events[:] = list(events), []
    expected = python_walker(log, score if with_score else None).run(n)
    assert (native, native_events) == (expected, events)
    assert contexts <= {("log", 7), ("score", 9)}
    assert ("log", 7) in contexts


def test_signature_reads_the_annotation_and_refuses_what_it_cannot_carry():
    assert host_callback_signature("Callable[[int, float], bool]") == {
        "schema": HOST_CALLBACK_SCHEMA,
        "annotation": "Callable[[int, float], bool]",
        "optional": False,
        "arguments": ["int64", "float64"],
        "result": "bool",
    }
    assert host_callback_signature("Callable[[int], None] | None")["optional"]
    with pytest.raises(ValueError):
        host_callback_signature("Callable[..., int]")
    with pytest.raises(ValueError):
        host_callback_signature("Callable[[str], int]")


def test_c_and_llvm_lanes_refuse_a_host_callback():
    from src.compiler.ssa_c_backend import emit_ssa_function_to_c
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
    from src.transmogrifier.ssa import (
        BasicBlock, Function, Instr, IRModule, SSAValue,
    )

    function_pointer, context, argument, result = (
        SSAValue(index, "int64") for index in range(4)
    )
    block = BasicBlock("entry", [
        Instr(
            "HostCallback", [function_pointer, context, argument], result,
            attributes={"host_callback_field": ("Walker", "log")},
        ),
        Instr("Ret", [result], None),
    ])
    module = IRModule(functions={"probe": Function(
        "probe", [function_pointer, context, argument], {"entry": block},
    )})
    for emit in (emit_ssa_function_to_c, emit_ssa_function_to_llvm):
        with pytest.raises(ValueError, match="host callback"):
            emit(module, "probe")


ESCAPE = '''
from typing import Callable


class Escape:
    def __init__(self, log: Callable[[int], None]) -> None:
        self.log = log

    def poke(self, n: int) -> int:
        f = self.log
        f(n)
        return n + 1
'''


def test_a_callable_field_used_as_a_value_refuses(tmp_path):
    # It used to compile, with the call through the local pruned: the host
    # was never called.
    with pytest.raises(Exception, match="host callback field"):
        compile_class_to_dll(
            ESCAPE, "Escape", str(tmp_path / "class_route"),
            progress=lambda _message: None,
        )
