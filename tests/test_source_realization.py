from __future__ import annotations

from src.common.tensors.source_realization import (
    authored_source_realization,
    install_authored_deployment,
)


class Surface:
    def calculate(self, value):
        return ("source", value)


def test_installed_call_uses_native_normally_and_source_during_compilation():
    authored = Surface.calculate

    installed = install_authored_deployment(
        Surface,
        "calculate",
        lambda self, value: ("native", value),
    )
    try:
        assert Surface().calculate(3) == ("native", 3)
        with authored_source_realization():
            assert Surface().calculate(3) == ("source", 3)
        assert installed.__turing_authored_source_callable__ is authored
    finally:
        Surface.calculate = authored


def test_target_scoped_compiler_deployment_only_reveals_its_own_source():
    authored = Surface.calculate
    installed = install_authored_deployment(
        Surface,
        "calculate",
        lambda self, value: ("native", value),
        identity="Surface.calculate",
        targeted=True,
    )
    try:
        with authored_source_realization(targets=("another.compiler.unit",)):
            assert Surface().calculate(3) == ("native", 3)
        with authored_source_realization(targets=("Surface.calculate",)):
            assert Surface().calculate(3) == ("source", 3)
        assert installed.__turing_authored_source_callable__ is authored
    finally:
        Surface.calculate = authored
