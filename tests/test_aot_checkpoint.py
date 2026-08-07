from sympy.core.parameters import global_parameters

from src.common.tensors.accelerator_backends.aot_checkpoint import (
    AOTCheckpointStore,
)


def test_checkpoint_restores_sympy_global_parameter_singleton(tmp_path):
    store = AOTCheckpointStore({"case": "sympy-parameters"}, tmp_path)

    store.store(
        "compiled_plan",
        "implementation",
        {"parameters": global_parameters},
    )
    restored = store.load("compiled_plan", "implementation")

    assert store.last_load_status == "hit"
    assert restored["parameters"] is global_parameters
    assert restored["parameters"].evaluate is True
