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


def test_prune_deletes_superseded_phases(tmp_path):
    from src.common.tensors.accelerator_backends.aot_checkpoint import AOTCheckpointStore
    store = AOTCheckpointStore({"case": "prune"}, tmp_path)
    store.store("compiled_plan", "impl", {"big": list(range(1000))})
    store.store("captured_program", "impl", {"small": 1})
    plan_pkl = store._paths("compiled_plan")[0]
    assert plan_pkl.exists()
    reclaimed = store.prune("compiled_plan")
    assert reclaimed > 0
    assert not plan_pkl.exists()
    # The superseding phase is untouched and still loads.
    assert store.load("captured_program", "impl") == {"small": 1}
    # Pruning an absent phase is a harmless no-op.
    assert store.prune("compiled_plan") == 0
