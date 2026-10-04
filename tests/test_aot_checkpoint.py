import pickle

import pytest
from sympy.core.parameters import global_parameters

from src.common.tensors.accelerator_backends.aot_checkpoint import (
    AOTCheckpointStore,
    ReductionArtifactStore,
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


@pytest.mark.parametrize("payload", [b"!invalid pickle", b"\x80\x05\x95"])
def test_corrupt_checkpoint_is_a_miss_and_can_be_replaced(tmp_path, payload):
    store = AOTCheckpointStore({"case": "corrupt"}, tmp_path)
    path = store.store("compiled_plan", "impl", {"old": True})
    path.write_bytes(payload)

    assert store.load("compiled_plan", "impl") is None
    assert store.last_load_status.startswith("miss: UnpicklingError:")

    store.store("compiled_plan", "impl", {"rebuilt": True})
    assert store.load("compiled_plan", "impl") == {"rebuilt": True}
    assert store.last_load_status == "hit"


@pytest.mark.parametrize("payload", [b"!invalid pickle", b"\x80\x05\x95"])
def test_corrupt_reduction_is_rebuilt_then_reused(tmp_path, payload):
    store = ReductionArtifactStore("impl", tmp_path)
    store.get_or_compute("region", lambda: {"old": True})
    next(store.directory.glob("*.pkl")).write_bytes(payload)
    calls = []

    def rebuild():
        calls.append("rebuilt")
        return {"rebuilt": True}

    assert store.get_or_compute("region", rebuild) == ({"rebuilt": True}, False)
    assert store.get_or_compute("region", rebuild) == ({"rebuilt": True}, True)
    assert calls == ["rebuilt"]
    assert (store.misses, store.hits) == (2, 1)


def test_reduction_compute_unpickling_error_is_not_hidden(tmp_path):
    store = ReductionArtifactStore("impl", tmp_path)

    def fail():
        raise pickle.UnpicklingError("producer failure")

    with pytest.raises(pickle.UnpicklingError, match="producer failure"):
        store.get_or_compute("region", fail)
