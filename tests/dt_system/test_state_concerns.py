"""The shared state union, concern masks, and the vectorized delta.

These pin the behaviour the dt-system coordinator relies on to decide who
was invalidated by a step without walking a coupling graph: a union built
from what participants declare, one mask each, one delta over the union.
"""

import pytest

from src.common.dt_system.state_concerns import (
    ConcernUnion,
    changed_mask,
    entry_changed,
    invalidated,
    touches,
)


@pytest.mark.dt
@pytest.mark.fast
def test_union_is_first_appearance_order():
    """Matches ``column_names_of``: a union and a column layout built from
    the same declarations must index identically, or a mask made against
    one silently selects the wrong entry of the other."""

    union = ConcernUnion.of([("p", "v"), ("v", "rho"), ("p",)])
    assert union.names == ("p", "v", "rho")
    assert union.position("v") == 1
    assert len(union) == 3


@pytest.mark.dt
@pytest.mark.fast
def test_mask_selects_only_declared_concerns():
    union = ConcernUnion.of([("p", "v", "rho")])
    mask = union.mask(("p", "rho"))
    assert [bool(x) for x in mask.tolist()] == [True, False, True]


@pytest.mark.dt
@pytest.mark.fast
def test_mask_refuses_a_concern_outside_the_union():
    """A participant naming state nobody publishes is a declaration error.

    Silently producing an all-false mask would make that participant
    permanently un-invalidated -- it would dilate forever on state that
    does not exist, which is exactly the failure that must be loud."""

    union = ConcernUnion.of([("p",)])
    with pytest.raises(KeyError):
        union.mask(("p", "not_published"))


@pytest.mark.dt
@pytest.mark.fast
def test_delta_marks_only_entries_that_moved():
    union = ConcernUnion.of([("p", "v", "rho")])
    before = {"p": [1.0, 2.0], "v": [0.0, 0.0], "rho": [5.0]}
    after = {"p": [1.0, 2.0], "v": [0.0, 1.0], "rho": [5.0]}
    changed = changed_mask(union, before, after)
    assert [bool(x) for x in changed.tolist()] == [False, True, False]


@pytest.mark.dt
@pytest.mark.fast
def test_invalidation_is_mask_intersect_delta():
    """The coordinator's whole question, answered for every participant at
    once from one delta."""

    union = ConcernUnion.of([("p", "v", "rho")])
    masks = {
        "solver": union.mask(("p", "v")),
        "reactor": union.mask(("rho",)),
        "probe": union.mask(("p",)),
    }
    before = {"p": [1.0], "v": [0.0], "rho": [5.0]}
    after = {"p": [1.0], "v": [1.0], "rho": [5.0]}

    changed = changed_mask(union, before, after)
    assert invalidated(changed, masks) == ("solver",)
    assert touches(changed, masks["solver"]) is True
    assert touches(changed, masks["reactor"]) is False


@pytest.mark.dt
@pytest.mark.fast
def test_a_dilating_participant_stays_valid_while_its_concerns_are_still():
    """The reaction-waiting-on-fed-inputs case.

    While nothing it cares about moves, a dilating participant is absent
    from the invalidated set, so its closed-form evaluation stays valid and
    it need not be stepped.  The step its inputs finally move, it comes
    back and must re-bind."""

    union = ConcernUnion.of([("heat", "reagent", "unrelated")])
    reactor = union.mask(("heat", "reagent"))

    quiet_before = {"heat": [300.0], "reagent": [1.0], "unrelated": [0.0]}
    quiet_after = {"heat": [300.0], "reagent": [1.0], "unrelated": [7.0]}
    assert touches(changed_mask(union, quiet_before, quiet_after), reactor) is False

    fed_after = {"heat": [305.0], "reagent": [1.0], "unrelated": [7.0]}
    assert touches(changed_mask(union, quiet_before, fed_after), reactor) is True


@pytest.mark.dt
@pytest.mark.fast
def test_change_is_exact_not_tolerant():
    """Energy accounting audits a dilation claim against these deltas, so a
    tolerance would be a place for an unaccounted exchange to hide."""

    assert entry_changed([1.0], [1.0 + 1e-18]) is False  # below float64 resolution
    assert entry_changed([1.0], [1.0 + 1e-12]) is True


@pytest.mark.dt
def test_union_matches_the_real_piece_column_layout():
    """The union is not a parallel structure -- it is the one
    ``llvm_dt_system`` already builds.

    ``column_names_of(pieces)`` and ``ConcernUnion.of`` must agree in
    ORDER, not merely in membership: a mask built against one indexes
    positionally into the other, so a divergence silently selects the wrong
    column rather than failing.  Skips when the stored pieces are absent;
    it is an integration check against real artifacts, not a unit test.
    """

    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "artifacts/llvm_pieces/voxel_air_step/b1/voxel_air_step.piece",
        root / "artifacts/llvm_pieces/pool_step/b1/pool_step.piece",
    ]
    if not all(path.exists() for path in paths):
        pytest.skip("stored llvm pieces are not present")

    import sys

    sys.path.insert(0, str(root / "examples"))
    from llvm_dt_system import column_names_of  # noqa: E402
    from src.compiler.native_law_kernels import LLVMPiece  # noqa: E402

    pieces = [LLVMPiece.load(path) for path in paths]
    union = ConcernUnion.of(piece.argument_names for piece in pieces)
    # ``column_names_of`` drops ``dt``: it is the step's own column, not a
    # column some piece reads from the shared state.
    assert tuple(n for n in union.names if n != "dt") == column_names_of(pieces)

    masks = {
        piece.artifact.name.split("__")[0]: union.mask(piece.argument_names)
        for piece in pieces
    }
    base = {name: [0.0] for name in union.names}

    def moved(column):
        after = dict(base)
        after[column] = [1.0]
        return invalidated(changed_mask(union, base, after), masks)

    # A column only one piece reads isolates; a physically shared one
    # (here the gas constant and the temperature both simulations couple
    # through) invalidates both, with no coupling graph consulted.
    assert moved("C_cond") == ("voxel_air_step",)
    assert moved("m_p") == ("pool_step",)
    assert moved("T") == ("voxel_air_step", "pool_step")
    assert moved("R_a") == ("voxel_air_step", "pool_step")


@pytest.mark.dt
@pytest.mark.fast
def test_absence_is_distinct_from_zero():
    """An entry nothing published is not an entry published as zero -- the
    same distinction the dt controller's presence tests depend on."""

    assert entry_changed(None, None) is False
    assert entry_changed(None, [0.0]) is True
    assert entry_changed([0.0], None) is True
