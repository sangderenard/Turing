from turing_ssa import ProvenanceTM, SSAView


def test_tape_and_head_views_and_patch():
    tm = ProvenanceTM(width=4)
    tm.init_tape("0101")
    tm.init_head(1)
    tape0 = tm.tape_view(0)
    assert isinstance(tape0[0], SSAView)
    assert [v.value for v in tape0] == [0,1,0,1]
    head0 = tm.head_view(0)
    assert head0.value == int("0100", 2)
    old_id = tape0[1].node_id
    new_val = tape0[1].value ^ 1
    tm.patch_cell(0, 1, new_val)
    tape1 = tm.tape_view(1)
    assert tape1[1].value == new_val
    assert tape1[1].node_id != old_id
    assert tm.graph.nodes


def test_init_tape_from_various_sources():
    tm = ProvenanceTM(width=8)
    tm.init_tape(0b1011)
    assert [v.value for v in tm.tape_view(0)] == [0,0,0,0,1,0,1,1]
    tm.init_tape(b"\xf0")
    assert [v.value for v in tm.tape_view(0)] == [1,1,1,1,0,0,0,0]
    tm.init_tape([1,0,1])
    assert [v.value for v in tm.tape_view(0)][:3] == [1,0,1]
