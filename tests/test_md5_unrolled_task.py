import numpy as np
from learning_tasks.md5_unrolled_task import make_sample


def test_make_sample_shapes():
    msg = b"abc"
    inp, tgt, cat = make_sample(msg, supervise_every=2)
    assert inp.shape[0] == 640
    assert tgt.shape[0] == 128
    expected_depth = 64 // 2
    assert inp.shape[1] == expected_depth
    assert tgt.shape[1] == expected_depth
    assert cat["message_hex"] == msg.hex()

