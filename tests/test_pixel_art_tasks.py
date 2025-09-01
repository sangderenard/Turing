import threading
from queue import Queue

import numpy as np

from learning_tasks.pixel_art_classifier_task import pump_queue as clf_pump
from learning_tasks.pixel_art_reconstruct_task import pump_queue as rec_pump
from learning_tasks.pixel_shapes import SHAPE_NAMES, SHAPES


def _get_one(pump):
    q: Queue = Queue()
    stop = threading.Event()
    thread = threading.Thread(target=pump, args=(q, (8, 8), 1), kwargs={"stop_event": stop})
    thread.start()
    item = q.get(timeout=2)
    stop.set()
    thread.join(timeout=2)
    return item


def test_classifier_pump_queue_shapes():
    inp, tgt, cat = _get_one(clf_pump)
    assert inp.shape == (1, 8, 8)
    assert tgt.shape == (1, 8, 8)
    assert cat["name"] in SHAPE_NAMES
    assert 0 <= cat["label"] < len(SHAPE_NAMES)


def test_reconstruct_pump_queue_shapes():
    inp, tgt, cat = _get_one(rec_pump)
    assert inp.shape == (1, 8, 8)
    assert tgt.shape == (1, 8, 8)
    assert not np.allclose(inp, tgt)
    assert cat["name"] in SHAPE_NAMES
    assert np.array_equal(tgt[0], SHAPES[cat["name"]])
