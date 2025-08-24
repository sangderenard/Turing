import numpy as np

from src.rendering.ascii_diff.ascii_kernel_classifier import AsciiKernelClassifier
from src.common.tensors import AbstractTensor
from src.common.tensors.numpy_backend import NumPyTensorOperations


def test_nn_classifier_matches_reference_bitmasks():
    ramp = " .:"
    classifier = AsciiKernelClassifier(ramp, use_nn=True, epsilon=1e-3, max_epochs=50)
    assert classifier.charBitmasks is not None
    np_backend = AbstractTensor.get_tensor(cls=NumPyTensorOperations)
    batch = np.stack([bm.to_backend(np_backend).numpy() for bm in classifier.charBitmasks], axis=0) * 255.0
    result = classifier.classify_batch(batch)
    assert result["chars"] == classifier.charset
    assert classifier.nn_trained
