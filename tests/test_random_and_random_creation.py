import pytest
from src.common.tensors.abstraction_methods import creation
from src.common.tensors.abstraction_methods.random import random_generator, Random, RANDOM_KIND, PRNG_ALGO

SEED = 12345

# Helper to compare two tensors/arrays for equality

def arrays_equal(a, b):
    return all(x == y for x, y in zip(a, b))

def test_random_tensor_determinism():
    t1 = creation.random_tensor((5,), seed=SEED)
    t2 = creation.random_tensor((5,), seed=SEED)
    assert arrays_equal(t1.tolist(), t2.tolist()), "random_tensor not deterministic with same seed"

def test_randn_determinism():
    t1 = creation.randn((5,), seed=SEED)
    t2 = creation.randn((5,), seed=SEED)
    assert arrays_equal(t1.tolist(), t2.tolist()), "randn not deterministic with same seed"

def test_randint_determinism():
    t1 = creation.randint((5,), 0, 10, seed=SEED)
    t2 = creation.randint((5,), 0, 10, seed=SEED)
    assert arrays_equal(t1.tolist(), t2.tolist()), "randint not deterministic with same seed"

def test_random_class_determinism():
    r1 = Random(seed=SEED)
    r2 = Random(seed=SEED)
    vals1 = [r1.random() for _ in range(5)]
    vals2 = [r2.random() for _ in range(5)]
    assert arrays_equal(vals1, vals2), "Random.random() not deterministic with same seed"
    # Test gauss
    r1 = Random(seed=SEED)
    r2 = Random(seed=SEED)
    vals1 = [r1.gauss(0, 1) for _ in range(5)]
    vals2 = [r2.gauss(0, 1) for _ in range(5)]
    assert arrays_equal(vals1, vals2), "Random.gauss() not deterministic with same seed"

def test_random_tensor_shapes():
    t = creation.random_tensor((2, 3), seed=SEED)
    assert t.shape == (2, 3)
    t = creation.randn((4, 2), seed=SEED)
    assert t.shape == (4, 2)
    t = creation.randint((3, 2), 0, 5, seed=SEED)
    assert t.shape == (3, 2)

def test_randn_distribution():
    # Check mean and std are roughly correct for randn
    import numpy as np
    t = creation.randn((10000,), seed=SEED)
    arr = t.tolist()
    mean = np.mean(arr)
    std = np.std(arr)
    assert abs(mean) < 0.1, f"randn mean too far from 0: {mean}"
    assert abs(std - 1) < 0.1, f"randn std too far from 1: {std}"
