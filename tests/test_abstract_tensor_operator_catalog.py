from __future__ import annotations

import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.numpy_backend import NumPyTensorOperations
from src.common.tensors.operator_catalog import (
    NON_OPERATOR_PUBLIC_API,
    OPERATOR_ALIASES,
    PUBLIC_ABSTRACT_TENSOR_OPERATOR_NAMES,
)
from src.transmogrifier.operator_defs import (
    abstract_tensor_funcs,
    abstract_tensor_sigs,
    operator_signatures,
)


def test_public_abstract_tensor_api_is_explicitly_classified():
    public = {
        name for name in dir(AbstractTensor)
        if not name.startswith("_")
    }

    assert not (
        public
        - PUBLIC_ABSTRACT_TENSOR_OPERATOR_NAMES
        - NON_OPERATOR_PUBLIC_API
    )


def test_every_catalogued_operator_has_a_handler_and_schema():
    expected = PUBLIC_ABSTRACT_TENSOR_OPERATOR_NAMES

    assert expected <= abstract_tensor_funcs.keys()
    assert expected <= abstract_tensor_sigs.keys()
    assert expected <= operator_signatures.keys()
    assert all(callable(abstract_tensor_funcs[name]) for name in expected)


def test_compatibility_spellings_share_the_canonical_handler():
    for alias, canonical in OPERATOR_ALIASES.items():
        assert abstract_tensor_funcs[alias] is abstract_tensor_funcs[canonical]
        assert abstract_tensor_sigs[alias] is abstract_tensor_sigs[canonical]


def test_newly_catalogued_handlers_execute_through_abstract_tensor():
    value = NumPyTensorOperations.tensor(
        np.arange(6, dtype=np.float32).reshape(2, 3)
    )

    reshaped = abstract_tensor_funcs["reshape"](value, 3, 2)
    clamped = abstract_tensor_funcs["clamp"](value, 1.0, 4.0)
    transposed = abstract_tensor_funcs["T"](value)

    np.testing.assert_array_equal(
        np.asarray(reshaped.tolist()),
        np.arange(6, dtype=np.float32).reshape(3, 2),
    )
    np.testing.assert_array_equal(
        np.asarray(clamped.tolist()),
        np.asarray([[1, 1, 2], [3, 4, 4]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(transposed.tolist()),
        np.arange(6, dtype=np.float32).reshape(2, 3).T,
    )
