"""Declared argument facts keep their fields and aliases through archives."""

import base64
import copy
import pickle

import pytest

from src.compiler.concordance_declarations import ArgumentBindingFact


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_argument_fact_archive_preserves_fields_and_source_alias(protocol):
    source = {"value": 76, "nested": [None, 17]}
    fact = ArgumentBindingFact(kind=27, source=source)
    restored, restored_source = pickle.loads(pickle.dumps((fact, source), protocol=protocol))
    assert type(restored) is ArgumentBindingFact
    assert restored.kind == "27"
    assert restored.source == source
    assert restored.source is restored_source
    assert fact.source is source


def test_argument_fact_deepcopy_preserves_shared_source():
    source = {"value": [17, 19]}
    fact = ArgumentBindingFact("caller_value", source)
    restored, restored_source = copy.deepcopy((fact, source))
    assert type(restored) is ArgumentBindingFact
    assert restored == fact
    assert restored.source is restored_source
    assert restored.source is not source


def test_argument_fact_restores_an_actual_legacy_protocol_four_archive():
    # Produced before __getnewargs__: tuple.__getnewargs__ supplied ONE pair
    # to NEWOBJ. These are archived bytes, not a call to the repaired writer.
    legacy = base64.b64decode(
        "gASVbgAAAAAAAACMJXNyYy5jb21waWxlci5jb25jb3JkYW5jZV9kZWNsYXJhdGlvbnOU"
        "jBNBcmd1bWVudEJpbmRpbmdGYWN0lJOUjAxjYWxsZXJfdmFsdWWUfZQojAV2YWx1ZZRL"
        "TIwGc291cmNllE51hpSFlIGULg==")
    restored = pickle.loads(legacy)
    assert type(restored) is ArgumentBindingFact
    assert restored.kind == "caller_value"
    assert restored.source == {"value": 76, "source": None}


def test_argument_fact_keeps_explicit_none_and_rejects_nonlegacy_arity():
    fact = ArgumentBindingFact(("a", "b"), None)
    assert fact.kind == "('a', 'b')" and fact.source is None
    for argument in ("caller_value", [], ["caller_value", 76], (), ("caller_value",), (1, 2, 3)):
        with pytest.raises(TypeError):
            ArgumentBindingFact(argument)
