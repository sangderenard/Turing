from src.common.tensors.accelerator_backends.glsl_backend import (
    _canonical_cache_value,
    _semantic_cache_digest,
)


def test_semantic_cache_encodes_binary_precompile_attributes_stably():
    payload = b"\xff\xd8\xff\xe0JFIF\x00"

    canonical = _canonical_cache_value(payload)

    assert canonical["bytes_type"] == "builtins.bytes"
    assert canonical["bytes_length"] == len(payload)
    assert _semantic_cache_digest({"constant": payload}) == (
        _semantic_cache_digest({"constant": bytes(payload)})
    )


def test_semantic_cache_binary_identity_changes_with_payload_or_type():
    assert _semantic_cache_digest(b"packet-a") != _semantic_cache_digest(
        b"packet-b"
    )
    assert _semantic_cache_digest(b"packet") != _semantic_cache_digest(
        bytearray(b"packet")
    )
