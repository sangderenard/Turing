from __future__ import annotations

import string

from src.compiler import site_bundle


SOURCE = """
TURING_PAGE = {
    "entrypoint": "kernel",
    "title": "Compiler Identity Probe",
    "slug": "compiler-identity-probe",
}

def kernel(x):
    return x + 1.0
"""


def test_bundle_compiler_digest_is_stable_sha256():
    first = site_bundle._bundle_compiler_digest()
    second = site_bundle._bundle_compiler_digest()

    assert first == second
    assert len(first) == 64
    assert set(first) <= set(string.hexdigits.lower())


def test_content_version_changes_with_compiler_implementation(monkeypatch):
    contract = site_bundle.discover_source_contract(SOURCE)
    monkeypatch.setattr(
        site_bundle, "_bundle_compiler_digest", lambda: "a" * 64
    )
    first_version, first_source = site_bundle._content_version(
        SOURCE, contract
    )
    monkeypatch.setattr(
        site_bundle, "_bundle_compiler_digest", lambda: "b" * 64
    )
    second_version, second_source = site_bundle._content_version(
        SOURCE, contract
    )

    assert first_source == second_source
    assert first_version != second_version
