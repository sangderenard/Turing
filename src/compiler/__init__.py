"""Compiler package; select the complete provenance backend once at startup."""

from .compiler_bootstrap_runtime import activate_native_provenance_core

activate_native_provenance_core()
