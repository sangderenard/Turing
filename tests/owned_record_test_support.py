"""Expose the pre-publication boundary for source/ownership proof unit tests.

End-to-end tests use the real entry without this patch. This only withholds the
last new publication step, never an existing compiler analysis or safety gate.
"""
from unittest.mock import patch
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


def lower_before_owned_record_boundary(*args, **kwargs):
    with patch('src.compiler.source_owned_record_transfer.integrate_owned_record_public_returns',
               return_value=()):
        return lower_ast_source_to_ssa(*args, **kwargs)
