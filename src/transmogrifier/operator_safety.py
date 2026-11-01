"""Failure outcomes are part of fundamental SSA operator definitions.

Enabling arms does not insert a source/SSA filtering pass. Instruction creation
retains the compile policy; each backend materializes the operator's own arms.
A failed result is unavailable to ordinary consumers until failure is handled.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorFailureArm:
    condition: str
    exception_type: str
    code: int
    operand_positions: tuple[int, ...]
    applicability: str = "all_numeric_divisors"


DIVISION_FAILURE_ARMS = (
    OperatorFailureArm('divisor_zero', 'builtins.ZeroDivisionError', 88, (1,)),
    OperatorFailureArm('signed_quotient_out_of_range', 'builtins.OverflowError', 86, (0, 1), 'signed_integral_quotient'),
)
OPERATOR_FAILURE_ARMS = {name: DIVISION_FAILURE_ARMS for name in ('Div', 'FloorDiv', 'Mod')}
_configuration = ContextVar('ssa_operator_safety_configuration', default=None)


@contextmanager
def operator_safety_configuration(enabled, book):
    from ..compiler.concordance_declarations import COMPILE_ENTRY, COMPILE_POLICY, POLICY_DECLARATION
    from ..compiler.identity_concordance import Mode, Novel
    if type(enabled) is not bool:
        raise TypeError('operator_safety_arms must be bool')
    policy = book.post(COMPILE_POLICY, ('operator_safety_arms',), 'on' if enabled else 'off',
                       stage=COMPILE_ENTRY, provenance=Novel(POLICY_DECLARATION, ()), mode=Mode.CONCORD)
    token = _configuration.set((book, policy))
    try:
        yield policy
    finally:
        _configuration.reset(token)


def instruction_safety_attributes(op):
    if str(op) not in OPERATOR_FAILURE_ARMS or _configuration.get() is None:
        return {}
    from ..compiler.concordance_declarations import COMPILE_POLICY
    book, policy = _configuration.get()
    enabled = book.page(COMPILE_POLICY).cell(policy.row, policy.column) == "on"
    return {'operator_safety_arms': enabled, 'operator_safety_policy': policy,
            **({'effectful': True, 'ordered_effect': True} if enabled else {})}
