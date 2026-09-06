"""Compatibility import for the renamed compiled machine program.

New code should import :class:`BinaryMachineProgram` from
``binary_machine_program``. No Python interpreter is implied by this module's
historical name.
"""

from .binary_machine_program import BinaryMachineProgram, SubjectDeviceBuffers

BinaryMachineRuntime = BinaryMachineProgram

__all__ = [
    "BinaryMachineProgram", "BinaryMachineRuntime", "SubjectDeviceBuffers",
]
