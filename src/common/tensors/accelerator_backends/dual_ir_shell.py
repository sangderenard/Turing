"""The numeric/control shell with its accompanying map IR.

Just the IR object -- not a compiler, not an executor.  ``DualIRShell``
pairs the programs the AOT pipeline already produces and already names
(``aot_compile.AOTCompilation.compiled_shell_program`` /
``.shell_control_program``): the numeric program (``FusedProgram``, a flat
list of ``OpStep``, no control flow) and the control program
(``ControlProgram``, the loop/branch shell that references numeric regions
via ``__scheduled_region_N__`` markers). ``map_ir`` carries graph identities,
object/resource identities, and their declared permissions.

A shell is more than just numeric+control, though: it also carries the
profiling and logging the instrumented C shell (``profiled_c_shell.py``)
already produces when something actually runs it, and it composes
hierarchically -- a shell may have child shells (e.g. one per scheduled
region), and ``rollup_profile``/``rollup_log`` walk down and report back up
to the root.  ``profile``/``log_messages``/``children`` are all optional:
a bare shell with none of them is exactly the numeric+control pair above,
nothing more.

Composing one from real AOT output is ``compose_dual_ir_shell`` below --
it only reads an already-produced ``AOTCompilation`` and an already-produced
``CLaunchProfile`` and arranges them into this object.  It does not compile
or execute anything itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TYPE_CHECKING, Optional

from ....compiler.control_source import ControlProgram
from ..fused_ir import FusedProgram
from .profiled_c_shell import CLaunchProfile, ShellLanguage

if TYPE_CHECKING:
    # aot_compile.py imports DualIRShell, so the reverse import can only be
    # a type-checking-time one -- annotations are strings at runtime
    # (from __future__ import annotations) so this never actually executes.
    from .aot_compile import AOTCompilation


@dataclass(frozen=True)
class DualIRShell:
    """The control meta and numeric meta the AOT already uses, plus the
    optional profiling/logging/hierarchy that turns the pair into a full
    shell description."""

    compiled_shell_program: FusedProgram
    shell_control_program: Optional[ControlProgram] = None
    map_ir: Optional[Mapping[str, Any]] = None
    name: Optional[str] = None
    profile: Optional[CLaunchProfile] = None
    log_messages: tuple[str, ...] = ()
    children: tuple["DualIRShell", ...] = field(default_factory=tuple)

    def rollup_profile(self) -> Optional[CLaunchProfile]:
        """This shell's own profile combined with every descendant's,
        recursively, up to this call's root.  Timings sum (a parent shell's
        wall/device time is the total time its children took), status is
        the worst of any launch in the tree, and language is kept only if
        every shell in the tree agrees on one."""

        profiles = [
            p
            for p in (self.profile, *(c.rollup_profile() for c in self.children))
            if p is not None
        ]
        if not profiles:
            return None
        languages = {p.language for p in profiles}
        return CLaunchProfile(
            shell_ns=sum(p.shell_ns for p in profiles),
            device_ns=sum(p.device_ns for p in profiles),
            status=max(p.status for p in profiles),
            language=next(iter(languages)) if len(languages) == 1 else ShellLanguage.UNKNOWN,
        )

    def rollup_log(self) -> tuple[str, ...]:
        """This shell's own log messages followed by every descendant's,
        in depth-first order, up to this call's root."""

        messages = list(self.log_messages)
        for child in self.children:
            messages.extend(child.rollup_log())
        return tuple(messages)


def compose_dual_ir_shell(
    aot: AOTCompilation,
    *,
    profile: Optional[CLaunchProfile] = None,
    log_messages: tuple[str, ...] = (),
    children: tuple[DualIRShell, ...] = (),
    name: Optional[str] = None,
) -> DualIRShell:
    """Arrange an already-produced ``AOTCompilation`` (and, optionally, an
    already-produced launch profile) into a ``DualIRShell``.  Reads
    ``aot.compiled_shell_program``/``aot.shell_control_program`` -- the same
    numeric/control/map members ``compile_ast_aot`` already returns -- and does not
    compile or execute anything itself."""

    return DualIRShell(
        compiled_shell_program=aot.compiled_shell_program,
        shell_control_program=aot.shell_control_program,
        map_ir=getattr(aot, "map_ir", None),
        name=name or aot.entrypoint,
        profile=profile,
        log_messages=log_messages,
        children=children,
    )


__all__ = ["DualIRShell", "compose_dual_ir_shell"]
