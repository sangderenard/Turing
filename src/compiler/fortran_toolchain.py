"""Central optimization and standalone-link policy for generated Fortran."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import re


def is_gnu_fortran(compiler: str) -> bool:
    return "gfortran" in Path(compiler).name.casefold()


def gnu_lto_flag() -> str:
    """Use the available hardware threads for GCC's partitioned LTO work.

    This controls build-time jobs, not program-loop concurrency. Bare -flto
    does not request a job count when the build has no make jobserver.
    """
    return f"-flto={max(1, os.cpu_count() or 1)}"


def aggressive_fortran_flags(
    compiler: str, *, precision_sections: bool = False,
) -> tuple[str, ...]:
    """High optimization without relaxing IEEE/NaN program semantics.

    ``precision_sections`` names the one obligation optimization flags can
    silently break: a module carrying limb arithmetic depends on every
    multiply and add rounding separately except where it spelled
    ``ieee_fma`` itself. The emitted expressions are already parenthesised
    -- Fortran forbids reassociating across parentheses -- but contraction
    is a rewrite the language leaves to the processor, and gfortran's
    default is ``-ffp-contract=fast``. Withdrawing it per-artifact is what
    lets the fortran lane claim SECTION_ISOLATION honestly: an explicit
    ``ieee_fma`` call is an intrinsic invocation, not a contraction, so it
    survives the flag. The cost lands only on modules that carry a
    section; everything else keeps the contraction the fast contract wants.
    """

    name = Path(compiler).name.casefold()
    contract = ("-ffp-contract=off",) if precision_sections else ()
    if "gfortran" in name:
        return (
            "-O3", "-march=native", gnu_lto_flag(), "-funroll-loops",
            "-fomit-frame-pointer", "-ffree-line-length-none", *contract,
        )
    if name.startswith(("ifx", "ifort")):
        # ifx spells the withdrawal through the value-safety model; there
        # is no narrower contraction-only switch that is also honoured.
        return (
            "-O3", "-xHost", "-ipo",
            *(("-fp-model=precise",) if precision_sections else ()),
        )
    if "flang" in name:
        return ("-O3", "-march=native", "-flto", *contract)
    return ("-O3",)


def native_fortran_flags(
    compiler: str, source: str, flags: tuple[str, ...],
) -> tuple[str, ...]:
    """Enable a generated native feature only after this compiler accepts it.

    The native literal/call regression's U64_MAX is unsigned, not a signed
    storage disguise. GNU requires -funsigned. Probe the exact compiler and
    compile flags instead of assuming that its executable name/version proves
    support. Ordinary modules do not require or pay for this feature.
    """
    if re.search(r"(?im)^\s*unsigned\s*\(|\buint\s*\(", source) is None:
        return tuple(flags)
    if not is_gnu_fortran(compiler):
        raise ValueError(
            "native unsigned Fortran requires a compiler supporting -funsigned; "
            f"selected compiler is {compiler!r}"
        )
    enabled = tuple(flags) + (() if "-funsigned" in flags else ("-funsigned",))
    environment = dict(os.environ)
    environment["PATH"] = str(Path(compiler).resolve().parent) + os.pathsep + environment.get("PATH", "")
    with tempfile.TemporaryDirectory(prefix="turing-fortran-unsigned-") as directory:
        probe = Path(directory) / "native_unsigned.f90"
        probe.write_text(
            "subroutine native_unsigned(x) bind(C)\n"
            "use iso_c_binding\n"
            "unsigned(c_int64_t), intent(out) :: x\n"
            "x = 18446744073709551615u_c_int64_t\n"
            "end subroutine\n", encoding="utf-8",
        )
        completed = subprocess.run(
            [compiler, *enabled, "-c", str(probe), "-o", str(probe.with_suffix(".o"))],
            capture_output=True, text=True, env=environment,
        )
    if completed.returncode != 0:
        raise ValueError(
            "selected Fortran compiler/flags do not support exact native unsigned "
            f"ISO C storage with -funsigned: {compiler!r}: {completed.stderr.strip()}"
        )
    return enabled


def aggressive_c_flags(fortran_compiler: str) -> tuple[str, ...]:
    """C-host flags compatible with the selected Fortran link driver."""

    if is_gnu_fortran(fortran_compiler):
        return (
            "-O3", "-march=native", gnu_lto_flag(), "-funroll-loops",
            "-fomit-frame-pointer",
        )
    return ("-O3",)


def stages_shared_runtime(compiler: str, *, shared: bool) -> bool:
    """Whether a standalone artifact carries its GNU runtime beside itself.

    A standalone artifact needs no runtime from the machine it runs on.
    Windows and executables get that by linking the runtime archives in.
    A Linux shared object cannot: distributions build ``libgfortran.a``
    without -fPIC, so ld refuses it inside a shared object ("failed to set
    dynamic section sizes: bad value" once ``error stop`` pulls the runtime).
    There the same promise is kept the other way: the runtime's shared
    objects are staged beside the artifact and found through ``$ORIGIN``.
    """

    return (
        shared and is_gnu_fortran(compiler)
        and os.name != "nt" and sys.platform != "darwin"
    )


def standalone_fortran_link_flags(
    compiler: str, *, shared: bool = False,
) -> tuple[str, ...]:
    """Link compiler runtimes into the artifact instead of requiring DLLs."""

    if not is_gnu_fortran(compiler):
        raise ValueError(
            "standalone runtime linkage is currently defined only for GNU "
            "Fortran; set standalone=False or add a verified compiler policy"
        )
    if stages_shared_runtime(compiler, shared=shared):
        # DT_RPATH, not DT_RUNPATH: RUNPATH serves only the artifact's own
        # needs, while libgfortran's libquadmath is found through its
        # loader's RPATH.  Pair with stage_standalone_fortran_runtime.
        return (gnu_lto_flag(), "-Wl,-rpath,$ORIGIN", "-Wl,--disable-new-dtags")
    flags = [gnu_lto_flag(), "-static-libgfortran", "-static-libgcc"]
    if os.name == "nt":
        # This also selects archive forms of libquadmath and transitive GNU
        # support libraries. Windows system DLLs remain normal OS ABI imports.
        flags.append("-static")
    return tuple(flags)


# libgfortran's ``string.o`` (pulled in by any module that touches the string
# or formatted-I/O runtime) calls ``strndup``.  A mingw-w64 CRT older than the
# installed gcc does not define it, so a ``-static`` link fails to resolve it
# while a DLL-runtime link never pulls the object at all.  Supplying our own
# definition satisfies the reference; the CRT's own archive member, if a later
# CRT has one, is then simply never pulled, so this cannot collide.  No header
# is included, so no platform prototype can conflict with it either.  The
# The standalone link supplies this definition directly.  It must be strong:
# a weak definition is not selected to satisfy libgfortran's archive reference
# on current mingw linkers.  The caller emits this shim only once.
_CRT_SHIM_SOURCE = """\
/* Generated by turing's Fortran toolchain policy.  See fortran_toolchain.py. */
#include <stdlib.h>

char *strndup(const char *s, size_t n)
{
    size_t len = 0;
    size_t i;
    char *out;

    while (len < n && s[len] != '\\0') {
        len++;
    }
    out = (char *)malloc(len + 1);
    if (out == NULL) {
        return NULL;
    }
    for (i = 0; i < len; i++) {
        out[i] = s[i];
    }
    out[len] = '\\0';
    return out;
}
"""


def standalone_runtime_shim_sources(
    compiler: str, directory: Path, standalone: bool = True
) -> tuple[str, ...]:
    """Emit CRT sources a standalone GNU link needs but this CRT lacks."""

    if not standalone or os.name != "nt" or not is_gnu_fortran(compiler):
        return ()
    path = Path(directory) / "turing_crt_shim.c"
    path.write_text(_CRT_SHIM_SOURCE, encoding="utf-8")
    return (str(path),)


GNU_RUNTIME_DLLS = (
    "libgfortran-5.dll",
    "libquadmath-0.dll",
    "libgcc_s_seh-1.dll",
    "libwinpthread-1.dll",
)


def stage_fortran_runtime_dependencies(
    compiler: str,
    destination: str | Path,
) -> tuple[Path, ...]:
    """Place GNU Fortran runtime DLLs beside a dynamically linked artifact."""

    if os.name != "nt" or not is_gnu_fortran(compiler):
        return ()
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    compiler_bin = Path(compiler).resolve().parent
    copied = []
    for name in GNU_RUNTIME_DLLS:
        source = compiler_bin / name
        if not source.is_file():
            continue
        output = target / name
        if source.resolve() != output.resolve():
            shutil.copy2(source, output)
        copied.append(output)
    return tuple(copied)


# The GNU Fortran runtime a standalone shared object carries: libgfortran
# and the GCC support libraries it names.  glibc's own objects (libc, libm,
# the loader) are the platform ABI, as System32 is on Windows.
GNU_RUNTIME_SONAME_PREFIXES = ("libgfortran.", "libquadmath.", "libgcc_s.")


def _elf_dynamic_strings(path: Path) -> tuple[str | None, tuple[str, ...]]:
    """DT_SONAME and DT_NEEDED of a 64-bit little-endian ELF object."""

    data = Path(path).read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError(f"{path} is not a 64-bit little-endian ELF object")
    section_offset, = struct.unpack_from("<Q", data, 0x28)
    entry_size, count = struct.unpack_from("<HH", data, 0x3A)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, section_offset + index * entry_size)
        for index in range(count)
    ]
    dynamic = next(
        (section for section in sections if section[1] == 6), None,
    )
    if dynamic is None:
        return None, ()
    strings = sections[dynamic[6]]
    def string(offset: int) -> str:
        start = strings[4] + offset
        return data[start:data.index(b"\0", start)].decode("utf-8")
    soname = None
    needed = []
    for position in range(dynamic[4], dynamic[4] + dynamic[5], 16):
        tag, value = struct.unpack_from("<qQ", data, position)
        if tag == 0:
            break
        if tag == 1:
            needed.append(string(value))
        elif tag == 14:
            soname = string(value)
    return soname, tuple(needed)


def _gnu_runtime_file(compiler: str, name: str) -> Path | None:
    completed = subprocess.run(
        [compiler, f"-print-file-name={name}"],
        capture_output=True, text=True,
    )
    located = Path(completed.stdout.strip())
    # GCC echoes the bare name back when it has no such file.
    if completed.returncode != 0 or not located.is_absolute() or not located.is_file():
        return None
    return located.resolve()


def stage_standalone_fortran_runtime(
    compiler: str,
    destination: str | Path,
    *,
    shared: bool,
) -> tuple[Path, ...]:
    """Stage the GNU runtime a standalone Linux shared object links to.

    Each object is staged under its DT_SONAME, the name the loader asks
    for, and the closure follows DT_NEEDED within the GNU runtime set.  A
    runtime the toolchain cannot produce is a named refusal, never a
    silently dependent artifact.
    """

    if not stages_shared_runtime(compiler, shared=shared):
        return ()
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    source = _gnu_runtime_file(compiler, "libgfortran.so")
    if source is None:
        raise ValueError(
            f"{compiler} names no shared libgfortran to stage beside a "
            "standalone shared object; set standalone=False"
        )
    pending = [source]
    staged: dict[str, Path] = {}
    while pending:
        source = pending.pop(0)
        soname, needed = _elf_dynamic_strings(source)
        name = soname or source.name
        if name in staged:
            continue
        output = target / name
        if source != output.resolve():
            shutil.copy2(source, output)
        staged[name] = output
        for dependency in needed:
            if dependency in staged or not dependency.startswith(
                GNU_RUNTIME_SONAME_PREFIXES
            ):
                continue
            located = _gnu_runtime_file(compiler, dependency)
            if located is None:
                raise ValueError(
                    f"{source.name} needs {dependency}, which {compiler} "
                    "cannot locate"
                )
            pending.append(located)
    return tuple(staged.values())


__all__ = [
    "GNU_RUNTIME_SONAME_PREFIXES",
    "aggressive_c_flags",
    "aggressive_fortran_flags",
    "native_fortran_flags",
    "GNU_RUNTIME_DLLS",
    "is_gnu_fortran",
    "stage_fortran_runtime_dependencies",
    "stage_standalone_fortran_runtime",
    "stages_shared_runtime",
    "standalone_fortran_link_flags",
    "standalone_runtime_shim_sources",
]
