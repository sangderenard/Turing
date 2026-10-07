"""The host-facing layout of a C module is a set of book rows, and the header
a host compiles against is printed from them.

A small dt-style program -- a record ``cell`` with two columns (``x``,
``v``) and a ``telemetry`` span, one bare scalar ``dt``, the update in a
callee so the root carries frame-linked formals -- is lowered with a real
``ExtractionContract`` through ``lower_ast_source_to_ssa`` and emitted with
``emit_ssa_module_to_c``.  Read only from the book the compile attached to
the module:

1. one ``program_abi_field_slot`` row per ProgramABI slot, each DERIVED (an
   inbound edge from the resident's ``ssa_value`` cell and from the entry's
   BUFFER_ORDER cell; a bare parameter also from its ``function_parameter``
   cell), none tagged unsourced;
2. the entry's API_CONTRACT row, DERIVED from the wrapper's FUNCTION_HEADER
   and FORMAL units, BUFFER_ORDER and every slot row, whose table names each
   buffer from those rows;
3. ``compile`` writes ``<entry>_layout.h`` beside ``<entry>.c`` and posts
   (SOURCE_FILE, "layout_header") DERIVED from the API_CONTRACT cell;
   ``compile_standalone`` posts the same part with the ``standalone`` suffix;
4. a tiny C program compiled with the C lane's toolchain against the emitted
   module and the header: for every named column it calls
   ``<entry>_bind_column`` on the ``buffers`` table it owns, gets back the
   very pointer it passed, prints the table the header carries (compared
   here with the rows), fills the columns from a file, runs the entry, and
   writes the columns back; they equal the Python lane (the same source run
   as Python on numpy arrays) bit for bit.

    python -u tools/compiler_probes/probe_host_layout_rows.py
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (  # noqa: E402
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (  # noqa: E402
    EMISSION_ARTIFACT,
    PROGRAM_ABI_FIELD_SLOT,
    ArtifactPart,
    Backend,
    LayoutKind,
)
from src.compiler.emission_concordance import (  # noqa: E402
    api_contract,
    program_abi_field_slots,
    value_cell,
)
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.identity_concordance import UNSOURCED_PAGE  # noqa: E402
from src.compiler.ssa_c_backend import emit_ssa_module_to_c  # noqa: E402
from src.compiler.work_contract import active_contract  # noqa: E402

BATCH = 4
SOURCE = '''
class Cell:
    pass

def advance(cell, dt):
    cell.x[...] = cell.x + dt * cell.v
    cell.v[...] = cell.v - dt * cell.x

def root(cell, dt):
    advance(cell, dt)
    cell.telemetry[0] = dt
'''

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if condition else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


def lower():
    span = lambda n: {  # noqa: E731
        "storage": "span", "dtype": "float64", "rank": 1, "shape": [n],
        "mutable": True,
    }
    policy = ExtractionContract(
        REPO / "extraction_contracts" / "program_extraction.yaml"
    ).with_program_abi({
        "records": {"Cell": {"identity": "probe.Cell", "fields": {
            "x": span(BATCH), "v": span(BATCH), "telemetry": span(2),
        }}},
        "bindings": [{"function": "*", "parameter": "cell", "record": "Cell"}],
        "values": [{
            "function": "root", "parameter": "dt", "storage": "scalar",
            "dtype": "float64", "rank": 0, "python_type": "builtins.float",
        }],
    }).with_execution_file(
        REPO / "extraction_contracts" / "vehicle_full_native_execution.yaml"
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCE, "root", name="hostrows", extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        runtime_closure_only=True,
    )
    return module, exports[0]


def python_lane(inputs: dict) -> dict:
    """The same source run as Python on numpy arrays."""

    namespace: dict = {}
    exec(compile(SOURCE, "<probe>", "exec"), namespace)  # noqa: S102
    cell = type("Cell", (), {})()
    cell.x, cell.v = inputs["cell.x"].copy(), inputs["cell.v"].copy()
    cell.telemetry = inputs["cell.telemetry"].copy()
    namespace["root"](cell, float(inputs["dt"][0]))
    return {"cell.x": cell.x, "cell.v": cell.v, "cell.telemetry": cell.telemetry,
            "dt": inputs["dt"]}


def host_program(entry: str) -> str:
    upper = entry.upper()
    return f'''#include <stdio.h>
#include <stdlib.h>
#include "{entry}_layout.h"

int main(int argc, char **argv) {{
    if (argc < 3) return 2;
    void *buffers[{upper}_BUFFER_COUNT];
    for (int i = 0; i < {entry}_layout_count; ++i) {{
        const turing_layout_entry *e = &{entry}_layout[i];
        buffers[i] = calloc(e->capacity, e->itemsize);   /* the host's own */
        printf("ENTRY %d %s %s %s dtype=%d kind=%d written=%d count=%llu capacity=%llu itemsize=%llu\\n",
               e->buffer_index, e->name ? e->name : "-",
               e->parameter ? e->parameter : "-", e->field ? e->field : "-",
               e->dtype, e->kind, e->written, (unsigned long long)e->count,
               (unsigned long long)e->capacity, (unsigned long long)e->itemsize);
    }}
    printf("COUNT %d BUFFER_COUNT %d BATCH %d\\n", {entry}_layout_count,
           {upper}_BUFFER_COUNT, {upper}_BATCH);
    FILE *in = fopen(argv[1], "rb");
    if (!in) return 3;
    for (int i = 0; i < {entry}_layout_count; ++i) {{
        const turing_layout_entry *e = &{entry}_layout[i];
        if (!e->name) continue;
        turing_column col;
        int32_t status = {entry}_bind_column(buffers, e->name, &col);
        printf("BIND %s status=%d same_pointer=%d bytes=%llu count=%llu itemsize=%llu dtype=%d index=%d\\n",
               e->name, (int)status, col.data == buffers[i],
               (unsigned long long)col.bytes, (unsigned long long)col.count,
               (unsigned long long)col.itemsize, col.dtype, col.buffer_index);
        if (fread(col.data, col.itemsize, col.count, in) != col.count) return 4;
    }}
    fclose(in);
    turing_column missing;
    printf("MISSING status=%d\\n", (int){entry}_bind_column(buffers, "no.such", &missing));
    long long extents[1] = {{0}};
    {entry}(buffers, extents);
    FILE *out = fopen(argv[2], "wb");
    if (!out) return 5;
    for (int i = 0; i < {entry}_layout_count; ++i) {{
        const turing_layout_entry *e = &{entry}_layout[i];
        if (!e->name) continue;
        turing_column col;
        {entry}_bind_column(buffers, e->name, &col);
        fwrite(col.data, 1, col.bytes, out);
    }}
    fclose(out);
    return 0;
}}
'''


def main() -> int:
    print("lowering ...", flush=True)
    module, entry = lower()
    artifact = emit_ssa_module_to_c(module, entry, batch=BATCH)
    check("C emission is complete", artifact.complete, str(artifact.shortfalls[:2]))
    book = module.metadata["identity_book"]
    root = module.functions[entry]
    backend = Backend.C_MODULE
    slot_page = book.pages[PROGRAM_ABI_FIELD_SLOT.name]
    unsourced = book.pages[UNSOURCED_PAGE.name]

    print("\n1. program_abi_field_slot rows")
    rows = {
        (row[2], row[3], row[4]): row for row in slot_page.scope_rows(artifact.name)
        if row[1] == backend
    }
    wanted = {("cell", "x"), ("cell", "v"), ("cell", "telemetry"), ("dt", None)}
    check("one row per ProgramABI slot", {k[:2] for k in rows} == wanted,
          f"{sorted(map(str, rows))}")
    buffer_order_cell = book.latest_ref(
        EMISSION_ARTIFACT, (artifact.name, backend, ArtifactPart.BUFFER_ORDER))
    for key, row in sorted(rows.items(), key=lambda item: str(item[0])):
        slot = slot_page.latest(row)
        ref = book.latest_ref(PROGRAM_ABI_FIELD_SLOT, row)
        sources = [source for source, _stage in book.edges_into(ref)]
        names = {source.page.name for source in sources}
        resident_cell = value_cell(book, root, slot.value_id)
        print(f"     {key[0]}.{key[1]} {row[:2]} -> {slot}")
        print(f"       edges_into: {[(s.page.name, s.row) for s in sources]}")
        check(f"{key[0]}.{key[1]}: DERIVED (edges_into non-empty)", bool(sources))
        check(f"{key[0]}.{key[1]}: from the resident's ssa_value cell",
              resident_cell in sources)
        check(f"{key[0]}.{key[1]}: from the BUFFER_ORDER cell", buffer_order_cell in sources)
        check(f"{key[0]}.{key[1]}: not tagged unsourced",
              not any(k[0] == slot_page.name and k[1] == row for k in unsourced.rows()))
        check(f"{key[0]}.{key[1]}: resident is a public buffer",
              slot.buffer_index == artifact.buffer_order.index(slot.value_id))
        if key[1] is None:
            check("dt: from its function_parameter cell", "function_parameter" in names)
    check("the reader returns the rows' facts",
          {k: v for k, v in program_abi_field_slots(module, artifact.name, backend).items()}
          == {k: slot_page.latest(r) for k, r in rows.items()})

    print("\n2. API_CONTRACT")
    contract_row = (artifact.name, backend, ArtifactPart.API_CONTRACT)
    contract_ref = book.latest_ref(EMISSION_ARTIFACT, contract_row)
    check("API_CONTRACT row exists", contract_ref is not None)
    sources = [source for source, _stage in book.edges_into(contract_ref)]
    print(f"     edges_into: {[(s.page.name, s.row[-1] if s.page.name == 'emission_unit' else s.row) for s in sources]}")
    check("API_CONTRACT DERIVED (edges_into non-empty)", bool(sources))
    check("from the entry's FUNCTION_HEADER unit",
          any(s.page.name == "emission_unit" for s in sources))
    check("from BUFFER_ORDER", buffer_order_cell in sources)
    check("from every slot row",
          all(book.latest_ref(PROGRAM_ABI_FIELD_SLOT, r) in sources for r in rows.values()))
    check("the artifact carries the API_CONTRACT cell", artifact.emission.api_contract == contract_ref)
    entry_name, batch, buffers = api_contract(book, artifact.name, backend)
    check("contract entry and declared batch", (entry_name, batch) == (artifact.name, BATCH))
    check("one LayoutBuffer per public buffer, in buffer order",
          [b.buffer_index for b in buffers] == list(range(len(artifact.buffer_order))))
    for buffer in buffers:
        resident = [r for r, s in ((r, slot_page.latest(r)) for r in rows.values())
                    if s.buffer_index == buffer.buffer_index]
        named = resident and (resident[0][2], resident[0][3]) == (buffer.parameter, buffer.field)
        check(f"buffer {buffer.buffer_index} named by its row",
              bool(named) if buffer.kind is not LayoutKind.OTHER else not resident,
              f"{buffer.parameter}.{buffer.field} {buffer.kind.name} count={buffer.count} "
              f"capacity={buffer.capacity}")

    print("\n3. <entry>_layout.h")
    build = pathlib.Path(tempfile.mkdtemp(prefix="hostrows_"))
    artifact.compile(build, optimization="O2")
    header = build / f"{artifact.name}_layout.h"
    check("<entry>_layout.h written beside <entry>.c", header.is_file() and (build / f"{artifact.name}.c").is_file())
    header_ref = book.latest_ref(
        EMISSION_ARTIFACT, (artifact.name, backend, (ArtifactPart.SOURCE_FILE, "layout_header")))
    check("(SOURCE_FILE, 'layout_header') row exists", header_ref is not None)
    header_sources = [source for source, _stage in book.edges_into(header_ref)]
    check("derived from the API_CONTRACT cell alone", header_sources == [contract_ref],
          str([(s.page.name, s.row[-1]) for s in header_sources]))
    fact = book.pages[EMISSION_ARTIFACT.name].latest(
        (artifact.name, backend, (ArtifactPart.SOURCE_FILE, "layout_header")))
    check("the row's hash and length are the file's",
          (fact.sha256, fact.byte_length)
          == (hashlib.sha256(header.read_bytes()).hexdigest(), len(header.read_bytes())))
    standalone = pathlib.Path(tempfile.mkdtemp(prefix="hostrows_sa_"))
    artifact.compile_standalone(standalone, {}, optimization="O0")
    standalone_ref = book.latest_ref(EMISSION_ARTIFACT, (
        artifact.name, backend, (ArtifactPart.SOURCE_FILE, "layout_header", "standalone")))
    check("compile_standalone posts the header part too",
          standalone_ref is not None and (standalone / f"{artifact.name}_layout.h").is_file())

    print("\n4. a C host against the artifact")
    host = build / "host.c"
    host.write_text(host_program(artifact.name), encoding="utf-8")
    exe = build / "host.exe"
    command = [
        sys.executable, "-m", "ziglang", "cc", "-O2", "-std=c11",
        *map(str, active_contract().compiler_flags), "-I", str(build),
        "-o", str(exe), str(host), str(build / f"{artifact.name}.c"),
    ]
    built = subprocess.run(command, capture_output=True, text=True)
    check("host.c + <entry>.c compile and link", built.returncode == 0 and exe.is_file(),
          (built.stderr or built.stdout)[-600:])
    if built.returncode != 0:
        return 1
    rng = np.random.default_rng(7)
    inputs = {
        "cell.x": rng.standard_normal(BATCH), "cell.v": rng.standard_normal(BATCH),
        "cell.telemetry": rng.standard_normal(2), "dt": np.array([0.1]),
    }
    named = [b for b in buffers if b.kind is not LayoutKind.OTHER]
    key_of = lambda b: b.parameter if b.field is None else f"{b.parameter}.{b.field}"  # noqa: E731
    input_file, output_file = build / "in.bin", build / "out.bin"
    input_file.write_bytes(b"".join(inputs[key_of(b)].tobytes() for b in named))
    ran = subprocess.run([str(exe), str(input_file), str(output_file)],
                         capture_output=True, text=True)
    print(ran.stdout.rstrip())
    check("host ran", ran.returncode == 0, ran.stderr[-300:])
    lines = ran.stdout.splitlines()
    table = [line.split() for line in lines if line.startswith("ENTRY ")]
    check("the header's table has one entry per buffer", len(table) == len(buffers))
    for printed, buffer in zip(table, buffers):
        expected = (
            str(buffer.buffer_index), key_of(buffer), str(buffer.parameter),
            "-" if buffer.field is None else buffer.field,
            f"dtype={['bool', 'int32', 'int64', 'float64'].index(buffer.dtype)}",
            f"kind={list(LayoutKind).index(buffer.kind)}", f"written={int(buffer.written)}",
            f"count={buffer.count}", f"capacity={buffer.capacity}", f"itemsize={buffer.itemsize}",
        )
        check(f"table[{buffer.buffer_index}] equals the row {key_of(buffer)}",
              tuple(printed[1:]) == expected, f"{printed[1:]} vs {expected}")
    check("COUNT / BUFFER_COUNT / BATCH constants",
          f"COUNT {len(buffers)} BUFFER_COUNT {len(buffers)} BATCH {BATCH}" in lines)
    binds = [line for line in lines if line.startswith("BIND ")]
    check("every named column bound to the pointer the host passed",
          len(binds) == len(named) and all("status=0 same_pointer=1" in line for line in binds))
    check("an unknown column is refused", "MISSING status=1" in lines)
    got = output_file.read_bytes()
    expected = python_lane(inputs)
    offset = 0
    for buffer in named:
        size = buffer.count * buffer.itemsize
        column = np.frombuffer(got[offset:offset + size], dtype=np.float64)
        offset += size
        reference = expected[key_of(buffer)]
        check(f"{key_of(buffer)} equals the Python lane bit for bit",
              np.array_equal(column.view(np.uint64), reference.view(np.uint64)),
              f"{column.tolist()} vs {reference.tolist()}")
    check("x and v moved", not np.array_equal(expected["cell.x"], inputs["cell.x"]))

    print(f"\nfailures: {len(failures)}")
    for label in failures:
        print(f"  - {label}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
