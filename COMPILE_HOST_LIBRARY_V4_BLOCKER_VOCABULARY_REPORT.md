# `compile()` v4 complete blocker and minimum-vocabulary report

Date: 2026-08-13
Authoritative input: `build/compile-host-library-report.json`
Related checkpoint: `HANDOFF_2026-08-13_COMPILE_HOST_LIBRARY_V4.md`

## Scope and accounting

This report accounts for **every effective blocker occurrence** emitted by the
completed `compile()` / `Py_CompileString` host-library extraction. It does not
rerun extraction and does not infer blockers from old logs.

The source ledger contains 6,518 effective occurrences. Repeated occurrences
arise when several cached export roots reach the same physical machine-code
site, when one site is represented in both a retained function and a control
funclet, or when many call sites require the same missing provider capability.
This report therefore presents three numbers where relevant:

- **occurrences**: exact duplicate-preserving count in the report;
- **physical sites**: deduplicated PE path + machine address + diagnostic;
- **capability family**: the minimum common compiler feature that can discharge
  those sites without special-casing one function.

The complete reconciliation is:

| Blocker front | Effective occurrences | Deduplicated evidence | Minimum capability families |
|---|---:|---:|---:|
| External module pursuit | 3,910 | 491 physical import sites; 154 normalized external identities | 2 |
| Indirect calls and jumps | 2,277 | 1,305 physical control sites | 2 core + 2 PE mapping cases |
| Instruction/control decode | 331 | 77 physical instruction sites; 55 normalized diagnostics | 11 form families |
| **Total** | **6,518** | | |

Nothing is omitted from the total, and no blocker is declared harmless merely
because it is control-only, repeated, or non-numerical.

## The actual minimum expansion set

The 6,518 occurrences do **not** require 6,518 bespoke handlers. The minimum
coherent expansion is the following shared vocabulary and provenance work:

1. **Windows API-set contract resolution.** Correctly map API-set DLL contract
   names to the local host PE images using the parsed API-set namespace.
2. **General PE export parsing for the local `ntdll.dll`.** Fix the one shared
   `BinaryFormatError` mechanism rather than stubbing 104 exports.
3. **Unified symbolic indirect-target value sets.** Propagate finite code-target
   sets through registers, based-memory loads, Phi/select operations, spills,
   reloads, and table indexing.
4. **PE interior-address ownership and initialized-slot vocabulary.** Recognize
   executable interior pointers, jump/callback tables, relocations, and loader-
   initialized slots even when they are not ordinary imports.
5. **Instruction-boundary and prefix/form dispatch correctness.** Many failures
   spell already-supported instructions. Repair shared decoding and ownership
   before adding duplicate instruction tokens.
6. **Declarative integer form matrix.** Complete operand-width, immediate-width,
   REX, legacy-prefix, and ModRM-extension combinations for existing arithmetic,
   bitwise, test, and conditional semantics.
7. **Condition-code matrix completion.** Add the missing parity rel32 branch form
   using the existing parity flag and conditional-branch semantics.
8. **String-memory iteration.** Express `MOVSB` as explicit pointer/count/memory
   state, reusable by repeated and non-repeated string forms.
9. **Machine control/status state.** Represent MXCSR load/store and ordering
   fences explicitly; retain prefetch as a declared hint operation rather than
   an unknown instruction.
10. **Width-parameterized vector vocabulary.** Use one vector move/XOR/shuffle/
    compare family parameterized by register bank, width, lane width, encoding,
    and upper-lane policy. Include MMX/x87 aliasing and VEX state explicitly.
11. **Complete register alias vocabulary.** Add AH/CH/DH/BH as real overlapping
    register slices with the architectural REX restriction.
12. **Atomic/non-atomic form distinction.** Reuse compare-exchange value/flag
    semantics while distinguishing locked ordering from the unlocked encoding.

Items 3 and 4 should be one machine-dataflow subsystem, and items 5 through 12
should be generated from the bidirectional instruction schema rather than hand-
maintained encode and decode forks. Semantically, this is a much smaller change
than the raw count suggests.

## Front A — external module pursuit

### Accounting

All 3,910 occurrences fall into exactly two failure mechanisms:

| Mechanism | Occurrences | Normalized identities | Required repair |
|---|---:|---:|---|
| API-set contract reported `module-unavailable` | 2,794 | 50 | API-set namespace/host resolution |
| `ntdll.dll` reported `module-unreadable:BinaryFormatError` | 1,116 | 104 | PE parser/export-reader correction |
| **Total** | **3,910** | **154** | |

There are no other effective external failure prefixes in the completed report:
no export-missing, forwarder-cycle, recursion-limit, or module-count-limit case.

### Every unresolved module family

| Module/contract | Occurrences | Unique symbols | Explanation |
|---|---:|---:|---|
| `ntdll.dll` | 1,116 | 104 | The file is found but rejected by the shared PE parser. One parser repair should expose all exports. |
| `api-ms-win-core-errorhandling-l1-1-0.dll` | 992 | 4 | API-set host mapping absent for error state/filter APIs. |
| `api-ms-win-core-processthreads-l1-1-0.dll` | 530 | 6 | API-set host mapping absent for process/thread/TLS APIs. |
| `api-ms-win-core-libraryloader-l1-1-0.dll` | 512 | 6 | API-set host mapping absent for loader/symbol APIs. |
| `api-ms-win-core-synch-l1-1-0.dll` | 475 | 7 | API-set host mapping absent for locks, semaphore, event, and wait APIs. |
| `api-ms-win-core-rtlsupport-l1-1-0.dll` | 91 | 4 | API-set host mapping absent for unwind/context APIs. |
| `api-ms-win-core-file-l1-1-0.dll` | 83 | 8 | API-set host mapping absent for file operations. |
| `api-ms-win-core-processthreads-l1-1-1.dll` | 45 | 1 | API-set host mapping absent for processor-feature query. |
| `api-ms-win-core-debug-l1-1-0.dll` | 30 | 2 | API-set host mapping absent for debugger operations. |
| `api-ms-win-core-processenvironment-l1-1-0.dll` | 18 | 4 | API-set host mapping absent for environment/standard-handle operations. |
| `api-ms-win-core-localization-l1-2-0.dll` | 16 | 6 | API-set host mapping absent for code-page/localization operations. |
| `api-ms-win-core-memory-l1-1-0.dll` | 2 | 2 | API-set host mapping absent for virtual allocation/free. |

The minimum correct action is not to create SSA stubs for `SetLastError`,
`GetProcAddress`, `TlsGetValue`, or the other names. Resolve each API contract to
its actual host PE and recursively ingest that code. Similarly, do not create
104 handcrafted `ntdll` functions: diagnose the exact structural field rejected
by `parse_pe_image`, correct the general PE reader, and let ordinary recursive
pursuit admit the exports.

Unblocking these dependencies may reveal additional machine-code blockers
inside newly readable units. Therefore 6,518 is the complete current ledger,
not a promise that provider repair alone produces the final zero ledger.

## Front B — indirect calls and jumps

### Complete deduplicated detail ledger

| Underlying shortfall | Calls | Jumps | Total occurrences | Minimum vocabulary |
|---|---:|---:|---:|---|
| Target depends on based-memory machine state | 1,908 | 1 | 1,909 | Symbolic memory-to-code target sets |
| Target depends on register machine state | 249 | 83 | 332 | Register/Phi/select target-set propagation |
| RIP-relative slot contains an executable pointer without import/code owner | 30 | 3 | 33 | PE interior code-owner and table-entry classification |
| RIP-relative slot is not file-backed | 3 | 0 | 3 | Loader/runtime-initialized slot state |
| **Total** | **2,190** | **87** | **2,277** | |

The based-memory group is entirely rooted in `python311.dll!Py_CompileString`.
The register group is mostly the same root (253 occurrences) with a smaller CRT
tail. This is not 1,909 different call mechanisms. It is one missing ability to
carry code identities through resident machine memory and recover the finite
target set at the consuming control operation.

The exact known-pointer slots are:

| Slot | Occurrences | Current contents |
|---|---:|---|
| `0x1804a12c0` | 22 | `0x1800693bc` |
| `0x18051e7e0` | 4 | `0x1800f572c` |
| `0x18051e7d8` | 3 | `0x1800f559c` |
| `0x1804a12a8` | 2 | `0x1800539ec` |
| `0x1804a12b0` | 1 | `0x1800769cc` |
| `0x1804a12b8` | 1 | `0x180065b74` |

These already contain concrete addresses. They need an image-wide interval/
entry ownership index and table provenance, not dynamic-call emulation. The
three non-file-backed slots (`0x180567040`, `0x180567058`, `0x1805670f0`) need
explicit initialized-storage provenance. They must not be guessed as zero or
dropped.

The unified target analysis should produce one of:

- one exact internal function reference;
- a finite table of exact references and explicit dispatch;
- a retained dynamic machine-state control operation whose target value and
  memory provenance are fully represented.

It must never replace an unresolved target with a Python call, numeric
projection, opaque handler, or arbitrary first target.

## Front C — instruction and control decode

The 216 `decode` plus 115 `machine-control-decode` occurrences are 331 report
occurrences but only 77 physical PE instruction sites. They reduce to eleven
form families:

| Family | Occurrences | Physical sites | Minimum expansion |
|---|---:|---:|---|
| Existing scalar/control spelling or boundary failure | 156 | 27 | Fix boundary/prefix/form dispatch; reuse existing semantics |
| Group-1 immediate width/prefix matrix | 54 | 20 | Generate `/1`, `/3`, `/4`, `/6`, `/7` forms by width/prefix |
| SSE/MMX move/shuffle/compare forms | 36 | 8 | Generic vector form matrix plus MMX alias state |
| Parity conditional jump rel32 | 28 | 1 | Add `JP/JPE rel32` form using PF |
| MXCSR and fence forms | 18 | 8 | Explicit MXCSR memory state and ordering token |
| Prefetch hint forms | 10 | 2 | Declared non-value hint semantics |
| `MOVSB` | 8 | 1 | String pointer/memory state transition |
| VEX vector forms | 7 | 3 | Width/encoding parameterization; no AVX2 forwarding |
| Legacy high-byte alias | 6 | 2 | AH/CH/DH/BH overlapping register slices |
| `TEST` / `BT` / `BTR` forms | 5 | 4 | Complete existing bit/test form matrix |
| Unlocked `CMPXCHG` | 3 | 1 | Reuse compare-exchange semantics without LOCK ordering |
| **Total** | **331** | **77** | |

### All normalized decode diagnostics

The following table is the complete normalized signature ledger. Addresses are
removed only to combine repeated occurrences; counts still sum to 331.

| Occurrences | Diagnostic signature |
|---:|---|
| 63 | bytes `ba 20 00 66` |
| 30 | bytes `75 12 48 c1` |
| 29 | bytes `0f 29 41 f0` |
| 28 | bytes `0f 8a 4c ff` |
| 27 | opcode `83` ModRM `/1` |
| 10 | bytes `73 17 66 41` |
| 10 | opcode `81` ModRM `/1` |
| 8 | bytes `a4 5e 5f c3` |
| 8 | bytes `0f 4c c2 48` |
| 8 | bytes `0f 18 84 11` |
| 7 | opcode `83` ModRM `/4` |
| 6 | bytes `0f ae e8 48` |
| 6 | opcode `81` ModRM `/4` |
| 6 | bytes `08 44 d1 3d` |
| 5 | bytes `d0 ee 40 22` |
| 5 | bytes `0f ae 5c 24` |
| 4 | bytes `34 01 c0 e0` |
| 4 | bytes `c5 f1 ef c9` |
| 3 | bytes `0f ae e8 8b` |
| 3 | bytes `0f b1 af 96` |
| 3 | opcode `f7` ModRM `/0` |
| 3 each | bytes `0f 47 08 66`, `0f 47 1a 66`, `0f 47 12 66`, `0f 47 0a 66`, `0f 47 f9 41` |
| 3 | bytes `c5 fe 6f 02` |
| 2 each | bytes `0f ae 54 24`, `0f ae 1c 24`, `0f 0d 0e 49`, `0f 74 c1 66`, `08 44 1a 18`, `08 4c f8 3d`, `08 5c 18 18`, `0f 70 c8 00`, `09 48 14 48`, `09 7b 14 48` |
| 2 | opcode `83` ModRM `/7` |
| 1 each | bytes `00 53 02 6b`, `09 43 14 8b`, `0f 9d c0 e9`, `8d 04 0a 25`, `0f 47 d1 41`, `0f 74 02 66`, `33 d0 48 c1`, `30 44 d1 3d`, `2d b8 00 00`, `1b c0 66 f7`, `0f 74 c8 66`, `0f 96 c0 44`, `0f 28 d8 66` |
| 1 each | opcode `83` ModRM `/3`, opcode `80` ModRM `/6`, opcode `0f ba` ModRM `/6`, opcode `0f ba` ModRM `/4` |
| 1 | legacy high-byte source needs a distinct register token |

### Why 156 occurrences must not cause 156 new tokens

The largest suspicious spellings are instructions already represented in the
repository vocabulary: `MOV r32, imm32` (`ba`), `JNE rel8` (`75`), `JAE rel8`
(`73`), `MOVAPS` (`0f 29`/`0f 28`), `CMOVL` (`0f 4c`), and `CMOVA` (`0f 47`).
For example, the 63 `ba 20 00 66` occurrences collapse onto a shared physical
site reached repeatedly through three KernelBase-rooted units and their control
funclets. Adding another MOV semantic would conceal the actual fault.

The decoder must first answer, for every such site:

1. Is the address an authoritative instruction boundary or an interior/data
   address admitted as code?
2. Were legacy/REX/VEX prefixes consumed but not represented in the diagnostic?
3. Did form selection reject a legal operand-width or register combination?
4. Did a parent decode failure cause resynchronization at the wrong byte?
5. Does the same site decode correctly through the ordinary function path but
   fail through machine-control funclet decoding?

Only after those checks should a genuinely absent form be added.

### Form expansions with mostly existing semantics

- Group-1 immediate failures are OR (`/1`), SBB (`/3`), AND (`/4`), XOR
  (`80 /6`), and CMP (`/7`) width/prefix combinations. The semantic operations
  already exist. Generate the missing encoding rows and flag/write-width rules.
- `f7 /0`, `0f ba /4`, and `0f ba /6` are TEST/BT/BTR forms. Their bit and flag
  semantics already exist elsewhere in the vocabulary.
- `0f b1` is the unlocked compare-exchange form. The locked 32/64-bit forms
  already exist; ordering must be a form attribute, not duplicated value logic.
- `0f 8a` is parity-set rel32. A parity rel8 token already exists, so this is a
  displacement form completion, not a new control abstraction.

### Genuinely broader state vocabulary

- `a4` requires string source/destination pointer updates and one byte memory
  transfer. The design should naturally extend to REP and other string widths.
- `0f ae /2`, `0f ae /3`, and `0f ae e8` require LDMXCSR, STMXCSR, and LFENCE.
  MXCSR is program state, and the fence is an ordering event; neither is a
  numerical expression.
- `0f 18` and `0f 0d` are prefetch-family hints. Retain them as explicit machine
  hint operations with declared architectural effects rather than deleting the
  instructions.
- The VEX spellings include vector XOR and unaligned vector move forms. They
  must lower through declared vector state and legal target selection. They
  must **not** be forwarded to this host as AVX2.
- The `0f 70`/`0f 74` cases include legacy packed shuffle/compare forms. Model
  lane width and register bank explicitly, including MMX's architectural alias
  relationship with x87 state where applicable.
- `d0 ee` and the explicit high-byte diagnostic prove the register table lacks
  a complete AH/CH/DH/BH slice model. Add those overlapping aliases and forbid
  them in forms carrying a REX prefix, as the ISA requires.

## Minimum implementation strategy

### One schema, not parallel tables

Expand the existing bidirectional instruction schema so each form declares:

- opcode bytes and legacy/REX/VEX constraints;
- ModRM extension and operand addressing;
- register bank, operand width, lane width, and upper-lane policy;
- semantic family;
- read/write register slices, flags, memory, MXCSR, and ordering effects;
- inverse encoding constraints.

Decoder selection, encoder selection, machine graph operators, machine SSA,
repository SSA lowering, and equivalence tests should consume that same record.
This is the minimum route to near-unity between the machine graph, machine SSA,
and repository SSA without contaminating repository SSA with a second PE-only
IR meaning.

### Repair sequence

1. Preserve the v4 report and all 107 referenced cache pickles.
2. Add cache repair at function/control-region granularity before changing the
   hashed vocabulary files. Do not rebuild 32,114 functions for one form.
3. Fix instruction boundary/prefix dispatch and re-evaluate the 27 suspicious
   existing-form sites. This may eliminate much of the 156-occurrence family
   without vocabulary growth.
4. Generate the missing integer/condition form matrix and add exact roundtrip,
   flags, and state tests for each normalized signature family.
5. Add the genuinely broader string, MXCSR/fence, hint, vector-width, and high-
   byte register state vocabulary.
6. Fix API-set resolution and `ntdll` PE parsing, then recursively admit only
   newly available units.
7. Run unified symbolic target-set analysis over the repaired machine graph and
   classify all remaining indirect sites.
8. Re-materialize the aggregate from reused and repaired cache records. Launch
   a complete recursive build only after the cache audit shows that existing
   functions will be reused.

## Completion criterion

The current ledger is discharged only when all three fronts reach zero without
filtering:

- all statically named external dependencies are recursively represented or
  carry an explicit in-program machine-state implementation;
- every indirect control operation has exact finite targets or a fully modeled
  dynamic target state;
- every instruction site has a bidirectional token, semantic effect, and legal
  repository SSA representation;
- `machine_state_complete` and `repository_ssa_complete` are both true.

Deduplication in this report is for engineering economy. The next authoritative
compile report must still retain every occurrence so that one fixed spelling at
one address cannot hide an unfixed occurrence elsewhere.
