# Shell system ports

System ports extend `turing-shell-io-requirements` without replacing the
compiled program API. Ordinary numerical parameters remain ordinary ABI
parameters. A system port groups the parameters that together represent a
shell-owned resource and gives that boundary one stable semantic identity.

## File parameters

An input file binds at least two fields:

```text
name = subject-binary
kind = file
direction = input
entry point = load_subject
data -> subject_bytes
length -> subject_length
```

The attachment pass resolves source names to final ABI-local parameter names.
The HTML shell removes those parameters from the numerical feed editor and
renders one byte-exact file picker. `window.TuringSystemPorts.publishFile()`
stores the `Uint8Array` and invokes the program-installed handler. The native
Python launcher generates `--file-<port>` options and retains a contiguous
`uint8` array plus exact length. The generated C/Fortran shell emits the same
option, bounds the file against the port's `maximum_bytes`, loads it into the
bound `uint8_t *`, and supplies its length parameter.

File request/completion mailboxes remain available for files opened later by
the running program. File parameters are initial named resources; broker
requests are dynamic operations. They use the same span convention but are not
the same lifecycle.

## Virtual filesystem

The file broker operates inside a persistent virtual namespace rather than on
ambient host paths. A manifest declares an absolute virtual current directory
and explicit mounts. Mounts are `memory`, `bundle`, `indexed_db`, `opfs`, or
`host_directory`, and each is read-only or read-write. Web shells accept
memory, bundle, IndexedDB, and origin-private-file-system mounts. IndexedDB and
OPFS sources are relative namespace names, not host paths. They hydrate into
the shell's synchronous virtual-file map before program execution; immediate
writes update that map and `flushVirtualFilesystem()` is the asynchronous
durability barrier. A real Chrome probe writes, evicts, and rehydrates exact
bytes through both backends. Browsers still cannot materialize
`host_directory`. Native shells may accept one, but its source and virtual
destination must both appear in the contract; no undeclared host path is
visible to the program.

The shared namespace uses canonical UTF-8 POSIX paths. A Windows guest sees a
stable projection (`/c/work` becomes `C:\work`), unrelated to the native
shell's own working directory. The broker declares open/create, read/write,
close/stat, list, mkdir/remove/rename, getcwd/chdir, and flush. Mutations are
ordered effects. In the binary machine they update an immutable
`VirtualFileSystemState`, so reversing a system-call completion restores file
contents, metadata, and current directory along with registers and memory.
Directory searches retain immutable VFS handles, match lists, and cursors, so
`FindFirstFileW`/`FindFirstFileExW` followed by `FindNextFileW` can be resumed
or reversed without consulting a changed host directory.

Ordinary Win32 file handles are equally virtual. `CreateFileW` enforces the
declared mount, creation disposition, access/share contract, bounded security
attribute shape, and supported synchronous flags before returning a VFS handle.
`ReadFile`, `WriteFile`, both file-pointer forms, truncation, size, flush,
close, attributes, timestamps, and volume queries produce only guest-memory,
device, or immutable VFS effects. Handle mode and cursor plus file creation,
access, modification times and attribute masks survive JSONL and segmented
tape checkpoints. Read-only mounts and attributes reject write admission;
overlapped, delete-on-close, unbuffered, encrypted, or otherwise unmodelled
shapes remain explicit fail-closed frontiers. `CloseHandle` accepts ordinary
file handles but not enumeration handles, which retain their distinct
`FindClose` lifecycle.

## Virtual registry

The Windows registry is another capability-owned immutable namespace, never a
view of the host registry. It begins with only the predefined root keys unless
the shell deliberately supplies additional virtual content. Keys retain typed
binary values, case-insensitive identities, display names, last-write metadata,
and access-bearing handles. Create/open, set/query, enumeration, value/key
deletion, and close are explicit `VirtualRegistryEffect` values applied by the
same external-completion journal as memory and VFS effects.

Win32 buffer-size and error contracts remain observable: queries return the
required byte count, short buffers report `ERROR_MORE_DATA`, enumeration
reports `ERROR_NO_MORE_ITEMS`, nonempty key deletion is refused, and handles
enforce their granted query/set/create/enumerate rights. `MAXIMUM_ALLOWED` is
resolved against the virtual namespace rather than forwarded to Windows.
Unsupported security descriptors and malformed shapes fail closed. Registry
deletion of a still-open key remains an explicit deferred-deletion frontier
rather than silently invalidating another guest handle. Registry
state is process-shared across guest threads, private to possible-world forks,
serialized in JSONL and segmented checkpoints, and catalogued as its own
effect domain during tape-to-SSA lifting.

## Virtual memory

The Windows address-space catalog is reversible state too. Initial image,
stack, TEB/PEB, and system-arena mappings are recorded as regions. Bounded
`VirtualAlloc` reserve-and-commit creates new page-backed private regions;
whole-region `VirtualFree(MEM_RELEASE)`, `VirtualQuery`, and current-process
`ReadProcessMemory` operate only on that catalog. Dynamic executable pages are
eligible for decoding and participate in page-version/cache invalidation, so
rewind restores both bytes and executable-code coherence. Reserve-only,
decommit, unsupported protections, partial release, overlapping requests, and
remote-process reads fail closed rather than borrowing Windows behavior.

## System devices

`kind = device` names a byte-oriented shell boundary such as
`console.input`, `console.output`, or a future declared sensor. Direction is
enforced independently of its adapter. The HTML shell exposes
`readDevice()`, `writeDevice()`, `publishDevice()`, subscriptions, and an
explicit adapter registration point. If a machine runtime is present, input
ports bind to its `injectDeviceBytes()` contract (with a console-input
compatibility spelling). The binary runtime journals that injection as a tape
edge; Win32 `ReadFile`/`ReadConsole` and output calls see constructed virtual
handles and reversible device buffers, never browser or Windows device handles.
This is the common seam where a shell may later install a permissioned real
adapter without changing guest semantics.

## External references

External-reference ports always state their domain:

- `bundle`: another published Turing bundle/component.
- `host_system`: a native shell facility or library, capability gated.
- `guest_binary`: a module mapped into the emulated subject environment.

HTML shells accept only `bundle`. They resolve an explicitly registered bundle
identity and export through `TuringSystemPorts`; they do not fetch arbitrary
URLs, load host libraries, or reinterpret guest DLLs. Cross-bundle discovery,
version negotiation, signatures, and dependency acquisition remain a larger
API project, but the domain and port identity are stable now.

Native shell profiles may advertise `host_references`. Guest-binary resolution
belongs to the binary-machine loader rather than the application shell. The
physical `turing-shell-io-abi` supplies external request/completion rings with
`resolve`, `call`, and `release` operations. Argument, result, and effect
payloads are offset/length spans in program-owned memory, allowing asynchronous
JavaScript/Wasm and native hosts to share one record layout.

For a loaded PE, imports become `guest_binary` references with stable numeric
IDs and synthetic target addresses. Calls pause as typed external requests.
Shell completions return a scalar result plus an effect list; effects may write
only existing guest mappings. Both request and completion are retained in the
reversible machine graph. `CapabilityGatedExternalPort` dispatches by exact,
case-insensitive `(library, symbol)` identity and has no catch-all host-call
path. The deterministic Windows bootstrap policy supplies caller-selected
virtual time, process/thread IDs, counters, and the main module handle.

Static imports and dynamically resolved exports share one durable external-link
catalog in the system-tape header. Guest pointers therefore retain the same
`(domain, library, symbol, reference ID, synthetic target)` identity after a
shell restart. External-completion graph nodes depend on the request node they
consume. Runtime dispatch nodes retain their executable PE targets and are
revalidated and decoded before a resumed core may use them. Conflicting target
or reference IDs fail tape loading instead of silently rebinding an address.

## Virtual child processes

Process creation is another exact capability, not permission to call host
`exec`. When a `VirtualProgramRegistry` is supplied, `CreateProcessW` may
resolve only a registered virtual path (including deterministic `PATH` and
`PATHEXT` search) to a declared bundle and executor. Unknown paths remain
pending. `InitializeProcThreadAttributeList`, `UpdateProcThreadAttribute`, and
`GetStartupInfoW` preserve the Windows x64 setup contract in reversible guest
memory; startup handles are the virtual console handles, never inherited host
handles.

Each accepted launch creates a canonical
`turing.virtual-child-process-tape.v1` containing the requested/resolved path,
arguments, environment, exact stdin/stdout/stderr, bundle/executor identities,
exit code, and execution units. Its SHA-256 digest is recorded in the parent
completion as `child-tape:sha256:<digest>`. The child tape can therefore be
replayed or inspected independently while its causal deployment remains a
typed node in the parent machine tape.

Win32 thread-local error state is part of this contract. Failed file searches
write `ERROR_FILE_NOT_FOUND`, exhausted enumerations write
`ERROR_NO_MORE_FILES`, and invalid closes write `ERROR_INVALID_HANDLE`; stale
errors must not redirect guest control flow. This distinction was required for
real `cmd.exe` to continue from its failed `.COM` probe to the registered
`.EXE` candidate.

Synchronization objects are capabilities too, never host kernel handles.
`CreateMutexExW`, `CreateSemaphoreExW`, `OpenSemaphoreW`, release operations,
and both single-object wait forms operate on integer handle records in the
shared reversible system state. Object names are case-insensitive and persist
on the exact tape. A zero-time unavailable wait returns `WAIT_TIMEOUT`; a
nonzero unavailable wait remains admitted but pending, allowing the shell's
core scheduler to run a producer. Security descriptors, unsupported creation
flags, and invalid count ranges fail closed instead of being silently ignored.

## Virtual pipes and descriptor aliases

Anonymous pipes are reversible byte devices, not host kernel objects.
`CreatePipe` allocates typed read/write handles and a bounded `pipe.<id>`
buffer. Empty reads remain pending while any writer is open, the final writer
close produces deterministic EOF, writes with no readers report a broken pipe,
and a full bounded buffer blocks until the guest consumes bytes.
`DuplicateHandle` preserves endpoint reference counts. `ReadFile`, `WriteFile`,
`FlushFileBuffers`, `GetFileType`, and `CloseHandle` all consult the same
capability-owned endpoint state.

MSVCRT `_pipe`, `_get_osfhandle`, `_open_osfhandle`, `_dup`, `_dup2`, and
`_close` build descriptor aliases over those same handles; there is no second
CRT-only transport. `STARTF_USESTDHANDLES` on a registered virtual child
process validates inherited endpoint direction and routes the child tape's
exact stdin/stdout/stderr through the selected pipe buffers. Pipe handles,
descriptor bindings, endpoint counts, buffered bytes, and their generations
survive JSONL or segmented replay. Tape-to-SSA lifting marks their changes with
a distinct `pipe` effect domain in addition to the underlying system/device
resources.

The x64 CRT `_setjmp` capability also records its corresponding shadow-call
stack in reversible system state. `longjmp` restores the jump buffer's
nonvolatile GPRs, RSP/RIP, MXCSR/FPCSR, XMM6-XMM15, and that shadow stack as one
nonlocal tape edge. This lets command-error recovery cross a pipe setup path
without losing control-flow provenance.

## Live display context transport

The native machine-program host and HTML interior display share a small
same-origin transport without sharing mutable execution objects. `GET
/snapshot?after=N` returns either one complete newer `TMSNAP01` generation or
HTTP 204. `POST /input` adds one bounded byte message to the terminal-input
queue. The host keeps only the latest display flip; exact history continues to
stream into the segmented machine tape and can be projected separately into
the provenance-preserving trace-SSA segment store.

Only the controller thread drains that queue, journals `console.input`,
services admitted external calls, advances the runner, and publishes snapshots.
HTTP threads therefore cannot interleave mutations with instruction execution.
The reference server accepts numeric loopback addresses only, imposes input and
queue bounds, disables caching, and provides no filesystem or generic host-call
route. `TuringMachineSnapshots.connect()` and `sendTerminalInput()` form the
browser side of this context contract; the shader remains presentation, never
the executor or clock authority.

## Safety rules

- A required port must have a matching requested shell capability.
- Each port and field name is unique.
- Bound fields must resolve to one parameter on the named entry point.
- Web emission fails if an external reference is not in the `bundle` domain.
- File data is byte-exact and never passes through the numerical value parser.
- Missing required native file paths and oversized files fail before execution.
- A virtual filesystem requires the `files` capability and a root mount.
- HTML planning and emission reject `host_directory` mounts.
- IndexedDB and OPFS mounts require explicit relative namespace sources.
- Device ports require the `system_devices` capability and enforce direction.
