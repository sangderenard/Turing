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
and explicit mounts. Mounts are `memory`, `bundle`, or `host_directory`, and
each is read-only or read-write. Web shells accept only memory and bundle
mounts. Native shells may accept a host-directory mount, but its source and
virtual destination must both appear in the contract; no undeclared host path
is visible to the program.

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
