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

## Safety rules

- A required port must have a matching requested shell capability.
- Each port and field name is unique.
- Bound fields must resolve to one parameter on the named entry point.
- Web emission fails if an external reference is not in the `bundle` domain.
- File data is byte-exact and never passes through the numerical value parser.
- Missing required native file paths and oversized files fail before execution.
