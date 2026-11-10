# Shader source precision and complete contact compilation

The user requested shader arithmetic to default to binary32 through the
precision system, with explicit overrides retained. This implementation is
source materialization, not an emitter-wide cast or an audit exception.

## Changes

- `precision_policy.ShaderPrecisionPolicy` supplies binary32 for unstamped
  floating literals. Existing authored floating widths and explicit per-value
  overrides remain authoritative. Integer and Boolean values are untouched.
- `ssa_python_materializer` can emit ordinary typed `AnnAssign` literals under
  that policy. Native callers without a shader policy retain their old source.
- `symbolic_abstract_tensor_source` accepts the policy. Automatically widened
  sections combined with this new policy fail visibly until an explicit
  section element contract is supplied; no precision-section guard was relaxed.
- Contact shader compilation now uses the complete canonical source entry,
  existing vehicle physical-span extraction ABI, actual contact lane count,
  and seven authored return values. It does not select one numerical region,
  overwrite every SSA dtype/shape, or introduce native double tensor kernels.

The previous native-reference probe produced 743 backend findings. The same
canonical source without native C specialization reduced these to 36 source
literal-width findings. Typed source declarations eliminate those findings:
all seven outputs survive and ordinary WGSL emission reports complete.

## Verification and limits

- Seven precision tests pass, including canonical source ingestion + normal
  backend binary32 success and explicit binary64 refusal (no silent narrowing).
- Actual contact compilation regression plus five initial policy tests:
  six passed, 40 deselected, 103 seconds.
- A broader materializer run was canceled, not claimed as passing. Targeted
  native materializer and packed-contact regression checks are pending.
- Four-lane isolated browser fixture compares the exact generated shader with
  the original SymPy equations, including a noncontact lane. GPU validation
  remains pending. Files are frozen for the browser worker.
- This establishes the shared materialization policy and opts the contact
  shader into it. It does not rewrite every existing shader caller globally.
- No server restart, live artifact publication, merge, or deployment occurred.

Evidence: `build/contact-shader32.json`, `build/contact-shader32.wgsl`,
`build/contact-shader32-qa.html`, `build/shader-precision-full.log`, and
`build/contact-shader32-integration.log`.
