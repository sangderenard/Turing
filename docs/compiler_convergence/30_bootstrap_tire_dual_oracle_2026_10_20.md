# Bootstrap/tire dual-oracle repair trial — 2026-10-20

This continues report 29 in the canonical repository
`C:/dev/Powershell/turing`. No bootstrap, tire, vehicle, or dt-system source
was changed. Every compiler repair was checked first with a focused regression,
then with a fresh full bootstrap receipt before the next tire trial.

## Outcome

The bootstrap topology regression supplied by the user is repaired. The
bootstrap now admits 20/330 units: 18 compiled-unverified, one partial, one
failed, and 310 pending. The tire advances past both its duplicate `dt_min`
public identity and its list-literal producer refusal. Neither target compiles
completely yet.

Three general compiler repairs were made:

1. Imported `typing.Any` and `object` remain open domains. They are no longer
   entered into the exact imported-class set, so an authored `item: Any` does
   not conflict with a `tuple[RowField, ...]` element projection.
2. Receiver-slot injection now restores the identity table's version-zero
   authored parameter as the one public ABI slot. A later lexical version may
   remain a frame argument, but it loses the public parameter name. This moved
   the tire's `dt_min` public identity from value 13 back to value 1.
3. Ordinary list literal materialization reads the reducer-owned
   `SOURCE_AGGREGATE_LAYOUT` member order. It no longer re-identifies copied AST
   objects by Python object identity. This repaired the 15-member
   `error_channels=[...]` list in `balloon_tire_managed_advance`.

## Bootstrap receipts

Command used for each independent full gate:

```powershell
python -u -m tools.bootstrap_compiler_concordance --lower-only --output <fresh-output>
```

After the `Any` repair and after the parameter repair, both receipts reached:

```text
admitted=19, compiled-unverified=18, partial=1, failed=0
unit 024 IdentityBook._admit_raw_write: partial
```

The partial is one unexplained validation predicate (value 11) on the terminal
`ConcordanceRefusal` raise. It has no dtype and no published accounting.

After the list-layout repair the bootstrap advanced one more unit:

```text
admitted=20, compiled-unverified=18, partial=1, failed=1
unit 024 IdentityBook._admit_raw_write: unchanged partial
unit 034 IdentityBook._virtual_edges_into:
  conditional union arm changes its physical ABI:
  value=40, arm='CompilerSSAExtent'
```

Receipt:
`build/bootstrap_dual_oracle_after_list_20261020/compartment.json`

Failure:
`build/bootstrap_dual_oracle_after_list_20261020/growth/product/units/unit_034/failure.json`

Log:
`artifacts/bootstrap_dual_oracle_after_list_20261020.log`

## Tire receipts

Command:

```powershell
python -u tools/build_balloon_tire_native.py --managed-dt --output <fresh-output>
```

The first rerun reproduced the duplicate public identity exactly:

```text
field-slot injection found two public identities for authored parameter
'dt_min': 13 != 1
```

After canonicalizing the authored parameter, the next run reached:

```text
List 243 lacks one exact element producer
```

The failed identity log identifies list 243 as the authored
`error_channels=AbstractTensor.tensor([...])` list in
`balloon_tire_managed_advance`, generated source line 524. Its concorded layout
already contained every member; only the consumer's AST-object re-match was
wrong.

After consuming that layout, the tire advanced to repository SSA settlement:

```text
post shape_transformation_concordance (... value 130 ...):
REVISE without a changed source
incumbent: shape=(8,4,14), dtype=bool
proposed:  shape=(8,4,14), dtype=float64
```

The same SSA value was revised from `bool` to `float64` by
`tensor_shape_enrichment`, but the proposed shape transformation cited only the
unchanged source set. The book correctly refused it.

Log:
`artifacts/tire_dual_oracle_after_list_20261020.log`

## Cross-reference and next repair

The new bootstrap and tire frontiers are now the same broad family: a value or
union arm acquires a second physical ABI after its first authoritative
publication. They are not yet proven to share one writer.

- Bootstrap unit 034 changes the physical ABI of the
  `CompilerSSAExtent` arm for value 40 during conditional-union merging.
- Tire value 130 changes dtype from `bool` to `float64` during repository SSA
  shape enrichment, then attempts a source-less revision of the existing
  transformation row.

The next narrow trial should trace the exact first and second writers for those
two values and compare their identity histories. Do not weaken either refusal
or treat the later dtype/arm as authoritative merely because it was observed
later. In particular, inspect the uncommitted sibling-occurrence propagation
in `tensor_ssa_lowering.py`: the tire receipt may expose a canonical-ID
collision that change made visible.

## Focused verification

```text
3 passed:
  test_any_local_does_not_override_declared_sequence_element_class
  test_field_only_parameter_replaces_later_public_lexical_version
  test_list_literal_uses_concorded_member_order_after_ast_copy
```

`py_compile` passes for the three changed compiler modules. `git diff --check`
reports only the repository's existing LF/CRLF warnings.

An existing open-type predicate test was also sampled and remains red because
its union alternatives omit `builtins.int`; that failure was not changed or
silenced in this trial.
