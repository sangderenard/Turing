"""Master correlation table over one lowered SSA module.

A side product of compilation: neither source, nor IR, nor artifact.  The
lowering keeps many partial records of what a value *is* -- ``parameter_names``,
``storage_formals``, ``program_abi_*`` accounting, sequence descriptors,
record tables, ``value_aliases``, ``callee_input_ids`` -- each written by a
different pass and each correlated through its own key (a name, a signature
position, a source coordinate, an id).  Every fault traced on 2026-09-17 was
two of those records disagreeing about one value while nothing compared them.

This module collects every such claim into one table, one row per
``(function, value_id)``, and reports the places where the claims disagree.
It writes nothing back.  The table is the thing a future single identity
authority would own; until then it is the audit that says where the
authority is missing.

Findings (each is one concrete disagreement, with the two claims):

``multiple-definition``
    one value id defined by more than one instruction (in/out ABI fields
    that are deliberately redefined by every producer are exempt).
``unaccounted-formal``
    a formal no record names: not a parameter, not ABI storage, not leased
    frame storage, not a closure/member formal.
``conflicting-storage-claims``
    a formal claimed as two different ABI fields, or as an ABI field and
    as leased frame storage at once.
``descriptor-member-unknown``
    a sequence descriptor names a member id the function neither receives
    nor defines.
``descriptor-member-shared``
    one storage id plays a role in two descriptors of one function.
``helper-operand-outside-descriptor``
    a keyed lookup helper call reads storage no descriptor of the function
    names (the rewrite bound the call but not the descriptor, or vice
    versa).
``duplicate-storage-across-call``
    a callee formal identified as an ABI field or a sequence member is fed
    freshly leased caller storage although the caller already owns a value
    with that same identity.
``use-not-dominated``
    an instruction reads a value whose definition does not dominate the
    read (Phi operands exempt; they arrive over predecessor edges).
``alias-target-missing``
    a recorded value alias points at a value the function never defines.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Claim:
    """One record's statement about a value's identity."""

    kind: str
    key: str
    source: str


@dataclass
class ValueRow:
    function: str
    value_id: int
    dtype: str | None = None
    is_formal: bool = False
    claims: list[Claim] = field(default_factory=list)
    definitions: list[tuple[str, int, str]] = field(default_factory=list)
    uses: list[tuple[str, int, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Finding:
    kind: str
    function: str
    value_id: int | None
    detail: str


_INOUT_REDEFINED = ("program_abi_mutable", "program_abi_field_written")


class CorrelationTable:
    """Every identity claim the module makes, keyed by (function, value id)."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, int], ValueRow] = {}
        self.sequence_roles: dict[tuple[str, int], list[tuple[int, str]]] = (
            defaultdict(list)
        )

    # ------------------------------------------------------------------ build
    def row(self, function: str, value_id: int, dtype: Any = None) -> ValueRow:
        key = (str(function), int(value_id))
        row = self.rows.get(key)
        if row is None:
            row = ValueRow(str(function), int(value_id), dtype=None)
            self.rows[key] = row
        if dtype is not None and row.dtype is None:
            row.dtype = str(dtype)
        return row

    def claim(self, function: str, value_id: int, kind: str, key: Any,
              source: str) -> None:
        self.row(function, value_id).claims.append(
            Claim(str(kind), str(key), str(source))
        )

    @classmethod
    def build(cls, module: Any) -> "CorrelationTable":
        table = cls()
        for name, function in module.functions.items():
            table._build_function(module, str(name), function)
        return table

    def _build_function(self, module: Any, name: str, function: Any) -> None:
        metadata = dict(getattr(function, "metadata", {}) or {})
        for formal in function.args:
            row = self.row(name, int(formal.id), formal.dtype)
            row.is_formal = True
            accounting = dict(formal.accounting or {})
            abi_field = accounting.get("program_abi_field")
            abi_parameter = accounting.get("program_abi_parameter")
            if abi_field is not None:
                self.claim(
                    name, formal.id, "abi-field",
                    f"{accounting.get('program_abi_record') or abi_parameter}"
                    f".{abi_field}",
                    "accounting.program_abi_field",
                )
            elif abi_parameter is not None:
                self.claim(
                    name, formal.id, "abi-parameter", abi_parameter,
                    "accounting.program_abi_parameter",
                )
            if accounting.get("program_abi_keyed_owner") is not None:
                self.claim(
                    name, formal.id, "keyed-part",
                    f"{accounting['program_abi_keyed_owner']}."
                    f"{accounting.get('program_abi_keyed_part')}",
                    "accounting.program_abi_keyed_owner",
                )
            if accounting.get("linked_call_frame_storage"):
                self.claim(
                    name, formal.id, "frame-storage",
                    f"{accounting['linked_call_frame_storage']}"
                    f"#{accounting.get('propagated_formal_id', '?')}",
                    "accounting.linked_call_frame_storage",
                )
            if accounting.get("compiler_frame_storage"):
                self.claim(
                    name, formal.id, "compiler-frame-storage",
                    accounting["compiler_frame_storage"],
                    "accounting.compiler_frame_storage",
                )
            if accounting.get("projected_row_source_id") is not None:
                self.claim(
                    name, formal.id, "projected-row",
                    f"{accounting['projected_row_source_id']}"
                    f"[{accounting.get('projected_row_column')}]",
                    "accounting.projected_row_source_id",
                )
        for label, value_id in metadata.get("parameter_names", ()) or ():
            self.claim(name, value_id, "parameter", label,
                       "metadata.parameter_names")
        for entry in metadata.get("storage_formals", ()) or ():
            self.claim(name, entry["value_id"], "storage-formal",
                       entry.get("kind", "storage"), "metadata.storage_formals")
        for entry in metadata.get("closure_formals", ()) or ():
            self.claim(name, entry["value_id"], "closure-formal",
                       entry.get("name", "?"), "metadata.closure_formals")
        for entry in metadata.get("parameter_member_formals", ()) or ():
            self.claim(
                name, entry["value_id"], "member-formal",
                f"{entry.get('parameter')}{list(entry.get('path', ()))}",
                "metadata.parameter_member_formals",
            )
        aliases = metadata.get("value_aliases", ()) or ()
        alias_pairs = (
            aliases.items() if isinstance(aliases, Mapping) else aliases
        )
        for pair in alias_pairs:
            try:
                alias, target = pair
            except (TypeError, ValueError):
                continue
            self.claim(name, alias, "alias-of", int(target),
                       "metadata.value_aliases")
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                if instruction.res is not None:
                    row = self.row(name, int(instruction.res.id),
                                   instruction.res.dtype)
                    row.definitions.append((block_name, index, instruction.op))
                for argument in instruction.args:
                    argument_id = getattr(argument, "id", None)
                    if argument_id is None:
                        continue
                    self.row(name, int(argument_id)).uses.append(
                        (block_name, index, instruction.op)
                    )
        sequence_table = (getattr(module, "sequence_tables", {}) or {}).get(name)
        if sequence_table is not None:
            for descriptor in sequence_table.sequences.values():
                roles = [
                    (int(descriptor.sequence_id), "handle"),
                    *(
                        (int(column), f"column{position}")
                        for position, column in enumerate(
                            descriptor.column_value_ids
                        )
                    ),
                    (int(descriptor.length_address_id), "length"),
                    (int(descriptor.capacity_value_id), "capacity"),
                ]
                if descriptor.status_address_id is not None:
                    roles.append((int(descriptor.status_address_id), "status"))
                if descriptor.live_flags_value_id is not None:
                    roles.append(
                        (int(descriptor.live_flags_value_id), "live_flags")
                    )
                for value_id, role in roles:
                    self.claim(
                        name, value_id, "sequence-member",
                        f"seq{int(descriptor.sequence_id)}.{role}",
                        "sequence_table",
                    )
                    self.sequence_roles[(name, value_id)].append(
                        (int(descriptor.sequence_id), role)
                    )
        record_table = (getattr(module, "record_tables", {}) or {}).get(name)
        if record_table is not None:
            for record in record_table.records.values():
                for record_field in record.fields:
                    for value_id in record_field.value_ids:
                        self.claim(
                            name, value_id, "record-field",
                            f"{record.identity}#{record.record_id}"
                            f".{record_field.name}",
                            "record_table",
                        )

    # --------------------------------------------------------------- findings
    def findings(self, module: Any) -> list[Finding]:
        found: list[Finding] = []
        for name, function in module.functions.items():
            found.extend(self._function_findings(module, str(name), function))
        return found

    def _identity_claims(self, row: ValueRow) -> list[Claim]:
        return [
            claim for claim in row.claims
            if claim.kind in {
                "abi-field", "abi-parameter", "keyed-part", "parameter",
                "storage-formal", "closure-formal", "member-formal",
                "frame-storage", "compiler-frame-storage", "projected-row",
            }
        ]

    @staticmethod
    def _is_authored(function: Any) -> bool:
        """Source-level function (not a planned region or generated helper).

        Regions and sequence/tensor helpers receive their formals by the
        planner's own slot convention and never carry ``parameter_names``;
        judging them by the authored-function records would only report
        the convention itself.
        """

        metadata = dict(getattr(function, "metadata", {}) or {})
        return bool(
            metadata.get("parameter_names")
            or metadata.get("authored_parameters")
        ) and not metadata.get("source_region_integral")

    def _function_findings(self, module: Any, name: str,
                           function: Any) -> list[Finding]:
        found: list[Finding] = []
        authored = self._is_authored(function)
        formals = {int(formal.id): formal for formal in function.args}
        rows = {
            value_id: row for (function_name, value_id), row in self.rows.items()
            if function_name == name
        }
        # multiple-definition
        for value_id, row in rows.items():
            if len(row.definitions) <= 1:
                continue
            formal = formals.get(value_id)
            accounting = dict(getattr(formal, "accounting", {}) or {})
            if formal is not None and all(
                accounting.get(key) for key in _INOUT_REDEFINED
            ):
                continue
            found.append(Finding(
                "multiple-definition", name, value_id,
                f"defined {len(row.definitions)} times: "
                f"{row.definitions[:4]!r}",
            ))
        # formal accountability and conflicts (authored functions only)
        for value_id, formal in formals.items():
            row = rows[value_id]
            identity = self._identity_claims(row)
            if not identity and not authored:
                continue
            if not identity:
                found.append(Finding(
                    "unaccounted-formal", name, value_id,
                    f"dtype={formal.dtype} accounting_keys="
                    f"{sorted(dict(formal.accounting or {}))[:6]!r}",
                ))
                continue
            abi_fields = {c.key for c in identity if c.kind == "abi-field"}
            frame = [c for c in identity if c.kind == "frame-storage"]
            accounting = dict(formal.accounting or {})
            if accounting.get("returned_record_storage") is not None:
                # Storage leased for a returned record's field carries that
                # field's identity by design; leased + field is the
                # convention here, not a conflict.
                frame = []
            if len(abi_fields) > 1 or (abi_fields and frame):
                found.append(Finding(
                    "conflicting-storage-claims", name, value_id,
                    f"claims={[(c.kind, c.key) for c in identity]!r}",
                ))
        # sequence descriptor members (authored functions only: a generated
        # helper's descriptor names the caller's cells it is handed)
        for (function_name, value_id), roles in self.sequence_roles.items():
            if function_name != name or not authored:
                continue
            row = rows.get(value_id)
            if row is None or (not row.is_formal and not row.definitions):
                found.append(Finding(
                    "descriptor-member-unknown", name, value_id,
                    f"roles={roles!r}: not a formal, never defined",
                ))
            distinct = {(sequence_id, role.rstrip("0123456789"))
                        for sequence_id, role in roles}
            sequences = {sequence_id for sequence_id, _role in distinct}
            if len(sequences) > 1:
                found.append(Finding(
                    "descriptor-member-shared", name, value_id,
                    f"roles={roles!r}",
                ))
        # keyed helper operands
        described = {
            value_id for (function_name, value_id) in self.sequence_roles
            if function_name == name
        }
        keyed_parts = {
            value_id for value_id, row in rows.items()
            if any(c.kind == "keyed-part" for c in row.claims)
        }
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                attributes = instruction.attributes or {}
                if attributes.get("keyed_lookup_owner") is None:
                    continue
                for position, argument in enumerate(instruction.args[:4]):
                    argument_id = int(argument.id)
                    if argument_id in described or argument_id in keyed_parts:
                        continue
                    found.append(Finding(
                        "helper-operand-outside-descriptor", name, argument_id,
                        f"{block_name}#{index} {attributes.get('callee')} "
                        f"operand {position} (owner "
                        f"{attributes.get('keyed_lookup_owner')!r})",
                    ))
        # duplicate storage across a call
        callee_identity: dict[str, dict[int, str]] = {}
        for callee_name, callee in module.functions.items():
            callee_identity[str(callee_name)] = {
                int(formal.id): key
                for formal in callee.args
                for key in (self._abi_or_member_key(str(callee_name), formal),)
                if key is not None
            }
        caller_by_key: dict[str, list[int]] = defaultdict(list)
        for value_id, formal in formals.items():
            key = self._abi_or_member_key(name, formal)
            if key is not None:
                caller_by_key[key].append(value_id)
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                if instruction.op not in {"Call", "call"}:
                    continue
                attributes = instruction.attributes or {}
                callee_name = str(attributes.get("callee") or "")
                declared = attributes.get("callee_input_ids")
                if callee_name not in callee_identity or declared is None:
                    continue
                for callee_id, argument in zip(declared, instruction.args):
                    key = callee_identity[callee_name].get(int(callee_id))
                    if key is None:
                        continue
                    argument_id = int(argument.id)
                    accounting = dict(getattr(argument, "accounting", {}) or {})
                    if not accounting.get("linked_call_frame_storage"):
                        continue
                    owned = [
                        owner for owner in caller_by_key.get(key, ())
                        if owner != argument_id
                    ]
                    if owned:
                        found.append(Finding(
                            "duplicate-storage-across-call", name, argument_id,
                            f"{block_name}#{index} -> {callee_name} formal "
                            f"{callee_id} ({key}) fed leased storage "
                            f"{argument_id} while caller owns {owned!r}",
                        ))
        # dominance
        found.extend(self._dominance_findings(name, function, formals))
        # alias targets
        for value_id, row in rows.items():
            for claim in row.claims:
                if claim.kind != "alias-of":
                    continue
                target = rows.get(int(claim.key))
                if target is None or (
                    not target.is_formal and not target.definitions
                ):
                    found.append(Finding(
                        "alias-target-missing", name, value_id,
                        f"alias of {claim.key} which is never defined",
                    ))
        return found

    def _abi_or_member_key(self, function: str, formal: Any) -> str | None:
        accounting = dict(getattr(formal, "accounting", {}) or {})
        abi_field = accounting.get("program_abi_field")
        if abi_field is not None:
            # Identity is per record INSTANCE: the parameter that carries
            # the record, or the call whose result record this storage
            # holds.  Two instances of one record type in one function
            # (``metrics`` and ``coerced = coerce_metrics(metrics)``)
            # legitimately own the same field name each.
            instance = accounting.get("returned_record_storage")
            if instance is not None:
                instance = f"{instance}@{accounting.get('callsite_id')}"
            else:
                instance = accounting.get("program_abi_parameter") or accounting.get("program_abi_record")
            return f"{instance}.{abi_field}"
        roles = self.sequence_roles.get((function, int(formal.id)))
        if roles:
            sequence_id, role = roles[0]
            return f"seq{sequence_id}.{role}"
        return None

    @staticmethod
    def _dominators(function: Any) -> dict[str, set[str]]:
        blocks = list(function.blocks)
        if not blocks:
            return {}
        entry = blocks[0]
        predecessors: dict[str, set[str]] = {b: set() for b in blocks}
        for block_name, block in function.blocks.items():
            for successor in block.successors:
                if successor in predecessors:
                    predecessors[successor].add(block_name)
        reachable = {entry}
        stack = [entry]
        while stack:
            current = stack.pop()
            for successor in function.blocks[current].successors:
                if successor in predecessors and successor not in reachable:
                    reachable.add(successor)
                    stack.append(successor)
        dominators = {b: set(reachable) for b in reachable}
        dominators[entry] = {entry}
        changed = True
        while changed:
            changed = False
            for block_name in reachable:
                if block_name == entry:
                    continue
                incoming = [
                    dominators[p] for p in predecessors[block_name]
                    if p in reachable
                ]
                updated = ({block_name} | set.intersection(*incoming)
                           if incoming else {block_name})
                if updated != dominators[block_name]:
                    dominators[block_name] = updated
                    changed = True
        return dominators

    def _dominance_findings(self, name: str, function: Any,
                            formals: Mapping[int, Any]) -> list[Finding]:
        found: list[Finding] = []
        dominators = self._dominators(function)
        if not dominators:
            return found
        definition_sites: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                if instruction.res is not None:
                    definition_sites[int(instruction.res.id)].append(
                        (block_name, index)
                    )
        for block_name, block in function.blocks.items():
            if block_name not in dominators:
                continue
            for index, instruction in enumerate(block.instrs):
                if instruction.op in {"Phi", "phi"}:
                    continue
                for argument in instruction.args:
                    argument_id = getattr(argument, "id", None)
                    if argument_id is None or int(argument_id) in formals:
                        continue
                    sites = definition_sites.get(int(argument_id))
                    if not sites:
                        continue
                    if any(
                        (site_block == block_name and site_index <= index)
                        or (site_block != block_name
                            and site_block in dominators[block_name])
                        for site_block, site_index in sites
                    ):
                        # ``site_index == index``: a region call names its
                        # own result among its operands (the out-pointer
                        # convention), which is a definition, not a read.
                        continue
                    found.append(Finding(
                        "use-not-dominated", name, int(argument_id),
                        f"{block_name}#{index} {instruction.op} reads a "
                        f"value defined at {sites[:3]!r}",
                    ))
        return found


def concordance_report(module: Any, *, limit: int = 12) -> str:
    """Build the table and render its findings as text."""

    table = CorrelationTable.build(module)
    findings = table.findings(module)
    by_kind: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        by_kind[finding.kind].append(finding)
    lines = [
        f"identity concordance: {len(table.rows)} rows across "
        f"{len(module.functions)} functions, {len(findings)} finding(s)"
    ]
    for kind in sorted(by_kind):
        entries = by_kind[kind]
        lines.append(f"  [{kind}] x{len(entries)}")
        for finding in entries[:limit]:
            lines.append(
                f"     {finding.function} value {finding.value_id}: "
                f"{finding.detail}"
            )
        if len(entries) > limit:
            lines.append(f"     ... {len(entries) - limit} more")
    return "\n".join(lines)
