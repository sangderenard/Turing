"""Classify authored and compiler-owned formals at the canonical native root."""

from __future__ import annotations

from collections import Counter, defaultdict
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.vehicle_python_compilation import (
    balloon_tire_python_compilation_inputs,
    lower_balloon_tire_python_ssa,
)


def main() -> None:
    lowered = lower_balloon_tire_python_ssa(batch_size=8)
    inputs = balloon_tire_python_compilation_inputs(8)
    function = lowered.module.functions[lowered.root_name]
    metadata = dict(function.metadata or {})
    authored_ids = {
        int(value_id) for _name, value_id in metadata.get("parameter_names", ())
    }
    output_ids = {
        int(value_id) for _name, value_id in metadata.get("named_outputs", ())
    }
    definitions = defaultdict(list)
    uses = Counter()
    use_sites = defaultdict(list)
    for block_name, block in function.blocks.items():
        for instruction in block.instrs:
            for position, value in enumerate(instruction.args):
                uses[int(value.id)] += 1
                use_sites[int(value.id)].append((
                    str(block_name),
                    str(instruction.op),
                    str(instruction.attributes.get("callee") or ""),
                    int(position),
                ))
            if instruction.res is not None:
                definitions[int(instruction.res.id)].append(
                    f"{block_name}:{instruction.op}"
                )

    groups = Counter()
    examples = defaultdict(list)
    for argument in function.args:
        value_id = int(argument.id)
        accounting = dict(argument.accounting or {})
        ownership = tuple(sorted(
            key for key, value in accounting.items()
            if value not in {None, "", False}
        ))
        category = (
            "authored" if value_id in authored_ids
            else "output" if value_id in output_ids
            else "defined-root" if value_id in definitions
            else "anonymous-live" if uses[value_id]
            else "anonymous-dead"
        )
        key = (
            category,
            str(argument.dtype or ""),
            bool(tuple(argument.shape or ())),
            ownership,
        )
        groups[key] += 1
        if len(examples[key]) < 6:
            examples[key].append((
                value_id,
                tuple(argument.shape or ()),
                uses[value_id],
                tuple(definitions[value_id]),
                tuple(use_sites[value_id]),
            ))

    print(
        f"root={lowered.root_name} args={len(function.args)} "
        f"authored_names={len(authored_ids)} feeds={len(inputs.feeds)}"
    )
    print(f"metadata_keys={tuple(sorted(metadata))!r}")
    for key, count in groups.most_common():
        print(f"count={count} key={key!r} examples={examples[key]!r}")

    anonymous_call_sites = Counter(
        (site[1], site[2])
        for argument in function.args
        if int(argument.id) not in authored_ids | output_ids
        for site in use_sites[int(argument.id)]
    )
    print(f"anonymous_call_sites={anonymous_call_sites.most_common()!r}")
    remaining_live = {
        int(argument.id): dict(argument.accounting or {})
        for argument in function.args
        if int(argument.id) not in authored_ids | output_ids
        and uses[int(argument.id)]
    }
    print(f"remaining_live_accounting={remaining_live!r}")
    print(f"value_aliases={metadata.get('value_aliases')!r}")
    print(f"value_names={metadata.get('value_names')!r}")
    call_outputs = []
    for block_name, block in function.blocks.items():
        for instruction in block.instrs:
            if instruction.op in {"Call", "call"}:
                call_outputs.append((
                    str(block_name),
                    str(instruction.attributes.get("callee") or ""),
                    tuple(instruction.attributes.get("output_ids") or ()),
                ))
    print(f"call_outputs={call_outputs!r}")
    inspected = 0
    for argument in function.args:
        value_id = int(argument.id)
        if (
            value_id in authored_ids | output_ids
            or definitions[value_id]
            or (argument.accounting or {}).get("linked_call_frame_storage")
        ):
            continue
        for block in function.blocks.values():
            for instruction in block.instrs:
                positions = [
                    position for position, actual in enumerate(instruction.args)
                    if int(actual.id) == value_id
                ]
                callee_name = str(instruction.attributes.get("callee") or "")
                callee = lowered.module.functions.get(callee_name)
                if not positions or callee is None:
                    continue
                for position in positions:
                    if position >= len(callee.args):
                        continue
                    formal = callee.args[position]
                    formal_uses = []
                    for callee_block_name, callee_block in callee.blocks.items():
                        for candidate in callee_block.instrs:
                            if any(
                                int(item.id) == int(formal.id)
                                for item in candidate.args
                            ):
                                formal_uses.append((
                                    str(callee_block_name),
                                    str(candidate.op),
                                    dict(candidate.attributes or {}),
                                ))
                    print(
                        f"anonymous_detail actual=%{value_id} "
                        f"shape={tuple(argument.shape or ())!r} "
                        f"callee={callee_name} position={position} "
                        f"formal=%{formal.id} accounting="
                        f"{dict(formal.accounting or {})!r} "
                        f"uses={formal_uses[:8]!r}"
                    )
                    inspected += 1
                    if inspected >= 16:
                        return


if __name__ == "__main__":
    main()
