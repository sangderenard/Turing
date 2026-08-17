"""A staged decision tree for "why doesn't this translate/compile/run right".

The pipeline is Python -> ProcessGraph -> repository SSA -> backend
(Fortran/LLVM) -> runtime buffers. A defect at any stage shows up at the
END as a wrong number or a compiler error, and the expensive mistake is
guessing which stage owns it. This walks the stages in order and reports
the FIRST one that is provably inconsistent, so the search starts where
the evidence is instead of where the symptom is.

Each check answers one question and says what it does NOT prove, because
a check that silently degrades (returning a default when it cannot
observe something) is worse than no check -- that exact failure mode
produced a day of false diagnosis here.

    python tools/diagnose_translation.py                # all stages
    python tools/diagnose_translation.py --ids 116,47   # focus on values

DECISION TREE
=============

STAGE 1  Lowering shortfalls
    Does the SSA even claim to be complete?
    FAIL -> read the shortfall; nothing downstream is meaningful.

STAGE 2  SSA well-formedness
    2a Duplicate producers: is one id produced by two DIFFERENT SSAValue
       objects? (Same object twice is in-place reuse and is FINE.)
    2b Phi arity: does every phi's incoming-block count match its
       operand count?
    2c Dangling operands: is any operand neither a formal, nor produced,
       nor a known constant?
    FAIL -> the defect is at/above precompile_to_ssa; the backend is
    faithfully rendering a broken graph.

STAGE 3  In-place aliasing safety  (in-place is good; unsafe is not)
    For every region call: which out-params share a pointer with a feed?
    3a An out-param that aliases a feed is IN-PLACE -- expected, fast.
    3b TWO DIFFERENT outputs sharing ONE pointer is a real defect: the
       second write destroys the first.
    3c An out-param aliasing a feed that the callee reads AFTER writing
       that output is order-dependent and must be verified.
    FAIL -> the fusion decision is wrong, not the arithmetic.

STAGE 4  ABI observability
    Which ids are actually in artifact.buffer_order?
    An id that is NOT there cannot be read; it is not zero. Any
    conclusion drawn from reading one is void.

STAGE 5  Influence (dye) -- what actually reaches a suspect value
    `influence_field.field_from_ssa` IS wired to lowered SSA (verified on
    this program: ~8k transports over the advance function). It answers
    "what reaches this, from where, and how much survived", which is the
    question static reading of IR text cannot answer cheaply.
    Read it as:
      - dominant BAKED  -> mostly compile-time-constant influence; per the
        module's own semantics such a node is constant-foldable, so a
        runtime-varying value that reads dominantly baked is suspicious.
      - dominant DYNAMIC -> genuinely runtime-fed.
      - RECURRENT       -> arrived through a loop-carried edge; state.
    Two values that SHOULD be independent but show identical weights are
    sharing an influence path -- though note the def-use view treats all
    outputs of one region call as one node, so equal weights for two
    outputs of the SAME call is expected, not evidence.

STAGE 6  Runtime
    Run and compare against independently computed truth.

WHAT THIS DELIBERATELY DOES NOT DO
    It does not mutate the program to observe it. Adding a source-level
    probe (`state.x = accumulator + 0.0`) shifts value ids and can rebind
    the very thing being measured -- that has already produced a wrong
    conclusion here. Prefer reading an EXISTING public id.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_id_spec(spec: str) -> tuple[int, ...]:
    """Parse an id selection: ``116``, ``116,47``, ``100-120``, mixed.

    Accepts commas and/or whitespace as separators and ``a-b`` (inclusive)
    or ``a..b`` as ranges, so a caller can sweep a numbering neighbourhood
    without typing every id. Order is preserved and duplicates collapse,
    because these ids are printed as report sections and a repeated
    section reads as a different value rather than the same one twice.
    """
    found: list[int] = []
    for token in str(spec).replace(",", " ").split():
        token = token.strip()
        if not token:
            continue
        body = token.replace("..", "-")
        if "-" in body.lstrip("-"):
            low_text, _, high_text = body.lstrip("-").partition("-")
            try:
                low, high = int(low_text), int(high_text)
            except ValueError:
                raise SystemExit(f"could not read id range {token!r}")
            if low > high:
                low, high = high, low
            if high - low > 4096:
                raise SystemExit(
                    f"id range {token!r} spans {high - low} ids; narrow it"
                )
            found.extend(range(low, high + 1))
            continue
        try:
            found.append(int(body))
        except ValueError:
            raise SystemExit(f"could not read id {token!r}")
    return tuple(dict.fromkeys(found))


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"         {msg}")


def stage_1_shortfalls(artifact: Any) -> bool:
    print("STAGE 1  lowering / emission shortfalls")
    shortfalls = tuple(getattr(artifact, "shortfalls", ()) or ())
    if not getattr(artifact, "complete", True) or shortfalls:
        _bad(f"{len(shortfalls)} shortfall(s); the SSA is not complete")
        for item in shortfalls[:10]:
            _info(str(item)[:160])
        return False
    _ok("artifact reports complete, no shortfalls")
    return True


def stage_2_ssa_wellformed(fn: Any) -> bool:
    print("STAGE 2  SSA well-formedness")
    healthy = True

    producers: dict[int, list[Any]] = {}
    for block in fn.blocks.values():
        for instr in block.instrs:
            if instr.res is not None:
                producers.setdefault(int(instr.res.id), []).append(instr.res)
    formals = {int(a.id): a for a in fn.args}

    # 2a -- duplicate producers by OBJECT, not by id.
    real_collisions = []
    inplace_reuse = []
    for vid, objs in producers.items():
        distinct = {id(o) for o in objs}
        formal = formals.get(vid)
        if formal is not None:
            if id(formal) in {id(o) for o in objs}:
                inplace_reuse.append(vid)
            else:
                real_collisions.append(vid)
        if len(distinct) > 1:
            real_collisions.append(vid)
    if real_collisions:
        healthy = False
        _bad(
            f"{len(sorted(set(real_collisions)))} id(s) produced by DIFFERENT "
            f"objects (true collision): {sorted(set(real_collisions))[:12]}"
        )
        _info("two distinct values sharing one id -- whichever renders last wins")
    else:
        _ok("no duplicate-producer collisions")
    if inplace_reuse:
        _ok(
            f"{len(inplace_reuse)} id(s) reuse their formal's cell in place "
            f"(same object) -- expected and fast: {sorted(inplace_reuse)[:12]}"
        )

    # 2b -- phi arity.
    bad_phis = []
    for block_name, block in fn.blocks.items():
        for instr in block.instrs:
            if str(instr.op) in {"Phi", "phi"}:
                blocks = tuple(instr.attributes.get("incoming_blocks") or ())
                if blocks and len(blocks) != len(instr.args):
                    bad_phis.append((block_name, int(instr.res.id)))
    if bad_phis:
        healthy = False
        _bad(f"phi arity mismatch: {bad_phis[:10]}")
    else:
        _ok("every phi's incoming blocks match its operands")

    # 2c -- dangling operands.
    defined = set(producers) | set(formals)
    dangling = set()
    for block in fn.blocks.values():
        for instr in block.instrs:
            for a in instr.args:
                if int(a.id) not in defined:
                    dangling.add(int(a.id))
    if dangling:
        healthy = False
        _bad(f"operands with no definition: {sorted(dangling)[:12]}")
    else:
        _ok("every operand is a formal or is produced")
    return healthy


def stage_3_inplace_safety(fn: Any, module_functions: dict) -> bool:
    print("STAGE 3  in-place aliasing safety")
    healthy = True
    calls = 0
    for block in fn.blocks.values():
        for instr in block.instrs:
            if str(instr.op) not in {"Call", "call"}:
                continue
            output_ids = tuple(map(int, instr.attributes.get("output_ids", ())))
            if not output_ids:
                continue
            calls += 1
            feed_ids = [int(a.id) for a in instr.args]
            callee_name = str(instr.attributes.get("callee") or "")

            # 3b -- two outputs on one id.
            if len(set(output_ids)) != len(output_ids):
                healthy = False
                dupes = [v for v in set(output_ids) if output_ids.count(v) > 1]
                _bad(
                    f"{callee_name.split('__')[-1]}: outputs repeat id(s) "
                    f"{dupes} -- the later write destroys the earlier"
                )

            # 3a -- in-place fusion (output id also a feed).
            fused = sorted(set(output_ids) & set(feed_ids))
            if fused:
                _ok(
                    f"{callee_name.split('__')[-1]}: {len(fused)} in-place "
                    f"out/feed fusion(s) {fused[:8]}"
                )
                # 3c -- order sensitivity inside the callee.
                callee = module_functions.get(callee_name)
                if callee is not None:
                    order: list[tuple[str, int]] = []
                    for cb in callee.blocks.values():
                        for ci in cb.instrs:
                            if ci.res is not None:
                                order.append(("w", int(ci.res.id)))
                            for ca in ci.args:
                                order.append(("r", int(ca.id)))
                    formal_ids = [int(a.id) for a in callee.args]
                    for position, vid in enumerate(output_ids):
                        if vid not in fused:
                            continue
                        try:
                            feed_pos = feed_ids.index(vid)
                        except ValueError:
                            continue
                        if feed_pos >= len(formal_ids):
                            continue
                        formal = formal_ids[feed_pos]
                        writes = [
                            i for i, (k, v) in enumerate(order)
                            if k == "w" and v == formal
                        ]
                        reads = [
                            i for i, (k, v) in enumerate(order)
                            if k == "r" and v == formal
                        ]
                        if writes and reads and max(reads) > min(writes):
                            healthy = False
                            _bad(
                                f"{callee_name.split('__')[-1]}: formal "
                                f"{formal} is READ after being WRITTEN while "
                                "fused in place -- order dependent"
                            )
    if calls and healthy:
        _ok(f"{calls} region call(s) checked; in-place fusions look safe")
    return healthy


def stage_4_observability(adv: Any, ids: tuple[int, ...]) -> None:
    print("STAGE 4  ABI observability")
    order = {int(v) for v in (adv.artifact.buffer_order or ())}
    _info(f"{len(order)} ids are in the public buffer ABI")
    for vid in ids:
        if vid in order:
            _ok(f"id {vid} IS observable")
        else:
            _bad(
                f"id {vid} is NOT observable (internal alloca). Reading it "
                "yields nothing; it is NOT zero."
            )


def stage_5_influence(fn: Any, ids: tuple[int, ...]) -> None:
    """Dye-trace what reaches each suspect value."""
    print("STAGE 5  influence (dye)")
    if not ids:
        _info("no --ids given; skipped")
        return
    try:
        from src.compiler.influence_field import (
            InfluenceContract, field_from_ssa,
        )
    except Exception as error:  # pragma: no cover - diagnostic aid only
        _info(f"influence field unavailable: {error}")
        return

    name = fn.name

    class _Module:
        functions = {name: fn}

    field = field_from_ssa(
        _Module(), InfluenceContract(enabled=True), functions=[name],
    )
    transports = field.propagate()
    _info(f"{transports} transports over {name.split('__')[-1]}")

    readings: dict[int, dict[str, float]] = {}

    def locate(vid: int):
        for label, block in fn.blocks.items():
            for instr in block.instrs:
                if instr.res is not None and int(instr.res.id) == vid:
                    return (name, label, instr.res.id), label
        return None, None

    for vid in ids:
        key, label = locate(vid)
        if key is None:
            _info(f"id {vid}: no producing instruction (formal or absent)")
            continue
        reading = field.reading(key)
        parts = {
            category: (
                round(getattr(entry, "weight", 0.0), 6),
                round(getattr(entry, "hue", 0.0), 6),
            )
            for category, entry in (reading.categories or {}).items()
        }
        _info(f"id {vid} in {label}: {parts}")
        readings[vid] = {
            category: round(getattr(entry, "weight", 0.0), 6)
            for category, entry in (reading.categories or {}).items()
        }

    # The signal here is comparative, so do the comparison rather than
    # leaving it to the reader: identical weights between two values that
    # should be independent means they share an influence path.
    identical: list[tuple[int, int]] = []
    seen = list(readings.items())
    for index, (left, left_weights) in enumerate(seen):
        for right, right_weights in seen[index + 1:]:
            if left_weights and left_weights == right_weights:
                identical.append((left, right))
    if identical:
        for left, right in identical:
            _bad(
                f"ids {left} and {right} have IDENTICAL influence weights -- "
                "they share an influence path. Expected if they are two "
                "outputs of the SAME region call (the def-use view treats "
                "that call as one node); a real finding otherwise."
            )
    elif len(readings) > 1:
        _ok("no two selected ids share an identical influence profile")

    _info(
        "Read these COMPARATIVELY, never absolutely. Baked dominance is the "
        "norm in this program (dt/dx/gravity/limits feed nearly everything), "
        "so 'dominantly baked' is NOT by itself a defect -- verified: the "
        "known-CORRECT max_wave_speed reads dominantly baked too. What "
        "carries signal is a value whose profile differs from a comparable "
        "value that is known to work, or two supposedly-independent values "
        "whose weights match exactly."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Staged decision tree for translation defects. See "
            "tools/TRANSLATION_DEBUGGING.md for the full questionnaire."
        ),
        epilog=(
            "ids accept commas, spaces, and inclusive ranges: "
            "--ids 141 | --ids 141,47 | --ids 100-120 | --ids 100-120,141"
        ),
    )
    parser.add_argument(
        "--ids", default="",
        help="value ids to inspect; commas/spaces/ranges (a-b or a..b)",
    )
    parser.add_argument(
        "--stages", default="",
        help="run only these stages, e.g. --stages 2,3 (default: all)",
    )
    args = parser.parse_args()
    ids = parse_id_spec(args.ids)
    wanted = set(parse_id_spec(args.stages)) if args.stages else set()

    def run(stage: int) -> bool:
        return not wanted or stage in wanted

    from src.compiler.symbolic_fluid_native_runtime import (
        compile_native_symbolic_fluid_advance,
    )

    build = ROOT / "build" / "diagnose-tmp"
    build.mkdir(parents=True, exist_ok=True)
    print(f"compiling into {build} ...\n")
    adv = compile_native_symbolic_fluid_advance(build)
    fn = adv.function
    module_functions = adv._module_functions or {}

    healthy = True
    if run(1):
        healthy &= stage_1_shortfalls(adv.artifact)
        print()
    if run(2):
        healthy &= stage_2_ssa_wellformed(fn)
        print()
    if run(3):
        healthy &= stage_3_inplace_safety(fn, module_functions)
        print()
    if run(4) and ids:
        stage_4_observability(adv, ids)
        print()
    if run(5):
        stage_5_influence(fn, ids)
        print()
    print("VERDICT:", "no stage proved inconsistent" if healthy
          else "a stage above is provably inconsistent -- start there")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
