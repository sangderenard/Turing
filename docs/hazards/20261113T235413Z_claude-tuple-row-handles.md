2026-11-13 A host-evaluated call a region captures is a region frontier (MEASURED, origin/main 9fdbe2aa + this change)

Context: owner direction (2026-11-13) to wash difficult Python with exact source identities before lowering. The
structure-of-arrays wash for YoungMan's local tuple-row lists (one plain list per column) produces index reads such
as `col_0[(i + 1) % len(col_0)]` for every column; those failed. Map of the wash work: shared notes
youngman-tuple-row-handles.md.

Repro (scratch modix3/c8, LLVM):
    def root(n):
        xs = []; ys = []
        for k in range(n): xs.append(k * 3); ys.append(k + 1)
        total = 0; zs = []
        for i in range(len(xs)):
            zs.append(xs[(i + 1) % len(xs)])
            zs.append(ys[(i + 1) % len(xs)])
        for z in zs: total = total * 10 + z
        return total
Refused: final concordance, 3 findings, "sequence_index_in_range.1#8 Lt reads a value defined at
sequence_index_in_range.2#9".  One modular read (c7), a plain read beside a modular read (c9), two reads in one
statement, and two plain reads appended all PASS.

Observed (root's terminal hierarchy plan): region_1 held both index computations, `Mod (19, 29) -> 30` (line 10) and
`Mod (22, 33) -> 34` (line 11), capturing 0, 29, 33.  29 and 33 are two host-evaluated `len(xs)` Call nodes, one
per statement; 33 is evaluated after line 10's append, so region_1 was emitted after it -- inside line 10's
in-range block -- while line 10's bounds check had already read 30.

Rule now holding (glsl `_control_partition_keys`): a node of type Call that is not a region member and feeds one is
a frontier exactly as a user call (callee_ref/method_ref) already is -- its descendants get their own key, so a
region cannot hold work from both sides of it.  Independent work keeps one key.

Measured after (branch claude/tuple-row-handles):
- c8 PASS LLVM, audit 0 findings; regions split into region_3 (`Mod -> 30`) and region_4 (`Mod -> 34`).
- c7, c9 PASS; scratch soa/c1 (four per-column lists, modular reads, zip of four) PASS; soa/c3 PASS.
- Regression gates, branch vs a baseline worktree at 9fdbe2aa without this change (Temp\wtr9):
  40 self-field programs (LLVM + Fortran): branch == baseline, every line.  74 for+if (LLVM) and 11 clues: every
  line that differs from the older summaries (loop74_fix8, allclues_fix2) is `LOWER-FAIL IndexError: list index out
  of range`; all 14 such programs (c124, c126, c209, c210, c236, c271, c275, c441, c68, c69, c794, c796,
  c1_two_geometries, c1_two_literal_formals) fail identically at 9fdbe2aa.  So this change alters no gate result.
- REGRESSION ON MAIN, not from this change (measured at 9fdbe2aa, also seen at fde37ed3): against this morning's
  summaries, self-field c424, c749, c750, c754, c759, c794, c796, c882, c941 and the 14 programs above now crash
  with IndexError in `identity_concordance.concord_compiler_frame_formals` (`functions[str(row[2])].blocks[str(row[3])]`);
  c426, c552, c643, c669_b refuse "formal_actual_occurrence_concordance disagreement"; c757 "signature pruning lacks
  its complete exact prior occurrence census"; c857/c858/c859/c875/c876 "record field input lacks exact binding
  causes".  Present at b510e3e2? No: all gates were identical there.  Introduced between b510e3e2 and fde37ed3
  (the AbstractNN / source-proven compiler integration merge 1057a3cc is the likely range; not bisected).
Note: scratch loopif_check.py hard-codes the shared `turing` checkout; branch measurements use loopif_check_root.py
with CHECK_ROOT.
