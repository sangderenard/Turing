2026-10-24 SILENT LANE ROUND 6 (MEASURED, cloud): six silent wrong answers
fixed, each wrong at `main` (doc 36 sections 43-47, corpus c482-c487).
Keyed deletes and a dict's or set's `len` now run at their authored position,
and the length query counts live rows (fuzz p136_20: a deleted key read as
present). A call through `f = inc if c else neg` lowers as a choice of direct
calls. Before, it dropped its parameter and return. A choice used as a value
is refused (`function_selection_value`). A list rebound in one arm and then
appended gets the right capacity and length operand (Lane D: 12 for 42,
error 75). On the class route every method uses the class table's field
slots (`Walker(7).poke(3)` gave 1.0 for 22). A region that reduces an arena
runs after a call that stores into it (fuzz p7_59). `tools/differential_fuzz.py`
gains `FUZZ_EXT=5`, an aimed tier. Gate at cc89aac9 vs main c5af44c9: failure
names are identical in all four parts. C corpus 386/387 (main 380/381); the
one failure, c428_class_keyword_construct, is WRONG at main c5af44c9 too.
FUZZ_EXT=5 seeds 139, 141 and 143 (60 each) and FUZZ_EXT=3 seed 144 (60): 0
WRONG. Seed 145 at this head: p145_9 WRONG, not yet checked against main.
