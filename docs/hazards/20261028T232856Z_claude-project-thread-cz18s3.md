2026-10-28 TABLE AND LIST READS IN SOURCE ORDER (MEASURED, cloud): three
silent wrong answers at main 75b910dc, rebuilt from #902. A dict's or set's
`len` is a lexical length query that counts live rows (c483: `t = len(d);
del d[2]` gave t one too large). A loop over `__iterable_extent_N__` reads
arena N, so a set built from a list waits for the list's appends (c488, fuzz
p145_9: 7 for 29). A query that `sequence_read_before_effect_edges` puts
before a keyed store precedes that store's carrier (c489: `t = len(d); d[7]
= 4` gave 327 for 227). Focused: 197 dict/set/list/table corpus programs
pass (main: the same 194 pass); 8 table/schedule test files fail the same
16 names on branch and main.
