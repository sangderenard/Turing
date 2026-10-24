# Hazards log entries

One file per entry of the test baseline and hazards log. This replaces
prepending to the top of `TEST_BASELINE_AND_HAZARDS.md`, which made every two
PRs that each wrote an entry conflict.

- **Name:** `YYYYMMDDTHHMMSSZ_<branch>.md`, the UTC time of writing and the
  branch with `/` spelled `-`. `python tools/hazards_log.py new` makes it.
  The name sorts by time and differs between branches, so files never collide.
- **Content:** the same as an old top entry: what was measured, at which
  commits, and what is not green. Start with the date and a title in capitals,
  as before.
- **One PR, one file.** To re-measure after merging main, edit your own file
  (its name may keep its first time).
- **Read order:** newest file name first, then
  `TEST_BASELINE_AND_HAZARDS.md`, whose entries predate this directory.
  `python tools/hazards_log.py` prints the newest 10 entries, then the top of
  the old log; `--all` prints everything.
