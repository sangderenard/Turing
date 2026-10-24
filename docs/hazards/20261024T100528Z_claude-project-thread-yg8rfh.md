2026-10-24 HAZARDS LOG SPLIT (MEASURED, cloud): new log entries are one file
each in docs/hazards/, named YYYYMMDDTHHMMSSZ_<branch>.md, instead of a block
prepended to TEST_BASELINE_AND_HAZARDS.md. Nearly every merge conflict on
2026-10-24 was two PRs prepending entries to that file. The old file keeps
every earlier entry unchanged under a short header that gives the new rule and
the read order (newest file first, then the old file).
`python tools/hazards_log.py` prints the log in that order; `new` creates this
branch's entry file. AGENTS.md (Testing, opening note) and
frontier_tasks/README.md name the new convention. Documentation and a tool
only; no compiler code changed. Gate at 3ea17296 vs main b6fba18 (container
without ziglang, so both trees show the same extra names): failure names
identical in all four parts. tests/test_hazards_log_tool.py: 4 passed.
