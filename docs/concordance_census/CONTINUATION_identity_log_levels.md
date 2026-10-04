# Identity-book log levels (2026-10-04)

Why: every compile dumped the whole book (one log reached 0.7 GB; 40 GB of
logs filled C:). Logs are a receipt, not a fact store; they never change
what compilation does.

## What was built
- `identity_concordance.py`: `IdentityLogLevel` (OFF < SUMMARY < FACTS < FULL),
  `Page.rows_level` (declared; default FACTS; not part of page equality),
  `declare_page(..., rows_level=)`, `iter_identity_book_lines`,
  `render_identity_book` (= join of the FULL lines, byte-identical to the old
  one, tested), `write_identity_log(book, path_stem, *, level, kind,
  extra_lines)` (lzma stream to a temp name, `os.replace` to
  `<stem>.<kind>.log.xz`, silent, never raises),
  `resolve_identity_log_level`, `identity_log_lzma_preset`.
- Levels: SUMMARY = header + one line per page + unsourced counts + the
  caller's `extra_lines` (the wrapper passes the detector findings it already
  computed). FACTS = pages with `rows_level <= FACTS` print each row's LAST
  span only, fact truncated at 240 chars with a `...[N chars]` marker and
  `[N spans]`; heavier pages print their header line only. FULL = the old dump.
- Defaults: OK compile -> SUMMARY, FAILED -> FULL. Override:
  `identity_log_level=` kwarg on `lower_ast_source_to_ssa` (popped like
  `identity_book=`), else env `TURING_IDENTITY_LOG_LEVEL=off|summary|facts|full`.
- Preset: `TURING_IDENTITY_LOG_LZMA_PRESET` (0-9, optional `e`), default 3.
- Declared FULL pages: `source_span`, `canonical_value`, `name_binding`,
  `identity_transition` (concordance_declarations.py) and the private
  `concordance_edge`, `concordance_dependents` (identity_concordance.py).
  Measured on a 735 MB log: canonical_value 417 MB, dependents 153 MB,
  edge 108 MB, identity_transition 29 MB, name_binding 10 MB; source_span small.
- Files are now `.log.xz` (`<label>.<stamp>.<ok|failed>.log.xz`,
  `<piece>.book.log.xz`, `<piece>.stale.book.log.xz`). No reader of the old
  names existed in the repo (git grep).
- `tools/compress_identity_logs.py DIR`: verified (sha256 round trip) before
  deleting the original; skips files modified in the last 10 minutes.

## Not done / open
- `turing/artifacts/identity_logs` holds ~26.9 GB of old `.log` files; the
  tool was only authorized and run on `engine_toy/artifacts/identity_logs`.
