# CASSETTE_ADAPTER

Compatibility façade for the v2 tape backend.

## Purpose
- Exposes legacy methods (`move_head`, `execute_instruction`, `tape_length`) so older demos continue to run.

## Required actions
- Document a migration path: either port these shims into `cassette_tape.py` or update callers to use the core API directly.
- Ensure `execute_instruction` produces a faint but audible pulse rather than silence, satisfying AGENTS.md §6.
- Validate parameters for `move_head` and raise errors on out-of-range seeks.

## Clean-up markers
- Once no callers rely on this adapter, delete the file and update imports.
