# SURVIVAL_COMPUTER

Guidance for `survival_computer.py`: bridge the gap between high-level bit ops and the analog tape simulator.

## Outstanding gaps
- Inputs never allocated; initial operands resolve to `None` and reads crash.
- `_execute_node` only handles `zeros`, `nand`, and `mu`; all others emit a placeholder instruction sound.

## Required work
1. **Graph preprocessing**
   - During compilation, allocate tape addresses for every node argument and write constant bit patterns before execution begins.
   - Guard against missing addresses; raise clear errors when operands are absent.
2. **Primitive translation**
   - Implement analog sequences for `sigma_L`, `sigma_R`, `slice`, `length`, and a real `concat` using cassette read/write operations.
   - Each primitive must call `cassette.read_wave`/`write_wave`; digital shortcuts are forbidden by AGENTS.md §6.
   - `length` should mechanically time a motor run to an end-marker and encode the frame count into PCM.
3. **Backend API alignment**
   - Import `CassetteTapeBackend` from `cassette_adapter` *or* add `execute_instruction` to the core backend.
   - Remove unsupported constructor args (`analogue_mode`, `frame_ms`) or map them to existing fields such as `time_scale_factor`.
4. **Error signaling & docs**
   - Replace silent fallbacks with explicit `NotImplementedError` or log warnings when primitives lack full analog support.
   - Expand module docstring to cite AGENTS.md §§6–7 so future edits honour lane, motor, and ADSR rules.

---
Every operation must leave an audible trace; digital shortcuts or silent stubs are considered failure under AGENTS.md §11.
