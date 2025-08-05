# CASSETTE_TAPE

Guidance for `cassette_tape.py` backend to maintain analog fidelity.

## Bit I/O through analog pipeline
- Replace `read_bit`/`write_bit` wrappers with versions that perform FFT-based lane analysis and render PCM frames with bias and motor carriers engaged.
- Reading must seek, scan, and decode via `dominant_tone`; writing must call `generate_bit_wave` and log every motor/head movement.

## Instruction execution hook
- Provide an `execute_instruction` method that emits a short instruction-track pulse so higher layers need not import the adapter.

## Motor and head modelling
- Validate head alignment after every seek/read/write; raise informative errors when misaligned.
- Use trapezoidal envelopes from `analog_spec.trapezoidal_motor_envelope` for all movement profiles.

## Testing
- Add tests verifying analog read/write round-trips for single bits and multi-lane frames, ensuring PCM→FFT→bit correctness.

---
Absence of audible frames or reliance on digital shortcuts violates AGENTS.md §11.
