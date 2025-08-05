# ANALOG_SPEC

Notes for `analog_spec.py`: placeholder maths must be replaced with physically credible PCM processing.

## Implement real analog operators
- Substitute list operations for `sigma_L`, `sigma_R`, `concat`, `slice`, `mu`, `length`, and `zeros` with lane-aware PCM transforms.
- Respect global constants (LANES, BIT_FRAME_MS, DATA_ADSR, etc.) when generating or timing frames.
- `mu` must perform VCA-style gating between X and Y lanes, driven by the selector's amplitude.
- `length` should run a simulated motor sweep to an end-stop marker and encode the elapsed frame count as binary PCM frames.

## Documentation & approximations
- Document any remaining simplifications (e.g., linear envelopes, integer quantisation) and reference the relevant AGENTS.md sections.
- Provide helper utilities for silence generation, frame concatenation, and FFT analysis that all honour `FRAME_SAMPLES`.
- Add unit tests verifying each primitive's round-trip: PCM → FFT → bits.

---
No silent stubs or pure digital math. Every operator must emit audio evidence of its effect per AGENTS.md §6.
