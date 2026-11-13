# Studio radio-to-light routing, 2026-11-13

The Studio camera routing grid can now opt in to its page's Nodus Radio widget.
The widget's accepted poll triggers an authenticated `action: radio` request;
only the configured server snapshot supplies measurement values. Browser values
are never accepted as measurements. Each account uses its existing grid panel.
The new button is Connect/Disconnect Nodus Radio. Open the Radio widget and
patch `radio:rms` or `radio:band:N` to existing light/filter receivers in the
original Spectral mixer. No automatic connections, new lights, FFT, or alternate
mixer were added. Closing the widget, stale feed, disconnect, or late replies
release the override; a two-second local expiry covers lost notifications.

Spectral dependency is the `codex/radio-light-jack-20261113` worktree. It extends
CameraTriggerGrid with original compile_nodes/GraphSolver level processing.
This first preview is explicitly zero-delay: NRAD snapshots are not a recorded
PCM clock, so delayed wires are rejected rather than mis-timed. A timed ray
exposure with patched radio sources is also rejected until recording of those
source values is connected. Normal camera pulses and unconnected lights remain
unchanged. This is not completed audio-synchronized ray capture.

Verification: real CPU Pygame/GraphSolver grid check passes, including chained
radio-to-bus-to-light transfer, save/restore, stale handling, delay rejection and
normal camera pulses. Three service tests pass for account isolation, server-only
measurements, missing grid/account and unavailable source. Node browser fixture
passes opt-in, disconnect, stale expiry, pagehide and late-result handling.
No GPU linking, renders, audio playback, live state, restart or deployment.

The installed global Python 3.11 can run the real CPU test. Turing's venv lacks
Torch; SpeakToMe's venv fails NumPy native import/Torch DLL import. The live
Torch-free service still has no verified worker-placement boundary for the
existing Spectral grid. Do not label this integration live until that is fixed.
