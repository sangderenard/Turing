# 2026-11-13 - OPTIONAL TRAINING SEED DISPLAY

Continuous trainer progress omits the optional seed field. The spectator UI
printed `seed null` even though the SQL run was running and validation records
contained valid epochs, losses and validation means. Latest read-only evidence:
epoch 626000, loss 0.0254777572, validation mean 0.0166401611, gate unmet.
This is missing display metadata, not an exception or a successful gate.

Only display supplied seed/epoch metadata, preserving a valid seed of zero.
Node syntax check and execution of the actual drawQualification function with
null, zero and nonzero seed records passed. No trainer, SQL write, service
restart or deployment performed. Live browser delivery remains unverified.
