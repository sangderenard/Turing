2026-10-24 GATE FAILS A PART THAT DID NOT RUN (MEASURED, cloud):
tools/gate/run_gate.sh now exits 1 and prints "NO PYTEST SUMMARY; the part
did not run" with the log tail for any part whose log has no pytest result
line (missing requirements, a collection error, the timeout). Before this,
such a part printed an empty NEW list, which reads as a pass (the Windows
parity lane hit exactly that). A part that ran keeps its old output, and the
script still exits 0 when every part ran, NEW names or not. Checked with
PYTHON pointing at an interpreter that cannot run pytest: all four parts
flagged, exit 1. The four real part logs of a cloud main run (aa 86 failed /
301 passed, ab 8/254, ac 26/475, rec 20/255) all match the summary pattern.
Tool only; no compiler code changed.
