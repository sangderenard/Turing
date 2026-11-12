2026-11-10 GENERIC HEADER CONTROL SUITES (Windows, local source only)

site-nav.js now registers policy-driven control suites using page-owned
samplers/actions. Living Data optionally registers its real viewport policy
and existing callbacks. No replacement input/physics engine, live asset copy,
server restart, account permission or network-setting change.

Passed: node --check site-nav.js; Python py_compile abstract_ui_div_map.py;
Node syntax of the extracted main Living Data JS constant;
node tests/site_control_suite_check.mjs (offline DOM lifecycle boundary).
Focused viewport/dynamics pytest: 7 passed, 1 failed. The monitor test expects
seven keyboard signals, while current policy supplies fourteen, starting with
additional respawn. This run did not establish an untouched-checkout baseline;
do not count it as a full green regression result.

The hazards helper refused pre-existing malformed filename
20261109T_material-library-paused-checkpoint.md. Read the two newest hazard
receipts directly and the legacy log opening instead; did not rename others'
receipts. Browser layout, hosted generated-page integration and native pointer
capture remain unverified by this worker. See docs/SITE_CONTROL_SUITES.md.
