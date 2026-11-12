# Shared site header clearance and collapse

The existing fixed `site-nav.js` banner covered the page's first controls.
It now reserves its measured height through its existing Shadow DOM host at
start of body (ResizeObserver plus resize updates), and offers Hide menu /
Show site menu. Collapse hides all account/suite actions, dismisses the open
panel and stops suite sampling, retaining a keyboard-accessible reopen button.
Login, SQL tickets, capabilities and server routing remain unchanged. Account
menu overflow is bounded for mobile screens; Escape never focuses a hidden
account button after collapse.

Verification: node tests/site_control_suite_check.mjs passed; node --check
examples/meowdoku/web/site-nav.js passed; git diff --check passed. Expanded
browser fixture checks guest/operator, 390/1280 widths, clearance and collapse.
Standalone Edge and Chrome both exited before DOM output with crashpad errors
(the historical fixture also records this executor issue). Actual layout remains
pending connected-browser verification. No Studio render, GPU link, server
restart, live publication or authentication change was performed.
