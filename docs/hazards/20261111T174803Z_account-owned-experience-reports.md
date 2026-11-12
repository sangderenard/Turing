2026-11-11 SQL EXPERIENCE REPORTS (Windows, source only):

New experience_reports.py holds the original DOC/TTICKET/AUDIT/LOG guestbook
in the existing Accounts database. It owns experience_reports and
experience_report_followups under schema module experience_reports v1.
Owner identity comes only from the current session, rechecked transactionally.
The report is immutable; explicit prompt history is preserved verbatim and
each explicit next step commits as a linked SQL row. Exact request-id retry
returns the original receipt; mismatched retry refuses. No historical import,
auto-diagnostics, attachment capture, execution, public sharing or deletion.

SpeakToMe owns the new reports.py REST contract v1. Small Turing loader/server
bindings expose /api/reports, /api/reports/map and /api/reports/{id}. MCP make
creates a report, enter map=reports uses existing cursors, and examine reads
the exact SQL record. Reports remain distinct from issue tickets. Original
body templates remain policy; category, explicit prompt history, next steps,
owner boundaries, request size and rate are enforced. Follow-ups are historical
records, not executable jobs or a completed assignment workflow.

Final focused checks:
- 16 new storage/real HTTP/MCP tests pass (independent rerun, 5.172 seconds).
- 29 existing Turing issue/schema/dependency/cursor/static tests and 10 subtests
  pass (9.27 seconds; existing cffi imp deprecation warning).
- 108 SpeakToMe MCP/manual/report-tool/REST tests and 14 subtests pass
  (12.41 seconds).
- Scoped diff whitespace checks and new-source AST parsing pass. Guestbook
  filename validation passes without applying changes.

Independent review found ASCII JSON escaping inflated a valid Unicode report
past the HTTP byte bound. MCP now serializes normal UTF-8 JSON; the actual Rest
transport/HTTP regression proves the fix. Additional tests cover pagination,
owner isolation, anonymous/viewer/dev_ui denial, lost/expired sessions, newer
schema refusal, concurrent/replayed requests, atomic next-step failure rollback,
malformed framing and connection closure without report-schema initialization.

The hazards-log helper refused a preexisting malformed filename. The newest
valid hazard entries and legacy opening were read directly before testing.
No full compiler/repository suite or live report acceptance is claimed.

Live boundary verified with the Studio publisher: PID 23588 / boot 80a815f06ed5
started at 17:37:11 UTC. All report source changes postdate that boot (first
module 17:38:06, host binding 17:38:48). Removing only the new report hunks in
memory reproduces the publisher's frozen server/loader hashes; /api/reports
still returned 404. No restart, deployment, direct live SQL write, commit or
push by this lane. The next service reload must explicitly include and verify
the matching SpeakToMe/Turing report source. Concurrent source remains preserved.

Policy, sources and remaining release step:
speaktome/AGENTS/experience_reports/1794419283_DOC_SQL_Experience_Reports.md
speaktome/todo/sql_experience_reports_release.stub.md
