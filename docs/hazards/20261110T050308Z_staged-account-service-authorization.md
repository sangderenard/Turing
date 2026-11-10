# Staged account and service authorization foundation

Staged only under build/dot-permissions-20261110. Proposed examples/meowdoku/authorization.py extends existing account SQLite with explicit principal/role/grant schema and a deny-by-default principal/action/resource/connection decision. No live source or database changes, no roles or credentials provisioned, no proxy/network activation or restart. Existing modified server.py and untracked accounts.py untouched.

13 focused unittest cases pass with isolated in-memory SQLite. Covered guest shader/login-only scope, explicit existing-user administrator, disabled service records, separate Apache/Nodus/multigame/coordinator roles, backend/peer/web isolation, resource ownership/grants, lease expiry/revocation/revision, CSRF and transport checks. No HTTP, browser, websocket or full repository suite run. Route adapter integration and production permission migration remain pending approval and exact administrator identity.

Static inspection documents current app and shader API dependencies. No cold browser execution verified. Legacy unindexed job ownership remains a migration hazard. Existing hazards helper has a known malformed filename; newest valid receipt and baseline were read directly.
