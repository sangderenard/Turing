"""One object that names every place Meowdoku keeps state, so no module derives a path from a checkout.

RULE: after the migration nothing mutable lives under a repo worktree.  Databases, user content, logs, secrets, per-process run
files and canvases live under ONE root (``--state-dir``, design: deploy/STATE_AND_FEDERATION_2026-11-10.md); ``--web-dir`` stays
per worktree because it is a product tree tied to a source revision.

Two layouts resolve through the same names:

  StatePaths(root)               the new layout   root/{sql,content,products,canvas,logs,secrets,run}
  StatePaths.legacy(repo_root)   today's layout   <repo_root>/build/meowdoku/... (everything beside web/, as before)
  StatePaths.legacy_at(base)     today's layout rooted at an explicit directory (what ``web_dir.parent`` was)

Every module that used to write ``web_dir.parent / <name>`` or ``STATE / <name>`` asks the instance instead; with no ``--state-dir``
the answer is byte-for-byte the old path.
"""
from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_PORT = 8766
HERE = Path(__file__).resolve().parent


class StatePaths:
    def __init__(self, root, *, _legacy: bool = False):
        self.root = Path(root)
        self.is_legacy = _legacy
        r = self.root
        if _legacy:      # everything beside web/, exactly where the modules put it before there was a state root
            self.sql_dir = self.content_dir = self.products_dir = self.secrets_dir = self.run_dir = r
            self.canvas_dir, self.logs_dir = r / "canvas", r / "logs"
            self.job_library_db = r / "marble_photos" / "job-library.sqlite"
            self.log_ring = r / "log_ring.jsonl"
            self.operator_credentials_dir = r / "operator_credentials"
        else:
            self.sql_dir, self.content_dir, self.products_dir = r / "sql", r / "content", r / "products"
            self.canvas_dir, self.logs_dir = r / "canvas", r / "logs"
            self.secrets_dir, self.run_dir = r / "secrets", r / "run"
            self.job_library_db = r / "sql" / "job-library.sqlite"
            self.log_ring = r / "logs" / "log_ring.jsonl"
            self.operator_credentials_dir = r / "secrets" / "operator_credentials"

    # ---- constructors
    @classmethod
    def legacy_at(cls, base) -> "StatePaths":
        return cls(base, _legacy=True)

    @classmethod
    def legacy(cls, repo_root) -> "StatePaths":
        return cls.legacy_at(Path(repo_root) / "build" / "meowdoku")

    @classmethod
    def coerce(cls, value) -> "StatePaths":
        """A StatePaths as is; a plain directory means the old meaning of a state dir (the legacy layout rooted there)."""
        return value if isinstance(value, cls) else cls.legacy_at(Path(value))

    # ---- sql
    @property
    def accounts_db(self) -> Path:
        return self.sql_dir / "accounts.db"

    @property
    def access_json(self) -> Path:
        return self.sql_dir / "access.json"

    # ---- jobs (source_divider: one directory per divide-and-lower job; legacy build/meowdoku/jobs)
    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    # ---- content
    @property
    def profiles_json(self) -> Path:
        return self.content_dir / "profiles.json"

    @property
    def marble_objects_dir(self) -> Path:
        return self.content_dir / "marble_objects"

    @property
    def marble_photos_dir(self) -> Path:
        return self.content_dir / "marble_photos"

    @property
    def users_dir(self) -> Path:
        return self.content_dir / "users"

    @property
    def obj_assets_dir(self) -> Path:
        return self.content_dir / "obj_assets"

    @property
    def shares_dir(self) -> Path:
        return self.content_dir / "shares"

    @property
    def tickets_dir(self) -> Path:
        return self.content_dir / "tickets"

    # ---- secrets
    @property
    def tls_dir(self) -> Path:
        return self.secrets_dir / "tls"

    def admin_token(self, port=None) -> Path:
        """admin_token.txt for the default port (or None), admin_token_<port>.txt for any other (lifecycle's per-port rule)."""
        suffix = "" if port in (None, DEFAULT_PORT) else f"_{port}"
        return self.secrets_dir / f"admin_token{suffix}.txt"

    # ---- run
    def sessions_dir(self, port=None) -> Path:
        return self.run_dir / f"sessions{'' if port in (None, DEFAULT_PORT) else f'_{port}'}"

    def launch_json(self, port=None) -> Path:
        return self.run_dir / f"launch{'' if port in (None, DEFAULT_PORT) else f'_{port}'}.json"

    def ensure(self, restrict: bool = True) -> list[str]:
        """Create the directories (the new layout's seven plus tls/ and operator_credentials/ under secrets/) and restrict secrets/ to
        the current user with the same icacls step operators.py uses.  The legacy layout only gets its base directory."""
        if self.is_legacy:
            self.root.mkdir(parents=True, exist_ok=True)
            return []
        for d in (self.sql_dir, self.content_dir, self.products_dir, self.canvas_dir, self.logs_dir, self.secrets_dir, self.run_dir,
                  self.tls_dir, self.operator_credentials_dir):
            d.mkdir(parents=True, exist_ok=True)
        if not restrict:
            return []
        if str(HERE / "deploy") not in sys.path:
            sys.path.append(str(HERE / "deploy"))
        import operators as operators_mod             # noqa: E402  (the one place the restriction is written)
        return [operators_mod._restrict(self.secrets_dir)]
