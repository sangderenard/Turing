"""Per-account Alpaca paper credentials in the restricted Meowdoku state secrets tree.

This is separate from user_files: no list, download, message subject, or static
route can return these bytes. The future paper market service may read them only
for the authenticated account whose id names the file.
"""
from __future__ import annotations

import json
import os
import secrets
from pathlib import Path


class CredentialError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _path(secrets_dir: Path, account_id: int) -> Path:
    if not isinstance(account_id, int) or account_id < 1:
        raise CredentialError("invalid account", 403)
    return Path(secrets_dir) / "user_credentials" / str(account_id) / "alpaca-paper.json"


def configured(secrets_dir: Path, account_id: int) -> bool:
    return _path(secrets_dir, account_id).is_file()


def save_paper(secrets_dir: Path, account_id: int, key_id: str, secret_key: str) -> None:
    for value in (key_id, secret_key):
        if (not isinstance(value, str) or not 8 <= len(value) <= 256 or
                value.strip() != value or any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in value)):
            raise CredentialError("provide a valid paper key ID and secret")
    target = _path(secrets_dir, account_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("." + secrets.token_hex(16) + ".tmp")
    payload = json.dumps({"key_id": key_id, "secret_key": secret_key}, separators=(",", ":")).encode("utf-8")
    try:
        with open(temporary, "xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_paper(secrets_dir: Path, account_id: int) -> tuple[str, str]:
    """Server-side only. Never pass this return value to an HTTP response or log."""
    try:
        data = json.loads(_path(secrets_dir, account_id).read_text(encoding="utf-8"))
        return data["key_id"], data["secret_key"]
    except (OSError, ValueError, KeyError, TypeError):
        raise CredentialError("paper credentials unavailable", 503) from None
