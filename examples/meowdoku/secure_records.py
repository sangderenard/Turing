"""Authenticated encryption for the two retained IP-bearing record streams.

The key belongs in StatePaths.secrets_dir. A missing key for an existing
encrypted stream is an error; never silently replace it. Existing plaintext
streams are converted before another record is appended.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

PREFIX = b"enc:v1:"


class RecordCipher:
    def __init__(self, key_path: Path, *, existing_streams=()):
        self.key_path = Path(key_path)
        if not self.key_path.exists():
            for stream in existing_streams:
                stream = Path(stream)
                if stream.exists():
                    with stream.open("rb") as saved:
                        if saved.read(len(PREFIX)) == PREFIX:
                            raise FileNotFoundError("records key is missing for encrypted history")
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as out:
                out.write(Fernet.generate_key())
                out.flush()
                os.fsync(out.fileno())
        self.fernet = Fernet(self.key_path.read_bytes().strip())

    def seal(self, line: bytes) -> bytes:
        return PREFIX + self.fernet.encrypt(line.rstrip(b"\r\n")) + b"\n"

    def open(self, line: bytes) -> bytes:
        if not line.startswith(PREFIX):
            raise ValueError("unencrypted retained record")
        try:
            return self.fernet.decrypt(line[len(PREFIX):].strip())
        except InvalidToken as error:
            raise ValueError("retained record failed authentication") from error

    def migrate(self, path: Path) -> None:
        """Convert a legacy plain stream atomically, then remove its old bytes."""
        path = Path(path)
        if not path.exists():
            return
        partial = None
        with path.open("rb") as source:
            first = source.readline()
            if not first or first.startswith(PREFIX):
                return
            partial = path.with_name(path.name + ".encrypting-" + secrets.token_hex(8))
            try:
                with partial.open("xb") as out:
                    out.write(self.seal(first))
                    for line in source:
                        out.write(self.seal(line))
                    out.flush()
                    os.fsync(out.fileno())
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
        # Windows refuses replacement while the source handle is open.
        try:
            os.replace(partial, path)
        finally:
            partial.unlink(missing_ok=True)
