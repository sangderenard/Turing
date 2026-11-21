from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import admin_records
from meowdoku.secure_records import RecordCipher


def test_legacy_access_migrates_and_metrics_are_read_from_ciphertext(tmp_path):
    path = tmp_path / "logs" / "apache-access.log"
    path.parent.mkdir()
    path.write_bytes(b'192.0.2.1 - - [16/Nov/2026:10:00:00 -0600] "GET /mcp/guest HTTP/1.1" 200 17\n'
                     b'192.0.2.2 - - [16/Nov/2026:10:00:01 -0600] "POST /mcp/guest HTTP/1.1" 429 -\n')
    key = tmp_path / "secrets" / "records.key"
    result = admin_records.read(path, key, limit=1)
    assert result["requests"] == 2 and result["bytes"] == 17
    assert result["by_status"] == {"200": 1, "429": 1}
    assert result["ip_history"][0]["ip"] == "192.0.2.2"
    assert b"192.0.2." not in path.read_bytes()
    assert RecordCipher(key).open(path.read_bytes().splitlines()[0]).startswith(b"192.0.2.1")


def test_unreadable_record_is_counted_not_fatal(tmp_path):
    """Ticket f62934c8 hazard 3: a torn or spliced line must not fail the whole admin records view."""
    key = tmp_path / "secrets" / "records.key"
    access = tmp_path / "logs" / "apache-access.log"
    access.parent.mkdir()
    cipher = RecordCipher(key)
    good = cipher.seal(b'192.0.2.7 - - [16/Nov/2026:10:00:00 -0600] "GET / HTTP/1.1" 200 5')
    access.write_bytes(good + good[:40] + b"\n" + b"enc:v1:not-a-token\n" + good)
    result = admin_records.read(access, key)
    assert result["requests"] == 2 and result["unreadable"] == 2
    assert [item["ip"] for item in result["ip_history"]] == ["192.0.2.7", "192.0.2.7"]


def test_missing_key_never_replaces_encrypted_history(tmp_path):
    key = tmp_path / "secrets" / "records.key"
    access = tmp_path / "logs" / "apache-access.log"
    access.parent.mkdir()
    access.write_bytes(RecordCipher(key).seal(b"private"))
    key.unlink()
    with pytest.raises(FileNotFoundError, match="records key is missing"):
        RecordCipher(key, existing_streams=(access,))
    assert not key.exists()
