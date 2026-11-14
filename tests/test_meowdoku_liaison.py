"""Authenticated local boundary used by the separate Windows liaison service."""
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku import liaison, remote_connections


class FixtureService:
    def __init__(self): self.calls = []
    def status(self, owner, provider):
        self.calls.append((owner, provider))
        return {"status": "authorized", "configured": owner == 7}


class LiaisonTests(unittest.TestCase):
    def test_authenticated_loopback_and_owner_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            key = Path(root) / "key"; key.write_bytes(os.urandom(32))
            fixture = FixtureService()
            server = liaison.LiaisonServer(key, fixture, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                good = liaison.LiaisonClient(key, port=server.server_address[1])
                self.assertTrue(good.status(7, "suno")["configured"])
                self.assertFalse(good.status(8, "suno")["configured"])
                self.assertEqual(fixture.calls, [(7, "suno"), (8, "suno")])
                bad = Path(root) / "bad"; bad.write_bytes(os.urandom(32))
                with self.assertRaises(RuntimeError):
                    liaison.LiaisonClient(bad, port=server.server_address[1]).status(7, "suno")
                self.assertEqual(len(fixture.calls), 2)
                with self.assertRaises(ValueError): good.status(7, "other")
                self.assertEqual(len(fixture.calls), 2)
                nonce = "a" * 32
                request = {"action": "status", "owner": 7, "body": {"provider": "suno"},
                           "nonce": nonce, "time": int(liaison.time.time())}
                wire = {**request, "mac": liaison._mac(key.read_bytes(), request)}
                for attempt in (1, 2):
                    with socket.create_connection(server.server_address, timeout=3) as sock:
                        liaison._send(sock, wire, liaison.MAX_REQUEST)
                        if attempt == 1:
                            self.assertTrue(liaison._recv(sock, liaison.MAX_RESPONSE)["ok"])
                        else:
                            with self.assertRaises(ConnectionError):
                                liaison._recv(sock, liaison.MAX_RESPONSE)
                self.assertEqual(len(fixture.calls), 3)
            finally:
                server.shutdown(); server.server_close()

    def test_web_factory_uses_canonical_account_state_key(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root) / "state"
            (state / "run").mkdir(parents=True)
            (state / "run" / "liaison.enabled").touch()
            sessions = SimpleNamespace(accounts_db=state / "sql" / "accounts.db")
            with patch.object(liaison, "LiaisonClient", wraps=liaison.LiaisonClient) as client:
                remote_connections.service_for(sessions)
                self.assertEqual(client.call_args.args[0], state / "secrets" / "liaison-ipc.key")

    def test_liaison_vault_is_owner_scoped(self):
        with tempfile.TemporaryDirectory() as root:
            read, write, clear = liaison.vault_callbacks(Path(root), 7, "suno")
            write({"tokens": {"access_token": "test-only"}})
            self.assertEqual(read()["tokens"]["access_token"], "test-only")
            self.assertIsNone(liaison.vault_callbacks(Path(root), 8, "suno")[0]())
            self.assertTrue((Path(root) / "7" / "suno.json").is_file())
            clear(); self.assertIsNone(read())


if __name__ == "__main__": unittest.main()
