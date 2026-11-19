import hashlib
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku.state_paths import StatePaths
from meowdoku.nodus_radio_jukebox import Jukebox
from meowdoku.radio_upload_assets import upload, owned, ControlError

# A valid FLAC marker and STREAMINFO for one second of mono 44.1 kHz PCM.
STREAM = ((44100 << 44) | (0 << 41) | (15 << 36) | 44100).to_bytes(8, "big")
FLAC = b"fLaC" + b"\x80\x00\x00\x22" + b"\x00\x10\x10\x00" + b"\x00" * 6 + STREAM + b"\x00" * 16


class RadioUploadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.state = StatePaths(Path(temporary.name))
        self.state.ensure(restrict=False)
        self.db = self.state.accounts_db

    def test_original_is_account_owned_and_offered_without_station_config(self):
        with patch("meowdoku.suno_radio_assets.run_codec", side_effect=lambda _s, _src, dst, _mode: dst.write_bytes(b"RIFF")) as codec:
            receipt = upload(self.state, self.db, 7, "Lantern", BytesIO(FLAC), len(FLAC))
        self.assertEqual(codec.call_args.args[3], "flac_decode")
        self.assertEqual(receipt["sha256"], hashlib.sha256(FLAC).hexdigest())
        self.assertEqual(len(owned(self.db, 7)), 1)
        self.assertEqual(owned(self.db, 8), [])
        catalogue = Jukebox(self.db, self.state.radio_json, state=self.state).catalogue()
        self.assertEqual(catalogue[receipt["id"]]["title"], "Lantern")
        self.assertEqual(Path(catalogue[receipt["id"]]["path"]).read_bytes(), FLAC)
        self.assertFalse(self.state.radio_json.exists())

    def test_missing_codec_refuses_publication(self):
        with self.assertRaises(OSError):
            upload(self.state, self.db, 7, "No codec", BytesIO(FLAC), len(FLAC))
        self.assertEqual(owned(self.db, 7), [])
        self.assertFalse(list(self.state.users_dir.rglob("*.flac")))

    def test_bad_and_truncated_flac_leave_no_asset(self):
        for body, length in ((b"not a flac" * 5, 50), (FLAC[:30], len(FLAC))):
            with self.assertRaises(ControlError):
                upload(self.state, self.db, 7, "Bad", BytesIO(body), length)
        self.assertEqual(owned(self.db, 7), [])
        self.assertFalse(list(self.state.users_dir.rglob("*.flac")))

    def test_codec_duration_ceiling(self):
        long_stream = ((44100 << 44) | (0 << 41) | (15 << 36) | (901 * 44100)).to_bytes(8, "big")
        body = FLAC[:18] + long_stream + FLAC[26:]
        with self.assertRaises(ControlError):
            upload(self.state, self.db, 7, "Too long", BytesIO(body), len(body))


if __name__ == "__main__":
    unittest.main()
