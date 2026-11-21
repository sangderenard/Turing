import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from meowdoku.state_paths import StatePaths
from meowdoku.nodus_radio_jukebox import Jukebox
from meowdoku.suno_radio_assets import import_song, owned, station_catalogue, ControlError

SONG = '11cd4c14-7835-4d28-b3aa-1b4de19d5fbc'
MP3 = b'ID3' + b'\x00' * 2045


class Provider:
    def __init__(self, credit=False):
        self.credit = credit
        self.calls = []

    def call(self, owner, provider, name, arguments):
        self.calls.append((owner, provider, name, arguments))
        if name == 'audio-retrieve':
            value = {'id': SONG, 'title': 'On and On', 'status': 'complete'}
        elif self.credit and not arguments['confirm']:
            value = {'status': 'requires_credit', 'credits_remaining': 27}
        else:
            value = {'url': 'https://cdn1.suno.ai/example.mp3'}
        return {'content': [{'type': 'text', 'text': json.dumps(value)}], 'isError': False}


async def fetch(_url, target):
    target.write_bytes(MP3)
    return len(MP3), hashlib.sha256(MP3).hexdigest()


class SunoAssetsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.state = StatePaths(Path(temporary.name))
        self.state.ensure(restrict=False)
        self.db = self.state.accounts_db

    def test_credit_check_does_not_import_or_spend(self):
        provider = Provider(credit=True)
        result = import_song(self.state, self.db, 7, SONG, 'mp3', False, provider, fetch=fetch)
        self.assertEqual(result['status'], 'requires_credit')
        self.assertEqual(owned(self.state, self.db, 7), [])
        self.assertFalse(list(self.state.users_dir.rglob('*.mp3')))
        self.assertFalse(provider.calls[-1][3]['confirm'])
        ready = import_song(self.state, self.db, 7, SONG, 'mp3', True, provider, fetch=fetch)
        self.assertEqual(ready['status'], 'ready')
        self.assertTrue(provider.calls[-1][3]['confirm'])

    def test_user_asset_and_jukebox_path_are_derived_from_owner(self):
        provider = Provider()
        result = import_song(self.state, self.db, 7, SONG, 'mp3', False, provider, fetch=fetch)
        self.assertEqual((result['title'], result['storage']), ('On and On', 'mp3'))
        self.assertEqual(owned(self.state, self.db, 8), [])
        item = owned(self.state, self.db, 7)[0]
        self.assertEqual(item['sha256'], hashlib.sha256(MP3).hexdigest())
        path = self.state.users_dir / 'account' / '7' / 'assets' / 'suno_radio' / SONG / 'audio.mp3'
        self.assertEqual(path.read_bytes(), MP3)
        catalogue = station_catalogue(self.state, self.db)
        self.assertEqual(catalogue[f'suno:7:{SONG}']['path'], str(path))
        jukebox = Jukebox(self.db, self.state.radio_json, state=self.state)
        self.assertIn(f'suno:7:{SONG}', jukebox.catalogue())
        self.assertEqual(import_song(self.state, self.db, 7, SONG, 'mp3', False, provider, fetch=fetch)['status'], 'ready')
        self.assertEqual(len(provider.calls), 2)

    def test_invalid_song_and_non_provider_download_are_refused(self):
        with self.assertRaises(ControlError):
            import_song(self.state, self.db, 7, '../song', 'mp3', False, Provider(), fetch=fetch)
        class BadProvider(Provider):
            def call(self, owner, provider, name, arguments):
                if name == 'audio-retrieve':
                    return super().call(owner, provider, name, arguments)
                return {'content':[{'type':'text', 'text':'{"url":"http://127.0.0.1/private"}'}]}
        with self.assertRaises(ControlError):
            import_song(self.state, self.db, 7, SONG, 'mp3', False, BadProvider(), fetch=fetch)
        self.assertEqual(owned(self.state, self.db, 7), [])


if __name__ == '__main__':
    unittest.main()
