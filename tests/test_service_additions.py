import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

path = Path(__file__).parents[1] / 'examples/meowdoku/deploy/install_service_additions.py'
spec = importlib.util.spec_from_file_location('additions', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class AdditionsTests(unittest.TestCase):
    def test_pins_reject_options_urls_and_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'r.txt'
            for value in ('--index-url https://bad', 'x @ https://bad', 'x>=1', '-r other'):
                p.write_text(value)
                with self.assertRaises(ValueError): m.pins(p)
            p.write_text('# test\nx==1.0\n')
            self.assertEqual(m.pins(p), ['x==1.0'])

    def test_plan_refuses_existing_and_direct(self):
        for item in ({'metadata': {'name':'old','version':'2'}},
                     {'metadata':{'name':'new','version':'1'},'is_direct':True}):
            with self.assertRaises(ValueError): m.check_plan({'install':[item]}, {'old':'1'})

    def test_plan_requires_approved_wheel_and_hash(self):
        good = {'metadata':{'name':'new','version':'1'}, 'download_info':{
            'url':'https://files.pythonhosted.org/new.whl',
            'archive_info':{'hashes':{'sha256':'a'*64}}}}
        self.assertEqual(m.check_plan({'install':[good]}, {})[0][0], 'new')
        for url in ('https://evil.test/new.whl', 'http://files.pythonhosted.org/new.whl',
                    'https://files.pythonhosted.org/new.tar.gz',
                    'https://user@files.pythonhosted.org/new.whl'):
            good['download_info']['url'] = url
            with self.assertRaises(ValueError): m.check_plan({'install':[good]}, {})
        good['download_info']['url'] = 'https://files.pythonhosted.org/new.whl'
        good['download_info']['archive_info']['hashes']['sha256'] = ''
        with self.assertRaises(ValueError): m.check_plan({'install':[good]}, {})

    def exercise(self, fail=False):
        state = {'old':'1'}
        calls = []
        def invoke(args, check, env):
            self.assertEqual(env["PIP_CONFIG_FILE"], m.os.devnull)
            self.assertIn("-I", args)
            calls.append(args)
            if '--dry-run' in args:
                Path(args[args.index('--report')+1]).write_text(json.dumps({'install':[{'metadata':{'name':'new','version':'1'}, 'download_info':{'url':'https://files.pythonhosted.org/new.whl', 'archive_info':{'hashes':{'sha256':'a'*64}}}}]}))
            elif 'install' in args:
                self.assertIn('--no-deps', args)
                self.assertIn('--require-hashes', args)
                self.assertIn('--no-index', args)
                self.assertIn('--hash=sha256:' + 'a'*64, Path(args[args.index('-r')+1]).read_text())
                state['new'] = '1'
                if fail: raise RuntimeError('partial failure')
            elif 'uninstall' in args:
                state.pop('new')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'r.txt'; p.write_text('new==1')
            with patch.object(m, 'installed', side_effect=lambda:dict(state)), patch.object(m.subprocess,'run',side_effect=invoke):
                if fail:
                    with self.assertRaises(RuntimeError): m.run(p)
                else: m.run(p)
        self.assertEqual(state, {'old':'1'} if fail else {'old':'1','new':'1'})
    def test_success_preserves_old(self): self.exercise()
    def test_partial_failure_rolls_back_only_new(self): self.exercise(True)

if __name__ == '__main__': unittest.main()
