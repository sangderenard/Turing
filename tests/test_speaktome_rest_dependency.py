"""The shared REST source must be explicit, unique and tensor-free to import."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples'))


def run_import(repo, extra=""):
    env = {**os.environ, "SPEAKTOME_REPO": str(repo)}
    return subprocess.run(
        [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(ROOT / 'examples')!r}); "
         + extra + "from meowdoku.rest_api import routes; "
         "assert routes.API_VERSION == 2; "
         "assert 'tensors' not in sys.modules; assert 'torch' not in sys.modules; "
         "print(routes.__file__)"], env=env, capture_output=True, text=True, timeout=15)


def test_selected_speaktome_package_is_the_executable_route_owner():
    from meowdoku.rest_api import REPOSITORY, routes
    result = run_import(REPOSITORY)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == REPOSITORY / 'speaktome_api' / 'routes.py'
    assert routes.messages_route.__module__ == 'speaktome_api.routes'
    assert str(REPOSITORY) not in sys.path


def test_missing_repository_fails_without_legacy_fallback(tmp_path):
    result = run_import(tmp_path)
    assert result.returncode != 0
    assert 'SPEAKTOME_REPO' in result.stderr and 'dependency is missing' in result.stderr


def test_different_preloaded_repository_is_refused(tmp_path):
    from meowdoku.rest_api import REPOSITORY
    fake = str(tmp_path / 'speaktome_api' / '__init__.py')
    result = run_import(REPOSITORY, "from types import SimpleNamespace; "
                        f"sys.modules['speaktome_api'] = SimpleNamespace(__file__={fake!r}); ")
    assert result.returncode != 0
    assert 'different checkout' in result.stderr


def test_incompatible_route_contract_is_refused(tmp_path):
    package = tmp_path / 'speaktome_api'
    package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'routes.py').write_text('API_VERSION = 1\n')
    result = run_import(tmp_path)
    assert result.returncode != 0
    assert 'requires SpeakToMe REST API_VERSION 2' in result.stderr
