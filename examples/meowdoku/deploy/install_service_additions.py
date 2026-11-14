"""Add pinned registry wheels without replacing existing service distributions.
Invoked only by Provision-ServicePython.ps1. Never upgrades existing packages.
"""
import argparse
import importlib.metadata as metadata
import json
import os
from urllib.parse import urlsplit
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def key(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def installed():
    return {key(d.metadata['Name']): d.version for d in metadata.distributions()}


def pins(path):
    result = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*==[A-Za-z0-9][A-Za-z0-9.!+_-]*', line):
            raise ValueError('Only exact name==version registry pins are accepted')
        result.append(line)
    if not result:
        raise ValueError('Empty requirements')
    return result


def check_plan(report, before):
    additions = []
    for item in report.get('install', []):
        name = key(item['metadata']['name'])
        version = item['metadata']['version']
        if name in before:
            raise ValueError('Refused replacement of installed distribution: ' + name)
        if item.get('is_direct'):
            raise ValueError('Direct artifacts are not supported')
        info = item.get('download_info', {})
        url = info.get('url', '')
        parsed = urlsplit(url)
        digest = info.get('archive_info', {}).get('hashes', {}).get('sha256', '')
        if (parsed.scheme != 'https' or parsed.hostname != 'files.pythonhosted.org'
                or parsed.port not in (None, 443) or parsed.username or parsed.password
                or parsed.query or parsed.fragment or not parsed.path.endswith('.whl')
                or any(c.isspace() for c in url)
                or not re.fullmatch(r'[a-fA-F0-9]{64}', digest)):
            raise ValueError('Expected a PyPI wheel with SHA256')
        additions.append((name, version, url, digest))
    return additions


def run(requirements):
    requested = pins(requirements)
    before = installed()
    pip = [sys.executable, '-I', '-m', 'pip', '--isolated', '--disable-pip-version-check']
    env = {k:v for k,v in os.environ.items()
           if not k.upper().startswith(('PIP_', 'PYTHON'))}
    env['PIP_CONFIG_FILE'] = os.devnull
    def invoke(args):
        subprocess.run(pip + args, check=True, env=env)
    # Refuse to layer new dependencies on an already inconsistent runtime.
    invoke(['check'])
    with tempfile.TemporaryDirectory(prefix='service-deps-') as tmp:
        constraints = Path(tmp) / 'constraints.txt'
        constraints.write_text(''.join(f'{n}=={v}\n' for n, v in sorted(before.items())))
        report = Path(tmp) / 'plan.json'
        invoke(['install', '--dry-run', '--no-cache-dir', '--only-binary=:all:',
                '--index-url', 'https://pypi.org/simple', '--constraint', str(constraints),
                '--report', str(report), *requested])
        additions = check_plan(json.loads(report.read_text()), before)
        if installed() != before:
            raise RuntimeError('Runtime changed concurrently; refusing install')
        if not additions:
            print('All requested distributions already installed; no changes.')
            return
        locked = Path(tmp) / 'wheels.txt'
        locked.write_text(''.join(f'{n} @ {u} --hash=sha256:{h}\n' for n,v,u,h in additions))
        try:
            # Exact resolved new packages only: pip cannot upgrade an existing dependency.
            invoke(['install', '--no-cache-dir', '--only-binary=:all:', '--no-deps',
                    '--no-index', '--require-hashes', '-r', str(locked)])
            after = installed()
            if any(after.get(n) != v for n, v in before.items()):
                raise RuntimeError('Existing distribution changed concurrently')
            if any(after.get(n) != v for n,v,u,h in additions):
                raise RuntimeError('Incomplete installation')
            invoke(['check'])
        except BaseException:
            # Roll back only our newly introduced distributions, including partial installs.
            current = installed()
            rollback = [n for n,v,u,h in additions if n in current]
            if rollback:
                invoke(['uninstall', '--yes', *rollback])
            if installed() != before:
                raise RuntimeError('Rollback could not restore inventory; operator review required')
            raise
        print('Added', len(additions), 'distributions; all existing versions unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('requirements')
    run(parser.parse_args().requirements)
