"""Add pinned registry wheels without replacing existing service distributions.
Invoked only by Provision-ServicePython.ps1. Never upgrades existing packages.
"""
import argparse
import importlib.metadata as metadata
import json
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
        additions.append(name + '==' + version)
    return additions


def run(requirements):
    requested = pins(requirements)
    before = installed()
    pip = [sys.executable, '-m', 'pip', '--isolated', '--disable-pip-version-check']
    def invoke(args):
        subprocess.run(pip + args, check=True)
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
        try:
            # Exact resolved new packages only: pip cannot upgrade an existing dependency.
            invoke(['install', '--no-cache-dir', '--only-binary=:all:', '--no-deps',
                    '--index-url', 'https://pypi.org/simple', *additions])
            after = installed()
            if any(after.get(n) != v for n, v in before.items()):
                raise RuntimeError('Existing distribution changed concurrently')
            if any(after.get(key(p.split('==')[0])) != p.split('==')[1] for p in additions):
                raise RuntimeError('Incomplete installation')
            invoke(['check'])
        except BaseException:
            # Roll back only our newly introduced distributions, including partial installs.
            current = installed()
            rollback = [p.split('==')[0] for p in additions if p.split('==')[0] in current]
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
