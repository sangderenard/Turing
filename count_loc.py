#!/usr/bin/env python3
from pathlib import Path

# directories to exclude from counting
EXCLUDE_DIRS = {'.git', '__pycache__'}


def iter_python_files(root: Path):
    for path in root.rglob('*.py'):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        yield path


def main():
    root = Path(__file__).resolve().parent
    total = 0
    for path in iter_python_files(root):
        with path.open('r', encoding='utf-8') as f:
            total += sum(1 for _ in f)
    print(total)


if __name__ == '__main__':
    main()
