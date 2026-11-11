"""Export a captured native Living Data mesh as standard OBJ/MTL plus source manifest."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples'))
from meowdoku.living_data_obj import write_obj_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path, help='JSON returned by livingDataStaticHost.meshSnapshot()')
    parser.add_argument('directory', type=Path)
    parser.add_argument('--stem', default='living-data')
    args = parser.parse_args()
    result = write_obj_bundle(json.loads(args.snapshot.read_text(encoding='utf-8')), args.directory, stem=args.stem)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
