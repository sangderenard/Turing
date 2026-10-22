import sys, pathlib
sys.argv, files = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import differential_fuzz as fuzz
for f in files:
    st, d = fuzz.isolated(open(f).read())
    print(pathlib.Path(f).stem, st, d[:300], flush=True)
