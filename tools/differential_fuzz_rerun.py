import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import differential_fuzz as fuzz


# Guarded: on Windows the fuzzer's spawned child re-imports this script.
if __name__ == "__main__":
    sys.argv, files = sys.argv[:1], sys.argv[1:]
    for f in files:
        st, d = fuzz.isolated(open(f).read())
        print(pathlib.Path(f).stem, st, d[:300], flush=True)
