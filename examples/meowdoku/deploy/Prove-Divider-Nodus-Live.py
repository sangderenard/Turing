"""Send ONE real divider piece through the live Nodus compile host and print what happened.

    python examples/meowdoku/deploy/Prove-Divider-Nodus-Live.py --port 47611 \
        --jobs-dir C:/dev/Powershell/srv/state/jobs/nodus-compile --work <empty temp dir> [--backend llvm]

The divider's own tables and source are kept in ``--work`` (a throwaway state root), so the live accounts database is
not touched. The request file, the product and the host's result land in ``--jobs-dir``, which is the live compile
host's jobs directory. This runs a real Turing compile on the canvas host's worker thread.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku import divider_nodus_bridge as bridge  # noqa: E402
from meowdoku import source_divider as divider  # noqa: E402
from meowdoku.state_paths import StatePaths  # noqa: E402

SOURCE = "def scale(x: AbstractTensor, y: AbstractTensor) -> AbstractTensor:\n    return x * y\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--jobs-dir", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--backend", default="llvm", choices=("c", "llvm", "fortran"))
    parser.add_argument("--service-id", type=int, default=9001)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--nodus-tools", default=str(ROOT.parent / "nodus" / "tools"))
    args = parser.parse_args(argv)

    state = StatePaths(Path(args.work) / "state")
    db = divider.connect(state.accounts_db)
    job = divider.submit(db, state, divider.intake_text(state, "prove_scale.py", SOURCE), owner_key="prove",
                         backends=(args.backend,))
    route = bridge.NodusCompileRoute("127.0.0.1", args.port, args.service_id, Path(args.jobs_dir),
                                     Path(args.nodus_tools), source_root=state.jobs_dir)
    started = time.time()
    print(json.dumps({"job": job, "backend": args.backend, "submitted_at": started}), flush=True)
    result = bridge.run_one(db, route, lease_owner="prove-divider-nodus-live", timeout_s=args.timeout)
    seconds = round(time.time() - started, 1)
    attempts = [dict(row) for row in db.execute(
        "SELECT piece_id, backend, state, stage FROM divider_attempts WHERE job_id=?", (job,))]
    print(json.dumps({"seconds": seconds, "result": result, "attempts": attempts}, default=str, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
