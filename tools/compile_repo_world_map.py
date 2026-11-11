"""Compile repo-world-map.v1 files from fresh shallow clones of GitHub repositories.

    python tools/compile_repo_world_map.py --out DIR [--repo URL[@ref]]...
        [--repos-file JSON_LIST] [--work DIR] [--force] [--keep] [--allow-local]

See docs/REPO_WORLD_MAP_COMPILE_JOB.md.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.compiler.repo_world_scan import REPO_WORLD_SCAN_VERSION, scan_repository  # noqa: E402

MAP_SCHEMA = "repo-world-map.v1"
_GITHUB = re.compile(r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$")


@dataclass(frozen=True)
class CompileResult:
    repo: str
    commit: str
    directory: Path
    map_path: Path
    regions: int
    rooms: int
    portals: int
    unresolved: int
    seconds: float
    map_bytes: int
    skipped: bool = False


# --- files ---------------------------------------------------------------------

def _dump(data) -> bytes:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _measure(path: Path) -> tuple[int, int]:
    """(bytes, lines); lines 0 when a NUL is in the first 8 KB."""
    try:
        if not path.is_file():
            return 0, 0
        size = path.stat().st_size
        lines = 0
        last = b"\n"
        first = True
        with path.open("rb") as handle:
            while True:
                block = handle.read(1 << 20)
                if not block:
                    break
                if first and b"\0" in block[:8192]:
                    return size, 0
                first = False
                lines += block.count(b"\n")
                last = block[-1:]
        if size and last != b"\n":
            lines += 1
        return size, lines
    except OSError:
        return 0, 0


def _specifier_head(kind: str, specifier: str) -> str:
    if "python" in kind:
        stripped = specifier.lstrip(".")
        dots = specifier[: len(specifier) - len(stripped)]
        return dots + stripped.split(".", 1)[0]
    if specifier.startswith("."):
        return specifier.split("/", 1)[0]
    parts = specifier.split("/")
    return "/".join(parts[:2]) if specifier.startswith("@") else parts[0]


# --- compile -------------------------------------------------------------------

def _call_scan(scan, clone: Path, import_roots):
    params = inspect.signature(scan).parameters
    kwargs = {}
    if import_roots and "import_roots" in params:
        kwargs["import_roots"] = tuple(import_roots)
    return scan(clone, **kwargs)


def build_map(world, clone: Path, repo: str, generated: float) -> dict:
    paths = sorted({("" if r.path == "." else r.path) for r in world.regions})
    region_id = {p: i for i, p in enumerate(paths)}
    identity_to_path = {r.identity: ("" if r.path == "." else r.path) for r in world.regions}
    rooms_sorted = sorted(world.rooms, key=lambda r: r.path)
    room_id = {r.identity: i for i, r in enumerate(rooms_sorted)}

    direct = collections.Counter()
    total = collections.Counter()
    nbytes = collections.Counter()
    out_count = collections.Counter()
    in_count = collections.Counter()
    for p in world.portals:
        out_count[room_id[p.source_room]] += 1
        in_count[room_id[p.target_room]] += 1

    rooms = []
    room_region = []
    for i, room in enumerate(rooms_sorted):
        rpath = identity_to_path[room.region]
        size, lines = _measure(clone / room.path)
        rid = region_id[rpath]
        room_region.append(rid)
        direct[rid] += 1
        cur = rpath
        while True:
            total[region_id[cur]] += 1
            nbytes[region_id[cur]] += size
            if not cur:
                break
            cur = cur.rpartition("/")[0]
        rooms.append({"id": i, "region": rid, "name": room.name, "language": room.language,
                      "bytes": size, "lines": lines, "out": out_count[i], "in": in_count[i]})
    regions = []
    for p in paths:
        i = region_id[p]
        parent = None if p == "" else region_id[p.rpartition("/")[0]]
        regions.append({"id": i, "path": p, "parent": parent, "files": direct[i],
                        "total_files": total[i], "bytes": nbytes[i]})

    kinds = sorted({p.reference_kind for p in world.portals})
    kind_index = {k: i for i, k in enumerate(kinds)}
    portals = sorted([room_id[p.source_room], room_id[p.target_room], p.source_line,
                      kind_index[p.reference_kind]] for p in world.portals)
    links = collections.Counter()
    for s, t, _l, _k in portals:
        a, b = room_region[s], room_region[t]
        if a != b:
            links[(a, b)] += 1
    region_links = [[a, b, c] for (a, b), c in sorted(links.items())]

    heads = collections.Counter(_specifier_head(u.reference_kind, u.specifier) for u in world.unresolved)
    top = sorted(heads.items(), key=lambda kv: (-kv[1], kv[0]))[:30]
    return {
        "schema": MAP_SCHEMA, "repo": repo, "origin": world.origin or "",
        "commit": world.commit, "generated": generated,
        "scan_version": REPO_WORLD_SCAN_VERSION,
        "regions": regions, "rooms": rooms, "kinds": kinds, "portals": portals,
        "region_links": region_links, "unresolved_count": len(world.unresolved),
        "unresolved_top": [[h, c] for h, c in top], "notes": list(world.notes),
    }


def _summary(m: dict, seconds: float, sizes: dict) -> dict:
    rooms, regions = m["rooms"], m["regions"]

    def label(r):
        base = regions[r["region"]]["path"]
        return f"{base}/{r['name']}" if base else r["name"]

    def top(key):
        ranked = sorted(rooms, key=lambda r: (-r[key], r["id"]))[:20]
        return [[label(r), r[key]] for r in ranked if r[key]]

    langs = collections.Counter(r["language"] for r in rooms)
    links = sorted(m["region_links"], key=lambda l: (-l[2], l[0], l[1]))[:20]
    return {
        "repo": m["repo"], "commit": m["commit"], "generated": m["generated"],
        "scan_version": m["scan_version"],
        "counts": {"regions": len(regions), "rooms": len(rooms), "portals": len(m["portals"]),
                   "region_links": len(m["region_links"]), "unresolved": m["unresolved_count"]},
        "seconds": round(seconds, 3), "sizes": sizes,
        "top_rooms_in": top("in"), "top_rooms_out": top("out"),
        "top_region_links": [[regions[a]["path"], regions[b]["path"], c] for a, b, c in links],
        "languages": dict(sorted(langs.items())),
    }


def compile_from_clone(clone: Path, out_root: Path, repo: str, *, import_roots=(),
                       scan=scan_repository, force: bool = False) -> CompileResult:
    clone, out_root = Path(clone), Path(out_root)
    owner, _, name = repo.partition("/")
    repo_dir = out_root / f"{owner}__{name}"
    start = time.time()
    commit = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True, encoding="utf-8").stdout.strip()
    target = repo_dir / commit
    map_path = target / "map.json"
    if map_path.is_file() and (target / "summary.json").is_file() and not force:
        s = json.loads((target / "summary.json").read_text(encoding="utf-8"))
        c = s["counts"]
        _publish(out_root, repo_dir, repo, commit, s)
        return CompileResult(repo, commit, target, map_path, c["regions"], c["rooms"], c["portals"],
                             c["unresolved"], 0.0, map_path.stat().st_size, True)
    if not import_roots:                      # the roots the clone itself declares (pyproject, pytest.ini, repo_world.toml); never guessed
        declared = getattr(sys.modules.get(getattr(scan, "__module__", "")), "declared_import_roots", None)
        import_roots = tuple(declared(clone)) if declared else ()
    world = _call_scan(scan, clone, import_roots)
    generated = time.time()
    m = build_map(world, clone, repo, generated)
    map_bytes = _dump(m)
    gz = gzip.compress(_dump(world.to_data()), mtime=0)
    seconds = time.time() - start
    summary = _summary(m, seconds, {"map_json": len(map_bytes), "world_json_gz": len(gz)})
    _atomic_write(target / "world.json.gz", gz)
    _atomic_write(map_path, map_bytes)
    _atomic_write(target / "summary.json", _dump(summary))
    _publish(out_root, repo_dir, repo, commit, summary)
    c = summary["counts"]
    return CompileResult(repo, commit, target, map_path, c["regions"], c["rooms"], c["portals"],
                         c["unresolved"], seconds, len(map_bytes))


def _publish(out_root: Path, repo_dir: Path, repo: str, commit: str, summary: dict) -> None:
    latest = {"commit": commit, "generated": summary["generated"],
              "map": f"{commit}/map.json", "summary": summary}
    _atomic_write(repo_dir / "latest.json", _dump(latest))
    index_path = out_root / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        index = []
    index = [e for e in index if e.get("repo") != repo]
    index.append({"repo": repo, "dir": repo_dir.name, "commit": commit, "generated": summary["generated"], "latest": latest})
    index.sort(key=lambda e: e["repo"])
    _atomic_write(index_path, _dump(index))


# --- cloning -------------------------------------------------------------------

def parse_repo_spec(spec: str, allow_local: bool = False) -> tuple[str, str | None, str]:
    """-> (url_or_path, ref, owner/name)."""
    url, ref = spec, None
    if spec.startswith("https://") and "@" in spec:
        url, _, ref = spec.rpartition("@")
    elif not spec.startswith("https://") and allow_local and not Path(spec).is_dir() and "@" in spec:
        url, _, ref = spec.rpartition("@")
    m = _GITHUB.match(url)
    if m:
        return url, ref or None, f"{m.group(1)}/{m.group(2)}"
    if allow_local and "://" not in url and Path(url).is_dir():
        p = Path(url).resolve()
        return str(p), ref or None, f"local/{p.name}"
    raise ValueError(f"refused: {spec!r} is not an https://github.com/<owner>/<repo> URL")


def _rmtree(path: Path) -> None:
    def onerror(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(path, onerror=onerror)


def fresh_clone(url: str, ref: str | None, dest: Path, timeout: float = 900.0) -> Path:
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError(f"{dest} is not empty")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "-c", "core.autocrlf=false", "-c", "core.longpaths=true", "clone", "--depth", "1"]
    if ref:
        cmd += ["--branch", ref]
    cmd += ["--no-tags", "--single-branch", url, str(dest)]
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"git clone failed: {r.stderr.strip()}")
    return dest


def compile_repo(spec: str, out_root: Path, *, work: Path | None = None, force=False, keep=False,
                 allow_local=False, scan=scan_repository) -> CompileResult:
    url, ref, repo = parse_repo_spec(spec, allow_local)
    base = Path(work) if work else Path(tempfile.mkdtemp(prefix="rw"))
    base.mkdir(parents=True, exist_ok=True)
    holder = Path(tempfile.mkdtemp(prefix="c", dir=base))
    dest = holder / "r"
    try:
        fresh_clone(url, ref, dest)
        return compile_from_clone(dest, out_root, repo, scan=scan, force=force)
    finally:
        if not keep:
            _rmtree(holder)
            if not work:
                _rmtree(base)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo", action="append", default=[])
    ap.add_argument("--repos-file", type=Path)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--allow-local", action="store_true")
    a = ap.parse_args(argv)
    specs = list(a.repo)
    if a.repos_file:
        specs += json.loads(a.repos_file.read_text(encoding="utf-8"))
    if not specs:
        ap.error("no repositories given")
    failed = 0
    for spec in specs:
        try:
            r = compile_repo(spec, a.out, work=a.work, force=a.force, keep=a.keep,
                             allow_local=a.allow_local)
            print(f"{r.repo} {r.commit[:10]} regions={r.regions} rooms={r.rooms} portals={r.portals} "
                  f"unresolved={r.unresolved} seconds={r.seconds:.1f} map_bytes={r.map_bytes}"
                  + (" (reused)" if r.skipped else ""))
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            failed += 1
            print(f"FAILED {spec}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
