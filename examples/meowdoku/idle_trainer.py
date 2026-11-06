"""The background trainer: studies recorded play whenever the server is idle.

Play is recorded as it happens, as SPANS (``autopilot.PlayTrace``): human play and the autopilot's own
(demo and inference) play, for every game with an autopilot, whether or not a game ever ends.  This
trainer, ON by default in ``server.py`` (``MEOWDOKU_IDLE_TRAIN=0`` or ``--no-idle-train`` turns it off),
runs in ONE low-priority background thread (its fit steps run in the trainer PROCESS of
``autopilot._trainer_pool``, below normal priority) and only while no realtime game has ticked for
``idle_after`` seconds.  It works in short chunks and looks at the clock between them, so a tick pauses it
within one chunk; the pass it was in is kept and resumes at the next idle moment.

A pass, for one game that has new spans (or has no weights yet but has teacher data):

1. SCORE: every stored span's loss under the CURRENT weights (a span the policy already imitates adds
   nothing; one where the human did what it would not is where the loss landscape moves), written back
   into the store.
2. WEIGHT, per span: ``(loss + eps) ** alpha`` (prioritised replay) x its source x its outcome:
   * human spans: ``HUMAN_WEIGHT`` x the outcome tag's worth (a room, a kill, a pickup, a score; damage
     is worth little);
   * the autopilot's own spans: ``SELF_WEIGHT`` (below the human's) x an ADVANTAGE-WEIGHTED factor:
     ``exp((return - baseline) / (beta * scale))``, clipped, from what followed the span (score,
     rooms, damage) against a running (exponential) baseline per game.  A bad span is down-weighted
     to nearly nothing; the imitation loss has no negative-example term, so it is never pushed away from;
   * redundant spans (loss ~ 0) are down-weighted hard.
3. FIT: minibatches drawn by those weights (the repo's ``PolicyTrainer``: autograd + Adam).
4. GUARD: the imitation loss on held-out spans (human, teacher and the good self-play) before vs after;
   the new weights are adopted only if NOT WORSE.  The previous weights are kept as ``<weights>.prev.npz``;
   the new ones are saved atomically and handed to the live pilots in place (``autopilot.adopt``: no
   restart, no recompile; an in-law pilot reads them at its next parameter hand-over).
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from meowdoku import autopilot as ap
from meowdoku import pilot_decl as pd

HUMAN_WEIGHT = float(os.environ.get("MEOWDOKU_HUMAN_WEIGHT", "1.0"))
SELF_WEIGHT = float(os.environ.get("MEOWDOKU_SELF_WEIGHT", "0.35"))      # below the human's by default
TEACHER_WEIGHT = 0.5
ALPHA, EPS, BETA = 0.7, 1e-3, 1.0
BASELINE_DECAY = 0.9
REDUNDANT = 2e-3                     # a span whose loss is below this adds almost nothing
BATCH = 128


# ------------------------------------------------------------------ the trainer process
_WORK: dict = {}


def low_priority() -> None:
    """Below-normal priority for the trainer process (Windows); a no-op elsewhere."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:  # noqa: BLE001
        try:
            os.nice(10)
        except Exception:  # noqa: BLE001
            pass


def fit_steps(key: str, params: dict | None, x, y, p, steps: int, seed: int = 0, lr: float = 1e-3):
    """Run ``steps`` minibatch steps on the trainer kept under ``key`` (created from ``params`` + the data on
    the first call); batches are drawn with probabilities ``p`` over the rows.  Returns (mean loss, params)."""
    if params is not None:
        _WORK[key] = {"trainer": ap.PolicyTrainer(params, lr=lr), "x": x, "y": y, "p": p, "rng": np.random.default_rng(seed)}
    w = _WORK[key]
    loss = 0.0
    for _ in range(steps):
        rows = w["rng"].choice(len(w["x"]), size=min(BATCH, len(w["x"])), p=w["p"])
        loss += w["trainer"].step(w["x"][rows], w["y"][rows])
    return loss / max(1, steps), w["trainer"].params()


def drop_work(key: str) -> None:
    _WORK.pop(key, None)


# ------------------------------------------------------------------ scoring
_BATCH_FWD = 64


def forward_rows(params: dict, x: np.ndarray) -> np.ndarray:
    """The policy's outputs for many rows: the NumPy PRODUCT (the traced, gradient-free forward) in
    fixed batches -- never an eager tape."""
    f, o = ap.widths_of(params)
    run = ap.numpy_product(_BATCH_FWD, f, o, arch=ap.arch_key(params), like=params)
    out = []
    for at in range(0, len(x), _BATCH_FWD):
        chunk = x[at:at + _BATCH_FWD]
        if len(chunk) < _BATCH_FWD:
            chunk = np.concatenate([chunk, np.zeros((_BATCH_FWD - len(chunk), f))])
        out.append(run(chunk, params))
    return np.concatenate(out)[:len(x)]


def row_loss(params: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((forward_rows(params, x) - y) ** 2).mean(axis=1)


def span_means(values: np.ndarray, span: np.ndarray) -> dict[int, float]:
    ids, inverse = np.unique(span, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    return {int(i): float(s / c) for i, s, c in zip(ids, sums, counts)}


def advantages(rec: dict) -> dict[int, float]:
    """Per self-play span: (return - running baseline) / scale.  The baseline is an exponential average of
    the spans' returns in the order they happened (span ids grow with time), the scale their spread."""
    ids, first = np.unique(rec["span"], return_index=True)
    returns = [float(rec["ret"][f]) for f in first]
    baseline, out = None, {}
    scale = float(np.std(returns)) + 1e-6 if len(returns) > 1 else 1.0
    for sid, r in zip(ids, returns):
        baseline = r if baseline is None else baseline
        out[int(sid)] = (r - baseline) / scale
        baseline = BASELINE_DECAY * baseline + (1.0 - BASELINE_DECAY) * r
    return out


def outcome_weight(adv: float) -> float:
    """Advantage-weighted imitation: exp(adv / beta), clipped; a bad span goes to ~0, never negative."""
    return float(np.clip(np.exp(adv / BETA), 0.02, 3.0))


# ------------------------------------------------------------------ the trainer
class IdleTrainer:
    def __init__(self, games=None, idle_after: float = 3.0, chunk_steps: int = 6, steps_per_pass: int = 120,
                 remote: bool = True, yield_s: float = 0.01, clock=time.monotonic, log=print,
                 human_weight: float | None = None, self_weight: float | None = None, headless_guard=None, lr: float = 1e-3):
        self.games = tuple(games) if games else tuple(ap.GAMES)
        self.idle_after, self.chunk_steps, self.steps_per_pass = idle_after, chunk_steps, steps_per_pass
        self.remote, self.yield_s, self.clock, self.log = remote, yield_s, clock, log
        self.human_weight = HUMAN_WEIGHT if human_weight is None else human_weight
        self.self_weight = SELF_WEIGHT if self_weight is None else self_weight
        self.last_tick = -1e9
        self.passes: dict[str, dict] = {}                 # a pass in progress per game (it survives a pause)
        self.results: list[dict] = []
        self.state_path = ap.BUILD / "idle_state.json"
        self.seen = self._load_state()
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.paused_count = 0
        self.lr = lr
        #: optional extra guard: (game, old params, new params) -> bool, e.g. a short headless score check
        self.headless_guard = headless_guard

    # ---------------------------------------------------------- clock
    def touch(self) -> None:
        """A realtime game ticked (or a player acted): play is on, training waits."""
        self.last_tick = self.clock()

    def idle(self) -> bool:
        return self.clock() - self.last_tick >= self.idle_after

    # ---------------------------------------------------------- state
    def _load_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_state(self) -> None:
        try:
            ap.BUILD.mkdir(parents=True, exist_ok=True)
            partial = self.state_path.with_suffix(".partial.json")
            partial.write_text(json.dumps(self.seen), encoding="utf-8")
            partial.replace(self.state_path)
        except OSError:
            pass

    def _counts(self, name: str) -> dict:
        out = {}
        for source in ("human", "selfplay"):
            rec = ap.load_records(name, source)
            out[source] = int(len(rec["x"])) if rec is not None else 0
        return out

    def has_work(self, name: str) -> bool:
        if name in self.passes:
            return True
        counts, seen = self._counts(name), self.seen.get(name, {})
        if counts["human"] > seen.get("human", 0) or counts["selfplay"] > seen.get("selfplay", 0):
            return True
        d = ap.declared(name)
        return bool(d is not None and ap.fitting_params(name) is None and (ap.BUILD / f"teacher_{name}.npz").exists())

    # ---------------------------------------------------------- the dataset of a pass
    def _params(self, name: str):
        params = ap.live_params(name)
        if params is None and ap.declared(name) is not None:
            d = ap.declared(name)
            params = d.init_params(pd.arch_for(name))
        return params

    def build(self, name: str, params: dict) -> dict | None:
        """The rows, their sampling probabilities and the held-out validation rows of one pass."""
        arch = ap.arch_of(params)
        d = ap.declared(name)
        cut = (lambda x: d.slice(arch, x)) if d is not None else (lambda x: x)
        parts = []                                                # (x, y, span key, base weight, source)
        report = {"human_spans": 0, "self_spans": 0, "teacher_rows": 0, "scored": 0}
        val_x, val_y = [], []
        human = ap.load_records(name, "human")
        if human is not None and len(human["x"]):
            x = cut(human["x"])
            loss = row_loss(params, x, human["y"])
            per = span_means(loss, human["span"])
            self._write_losses(name, "human", per)
            report["scored"] += len(per)
            for sid, rows in _groups(human["span"]):
                tag = int(human["tags"][rows[0]])
                w = self.human_weight * ap.tag_weight(tag) * (per[sid] + EPS) ** ALPHA
                w *= 0.1 if per[sid] < REDUNDANT else 1.0
                if sid % 7 == 0:                                   # held out
                    val_x.append(x[rows]); val_y.append(human["y"][rows])
                else:
                    parts.append((x[rows], human["y"][rows], w))
            report["human_spans"] = len(per)
        own = ap.load_records(name, "selfplay")
        if own is not None and len(own["x"]):
            x = cut(own["x"])
            loss = row_loss(params, x, own["y"])
            per = span_means(loss, own["span"])
            self._write_losses(name, "selfplay", per)
            adv = advantages(own)
            for sid, rows in _groups(own["span"]):
                if sid % 7 == 0 and adv[sid] > 0.0:                # a held-out GOOD span
                    val_x.append(x[rows]); val_y.append(own["y"][rows])
                    continue
                w = self.self_weight * outcome_weight(adv[sid]) * (per[sid] + EPS) ** ALPHA
                w *= 0.1 if per[sid] < REDUNDANT else 1.0
                parts.append((x[rows], own["y"][rows], w))
            report["self_spans"] = len(per)
            report["self_adv_mean"] = float(np.mean(list(adv.values())))
        t_path = ap.BUILD / f"teacher_{name}.npz"
        if d is not None and t_path.exists():
            t = np.load(t_path)
            tx, ty = cut(t["x"]), t["y"]
            n = len(tx)
            held = np.arange(n) % 10 == 0
            val_x.append(tx[held]); val_y.append(ty[held])
            tx, ty = tx[~held], ty[~held]
            # the teacher's rows in blocks (as spans), each at the base weight: it is the floor of the data
            for at in range(0, len(tx), ap.SPAN_ROWS):
                rows = slice(at, at + ap.SPAN_ROWS)
                parts.append((tx[rows], ty[rows], TEACHER_WEIGHT))
            report["teacher_rows"] = int(len(tx))
        if not parts:
            return None
        x = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        prob = np.concatenate([np.full(len(p[0]), p[2] / len(p[0])) for p in parts])
        prob = prob / prob.sum()
        vx = np.concatenate(val_x) if val_x else x[:0]
        vy = np.concatenate(val_y) if val_y else y[:0]
        return {"x": x, "y": y, "p": prob, "val_x": vx, "val_y": vy, "report": report}

    def _write_losses(self, name: str, source: str, per: dict[int, float]) -> None:
        """The scored loss goes back into the store (rows added since stay unscored)."""
        with ap.LEARN_LOCK:
            rec = ap.load_records(name, source)
            if rec is None:
                return
            for sid, v in per.items():
                rec["loss"][rec["span"] == sid] = v
            ap.save_records(name, rec, source)

    # ---------------------------------------------------------- a pass
    def val_loss(self, params: dict, data: dict) -> float:
        if not len(data["val_x"]):
            return float("nan")
        return float(row_loss(params, data["val_x"], data["val_y"]).mean())

    def train_game(self, name: str) -> dict:
        """One pass (or the rest of one paused earlier).  Returns its record: ``status`` is ``adopted``,
        ``kept`` (not better: the old weights stand), ``paused`` (play resumed) or ``nodata``."""
        if name not in self.passes:
            params = self._params(name)
            if params is None:
                return {"game": name, "status": "nodata"}
            data = self.build(name, params)
            if data is None:
                self.seen[name] = self._counts(name)
                return {"game": name, "status": "nodata"}
            self.passes[name] = {"params": params, "data": data, "done": 0, "before": self.val_loss(params, data),
                                 "counts": self._counts(name), "loss": float("nan"), "fitted": None, "started": self.clock()}
        ps = self.passes[name]
        data = ps["data"]
        key = f"{os.getpid()}:{name}"
        ap.TRAINING[name] = True                                    # the page's learn panel shows a pass running
        while ps["done"] < self.steps_per_pass:
            if not self.idle():                                     # play resumed: stop NOW, keep the pass
                self.paused_count += 1
                ap.TRAINING[name] = False
                return {"game": name, "status": "paused", "steps": ps["done"]}
            first = ps["done"] == 0
            args = (key, {n: v.copy() for n, v in ps["params"].items()} if first else None,
                    data["x"] if first else None, data["y"] if first else None, data["p"] if first else None, self.chunk_steps)
            loss, fitted = self._submit(*args)
            ps["loss"], ps["fitted"] = loss, fitted
            ps["done"] += self.chunk_steps
            time.sleep(self.yield_s)                                # yield between batches
        if self.remote:
            ap._trainer_pool().submit(drop_work, key)
        else:
            drop_work(key)
        fitted = dict(ps["fitted"])
        if "variant" in ps["params"]:
            fitted["variant"] = ps["params"]["variant"]
        for n in ap.param_names(ps["params"]):
            if n in ("GM1", "GM2"):
                fitted[n] = ps["params"][n]                         # a graph's masks are fixed
        after = self.val_loss(fitted, data)
        before = ps["before"]
        better = bool(np.isnan(before) or after <= before)         # the guard: adopt only if NOT WORSE
        if better and self.headless_guard is not None:
            better = bool(self.headless_guard(name, ps["params"], fitted))
        record = {"game": name, "arch": ap.arch_of(ps["params"]), "steps": ps["done"], "rows": int(len(data["x"])),
                  "val_before": before, "val_after": after, "train_loss": ps["loss"], **data["report"]}
        if better:
            path = ap.weights_path(name)
            if path.exists():
                shutil.copyfile(path, path.with_suffix(".prev.npz"))
            ap.save_params(fitted, path)                            # atomic (partial + replace)
            ap.adopt(name, fitted)
            record["status"] = "adopted"
        else:
            record["status"] = "kept"
        del self.passes[name]
        self.seen[name] = ps["counts"]
        self._save_state()
        self.results.append(record)
        ap.TRAINING[name] = False
        self._announce(name, record)
        self.log(f"[idle-trainer] {name} {record['arch']}: {record['rows']} rows "
                 f"(human spans {data['report']['human_spans']}, self {data['report']['self_spans']}, teacher {data['report']['teacher_rows']}) "
                 f"val loss {before:.4f} -> {after:.4f} {record['status']}")
        return record

    def _announce(self, name: str, record: dict) -> None:
        """The pass on the page: the same learn status and history the post-game trainer writes, so the
        learn panel flashes, charts the loss and names the pass (adopted or kept)."""
        rep = record
        label = (f"idle pass ({record['status']}): {rep.get('human_spans', 0)} of your spans, "
                 f"{rep.get('self_spans', 0)} self-play, val {record['val_before']:.4f} -> {record['val_after']:.4f}")
        ap.LEARN_STATUS[name] = {"last": label, "rows": record["rows"], "loss": round(float(record["val_after"]), 4),
                                 "at": time.strftime("%H:%M:%S"), "idle_t": time.time()}
        ap._history(name, kind="trained", loss=round(float(record["val_after"]), 5), rows=record["rows"], label=label)
        try:
            ap._save_learning()
        except Exception:  # noqa: BLE001 - the record is a convenience
            pass

    def _submit(self, key, params, x, y, p, steps):
        if self.remote:
            return ap._trainer_pool().submit(fit_steps, key, params, x, y, p, steps, 0, self.lr).result()
        return fit_steps(key, params, x, y, p, steps, 0, self.lr)

    # ---------------------------------------------------------- the loop
    def cycle(self) -> int:
        """One round over the games with work; returns how many passes were worked on."""
        worked = 0
        for name in self.games:
            if not self.idle():
                break
            if self.has_work(name):
                try:
                    self.train_game(name)
                except Exception as error:  # noqa: BLE001 - learning must never take the server down
                    self.log(f"[idle-trainer] {name}: pass failed: {type(error).__name__}: {str(error)[:120]}")
                    self.passes.pop(name, None)
                    ap.TRAINING[name] = False
                    self.seen[name] = self._counts(name)
                worked += 1
        return worked

    def run(self) -> None:
        while not self._stop.is_set():
            if not self.idle():
                self._stop.wait(0.25)
                continue
            if not self.cycle():
                self._stop.wait(2.0)

    def start(self) -> "IdleTrainer":
        ap.IDLE_TRAINER = self
        self.thread = threading.Thread(target=self.run, name="idle-trainer", daemon=True)
        self.thread.start()
        self.log(f"[idle-trainer] on: trains after {self.idle_after:.0f}s without play (MEOWDOKU_IDLE_TRAIN=0 turns it off); "
                 f"human weight {self.human_weight}, self-play weight {self.self_weight}")
        return self

    def stop(self) -> None:
        self._stop.set()
        if ap.IDLE_TRAINER is self:
            ap.IDLE_TRAINER = None


def _groups(span: np.ndarray):
    ids, inverse = np.unique(span, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.cumsum(np.bincount(inverse))
    start = 0
    for i, sid in enumerate(ids):
        yield int(sid), order[start:bounds[i]]
        start = bounds[i]


TRAINER: IdleTrainer | None = None


def touch() -> None:
    if TRAINER is not None:
        TRAINER.touch()


def enabled() -> bool:
    return os.environ.get("MEOWDOKU_IDLE_TRAIN", "1") != "0"


def start_default(log=print) -> IdleTrainer | None:
    global TRAINER
    if not enabled():
        return None
    TRAINER = IdleTrainer(log=log).start()
    return TRAINER
