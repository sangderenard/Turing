"""[LOADING] Server start-up readiness: bind first, build in the background, say what is ready.

The server used to run every prepare step (the engine laws, the autopilot policy, the page law graph products, the fountain
families) BEFORE it listened, so the first page waited minutes on a cold start.  Now ``server.main`` binds and serves at once and
hands the prepare steps to :class:`Lanes`; each product (``predictor``, ``fountain``, ...) is tracked in :class:`Readiness`, which
``/api/ready`` reports and which gates:

* the page: a game whose products are building shows the loading panel (``web/loading_panel.js``) and starts when they are ready;
* a request for a product FILE that is still building waits for it (``Readiness.wait_for_path``), so it never 404s into a
  broken page.

A product that FAILED is settled too (the page plays without it, as it always did); only ``pending``/``building`` hold anything.
The digest cache stays where it was: each prepare step rebuilds only what its source digest says, so a warm start is quick.
"""
from __future__ import annotations

import fnmatch
import threading
import time
from typing import Callable

PENDING, BUILDING, READY, FAILED, SKIPPED = "pending", "building", "ready", "failed", "skipped"
SETTLED = (READY, FAILED, SKIPPED)

#: the fountain families (a GPU product, then its CPU/wasm fallbacks: ``<name>`` and ``<name>_cpu``), most specific file prefix first
FOUNTAIN_FAMILIES = (
    ("fountain3d_wheel", "fountain3d_wheel"), ("fountain3d_flat", "fountain3d_flat"), ("fountain3d_toy", "fountain3d_toy"), ("fountain3d", "fountain3d"),
    ("fountain_wheel", "fountain_wheel"), ("fountain_purrline", "fountain_purrline"), ("fountain_mobroad", "fountain_mobroad"), ("fountain", "fountain"),
)
FAMILY_NAMES = tuple(name for _prefix, name in FOUNTAIN_FAMILIES)

#: what a game's page needs before it starts.  A list serves every browser; a dict splits by the browser's WebGPU
#: ("gpu": its compute product; "cpu": the wasm fallback).  Games not listed need nothing built: the Meowdoku puzzle, chess, checkers,
#: wordle, purrline, ... are playable the moment the server listens.
GAME_NEEDS = {
    "fountain": {"gpu": ["fountain"], "cpu": ["fountain_cpu"]},
    "fountain3d": {"gpu": ["fountain3d_flat"], "cpu": ["fountain3d_flat_cpu"]},
    "marble_madness": {"gpu": ["fountain3d_toy"], "cpu": ["fountain3d_toy_cpu"]},
    "tipjar": {"gpu": ["fountain3d_flat"], "cpu": ["fountain3d_flat_cpu"]},
    "mobroad": {"gpu": ["fountain_mobroad", "predictor"], "cpu": ["fountain_mobroad_cpu", "predictor"]},
    **{game: ["predictor"] for game in ("space_invaders", "bullet_hell", "platformer", "delve", "side_scroller", "tilequest",
                                         "boardwalk", "worldwalk")},
}


#: what the caption says a product is ("compiling <label>")
LABELS = {
    "predictor": "engine laws", "policy": "autopilot policy", "graph": "page law graph", "autopilot": "autopilot weights",
    "fountain": "fountain", "fountain_cpu": "fountain CPU fallback", "fountain_wheel": "fountain waterwheel",
    "fountain_wheel_cpu": "waterwheel CPU fallback", "fountain_purrline": "purrline sand", "fountain_purrline_cpu": "purrline sand CPU fallback",
    "fountain_mobroad": "mob road crowd", "fountain_mobroad_cpu": "mob road CPU fallback",
    "fountain3d_flat": "3-D fountain", "fountain3d_flat_cpu": "3-D fountain CPU fallback", "fountain3d": "3-D fountain SDF chambers",
    "fountain3d_cpu": "3-D SDF CPU fallback", "fountain3d_wheel": "3-D waterwheel", "fountain3d_wheel_cpu": "3-D waterwheel CPU fallback",
    "fountain3d_toy": "3-D oil toy", "fountain3d_toy_cpu": "3-D oil toy CPU fallback",
}


def label_of(name: str) -> str:
    return LABELS.get(name, name.replace("_", " "))


def product_for_file(name: str, predictor_stems=()) -> str | None:
    """The product a web file belongs to (by its base name), or None for a file no prepare step writes."""
    base = name.rsplit("/", 1)[-1]
    for stem in predictor_stems:
        if base in (f"{stem}.wasm", f"{stem}.abi.json"):
            return "predictor"
    if base.startswith(("policy.", "policy_")):
        return "policy"
    if base.split(".", 1)[0] in ("law", "pilot"):
        return "graph"
    for prefix, family in FOUNTAIN_FAMILIES:
        if base == f"{prefix}.manifest.json" or fnmatch.fnmatchcase(base, f"{prefix}_*"):
            if prefix == "fountain" and fnmatch.fnmatchcase(base, "fountain_wheel*"):
                continue                                         # (the table orders these; the guard is for a future prefix clash)
            cpu = base.endswith((".wasm", ".abi.json"))
            return f"{family}_cpu" if cpu else family
    return None


class Readiness:
    """Per-product build state, thread safe.  Products are declared up front (``pending``) so the page sees what is coming."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, predictor_stems=()):
        self._lock = threading.Condition()
        self._clock = clock
        self.started = clock()
        self.predictor_stems = tuple(predictor_stems)
        self.products: dict[str, dict] = {}
        self.extra: dict = {}                    # other facts /api/ready carries (the loading animation's presence)

    def declare(self, *names: str) -> None:
        with self._lock:
            for name in names:
                self.products.setdefault(name, {"state": PENDING, "detail": "", "progress": None, "started": None, "finished": None})

    def _set(self, name: str, state: str, detail: str = "", progress=None) -> None:
        with self._lock:
            rec = self.products.setdefault(name, {"state": PENDING, "detail": "", "progress": None, "started": None, "finished": None})
            now = self._clock() - self.started
            if state == BUILDING and rec["started"] is None:
                rec["started"] = now
            if state in SETTLED:
                rec["finished"] = now
                if rec["started"] is None:
                    rec["started"] = now
                rec["progress"] = 1.0 if state == READY else rec["progress"]
            elif progress is not None:
                rec["progress"] = progress
            rec["state"], rec["detail"] = state, detail
            self._lock.notify_all()

    def building(self, name: str, detail: str = "", progress=None) -> None:
        self._set(name, BUILDING, detail, progress)

    def ready(self, name: str, detail: str = "") -> None:
        self._set(name, READY, detail)

    def failed(self, name: str, detail: str = "") -> None:
        self._set(name, FAILED, detail)

    def skipped(self, name: str, detail: str = "") -> None:
        self._set(name, SKIPPED, detail)

    def state(self, name: str) -> str:
        with self._lock:
            return self.products[name]["state"] if name in self.products else READY    # an unknown product is nothing to wait for

    def settled(self, names) -> bool:
        return all(self.state(n) in SETTLED for n in names)

    def wait(self, names, timeout: float) -> bool:
        """Block until every named product is settled (ready, failed or skipped); False on timeout."""
        names = list(names)
        deadline = self._clock() + timeout
        with self._lock:
            while not all((self.products[n]["state"] if n in self.products else READY) in SETTLED for n in names):
                left = deadline - self._clock()
                if left <= 0:
                    return False
                self._lock.wait(min(left, 0.25))
        return True

    def wait_for_path(self, path: str, timeout: float) -> bool:
        """A request for the file ``path``: wait for the product that writes it.  True when it may be served (or is no product)."""
        product = product_for_file(path, self.predictor_stems)
        return True if product is None else self.wait([product], timeout)

    def snapshot(self) -> dict:
        with self._lock:
            now = self._clock() - self.started
            products = {}
            for name, rec in self.products.items():
                out = {"state": rec["state"], "label": label_of(name)}
                if rec["detail"]:
                    out["detail"] = rec["detail"]
                if rec["progress"] is not None:
                    out["progress"] = round(float(rec["progress"]), 3)
                if rec["started"] is not None:
                    out["seconds"] = round((rec["finished"] if rec["finished"] is not None else now) - rec["started"], 2)
                products[name] = out
            # WHAT is compiling, in the order the lanes run it: the building products first, then the pending ones
            order = sorted((n for n, r in self.products.items() if r["state"] in (BUILDING, PENDING)),
                           key=lambda n: (self.products[n]["state"] != BUILDING, n))
            compiling = [{"name": n, "label": label_of(n), "state": self.products[n]["state"]} for n in order]
            return {"uptime": round(now, 2), "compiling": compiling, "all_settled": all(r["state"] in SETTLED for r in self.products.values()),
                    "products": products, "needs": GAME_NEEDS, **self.extra}


class Lanes:
    """Run prepare steps in background daemon threads, one thread per lane, steps of a lane in order.

    A step is ``(products, function)``; ``function(ctx)`` may call ``ctx.ready(name)`` as parts of the work land (the fountain's
    GPU products before its wasm fallbacks).  Whatever the step did not mark is marked ready when it returns, failed when it raises
    (the exception is printed; the page plays without that product, as before)."""

    class Context:
        def __init__(self, readiness: Readiness, products):
            self.readiness, self.products = readiness, tuple(products)

        def building(self, name, detail="", progress=None):
            self.readiness.building(name, detail, progress)

        def ready(self, name, detail=""):
            self.readiness.ready(name, detail)

        def failed(self, name, detail=""):
            self.readiness.failed(name, detail)

        def skipped(self, name, detail=""):
            self.readiness.skipped(name, detail)

    def __init__(self, readiness: Readiness, log=print):
        self.readiness, self.log = readiness, log
        self.threads: list[threading.Thread] = []

    def declare(self, lanes: dict[str, list[tuple]]) -> None:
        for steps in lanes.values():
            for products, _fn in steps:
                self.readiness.declare(*products)

    def start(self, lanes: dict[str, list[tuple]]) -> list[threading.Thread]:
        self.declare(lanes)
        for name, steps in lanes.items():
            thread = threading.Thread(target=self._run, args=(name, steps), name=f"prepare-{name}", daemon=True)
            thread.start()
            self.threads.append(thread)
        return self.threads

    def _run(self, lane: str, steps: list[tuple]) -> None:
        for products, fn in steps:
            ctx = Lanes.Context(self.readiness, products)
            for product in products:
                self.readiness.building(product)
            try:
                fn(ctx)
            except BaseException as error:               # a failed step settles its products: nothing waits on it for ever
                self.log(f"[meowdoku] prepare step {products} failed ({type(error).__name__}: {error})")
                for product in products:
                    if self.readiness.state(product) not in SETTLED:
                        self.readiness.failed(product, f"{type(error).__name__}: {error}"[:300])
                continue
            for product in products:
                if self.readiness.state(product) not in SETTLED:
                    self.readiness.ready(product)
