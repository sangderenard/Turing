"""The color game's learning network, trained continuously and natively.

One compiled schedule (``palette_native_training.compile_palette_trainer``:
the repo compiler's ``compile_native_training_schedule`` on the LLVM lane)
holds two parameter groups.  ``shared`` is the network everyone's responses
train; it is the site's default.  ``personal`` is one participant's linear
layer, weighted ``ALPHA`` in every score and pulled toward zero, so it only
nudges the shared default.  The shared parameters live in ``shared.npz`` in
the model directory; each personal layer lives with its owner in SQLite.

Training is continuous: every ranking or slider score queues its owner, and
one worker thread runs native steps over the newest responses -- the shared
group on everyone's, then that owner's personal group on theirs.  Scores come
from the same compiled function with the learning rate at zero, so the model
that proposes a design is the one that was trained.  Until the schedule has
compiled (about a minute, once per model directory) ``ready`` is False and
the game offers random designs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np

from . import palette_designs as designs
from . import palette_native_training as trainer

SHARED_STEPS = 8         # native steps per training pass for the shared network
PERSONAL_STEPS = 8       # native steps per pass for the owner's personal layer
SHARED_RATE = 0.5
PERSONAL_RATE = 0.5
DIRECTORY_ENVIRONMENT = "MEOWDOKU_PALETTE_MODEL_DIR"


def model_directory(accounts_db) -> Path:
    """Where the compiled schedule and shared parameters live: beside the
    accounts database, unless ``MEOWDOKU_PALETTE_MODEL_DIR`` names one."""

    configured = os.environ.get(DIRECTORY_ENVIRONMENT)
    return Path(configured) if configured else Path(accounts_db).parent / "palette_model"


class PaletteModel:
    """The compiled network, its parameters and the training worker."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.lock = threading.RLock()
        self.schedule = None
        self.ids = None
        self.shared = None
        self.shared_steps = 0
        self.error = None
        self.pending: "queue.Queue[tuple | None]" = queue.Queue()
        self.compiled = threading.Event()
        threading.Thread(target=self._compile, name="palette-compile", daemon=True).start()
        threading.Thread(target=self._train_forever, name="palette-train", daemon=True).start()

    # -- compilation and parameters -------------------------------------
    def _compile(self) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            schedule_path = self.directory / "schedule.json"
            ids_path = self.directory / "ids.json"
            # The schedule is compiled for one layout of the network (the bindings' names and shapes); a schedule
            # compiled for another layout (an earlier design's features) is compiled again.
            layout_path = self.directory / "layout.json"
            layout = [[name, list(shape)] for name, shape in trainer.BINDINGS]
            same_layout = (layout_path.is_file()
                           and json.loads(layout_path.read_text(encoding="utf-8")) == layout)
            if same_layout and schedule_path.is_file() and ids_path.is_file():
                schedule = trainer.load_training_schedule(schedule_path)
                ids = json.loads(ids_path.read_text(encoding="utf-8"))
            else:
                schedule, ids = trainer.compile_palette_trainer(self.directory)
                ids_path.write_text(json.dumps(ids), encoding="utf-8")
                layout_path.write_text(json.dumps(layout), encoding="utf-8")
            shared = self._load_shared(ids)
            with self.lock:
                self.schedule, self.ids, self.shared = schedule, ids, shared
        except Exception as error:          # reported through ``status``; the game stays usable
            self.error = f"{type(error).__name__}: {error}"
        finally:
            self.compiled.set()

    def _load_shared(self, ids):
        path = self.directory / "shared.npz"
        buffers = trainer.initial_parameters(ids)
        if path.is_file():
            with np.load(path) as stored:
                values = {name: np.asarray(stored[name], dtype=np.float64) for name in trainer.SHARED if name in stored}
                # Parameters of another layout are not this network's: it starts from its initial parameters.
                if all(name in values and values[name].shape == buffers[ids[name]].shape for name in trainer.SHARED):
                    for name in trainer.SHARED:
                        buffers[ids[name]] = values[name]
                    self.shared_steps = int(stored["steps"]) if "steps" in stored else 0
        return {name: buffers[ids[name]] for name in trainer.SHARED}

    def _save_shared(self) -> None:
        temporary = self.directory / "shared.tmp.npz"
        np.savez(temporary, steps=np.asarray(self.shared_steps), **self.shared)
        temporary.replace(self.directory / "shared.npz")

    @property
    def ready(self) -> bool:
        return self.schedule is not None

    def status(self) -> dict:
        state = ("ready" if self.ready else "unavailable" if self.error
                 else "compiling")
        return {"state": state, "shared_steps": int(self.shared_steps),
                **({"error": self.error} if self.error else {})}

    # -- buffers ----------------------------------------------------------
    def _buffers(self, personal):
        ids = self.ids
        buffers = trainer.initial_parameters(ids)
        for name, value in self.shared.items():
            buffers[ids[name]] = value.copy()
        # A personal layer of another layout (an earlier design's features) starts again from zero.
        buffers[ids["personal.u"]] = (
            np.asarray(personal, dtype=np.float64).reshape(designs.FEATURES, 1)
            if personal is not None and len(personal) == designs.FEATURES
            else np.zeros((designs.FEATURES, 1))
        )
        trainer.feedback_batch(buffers, ids)
        return buffers

    # -- scoring ----------------------------------------------------------
    def score(self, rows, personal=None) -> list[float] | None:
        """Network scores for feature rows; ``personal`` None scores the
        shared default alone.  None while the schedule is not ready."""

        if not self.ready or not rows:
            return None
        with self.lock:
            buffers = self._buffers(personal)
            scores: list[float] = []
            for start in range(0, len(rows), trainer.RATINGS):
                chunk = rows[start:start + trainer.RATINGS]
                block = np.zeros((trainer.RATINGS, designs.FEATURES))
                block[:len(chunk)] = np.asarray(chunk, dtype=np.float64)
                buffers[self.ids["rated"]] = block
                execution = trainer.run_parameter_group(
                    self.schedule, "personal", buffers, steps=1, learning_rate=0.0,
                )
                values = np.asarray(
                    execution.buffers[self.schedule.outputs["scores"]], dtype=np.float64,
                ).reshape(trainer.RATINGS)
                scores.extend(float(value) for value in values[:len(chunk)])
            return scores

    # -- continuous training ----------------------------------------------
    def learn(self, owner: str, responses, store_personal) -> None:
        """Queue one training pass for ``owner``'s new response.

        ``responses(owner)`` returns ``(pairs, ratings, everyone_pairs,
        everyone_ratings, personal)`` as feature rows from the game's tables;
        ``store_personal(owner, weights, steps)`` keeps the trained layer.
        Both name the database of the server that queued the pass."""
        self.pending.put((owner, responses, store_personal))

    def drain(self, timeout: float = 120.0) -> bool:
        """Block until every queued pass has run (tests and shutdown)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.pending.unfinished_tasks == 0:
                return True
            time.sleep(0.05)
        return False

    def _train_forever(self) -> None:
        self.compiled.wait()
        while True:
            job = self.pending.get()
            try:
                if job is not None and self.ready:
                    self.train(*job)
            except Exception as error:      # one bad pass must not stop learning
                self.error = f"{type(error).__name__}: {error}"
            finally:
                self.pending.task_done()

    def train(self, owner: str, responses, store_personal) -> None:
        pairs, ratings, everyone_pairs, everyone_ratings, personal = responses(owner)
        trained = None
        with self.lock:
            if everyone_pairs or everyone_ratings:
                buffers = self._buffers(None)
                trainer.feedback_batch(
                    buffers, self.ids,
                    pairs=everyone_pairs[-trainer.PAIRS:],
                    ratings=everyone_ratings[-trainer.RATINGS:],
                )
                trainer.run_parameter_group(
                    self.schedule, "shared", buffers,
                    steps=SHARED_STEPS, learning_rate=SHARED_RATE,
                )
                self.shared = {name: buffers[self.ids[name]].copy() for name in trainer.SHARED}
                self.shared_steps += SHARED_STEPS
                self._save_shared()
            if pairs or ratings:
                buffers = self._buffers(personal)
                trainer.feedback_batch(
                    buffers, self.ids,
                    pairs=pairs[-trainer.PAIRS:], ratings=ratings[-trainer.RATINGS:],
                )
                trainer.run_parameter_group(
                    self.schedule, "personal", buffers,
                    steps=PERSONAL_STEPS, learning_rate=PERSONAL_RATE,
                )
                trained = buffers[self.ids["personal.u"]].reshape(-1).tolist()
        # Written after the model lock is released: a request scoring designs
        # holds the database only after it has finished with the model.
        if trained is not None:
            store_personal(owner, trained, PERSONAL_STEPS)


_MODELS: dict[str, PaletteModel] = {}
_MODELS_LOCK = threading.Lock()


def model_for(accounts_db) -> PaletteModel:
    """The one model of a model directory (started on first use)."""

    directory = model_directory(accounts_db)
    key = str(directory.resolve())
    with _MODELS_LOCK:
        model = _MODELS.get(key)
        if model is None:
            model = _MODELS[key] = PaletteModel(directory)
        return model
