"""[BASELINE] The hosted Llama runs the base model: no LoRA loaded and no training unless explicitly asked for."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "examples" / "meowdoku" / "deploy" / "hosted_llm_runner.py"
sys.path[:0] = [str(ROOT), str(ROOT / "examples"), r"C:\dev\Powershell\speaktome"]


def _runner():
    spec = importlib.util.spec_from_file_location("hosted_llm_runner_t", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_lora_and_no_training_by_default(tmp_path, monkeypatch):
    runner = _runner()
    adapter = tmp_path / "adapter.gguf"
    adapter.write_bytes(b"lora")                         # an adapter on disk is not enough to load it
    seen = {}

    class Proc:
        def poll(self):
            raise RuntimeError("stop here")

    monkeypatch.setattr(runner.subprocess, "Popen", lambda args, **kw: seen.setdefault("args", args) and Proc())
    try:
        runner.LocalModel(Path("llama-server.exe"), Path("m.gguf"), None, 1, tmp_path).start()
    except RuntimeError:
        pass
    assert "--lora" not in seen["args"]
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: calls.append(a))
    runner.train_due(SimpleNamespace(train=False), None, None)
    assert calls == [], "no training without --train"


def test_both_are_off_unless_asked_for():
    text = RUNNER.read_text(encoding="utf-8")
    assert '"--use-adapter", action="store_true"' in text and '"--train", action="store_true"' in text
    launcher = (RUNNER.parent / "Run-HostedLlm.ps1").read_text(encoding="utf-8")
    assert "--use-adapter" not in launcher and "--train " not in launcher, "the launcher runs the baseline"
