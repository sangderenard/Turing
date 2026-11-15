"""Train a small Llama LoRA on closed, successful emoji-target attempts.

Each example conditions on a randomly assigned target emoji. Only a reaction by
another account in the ten-minute window is a positive example. Checkpoints are
versioned, converted to GGUF, and published only after conversion succeeds.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time


def examples(db, limit):
    return db.execute("SELECT id,prompt,response,target_kind FROM attempts "
                      "WHERE reward=1 AND trained_at IS NULL ORDER BY id LIMIT ?", (limit,)).fetchall()


def encoded(tokenizer, prompt, response, target, device):
    system = ("You are Llama, an intern account in SpeakToMe. Participate plainly and courteously. "
              f"Naturally invite a {target} emoji reaction without asking for reactions directly. "
              "Use only the supplied chat logs.")
    prefix = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    full = prefix + [{"role": "assistant", "content": response}]
    beginning = tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=True)
    whole = tokenizer.apply_chat_template(full, tokenize=True, add_generation_prompt=False)
    # Transformers 5 returns BatchEncoding; older releases return token-id lists.
    if not isinstance(beginning, list):
        beginning = beginning["input_ids"]
    if not isinstance(whole, list):
        whole = whole["input_ids"]
    if len(whole) > 768:
        return None
    import torch
    ids = torch.tensor([whole], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[:, :len(beginning)] = -100
    if not torch.any(labels != -100):
        return None
    return ids, labels


def train(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training is unavailable")
    db = sqlite3.connect(args.db, timeout=10, isolation_level=None)
    rows = examples(db, args.limit)
    if not rows:
        print("No new positive reactions; weights unchanged.")
        return 0
    root = Path(args.adapter_root)
    root.mkdir(parents=True, exist_ok=True)
    pointer = root / "current.txt"
    previous = Path(pointer.read_text(encoding="utf-8").strip()) if pointer.is_file() else None
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True, dtype=torch.float16,
                                                 attn_implementation="sdpa", device_map="cuda:0")
    if previous and previous.is_dir():
        model = PeftModel.from_pretrained(base, previous, is_trainable=True)
    else:
        config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
                            target_modules=["q_proj", "v_proj", "o_proj"])
        model = get_peft_model(base, config)
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=8e-5)
    completed = []
    for item_id, prompt, response, target in rows:
        sample = encoded(tokenizer, prompt, response, target, "cuda:0")
        if sample is None:
            continue
        ids, labels = sample
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=labels).loss
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        completed.append(item_id)
        print(f"Trained attempt {item_id}: loss {float(loss.detach()):.4f}", flush=True)
    if not completed:
        print("All new examples exceeded the token limit; weights unchanged.")
        return 0
    checkpoint = root / f"step-{completed[-1]:09d}-{time.time_ns()}"
    if checkpoint.exists():
        raise RuntimeError("checkpoint already exists; refusing to overwrite")
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    del model, base, optimizer
    torch.cuda.empty_cache()
    candidate = Path(args.gguf_out).with_suffix(".new.gguf")
    if candidate.exists():
        candidate.unlink()
    env = dict(os.environ)
    env.pop("NO_LOCAL_GGUF", None)  # use the converter's pinned gguf-py, not an unrelated installed release
    command = [sys.executable, args.converter, "--base", args.base, "--outfile", str(candidate),
               "--outtype", "f16", str(checkpoint)]
    subprocess.run(command, check=True, env=env, timeout=600)
    if not candidate.is_file() or candidate.stat().st_size < 1024:
        raise RuntimeError("LoRA conversion produced no adapter")
    os.replace(candidate, args.gguf_out)
    temporary = pointer.with_suffix(".new")
    temporary.write_text(str(checkpoint), encoding="utf-8")
    os.replace(temporary, pointer)
    db.executemany("UPDATE attempts SET trained_at=? WHERE id=?",
                   [(time.time(), item_id) for item_id in completed])
    print(f"Published adapter {args.gguf_out} from {len(completed)} positive attempts.", flush=True)
    return len(completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter-root", required=True)
    parser.add_argument("--converter", required=True)
    parser.add_argument("--gguf-out", required=True)
    parser.add_argument("--limit", type=int, default=8)
    return train(parser.parse_args())


if __name__ == "__main__":
    main()
