"""Byte-guarded audio-only repair of the preserved newer game; never publishes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE_SHA256 = "2cd5c5e23f1491243346dd46586f5ebc10a4391187a4e991fa8f151f8dfd4ef4"
OLD_CALL = "      this.bank.set(descriptor.identity,{instance,views,output,run:()=>instance.exports[descriptor.entrypoint](this.blockSize,...inputOffsets,outputOffset)});}"
OLD_DESCRIPTOR = "      reservedBytes:kernel.reserved_bytes,binary:base64Bytes(kernel.binary_base64)})),profile=engineSoundProfileForPreset("
NEW_DESCRIPTOR = "      reservedBytes:kernel.reserved_bytes,inputNames:kernel.input_names,binary:base64Bytes(kernel.binary_base64)})),profile=engineSoundProfileForPreset("


def repair(source: bytes) -> tuple[bytes, dict]:
    if hashlib.sha256(source).hexdigest() != BASE_SHA256:
        raise ValueError("engine audio repair requires exact preserved newer base")
    from src.compiler.engine_pcm_wasm import engine_pcm_kernel_bank_model
    from src.compiler.abstract_ui_div_map import DIV_MAP_JAVASCRIPT
    text = source.decode("utf-8")
    marker = '"engine_audio":'
    if text.count(marker) != 1:
        raise ValueError("engine audio model is not unique")
    start = text.index(marker) + len(marker)
    original, end = json.JSONDecoder().raw_decode(text[start:])
    current = engine_pcm_kernel_bank_model()
    # Changing feed names changes the ABI description, not a single opcode.
    # Never attach new names to a binary that was not proved byte-identical.
    old_kernels = {k["identity"]: k for k in original["kernels"]}
    if set(old_kernels) != {k["identity"] for k in current["kernels"]}:
        raise ValueError("engine kernel identities changed")
    for kernel in current["kernels"]:
        prior = old_kernels[kernel["identity"]]
        for field in ("binary_base64", "entrypoint", "reserved_bytes"):
            if prior[field] != kernel[field]:
                raise ValueError(f"engine kernel {kernel['identity']} changed {field}")
    if original["inputs"] != current["inputs"]:
        raise ValueError("engine telemetry names changed")
    # Preserve every old field except the two proven repairs.
    replacement = json.loads(json.dumps(original))
    replacement["mid_high_rolloff"]["coefficients"] = current["mid_high_rolloff"]["coefficients"]
    names = {k["identity"]: k["input_names"] for k in current["kernels"]}
    for kernel in replacement["kernels"]:
        kernel["input_names"] = names[kernel["identity"]]
    first = DIV_MAP_JAVASCRIPT.index("      if(!Array.isArray(descriptor.inputNames)")
    last = DIV_MAP_JAVASCRIPT.index("\n    this.port.onmessage", first)
    new_call = DIV_MAP_JAVASCRIPT[first:last]
    replacements = [(text[start:start+end], json.dumps(replacement, ensure_ascii=False, separators=(",", ":"))),
                    (OLD_CALL, new_call), (OLD_DESCRIPTOR, NEW_DESCRIPTOR)]
    result = text
    for before, after in replacements:
        if result.count(before) != 1:
            raise ValueError("audio replacement seam is not unique")
        result = result.replace(before, after, 1)
    restored = result
    for before, after in reversed(replacements):
        if restored.count(after) != 1:
            raise ValueError("audio reverse seam is not unique")
        restored = restored.replace(after, before, 1)
    if restored.encode("utf-8") != source:
        raise ValueError("audio reverse-byte restoration failed")
    output = result.encode("utf-8")
    return output, {"base_sha256": BASE_SHA256, "candidate_sha256": hashlib.sha256(output).hexdigest(),
                    "reverse_byte_restore": True, "engine_binaries_unchanged": True,
                    "replacements": len(replacements), "input_names": names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve() or args.output.exists():
        raise ValueError("output must be a new separate candidate")
    output, report = repair(args.source.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.output.with_suffix(".audio-repair.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
