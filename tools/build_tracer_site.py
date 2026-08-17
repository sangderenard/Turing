"""Generate a static site: pick a source, pick a layer, read the colour.

One page, two dropdowns and a switch.

* **source** -- which representation's text you are reading: the authored
  Python, the authored SymPy equations, or the lowered SSA;
* **layer** -- which stage's influence field supplies the colour;
* **blended / categories** -- how the colour is formed.

The two dropdowns are independent on purpose. "The Python source coloured
by the SSA field" is the question that is hard to answer any other way,
and it is just a pair of selections here.

Blended versus categories
-------------------------
`categories` is the field's own convention: each binding-time category
collapses to its own hue and they are kept apart, because averaging a
dynamic hue with a baked one reports a dispersion that describes nothing.

`blended` uses the retained ``Spectrum``: every individual contributing
frequency is converted to colour and ADDED, the way light adds. It is the
真 mix rather than a centroid -- a location fed by six origins looks like
those six mixed, not like their mean, and two different origin sets that
share a mean no longer look identical.

Both are computed HERE, in Python, through `influence_field_image.dye_rgb`.
The page only swaps between precomputed colours: putting a colour model in
JavaScript would be a second model that disagrees with the shader about
the same field.

Correlation uses the strongest identity each pane has. Where a text
states which value it means (SSA tN tokens) it is read by key. Where it
names an authored quantity it is read by name. And a stage boundary is
not 1:1: an authored equation SPREADS over every instruction its
expansion produced, and instructions reached by several equations MIX
them by adding spectra. Nothing is matched on position.

    python tools/build_tracer_site.py
    python tools/build_tracer_site.py --out build/tracer_site.html
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ADVANCE = "symbolic_fluid_control__symbolic_fluid_advance"
STEP = "symbolic_fluid_control__symbolic_fluid_step"
REGION = STEP + "__planned_region_0"
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
UNCOLOURED = "#1a1a20"


def newest_lowering() -> Path:
    found = sorted(
        (ROOT / "build").glob("*/control_repository_ssa.pkl"),
        key=lambda path: path.stat().st_mtime, reverse=True,
    )
    if not found:
        raise SystemExit("no lowered SSA under build/")
    return found[0]


def to_hex(triple) -> str:
    red, green, blue = (max(0, min(255, int(round(c * 255)))) for c in triple)
    return f"#{red:02x}{green:02x}{blue:02x}"


def category_colour(reading: Any) -> str:
    from src.rendering.influence_field_image import dye_rgb

    category = (reading.categories or {}).get("dynamic")
    if category is None:
        return UNCOLOURED
    return to_hex(dye_rgb(
        getattr(category, "hue", 0.0),
        getattr(category, "saturation", 0.0),
        max(0.25, min(1.0, float(reading.value))),
    ))


def blended_colour(accumulator: Any) -> str:
    """Add every individual hue, weighted -- real mixing, not a centroid."""
    from src.rendering.influence_field_image import dye_rgb

    lines = tuple(getattr(accumulator, "lines", ()) or ())
    if not lines:
        return UNCOLOURED
    red = green = blue = 0.0
    for frequency, weight in lines:
        r, g, b = dye_rgb(float(frequency), 1.0, 1.0)
        red, green, blue = red + r * weight, green + g * weight, blue + b * weight
    peak = max(red, green, blue)
    if peak <= 0.0:
        return UNCOLOURED
    return to_hex((red / peak, green / peak, blue / peak))


def build_layers(contract: Any) -> tuple[dict, dict]:
    """{layer id: {name: (reading, accumulator)}} plus labels."""
    from src.compiler.influence_field import (
        field_from_ssa, field_from_sympy,
    )
    from src.compiler.symbolic_fluid_model import (
        symbolic_viscous_shallow_water_equations,
    )

    layers: dict[str, dict[str, tuple]] = {}
    keyed: dict[str, dict[Any, tuple]] = {}
    labels: dict[str, str] = {}

    model = symbolic_viscous_shallow_water_equations()
    field = field_from_sympy(model.equations, contract)
    table = {reading.key: reading for reading in field.table()}
    entries: dict[str, tuple] = {}
    for equation in model.equations:
        if equation.lhs in table:
            entries[str(equation.lhs)] = (
                table[equation.lhs],
                field.moments(equation.lhs).get("dynamic"),
            )
    for symbol in model.symbols.values():
        if symbol in table:
            entries[str(symbol)] = (
                table[symbol], field.moments(symbol).get("dynamic"),
            )
    layers["sympy"] = entries
    labels["sympy"] = "authored SymPy equations"

    with newest_lowering().open("rb") as stream:
        module, _outputs, _exports = pickle.load(stream)
    field = field_from_ssa(module, contract, functions=[ADVANCE, STEP, REGION])
    table = {reading.key: reading for reading in field.table()}
    entries = {}
    for source_name in (REGION, STEP, ADVANCE):
        function = module.functions.get(source_name)
        if function is None:
            continue
        pairs = list(function.metadata.get("value_names") or ())
        pairs += list(function.metadata.get("parameter_names") or ())
        if source_name == STEP:
            pairs += list(function.metadata.get("named_outputs") or ())
        home = REGION if source_name == STEP else source_name
        target = module.functions.get(home)
        if target is None:
            continue
        for label, value in pairs:
            for block_name, block in target.blocks.items():
                for instruction in block.instrs:
                    if (
                        instruction.res is not None
                        and int(instruction.res.id) == int(value)
                    ):
                        key = (home, block_name, int(value))
                        reading = table.get(key)
                        if reading is not None and str(label) not in entries:
                            entries[str(label)] = (
                                reading, field.moments(key).get("dynamic"),
                            )
    layers["ssa"] = entries
    labels["ssa"] = "lowered repository SSA"
    # The same field, addressed by key instead of by name. A representation
    # that states which value it means should be read that way.
    keyed["ssa"] = {
        reading.key: (reading, field.moments(reading.key).get("dynamic"))
        for reading in field.table()
    }
    return layers, keyed, labels, module, model


def source_texts(module: Any, model: Any) -> tuple[dict, dict]:
    from src.compiler.symbolic_fluid_dt import SYMBOLIC_FLUID_DT_SOURCE

    direct: dict[str, list] = {}
    texts = {"python": SYMBOLIC_FLUID_DT_SOURCE}
    labels = {"python": "authored Python (the traversal)"}

    texts["sympy"] = "\n".join(
        f"{equation.lhs} = {equation.rhs}" for equation in model.equations
    )
    labels["sympy"] = "authored SymPy equations"

    region = module.functions.get(REGION)
    if region is not None:
        names = {
            int(value): str(label)
            for label, value in (
                module.functions[STEP].metadata.get("named_outputs") or ()
            )
        }
        # Each rendered line remembers its block, so every tN token can be
        # resolved to the field key it denotes. Colouring this pane by NAME
        # lights only the values whose authored name reached a comment --
        # eleven highlights over three hundred lines of real instructions,
        # which is what made it look like the tracer had nothing to say
        # about the SSA.
        rows: list[tuple[str, str]] = []
        for block_name, block in region.blocks.items():
            rows.append((f"{block_name}:", block_name))
            for instruction in block.instrs:
                result = instruction.res
                target = "" if result is None else f"t{int(result.id)}"
                named = names.get(int(result.id)) if result is not None else None
                arguments = ", ".join(
                    f"t{int(a.id)}" for a in instruction.args
                )
                label = f"  {target:>7} = {instruction.op}({arguments})"
                rows.append((
                    label + (f"    # {named}" if named else ""), block_name,
                ))
        tokens = []
        for row, (line, block_name) in enumerate(rows):
            for match in re.finditer(r"t(\d+)", line):
                tokens.append((
                    row, match.start(), match.end(),
                    (REGION, block_name, int(match.group(1))),
                ))
        direct["ssa"] = tokens
        texts["ssa"] = "\n".join(line for line, _block in rows)
        labels["ssa"] = f"lowered SSA: {REGION.split('__')[-1]}"
    return texts, labels, direct


def spread_and_mix(module: Any, sympy_entries: dict) -> dict:
    """Carry each authored quantity onto every value its expansion produced.

    Provenance is a relation, not a function, and treating it as 1:1 is why
    the cross-stage panes were nearly empty. A translation does two things
    a one-to-one map cannot express:

    * it SPREADS -- one authored equation becomes tens of instructions, and
      every one of them descends from that equation;
    * it MIXES -- an instruction reached by several equations descends from
      all of them, which is what common subexpression means.

    Both fall out of the backward cone plus the Spectrum's algebra. A value
    in one cone takes that equation's spectrum; a value in six takes the
    SUM of six, which is a real mixture rather than a winner. Addition is
    the merge, so the result does not depend on which cone is walked first.
    """
    region = module.functions.get(REGION)
    step = module.functions.get(STEP)
    if region is None or step is None:
        return {}
    produced: dict[int, Any] = {}
    home: dict[int, str] = {}
    for block_name, block in region.blocks.items():
        for instruction in block.instrs:
            if instruction.res is not None:
                produced[int(instruction.res.id)] = instruction
                home[int(instruction.res.id)] = block_name

    def cone(root: int) -> set[int]:
        seen: set[int] = set()
        pending = [root]
        while pending:
            value_id = pending.pop()
            if value_id in seen or value_id not in produced:
                continue
            seen.add(value_id)
            pending.extend(int(a.id) for a in produced[value_id].args)
        return seen

    carried: dict[Any, tuple] = {}
    for label, value in tuple(step.metadata.get("named_outputs") or ()):
        entry = sympy_entries.get(str(label))
        if entry is None or int(value) not in produced:
            continue
        reading, accumulator = entry
        if accumulator is None:
            continue
        for value_id in cone(int(value)):
            key = (REGION, home[value_id], value_id)
            previous = carried.get(key)
            carried[key] = (
                reading if previous is None else previous[0],
                accumulator if previous is None else previous[1] + accumulator,
            )
    return carried


def colour_by_key(tokens: list, keyed: dict, mode: str) -> list:
    """Colour tokens that name a field key outright.

    Where a representation states which value it means, use that. Falling
    back to name matching there would colour only the few values whose
    authored name survived into a comment, which is what made the SSA pane
    read as a couple of highlights over three hundred lines of real
    instructions.
    """
    spans = []
    for row, start, end, key in tokens:
        found = keyed.get(key)
        if found is None:
            continue
        reading, accumulator = found
        colour = (
            blended_colour(accumulator) if mode == "blended"
            else category_colour(reading)
        )
        if colour != UNCOLOURED:
            spans.append([row, start, end, colour])
    return spans


def colour_spans(text: str, entries: dict, mode: str) -> list:
    spans = []
    for row, line in enumerate(text.splitlines()):
        for match in IDENTIFIER.finditer(line):
            found = entries.get(match.group(0))
            if found is None:
                continue
            reading, accumulator = found
            colour = (
                blended_colour(accumulator) if mode == "blended"
                else category_colour(reading)
            )
            if colour != UNCOLOURED:
                spans.append([row, match.start(), match.end(), colour])
    return spans


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="build/tracer_site.html")
    arguments = parser.parse_args()

    from src.compiler.influence_field import InfluenceContract

    # spectral=True is what makes `blended` possible at all: without the
    # retained lines there is nothing to mix, only a centroid to restate.
    contract = InfluenceContract(enabled=True, spectral=True)
    layers, keyed, layer_labels, module, model = build_layers(contract)
    texts, text_labels, direct = source_texts(module, model)
    # The authored mathematics, spread across every SSA value its expansion
    # produced and mixed where expansions overlap.
    keyed["sympy"] = spread_and_mix(module, layers["sympy"])

    payload = {
        "sources": {
            key: {"label": text_labels[key], "lines": value.splitlines()}
            for key, value in texts.items()
        },
        "layers": {key: layer_labels[key] for key in layers},
        "colours": {
            source_key: {
                layer_key: {
                    mode: (
                        colour_by_key(
                            direct[source_key], keyed[layer_key], mode,
                        )
                        if source_key in direct and layer_key in keyed
                        else colour_spans(texts[source_key], entries, mode)
                    )
                    for mode in ("categories", "blended")
                }
                for layer_key, entries in layers.items()
            }
            for source_key in texts
        },
        "counts": {
            layer_key: len(entries) for layer_key, entries in layers.items()
        },
    }

    destination = ROOT / arguments.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_page(payload), encoding="utf-8")
    for layer_key, entries in layers.items():
        print(f"layer {layer_key}: {len(entries)} authored names carry a reading")
    for source_key, text in texts.items():
        print(f"source {source_key}: {len(text.splitlines())} lines")
    print(f"wrote {destination}")
    return 0


def _page(payload: dict) -> str:
    data = json.dumps(payload)
    return (
        "<!doctype html><meta charset='utf-8'><title>tracer</title>"
        "<style>"
        "body{background:#0d0d11;color:#dcdce4;font:13px/1.55 ui-monospace,"
        "Consolas,monospace;margin:0;padding:16px 20px}"
        "header{display:flex;gap:14px;align-items:center;flex-wrap:wrap;"
        "margin-bottom:12px}"
        "select,label{font:12px ui-sans-serif,system-ui}"
        "select{background:#191922;color:#dcdce4;border:1px solid #333;"
        "border-radius:4px;padding:4px 6px}"
        "#note{color:#7b7b8c;font:12px ui-sans-serif,system-ui;margin:0 0 12px}"
        ".l{white-space:pre}.n{color:#3f3f52;display:inline-block;width:3.5em;"
        "text-align:right;padding-right:1em;user-select:none}"
        "b{font-weight:600;color:#111;border-radius:2px}"
        "</style>"
        "<header>"
        "<label>source <select id='source'></select></label>"
        "<label>layer <select id='layer'></select></label>"
        "<label><input type='checkbox' id='blend'> blended individual hues"
        "</label>"
        "</header><p id='note'></p><div id='code'></div>"
        f"<script>const DATA={data};"
        "const source=document.getElementById('source'),"
        "layer=document.getElementById('layer'),"
        "blend=document.getElementById('blend'),"
        "code=document.getElementById('code'),note=document.getElementById('note');"
        "for(const k in DATA.sources){const o=document.createElement('option');"
        "o.value=k;o.textContent=DATA.sources[k].label;source.append(o);}"
        "for(const k in DATA.layers){const o=document.createElement('option');"
        "o.value=k;o.textContent=DATA.layers[k];layer.append(o);}"
        "function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;')"
        ".replace(/>/g,'&gt;');}"
        "function draw(){const s=source.value,l=layer.value,"
        "m=blend.checked?'blended':'categories';"
        "const lines=DATA.sources[s].lines,spans=DATA.colours[s][l][m];"
        "const byRow={};for(const sp of spans){(byRow[sp[0]]||(byRow[sp[0]]=[]))"
        ".push(sp);}"
        "let html='';"
        "for(let r=0;r<lines.length;r++){const line=lines[r];let out='',at=0;"
        "for(const sp of (byRow[r]||[]).sort((a,b)=>a[1]-b[1])){"
        "if(sp[1]<at)continue;out+=esc(line.slice(at,sp[1]));"
        "out+='<b style=\"background:'+sp[3]+'\">'+esc(line.slice(sp[1],sp[2]))"
        "+'</b>';at=sp[2];}"
        "out+=esc(line.slice(at));"
        "html+='<div class=\"l\"><span class=\"n\">'+(r+1)+'</span>'+out+'</div>';}"
        "code.innerHTML=html;"
        "const n=spans.length;note.textContent=DATA.sources[s].label+', coloured "
        "by '+DATA.layers[l]+' ('+DATA.counts[l]+' authored names carry a "
        "reading; '+n+' occurrences coloured here). '+(m==='blended'?'Blended: "
        "every contributing frequency added, the way light adds.':'Categories: "
        "each binding-time category collapsed to its own hue, kept apart.')"
        "+' Uncoloured means no reading under that name in this layer.';}"
        "source.onchange=layer.onchange=blend.onchange=draw;draw();"
        "</script>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
