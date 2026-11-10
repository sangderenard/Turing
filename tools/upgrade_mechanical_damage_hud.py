"""Stage the existing HUD metric upgrade without recompiling physics or publishing."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def patch(source: bytes):
    text = source.decode('utf-8')
    original = text
    generator = (ROOT / 'src/compiler/abstract_ui_div_map.py').read_text(encoding='utf-8')
    replacements = []

    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError('Expected one original HUD seam: ' + old[:80])
        text = text.replace(old, new, 1)
        replacements.append((old, new))

    seam = 'function drawVehicleShaderHud(gl,width,height){'
    a = generator.index('// Read-only analysis of the authored mechanical graph')
    b = generator.index(seam, a)
    replace(seam, generator[a:b] + seam)
    replace('panelWidth=154*ratio,panelHeight=286*ratio', 'panelWidth=154*ratio,panelHeight=321*ratio')
    a = text.index('  // Actual mechanical graph damage, projected top-down.')
    b = text.index('  gl.disable(gl.BLEND);gl.enable(gl.DEPTH_TEST);', a)
    c = generator.index('  // Every authored graph node is shown')
    d = generator.index('  gl.disable(gl.BLEND);gl.enable(gl.DEPTH_TEST);', c)
    replace(text[a:b], generator[c:d])
    seam = '    glyphText=(text,x,y,scale,color)=>'
    c = generator.index('    extraGlyphs=')
    d = generator.index(seam, c)
    replace(seam, generator[c:d] + seam)
    replace('const rows=glyphs[character]||[]', 'const rows=glyphs[character]||extraGlyphs[character]||[]')
    replace('  if(hit.action==="traction-control")', '  if(hit.action==="damage-metric")vehicleRuntime.damageMetric=hit.metric;\n  else if(hit.action==="traction-control")')
    seam = 'vehicleRuntime.electrical={...vehicleRuntime.electrical,...(message.energy||{})};'
    replace(seam, seam + '\n            if(message.damageMetrics)vehicleRuntime.damage=message.damageMetrics;')

    # Decode only the original worker JSON string; keep every other model byte.
    model_start = text.index('id="abstract-ui-model"')
    model_start = text.index('>', model_start) + 1
    model_end = text.index('</script>', model_start)
    model_text = text[model_start:model_end]
    model = json.loads(model_text)
    workers = model['loop_deployment']['workers']
    matches = [w['source'] for w in workers if 'function ensureVehicleDamage' in w.get('source', '')]
    if len(matches) != 1:
        raise ValueError('Expected one original vehicle worker')
    worker = matches[0]
    prefix = json.dumps(worker[:80], ensure_ascii=False)[1:-1]
    offset = model_text.index(prefix) - 1
    decoded, length = json.JSONDecoder().raw_decode(model_text[offset:])
    if decoded != worker:
        raise ValueError('Worker JSON identity mismatch')
    old_literal = model_text[offset:offset + length]
    seam = '    driverAssistance:{...ensureVehicleDriverAssistance(body)}});}'
    if worker.count(seam) != 1:
        raise ValueError('Expected one 250ms energy publication')
    updated = worker.replace(seam, '    damageMetrics:body.damage||null,\n' + seam, 1)
    new_literal = json.dumps(updated, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    replace(old_literal, new_literal)
    recovered = text
    for old, new in reversed(replacements):
        if recovered.count(new) != 1:
            raise AssertionError('Ambiguous reverse patch')
        recovered = recovered.replace(new, old, 1)
    assert recovered == original
    return text.encode('utf-8'), {'original_page_recovered_exactly': True, 'compiled_kernels_unchanged': True,
        'changes': len(replacements), 'source_sha256': hashlib.sha256(source).hexdigest(),
        'output_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve() or args.output.exists():
        parser.error('Output must be a new staged file; original/live pages are not overwritten')
    result, receipt = patch(args.source.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result)
    args.output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
