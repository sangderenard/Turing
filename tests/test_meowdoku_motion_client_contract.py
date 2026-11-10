"""Existing published browser predictor vs exact Python engine, not native proof."""
import hashlib
import inspect
import json
from pathlib import Path
import random
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'examples')]
from meowdoku import entity_motion as motion

def test_actual_damped_axis_wasm():
    abi=json.loads((ROOT/'build/meowdoku/web/motion.abi.json').read_text())
    source='import math\n'+inspect.getsource(motion.damped_axis)
    assert abi['source_digest']==hashlib.sha256(source.encode()).hexdigest()[:16]
    assert abi['inputs']==list(motion.NAMES) and abi['outputs']==list(motion.OUTPUTS)
    r=random.Random(731)
    rows=[]
    for _ in range(128):
        values=[r.uniform(-1,1),r.uniform(-2,2),r.uniform(-1,1),r.choice([-1,r.uniform(0,1)]),-1,1,2,0.2,0.3,r.uniform(.001,.1)]
        rows.append(dict(zip(motion.NAMES,values)))
    js="""
import fs from 'node:fs';
const shell=fs.readFileSync('examples/mobile_reference_app/web/turing-wasm-shell.js','utf8');
const {TuringWasmProgram}=await import('data:text/javascript;base64,'+Buffer.from(shell).toString('base64'));
const abi=JSON.parse(fs.readFileSync('build/meowdoku/web/motion.abi.json'));
const engine=await TuringWasmProgram.fromBytes(fs.readFileSync('build/meowdoku/web/motion.wasm'),abi);
let input=''; for await(const c of process.stdin)input+=c;
console.log(JSON.stringify(JSON.parse(input).map(row=>engine.invoke(row))));
"""
    result=subprocess.run(['node','--input-type=module','-e',js],input=json.dumps(rows),text=True,capture_output=True,cwd=ROOT,check=True)
    outputs=json.loads(result.stdout)
    for row,actual in zip(rows,outputs):
        expected=motion.damped_axis(**row)
        for name,value in zip(motion.OUTPUTS,expected):
            assert abs(actual[name]-value)<1e-12
