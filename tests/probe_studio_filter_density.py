"""Execute the real compiler/native filter product, or preserve its exact gate.

No fallback evaluator is used. A failed compiler probe exits nonzero and writes
a blocked receipt; it is not a passing runtime test.
"""
import json
import math
from pathlib import Path
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from examples.meowdoku.studio_filter_density import build_density,neutral_density_sigma


def run(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    try:
        product=build_density(directory)
        rows=[]
        for u,v in ((0.,0.),(.25,.75),(.5,.5),(1.,1.)):
            for nd in (0.,.3,1.,2.):
                controls={f'density_{i}_{j}':i/3 for i in range(4) for j in range(4)}
                actual=product.sample(dict(controls,u=u,v=v,sigma_a_per_m=neutral_density_sigma(nd,.001),path_m=.001))
                expected=10**(-nd*u)
                if not math.isclose(actual['relative_density'],u,rel_tol=2e-5,abs_tol=2e-6):raise AssertionError(actual)
                if not math.isclose(actual['transmittance'],expected,rel_tol=2e-5,abs_tol=2e-6):raise AssertionError(actual)
                rows.append({'u':u,'v':v,'nd':nd,'native':actual,'expected_T':expected})
        receipt={'passed':True,'native_samples':rows,'manifest':product.manifest,
                 'scope':'Actual SSA C execution; LLVM/WGSL emitted, not executed; no production reader registration'}
    except Exception as error:
        receipt={'passed':False,'status':'blocked','error_type':type(error).__name__,
                 'error':str(error),'traceback':traceback.format_exc(),
                 'fallback_used':False}
    (directory/'probe-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


if __name__=='__main__':
    result=run(Path(__file__).resolve().parents[1]/'build'/'studio-filter-density-probe')
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
