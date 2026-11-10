"""Guarded presentation/diagnostic upgrade of the verified audio base. Never publishes."""
from pathlib import Path
import hashlib, json, argparse
ROOT=Path(__file__).resolve().parents[1]
BASE="4d840bb96ec1d9673aa8650643d9084a514af23e688146c3a89f679e8df47bae"
def upgrade(data):
    if hashlib.sha256(data).hexdigest()!=BASE: raise ValueError("Unknown game base")
    source=(ROOT/"src/compiler/abstract_ui_div_map.py").read_text(encoding="utf-8")
    original=data.decode("utf-8"); text=original; replacements=[]
    def replace(old,new):
        nonlocal text
        if text.count(old)!=1: raise ValueError("Nonunique HUD seam: "+old[:80])
        text=text.replace(old,new,1); replacements.append((old,new))
    def function(s,name):
        start=s.index("function "+name+"("); end=s.index("\nfunction ",start+1)
        return s[start:end]
    helper_start=source.index("function vehicleGraphProblem(")
    helper_end=source.index("function drawVehicleShaderHud(",helper_start)
    replace(function(text,"drawVehicleShaderHud"),source[helper_start:helper_end]+function(source,"drawVehicleShaderHud"))
    replace(function(text,"handleVehicleShaderHudPointer"),function(source,"handleVehicleShaderHudPointer"))
    old="const edge=member.mechanical_edge,a=[...positions.get(edge.a)],authoredB=[...positions.get(edge.b)],"
    start=source.index("const edge=member.mechanical_edge,pointA="); end=source.index("\n      b=edge.runtime_failed",start)
    replace(old,source[start:end])
    marker='    if(box.mechanical_edge?.constraint==="spring-damper"){'
    start=source.index("    // Add three pulsing collars"); end=source.index(marker,start)
    replace(marker,source[start:end]+marker)
    old='            vehicleRuntime.electrical={...vehicleRuntime.electrical,...(message.energy||{})};'
    replace(old,old+'\n            if(message.damageMetrics)vehicleRuntime.damage=message.damageMetrics;')
    # Transport the existing worker diagnostic record only; no solver or ABI changes.
    old=r'jointBushingDissipatedEnergyJ:Number(body.damage?.dissipatedEnergyJ||0)},\n    driverAssistance:'
    replace(old,r'jointBushingDissipatedEnergyJ:Number(body.damage?.dissipatedEnergyJ||0)},\n    damageMetrics:body.damage||null,driverAssistance:')
    restored=text
    for old,new in reversed(replacements):
        if restored.count(new)!=1: raise ValueError("Nonunique reverse seam")
        restored=restored.replace(new,old,1)
    if restored.encode("utf-8")!=data: raise ValueError("Reverse restoration failed")
    result=text.encode("utf-8")
    return result,{"base_sha256":BASE,"candidate_sha256":hashlib.sha256(result).hexdigest(),
        "reverse_bytes_verified":True,"replacements":len(replacements),
        "limitations":["Axial small-displacement presentation energy estimate, not total machine energy",
          "No global conservation or multiplayer diagnostic channel", "Browser visual QA pending"]}
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("base",type=Path);parser.add_argument("output",type=Path);args=parser.parse_args()
    result,receipt=upgrade(args.base.read_bytes());args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(result);args.output.with_suffix(".graph-hud.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))
