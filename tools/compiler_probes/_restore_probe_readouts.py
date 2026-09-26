"""Put back the probe readouts that located this defect.

The findings echo, the callsite argument resolution, the structural rebinding
receipts and the SSA block dumps are what turned "the output is NaN" into a
named instruction with a named pass on it.  None of them should have been
removed.
"""

from pathlib import Path

probe = Path(__file__).resolve().parents[2] / "tools/compiler_probes/probe_solve_numeric.py"
text = probe.read_text(encoding="utf-8")

AFTER_REPORT = "print(concordance_report(module, limit=8), flush=True)"

READOUTS = AFTER_REPORT + '''
from src.compiler.identity_concordance import (
    CorrelationTable as _CT, identity_book as _ib,
    loop_scope_declarations as _lsd,
)
for _fd in _CT.build(module).findings(module):
    if _fd.kind.startswith("loop-scope") or _fd.kind == "stale-carried-read":
        print("  >>", _fd.kind, _fd.function, _fd.value_id, _fd.detail,
              flush=True)

# What each scheduled callsite resolved its arguments to.  A graph id that
# resolves to itself inside a loop body means the carried machinery never
# mapped it to the current iteration.
_book = _ib(module)
_outer = set()
for _name in module.functions:
    for _decl in _lsd(_book, str(_name)):
        _outer.update(int(r["outer"]) for r in _decl["rebinds"])
_page = _book.page("callsite_argument")
print("CALLSITE ARGUMENTS: outer generations =", sorted(_outer), flush=True)
_seen = 0
for _row in _page.rows():
    _fact = _page.latest(_row)
    if not (isinstance(_fact, tuple) and len(_fact) == 3):
        continue
    _graph, _resolved, _in_loop = _fact
    if not _in_loop or int(_graph) not in _outer:
        continue
    _seen += 1
    print(f"  callsite {_row[1]} arg {_row[2]} in {_row[0]}: graph {_graph}"
          f" -> {_resolved}"
          f" {'SELF (unmapped)' if _graph == _resolved else 'mapped'}",
          flush=True)
print(f"CALLSITE ARGUMENTS: {_seen} in-loop use(s) of an outer generation,"
      f" out of {len(_page.rows())} recorded argument(s)", flush=True)

# Every operand rewrite is receipted in function metadata already.
print("STRUCTURAL REBINDINGS touching a declared loop scope:", flush=True)
_hits = 0
for _fname, _fn in module.functions.items():
    _decls = _lsd(_book, str(_fname))
    if not _decls:
        continue
    _gen = {}
    for _d in _decls:
        for _r in _d["rebinds"]:
            _gen[int(_r["outer"])] = "outer"
            _gen[int(_r["carried"])] = "carried"
            _gen[int(_r["inner"])] = "inner"
    for _c in _fn.metadata.get("structural_identity_rebindings", ()):
        _was, _now = int(_c["value_id"]), int(_c["replacement_value_id"])
        if _was not in _gen and _now not in _gen:
            continue
        _hits += 1
        print(f"  {_fname.split('__')[-1]} {_c['block']}"
              f"#{_c['instruction_index']} operand {_c['operand_index']}:"
              f" {_was} ({_gen.get(_was, '-')}) -> {_now}"
              f" ({_gen.get(_now, '-')})"
              f" [{_c.get('priority')} / {_c.get('tie_policy')}]", flush=True)
print(f"STRUCTURAL REBINDINGS: {_hits} touching a loop scope", flush=True)'''

BEFORE_RECONCILE = 'reconciliation = book.page("loop_result_reconciliation")'

BLOCK_DUMP = '''for _n, _f in module.functions.items():
    if not _n.endswith("___back_substitute"):
        continue
    print("FORMALS", [int(a.id) for a in getattr(_f, "args", ())], flush=True)
    print("BLOCKS", list(_f.blocks), flush=True)
    for _want in (20, 24, 12, 63):
        _where = [(b, i, d.op) for b, blk in _f.blocks.items()
                  for i, d in enumerate(blk.instrs)
                  if d.res is not None and int(d.res.id) == _want]
        print("  def", _want, "->", _where or "NOT DEFINED", flush=True)
    _uses63 = [(b, i, d.op) for b, blk in _f.blocks.items()
               for i, d in enumerate(blk.instrs)
               if 63 in [int(getattr(a, "id", -1)) for a in d.args]]
    print("USES OF 63:", _uses63, flush=True)
    for _bn in ("loop_body", "loop_header"):
        _blk = _f.blocks.get(_bn)
        if _blk is None:
            continue
        print("== block", _bn, flush=True)
        for _i, _d in enumerate(_blk.instrs):
            print(f"  {_i:3d} {_d.op:14s} res={getattr(_d.res,'id',None)} "
                  f"args={[int(getattr(a,'id',-1)) for a in _d.args]}",
                  flush=True)
            print(f"        ALLATTRS {dict(_d.attributes)}"[:400], flush=True)
    for _cn, _cf in module.functions.items():
        if not _cn.endswith("___back_substitute__planned_region_5"):
            continue
        print("== planned_region_5 formals=",
              [int(a.id) for a in getattr(_cf, "args", ())], flush=True)
        for _bn2, _blk2 in _cf.blocks.items():
            for _i2, _d2 in enumerate(_blk2.instrs):
                if _d2.op == "Const":
                    continue
                print(f"   {_i2:3d} {_d2.op:14s} "
                      f"res={getattr(_d2.res,'id',None)} "
                      f"args={[int(getattr(a,'id',-1)) for a in _d2.args]} "
                      f"attrs={ {k: v for k, v in _d2.attributes.items() if k == 'callee'} }",
                      flush=True)
    for _ins in _f.blocks["loop_header"].instrs:
        if str(_ins.op).lower() != "phi":
            continue
        print("PHIATTRS", dict(_ins.attributes), flush=True)
        print("HDRPHI res=", getattr(_ins.res, "id", None),
              "args=", [int(getattr(a, "id", -1)) for a in _ins.args],
              flush=True)
        for _a in _ins.args:
            _aid = int(getattr(_a, "id", -1))
            for _b, _blk in _f.blocks.items():
                for _i, _d in enumerate(_blk.instrs):
                    if _d.res is not None and int(_d.res.id) == _aid:
                        print("   def of", _aid, "at", _b, _i, _d.op,
                              flush=True)
    break

''' + BEFORE_RECONCILE

assert text.count(AFTER_REPORT) == 1
assert text.count(BEFORE_RECONCILE) == 1
text = text.replace(AFTER_REPORT, READOUTS)
text = text.replace(BEFORE_RECONCILE, BLOCK_DUMP)
probe.write_text(text, encoding="utf-8")
print("probe readouts restored")
