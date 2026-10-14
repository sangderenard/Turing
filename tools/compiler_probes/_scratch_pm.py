import sys, runpy, traceback, time
sys.argv=["probe","llvm"]
t=time.time()
try:
    runpy.run_path("tools/compiler_probes/probe_solve_in_loop.py", run_name="__main__")
except BaseException as e:
    print("ELAPSED", time.time()-t)
    tb=e.__traceback__
    while tb:
        f=tb.tb_frame
        if f.f_code.co_name=="post_shape_statement":
            L=f.f_locals
            print("row",L["row"]); print("statement",L["statement"])
            print("latest",L["latest"], L["_cell_fact"](L["latest"]) if L["latest"] else None)
            print("cells",L["cells"]); print("source",[ (getattr(s,'graph',{}).get('function_name') if hasattr(s,'graph') else s) for s in (L["source"] or ())])
            print("owner",L["owner"].graph.get("function_name"), "value", L["value_id"])
        if f.f_code.co_name=="note_shape":
            print("note_shape fact", f.f_locals["fact"], "src", f.f_locals["source"] and f.f_locals["source"][1])
        tb=tb.tb_next
