import runpy

from src.compiler import fortran_c_shell as shell


original = shell._field_slot_ops


def inspect(graph, **kwargs):
    result = original(graph, **kwargs)
    raw = getattr(graph, "G", graph)
    if raw.graph.get("function_name") == "update_dt_max":
        self_record = dict(
            (raw.graph.get("parameter_record_abi") or {}).get("self") or {}
        )
        print(
            "FIELD_SLOTS",
            "graph=", id(raw),
            "owner=", raw.graph.get("method_owner"),
            "fields=", result[4],
            "abi_fields=", tuple(dict(self_record.get("fields") or {})),
            "ops=", result[1],
            flush=True,
        )
    return result


shell._field_slot_ops = inspect
runpy.run_module("tools.scan_managed_duplicates", run_name="__main__")
