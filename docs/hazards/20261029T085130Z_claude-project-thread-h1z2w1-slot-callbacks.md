2026-10-29 External runtime slots called back into the host interpreter without reading the execution contract (MEASURED, external_functions._slot_address)

Program: a SymPy law with an undefined Function `f` (tools/compiler_probes/
probe_external_function.py), lowered under `batch_contract`, whose execution
overlay is vehicle_full_native_execution.yaml (`python_callbacks: reject`).
Wrong record: `bind_external_slots` -> `_slot_address` decided by the value's
type -- `callable(implementation)` -- and wrapped any Python callable in a
`ctypes.CFUNCTYPE`, so native code called back into the live interpreter
although the contract the module was lowered under rejects Python callbacks.
Rule: `external_slot_rows` copies `python_callbacks` from the lowered module's
`metadata["extraction_contract"]["execution"]` onto every slot row, and a
Python callable binds only when that is `contract_only`; otherwise a named
refusal. Native addresses and built LLVM pieces bind as before.

Not this entry: the probe itself stops earlier on main, before binding
(`'ExternalSlotABI' object has no attribute 'buffer_element_dtypes'`,
fortran_c_shell `_lower_ast_source_to_ssa_impl`).
