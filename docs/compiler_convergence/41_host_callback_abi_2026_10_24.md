# Host callback ABI for stored callable fields (2026-10-24)

Schema: `turing-host-callback-v1` (`src/compiler/runtime_abi.py`).

## What it is

A class field annotated `Callable[[...], R]` (or `Optional[...]`,
`... | None`) holds code the host owns.  A compiled method that calls it,
`self.log(i)`, calls the host through a C function pointer.  A Python callable
cannot live in a native record slot, so the host keeps it:

- The constructor's store `self.log = log` (a parameter, or `None`) is the
  host's binding.  The constructor takes no native input for `log`; the field
  has no record slot.
- Every method that calls the field, or tests it against `None`, takes two
  extra inputs per field: the function pointer and an opaque context.  The
  host passes the callable it bound at construction on every such call.
- `self.log is None` / `is not None` tests the pointer against null.  A null
  pointer means the field is `None`.

Any other use of the field (`f = self.log`, passing it on, rebinding it
outside `__init__`) refuses.  The C and LLVM buffer-vector lanes refuse every
host callback; only the Fortran class surface carries these inputs.

## C shape

For `Callable[[int, float], float]` the pointer has the C type

    double callback(intptr_t context, int64_t a0, double a1);

The context comes first and is passed back unchanged; the compiled code never
reads it.  Argument and result types follow the annotation: `int` is
`int64_t`, `float` is `double`, `bool` is a one-byte C bool, `Any`/`object`
is an `intptr_t` opaque reference.  A `None` result is a `void` function.
Other annotation types (`str`, containers, `...` argument lists) refuse at
compile time with the type named.

Both inputs are passed by value as `int64_t`: the pointer's address and the
context.

## Where the program API states it

In `<Class>.api.yaml` metadata:

```yaml
host_callback_schema: turing-host-callback-v1
host_callbacks:
  Walker__run:                       # entry symbol
  - owner: Walker
    field: score
    function_parameter: score.__callback_function
    context_parameter: score.__callback_context   # null if only tested
    signature:
      schema: turing-host-callback-v1
      annotation: Optional[Callable[[int, float], float]]
      optional: true
      arguments: [int64, float64]
      result: float64                # null for a void callback
host_callback_bindings:
  Walker____init__:
  - {owner: Walker, field: score, parameter: score, signature: {...}}
```

`function_parameter` and `context_parameter` are the `source_name` of the
entry's parameters in `entry_points`; find the slot there, never by position.

## Order

Host callbacks are observable effects.  They run in source order relative to
each other, including across a branch: `log(i); if score: score(i, t);
log(-i)` calls `log`, then `score`, then `log`, as Python does.  The order is
published on the graph (`host_callback_order_edges`) and read by node
dependency order, so the region schedule and the SSA lowering keep it.

## Checked against Python

`tests/test_host_callback_fields.py` compiles a class with one required and
one optional callback.  It calls the native method through ctypes callbacks
and compares the result and the full call log with Python for `n` in 0, 1
and 4, with and without the optional callback.

## Not carried yet

The SSA interpreter's `trace: Callable[[str, Any, dict[int, Any]], None] |
None` refuses on `str`.  A `str` argument needs a (pointer, length) pair and a
`dict` needs an opaque handle the host can read; neither is declared here
yet.
