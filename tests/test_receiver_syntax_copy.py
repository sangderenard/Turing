"""A rewritten call copies its syntax, never the compiler's annotations.

``_resolve_grounded_method_references`` rewrites ``f(x)`` into
``receiver.method(x)``.  It used ``copy.deepcopy`` on the authored call, which
also cloned private annotations hung on its nodes; with a module's bindings
among them (``re.compile`` pursued into ``re._compiler``) that crashed with
"cannot pickle 'mappingproxy' object".
"""

from __future__ import annotations

import ast
import types

from src.compiler.receiver_resolution_prepasses import _syntax_copy


def test_syntax_copy_clones_fields_and_shares_annotations():
    call = ast.parse("f(a, b.c)").body[0].value
    bindings = {"module": types.MappingProxyType({"x": 1})}
    call._python_bindings = bindings
    call.args[1]._python_bindings = bindings

    copied = _syntax_copy(call)

    assert ast.dump(copied, include_attributes=True) == ast.dump(
        call, include_attributes=True,
    )
    assert copied is not call and copied.args[0] is not call.args[0]
    assert copied._python_bindings is bindings
    assert copied.args[1]._python_bindings is bindings
    copied.args.append(ast.Name(id="z", ctx=ast.Load()))
    assert len(call.args) == 2
