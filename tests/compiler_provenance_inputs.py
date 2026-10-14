"""Declare the handwritten SSA graphs that tests supply as compiler inputs.

The fixture author is the input boundary. Compiler rewrites must derive from
these declarations; they must not create a substitute root for missing inputs.
"""
def publish_authored_ssa(functions):
    from src.compiler.identity_concordance import (
        REGISTRY, Registry, RowField, RowFieldKind, Novel, Derived, Mode,
        Unresolved, current_identity_book,
    )
    from src.compiler.concordance_declarations import (
        SSA_VALUE, SSAValueFact, SSAValueOrigin, SSA_BLOCK, SSA_BLOCK_OWNER_UNROUTED,
    )
    from src.compiler.ssa_record_return_state import function_scope_of
    book = current_identity_book()
    if book.registry is REGISTRY:
        registry = Registry()
        for name in ("pages", "stages", "transforms", "reasons"):
            getattr(registry, name).update(getattr(REGISTRY, name))
        registry.private_pages.update(REGISTRY.private_pages)
        book.registry = registry  # Keep test-only declarations out of the global catalogue.
    page = book.registry.declare_page("test_authored_ssa", (
        RowField("function", RowFieldKind.SCOPE),), tuple)
    stage = book.registry.declare_stage("test_ssa_input")
    origin = book.registry.declare_transform("test_authored_ssa_input", 0)
    for function in functions.values():
        if function.metadata.get("_test_source_book") is book:
            continue  # Never re-author the output of a compiler rewrite.
        scope = function_scope_of(function)
        image = (tuple((value.id, value.dtype, tuple(value.shape), repr(value.accounting)) for value in function.args),
                 tuple((label, tuple((item.op, tuple(value.id for value in item.args),
                                     None if item.res is None else item.res.id, repr(item.attributes),
                                     tuple((value.id, value.dtype, tuple(value.shape), repr(value.accounting))
                                           for value in (*item.args, *((item.res,) if item.res is not None else ()))))
                                    for item in block.instrs), tuple(block.successors))
                       for label, block in function.blocks.items()))
        source = book.post(page, ((scope, function.name),), image,
                           stage=stage, provenance=Novel(origin, ()), mode=Mode.CONCORD)
        values = {int(value.id): value for value in function.args}
        values.update((int(item.res.id), item.res) for block in function.blocks.values()
                      for item in block.instrs if item.res is not None)
        for block in function.blocks.values():
            for item in block.instrs:
                for value in item.args:
                    values.setdefault(int(value.id), value)
        for value_id, value in values.items():
            book.post(SSA_VALUE, (scope, value_id),
                      SSAValueFact(value.dtype, tuple(value.shape), SSAValueOrigin.ADOPTED_GRAPH_ID),
                      stage=stage, provenance=Derived((source,)), mode=Mode.CONCORD)
        for label in function.blocks:
            book.post(SSA_BLOCK, (scope, function.name, label),
                      Unresolved(SSA_BLOCK_OWNER_UNROUTED, (source,)),
                      stage=stage, provenance=Derived((source,)), mode=Mode.CONCORD)
        function.metadata["_test_source_book"] = book
    return book
