# A literal specialization that folds the callee's body to a constant drops
# the formal it no longer reads (check(0, xs) never reads values). The
# authored binding stays as provenance, but no value ABI may be settled onto
# the dropped formal: it has no class row in that specialization's scope.
# Before the fix the final concordance refused "ordinary type ABI does not
# derive directly from its exact source_value_class cell" (Token Orchard's
# validate_batch_envelope with offsets=[1, 1] and _valid_atom with domain=0).
def check(flag: int, values: list[int]) -> int:
    if flag == 0:
        return 1
    return len(values) + values[0]


def root(n: int) -> int:
    xs = [n, 2]
    return check(0, xs) + check(1, xs)

# INPUTS [{'n': 0}, {'n': 5}, {'n': -3}]
