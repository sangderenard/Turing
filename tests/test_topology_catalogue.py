"""Grouping region programs by canonical topology.

The signature is invariant to value-id renaming (the property a content hash
lacked, which is why the earlier reduction cache never reused across builds) and
abstracts constant data, so "the same algorithm over different data" groups
together. That collapses a program that repeats one operation many times into a
small master table of the algorithms actually present.
"""
from __future__ import annotations

import tempfile

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.topology_catalogue import (
    TopologyCatalogue,
    algebraic_profile,
    canonical_formula,
    invocation_name,
    kernel_name,
    topology_name,
    topology_signature,
)


def _ops_program(*ops):
    steps = [OpStep(i, op, [1, 2], {}, 10 + i) for i, op in enumerate(ops)]
    return _program(steps, feeds=(1, 2), outputs={"r": 10 + len(ops) - 1})


def test_algebraic_profile_asserts_the_number_system():
    # Bitwise operators are GF(2); their laws include xor's self-inverse.
    gf2 = algebraic_profile(_ops_program("bitand", "bitxor"))
    assert gf2["domain"] == "GF(2) boolean ring" and gf2["closed"]
    assert "self_inverse" in gf2["symmetries"]
    # Integer arithmetic is the modular ring.
    ring = algebraic_profile(_ops_program("add", "mul"))
    assert ring["domain"].startswith("ring Z/2^n") and ring["closed"]
    # min/max is a lattice.
    assert algebraic_profile(_ops_program("minimum", "maximum"))["domain"] == "min/max semilattice"
    # Pure memory op: no arithmetic to classify.
    assert algebraic_profile(_ops_program("index_set"))["domain"] == "none"
    # Spanning two structures is mixed, not closed.
    mixed = algebraic_profile(_ops_program("add", "bitand"))
    assert mixed["domain"] == "mixed" and not mixed["closed"]


def test_catalogue_entry_carries_the_algebraic_domain():
    catalogue = TopologyCatalogue(root=__import__("tempfile").mkdtemp())
    signature = catalogue.record(_ops_program("bitand", "bitxor"))
    assert catalogue.entries[signature]["domain"] == "GF(2) boolean ring"


def _program(steps, feeds, outputs):
    return FusedProgram(
        version=1, feeds=set(feeds), steps=list(steps), outputs=dict(outputs)
    )


def _shift(program, k):
    return _program(
        [OpStep(s.step_id, s.op_name, [i + k for i in s.input_ids],
                dict(s.attrs), s.result_id + k) for s in program.steps],
        {f + k for f in program.feeds},
        {n: v + k for n, v in program.outputs.items()},
    )


_BASE = _program(
    [OpStep(0, "add", [1, 2], {}, 3),
     OpStep(1, "index_set", [3, 2], {"slices": 2}, 4)],
    feeds=(1, 2), outputs={"r": 4},
)


def test_signature_is_invariant_to_value_id_renaming():
    assert topology_signature(_BASE) == topology_signature(_shift(_BASE, 10_000))


def test_constant_data_is_abstracted_but_wiring_is_not():
    same_shape = _program(
        [OpStep(0, "add", [1, 2], {}, 3),
         OpStep(1, "index_set", [3, 2], {"slices": 999}, 4)],
        feeds=(1, 2), outputs={"r": 4},
    )
    different_op = _program(
        [OpStep(0, "mul", [1, 2], {}, 3),
         OpStep(1, "index_set", [3, 2], {"slices": 2}, 4)],
        feeds=(1, 2), outputs={"r": 4},
    )
    assert topology_signature(same_shape) == topology_signature(_BASE)
    assert topology_signature(different_op) != topology_signature(_BASE)


def test_canonical_formula_is_analyzable_and_leveled():
    # Topology: named inputs, constants abstracted; ``in1`` reused shows the DAG
    # sharing (the value stored is the same input added).
    formula = canonical_formula(_BASE, keep_data=False)
    assert formula == "index_set(add(in0,in1),in1){slices=*}"
    assert topology_name(_BASE) == "T:" + formula
    # Kernel keeps the constant and the dtype; a byte-distinct compiled body.
    assert kernel_name(_BASE, "i64") == "K:index_set(add(in0,in1),in1){slices=2}@i64"
    # Invocation binds a kernel to resident field slots.
    inv = invocation_name(kernel_name(_BASE, "i64"), [3, 7], [3])
    assert inv.startswith("I:K:") and "<in:s3,s7;out:s3>" in inv


def test_formula_uses_an_explicit_stack_on_a_deep_chain():
    # A long linear chain must not overflow a language recursion stack, and past
    # the cap it degrades to a compact genus-level name rather than a huge one.
    steps = [OpStep(0, "add", [0, 1], {}, 2)]
    for k in range(1, 200):
        steps.append(OpStep(k, "add", [k + 1, 1], {}, k + 2))
    deep = _program(steps, feeds=(0, 1), outputs={"r": 201})
    name = canonical_formula(deep)
    assert name.startswith("add200") and "~" in name  # genus-level fallback


def test_different_constant_is_same_topology_but_different_kernel():
    other = _program(
        [OpStep(0, "add", [1, 2], {}, 3),
         OpStep(1, "index_set", [3, 2], {"slices": 999}, 4)],
        feeds=(1, 2), outputs={"r": 4},
    )
    assert topology_name(other) == topology_name(_BASE)
    assert kernel_name(other, "i64") != kernel_name(_BASE, "i64")


def test_catalogue_groups_and_persists():
    root = tempfile.mkdtemp()
    catalogue = TopologyCatalogue(root=root)
    # 5 members of one topology (renamed ids), 1 of another.
    for k in range(5):
        catalogue.record(_shift(_BASE, k * 100))
    other = _program([OpStep(0, "mul", [1, 2], {}, 3)], feeds=(1, 2), outputs={"r": 3})
    catalogue.record(other)

    assert len(catalogue.entries) == 2
    base_sig = topology_signature(_BASE)
    assert catalogue.entries[base_sig]["members"] == 5
    catalogue.save()

    # A fresh catalogue over the same root reloads the master table.
    reloaded = TopologyCatalogue(root=root)
    assert reloaded.entries[base_sig]["members"] == 5
