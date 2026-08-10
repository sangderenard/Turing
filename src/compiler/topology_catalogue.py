"""A master catalogue of algorithm topologies.

Group region programs by *topology* -- the structural shape of their dataflow --
rather than by content. The signature is computed over a canonical relabeling of
value ids (first appearance order), so it is invariant to the value-id numbering
that shifts build to build. That is the property a content hash lacked: two
regions that compute the same algorithm over different data (and under different
id numbering) land in the same group.

This is the stable key the earlier reduction cache needed. It also collapses a
program that repeats one operation thousands of times (a decoder's per-state
writes, say) into a handful of distinct topologies -- a master table of the
algorithms actually present, each with a canonical name.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..common.tensors.accelerator_backends.artifact_cache import (
    repository_cache_root,
)


# Attributes whose *value* changes the operation's structure (kept in the
# signature); every other attribute is data (a constant, a subscript value) and
# is abstracted to its presence only, so "the same algorithm, different data"
# groups together.
_STRUCTURAL_ATTRS = ("dim", "reverse", "axis", "keepdims")

_SCHEMA = "turing-topology-catalogue-v1"


def _attr_signature(attrs: Mapping[str, Any], keep_data: bool = False) -> tuple:
    parts = []
    for key in sorted(attrs):
        if key in _STRUCTURAL_ATTRS or keep_data:
            parts.append((key, repr(attrs[key])))
        else:
            parts.append((key, "*"))  # data: presence only
    return tuple(parts)


def _canonical_form(program: Any, keep_data: bool = False) -> tuple:
    """A value-id-independent structural form of one region program.

    With ``keep_data`` the constant/subscript values are retained -- the form
    then identifies a byte-distinct *kernel* rather than the *topology*.
    """

    relabel: dict[int, int] = {}

    def canon(value_id: int) -> int:
        # First appearance order -- independent of the id's actual number.
        if value_id not in relabel:
            relabel[value_id] = len(relabel)
        return relabel[value_id]

    steps = tuple(
        (
            step.op_name,
            tuple(canon(int(v)) for v in step.input_ids),
            canon(int(step.result_id)),
            _attr_signature(dict(step.attrs), keep_data),
        )
        for step in getattr(program, "steps", ())
    )
    outputs = tuple(
        canon(int(v)) for v in getattr(program, "outputs", {}).values()
    )
    # Feeds that never appear in a step still shape the interface; append the
    # count of unreferenced feeds so two otherwise-identical bodies with
    # different fan-in are not merged.
    unreferenced = sum(1 for f in getattr(program, "feeds", ()) if int(f) not in relabel)
    return (steps, outputs, unreferenced)


def _digest(canonical: tuple) -> str:
    return hashlib.sha256(
        repr(canonical).encode("utf-8", errors="backslashreplace")
    ).hexdigest()


def topology_signature(program: Any) -> str:
    """A stable digest of a region's topology: the symmetry-invariant canonical
    formula (data abstracted), so commutative variants share one signature and
    therefore one catalogue group."""

    return _digest(canonical_formula(program, keep_data=False))


def kernel_signature(program: Any) -> str:
    """Like ``topology_signature`` but with the constants fixed -- one
    byte-distinct compiled kernel, still value-id- and symmetry-invariant."""

    return _digest(canonical_formula(program, keep_data=True))


def _attr_formula(attrs: Mapping[str, Any], keep_data: bool) -> str:
    """The attribute clause of one operator in a canonical formula.

    Structural attributes (that change what the operator computes) always show
    their value; data attributes (a constant, a subscript value) show their
    value only for a *kernel* formula (``keep_data``) and are abstracted to
    ``*`` for a *topology* formula, so "same algorithm, different data" reads
    the same at the topology level.
    """

    if not attrs:
        return ""
    parts = []
    for key in sorted(attrs):
        if key in _STRUCTURAL_ATTRS or keep_data:
            parts.append(f"{key}={_scalar_repr(attrs[key])}")
        else:
            parts.append(f"{key}=*")
    return "{" + ",".join(parts) + "}"


def _scalar_repr(value: Any) -> str:
    for accessor in (value, getattr(value, "data", None)):
        if accessor is None:
            continue
        try:
            return repr(int(accessor))
        except (TypeError, ValueError):
            pass
        item = getattr(accessor, "item", None)
        if callable(item):
            try:
                return repr(item())
            except Exception:
                pass
    text = repr(value)
    return text if len(text) <= 24 else text[:21] + "..."


# Operators whose arguments may be reordered without changing the result. Their
# argument lists are sorted in the canonical formula so ``add(a,b)`` and
# ``add(b,a)`` name the same thing -- a first, concrete use of an algebraic
# symmetry to tighten the canonical form (see ``algebraic_profile``).
_COMMUTATIVE = frozenset({
    "add", "mul", "bitand", "bitor", "bitxor",
    "minimum", "maximum", "equal", "not_equal",
    "logical_and", "logical_or",
})

# Above this many steps a nested formula stops being a readable "name"; fall
# back to a compact genus-level form. The cap also bounds the explicit-stack
# traversal below.
_FORMULA_STEP_CAP = 48


def _symmetry_colors(program: Any) -> dict:
    """Weisfeiler-Lehman color refinement -> a symmetry-invariant color per value.

    Iterative (an explicit fixed-point loop, no recursion). Each value's color is
    refined from its own operator/attrs, the colors of its inputs (as an
    unordered multiset for commutative operators, an ordered tuple otherwise),
    and the colors of its consumers -- so two values that are structurally
    interchangeable (e.g. the two operands of an ``add``) end up the same color,
    while values used differently stay distinct. Naming inputs by color order
    then makes ``add(a,b)`` and ``add(b,a)`` render identically.
    """

    steps = list(getattr(program, "steps", ()))
    steps_by_result = {int(s.result_id): s for s in steps}
    values: set[int] = set()
    for step in steps:
        values.add(int(step.result_id))
        values.update(int(i) for i in step.input_ids)
    for v in getattr(program, "outputs", {}).values():
        values.add(int(v))

    consumers: dict[int, list] = {v: [] for v in values}
    for step in steps:
        commutative = step.op_name in _COMMUTATIVE
        for position, input_id in enumerate(step.input_ids):
            consumers[int(input_id)].append(
                (step.op_name, "c" if commutative else position, int(step.result_id))
            )

    color = {
        v: (
            "leaf" if v not in steps_by_result
            else f"{steps_by_result[v].op_name}|{_attr_signature(dict(steps_by_result[v].attrs))}"
        )
        for v in values
    }

    # A small fixed number of rounds (WL-k): color refinement stabilizes in a
    # few passes on the shallow DAGs here, and an unbounded fixed point would
    # cost O(V^2) per region for no extra discrimination in practice.
    for _ in range(min(_WL_ROUNDS, len(values) + 1)):
        updated = {}
        for v in values:
            step = steps_by_result.get(v)
            if step is None:
                in_sig: Any = ()
            else:
                child = [color[int(i)] for i in step.input_ids]
                in_sig = tuple(sorted(child)) if step.op_name in _COMMUTATIVE else tuple(child)
            out_sig = tuple(sorted(
                (op, position, color[result]) for op, position, result in consumers[v]
            ))
            updated[v] = _digest((color[v], in_sig, out_sig))
        if len(set(updated.values())) == len(set(color.values())):
            color = updated
            break
        color = updated
    return color


_WL_ROUNDS = 4


def canonical_formula(program: Any, *, keep_data: bool = False) -> str:
    """A canonical, analyzable structural formula for a region program.

    Reads like a scientific name: a nested operator expression over the region's
    dataflow. Inputs are named ``in0, in1, ...`` in canonical first-appearance
    order (a shared input is simply the same name reused); a computed value used
    in more than one place becomes a ``&k`` let-binding so a DAG is written once,
    not duplicated. Commutative operators have their arguments sorted, so an
    algebraic symmetry does not split one algorithm into two names. Derived from
    the value-id-invariant canonical order, so the same algorithm always renders
    identically; the exact ``topology_signature`` remains the unique key.

    No language recursion: the traversal uses an explicit work stack. Regions
    larger than ``_FORMULA_STEP_CAP`` steps use a compact genus-level fallback.

    Example (topology): ``index_set(add(in0,in1),in1){slices=*}`` -- a subscript
    store of ``a+b`` at ``b``. Kernel form keeps the constant: ``{slices=2}``.
    """

    steps = list(getattr(program, "steps", ()))
    if len(steps) > _FORMULA_STEP_CAP:
        histogram = Counter(step.op_name for step in steps)
        slug = "-".join(f"{op}{n}" for op, n in sorted(histogram.items())) or "empty"
        # Raw canonical form (not the formula) to avoid recursing into the
        # signature, which now rides on this function.
        digest = _digest(_canonical_form(program, keep_data=keep_data))
        return f"{slug}#{len(steps)}~{digest[:8]}"

    steps_by_result = {int(s.result_id): s for s in steps}
    uses: Counter = Counter()
    for step in steps:
        for value_id in step.input_ids:
            uses[int(value_id)] += 1
    output_ids = [int(v) for v in getattr(program, "outputs", {}).values()]
    for value_id in output_ids:
        uses[value_id] += 1

    reference: dict[int, str] = {}     # value_id -> its rendered reference
    definitions: list[str] = []        # "&k=expr" for shared computed values

    # Name inputs by symmetry-invariant color, not traversal order, so
    # commutative variants render identically. Interchangeable (same-color)
    # inputs make the formula invariant to their relative order regardless of
    # the id tie-break.
    colors = _symmetry_colors(program)
    producers = set(steps_by_result)
    feed_values = {int(i) for s in steps for i in s.input_ids} - producers
    feed_values |= {v for v in output_ids if v not in producers}
    feed_name = {
        value_id: f"in{index}"
        for index, value_id in enumerate(
            sorted(feed_values, key=lambda f: (colors.get(f, ""), f))
        )
    }

    def feed_reference(value_id: int) -> str:
        return feed_name.get(value_id, "in?")

    # Explicit-stack post-order: children resolve before their parent, so every
    # value's inputs already have a reference when its inline form is built.
    for root in output_ids:
        stack = [(int(root), False)]
        while stack:
            value_id, expanded = stack.pop()
            value_id = int(value_id)
            if value_id in reference:
                continue
            step = steps_by_result.get(value_id)
            if step is None:
                reference[value_id] = feed_reference(value_id)
                continue
            if not expanded:
                stack.append((value_id, True))
                for input_id in reversed(step.input_ids):
                    if int(input_id) not in reference:
                        stack.append((int(input_id), False))
                continue
            argument_refs = [reference[int(i)] for i in step.input_ids]
            if step.op_name in _COMMUTATIVE:
                argument_refs = sorted(argument_refs)
            inline = (
                f"{step.op_name}({','.join(argument_refs)})"
                f"{_attr_formula(dict(step.attrs), keep_data)}"
            )
            if uses[value_id] > 1:
                label = f"&{len(definitions)}"
                definitions.append(f"{label}={inline}")
                reference[value_id] = label
            else:
                reference[value_id] = inline

    body = ";".join(sorted(reference[v] for v in output_ids)) or "()"
    if definitions:
        return ";".join(definitions) + ";" + body
    return body


def topology_name(program: Any) -> str:
    """The topology-level (``T:``) name: the canonical formula with data
    abstracted. The algorithm, independent of its constants and bindings."""

    return f"T:{canonical_formula(program, keep_data=False)}"


def kernel_name(program: Any, dtype: str) -> str:
    """The kernel-level (``K:``) name: a byte-distinct compiled body -- the
    topology plus its concrete constants and working dtype."""

    return f"K:{canonical_formula(program, keep_data=True)}@{dtype}"


def invocation_name(
    kernel: str, input_slots, output_slots
) -> str:
    """The invocation-level (``I:``) name: one call of a kernel bound to specific
    resident field slots. The kernel is the method; the slots are the receiver."""

    ins = ",".join(f"s{int(s)}" for s in input_slots)
    outs = ",".join(f"s{int(s)}" for s in output_slots)
    return f"I:{kernel}<in:{ins};out:{outs}>"


# Algebraic laws each operator obeys. Only operators with meaningful structure
# are listed; the laws drive both symmetry-aware naming and the number-system
# classification below.
_OP_LAWS: Mapping[str, frozenset] = {
    "add": frozenset({"commutative", "associative", "identity", "invertible"}),
    "sub": frozenset({"identity"}),
    "mul": frozenset({"commutative", "associative", "identity", "absorbing"}),
    "bitand": frozenset({"commutative", "associative", "idempotent", "absorbing", "identity"}),
    "bitor": frozenset({"commutative", "associative", "idempotent", "identity"}),
    "bitxor": frozenset({"commutative", "associative", "identity", "self_inverse"}),
    "minimum": frozenset({"commutative", "associative", "idempotent"}),
    "maximum": frozenset({"commutative", "associative", "idempotent"}),
    "logical_and": frozenset({"commutative", "associative", "idempotent"}),
    "logical_or": frozenset({"commutative", "associative", "idempotent"}),
    "neg": frozenset({"involution"}),
    "invert": frozenset({"involution"}),
}

# A region whose *algebraic* operators all fall inside one of these closed sets
# computes in that number system / structure -- a math-system assertion.
_DOMAINS: tuple = (
    ("GF(2) boolean ring", frozenset({"bitand", "bitor", "bitxor", "invert", "shl", "shr"})),
    ("ring Z/2^n (modular integer arithmetic)", frozenset({"add", "sub", "mul", "neg"})),
    ("min/max semilattice", frozenset({"minimum", "maximum"})),
)

# Addressing / memory / comparison / reduction carriers -- not part of the
# algebraic-domain classification (they move or compare values, they are not the
# arithmetic the number system is about).
_NON_ALGEBRAIC: frozenset = frozenset({
    "index_set", "IndexedStore", "Indexed", "gather", "where", "tensor_from_list",
    "reshape", "tobytes", "less", "less_equal", "greater", "greater_equal",
    "equal", "not_equal", "sum", "mean", "sign", "floor", "ceil", "trunc", "round",
})

_ANALYZABLE_SYMMETRIES = frozenset(
    {"commutative", "associative", "idempotent", "involution", "self_inverse"}
)


def algebraic_profile(program: Any) -> dict:
    """Assert the algebraic character of a region from its operators' laws.

    Returns the laws present, the symmetries exhibited, and -- when the region's
    arithmetic operators are closed within a known structure -- the number
    system it computes in (GF(2), the ring Z/2^n, a lattice). ``domain`` is
    ``"mixed"`` when algebraic ops span structures and ``"none"`` when the region
    is pure addressing/comparison with no arithmetic.
    """

    operators = tuple(sorted({step.op_name for step in getattr(program, "steps", ())}))
    laws = {op: sorted(_OP_LAWS[op]) for op in operators if op in _OP_LAWS}
    symmetries = tuple(sorted(
        {law for law_set in laws.values() for law in law_set} & _ANALYZABLE_SYMMETRIES
    ))
    algebraic = frozenset(op for op in operators if op not in _NON_ALGEBRAIC)
    domain, closed = "none", False
    if algebraic:
        domain, closed = "mixed", False
        for name, op_set in _DOMAINS:
            if algebraic <= op_set:
                domain, closed = name, True
                break
    return {
        "operators": list(operators),
        "laws": laws,
        "symmetries": list(symmetries),
        "domain": domain,
        "closed": closed,
    }


class TopologyCatalogue:
    """A persistent master table keyed by topology signature.

    Records one representative name per distinct topology and how many members
    it has seen. The manifest is a plain JSON map so the catalogue is
    inspectable and survives across builds.
    """

    def __init__(self, root: str | Path | None = None):
        base = Path(root).expanduser().resolve() if root else repository_cache_root()
        self.directory = base / "algorithm-catalogue"
        self.entries: dict[str, dict[str, Any]] = {}
        self._load()

    @property
    def manifest_path(self) -> Path:
        return self.directory / "topologies.json"

    def _load(self) -> None:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if payload.get("schema") == _SCHEMA:
                self.entries = dict(payload.get("topologies", {}))
        except (OSError, ValueError):
            self.entries = {}

    def classify(self, program: Any) -> tuple[str, str]:
        """Return ``(signature, name)`` for a program without recording it."""

        return topology_signature(program), topology_name(program)

    def record(self, program: Any) -> str:
        """Add ``program`` to the catalogue; return its topology signature.

        Idempotent per signature: the first program of a topology defines its
        name and op histogram; later members only bump the count.
        """

        signature = topology_signature(program)
        entry = self.entries.get(signature)
        if entry is None:
            profile = algebraic_profile(program)
            self.entries[signature] = {
                "name": topology_name(program),
                "op_histogram": dict(
                    Counter(s.op_name for s in getattr(program, "steps", ()))
                ),
                "step_count": len(getattr(program, "steps", ())),
                # The math-system assertion for this topology.
                "domain": profile["domain"],
                "closed": profile["closed"],
                "symmetries": profile["symmetries"],
                "members": 1,
            }
        else:
            entry["members"] = int(entry.get("members", 0)) + 1
        return signature

    def record_all(self, programs) -> "TopologyCatalogue":
        for program in programs:
            self.record(program)
        return self

    def save(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": _SCHEMA,
            "distinct_topologies": len(self.entries),
            "topologies": self.entries,
        }
        temporary = self.manifest_path.with_suffix(f".json.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, self.manifest_path)
        return self.manifest_path


def math_torture_programs():
    """Region programs that exercise the math operator space and the algebraic
    structures, for seeding the catalogue with a canonical named library rather
    than only whatever one build happens to contain.

    Covers every elementwise unary/binary operator on its own, plus curated
    compositions that name the number systems: a multiply-add (ring), an
    and/xor chain (GF(2)), a min/max chain (lattice), and involutions.
    """

    from ..common.tensors.fused_ir import (
        FusedProgram, OpStep, ELEMENTWISE_BINARY, ELEMENTWISE_UNARY,
    )

    programs = []

    def add(steps, output):
        programs.append(FusedProgram(
            version=1, feeds={1, 2, 3}, steps=list(steps), outputs={"r": output},
        ))

    for op in sorted(ELEMENTWISE_BINARY):
        add([OpStep(0, op, [1, 2], {}, 4)], 4)
    for op in sorted(ELEMENTWISE_UNARY):
        add([OpStep(0, op, [1], {}, 4)], 4)

    # Algebraic-structure witnesses -- each names a number system in the profile.
    add([OpStep(0, "mul", [1, 2], {}, 4), OpStep(1, "add", [4, 3], {}, 5)], 5)      # ring: a*b+c
    add([OpStep(0, "bitand", [1, 2], {}, 4), OpStep(1, "bitxor", [4, 3], {}, 5)], 5)  # GF(2)
    add([OpStep(0, "minimum", [1, 2], {}, 4), OpStep(1, "maximum", [4, 3], {}, 5)], 5)  # lattice
    add([OpStep(0, "neg", [1], {}, 4), OpStep(1, "neg", [4], {}, 5)], 5)            # involution
    add([OpStep(0, "invert", [1], {}, 4), OpStep(1, "invert", [4], {}, 5)], 5)      # involution
    # Subscript get/set (the reference-operator pair).
    add([OpStep(0, "index_set", [1, 2], {"slices": 0}, 4)], 4)
    return programs


def seed_catalogue_with_math_torture(root=None) -> "TopologyCatalogue":
    """Populate (and persist) a catalogue from the math torture programs."""

    catalogue = TopologyCatalogue(root=root)
    catalogue.record_all(math_torture_programs())
    catalogue.save()
    return catalogue


__all__ = [
    "topology_signature",
    "kernel_signature",
    "canonical_formula",
    "topology_name",
    "kernel_name",
    "invocation_name",
    "algebraic_profile",
    "math_torture_programs",
    "seed_catalogue_with_math_torture",
    "TopologyCatalogue",
]
