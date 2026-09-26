"""Give an elementwise binary operation a broadcast descriptor.

`_tensor_descriptor` knew how to follow a unary op, a cast, a subscript, a
store and a loop result, but had no rule for `a * b`.  Authored source that
indexes and slices carries explicit shapes at every step, so that gap stayed
invisible.  A tensorized source is almost entirely broadcasting binary ops --
`index.unsqueeze(-1) == index.unsqueeze(-2)` is how it builds an identity
matrix -- so without this rule nearly every intermediate has no shape, and the
first function to return one hands its caller a rank-0 scalar where a matrix
belongs.
"""

from pathlib import Path

ANCHOR = """        if operation in {
            "neg", "abs", "sin", "cos", "tan", "exp", "log", "sqrt",
            "tanh", "clone", "copy", "identity", "real", "imag", "conj",
        }:"""

BINARY_RULE = '''        if operation in _ELEMENTWISE_BINARY_OPERATIONS:
            operands = tuple(
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role).casefold() in {
                    "lhs", "rhs", "left", "right", "operand", "value", "other",
                }
                and int(parent) in graph.G
            )
            if len(operands) == 2:
                sides = []
                for operand in operands:
                    side = _tensor_descriptor(graph, operand, seen)
                    if side is None:
                        # An authored scalar literal is rank 0, the same
                        # reading `_operand_descriptor` gives it.
                        node = graph.G.nodes[operand]
                        if str(node.get("type")) in {
                            "Constant", "Const", "const",
                        }:
                            literal = node.get("constant")
                            if literal is None:
                                literal = (
                                    node.get("attributes") or {}
                                ).get("value")
                            if isinstance(literal, (int, float, bool)):
                                side = {"shape": (), "dtype": "float64"}
                    sides.append(side)
                if all(side is not None for side in sides) and all(
                    descriptor_states_a_shape(side) for side in sides
                ):
                    left_shape = tuple(sides[0].get("shape") or ())
                    right_shape = tuple(sides[1].get("shape") or ())
                    rank = max(len(left_shape), len(right_shape))
                    left_aligned = (1,) * (rank - len(left_shape)) + left_shape
                    right_aligned = (
                        (1,) * (rank - len(right_shape)) + right_shape
                    )
                    broadcast: list[int] = []
                    compatible = True
                    for left_extent, right_extent in zip(
                        left_aligned, right_aligned
                    ):
                        if left_extent == right_extent:
                            broadcast.append(int(left_extent))
                        elif left_extent == 1:
                            broadcast.append(int(right_extent))
                        elif right_extent == 1:
                            broadcast.append(int(left_extent))
                        else:
                            # Not broadcastable: say nothing rather than
                            # invent an extent neither operand has.
                            compatible = False
                            break
                    if compatible:
                        dtype = next((
                            str(side.get("dtype"))
                            for side in sides
                            if str(side.get("dtype") or "unknown") != "unknown"
                        ), "float64")
                        return {
                            "shape": tuple(broadcast),
                            "dtype": dtype,
                            "rank": len(broadcast),
                        }
''' + ANCHOR

SET_ANCHOR = "def descriptor_states_a_shape(descriptor: Any) -> bool:"

OPERATION_SET = '''# Elementwise operations whose result shape is the broadcast of its operands.
# ``matmul`` is deliberately absent: it contracts rather than broadcasts and
# has its own rule where it is lowered.
_ELEMENTWISE_BINARY_OPERATIONS = frozenset({
    "add", "sub", "mul", "div", "truediv", "floordiv", "mod", "pow",
    "eq", "ne", "lt", "le", "gt", "ge",
    "maximum", "minimum", "logical_and", "logical_or", "logical_xor",
    "bitwise_and", "bitwise_or", "bitwise_xor",
})


''' + SET_ANCHOR

path = Path(__file__).resolve().parents[2] / "src/compiler/glsl_deployment_strategy.py"
text = path.read_text(encoding="utf-8")
assert text.count(ANCHOR) == 1, text.count(ANCHOR)
text = text.replace(ANCHOR, BINARY_RULE)
assert text.count(SET_ANCHOR) == 1, text.count(SET_ANCHOR)
text = text.replace(SET_ANCHOR, OPERATION_SET)
path.write_text(text, encoding="utf-8")
print("elementwise binary operations broadcast their operand shapes")
