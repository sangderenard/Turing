"""Build a Python AST from the supported GLSL source slice.

This is the inspectable alternate route to repository SSA.  It shares the
GLSL tokenizer and lexical tables with direct SSA lowering, but produces only
ordinary Python AST nodes already understood by ProcessGraph's AST ingestion.
Composite GLSL conveniences are expanded while the AST is built; no GLSL-only
operator node is retained.

Vector and struct types enter as the compiler's existing record faculty: each
``vecN``/``ivecN``/``uvecN``/``bvecN`` and each source ``struct`` the document
uses becomes one authored ``@dataclass`` record whose fields carry exact GLSL
widths.  Vector arithmetic, swizzles and vector built-ins are expanded here
into per-component scalar operations on those records, so no vector operator
survives into the graph.  ``out``/``inout`` helper parameters become
additional helper results unpacked by the caller.  Anything this reader cannot
represent remains a named shortfall.
"""

from __future__ import annotations

import ast
import copy
import keyword
from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from .glsl_source_ingestion import (
    GLSLSourceShortfall,
    glsl_number_literal,
    _ExpressionError,
    _GLOBAL,
    _dtype,
    _QUALIFIERS,
    _SCALAR_TYPES,
    _Token,
    _main_body,
    _statements,
    _tokenize,
    _without_comments,
)
from .glsl_source_tables import (
    GLSL_CASTS,
    GLSL_DIRECT_CALLS,
    GLSL_UNARY_TO_SSA,
    WEBGL_UNLOWERED_CALLS,
)


@dataclass(frozen=True)
class GLSLASTLoweringResult:
    module: ast.Module
    shortfalls: tuple[GLSLSourceShortfall, ...]
    #: ``(member, lane dtype, extent, stride)`` for each storage-buffer member
    #: the entry takes, in declaration order; extent is None for a
    #: runtime-sized ``T name[]``, N for ``T name[N]`` and 1 for a scalar
    #: member.  Each is an AbstractTensor parameter, caller-owned memory: a
    #: rank-1 span when stride is None, else its std430 region as a rank-2
    #: ``[extent, stride]`` span of lanes.  That is what the extraction
    #: contract's ``program_abi`` declares for it.
    storage_buffers: tuple[tuple[str, str, int | None, int | None], ...] = ()
    #: ``(name, texel dtype, dims)`` for each image or sampler the entry
    #: takes: a caller-owned one-channel texture, a rank-``dims`` tensor in
    #: ``[z, y, x]`` order (``_source_textures``).
    textures: tuple[tuple[str, str, int], ...] = ()

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    def shortfall_report(self) -> str:
        if not self.shortfalls:
            return "GLSL source to AST: complete"
        return "GLSL source to AST shortfalls:\n" + "\n".join(
            f"- {item.format()}" for item in self.shortfalls
        )


_BINARY_AST = {
    "+": ast.Add,
    "-": ast.Sub,
    "*": ast.Mult,
    "/": ast.Div,
    "%": ast.Mod,
    "<<": ast.LShift,
    ">>": ast.RShift,
    "&": ast.BitAnd,
    "|": ast.BitOr,
    "^": ast.BitXor,
}
_BOOLEAN_AST = {"&&": ast.And, "||": ast.Or}
_COMPARE_AST = {
    "==": ast.Eq,
    "!=": ast.NotEq,
    "<": ast.Lt,
    "<=": ast.LtE,
    ">": ast.Gt,
    ">=": ast.GtE,
}
_PRECEDENCE = {
    "||": 1, "&&": 2, "|": 3, "^": 4, "&": 5,
    "==": 6, "!=": 6, "<": 7, "<=": 7, ">": 7, ">=": 7,
    "<<": 8, ">>": 8, "+": 9, "-": 9, "*": 10, "/": 10, "%": 10,
}

# GLSL vector families and their scalar component types.
_VECTOR_COMPONENT = {"vec": "float", "ivec": "int", "uvec": "uint", "bvec": "bool"}
_VECTOR_PREFIX = {component: prefix for prefix, component in _VECTOR_COMPONENT.items()}
_COMPONENTS = "xyzw"
_SWIZZLE = {letter: index for names in ("xyzw", "rgba", "stpq")
            for index, letter in enumerate(names)}
# Built-ins that GLSL applies per component to vector operands.  Each is
# lowered by the existing scalar rules below, once per component.
_COMPONENTWISE_CALLS = frozenset({
    "abs", "acos", "acosh", "asin", "asinh", "atan", "atanh", "ceil",
    "clamp", "cos", "cosh", "exp", "exp2", "floor", "inversesqrt", "isinf", "isnan",
    "log", "max", "min", "mix", "mod", "pow", "round", "sign", "sin", "sinh",
    "smoothstep", "sqrt", "step", "tan", "tanh", "trunc",
})
_PARAMETER_QUALIFIERS = {"in", "out", "inout"}
# ``op=`` assignment spellings; the operator is the spelling without '='.
_ASSIGNMENTS = frozenset({
    "=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "|=", "^=",
})
# GLSL bit reinterpretations -> (``bit_reinterpretation`` function, result
# component type).  Each is the compiler's direct ``BitCast``.
_BIT_REINTERPRETATIONS = {
    "floatBitsToUint": ("float_bits_to_uint", "uint"),
    "floatBitsToInt": ("float_bits_to_int", "int"),
    "uintBitsToFloat": ("uint_bits_to_float", "float"),
    "intBitsToFloat": ("int_bits_to_float", "float"),
}
# Buffer atomics and the AbstractTensor atomic update each one is
# (``indexing.ATOMIC_UPDATE_OPERATIONS``).
_BUFFER_ATOMICS = {"atomicAdd": "atomic_add", "atomicOr": "atomic_or"}
# Image atomics: the same declared atomic updates, on one texel.
_BUFFER_ATOMICS_IMAGE = {"imageAtomicAdd": "atomic_add", "imageAtomicOr": "atomic_or"}
_TEXTURE_CALLS = frozenset({*_BUFFER_ATOMICS_IMAGE, "imageLoad", "imageStore", "texelFetch"})
# A storage-buffer block: ``layout(std430, binding=N) buffer Name { ... };``
_BUFFER_BLOCK = re.compile(
    r"(?:layout\s*\([^)]*\)\s*)?"
    r"(?:(?:readonly|writeonly|coherent|restrict|volatile)\s+)*"
    r"\bbuffer\s+(?P<block>[A-Za-z_]\w*)\s*\{(?P<body>[^{}]*)\}"
    r"\s*(?P<instance>[A-Za-z_]\w*)?\s*;"
)
# GLSL scalar element type -> the canonical dtype a span declares.
_BUFFER_ELEMENT_DTYPES = {"float": "float32", "int": "int32", "uint": "uint32"}


def _vector_shape(glsl_type: str | None) -> tuple[str, int] | None:
    match = re.fullmatch(r"([iub]?vec)([234])", str(glsl_type or ""))
    if match is None:
        return None
    return _VECTOR_COMPONENT[match.group(1)], int(match.group(2))


def _vector_type(component: str, width: int) -> str:
    return f"{_VECTOR_PREFIX[component]}{width}"


def _typed(node: ast.AST, glsl_type: str | None) -> ast.AST:
    node._glsl_type = glsl_type
    return node


def _type_of(node: ast.AST) -> str | None:
    return getattr(node, "_glsl_type", None)


def _name(value: str) -> ast.Name:
    return ast.Name(id=value, ctx=ast.Load())


def _call(name: str, args: Sequence[ast.expr]) -> ast.Call:
    return ast.Call(func=_name(name), args=list(args), keywords=[])


def _binary(left: ast.expr, operation: str, right: ast.expr) -> ast.expr:
    if operation in _BINARY_AST:
        return ast.BinOp(left=left, op=_BINARY_AST[operation](), right=right)
    if operation in _BOOLEAN_AST:
        return ast.BoolOp(op=_BOOLEAN_AST[operation](), values=[left, right])
    return ast.Compare(
        left=left,
        ops=[_COMPARE_AST[operation]()],
        comparators=[right],
    )


def _scalar_result_type(operation: str, left: str | None, right: str | None) -> str | None:
    if operation in _COMPARE_AST or operation in _BOOLEAN_AST:
        return "bool"
    if left == right or right is None:
        return left
    if left is None:
        return right
    return "float" if "float" in (left, right) else left


class _GLSLTypes:
    """The document's declared GLSL types and the records they become.

    ``structs`` maps a source struct name to its ordered ``(type, field)``
    members.  ``record`` returns the record class name for a vector or struct
    type and remembers it, so the module defines exactly the records used.
    """

    def __init__(self, structs: dict[str, list[tuple[str, str]]]):
        self.structs = structs
        self.records: dict[str, list[tuple[str, str]]] = {}

    def known(self, glsl_type: str | None) -> bool:
        return (glsl_type in _SCALAR_TYPES or _vector_shape(glsl_type) is not None
                or glsl_type in self.structs)

    def record(self, glsl_type: str) -> str | None:
        shape = _vector_shape(glsl_type)
        if shape is not None:
            component, width = shape
            self.records.setdefault(
                glsl_type, [(axis, component) for axis in _COMPONENTS[:width]])
            return glsl_type
        if glsl_type in self.structs:
            if glsl_type not in self.records:
                for member_type, _ in self.structs[glsl_type]:
                    if member_type not in _SCALAR_TYPES:
                        self.record(member_type)
                self.records[glsl_type] = [
                    (field, member_type)
                    for member_type, field in self.structs[glsl_type]]
            return glsl_type
        return None

    def fields(self, glsl_type: str) -> list[tuple[str, str]]:
        self.record(glsl_type)
        return self.records[glsl_type]

    def annotation(self, glsl_type: str) -> ast.expr:
        if glsl_type in _SCALAR_TYPES:
            return ast.Constant(_dtype(glsl_type))
        record = self.record(glsl_type)
        if record is None:
            raise ValueError(f"no record for GLSL type {glsl_type!r}")
        return _name(record)

    def class_definitions(self) -> list[ast.stmt]:
        definitions: list[ast.stmt] = []
        for record, fields in self.records.items():
            definitions.append(ast.ClassDef(
                name=record, bases=[], keywords=[],
                body=[ast.AnnAssign(
                    target=ast.Name(id=field, ctx=ast.Store()),
                    annotation=self.annotation(member_type),
                    value=None, simple=1) for field, member_type in fields],
                decorator_list=[_name("dataclass")],
            ))
        return definitions


class _Temporaries:
    def __init__(self, used=()):
        self.count = 0
        self.used = set(used)

    def fresh(self) -> str:
        while True:
            name = f"__glsl_value_{self.count}"
            self.count += 1
            if name not in self.used:
                self.used.add(name)
                return name


@dataclass
class _HelperSignature:
    parameters: list[tuple[str, str, str]]  # (qualifier, GLSL type, name)
    result_type: str
    # Storage-buffer members the helper reads or writes, directly or through
    # the helpers it calls.  Each is a caller-owned AbstractTensor handed in
    # by reference after the authored parameters; the compiler passes each
    # copy only the spans it touches.
    buffers: tuple[str, ...] = ()
    # Mutable globals (per-invocation state ``main`` owns) the helper reads
    # or writes, directly or through its callees: ``(name, GLSL type,
    # written)``.  Each follows the authored parameters; a written scalar,
    # vector or struct is ``inout`` (returned), an array is passed as its
    # list and written in place (``_source_mutable_globals``).
    globals: tuple[tuple[str, str, bool], ...] = ()

    @property
    def carried_parameters(self) -> list[tuple[str, str, str]]:
        return [*self.parameters, *(
            ("inout" if written and not _array_type(kind) else "in", kind, name)
            for name, kind, written in self.globals)]

    @property
    def outputs(self) -> list[int]:
        return [i for i, (qualifier, _, _) in enumerate(self.carried_parameters)
                if qualifier in {"out", "inout"}]


class _ReaderContext:
    """State shared by every expression and statement reader of a document."""

    def __init__(self, types: _GLSLTypes, helpers: dict[str, _HelperSignature],
                 temporaries: _Temporaries):
        self.types = types
        self.helpers = helpers
        self.temporaries = temporaries
        # GLSL type of each Python-side name in the current function.
        self.scope: dict[str, str | None] = {}
        # GLSL block scopes, innermost last: source name -> Python name.  A
        # declaration that reuses a name already declared in this function
        # gets a fresh Python name, because a Python function has one scope.
        self.lexical: list[dict[str, str]] = [{}]
        # Storage-buffer members: name -> element GLSL type.
        self.buffers: dict[str, str] = {}
        # Document constant arrays: name -> (element type, extent, width, lanes).
        self.constant_arrays: dict[str, tuple] = {}
        # Images and samplers: name -> (kind, texel component type, dims).
        self.textures: dict[str, tuple[str, str, int]] = {}
        # Scalar buffer members (``uint count;``), read and written at 0.
        self.buffer_scalars: set[str] = set()
        # Arrays of vectors or structs: name -> ``_std430_element`` layout.
        self.buffer_layouts: dict[str, tuple] = {}
        # ``bit_reinterpretation`` functions the document calls.
        self.imports: set[str] = set()
        # Local ``const int K = 8;`` integral constants, by source name: a
        # GLSL constant expression, so it may size a local array.
        self.constants: dict[str, int] = {}

    def resolve(self, name: str) -> str:
        for scope in reversed(self.lexical):
            if name in scope:
                return scope[name]
        return name


def _zero_value(context: _ReaderContext, glsl_type: str, token: _Token) -> ast.expr:
    """The value of an uninitialized GLSL local or ``out`` parameter.

    GLSL leaves such a value undefined, so every value is a conforming
    refinement; zero of the declared type is used and is deterministic.
    """

    if glsl_type in _SCALAR_TYPES:
        value = {"bool": False, "float": 0.0, "double": 0.0}.get(glsl_type, 0)
        node = ast.Constant(value=value)
        if glsl_type != "bool":
            node._glsl_scalar_dtype = {"float": "float32", "double": "float64",
                                       "uint": "uint32"}.get(glsl_type, "int32")
        return _typed(node, glsl_type)
    if not context.types.known(glsl_type):
        raise _ExpressionError(token, f"unknown GLSL type {glsl_type!r}")
    fields = context.types.fields(glsl_type)
    return _typed(_call(context.types.record(glsl_type), [
        _zero_value(context, member_type, token) for _, member_type in fields
    ]), glsl_type)


class _ASTExpressionParser:
    def __init__(self, tokens: Sequence[_Token], *, source_calls=None, context=None):
        self.tokens = list(tokens)
        self.index = 0
        self.source_calls = dict(source_calls or {})
        self.context = context
        # Statements a vector expansion or an out-parameter call must execute
        # before the expression that consumes it.  A conditionally evaluated
        # operand (right of && / ||, an arm of ?:) collects its own prelude,
        # which is placed inside the branch that evaluates it.
        self.prelude: list[ast.stmt] = []

    def parse(self) -> ast.expr:
        if not self.tokens:
            raise _ExpressionError(_Token("", 1, 1), "empty expression")
        value = self._expression(0)
        if self.index != len(self.tokens):
            token = self.tokens[self.index]
            raise _ExpressionError(token, f"unexpected token {token.text!r}")
        if not hasattr(value, "_glsl_source_location"):
            token=self.tokens[0]
            value._glsl_source_location=(token.line,token.column,token.text)
        return value

    def _peek(self, spelling: str | None = None) -> _Token | None:
        if self.index >= len(self.tokens):
            return None
        token = self.tokens[self.index]
        return token if spelling is None or token.text == spelling else None

    def _take(self, spelling: str | None = None) -> _Token:
        token = self._peek(spelling)
        if token is None:
            anchor = self.tokens[min(self.index, len(self.tokens) - 1)]
            found = "end of expression" if self._peek() is None else repr(
                self._peek().text
            )
            raise _ExpressionError(anchor, f"expected {spelling!r}, found {found}")
        self.index += 1
        return token

    # -- records and hoisting -------------------------------------------------

    def _stable(self, value: ast.expr, token: _Token) -> ast.expr:
        """An expression safe to read more than once: a name or field path.

        Anything else is evaluated once into a typed temporary first, so a
        per-component expansion never re-evaluates (or re-orders) its operand.
        """

        probe = value
        while isinstance(probe, ast.Attribute):
            probe = probe.value
        if isinstance(probe, ast.Name):
            return value
        glsl_type = _type_of(value)
        if self.context is None or not self.context.types.known(glsl_type):
            raise _ExpressionError(
                token, f"cannot hold a value of unknown GLSL type {glsl_type!r}")
        name = self.context.temporaries.fresh()
        self.context.scope[name] = glsl_type
        self.prelude.append(ast.AnnAssign(
            target=ast.Name(id=name, ctx=ast.Store()),
            annotation=self.context.types.annotation(glsl_type),
            value=value, simple=1))
        return _typed(_name(name), glsl_type)

    def _reuse(self, value: ast.expr) -> ast.expr:
        return _typed(copy.deepcopy(value), _type_of(value))

    def _components(self, value: ast.expr, width: int, token: _Token) -> list[ast.expr]:
        shape = _vector_shape(_type_of(value))
        stable = self._stable(value, token)
        if shape is None:
            return [self._reuse(stable) for _ in range(width)]
        component, actual = shape
        if actual != width:
            raise _ExpressionError(
                token, f"vector width mismatch: {_type_of(value)} where width {width}")
        return [_typed(ast.Attribute(value=self._reuse(stable), attr=axis,
                                     ctx=ast.Load()), component)
                for axis in _COMPONENTS[:width]]

    def _construct(self, glsl_type: str, parts: Sequence[ast.expr]) -> ast.expr:
        record = self.context.types.record(glsl_type)
        return _typed(_call(record, parts), glsl_type)

    def _vector_width(self, values: Sequence[ast.expr], token: _Token) -> tuple[str, int] | None:
        shapes = [_vector_shape(_type_of(v)) for v in values]
        shapes = [s for s in shapes if s is not None]
        if not shapes:
            return None
        if len({width for _, width in shapes}) != 1:
            raise _ExpressionError(token, "vector operands of different widths")
        return shapes[0]

    def _vector_binary(self, token: _Token, left: ast.expr, right: ast.expr) -> ast.expr:
        if token.text in _COMPARE_AST or token.text in _BOOLEAN_AST:
            raise _ExpressionError(
                token, f"vector operator {token.text!r} has no component-wise "
                "meaning; GLSL spells it with equal()/lessThan()/all()/any()")
        component, width = self._vector_width((left, right), token)
        lefts = self._components(left, width, token)
        rights = self._components(right, width, token)
        parts = [_typed(_binary(a, token.text, b), component)
                 for a, b in zip(lefts, rights)]
        return self._construct(_vector_type(component, width), parts)

    # -- expressions -------------------------------------------------------------

    def _expression(self, minimum: int) -> ast.expr:
        left = self._unary()
        while True:
            token = self._peek()
            precedence = -1 if token is None else _PRECEDENCE.get(token.text, -1)
            if precedence < minimum:
                break
            operation = self._take()
            if operation.text in _BOOLEAN_AST:
                right, branch = self._branch(lambda: self._expression(precedence + 1))
                if branch:
                    left = self._short_circuit(operation, left, right, branch)
                    continue
            else:
                right = self._expression(precedence + 1)
            if self.context is not None and (
                    _vector_shape(_type_of(left)) or _vector_shape(_type_of(right))):
                left = self._vector_binary(operation, left, right)
            else:
                result_type = _scalar_result_type(
                    operation.text, _type_of(left), _type_of(right))
                left = _typed(_binary(left, operation.text, right), result_type)
            left._glsl_source_location = (operation.line,operation.column,operation.text)
        if minimum == 0 and self._peek("?") is not None:
            question = self._take("?")
            if_true, true_branch = self._branch(lambda: self._expression(0))
            self._take(":")
            if_false, false_branch = self._branch(lambda: self._expression(0))
            result_type = _type_of(if_true) or _type_of(if_false)
            if true_branch or false_branch:
                # An arm that needs statements runs them only when selected.
                name = self._result_temporary(result_type, question)
                self.prelude.append(ast.If(
                    test=left,
                    body=true_branch + [ast.Assign(
                        targets=[ast.Name(id=name, ctx=ast.Store())], value=if_true)],
                    orelse=false_branch + [ast.Assign(
                        targets=[ast.Name(id=name, ctx=ast.Store())], value=if_false)]))
                return _typed(_name(name), result_type)
            left = _typed(ast.IfExp(test=left, body=if_true, orelse=if_false),
                          result_type)
        return left

    def _branch(self, read):
        """Read a conditionally evaluated operand with its own prelude."""

        outer = self.prelude
        self.prelude = []
        try:
            value = read()
            return value, self.prelude
        finally:
            self.prelude = outer

    def _result_temporary(self, glsl_type, token):
        if self.context is None or not self.context.types.known(glsl_type):
            raise _ExpressionError(
                token, f"cannot hold a value of unknown GLSL type {glsl_type!r}")
        name = self.context.temporaries.fresh()
        self.context.scope[name] = glsl_type
        self.prelude.append(ast.AnnAssign(
            target=ast.Name(id=name, ctx=ast.Store()),
            annotation=self.context.types.annotation(glsl_type),
            value=_zero_value(self.context, glsl_type, token), simple=1))
        return name

    def _short_circuit(self, operation, left, right, branch):
        """``a && b`` / ``a || b`` whose right operand needs statements.

        The right operand's statements run only when GLSL would evaluate it.
        """

        name = self._result_temporary("bool", operation)
        self.prelude.append(ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())], value=left))
        test = _name(name) if operation.text == "&&" else ast.UnaryOp(
            op=ast.Not(), operand=_name(name))
        self.prelude.append(ast.If(test=test, body=branch + [ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())], value=right)], orelse=[]))
        return _typed(_name(name), "bool")

    def _unary(self) -> ast.expr:
        token = self._peek()
        if token is not None and token.text in GLSL_UNARY_TO_SSA:
            self._take()
            operation = {
                "-": ast.USub,
                "!": ast.Not,
                "~": ast.Invert,
            }[token.text]
            operand = self._unary()
            shape = _vector_shape(_type_of(operand))
            if shape is not None and self.context is not None:
                if token.text == "!":
                    raise _ExpressionError(token, "'!' on a vector is not GLSL; use not()")
                component, width = shape
                return self._construct(_vector_type(component, width), [
                    _typed(ast.UnaryOp(op=operation(), operand=part), component)
                    for part in self._components(operand, width, token)])
            return _typed(ast.UnaryOp(op=operation(), operand=operand),
                          "bool" if token.text == "!" else _type_of(operand))
        if token is not None and token.text == "+":
            self._take()
            return self._unary()
        if token is not None and token.text in {"++", "--"} and self.context is not None:
            # ++x inside an expression: update first, then read the variable.
            self._take()
            operand = self._unary()
            self.prelude.append(self._increment(operand, token))
            return self._reuse(operand)
        value = self._primary()
        while self._peek() is not None:
            if self._peek("++") or self._peek("--"):
                if self.context is None:
                    break
                # x++ inside an expression: read the old value, then update.
                operator = self._take()
                old = self._result_temporary(_type_of(value), operator)
                self.prelude.append(ast.Assign(
                    targets=[ast.Name(id=old, ctx=ast.Store())], value=self._reuse(value)))
                self.prelude.append(self._increment(value, operator))
                value = _typed(_name(old), _type_of(value))
                continue
            if self._peek("["):
                bracket = self._take("[")
                index = self._expression(0)
                self._take("]")
                value = self._index(value, index, bracket)
            elif self._peek("."):
                self._take(".")
                field = self._take()
                if not re.fullmatch(r"[A-Za-z_]\w*", field.text):
                    raise _ExpressionError(field, "Expected field name")
                if (field.text == "length" and self._peek("(")
                        and str(_type_of(value) or "").endswith("[]")):
                    # ``buffer.length()``: the runtime extent of the
                    # caller's span, which Python spells ``len``.
                    self._take("(")
                    self._take(")")
                    value = _typed(_call("len", [value]), "int")
                    continue
                value = self._field(value, field)
            else:
                break
        return value

    def _increment(self, operand: ast.expr, token: _Token) -> ast.stmt:
        if not isinstance(operand, (ast.Name, ast.Attribute, ast.Subscript)):
            raise _ExpressionError(token, f"{token.text} requires an assignable operand")
        target = copy.deepcopy(operand)
        target.ctx = ast.Store()
        one = _typed(ast.Constant(value=1), "int")
        if _type_of(operand) == "uint":
            one._glsl_scalar_dtype = "uint32"
        elif _type_of(operand) == "float":
            one = _typed(ast.Constant(value=1.0), "float")
        return ast.AugAssign(target=target,
                             op=ast.Add() if token.text == "++" else ast.Sub(),
                             value=one)

    def _buffer_lanes(self, buffer: str, row: ast.expr, lanes: Sequence[int],
                      glsl_type: str, component: str) -> ast.expr:
        """The value held in ``lanes`` of one row of a std430 buffer.

        Each lane is one element of the caller's rank-2 span, ``buffer[row,
        lane]``.  The value is tagged with its lanes so a field, swizzle or
        store narrows to exactly the lanes it names.
        """

        def lane(offset):
            column = ast.Constant(value=int(offset))
            column._glsl_raw_literal = True
            return _typed(ast.Subscript(
                value=_name(buffer),
                slice=ast.Tuple(elts=[self._reuse(row), column], ctx=ast.Load()),
                ctx=ast.Load()), component)

        value = (lane(lanes[0]) if len(lanes) == 1
                 else self._construct(glsl_type, [lane(offset) for offset in lanes]))
        value._glsl_buffer_lanes = (buffer, row, tuple(lanes), glsl_type, component)
        return value

    def _buffer_element(self, buffer: str, index: ast.expr, token: _Token) -> ast.expr:
        """``buffer[index]`` for an array of vectors or of structs of vectors."""

        stride, fields, component = self.context.buffer_layouts[buffer]
        element = self.context.buffers[buffer]
        row = self._stable(index, token)
        if fields is None:
            width = (_vector_shape(element) or (component, 1))[1]
            return self._buffer_lanes(buffer, row, range(width), element, component)
        parts = []
        for kind, name in self.context.types.structs[element]:
            offset, _kind = fields[name]
            width = (_vector_shape(kind) or (component, 1))[1]
            parts.append(self._buffer_lanes(
                buffer, row, range(offset, offset + width), kind, component))
        value = self._construct(element, parts)
        value._glsl_buffer_struct = (buffer, row, fields, component)
        return value

    def _index(self, value: ast.expr, index: ast.expr, token: _Token) -> ast.expr:
        if (self.context is not None and isinstance(value, ast.Name)
                and value.id in self.context.buffer_layouts):
            return self._buffer_element(value.id, index, token)
        if (self.context is not None and isinstance(value, ast.Name)
                and value.id in self.context.constant_arrays):
            # ``B[i]`` of a constant array: lanes ``i * width + k`` of its
            # typed lane list.
            kind, _extent, width, _values = self.context.constant_arrays[value.id]
            if width == 1:
                return _typed(ast.Subscript(value=value, slice=index, ctx=ast.Load()), kind)
            row = self._stable(index, token)
            lanes = []
            for lane in range(width):
                stride = ast.Constant(value=width)
                stride._glsl_raw_literal = True
                offset = ast.Constant(value=lane)
                offset._glsl_raw_literal = True
                position = ast.BinOp(left=ast.BinOp(left=self._reuse(row), op=ast.Mult(),
                                                    right=stride),
                                     op=ast.Add(), right=offset)
                lanes.append(_typed(ast.Subscript(value=_name(value.id), slice=position,
                                                  ctx=ast.Load()), "float"))
            return self._construct(kind, lanes)
        glsl_type = _type_of(value)
        shape = _vector_shape(glsl_type)
        if shape is not None and self.context is not None:
            component, width = shape
            if isinstance(index, ast.Constant) and isinstance(index.value, int):
                if not 0 <= index.value < width:
                    raise _ExpressionError(token, "vector component index out of range")
                return _typed(ast.Attribute(value=value, attr=_COMPONENTS[index.value],
                                            ctx=ast.Load()), component)
            # A variable component index reads one record field selected by
            # the index: v[k] = k==0 ? v.x : k==1 ? v.y : ... (in range by
            # GLSL's own contract; the last component is the final arm).
            stable = self._stable(value, token)
            position = self._stable(index, token)
            selected = _typed(ast.Attribute(value=self._reuse(stable),
                                            attr=_COMPONENTS[width - 1], ctx=ast.Load()),
                              component)
            for axis in reversed(range(width - 1)):
                test = ast.Compare(left=self._reuse(position), ops=[ast.Eq()],
                                   comparators=[_typed(ast.Constant(value=axis), "int")])
                selected = _typed(ast.IfExp(
                    test=test,
                    body=_typed(ast.Attribute(value=self._reuse(stable),
                                              attr=_COMPONENTS[axis], ctx=ast.Load()),
                                component),
                    orelse=selected), component)
            # A store through the same index writes the selected component.
            selected._glsl_component_select = (stable, position, width)
            return selected
        element = None
        match = re.fullmatch(r"(\w+)\[\d*\]", str(glsl_type or ""))
        if match is not None:
            element = match.group(1)
        return _typed(ast.Subscript(value=value, slice=index, ctx=ast.Load()), element)

    def _field(self, value: ast.expr, field: _Token) -> ast.expr:
        texel = getattr(value, "_glsl_texel", None)
        if texel is not None:
            # A swizzle of a texel read names only the lanes its declared
            # format stores (``_texture_call``).
            component, channels = texel
            indices = [_SWIZZLE.get(letter) for letter in field.text]
            if (any(i is None or i >= channels for i in indices) or len(indices) > 4
                    or len({_family(letter) for letter in field.text}) != 1):
                raise _ExpressionError(
                    field, f"swizzle .{field.text} names lanes a {channels}-channel "
                           f"texel does not store")
            if len(indices) == 1:
                return _typed(value, component)
            stable = self._stable(_typed(value, component), field)
            return self._construct(_vector_type(component, len(indices)), [
                _typed(self._reuse(stable), component) for _ in indices])
        glsl_type = _type_of(value)
        shape = _vector_shape(glsl_type)
        struct_row = getattr(value, "_glsl_buffer_struct", None)
        if struct_row is not None:
            # A field of a buffer row: only that field's lanes are read.
            buffer, row, fields, component = struct_row
            if field.text not in fields:
                raise _ExpressionError(field, f"{glsl_type} has no field {field.text!r}")
            offset, kind = fields[field.text]
            width = (_vector_shape(kind) or (component, 1))[1]
            return self._buffer_lanes(
                buffer, row, range(offset, offset + width), kind, component)
        lanes_row = getattr(value, "_glsl_buffer_lanes", None)
        if lanes_row is not None and shape is not None and self.context is not None:
            # A swizzle of a buffer vector reads just the lanes it names.
            buffer, row, lanes, _kind, component = lanes_row
            indices = [_SWIZZLE.get(letter) for letter in field.text]
            if (any(i is None or i >= shape[1] for i in indices) or len(indices) > 4
                    or len({_family(letter) for letter in field.text}) != 1):
                raise _ExpressionError(field, f"invalid swizzle .{field.text} of {glsl_type}")
            return self._buffer_lanes(
                buffer, row, [lanes[i] for i in indices],
                component if len(indices) == 1 else _vector_type(component, len(indices)),
                component)
        if shape is not None and self.context is not None:
            component, width = shape
            indices = [_SWIZZLE.get(letter) for letter in field.text]
            if (any(i is None or i >= width for i in indices) or len(indices) > 4
                    or len({_family(letter) for letter in field.text}) != 1):
                raise _ExpressionError(field, f"invalid swizzle .{field.text} of {glsl_type}")
            if len(indices) == 1:
                return _typed(ast.Attribute(value=value, attr=_COMPONENTS[indices[0]],
                                            ctx=ast.Load()), component)
            stable = self._stable(value, field)
            result = self._construct(_vector_type(component, len(indices)), [
                _typed(ast.Attribute(value=self._reuse(stable), attr=_COMPONENTS[i],
                                     ctx=ast.Load()), component) for i in indices])
            result._glsl_swizzle = (stable, indices)
            return result
        member_type = None
        if self.context is not None and glsl_type in self.context.types.structs:
            member_type = dict(
                (name, kind) for kind, name in self.context.types.structs[glsl_type]
            ).get(field.text)
            if member_type is None:
                raise _ExpressionError(field, f"{glsl_type} has no field {field.text!r}")
        return _typed(ast.Attribute(value=value, attr=field.text, ctx=ast.Load()),
                      member_type)

    def _primary(self) -> ast.expr:
        token = self._take()
        if token.text == "(":
            value = self._expression(0)
            self._take(")")
            return value
        if token.text in {"true", "false"}:
            return _typed(ast.Constant(value=token.text == "true"), "bool")
        if token.text[0].isdigit() or token.text[0] == ".":
            value, dtype = glsl_number_literal(token.text)
            node = ast.Constant(value=value)
            node._glsl_scalar_dtype = {"u32": "uint32", "i32": "int32"}.get(dtype, dtype)
            node._glsl_source_location = (token.line,token.column,token.text)
            return _typed(node, {"uint32": "uint", "float32": "float",
                                 "int32": "int"}[node._glsl_scalar_dtype])
        if not re.fullmatch(r"[A-Za-z_]\w*", token.text):
            raise _ExpressionError(token, f"expected value, found {token.text!r}")
        if self._peek("(") is None:
            if self.context is None:
                return _name(token.text)
            name = self.context.resolve(token.text)
            if name in self.context.buffer_scalars:
                # A scalar buffer member is element 0 of its one-element span.
                zero = _typed(ast.Constant(value=0), "int")
                return self._index(_typed(_name(name), self.context.scope.get(name)),
                                   zero, token)
            return _typed(_name(name), self.context.scope.get(name))
        self._take("(")
        args: list[ast.expr] = []
        if self._peek(")") is None:
            while True:
                args.append(self._expression(0))
                if self._peek(",") is None:
                    break
                self._take(",")
        self._take(")")
        return self._lower_call(token, args)

    @staticmethod
    def _arity(token: _Token, args: Sequence[ast.expr], expected: int) -> None:
        if len(args) != expected:
            raise _ExpressionError(
                token,
                f"{token.text} expects {expected} operands, got {len(args)}",
            )

    def _texture_call(self, token: _Token, args: Sequence[ast.expr]) -> ast.expr:
        """Texel reads, writes and atomics on caller-owned texture memory.

        A one-channel texture is a rank-``dims`` tensor in ``[z, y, x]``
        order; coordinate ``q`` addresses texel ``(q.z, q.y, q.x)``.
        ``imageAtomicAdd``/``imageAtomicOr`` are AbstractTensor's declared
        atomic updates on that texel; ``imageLoad`` and ``texelFetch`` (mip
        0 only) read the one stored channel, through a swizzle that names
        only it; ``imageStore`` writes the value's first component.
        """

        name = token.text
        expected = {"texelFetch": 3, "imageLoad": 2, "imageStore": 3}.get(name, 3)
        self._arity(token, args, expected)
        texture = args[0]
        if not (isinstance(texture, ast.Name) and texture.id in self.context.textures):
            raise _ExpressionError(token, f"{name} needs an image or sampler uniform")
        kind, component, dims = self.context.textures[texture.id]
        if (kind == "sampler") != (name == "texelFetch"):
            raise _ExpressionError(token, f"{name} does not apply to a {kind}")
        coordinate = args[1]
        parts = (self._components(coordinate, dims, token) if dims > 1
                 else [coordinate])
        texel = ast.Tuple(elts=list(reversed(parts)), ctx=ast.Load())
        if name in _BUFFER_ATOMICS_IMAGE:
            method = ast.Attribute(value=_name("AbstractTensor"),
                                   attr=_BUFFER_ATOMICS_IMAGE[name], ctx=ast.Load())
            return _typed(ast.Call(func=method, args=[_name(texture.id), texel, args[2]],
                                   keywords=[]), component)
        if name == "imageStore":
            value = args[2]
            first = (self._components(value, 4, token)[0]
                     if _vector_shape(_type_of(value)) is not None else value)
            self.prelude.append(ast.Assign(
                targets=[ast.Subscript(value=_name(texture.id), slice=texel,
                                       ctx=ast.Store())],
                value=first))
            return _typed(ast.Constant(value=None), "void")
        if name == "texelFetch":
            lod = args[2]
            if not (isinstance(lod, ast.Constant) and lod.value == 0):
                raise _ExpressionError(token, "texelFetch reads mip level 0 only")
        # The texel holds the one channel its declared format stores.  GLSL
        # types the read ``gvec4``; the lanes the format does not store are
        # not in caller memory, so only a swizzle of the stored lane is
        # lowered (``_field``), and the read is not used whole.
        if self._peek(".") is None:
            raise _ExpressionError(
                token, f"{name} of a one-channel texture is lowered through a "
                       f"swizzle of its stored lane (.r/.x)")
        read = _typed(ast.Subscript(value=_name(texture.id), slice=texel,
                                    ctx=ast.Load()), component)
        read._glsl_texel = (component, 1)
        return read

    def _bit_reinterpretation(self, token: _Token, args: Sequence[ast.expr]) -> ast.expr:
        """``floatBitsToUint(x)``: the same bits typed as the other scalar.

        A scalar becomes one ``bit_reinterpretation`` call (the compiler's
        ``BitCast``); a vector reinterprets each component.
        """

        self._arity(token, args, 1)
        function, component = _BIT_REINTERPRETATIONS[token.text]
        self.context.imports.add(function)
        shape = _vector_shape(_type_of(args[0]))
        if shape is None:
            return _typed(_call(function, [args[0]]), component)
        parts = [_typed(_call(function, [part]), component)
                 for part in self._components(args[0], shape[1], token)]
        return self._construct(_vector_type(component, shape[1]), parts)

    def _buffer_atomic(self, token: _Token, args: Sequence[ast.expr]) -> ast.expr:
        """``atomicAdd(words[i], v)``: AbstractTensor's declared atomic update.

        The first operand names one element of a storage buffer; the call
        becomes ``AbstractTensor.atomic_add(words, i, v)``, which returns the
        element's old value and writes the caller's span in place.  Its
        pursued body is one ``source_atomic_site``.
        """

        self._arity(token, args, 2)
        target, value = args
        if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                and target.value.id in self.context.buffers):
            raise _ExpressionError(
                token, f"{token.text} needs a storage buffer element as its first operand")
        element = self.context.buffers[target.value.id]
        method = ast.Attribute(value=_name("AbstractTensor"),
                               attr=_BUFFER_ATOMICS[token.text], ctx=ast.Load())
        return _typed(ast.Call(func=method, args=[
            _name(target.value.id), target.slice, value], keywords=[]), element)

    def _helper_call(self, token: _Token, signature: _HelperSignature,
                     args: Sequence[ast.expr]) -> ast.expr:
        self._arity(token, args, len(signature.parameters))
        # The mutable globals the helper touches follow its authored
        # arguments (the caller's current values); then the storage buffers,
        # each the caller's own span, handed by reference.
        args = [*args, *(
            _typed(_name(self.context.resolve(name)), kind)
            for name, kind, _written in signature.globals)]
        call = _call(token.text, [*args, *map(_name, signature.buffers)])
        outputs = signature.outputs
        if not outputs:
            return _typed(call, signature.result_type)
        targets: list[ast.expr] = []
        result = None
        if signature.result_type != "void":
            name = self.context.temporaries.fresh()
            self.context.scope[name] = signature.result_type
            targets.append(ast.Name(id=name, ctx=ast.Store()))
            result = _typed(_name(name), signature.result_type)
        for position in outputs:
            target = copy.deepcopy(args[position])
            if not isinstance(target, (ast.Name, ast.Attribute, ast.Subscript)):
                raise _ExpressionError(token, "an out argument must be assignable")
            target.ctx = ast.Store()
            targets.append(target)
        self.prelude.append(ast.Assign(
            targets=[targets[0] if len(targets) == 1
                     else ast.Tuple(elts=targets, ctx=ast.Store())],
            value=call))
        if result is None:
            return _typed(ast.Constant(value=None), "void")
        return result

    def _vector_constructor(self, token: _Token, args: Sequence[ast.expr]) -> ast.expr:
        component, width = _vector_shape(token.text)
        if len(args) == 1 and _vector_shape(_type_of(args[0])) is None:
            parts = self._components(args[0], width, token)
        else:
            parts = []
            for arg in args:
                shape = _vector_shape(_type_of(arg))
                parts.extend(self._components(arg, shape[1], token) if shape else [arg])
            if len(parts) != width:
                raise _ExpressionError(
                    token, f"{token.text} needs {width} components, got {len(parts)}")
        parts = [part if _type_of(part) in {component, None} or component == "float"
                 and _type_of(part) == "float"
                 else _typed(_call(component, [part]), component) for part in parts]
        return self._construct(token.text, parts)

    def _vector_call(self, token: _Token, args: Sequence[ast.expr],
                     vector: tuple[str, int]) -> ast.expr:
        name = token.text
        component, width = vector
        if name in {"dot", "length", "distance", "normalize"}:
            if name == "distance":
                self._arity(token, args, 2)
                difference = self._vector_binary(_Token("-", token.line, token.column), *args)
                return self._vector_call(_Token("length", token.line, token.column),
                                         [difference], vector)
            if name == "dot":
                self._arity(token, args, 2)
                lefts = self._components(args[0], width, token)
                rights = self._components(args[1], width, token)
            else:
                self._arity(token, args, 1)
                lefts = self._components(args[0], width, token)
                rights = [self._reuse(part) for part in lefts]
            total = _typed(_binary(lefts[0], "*", rights[0]), component)
            for a, b in zip(lefts[1:], rights[1:]):
                total = _typed(_binary(total, "+", _typed(_binary(a, "*", b), component)),
                               component)
            if name == "dot":
                return total
            magnitude = _typed(_call("sqrt", [total]), component)
            if name == "length":
                return magnitude
            # GLSL defines normalize(v) as v / length(v): divide each
            # component, never multiply by a reciprocal.
            divisor = self._stable(magnitude, token)
            return self._construct(_vector_type(component, width), [
                _typed(_binary(self._reuse(part), "/", self._reuse(divisor)), component)
                for part in self._components(args[0], width, token)])
        comparisons = {"lessThan": "<", "lessThanEqual": "<=", "greaterThan": ">",
                       "greaterThanEqual": ">=", "equal": "==", "notEqual": "!="}
        if name in comparisons:
            self._arity(token, args, 2)
            lefts = self._components(args[0], width, token)
            rights = self._components(args[1], width, token)
            return self._construct(_vector_type("bool", width), [
                _typed(_binary(a, comparisons[name], b), "bool")
                for a, b in zip(lefts, rights)])
        if name in {"any", "all"}:
            self._arity(token, args, 1)
            if component != "bool":
                raise _ExpressionError(token, f"{name} requires a bvec")
            parts = self._components(args[0], width, token)
            return _typed(ast.BoolOp(op=ast.Or() if name == "any" else ast.And(),
                                     values=parts), "bool")
        if name == "not":
            self._arity(token, args, 1)
            return self._construct(_vector_type("bool", width), [
                _typed(ast.UnaryOp(op=ast.Not(), operand=part), "bool")
                for part in self._components(args[0], width, token)])
        if name == "refract":
            # GLSL: k = 1 - eta*eta*(1 - dot(N,I)*dot(N,I));
            # k < 0 ? 0 : eta*I - (eta*dot(N,I) + sqrt(k))*N.
            self._arity(token, args, 3)
            incident = self._stable(args[0], token)
            normal = self._stable(args[1], token)
            eta = self._stable(args[2], token)
            cosine = self._stable(self._vector_call(
                _Token("dot", token.line, token.column),
                [self._reuse(normal), self._reuse(incident)], vector), token)
            def scalar(value):
                return _typed(ast.Constant(value=value), component)
            k = self._stable(_typed(_binary(scalar(1.0), "-", _typed(_binary(
                _typed(_binary(self._reuse(eta), "*", self._reuse(eta)), component), "*",
                _typed(_binary(scalar(1.0), "-", _typed(_binary(
                    self._reuse(cosine), "*", self._reuse(cosine)), component)), component)),
                component)), component), token)
            parts = []
            for i, n in zip(self._components(incident, width, token),
                            self._components(normal, width, token)):
                # sqrt(k) stays inside the selected arm: it is evaluated only
                # where GLSL evaluates it (k >= 0).
                factor = _typed(_binary(
                    _typed(_binary(self._reuse(eta), "*", self._reuse(cosine)), component),
                    "+", _typed(_call("sqrt", [self._reuse(k)]), component)), component)
                refracted = _typed(_binary(
                    _typed(_binary(self._reuse(eta), "*", i), component), "-",
                    _typed(_binary(factor, "*", n), component)), component)
                parts.append(_typed(ast.IfExp(
                    test=ast.Compare(left=self._reuse(k), ops=[ast.Lt()],
                                     comparators=[scalar(0.0)]),
                    body=scalar(0.0), orelse=refracted), component))
            return self._construct(_vector_type(component, width), parts)
        if name == "reflect":
            # GLSL: reflect(I, N) = I - 2.0 * dot(N, I) * N, in that order.
            self._arity(token, args, 2)
            incident = self._stable(args[0], token)
            normal = self._stable(args[1], token)
            scale = self._stable(_typed(_binary(
                _typed(ast.Constant(value=2.0), component), "*",
                self._vector_call(_Token("dot", token.line, token.column),
                                  [self._reuse(normal), self._reuse(incident)], vector)),
                component), token)
            return self._construct(_vector_type(component, width), [
                _typed(_binary(i, "-", _typed(_binary(self._reuse(scale), "*", n),
                                              component)), component)
                for i, n in zip(self._components(incident, width, token),
                                self._components(normal, width, token))])
        if name == "cross":
            self._arity(token, args, 2)
            if width != 3:
                raise _ExpressionError(token, "cross requires vec3")
            a = self._components(args[0], 3, token)
            b = self._components(args[1], 3, token)
            def term(i, j):
                return _typed(_binary(self._reuse(a[i]), "*", self._reuse(b[j])), component)
            return self._construct(_vector_type(component, 3), [
                _typed(_binary(term(j, k), "-", term(k, j)), component)
                for j, k in ((1, 2), (2, 0), (0, 1))])
        if name in _COMPONENTWISE_CALLS:
            columns = [self._components(arg, width, token) for arg in args]
            parts = []
            for i in range(width):
                part = self._lower_call(token, [column[i] for column in columns])
                parts.append(_typed(part, "bool" if name in {"isinf", "isnan"} else component))
            result_component = "bool" if name in {"isinf", "isnan"} else component
            return self._construct(_vector_type(result_component, width), parts)
        raise _ExpressionError(token, f"vector GLSL call {name!r} is not lowered")

    def _lower_call(self, token: _Token, args: Sequence[ast.expr]) -> ast.expr:
        name = token.text
        context = self.context
        if context is not None and name in _BUFFER_ATOMICS:
            return self._buffer_atomic(token, args)
        if context is not None and name in _BIT_REINTERPRETATIONS:
            return self._bit_reinterpretation(token, args)
        if context is not None and name in _TEXTURE_CALLS:
            return self._texture_call(token, args)
        if context is not None and name in context.helpers:
            return self._helper_call(token, context.helpers[name], args)
        if name in self.source_calls:
            self._arity(token, args, self.source_calls[name])
            return _call(name, args)
        if context is not None and _vector_shape(name) is not None:
            return self._vector_constructor(token, args)
        if context is not None and name in context.types.structs:
            fields = context.types.fields(name)
            self._arity(token, args, len(fields))
            return _typed(_call(context.types.record(name), args), name)
        if context is not None:
            vector = self._vector_width(args, token)
            if vector is not None:
                return self._vector_call(token, args, vector)
        if name in GLSL_CASTS:
            self._arity(token, args, 1)
            return _typed(_call(name, args), name)
        result_type = _type_of(args[0]) if args else None
        if name == "pow":
            self._arity(token, args, 2)
            return _typed(ast.BinOp(left=args[0], op=ast.Pow(), right=args[1]), result_type)
        if name == "exp2":
            # GLSL defines exp2(x) as 2 raised to x (its precision is the
            # implementation's); the compiler's Pow at x's own width.
            self._arity(token, args, 1)
            two = _typed(ast.Constant(value=2.0), "float")
            return _typed(ast.BinOp(left=two, op=ast.Pow(), right=args[0]), result_type)
        if name == "mod":
            self._arity(token, args, 2)
            return _typed(ast.BinOp(left=args[0], op=ast.Mod(), right=args[1]), result_type)
        if name in {"min", "max"}:
            self._arity(token, args, 2)
            comparison = ast.Lt() if name == "min" else ast.Gt()
            condition = ast.Compare(
                left=args[0], ops=[comparison], comparators=[args[1]]
            )
            return _typed(ast.IfExp(test=condition, body=args[0], orelse=args[1]),
                          result_type)
        if name == "clamp":
            self._arity(token, args, 3)
            lower = self._lower_call(_Token("max", token.line, token.column), args[:2])
            return self._lower_call(
                _Token("min", token.line, token.column), [lower, args[2]]
            )
        if name == "mix":
            self._arity(token, args, 3)
            return _typed(ast.BinOp(
                left=args[0],
                op=ast.Add(),
                right=ast.BinOp(
                    left=ast.BinOp(left=args[1], op=ast.Sub(), right=args[0]),
                    op=ast.Mult(),
                    right=args[2],
                ),
            ), result_type)
        if name == "step":
            self._arity(token, args, 2)
            condition = ast.Compare(
                left=args[1], ops=[ast.Lt()], comparators=[args[0]]
            )
            return _typed(ast.IfExp(
                test=condition,
                body=ast.Constant(value=0.0),
                orelse=ast.Constant(value=1.0),
            ), "float")
        if name == "smoothstep":
            self._arity(token, args, 3)
            ratio = ast.BinOp(
                left=ast.BinOp(left=args[2], op=ast.Sub(), right=args[0]),
                op=ast.Div(),
                right=ast.BinOp(left=args[1], op=ast.Sub(), right=args[0]),
            )
            t = self._lower_call(
                _Token("clamp", token.line, token.column),
                [ratio, ast.Constant(value=0.0), ast.Constant(value=1.0)],
            )
            return _typed(ast.BinOp(
                left=ast.BinOp(left=t, op=ast.Mult(), right=t),
                op=ast.Mult(),
                right=ast.BinOp(
                    left=ast.Constant(value=3.0),
                    op=ast.Sub(),
                    right=ast.BinOp(
                        left=ast.Constant(value=2.0), op=ast.Mult(), right=t
                    ),
                ),
            ), "float")
        if name == "inversesqrt":
            self._arity(token, args, 1)
            return _typed(ast.BinOp(
                left=ast.Constant(value=1.0),
                op=ast.Div(),
                right=_call("sqrt", args),
            ), result_type)
        if name == "length":
            self._arity(token, args, 1)
            return _typed(_call("sqrt", [
                ast.BinOp(left=args[0], op=ast.Mult(), right=args[0])
            ]), result_type)
        if name == "distance":
            self._arity(token, args, 2)
            return self._lower_call(
                _Token("length", token.line, token.column),
                [ast.BinOp(left=args[0], op=ast.Sub(), right=args[1])],
            )
        if name in GLSL_DIRECT_CALLS:
            return _typed(_call(GLSL_DIRECT_CALLS[name], args),
                          "bool" if name in {"isinf", "isnan"} else result_type)
        if name in WEBGL_UNLOWERED_CALLS:
            raise _ExpressionError(
                token,
                f"{name} has no exact existing SSA/ProcessGraph operation",
            )
        raise _ExpressionError(token, f"unsupported GLSL call {name!r}")


def _family(letter: str) -> int:
    for family, names in enumerate(("xyzw", "rgba", "stpq")):
        if letter in names:
            return family
    return -1


class _ASTStatementReader(_ASTExpressionParser):
    """Read statement syntax into the existing ordinary AST ingestion surface.

    There is no shader IR or direct SSA writer here. The public source compiler
    remains responsible for control flow, aliasing, types and identity records.
    """
    def __init__(self, tokens, *, reserved_names=(), source_calls=None, context=None):
        super().__init__(tokens, source_calls=source_calls, context=context)
        self.declarations = set(reserved_names)

    def expression(self, tokens):
        """Parse one expression; return it and the statements it needs first."""

        parser = _ASTExpressionParser(tokens, source_calls=self.source_calls,
                                      context=self.context)
        value = parser.parse()
        return value, parser.prelude

    def read_all(self, closing=None):
        body = []
        while self._peek() is not None and (closing is None or not self._peek(closing)):
            body.extend(self.statement())
        if closing:
            self._take(closing)
        return body

    def delimited(self, opening, closing):
        self._take(opening)
        begin = self.index
        depth = 1
        while depth:
            token = self._take()
            depth += (token.text == opening) - (token.text == closing)
        return self.tokens[begin:self.index - 1]

    def statement(self):
        token = self._peek()
        statements = self._statement()
        for node in statements:
            node._glsl_source_location = (token.line,token.column,token.text)
        return statements

    def _scoped(self, read):
        if self.context is None:
            return read()
        self.context.lexical.append({})
        try:
            return read()
        finally:
            self.context.lexical.pop()

    def _statement(self):
        token = self._peek()
        if token.text == "{":
            self._take()
            return self._scoped(lambda: self.read_all("}"))
        if token.text == "for":
            # The for header's declarations belong to the loop's own scope.
            return self._scoped(self._for_statement)
        if token.text in {"if", "while"}:
            self._take()
            test, prelude = self.expression(self.delimited("(", ")"))
            if prelude and token.text == "while":
                raise _ExpressionError(
                    token, "a while condition that needs prior statements "
                    "requires loop-condition lowering")
            body = self.statement() or [ast.Pass()]
            otherwise = []
            if token.text == "if" and self._peek("else"):
                self._take()
                otherwise = self.statement()
            node = (ast.If if token.text == "if" else ast.While)(
                test=test, body=body, orelse=otherwise)
            return prelude + [node]
        begin = self.index
        while self._peek() and not self._peek(";"):
            if self._peek("}"):
                raise _ExpressionError(self._peek(), "Missing statement semicolon")
            self._take()
        self._take(";")
        return self.simple(self.tokens[begin:self.index - 1])

    def _for_statement(self):
        token = self._peek()
        if True:
            # Preserve C for semantics, including increment on continue. This
            # bounded first slice accepts a simple counter increment only.
            self._take()
            header = self.delimited("(", ")")
            parts = [[]]
            for item in header:
                if item.text == ";": parts.append([])
                else: parts[-1].append(item)
            if len(parts) != 3 or not all(parts):
                raise _ExpressionError(token, "Expected complete for header")
            initial = self.simple(parts[0])
            test, prelude = self.expression(parts[1])
            if prelude:
                raise _ExpressionError(
                    token, "a for condition that needs prior statements "
                    "requires loop-condition lowering")
            update = self.simple(parts[2])
            if len(update) != 1 or not isinstance(update[0], ast.AugAssign) or not isinstance(update[0].target, ast.Name):
                raise _ExpressionError(token, "For update requires simple counter increment")
            body = self.statement()
            class ContinueUpdate(ast.NodeTransformer):
                def visit_Continue(self, node): return copy.deepcopy(update) + [node]
                def visit_For(self, node): return node
                def visit_While(self, node): return node
            body = [ContinueUpdate().visit(item) for item in body]
            body = [node for item in body for node in (item if isinstance(item, list) else [item])]
            return initial + [ast.While(test=test, body=body + update, orelse=[])]

    def _declared_type(self, tokens):
        """Split a leading declared type from ``tokens``; (type, rest)."""

        while tokens and tokens[0].text in _QUALIFIERS:
            tokens = tokens[1:]
        if not tokens:
            return None, tokens
        known = (tokens[0].text in _SCALAR_TYPES if self.context is None
                 else self.context.types.known(tokens[0].text))
        if known and len(tokens) > 1 and re.fullmatch(r"[A-Za-z_]\w*", tokens[1].text):
            return tokens[0].text, tokens[1:]
        return None, tokens

    def _declare(self, declared, tokens, anchor, *, constant=False):
        """Lower ``T a [= e], b [= e];`` into typed annotated assignments."""

        declarators = [[]]
        depth = 0
        for item in tokens:
            depth += (item.text in "([") - (item.text in ")]")
            if item.text == "," and depth == 0:
                declarators.append([])
            else:
                declarators[-1].append(item)
        statements = []
        for declarator in declarators:
            if not declarator or not re.fullmatch(r"[A-Za-z_]\w*", declarator[0].text):
                raise _ExpressionError(anchor, "Expected local declaration name")
            name = declarator[0]
            local = name.text
            if name.text in self.declarations:
                if self.context is None:
                    raise _ExpressionError(name, "Shadowed/redeclared local requires lexical-scope lowering")
                # GLSL block scoping: a redeclaration in another block is a
                # different variable.  Python has one function scope, so the
                # inner variable gets its own name for the rest of its block.
                local = self.context.temporaries.fresh().replace(
                    "__glsl_value_", f"{name.text}__glsl_scope_")
            extent = None
            if len(declarator) > 1 and declarator[1].text == "[":
                # The extent is a literal or a recorded ``const int``.
                constants = {} if self.context is None else self.context.constants
                size = (int(declarator[2].text) if len(declarator) > 2
                        and declarator[2].text.isdigit()
                        else constants.get(declarator[2].text) if len(declarator) > 2
                        else None)
                if (len(declarator) < 4 or size is None
                        or size <= 0 or declarator[3].text != "]"):
                    raise _ExpressionError(declarator[0], "Local array requires positive constant extent")
                if len(declarator) > 4 or declared not in _SCALAR_TYPES:
                    raise _ExpressionError(declarator[0],
                        "Local record arrays and array initializers require typed storage lowering")
                extent = int(size)
            elif (constant and self.context is not None and declared in {"int", "uint"}
                  and len(declarator) == 3 and declarator[1].text == "="
                  and re.fullmatch(r"\d+[uU]?", declarator[2].text)):
                # ``const int K = 8;``: an integral constant expression.
                self.context.constants[name.text] = int(declarator[2].text.rstrip("uU"))
            self.declarations.add(name.text)
            self.declarations.add(local)
            if extent is not None:
                # A local scalar array: one typed list of ``extent`` zeros
                # (an uninitialized GLSL array is undefined; zero conforms).
                value, prelude = ast.BinOp(
                    left=ast.List(elts=[_zero_value(self.context, declared, name)],
                                  ctx=ast.Load()),
                    op=ast.Mult(), right=ast.Constant(value=extent)), []
                # ``[zero] * extent`` is the existing fill special case; its
                # literals stay raw so ingestion recognises it, and the
                # declared list annotation carries the element width.
                value.right._glsl_raw_literal = True
                value.left.elts[0]._glsl_raw_literal = True
            elif len(declarator) == 1:
                value, prelude = _zero_value(self.context, declared, name), []
            elif declarator[1].text == "=" and len(declarator) > 2:
                value, prelude = self.expression(declarator[2:])
                # A declaration initialized from existing storage is a
                # copy of it, as an assignment is (_value_copy).
                value = _copied_read(self.context, value, name)
            else:
                raise _ExpressionError(name, "Expected '=' or ',' after declaration name")
            if self.context is not None:
                self.context.lexical[-1][name.text] = local
                self.context.scope[local] = (
                    declared if extent is None else f"{declared}[{extent}]")
            annotation = (ast.Constant(_dtype(declared)) if self.context is None
                          else self.context.types.annotation(declared))
            if extent is not None:
                annotation = ast.Subscript(value=_name("list"), slice=annotation,
                                           ctx=ast.Load())
            statements.extend(prelude)
            statements.append(ast.AnnAssign(
                target=ast.Name(id=local, ctx=ast.Store()),
                annotation=annotation, value=value, simple=1))
        return statements

    def simple(self, tokens):
        if not tokens: return []
        token = tokens[0]
        if token.text == "return":
            if len(tokens) == 1:
                return [ast.Return(value=None)]
            value, prelude = self.expression(tokens[1:])
            return prelude + [ast.Return(value=value)]
        if token.text in {"break", "continue"} and len(tokens) == 1:
            return [(ast.Break if token.text == "break" else ast.Continue)()]
        constant = any(t.text == "const" for t in tokens[:3])
        declared, rest = self._declared_type(tokens)
        if declared is not None:
            return self._declare(declared, rest, token, constant=constant)
        while tokens and tokens[0].text in _QUALIFIERS: tokens = tokens[1:]
        if len(tokens) == 2 and (tokens[0].text in {"++", "--"} or tokens[-1].text in {"++", "--"}):
            op = tokens[0] if tokens[0].text in {"++", "--"} else tokens[-1]
            name = tokens[-1] if op is tokens[0] else tokens[0]
            local = name.text if self.context is None else self.context.resolve(name.text)
            return [ast.AugAssign(target=ast.Name(id=local, ctx=ast.Store()),
                op=ast.Add() if op.text=="++" else ast.Sub(), value=ast.Constant(1))]
        depth = 0
        assignment = None
        for i, t in enumerate(tokens):
            depth += (t.text in "([") - (t.text in ")]")
            if depth == 0 and t.text in _ASSIGNMENTS:
                assignment = i
                break
        if assignment is None:
            # An expression statement: a void call, or a call whose result is
            # discarded.  Anything else has no effect GLSL can observe.
            value, prelude = self.expression(tokens)
            if isinstance(value, ast.Constant) and _type_of(value) == "void":
                return prelude
            if not isinstance(value, ast.Call):
                raise _ExpressionError(token, "Expected assignment or supported statement")
            return prelude + [ast.Expr(value=value)]
        target, target_prelude = self.expression(tokens[:assignment])
        value, prelude = self.expression(tokens[assignment+1:])
        op = tokens[assignment].text
        lanes_row = getattr(target, "_glsl_buffer_lanes", None)
        if lanes_row is not None:
            return target_prelude + prelude + self._buffer_store(
                target, lanes_row, op, value, tokens[assignment])
        if getattr(target, "_glsl_buffer_struct", None) is not None:
            raise _ExpressionError(
                token, "a whole struct buffer element is stored field by field")
        component_select = getattr(target, "_glsl_component_select", None)
        if component_select is not None:
            return target_prelude + prelude + self._component_select_store(
                target, component_select, op, value, tokens[assignment])
        swizzle = getattr(target, "_glsl_swizzle", None)
        if swizzle is not None:
            return target_prelude + prelude + self._swizzle_store(
                target, swizzle, op, value, tokens[assignment])
        if not isinstance(target,(ast.Name,ast.Subscript,ast.Attribute)):
            raise _ExpressionError(token,"Invalid assignment target")
        if op != "=" and _vector_shape(_type_of(target)) and self.context is not None:
            parser = _ASTExpressionParser((), source_calls=self.source_calls,
                                          context=self.context)
            value = parser._vector_binary(
                _Token(op[:-1], tokens[assignment].line, tokens[assignment].column),
                self._reuse(target), value)
            prelude = prelude + parser.prelude
            op = "="
        target.ctx=ast.Store()
        if op=="=":
            # GLSL assignment copies a vector, struct or array (_value_copy).
            value = _copied_read(self.context, value, token)
            return target_prelude + prelude + [ast.Assign(targets=[target],value=value)]
        return target_prelude + prelude + [ast.AugAssign(target=target,op=_BINARY_AST[op[:-1]](),value=value)]

    def _component_select_store(self, target, component_select, op, value, token):
        """``q[k] = v`` with a runtime ``k``: store the selected component.

        The read ``q[k]`` is a select over components; the store is the same
        select as statements: ``if k == 0: q.x = v`` ... ``else: q.<last> =
        v``.  ``v`` (and, for ``op=``, the combined value) is evaluated once,
        before the branches.
        """

        base, position, width = component_select
        component = _type_of(target)
        if op != "=":
            value = _typed(_binary(target, op[:-1], value), component)
        name = self.context.temporaries.fresh()
        self.context.scope[name] = component
        statements = [ast.AnnAssign(target=ast.Name(id=name, ctx=ast.Store()),
                                    annotation=self.context.types.annotation(component),
                                    value=value, simple=1)]

        def store(axis):
            destination = ast.Attribute(value=copy.deepcopy(base),
                                        attr=_COMPONENTS[axis], ctx=ast.Store())
            return [ast.Assign(targets=[destination], value=_name(name))]

        branch = store(width - 1)
        for axis in reversed(range(width - 1)):
            test = ast.Compare(left=copy.deepcopy(position), ops=[ast.Eq()],
                               comparators=[_typed(ast.Constant(value=axis), "int")])
            branch = [ast.If(test=test, body=store(axis), orelse=branch)]
        return statements + branch

    def _buffer_store(self, target, lanes_row, op, value, token):
        """Store a value into the lanes of one buffer row it names.

        ``segs[k] = vec4(...)`` and ``pending_rays[i].packet = p`` write each
        lane of the caller's span in place; ``op=`` combines with the lanes'
        current value first.
        """

        buffer, row, lanes, glsl_type, _component = lanes_row
        parser = _ASTExpressionParser((), source_calls=self.source_calls,
                                      context=self.context)
        if op != "=":
            operator = _Token(op[:-1], token.line, token.column)
            value = (parser._vector_binary(operator, target, value)
                     if len(lanes) > 1
                     else _typed(_binary(target, op[:-1], value), glsl_type))
        parts = parser._components(value, len(lanes), token) if len(lanes) > 1 else [value]
        statements = list(parser.prelude)
        for lane, part in zip(lanes, parts):
            column = ast.Constant(value=int(lane))
            column._glsl_raw_literal = True
            destination = ast.Subscript(
                value=_name(buffer),
                slice=ast.Tuple(elts=[copy.deepcopy(row), column], ctx=ast.Load()),
                ctx=ast.Store())
            statements.append(ast.Assign(targets=[destination], value=part))
        return statements

    def _swizzle_store(self, target, swizzle, op, value, token):
        base, indices = swizzle
        if len(set(indices)) != len(indices):
            raise _ExpressionError(token, "a swizzle assignment may not repeat a component")
        parser = _ASTExpressionParser((), source_calls=self.source_calls,
                                      context=self.context)
        if op != "=":
            value = parser._vector_binary(
                _Token(op[:-1], token.line, token.column), target, value)
        parts = parser._components(value, len(indices), token)
        statements = list(parser.prelude)
        for index, part in zip(indices, parts):
            destination = ast.Attribute(value=copy.deepcopy(base),
                                        attr=_COMPONENTS[index], ctx=ast.Store())
            statements.append(ast.Assign(targets=[destination], value=part))
        return statements


def _retain_literal_widths(function):
    """Retain typed source literals through the normal annotated AST lane.

    Ordinary Python literals alone would reclassify GLSL binary32 as binary64.
    Their immutable typed declarations carry the original spelling/location;
    no arithmetic or backend instruction is supplied here.
    """
    used = {node.id for node in ast.walk(function) if isinstance(node,ast.Name)}
    used.update(arg.arg for arg in function.args.args)
    declarations=[]
    class LiteralDeclarations(ast.NodeTransformer):
        def visit_Constant(self,node):
            if getattr(node,"_glsl_raw_literal",False):return node
            dtype=getattr(node,"_glsl_scalar_dtype",
                "float32" if type(node.value) is float else "int32" if type(node.value) is int else None)
            if dtype is None:return node
            name="__glsl_literal_"+str(len(declarations))
            while name in used:name+="_"
            used.add(name)
            declaration=ast.AnnAssign(target=ast.Name(id=name,ctx=ast.Store()),
                annotation=ast.Constant(dtype),value=node,simple=1)
            declaration._glsl_source_location=getattr(node,"_glsl_source_location",None)
            declarations.append(declaration)
            reference=ast.Name(id=name,ctx=ast.Load())
            reference._glsl_source_location=getattr(node,"_glsl_source_location",None)
            return reference
        def visit_AnnAssign(self,node):
            # A declared type annotation is not a value; only the assigned
            # value carries literals.
            if node.value is not None:
                node.value=self.visit(node.value)
            return node
    function.body=[LiteralDeclarations().visit(statement) for statement in function.body]
    function.body=declarations+function.body
    return function


def _source_tokens(body, line, column):
    tokens=_tokenize(body,start_line=line)
    return [_Token(t.text,t.line,t.column+column-1 if t.line==line else t.column) for t in tokens]


def retain_glsl_source_evidence(module, source):
    """Use the existing generated-source declaration mechanism, not SSA roots.

    Each emitted AST span derives from its nearest exact GLSL token; structural
    module/function nodes additionally retain the complete submitted document.
    The public compiler must preserve this AST rather than unparse/reparse it.
    """
    import hashlib
    from .concordance_declarations import SOURCE_SPAN, SpanFact
    from ..transmogrifier.graph.graph_express2 import SourceDefinition
    digest=hashlib.sha256(source.encode()).hexdigest()
    lines=source.splitlines() or [""]
    document=SourceDefinition(SOURCE_SPAN,("glsl:"+digest,"document",()),
        SpanFact("GLSLDocument",1,0,len(lines),len(lines[-1]),digest))
    def visit(node,inherited):
        location=getattr(node,"_glsl_source_location",None)
        cells=inherited
        if location is not None:
            line,column,spelling=location
            if not (1<=line<=len(lines)) or lines[line-1][column-1:column-1+len(spelling)]!=spelling:
                raise ValueError("GLSL source token correspondence is invalid")
            token=SourceDefinition(SOURCE_SPAN,("glsl:"+digest,"tokens",(line,column)),
                SpanFact("GLSLToken",line,column-1,line,column-1+len(spelling),
                    hashlib.sha256(spelling.encode()).hexdigest()))
            cells=(document,token)
        node._turing_source_cells=cells
        if isinstance(node,(ast.expr,ast.stmt)):
            # Existing owner stamping consumes this declaration, including
            # after rewrites. Never stamp shared operator/context singletons.
            node._turing_source_definition=cells[-1]
        for child in ast.iter_child_nodes(node):visit(child,cells)
    visit(module,(document,))
    module._glsl_definition_sources={node.name:document for node in module.body
        if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    return module


_STRUCT = re.compile(r"\bstruct\s+([A-Za-z_]\w*)\s*\{([^{}]*)\}\s*;")


def _source_structs(cleaned, shortfalls):
    """Read ``struct`` declarations as ordered ``(type, field)`` members."""

    structs: dict[str, list[tuple[str, str]]] = {}
    for match in _STRUCT.finditer(cleaned):
        name, body = match.groups()
        line = cleaned.count("\n", 0, match.start()) + 1
        members = []
        valid = True
        for declaration in body.split(";"):
            words = declaration.replace(",", " , ").split()
            if not words:
                continue
            while words and words[0] in _QUALIFIERS:
                words = words[1:]
            member_type, names = words[0], [w for w in words[1:] if w != ","]
            for member in names:
                if "[" in member or not re.fullmatch(r"[A-Za-z_]\w*", member):
                    shortfalls.append(GLSLSourceShortfall(
                        "GLSL_STRUCT", line, 1,
                        f"struct {name} member {member!r} needs fixed-array record fields"))
                    valid = False
                    continue
                members.append((member_type, member))
        if valid:
            structs[name] = members
    for name, members in list(structs.items()):
        for member_type, member in members:
            if (member_type not in _SCALAR_TYPES and _vector_shape(member_type) is None
                    and member_type not in structs):
                shortfalls.append(GLSLSourceShortfall(
                    "GLSL_STRUCT", 1, 1,
                    f"struct {name} member {member} has unknown type {member_type!r}"))
                structs.pop(name, None)
                break
    return structs


def _expand_object_macros(cleaned, shortfalls):
    """Apply ``#define NAME body`` the way the GLSL preprocessor does.

    An object-like macro is textual replacement of the identifier token
    ``NAME`` by its body, repeated until no defined name remains (a macro
    may name another).  Each directive line becomes a blank line so every
    reported line number stays the source's.  A function-like macro and
    the conditional directives are refused by name, not approximated.
    """

    macros: dict[str, str] = {}
    lines = cleaned.split("\n")
    for index, text in enumerate(lines):
        directive = re.match(r"\s*#\s*(\w+)(.*)", text)
        if directive is None:
            continue
        keyword, rest = directive.groups()
        if keyword == "define":
            definition = re.match(r"\s+([A-Za-z_]\w*)(\(?)(.*)", rest)
            if definition is None or definition.group(2):
                shortfalls.append(GLSLSourceShortfall(
                    "GLSL_PREPROCESSOR", index + 1, 1,
                    "only object-like #define macros are expanded: " + text.strip()))
            else:
                macros[definition.group(1)] = definition.group(3).strip()
            lines[index] = ""
        elif keyword not in {"version", "extension"}:
            shortfalls.append(GLSLSourceShortfall(
                "GLSL_PREPROCESSOR", index + 1, 1,
                f"#{keyword} is not expanded"))
    text = "\n".join(lines)
    if not macros:
        return text
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, macros)) + r")\b")
    for _ in range(32):
        expanded = pattern.sub(lambda match: macros[match.group(1)], text)
        if expanded == text:
            return text
        text = expanded
    shortfalls.append(GLSLSourceShortfall(
        "GLSL_PREPROCESSOR", 1, 1, "#define expansion did not settle (a macro names itself)"))
    return text


def _std430_element(glsl_type, types):
    """The std430 layout of one array element, in 4-byte lanes, or None.

    Returns ``(stride, fields, component)``: the element's row width in
    lanes, ``{field: (lane offset, GLSL type)}`` for a struct (None for a
    vector), and the one scalar type every lane holds.  std430 aligns a
    scalar to 1 lane, vec2 to 2, vec3 and vec4 to 4 (vec3 occupies 3); a
    struct aligns to its widest member and its stride rounds up to that.
    Only elements whose lanes share one scalar type are read; anything else
    (a nested struct, a bool, mixed lane types) is refused by the caller.
    """

    def vector(kind):
        if kind in _BUFFER_ELEMENT_DTYPES:
            return kind, 1, 1, 1
        shape = _vector_shape(kind)
        if shape is None or shape[0] not in _BUFFER_ELEMENT_DTYPES:
            return None
        component, width = shape
        align = {2: 2, 3: 4, 4: 4}[width]
        return component, width, align, width

    plain = vector(glsl_type)
    if plain is not None:
        component, _width, align, size = plain
        return -(-size // align) * align, None, component
    members = types.structs.get(glsl_type)
    if not members:
        return None
    offset, struct_align, fields, components = 0, 1, {}, set()
    for kind, name in members:
        layout = vector(kind)
        if layout is None:
            return None
        component, _width, align, size = layout
        offset = -(-offset // align) * align
        fields[name] = (offset, kind)
        components.add(component)
        offset += size
        struct_align = max(struct_align, align)
    if len(components) != 1:
        return None
    return -(-offset // struct_align) * struct_align, fields, components.pop()


_GLOBAL_SCALAR_CONSTANT = re.compile(
    r"\bconst\s+(?:(?:highp|mediump|lowp)\s+)?(?:int|uint|float|bool)\s+"
    r"([A-Za-z_]\w*)\s*=\s*([-+]?(?:0[xX][0-9a-fA-F]+[uU]?|\d+\.?\d*(?:[eE][-+]?\d+)?[fFuU]?|"
    r"\.\d+(?:[eE][-+]?\d+)?[fF]?|true|false))\s*;"
)


_GLOBAL_CONSTANT_ARRAY = re.compile(
    r"\bconst\s+(?P<type>[A-Za-z_]\w*)\s+(?P<name>[A-Za-z_]\w*)\s*\[\s*(?P<extent>\d+)\s*\]"
    r"\s*=\s*(?P=type)\s*\[\s*\d*\s*\]\s*\((?P<body>[^;]*)\)\s*;"
)
_FLOAT_LITERAL = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?[fF]?")


def _source_constant_arrays(cleaned, shortfalls):
    """Read document ``const`` arrays of float scalars or float vectors.

    ``const vec3 B[12] = vec3[](vec3(...), ...);`` is constant data: its
    lanes, ``extent * width`` float literals in order.  It is blanked from
    the text (line numbers kept); each function that reads it declares it
    as a typed lane list (``_ASTStatementReader``) and ``B[i]`` builds the
    vector from lanes ``i * width + k``.
    """

    arrays: dict[str, tuple[str, int, int, list[float]]] = {}
    pieces, position = [], 0
    for match in _GLOBAL_CONSTANT_ARRAY.finditer(cleaned):
        kind = match.group("type")
        shape = _vector_shape(kind)
        width = 1 if kind == "float" else (shape[1] if shape and shape[0] == "float" else None)
        extent = int(match.group("extent"))
        body = re.sub(r"\b" + re.escape(kind) + r"\s*\(", "(", match.group("body"))
        values = [float(text.rstrip("fF")) for text in _FLOAT_LITERAL.findall(body)]
        line = cleaned.count("\n", 0, match.start()) + 1
        if width is None or len(values) != extent * width:
            shortfalls.append(GLSLSourceShortfall(
                "GLSL_CONSTANT", line, 1,
                f"const array {match.group('name')} needs {extent} literal "
                f"float or float-vector elements"))
            continue
        arrays[match.group("name")] = (kind, extent, width, values)
        pieces.append(cleaned[position:match.start()])
        pieces.append(re.sub(r"[^\n]", " ", match.group(0)))
        position = match.end()
    pieces.append(cleaned[position:])
    return "".join(pieces), arrays


def _inline_global_scalar_constants(cleaned):
    """Replace each global ``const`` scalar with its literal initializer.

    ``const int N = 12;`` at document scope is a GLSL constant expression:
    every use means the literal, which keeps its own spelling (``12`` an
    int, ``5u`` a uint, ``1.5`` a float).  The declaration is blanked in
    place so line numbers stay the source's.  A constant whose initializer
    is not a single literal is left as a declaration.
    """

    constants: dict[str, str] = {}
    depth = 0
    pieces = []
    position = 0
    for match in _GLOBAL_SCALAR_CONSTANT.finditer(cleaned):
        depth += cleaned.count("{", position, match.start()) - cleaned.count(
            "}", position, match.start())
        pieces.append(cleaned[position:match.start()])
        if depth == 0:
            constants[match.group(1)] = match.group(2)
            pieces.append(" " * (match.end() - match.start()))
        else:
            pieces.append(match.group(0))
        position = match.end()
    pieces.append(cleaned[position:])
    text = "".join(pieces)
    if not constants:
        return text
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, constants)) + r")\b")
    # A negative literal is parenthesized so ``x-N`` stays a subtraction.
    return pattern.sub(
        lambda match: (f"({constants[match.group(1)]})"
                       if constants[match.group(1)][:1] in "+-"
                       else constants[match.group(1)]), text)


_TEXTURE = re.compile(
    r"(?:layout\s*\((?P<layout>[^)]*)\)\s*)?\buniform\s+"
    r"(?:(?:highp|mediump|lowp|readonly|writeonly|coherent|restrict|volatile)\s+)*"
    r"(?P<prefix>[iu]?)(?P<kind>image|sampler)(?P<dims>[123])D\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*;"
)
_TEXTURE_COMPONENTS = {"": "float", "i": "int", "u": "uint"}
_IMAGE_FORMATS = {"r32f": "float", "r32i": "int", "r32ui": "uint"}


def _source_textures(cleaned, shortfalls, texture_fields=None):
    """Read image and sampler uniforms as caller-owned texel memory.

    A one-channel 32-bit texture is a rank-``dims`` tensor of its component
    type in ``[z, y, x]`` order (x fastest, as GLSL stores texels).  An
    image's channel count is its declared ``layout`` format, which must be
    ``r32f``/``r32i``/``r32ui``.  A sampler states none in GLSL: its texel
    format is the caller's declared field in ``texture_fields`` (the
    ``program_abi`` value of that parameter), whose rank is ``dims`` for one
    channel.  An undeclared sampler, or one declared with more channels, is
    a shortfall; no channel count is assumed.
    Returns ``name -> (kind, component, dims)``.
    """

    fields = dict(texture_fields or {})
    textures: dict[str, tuple[str, str, int]] = {}
    for match in _TEXTURE.finditer(cleaned):
        line = cleaned.count("\n", 0, match.start()) + 1
        name = match.group("name")
        component = _TEXTURE_COMPONENTS[match.group("prefix")]
        dims = int(match.group("dims"))
        if match.group("kind") == "image":
            formats = {item.strip() for item in (match.group("layout") or "").split(",")}
            declared = {_IMAGE_FORMATS[f] for f in formats if f in _IMAGE_FORMATS}
            if declared != {component}:
                shortfalls.append(GLSLSourceShortfall(
                    "GLSL_TEXTURE", line, 1,
                    f"image {name} needs a one-channel 32-bit "
                    f"layout format matching its {component} texels"))
                continue
        else:
            field = fields.get(name)
            if field is None:
                shortfalls.append(GLSLSourceShortfall(
                    "GLSL_TEXTURE", line, 1,
                    f"sampler {name} has no texel format in GLSL; the caller's "
                    f"program_abi must declare its dtype and rank"))
                continue
            dtype = str(getattr(field, "dtype", None) if not isinstance(field, Mapping)
                        else field.get("dtype"))
            rank = getattr(field, "rank", None) if not isinstance(field, Mapping) \
                else field.get("rank")
            if dtype != _BUFFER_ELEMENT_DTYPES[component] or rank is None or int(rank) != dims:
                shortfalls.append(GLSLSourceShortfall(
                    "GLSL_TEXTURE", line, 1,
                    f"sampler {name} is declared {dtype} rank {rank}; a "
                    f"{component} {dims}D one-channel texture is "
                    f"{_BUFFER_ELEMENT_DTYPES[component]} rank {dims} "
                    f"(multi-channel texel reads are not lowered)"))
                continue
        textures[name] = (match.group("kind"), component, dims)
    return textures


def _source_mutable_globals(cleaned, type_names, shortfalls):
    """The file-scope variables with no storage qualifier.

    ``vec3 studio_response_direction;`` is per-invocation state: it starts
    undefined when the invocation does, every function of the invocation
    sees the one copy, and it ends with the invocation.  ``main`` owns it
    (its declaration statement, read at its own source position, becomes
    ``main``'s first statement: initializer, or zero) and each helper that
    touches it takes it after its authored parameters
    (``_HelperSignature.globals``).  Reading each use as a Python local
    silently dropped the write another function reads.  A ``shared``
    variable is workgroup memory, not carried: a named shortfall.  ``const``
    globals are constant expressions (inlined); uniforms, inputs, outputs,
    buffers, images and samplers are caller memory read elsewhere.

    Returns ``[(statement text, line, column, ((name, GLSL type), ...))]``,
    a fixed array's type spelled ``T[N]``.
    """

    declarations = []

    depth, segment, segment_start, segments = 0, [], 0, []
    for index, character in enumerate(cleaned):
        if character == "{":
            if depth == 0:
                segment.append(";")
            depth += 1
        elif character == "}":
            depth -= 1
        elif depth == 0:
            if not segment:
                segment_start = index
            segment.append(character)
            if character == ";":
                segments.append(("".join(segment), segment_start))
                segment = []
    for text, start in segments:
        statement = text.strip().rstrip(";").strip()
        if not statement or statement.startswith("#"):
            continue
        head = re.match(r"(?P<qualifiers>(?:(?:shared|coherent|volatile|restrict|"
                        r"precise|highp|mediump|lowp)\s+)*)(?P<type>[A-Za-z_]\w*)\s+",
                        statement)
        if head is None or head.group("type") not in type_names:
            continue
        line = cleaned.count("\n", 0, start) + 1
        # The declarators, split at top-level commas (an initializer may
        # hold a call, ``vec3 a = vec3(0.0), b;``).
        declarators, depth, current = [], 0, []
        for character in statement[head.end():]:
            depth += (character in "([") - (character in ")]")
            if character == "," and depth == 0:
                declarators.append("".join(current))
                current = []
            else:
                current.append(character)
        declarators.append("".join(current))
        names = []
        for declarator in declarators:
            match = re.match(r"\s*(?P<name>[A-Za-z_]\w*)\s*(?:\[\s*(?P<extent>\w*)\s*\])?"
                             r"\s*(?:=.*)?$", declarator, re.S)
            if match is None:
                names = None
                break
            extent = match.group("extent")
            if extent is not None and not extent.isdigit():
                names = None
                break
            names.append((match.group("name"), head.group("type")
                          if extent is None else f"{head.group('type')}[{extent}]"))
        if names is None:
            # A function prototype or a declaration form not read here.
            if "(" not in statement.split("=", 1)[0]:
                shortfalls.append(GLSLSourceShortfall(
                    "GLSL_GLOBAL", line, 1,
                    f"file-scope declaration is not read: {statement[:80]}"))
            continue
        if "shared" in head.group("qualifiers"):
            for name, _kind in names:
                shortfalls.append(GLSLSourceShortfall(
                    "GLSL_GLOBAL", line, 1,
                    f"workgroup-shared variable {name} is not carried"))
            continue
        text_start = cleaned.index(statement, start)
        line = cleaned.count("\n", 0, text_start) + 1
        column = text_start - cleaned.rfind("\n", 0, text_start)
        declarations.append((statement + ";", line, column, tuple(names)))
    return declarations


def _source_buffers(cleaned, shortfalls, types):
    """Read storage-buffer blocks as ``member -> (element, extent, scalar, layout)``.

    A member is caller-owned memory: ``uint words[];`` (runtime extent),
    ``uint words[N];``, or a scalar member ``uint count;`` (one element, a
    span of extent 1).  An array of vectors or of structs of vectors is its
    std430 region itself: a rank-2 ``[rows, stride]`` tensor of the lane
    type, whose ``layout`` (``_std430_element``) says where each field's
    lanes are.  Blocks addressed through an instance name are refused by
    name rather than guessed into a layout.
    """

    buffers: dict[str, tuple] = {}
    for match in _BUFFER_BLOCK.finditer(cleaned):
        line = cleaned.count("\n", 0, match.start()) + 1
        if match.group("instance"):
            shortfalls.append(GLSLSourceShortfall(
                "GLSL_STORAGE", line, 1,
                f"buffer {match.group('block')} is addressed through instance "
                f"{match.group('instance')!r}; only unnamed blocks are read"))
            continue
        for declaration in match.group("body").split(";"):
            words = declaration.split()
            while words and words[0] in _QUALIFIERS:
                words = words[1:]
            if not words:
                continue
            # ``uint words[];`` (runtime extent), ``uint words[N];`` or a
            # scalar member ``uint count;``, which is one element of the
            # caller's memory: a span of extent 1 read and written at 0.
            member = re.fullmatch(
                r"([A-Za-z_]\w*)\s*(?:\[\s*(\d*)\s*\])?", " ".join(words[1:]))
            bracketed = "[" in declaration
            layout = (
                None if len(words) < 2 or words[0] in _BUFFER_ELEMENT_DTYPES
                or not bracketed else _std430_element(words[0], types)
            )
            if (len(words) < 2 or member is None
                    or (words[0] not in _BUFFER_ELEMENT_DTYPES and layout is None)):
                shortfalls.append(GLSLSourceShortfall(
                    "GLSL_STORAGE", line, 1,
                    f"buffer {match.group('block')} member {declaration.strip()!r} "
                    "needs a float, int or uint element, or an array of vectors "
                    "or of structs of vectors with one lane type"))
                continue
            extent = (1 if not bracketed
                      else int(member.group(2)) if member.group(2) else None)
            buffers[member.group(1)] = (words[0], extent, not bracketed, layout)
    return buffers


def lower_glsl_source_to_ast(
    source: str,
    *,
    function_name: str | None = None,
    texture_fields: Mapping[str, Any] | None = None,
) -> GLSLASTLoweringResult:
    """Parse the supported GLSL slice into an ordinary Python ``ast.Module``.

    ``texture_fields`` is the caller's declared ABI field (``dtype``,
    ``rank``, ``shape``) of each sampler uniform, by name: GLSL states no
    texel format for a sampler, so the caller that allocated the texture
    declares it (``_source_textures``).
    """

    shortfalls: list[GLSLSourceShortfall] = []
    cleaned = _inline_global_scalar_constants(_without_comments(_expand_object_macros(
        _without_comments(str(source), keep_directives=True), shortfalls)))
    cleaned, constant_arrays = _source_constant_arrays(cleaned, shortfalls)
    try:
        body, body_line, entry_name = _main_body(cleaned)
    except ValueError as error:
        shortfalls.append(GLSLSourceShortfall("GLSL_ENTRY", 1, 1, str(error)))
        return GLSLASTLoweringResult(ast.Module(body=[], type_ignores=[]), tuple(shortfalls))

    types = _GLSLTypes(_source_structs(cleaned, shortfalls))
    type_names = sorted(set(_SCALAR_TYPES) | set(types.structs)
                        | {f"{p}{n}" for p in _VECTOR_PREFIX.values() for n in (2, 3, 4)},
                        key=len, reverse=True)
    type_pattern = "|".join(map(re.escape, ["void", *type_names]))
    mutable_globals = _source_mutable_globals(cleaned, set(type_names), shortfalls)
    helpers: dict[str, _HelperSignature] = {}
    source_functions = {}
    for match in re.finditer(r"\b(" + type_pattern + r")\s+([A-Za-z_]\w*)\s*\(([^()]*)\)\s*\{", cleaned):
        result_type, helper_name, parameters = match.groups()
        line = cleaned.count("\n", 0, match.start()) + 1
        if helper_name == entry_name:
            continue
        if helper_name in helpers:
            shortfalls.append(GLSLSourceShortfall("GLSL_FUNCTION", line, 1,
                "Overloaded/redeclared functions require signature resolution: " + helper_name))
            continue
        signature = []
        valid = True
        for parameter in parameters.split(",") if parameters.strip() and parameters.strip() != "void" else []:
            fields = parameter.strip().split()
            qualifier = "in"
            while fields and (fields[0] in _QUALIFIERS or fields[0] in _PARAMETER_QUALIFIERS):
                if fields[0] in _PARAMETER_QUALIFIERS:
                    qualifier = fields[0]
                fields = fields[1:]
            if len(fields) != 2 or not types.known(fields[0]) or not re.fullmatch(r"[A-Za-z_]\w*", fields[1]):
                shortfalls.append(GLSLSourceShortfall("GLSL_PARAMETER", line, 1,
                    "Helper parameter requires a known scalar, vector or struct type: " + parameter.strip()))
                valid=False;break
            signature.append((qualifier, fields[0], fields[1]))
        if not valid: continue
        end=match.end();depth=1
        while end<len(cleaned) and depth:
            depth+=(cleaned[end]=="{")-(cleaned[end]=="}");end+=1
        if depth:
            shortfalls.append(GLSLSourceShortfall("GLSL_FUNCTION",line,1,"Unclosed helper body"));continue
        helpers[helper_name] = _HelperSignature(signature, result_type)
        source_functions[helper_name]=(cleaned[match.end():end-1],line,match.end()-cleaned.rfind("\n",0,match.end()))

    prefix = cleaned[:cleaned.find(body)]
    arguments: list[str] = []
    output_names: list[str] = []
    argument_types = {}
    textures = _source_textures(cleaned, shortfalls, texture_fields)
    for match in _GLOBAL.finditer(prefix):
        storage, name = match.group("storage"), match.group("name")
        if name in textures:
            # A texture is caller memory, threaded like a buffer below.
            continue
        (arguments if storage in {"uniform", "in"} else output_names).append(name)
        argument_types[name] = match.group("type")

    used_names = set(re.findall(r"[A-Za-z_]\w*", cleaned))
    context = _ReaderContext(types, helpers, _Temporaries(used_names))
    buffers = _source_buffers(cleaned, shortfalls, types)
    context.buffers = {name: member[0] for name, member in buffers.items()}
    context.buffer_scalars = {name for name, member in buffers.items() if member[2]}
    context.buffer_layouts = {
        name: member[3] for name, member in buffers.items() if member[3] is not None
    }
    context.constant_arrays = dict(constant_arrays)

    def constant_array_declarations(text):
        """A typed lane list for each constant array ``text`` reads."""

        declarations = []
        for name in sorted(identifiers(text) & set(constant_arrays)):
            _kind, _extent, _width, values = constant_arrays[name]
            elements = []
            for value in values:
                literal = ast.Constant(value=float(value))
                literal._glsl_raw_literal = True
                elements.append(literal)
            declarations.append(ast.AnnAssign(
                target=ast.Name(id=name, ctx=ast.Store()),
                annotation=ast.Subscript(value=_name("list"),
                                         slice=ast.Constant("float32"), ctx=ast.Load()),
                value=ast.List(elts=elements, ctx=ast.Load()), simple=1))
        return declarations

    # Which buffers each function touches: its own reads and writes, plus
    # those of every helper it calls (a fixed point over the call graph).
    def identifiers(text):
        return set(re.findall(r"[A-Za-z_]\w*", text))

    # Buffers and textures are both caller memory handed in by reference.
    context.textures = dict(textures)
    resources = [*buffers, *textures]
    buffer_uses = {
        helper: identifiers(text) & set(resources)
        for helper, (text, _line, _column) in source_functions.items()
    }
    helper_calls = {
        helper: identifiers(text) & set(source_functions)
        for helper, (text, _line, _column) in source_functions.items()
    }
    changed = True
    while changed:
        changed = False
        for helper, callees in helper_calls.items():
            grown = buffer_uses[helper].union(*(buffer_uses[c] for c in callees))
            if grown != buffer_uses[helper]:
                buffer_uses[helper] = grown
                changed = True
    for helper in source_functions:
        helpers[helper].buffers = tuple(b for b in resources if b in buffer_uses[helper])

    # Mutable globals: which each helper reads and which it writes, directly
    # or through its callees (the same fixed point as the buffers).  A write
    # is an assignment, a compound assignment or ``++``/``--`` of the global
    # or of an element or field of it.  A parameter of the same name shadows.
    mutable_global_types = {
        name: kind
        for _text, _line, _column, names in mutable_globals for name, kind in names
    }

    def global_writes(text):
        tokens = [token.text for token in _tokenize(text, start_line=1)]
        written = set()
        for index, spelling in enumerate(tokens):
            if spelling not in mutable_global_types:
                continue
            if index and tokens[index - 1] in {"++", "--"}:
                written.add(spelling)
                continue
            after = index + 1
            while after < len(tokens) and tokens[after] in {"[", "."}:
                if tokens[after] == ".":
                    after += 2
                    continue
                depth = 0
                while after < len(tokens):
                    depth += (tokens[after] == "[") - (tokens[after] == "]")
                    after += 1
                    if depth == 0:
                        break
            if after < len(tokens) and tokens[after] in {
                "=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=",
                "++", "--",
            }:
                written.add(spelling)
        return written

    global_uses, global_written = {}, {}
    for helper, (text, _line, _column) in source_functions.items():
        shadowed = {name for _, _, name in helpers[helper].parameters}
        global_uses[helper] = (identifiers(text) & set(mutable_global_types)) - shadowed
        global_written[helper] = global_writes(text) - shadowed
    changed = True
    while changed:
        changed = False
        for helper, callees in helper_calls.items():
            uses = global_uses[helper].union(*(global_uses[c] for c in callees))
            writes = global_written[helper].union(*(global_written[c] for c in callees))
            if uses != global_uses[helper] or writes != global_written[helper]:
                global_uses[helper], global_written[helper] = uses, writes
                changed = True
    for helper in source_functions:
        helpers[helper].globals = tuple(
            (name, kind, name in global_written[helper])
            for name, kind in mutable_global_types.items()
            if name in global_uses[helper])
    main_names = identifiers(body)
    main_buffers = tuple(b for b in resources if b in main_names or any(
        b in buffer_uses[helper] for helper in main_names & set(source_functions)))

    def annotation(glsl_type):
        if types.known(glsl_type):
            return types.annotation(glsl_type)
        return ast.Constant(_dtype(glsl_type))

    global_types = {name: argument_types[name] for name in arguments + output_names}
    # A buffer member is indexed as ``T[]``; ``_index`` reads its element.
    global_types.update({name: f"{member[0]}[]" for name, member in buffers.items()})
    global_types.update({name: f"{array[0]}[{array[1]}]"
                         for name, array in constant_arrays.items()})
    global_types.update({name: f"{texture[1]}{texture[0]}{texture[2]}D"
                         for name, texture in textures.items()})
    context.scope = dict(global_types)
    context.lexical = [{}]
    try:
        # ``main`` owns the invocation's mutable globals: their declarations,
        # read at their own source positions, are its first statements.
        global_declaration_tokens = [
            token
            for text, line, column, _names in mutable_globals
            for token in _source_tokens(text, line, column)
        ]
        reader = _ASTStatementReader([*global_declaration_tokens, *_source_tokens(body, body_line, cleaned.find(body)-cleaned.rfind("\n",0,cleaned.find(body)))], reserved_names=[*arguments, *main_buffers], context=context)
        statements = constant_array_declarations(body) + reader.read_all()
    except _ExpressionError as error:
        shortfalls.append(GLSLSourceShortfall(
            "GLSL_STATEMENT", error.token.line, error.token.column, str(error)))
        statements = []
    assigned_outputs = [name for name in output_names if any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == name
        for statement in statements for node in ast.walk(statement))]
    explicit_return = any(isinstance(node, ast.Return)
        for statement in statements for node in ast.walk(statement))

    if not explicit_return and assigned_outputs:
        value: ast.expr = _name(assigned_outputs[0])
        if len(assigned_outputs) > 1:
            value = ast.Tuple(
                elts=[_name(name) for name in assigned_outputs], ctx=ast.Load()
            )
        statements.append(ast.Return(value=value))
    elif not explicit_return and main_buffers:
        # A compute entry's outputs are its writes into the caller's
        # storage buffers; it returns nothing.
        pass
    elif not explicit_return:
        shortfalls.append(GLSLSourceShortfall(
            "GLSL_OUTPUT", body_line, 1,
            "entry function produced no return value or declared output",
        ))

    function = ast.FunctionDef(
        name=function_name or entry_name,
        args=ast.arguments(
            posonlyargs=[],
            args=[*(ast.arg(arg=name,annotation=annotation(argument_types[name])) for name in arguments),
                  *(ast.arg(arg=name,annotation=_name("AbstractTensor")) for name in main_buffers)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=statements or [ast.Pass()],
        decorator_list=[],
    )
    helper_definitions=[]
    for helper_name,(helper_body,helper_line,helper_column) in source_functions.items():
        signature = helpers[helper_name]
        try:
            # Document globals are visible in every helper; parameters shadow.
            context.scope = dict(global_types)
            context.scope.update({name: kind for _, kind, name in signature.carried_parameters})
            context.lexical = [{}]
            helper_reader=_ASTStatementReader(_source_tokens(helper_body,helper_line,helper_column),
                reserved_names=[*(name for _, _, name in signature.carried_parameters),
                                *signature.buffers],context=context)
            helper_statements=(constant_array_declarations(helper_body)
                               + helper_reader.read_all())
            helper_definitions.append(_helper_definition(
                context, helper_name, signature, helper_statements, _Token(helper_name, helper_line, 1)))
        except _ExpressionError as error:
            shortfalls.append(GLSLSourceShortfall("GLSL_HELPER",error.token.line,
                error.token.column,str(error)))
    # A helper that rebinds a global it does not hand back (an array assigned
    # whole, or a write the token scan did not see, such as an ``out``
    # argument) would drop that write: refuse it by name.
    for definition in helper_definitions:
        signature = helpers[definition.name]
        returned = {name for qualifier, _kind, name in signature.carried_parameters
                    if qualifier == "inout"}
        for name, _kind, _written in signature.globals:
            if name not in returned and any(
                isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                and node.id == name for node in ast.walk(definition)
            ):
                shortfalls.append(GLSLSourceShortfall(
                    "GLSL_GLOBAL", 1, 1,
                    f"helper {definition.name} rebinds global {name}, which is "
                    "not handed back to its caller"))
    records = types.class_definitions()
    prelude = ([ast.ImportFrom(module="dataclasses", names=[ast.alias(name="dataclass")], level=0)]
               if records else [])
    if main_buffers or any(signature.buffers for signature in helpers.values()):
        # Storage buffers are AbstractTensor state pages: caller-owned memory
        # the tensor state library can route to any device.
        prelude.append(ast.ImportFrom(module="src.common.tensors.abstraction",
                                      names=[ast.alias(name="AbstractTensor")], level=0))
    if context.imports:
        prelude.append(ast.ImportFrom(
            module="src.common.tensors.abstraction_methods.bit_reinterpretation",
            names=[ast.alias(name=name) for name in sorted(context.imports)], level=0))
    module = _python_identifiers(ast.fix_missing_locations(ast.Module(
        body=prelude + records + [_retain_literal_widths(fn) for fn in helper_definitions+[function]],
        type_ignores=[]
    )))
    def buffer_abi(name):
        element, extent, _scalar, layout = buffers[name]
        if layout is None:
            return _python_identifier(name), _BUFFER_ELEMENT_DTYPES[element], extent, None
        stride, _fields, component = layout
        return _python_identifier(name), _BUFFER_ELEMENT_DTYPES[component], extent, stride

    return GLSLASTLoweringResult(
        module, tuple(shortfalls),
        tuple(buffer_abi(name) for name in main_buffers if name in buffers),
        tuple((_python_identifier(name), _BUFFER_ELEMENT_DTYPES[textures[name][1]],
               textures[name][2])
              for name in main_buffers if name in textures))


def _python_identifier(name):
    """A GLSL identifier as a Python one.

    GLSL allows Python's keywords as names (the Studio tracer's ``uint
    from``).  Such a name gets the suffix ``__glsl_keyword``; GLSL reserves
    every identifier containing ``__``, so no source name can collide.
    """
    return f"{name}__glsl_keyword" if keyword.iskeyword(name) else name


class _NegativeNumbers(ast.NodeTransformer):
    """A negative number in the form Python parses it: ``-x`` is
    ``UnaryOp(USub, Constant(x))``.  The reader's tree must be the tree of
    its own text (``ast.parse(ast.unparse(tree))``) node for node, so the
    compiler can give it that text's source positions; a ``Constant(-x)``
    was the one node that did not round-trip.  The reader's annotations
    stay on the outer node."""

    def visit_Constant(self, node):  # noqa: N802
        value = node.value
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and (value < 0 or str(value).startswith("-"))):
            negated = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=-value))
            for name, attribute in vars(node).items():
                if name.startswith("_"):
                    setattr(negated, name, attribute)
            return ast.copy_location(negated, node)
        return node


def _python_identifiers(module):
    """Apply ``_python_identifier`` to every name the AST binds or reads:
    variables, parameters, functions, record fields and attributes; and put
    negative numbers in Python's own form (``_NegativeNumbers``)."""
    module = ast.fix_missing_locations(_NegativeNumbers().visit(module))
    for node in ast.walk(module):
        if isinstance(node, ast.Name):
            node.id = _python_identifier(node.id)
        elif isinstance(node, ast.arg):
            node.arg = _python_identifier(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node.name = _python_identifier(node.name)
        elif isinstance(node, ast.Attribute):
            node.attr = _python_identifier(node.attr)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            node.arg = _python_identifier(node.arg)
    return module


def _value_copy(context, value, glsl_type, token):
    """``value`` as a new object of its GLSL type, member by member.

    Program: the Studio tracer's ``studio_response_begin`` does
    ``studio_response_direction = direction`` and ``studio_boundary_step``
    returns its ``inout vec3 rd``; and ``vec3 a = b; a.x = 9.0;``.  Wrong
    record: GLSL assigns, passes and returns vectors, structs and arrays BY
    VALUE, but the reader spelled each as a Python binding of the same
    object.  As Python, ``a.x = 9.0`` then also wrote ``b.x`` (a silent
    wrong value), and every record a helper returned or rebound was an
    escape of its caller's object, so the tracer's ``rd.x`` was refused
    (record_stored_by_call).  Rule: where GLSL copies, the Python names a
    new object: a record is its constructor over its copied members, a
    fixed array the list of its copied elements.  A scalar is a value; a
    resource handle (buffer, image, sampler) is not copied by GLSL either.
    """

    if glsl_type is None or glsl_type in _SCALAR_TYPES:
        return value
    array = _array_type(glsl_type)
    if array is not None:
        element, extent = array
        members = []
        for index in range(extent):
            position = ast.Constant(value=index)
            position._glsl_raw_literal = True
            members.append(_value_copy(context, _typed(ast.Subscript(
                value=copy.deepcopy(value), slice=position, ctx=ast.Load(),
            ), element), element, token))
        return _typed(ast.List(elts=members, ctx=ast.Load()), glsl_type)
    record = context.types.record(glsl_type)
    if record is None:
        return value
    return _typed(_call(record, [
        _value_copy(context, _typed(ast.Attribute(
            value=copy.deepcopy(value), attr=field, ctx=ast.Load(),
        ), member), member, token)
        for field, member in context.types.fields(glsl_type)
    ]), glsl_type)


def _copied_read(context, value, token, glsl_type=None):
    """A GLSL value read from existing storage (a name, field or element),
    copied as GLSL copies it on assignment and return.  A value computed
    here (a constructor, call or operator result) is already its own
    object."""

    if context is None or not isinstance(
        value, (ast.Name, ast.Attribute, ast.Subscript),
    ):
        return value
    return _value_copy(
        context, value, _type_of(value) if glsl_type is None else glsl_type,
        token,
    )


def _stored_roots(statements):
    """Names whose fields or elements ``statements`` store into."""

    roots = set()
    for statement in statements:
        for node in ast.walk(statement):
            targets = (
                node.targets if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else []
            )
            for target in targets:
                for element in (
                    target.elts if isinstance(target, ast.Tuple) else [target]
                ):
                    if not isinstance(element, (ast.Attribute, ast.Subscript)):
                        continue
                    root = element
                    while isinstance(root, (ast.Attribute, ast.Subscript)):
                        root = root.value
                    if isinstance(root, ast.Name):
                        roots.add(root.id)
    return roots


def _helper_definition(context, name, signature, statements, token):
    """One helper as a Python function; out/inout values become results.

    GLSL copies ``out``/``inout`` parameters back to the caller on return.
    Each ``return`` therefore returns ``(result, out...)`` (or the outs alone
    for a void helper); the caller's call site unpacks them in order.  An
    ``out`` parameter is undefined on entry and starts at its type's zero.
    """

    carried = signature.carried_parameters
    outputs = [carried[i] for i in signature.outputs]
    body = list(statements)

    class ReturnCopies(ast.NodeTransformer):
        """A returned vector, struct or array is a copy (_value_copy)."""

        def visit_Return(self, node):
            if node.value is not None:
                node.value = _copied_read(
                    context, node.value, token,
                    _type_of(node.value) or signature.result_type,
                )
            return node

        def visit_FunctionDef(self, node):
            return node

    body = [ReturnCopies().visit(statement) for statement in body]
    # An ``in``/``inout`` parameter is the helper's own copy of its actual:
    # one whose fields or elements the helper stores into starts as a copy,
    # so the caller's object is never written (_value_copy).  The copy is a
    # block-to-open spelling change: it gets its own Python name
    # (``d__glsl_scope_N``, as a redeclared block local does) and the body
    # uses that name, so the parameter itself is only ever read.  A mutable
    # global array is the caller's storage by design (carried, not a
    # parameter) and is written in place.
    stored = _stored_roots(body)
    copies = {
        parameter: (kind, context.temporaries.fresh().replace(
            "__glsl_value_", f"{parameter}__glsl_scope_"))
        for qualifier, kind, parameter in signature.parameters
        if qualifier != "out" and parameter in stored
        and not (kind in _SCALAR_TYPES or (
            _array_type(kind) is None and context.types.record(kind) is None
        ))
    }

    class CopyNames(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in copies:
                node.id = copies[node.id][1]
            return node

        def visit_FunctionDef(self, node):
            return node

    body = [CopyNames().visit(statement) for statement in body]
    body = [
        ast.AnnAssign(
            target=ast.Name(id=local, ctx=ast.Store()),
            annotation=_value_annotation(context, kind),
            value=_value_copy(
                context, _typed(_name(parameter), kind), kind, token,
            ),
            simple=1,
        )
        for parameter, (kind, local) in copies.items()
    ] + body
    if outputs:
        entry = [ast.AnnAssign(target=ast.Name(id=parameter, ctx=ast.Store()),
                               annotation=context.types.annotation(kind),
                               value=_zero_value(context, kind, token), simple=1)
                 for qualifier, kind, parameter in outputs if qualifier == "out"]
        # Each out/inout value goes back to the caller as GLSL copies it.
        names = [
            _value_copy(context, _typed(_name(
                copies[parameter][1] if parameter in copies else parameter
            ), kind), kind, token)
            for _, kind, parameter in outputs
        ]

        class OutputReturns(ast.NodeTransformer):
            def visit_Return(self, node):
                values = ([] if node.value is None else [node.value]) + copy.deepcopy(names)
                node.value = values[0] if len(values) == 1 else ast.Tuple(elts=values, ctx=ast.Load())
                return node

            def visit_FunctionDef(self, node):
                return node

        body = [OutputReturns().visit(statement) for statement in body]
        if signature.result_type == "void" and not (body and isinstance(body[-1], ast.Return)):
            body.append(OutputReturns().visit(ast.Return(value=None)))
        body = entry + body
    if outputs:
        # The returned values' declared types: the result (unless void) and
        # each out/inout value, as a ``tuple[...]`` when there are several.
        # Undeclared, a returned ``int`` member took the Python class
        # default (int64) against its caller's ``'i32'`` binding.
        returned = ([signature.result_type] if signature.result_type != "void" else []) + [
            kind for _qualifier, kind, _name in outputs]
        annotations = [_value_annotation(context, kind) for kind in returned]
        returns = (annotations[0] if len(annotations) == 1 else ast.Subscript(
            value=_name("tuple"), slice=ast.Tuple(elts=annotations, ctx=ast.Load()),
            ctx=ast.Load()))
    else:
        returns = (None if signature.result_type == "void"
                   else context.types.annotation(signature.result_type)
                   if context.types.known(signature.result_type)
                   else ast.Constant(_dtype(signature.result_type)))
    return ast.FunctionDef(
        name=name,
        args=ast.arguments(posonlyargs=[], args=[
            *(ast.arg(arg=parameter, annotation=_value_annotation(context, kind))
              for _, kind, parameter in carried),
            *(ast.arg(arg=buffer, annotation=_name("AbstractTensor"))
              for buffer in signature.buffers)],
            kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=body or [ast.Pass()], decorator_list=[], returns=returns)


_ARRAY_TYPE = re.compile(r"(?P<element>[A-Za-z_]\w*)\[(?P<extent>\d+)\]")


def _array_type(glsl_type):
    """``(element, extent)`` of a fixed array type ``T[N]``, or None."""
    match = _ARRAY_TYPE.fullmatch(str(glsl_type))
    return None if match is None else (match.group("element"), int(match.group("extent")))


def _value_annotation(context, glsl_type):
    """The Python annotation of a value of ``glsl_type``: its record or
    scalar dtype, and ``list[...]`` of that for a fixed array (the form a
    local array is declared with)."""
    array = _array_type(glsl_type)
    if array is None:
        return context.types.annotation(glsl_type)
    return ast.Subscript(value=_name("list"),
                         slice=context.types.annotation(array[0]), ctx=ast.Load())


__all__ = ["GLSLASTLoweringResult", "lower_glsl_source_to_ast"]
