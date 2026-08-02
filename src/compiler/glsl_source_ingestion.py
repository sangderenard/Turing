"""Lower a conservative GLSL expression subset directly into repository SSA.

The reader is deliberately not a second GLSL IR.  Recognised syntax evaporates
into existing ``Handler`` instructions immediately.  A construct that cannot be
represented is omitted and returned through the lowering shortfall list, which
is the same fail-visible boundary used by the other compiler front ends.

The first supported slice is straight-line shader arithmetic: declarations,
assignments, scalar calls, ternaries, and returns/output writes.  Common GLSL
conveniences such as ``mix``, ``clamp`` and ``smoothstep`` are decomposed into
existing arithmetic/comparison/select instructions instead of becoming ops.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

from ..transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue
from ..transmogrifier.ssa_registry import Handler
from .glsl_source_tables import (
    GLSL_BINARY_TO_SSA,
    GLSL_CASTS,
    GLSL_DIRECT_CALLS,
    GLSL_UNARY_TO_SSA,
    WEBGL_UNLOWERED_CALLS,
)


# MEMORY-SPECIALIZATION TODO:
# Desktop GLSL needs its own SSA memory handler before source ingestion grows
# storage operations.  The GLSL backend deliberately packs values into a
# mildly unique uint-word arena because SSBO binding/channel limits make the
# ordinary "one logical value/region per binding" model untenable.  Loads,
# stores, addressing, alias/lifetime accounting, and arena views must therefore
# lower through a GLSL-arena specialization selected from the existing SSA
# memory operations; do not encode arena mechanics as new numerical opcodes.
# WebGL remains a separate texture/raster ABI and must not borrow that handler.


@dataclass(frozen=True, order=True)
class GLSLSourceShortfall:
    code: str
    line: int
    column: int
    message: str

    def format(self) -> str:
        return f"{self.code} at {self.line}:{self.column}: {self.message}"


@dataclass(frozen=True)
class GLSLSSALoweringResult:
    module: IRModule
    shortfalls: tuple[GLSLSourceShortfall, ...]

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    def shortfall_report(self) -> str:
        if not self.shortfalls:
            return "GLSL source to SSA: complete"
        return "GLSL source to SSA shortfalls:\n" + "\n".join(
            f"- {item.format()}" for item in self.shortfalls
        )


@dataclass(frozen=True)
class _Token:
    text: str
    line: int
    column: int


class _ExpressionError(ValueError):
    def __init__(self, token: _Token, message: str):
        self.token = token
        super().__init__(message)


_TOKEN = re.compile(
    r"(?P<space>\s+)"
    r"|(?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFuU]?)"
    r"|(?P<name>[A-Za-z_]\w*)"
    r"|(?P<operator>==|!=|<=|>=|&&|\|\||<<|>>|\+=|-=|\*=|/=|%=|"
    r"[+\-*/%<>&|^!~?:=(),.\[\]])"
)

_BINARY_PRECEDENCE = {
    "||": 1,
    "&&": 2,
    "|": 3,
    "^": 4,
    "&": 5,
    "==": 6,
    "!=": 6,
    "<": 7,
    "<=": 7,
    ">": 7,
    ">=": 7,
    "<<": 8,
    ">>": 8,
    "+": 9,
    "-": 9,
    "*": 10,
    "/": 10,
    "%": 10,
}

_QUALIFIERS = {
    "const", "highp", "lowp", "mediump", "precise", "readonly",
    "restrict", "volatile", "writeonly",
}
_SCALAR_TYPES = frozenset(GLSL_CASTS)


def _without_comments(source: str) -> str:
    source = re.sub(
        r"/\*.*?\*/",
        lambda match: "\n" * match.group(0).count("\n"),
        source,
        flags=re.DOTALL,
    )
    source = re.sub(r"//[^\n]*", "", source)
    return re.sub(r"^[ \t]*#[^\n]*", "", source, flags=re.MULTILINE)


def _tokenize(text: str, *, start_line: int = 1) -> list[_Token]:
    tokens: list[_Token] = []
    position = 0
    line = start_line
    column = 1
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None:
            token = _Token(text[position], line, column)
            raise _ExpressionError(token, f"unrecognised character {token.text!r}")
        spelling = match.group(0)
        if match.lastgroup != "space":
            tokens.append(_Token(spelling, line, column))
        lines = spelling.splitlines(keepends=True)
        if len(lines) > 1:
            line += len(lines) - 1
            column = len(lines[-1]) + 1
        else:
            column += len(spelling)
        position = match.end()
    return tokens


def _dtype(glsl_type: str | None) -> str | None:
    return {
        "bool": "bool",
        "float": "float32",
        "double": "float64",
        "int": "i32",
        "uint": "u32",
    }.get(str(glsl_type), glsl_type)


class _SSABuilder:
    def __init__(self, name: str):
        self.name = name
        self.next_id = 0
        self.arguments: list[SSAValue] = []
        self.instructions: list[Instr] = []
        self.bindings: dict[str, SSAValue] = {}
        self.binding_types: dict[str, str | None] = {}
        self.outputs: list[tuple[str, SSAValue]] = []
        self.shortfalls: list[GLSLSourceShortfall] = []

    def fresh(self, dtype: str | None = None) -> SSAValue:
        value = SSAValue(self.next_id, dtype=dtype)
        self.next_id += 1
        return value

    def argument(self, name: str, dtype: str | None) -> SSAValue:
        value = self.fresh(_dtype(dtype))
        self.arguments.append(value)
        self.bindings[name] = value
        self.binding_types[name] = dtype
        return value

    def emit(
        self,
        handler: Handler,
        args: Sequence[SSAValue] = (),
        *,
        dtype: str | None = None,
        attributes: dict | None = None,
        token: _Token | None = None,
    ) -> SSAValue:
        value = self.fresh(dtype)
        span = None if token is None else {
            "line": token.line,
            "column": token.column,
            "surface": "glsl",
        }
        self.instructions.append(Instr(
            handler.value,
            list(args),
            value,
            attributes=dict(attributes or {}),
            source_span=span,
        ))
        return value

    def constant(self, token: _Token) -> SSAValue:
        spelling = token.text
        if spelling in {"true", "false"}:
            value, dtype = spelling == "true", "bool"
        else:
            suffix = spelling[-1:] if spelling[-1:] in "fFuU" else ""
            number = spelling[:-1] if suffix else spelling
            if suffix.lower() == "u":
                value, dtype = int(number, 10), "u32"
            elif any(character in number for character in ".eE"):
                value, dtype = float(number), "float32"
            else:
                value, dtype = int(number, 10), "i32"
        return self.emit(
            Handler.Const,
            dtype=dtype,
            attributes={"value": value},
            token=token,
        )

    def report(self, token: _Token, code: str, message: str) -> None:
        self.shortfalls.append(GLSLSourceShortfall(
            code, token.line, token.column, message
        ))

    def call(self, token: _Token, args: Sequence[SSAValue]) -> SSAValue:
        name = token.text
        if name in GLSL_CASTS:
            if len(args) != 1:
                raise _ExpressionError(
                    token, f"scalar constructor {name} expects one operand"
                )
            return self.emit(
                GLSL_CASTS[name], args, dtype=_dtype(name),
                attributes={"target_type": _dtype(name)}, token=token,
            )
        if name == "pow":
            return self._arity_emit(token, Handler.Pow, args, 2)
        if name == "mod":
            return self._arity_emit(token, Handler.Mod, args, 2)
        if name in {"min", "max"}:
            self._require_arity(token, args, 2)
            comparison = Handler.Lt if name == "min" else Handler.Gt
            condition = self.emit(comparison, args, dtype="bool", token=token)
            return self.emit(
                Handler.Select,
                [condition, args[0], args[1]],
                token=token,
            )
        if name == "clamp":
            self._require_arity(token, args, 3)
            lower = self.call(_Token("max", token.line, token.column), args[:2])
            return self.call(
                _Token("min", token.line, token.column), [lower, args[2]]
            )
        if name == "mix":
            self._require_arity(token, args, 3)
            delta = self.emit(Handler.Sub, [args[1], args[0]], token=token)
            scaled = self.emit(Handler.Mul, [delta, args[2]], token=token)
            return self.emit(Handler.Add, [args[0], scaled], token=token)
        if name == "step":
            self._require_arity(token, args, 2)
            condition = self.emit(
                Handler.Lt, [args[1], args[0]], dtype="bool", token=token
            )
            zero = self.constant(_Token("0.0", token.line, token.column))
            one = self.constant(_Token("1.0", token.line, token.column))
            return self.emit(
                Handler.Select, [condition, zero, one], token=token
            )
        if name == "smoothstep":
            self._require_arity(token, args, 3)
            numerator = self.emit(Handler.Sub, [args[2], args[0]], token=token)
            denominator = self.emit(Handler.Sub, [args[1], args[0]], token=token)
            ratio = self.emit(Handler.Div, [numerator, denominator], token=token)
            zero = self.constant(_Token("0.0", token.line, token.column))
            one = self.constant(_Token("1.0", token.line, token.column))
            t = self.call(
                _Token("clamp", token.line, token.column), [ratio, zero, one]
            )
            t2 = self.emit(Handler.Mul, [t, t], token=token)
            two = self.constant(_Token("2.0", token.line, token.column))
            three = self.constant(_Token("3.0", token.line, token.column))
            twice_t = self.emit(Handler.Mul, [two, t], token=token)
            curve = self.emit(Handler.Sub, [three, twice_t], token=token)
            return self.emit(Handler.Mul, [t2, curve], token=token)
        if name == "inversesqrt":
            self._require_arity(token, args, 1)
            root = self._direct_call(token, "sqrt", args)
            one = self.constant(_Token("1.0", token.line, token.column))
            return self.emit(Handler.Div, [one, root], token=token)
        if name == "length":
            self._require_arity(token, args, 1)
            square = self.emit(Handler.Mul, [args[0], args[0]], token=token)
            return self._direct_call(token, "sqrt", [square])
        if name == "distance":
            self._require_arity(token, args, 2)
            delta = self.emit(Handler.Sub, [args[0], args[1]], token=token)
            return self.call(
                _Token("length", token.line, token.column), [delta]
            )
        if name in GLSL_DIRECT_CALLS:
            return self._direct_call(token, GLSL_DIRECT_CALLS[name], args)
        if name in WEBGL_UNLOWERED_CALLS:
            raise _ExpressionError(
                token,
                f"{name} has no exact existing SSA/ProcessGraph operation",
            )
        raise _ExpressionError(token, f"unsupported GLSL call {name!r}")

    def _direct_call(
        self, token: _Token, callee: str, args: Sequence[SSAValue]
    ) -> SSAValue:
        return self.emit(
            Handler.Call,
            args,
            attributes={"callee": callee},
            token=token,
        )

    @staticmethod
    def _require_arity(
        token: _Token, args: Sequence[SSAValue], expected: int
    ) -> None:
        if len(args) != expected:
            raise _ExpressionError(
                token,
                f"{token.text} expects {expected} operands, got {len(args)}",
            )

    def _arity_emit(
        self,
        token: _Token,
        handler: Handler,
        args: Sequence[SSAValue],
        arity: int,
    ) -> SSAValue:
        self._require_arity(token, args, arity)
        return self.emit(handler, args, token=token)

    def finish(self) -> IRModule:
        returns = [value for _, value in self.outputs]
        self.instructions.append(Instr(Handler.Ret.value, returns, None))
        function = Function(
            self.name,
            self.arguments,
            {"entry": BasicBlock("entry", self.instructions)},
        )
        return IRModule({self.name: function})


class _ExpressionParser:
    def __init__(self, tokens: Sequence[_Token], builder: _SSABuilder):
        self.tokens = list(tokens)
        self.builder = builder
        self.index = 0

    def parse(self) -> SSAValue:
        if not self.tokens:
            raise _ExpressionError(_Token("", 1, 1), "empty expression")
        value = self._expression(0)
        if self.index != len(self.tokens):
            token = self.tokens[self.index]
            raise _ExpressionError(token, f"unexpected token {token.text!r}")
        return value

    def _peek(self, spelling: str | None = None) -> _Token | None:
        if self.index >= len(self.tokens):
            return None
        token = self.tokens[self.index]
        return token if spelling is None or token.text == spelling else None

    def _take(self, spelling: str | None = None) -> _Token:
        token = self._peek(spelling)
        if token is None:
            found = "end of expression" if self._peek() is None else repr(
                self._peek().text
            )
            anchor = self.tokens[min(self.index, len(self.tokens) - 1)]
            raise _ExpressionError(anchor, f"expected {spelling!r}, found {found}")
        self.index += 1
        return token

    def _expression(self, minimum: int) -> SSAValue:
        left = self._unary()
        while True:
            token = self._peek()
            precedence = -1 if token is None else _BINARY_PRECEDENCE.get(
                token.text, -1
            )
            if precedence < minimum:
                break
            operation = self._take()
            right = self._expression(precedence + 1)
            left = self.builder.emit(
                GLSL_BINARY_TO_SSA[operation.text],
                [left, right],
                dtype=(
                    "bool"
                    if operation.text in {
                        "&&", "||", "==", "!=", "<", "<=", ">", ">="
                    }
                    else None
                ),
                token=operation,
            )
        if minimum == 0 and self._peek("?") is not None:
            token = self._take("?")
            if_true = self._expression(0)
            self._take(":")
            if_false = self._expression(0)
            left = self.builder.emit(
                Handler.Select, [left, if_true, if_false], token=token
            )
        return left

    def _unary(self) -> SSAValue:
        token = self._peek()
        if token is not None and token.text in GLSL_UNARY_TO_SSA:
            self._take()
            return self.builder.emit(
                GLSL_UNARY_TO_SSA[token.text],
                [self._unary()],
                dtype="bool" if token.text == "!" else None,
                token=token,
            )
        if token is not None and token.text == "+":
            self._take()
            return self._unary()
        return self._primary()

    def _primary(self) -> SSAValue:
        token = self._take()
        if token.text == "(":
            value = self._expression(0)
            self._take(")")
            return value
        if re.fullmatch(_TOKEN.pattern, token.text) and (
            token.text[0].isdigit() or token.text[0] == "."
        ):
            return self.builder.constant(token)
        if token.text in {"true", "false"}:
            return self.builder.constant(token)
        if not re.fullmatch(r"[A-Za-z_]\w*", token.text):
            raise _ExpressionError(token, f"expected value, found {token.text!r}")
        if self._peek("(") is not None:
            self._take("(")
            args: list[SSAValue] = []
            if self._peek(")") is None:
                while True:
                    args.append(self._expression(0))
                    if self._peek(",") is None:
                        break
                    self._take(",")
            self._take(")")
            return self.builder.call(token, args)
        value = self.builder.bindings.get(token.text)
        if value is None:
            raise _ExpressionError(token, f"unbound GLSL name {token.text!r}")
        return value


def _main_body(source: str) -> tuple[str, int, str]:
    match = re.search(r"\bvoid\s+(\w+)\s*\([^)]*\)\s*\{", source)
    if match is None:
        raise ValueError("GLSL source has no void entry function")
    depth = 1
    position = match.end()
    while position < len(source) and depth:
        depth += (source[position] == "{") - (source[position] == "}")
        position += 1
    if depth:
        raise ValueError("GLSL entry function has an unmatched opening brace")
    body_start = match.end()
    return (
        source[body_start:position - 1],
        source.count("\n", 0, body_start) + 1,
        match.group(1),
    )


def _statements(body: str, start_line: int) -> Iterable[tuple[str, int]]:
    depth = 0
    begin = 0
    line = start_line
    statement_line = line
    for index, character in enumerate(body):
        if character == "\n":
            line += 1
        elif character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        elif character == ";" and depth == 0:
            yield body[begin:index].strip(), statement_line
            begin = index + 1
            statement_line = line
        elif character in "{}" and depth == 0:
            # Control blocks are outside the first straight-line tranche.  Keep
            # them together so the caller emits one useful shortfall.
            depth += 1 if character == "{" else -1
    tail = body[begin:].strip()
    if tail:
        yield tail, statement_line


_GLOBAL = re.compile(
    r"(?:layout\s*\([^)]*\)\s*)?"
    r"\b(?P<storage>uniform|in|out)\s+"
    r"(?:(?:highp|mediump|lowp|flat|smooth|noperspective|centroid)\s+)*"
    r"(?P<type>[A-Za-z_]\w*)\s+(?P<name>[A-Za-z_]\w*)"
    r"(?:\s*\[[^]]*\])?\s*;"
)


def lower_glsl_source_to_ssa(
    source: str,
    *,
    function_name: str | None = None,
) -> GLSLSSALoweringResult:
    """Lower one scalar, straight-line GLSL entry function to existing SSA."""

    cleaned = _without_comments(str(source))
    try:
        body, body_line, entry_name = _main_body(cleaned)
    except ValueError as error:
        name = function_name or "main"
        builder = _SSABuilder(name)
        builder.shortfalls.append(GLSLSourceShortfall(
            "GLSL_ENTRY", 1, 1, str(error)
        ))
        return GLSLSSALoweringResult(
            builder.finish(), tuple(builder.shortfalls)
        )

    builder = _SSABuilder(function_name or entry_name)
    output_names: list[str] = []
    prefix = cleaned[:cleaned.find(body)]
    for match in _GLOBAL.finditer(prefix):
        storage, glsl_type, name = (
            match.group("storage"), match.group("type"), match.group("name")
        )
        if storage in {"uniform", "in"}:
            builder.argument(name, glsl_type)
        else:
            output_names.append(name)
            builder.binding_types[name] = glsl_type

    for statement, line in _statements(body, body_line):
        if not statement:
            continue
        if "{" in statement or "}" in statement or re.match(
            r"^(if|for|while|switch|do)\b", statement
        ):
            builder.shortfalls.append(GLSLSourceShortfall(
                "GLSL_CONTROL", line, 1,
                "control-flow statement is not in the straight-line source subset",
            ))
            continue
        try:
            tokens = _tokenize(statement, start_line=line)
        except _ExpressionError as error:
            builder.report(error.token, "GLSL_TOKEN", str(error))
            continue
        if not tokens:
            continue

        declaration_type = None
        cursor = 0
        while cursor < len(tokens) and tokens[cursor].text in _QUALIFIERS:
            cursor += 1
        if cursor < len(tokens) and (
            tokens[cursor].text in _SCALAR_TYPES
            or re.fullmatch(r"[biu]?vec[234]", tokens[cursor].text)
            or re.fullmatch(r"d?mat[234](?:x[234])?", tokens[cursor].text)
        ):
            declaration_type = tokens[cursor].text
            cursor += 1
        tokens = tokens[cursor:]
        if not tokens:
            continue

        if tokens[0].text == "return":
            try:
                value = _ExpressionParser(tokens[1:], builder).parse()
            except _ExpressionError as error:
                builder.report(error.token, "GLSL_EXPRESSION", str(error))
            else:
                builder.outputs.append(("return", value))
            continue

        assignment = next((
            index for index, token in enumerate(tokens)
            if token.text in {"=", "+=", "-=", "*=", "/=", "%="}
        ), None)
        if assignment is None:
            builder.report(
                tokens[0], "GLSL_STATEMENT",
                "statement has no representable SSA assignment or return",
            )
            continue
        if assignment != 1 or not re.fullmatch(
            r"[A-Za-z_]\w*", tokens[0].text
        ):
            builder.report(
                tokens[0], "GLSL_LVALUE",
                "only direct-name assignments are in the source subset",
            )
            continue
        name, operation = tokens[0].text, tokens[assignment].text
        try:
            value = _ExpressionParser(tokens[assignment + 1:], builder).parse()
            if operation != "=":
                previous = builder.bindings.get(name)
                if previous is None:
                    raise _ExpressionError(
                        tokens[0], f"compound assignment to unbound name {name!r}"
                    )
                value = builder.emit(
                    GLSL_BINARY_TO_SSA[operation[0]],
                    [previous, value],
                    token=tokens[assignment],
                )
        except _ExpressionError as error:
            builder.report(error.token, "GLSL_EXPRESSION", str(error))
            continue
        builder.bindings[name] = value
        if declaration_type is not None:
            builder.binding_types[name] = declaration_type
            value.dtype = _dtype(declaration_type)
        if name in output_names:
            builder.outputs = [
                item for item in builder.outputs if item[0] != name
            ]
            builder.outputs.append((name, value))

    if not builder.outputs:
        builder.shortfalls.append(GLSLSourceShortfall(
            "GLSL_OUTPUT", body_line, 1,
            "entry function produced no return value or declared output",
        ))
    return GLSLSSALoweringResult(
        builder.finish(), tuple(builder.shortfalls)
    )


__all__ = [
    "GLSLSSALoweringResult",
    "GLSLSourceShortfall",
    "lower_glsl_source_to_ssa",
]
