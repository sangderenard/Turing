"""Build a Python AST from the supported GLSL source slice.

This is the inspectable alternate route to repository SSA.  It shares the
GLSL tokenizer and lexical tables with direct SSA lowering, but produces only
ordinary Python AST nodes already understood by ProcessGraph's AST ingestion.
Composite GLSL conveniences are expanded while the AST is built; no GLSL-only
operator node is retained.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import re
from typing import Sequence

from .glsl_source_ingestion import (
    GLSLSourceShortfall,
    _ExpressionError,
    _GLOBAL,
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


class _ASTExpressionParser:
    def __init__(self, tokens: Sequence[_Token]):
        self.tokens = list(tokens)
        self.index = 0

    def parse(self) -> ast.expr:
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
            anchor = self.tokens[min(self.index, len(self.tokens) - 1)]
            found = "end of expression" if self._peek() is None else repr(
                self._peek().text
            )
            raise _ExpressionError(anchor, f"expected {spelling!r}, found {found}")
        self.index += 1
        return token

    def _expression(self, minimum: int) -> ast.expr:
        left = self._unary()
        while True:
            token = self._peek()
            precedence = -1 if token is None else _PRECEDENCE.get(token.text, -1)
            if precedence < minimum:
                break
            operation = self._take().text
            left = _binary(left, operation, self._expression(precedence + 1))
        if minimum == 0 and self._peek("?") is not None:
            self._take("?")
            if_true = self._expression(0)
            self._take(":")
            if_false = self._expression(0)
            left = ast.IfExp(test=left, body=if_true, orelse=if_false)
        return left

    def _unary(self) -> ast.expr:
        token = self._peek()
        if token is not None and token.text in GLSL_UNARY_TO_SSA:
            self._take()
            operation = {
                "-": ast.USub,
                "!": ast.Not,
                "~": ast.Invert,
            }[token.text]
            return ast.UnaryOp(op=operation(), operand=self._unary())
        if token is not None and token.text == "+":
            self._take()
            return self._unary()
        return self._primary()

    def _primary(self) -> ast.expr:
        token = self._take()
        if token.text == "(":
            value = self._expression(0)
            self._take(")")
            return value
        if token.text in {"true", "false"}:
            return ast.Constant(value=token.text == "true")
        if token.text[0].isdigit() or token.text[0] == ".":
            suffix = token.text[-1:] if token.text[-1:] in "fFuU" else ""
            spelling = token.text[:-1] if suffix else token.text
            value = (
                float(spelling)
                if any(character in spelling for character in ".eE")
                else int(spelling, 10)
            )
            return ast.Constant(value=value)
        if not re.fullmatch(r"[A-Za-z_]\w*", token.text):
            raise _ExpressionError(token, f"expected value, found {token.text!r}")
        if self._peek("(") is None:
            return _name(token.text)
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

    def _lower_call(self, token: _Token, args: Sequence[ast.expr]) -> ast.expr:
        name = token.text
        if name in GLSL_CASTS:
            self._arity(token, args, 1)
            return _call(name, args)
        if name == "pow":
            self._arity(token, args, 2)
            return ast.BinOp(left=args[0], op=ast.Pow(), right=args[1])
        if name == "mod":
            self._arity(token, args, 2)
            return ast.BinOp(left=args[0], op=ast.Mod(), right=args[1])
        if name in {"min", "max"}:
            self._arity(token, args, 2)
            comparison = ast.Lt() if name == "min" else ast.Gt()
            condition = ast.Compare(
                left=args[0], ops=[comparison], comparators=[args[1]]
            )
            return ast.IfExp(test=condition, body=args[0], orelse=args[1])
        if name == "clamp":
            self._arity(token, args, 3)
            lower = self._lower_call(_Token("max", token.line, token.column), args[:2])
            return self._lower_call(
                _Token("min", token.line, token.column), [lower, args[2]]
            )
        if name == "mix":
            self._arity(token, args, 3)
            return ast.BinOp(
                left=args[0],
                op=ast.Add(),
                right=ast.BinOp(
                    left=ast.BinOp(left=args[1], op=ast.Sub(), right=args[0]),
                    op=ast.Mult(),
                    right=args[2],
                ),
            )
        if name == "step":
            self._arity(token, args, 2)
            condition = ast.Compare(
                left=args[1], ops=[ast.Lt()], comparators=[args[0]]
            )
            return ast.IfExp(
                test=condition,
                body=ast.Constant(value=0.0),
                orelse=ast.Constant(value=1.0),
            )
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
            return ast.BinOp(
                left=ast.BinOp(left=t, op=ast.Mult(), right=t),
                op=ast.Mult(),
                right=ast.BinOp(
                    left=ast.Constant(value=3.0),
                    op=ast.Sub(),
                    right=ast.BinOp(
                        left=ast.Constant(value=2.0), op=ast.Mult(), right=t
                    ),
                ),
            )
        if name == "inversesqrt":
            self._arity(token, args, 1)
            return ast.BinOp(
                left=ast.Constant(value=1.0),
                op=ast.Div(),
                right=_call("sqrt", args),
            )
        if name == "length":
            self._arity(token, args, 1)
            return _call("sqrt", [
                ast.BinOp(left=args[0], op=ast.Mult(), right=args[0])
            ])
        if name == "distance":
            self._arity(token, args, 2)
            return self._lower_call(
                _Token("length", token.line, token.column),
                [ast.BinOp(left=args[0], op=ast.Sub(), right=args[1])],
            )
        if name in GLSL_DIRECT_CALLS:
            return _call(GLSL_DIRECT_CALLS[name], args)
        if name in WEBGL_UNLOWERED_CALLS:
            raise _ExpressionError(
                token,
                f"{name} has no exact existing SSA/ProcessGraph operation",
            )
        raise _ExpressionError(token, f"unsupported GLSL call {name!r}")


def lower_glsl_source_to_ast(
    source: str,
    *,
    function_name: str | None = None,
) -> GLSLASTLoweringResult:
    """Parse the supported GLSL slice into an ordinary Python ``ast.Module``."""

    cleaned = _without_comments(str(source))
    shortfalls: list[GLSLSourceShortfall] = []
    try:
        body, body_line, entry_name = _main_body(cleaned)
    except ValueError as error:
        shortfalls.append(GLSLSourceShortfall("GLSL_ENTRY", 1, 1, str(error)))
        return GLSLASTLoweringResult(ast.Module(body=[], type_ignores=[]), tuple(shortfalls))

    prefix = cleaned[:cleaned.find(body)]
    arguments: list[str] = []
    output_names: list[str] = []
    for match in _GLOBAL.finditer(prefix):
        storage, name = match.group("storage"), match.group("name")
        (arguments if storage in {"uniform", "in"} else output_names).append(name)

    statements: list[ast.stmt] = []
    assigned_outputs: list[str] = []
    explicit_return = False
    for statement, line in _statements(body, body_line):
        if not statement:
            continue
        if "{" in statement or "}" in statement or re.match(
            r"^(if|for|while|switch|do)\b", statement
        ):
            shortfalls.append(GLSLSourceShortfall(
                "GLSL_CONTROL", line, 1,
                "control-flow statement is not in the straight-line source subset",
            ))
            continue
        try:
            tokens = _tokenize(statement, start_line=line)
        except _ExpressionError as error:
            shortfalls.append(GLSLSourceShortfall(
                "GLSL_TOKEN", error.token.line, error.token.column, str(error)
            ))
            continue
        cursor = 0
        while cursor < len(tokens) and tokens[cursor].text in _QUALIFIERS:
            cursor += 1
        if cursor < len(tokens) and (
            tokens[cursor].text in _SCALAR_TYPES
            or re.fullmatch(r"[biu]?vec[234]", tokens[cursor].text)
            or re.fullmatch(r"d?mat[234](?:x[234])?", tokens[cursor].text)
        ):
            cursor += 1
        tokens = tokens[cursor:]
        if not tokens:
            continue
        try:
            if tokens[0].text == "return":
                statements.append(ast.Return(
                    value=_ASTExpressionParser(tokens[1:]).parse()
                ))
                explicit_return = True
                continue
            assignment = next((
                index for index, token in enumerate(tokens)
                if token.text in {"=", "+=", "-=", "*=", "/=", "%="}
            ), None)
            if assignment != 1:
                raise _ExpressionError(
                    tokens[0], "only direct-name assignments are in the source subset"
                )
            target = tokens[0].text
            value = _ASTExpressionParser(tokens[2:]).parse()
            operation = tokens[1].text
            if operation != "=":
                value = _binary(_name(target), operation[0], value)
            statements.append(ast.Assign(
                targets=[ast.Name(id=target, ctx=ast.Store())], value=value
            ))
            if target in output_names and target not in assigned_outputs:
                assigned_outputs.append(target)
        except _ExpressionError as error:
            shortfalls.append(GLSLSourceShortfall(
                "GLSL_EXPRESSION",
                error.token.line,
                error.token.column,
                str(error),
            ))

    if not explicit_return and assigned_outputs:
        value: ast.expr = _name(assigned_outputs[0])
        if len(assigned_outputs) > 1:
            value = ast.Tuple(
                elts=[_name(name) for name in assigned_outputs], ctx=ast.Load()
            )
        statements.append(ast.Return(value=value))
    elif not explicit_return:
        shortfalls.append(GLSLSourceShortfall(
            "GLSL_OUTPUT", body_line, 1,
            "entry function produced no return value or declared output",
        ))

    function = ast.FunctionDef(
        name=function_name or entry_name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in arguments],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=statements or [ast.Pass()],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(
        body=[function], type_ignores=[]
    ))
    return GLSLASTLoweringResult(module, tuple(shortfalls))


__all__ = ["GLSLASTLoweringResult", "lower_glsl_source_to_ast"]
