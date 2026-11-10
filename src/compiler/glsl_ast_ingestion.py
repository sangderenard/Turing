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
    def __init__(self, tokens: Sequence[_Token], *, source_calls=None):
        self.tokens = list(tokens)
        self.index = 0
        self.source_calls = dict(source_calls or {})

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

    def _expression(self, minimum: int) -> ast.expr:
        left = self._unary()
        while True:
            token = self._peek()
            precedence = -1 if token is None else _PRECEDENCE.get(token.text, -1)
            if precedence < minimum:
                break
            operation = self._take()
            left = _binary(left, operation.text, self._expression(precedence + 1))
            left._glsl_source_location = (operation.line,operation.column,operation.text)
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
        value = self._primary()
        while self._peek() is not None:
            if self._peek("["):
                self._take("[")
                index = self._expression(0)
                self._take("]")
                value = ast.Subscript(value=value, slice=index, ctx=ast.Load())
            elif self._peek("."):
                self._take(".")
                field = self._take()
                if not re.fullmatch(r"[A-Za-z_]\w*", field.text):
                    raise _ExpressionError(field, "Expected field name")
                value = ast.Attribute(value=value, attr=field.text, ctx=ast.Load())
            else:
                break
        return value

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
            node = ast.Constant(value=value)
            node._glsl_scalar_dtype = "uint32" if suffix.lower()=="u" else "float32" if isinstance(value,float) else "int32"
            node._glsl_source_location = (token.line,token.column,token.text)
            return node
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
        if name in self.source_calls:
            self._arity(token, args, self.source_calls[name])
            return _call(name, args)
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


class _ASTStatementReader(_ASTExpressionParser):
    """Read statement syntax into the existing ordinary AST ingestion surface.

    There is no shader IR or direct SSA writer here. The public source compiler
    remains responsible for control flow, aliasing, types and identity records.
    """
    def __init__(self, tokens, *, reserved_names=(), source_calls=None):
        super().__init__(tokens, source_calls=source_calls)
        self.declarations = set(reserved_names)

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

    def _statement(self):
        token = self._peek()
        if token.text == "{":
            self._take()
            return self.read_all("}")
        if token.text in {"if", "while"}:
            self._take()
            test = _ASTExpressionParser(self.delimited("(", ")"), source_calls=self.source_calls).parse()
            body = self.statement() or [ast.Pass()]
            otherwise = []
            if token.text == "if" and self._peek("else"):
                self._take()
                otherwise = self.statement()
            node = (ast.If if token.text == "if" else ast.While)(
                test=test, body=body, orelse=otherwise)
            return [node]
        if token.text == "for":
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
            test = _ASTExpressionParser(parts[1], source_calls=self.source_calls).parse()
            update = self.simple(parts[2])
            if len(update) != 1 or not isinstance(update[0], ast.AugAssign) or not isinstance(update[0].target, ast.Name):
                raise _ExpressionError(token, "For update requires simple counter increment")
            body = self.statement()
            import copy
            class ContinueUpdate(ast.NodeTransformer):
                def visit_Continue(self, node): return copy.deepcopy(update) + [node]
                def visit_For(self, node): return node
                def visit_While(self, node): return node
            body = [ContinueUpdate().visit(item) for item in body]
            body = [node for item in body for node in (item if isinstance(item, list) else [item])]
            return initial + [ast.While(test=test, body=body + update, orelse=[])]
        begin = self.index
        while self._peek() and not self._peek(";"):
            if self._peek("}"):
                raise _ExpressionError(self._peek(), "Missing statement semicolon")
            self._take()
        self._take(";")
        return self.simple(self.tokens[begin:self.index - 1])

    def simple(self, tokens):
        if not tokens: return []
        token = tokens[0]
        if token.text == "return":
            return [ast.Return(value=_ASTExpressionParser(tokens[1:], source_calls=self.source_calls).parse() if len(tokens)>1 else None)]
        if token.text in {"break", "continue"} and len(tokens) == 1:
            return [(ast.Break if token.text == "break" else ast.Continue)()]
        declaration = None
        while tokens and tokens[0].text in _QUALIFIERS: tokens = tokens[1:]
        if tokens and tokens[0].text in _SCALAR_TYPES:
            declaration = tokens[0].text
            tokens = tokens[1:]
            if not tokens or not re.fullmatch(r"[A-Za-z_]\w*", tokens[0].text):
                raise _ExpressionError(token, "Expected local declaration name")
            if tokens[0].text in self.declarations:
                raise _ExpressionError(tokens[0], "Shadowed/redeclared local requires lexical-scope lowering")
            self.declarations.add(tokens[0].text)
        if len(tokens) == 2 and (tokens[0].text in {"++", "--"} or tokens[-1].text in {"++", "--"}):
            op = tokens[0] if tokens[0].text in {"++", "--"} else tokens[-1]
            name = tokens[-1] if op is tokens[0] else tokens[0]
            return [ast.AugAssign(target=ast.Name(id=name.text, ctx=ast.Store()),
                op=ast.Add() if op.text=="++" else ast.Sub(), value=ast.Constant(1))]
        if declaration and len(tokens)==4 and tokens[1].text=="[" and tokens[3].text=="]":
            if not tokens[2].text.isdigit() or int(tokens[2].text)<=0:
                raise _ExpressionError(tokens[2], "Local array requires positive constant extent")
            raise _ExpressionError(tokens[0],
                "Local arrays require typed storage and definite-initialization lowering")
        assignment = next((i for i,t in enumerate(tokens) if t.text in {"=","+=","-=","*=","/=","%="}),None)
        if assignment is None:
            raise _ExpressionError(token, "Expected assignment or supported statement")
        target = _ASTExpressionParser(tokens[:assignment], source_calls=self.source_calls).parse()
        if not isinstance(target,(ast.Name,ast.Subscript,ast.Attribute)):
            raise _ExpressionError(token,"Invalid assignment target")
        target.ctx=ast.Store()
        value=_ASTExpressionParser(tokens[assignment+1:], source_calls=self.source_calls).parse()
        op=tokens[assignment].text
        if op=="=":
            if declaration:
                return [ast.AnnAssign(target=target,annotation=ast.Constant(_dtype(declaration)),
                    value=value,simple=int(isinstance(target,ast.Name)))]
            return [ast.Assign(targets=[target],value=value)]
        return [ast.AugAssign(target=target,op=_BINARY_AST[op[0]](),value=value)]


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

    source_functions = {}
    source_calls = {}
    for match in re.finditer(r"\b(void|float|double|int|uint|bool)\s+([A-Za-z_]\w*)\s*\(([^()]*)\)\s*\{", cleaned):
        result_type, helper_name, parameters = match.groups()
        if helper_name == entry_name:
            continue
        if helper_name in source_functions:
            shortfalls.append(GLSLSourceShortfall("GLSL_FUNCTION", 1, 1,
                "Overloaded/redeclared functions require signature resolution"))
            continue
        names=[]
        types=[]
        valid=True
        for parameter in parameters.split(",") if parameters.strip() else []:
            fields=parameter.strip().split()
            if fields and fields[0] in {"in", "const"}: fields=fields[1:]
            if len(fields)!=2 or fields[0] not in _SCALAR_TYPES:
                shortfalls.append(GLSLSourceShortfall("GLSL_PARAMETER", 1, 1,
                    "Helper parameter requires supported scalar in semantics: " + parameter))
                valid=False;break
            names.append(fields[1]);types.append(_dtype(fields[0]))
        if not valid: continue
        end=match.end();depth=1
        while end<len(cleaned) and depth:
            depth+=(cleaned[end]=="{")-(cleaned[end]=="}");end+=1
        if depth:
            shortfalls.append(GLSLSourceShortfall("GLSL_FUNCTION",1,1,"Unclosed helper body"));continue
        source_functions[helper_name]=(names,types,result_type,cleaned[match.end():end-1],cleaned.count("\n",0,match.end())+1,match.end()-cleaned.rfind("\n",0,match.end()))
        source_calls[helper_name]=len(names)

    prefix = cleaned[:cleaned.find(body)]
    arguments: list[str] = []
    output_names: list[str] = []
    argument_types = {}
    for match in _GLOBAL.finditer(prefix):
        storage, name = match.group("storage"), match.group("name")
        (arguments if storage in {"uniform", "in"} else output_names).append(name)
        if storage in {"uniform", "in"}: argument_types[name]=_dtype(match.group("type"))

    try:
        reader = _ASTStatementReader(_source_tokens(body, body_line, cleaned.find(body)-cleaned.rfind("\n",0,cleaned.find(body))), reserved_names=arguments, source_calls=source_calls)
        statements = reader.read_all()
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
    elif not explicit_return:
        shortfalls.append(GLSLSourceShortfall(
            "GLSL_OUTPUT", body_line, 1,
            "entry function produced no return value or declared output",
        ))

    function = ast.FunctionDef(
        name=function_name or entry_name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name,annotation=ast.Constant(argument_types[name])) for name in arguments],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=statements or [ast.Pass()],
        decorator_list=[],
    )
    helpers=[]
    for helper_name,(names,types,result_type,helper_body,helper_line,helper_column) in source_functions.items():
        try:
            helper_reader=_ASTStatementReader(_source_tokens(helper_body,helper_line,helper_column),
                reserved_names=names,source_calls=source_calls)
            helper_statements=helper_reader.read_all()
            helpers.append(ast.FunctionDef(name=helper_name,
                args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=name,annotation=ast.Constant(kind)) for name,kind in zip(names,types)],
                    kwonlyargs=[],kw_defaults=[],defaults=[]),
                body=helper_statements or [ast.Pass()],decorator_list=[],
                returns=ast.Constant(_dtype(result_type)) if result_type!="void" else None))
        except _ExpressionError as error:
            shortfalls.append(GLSLSourceShortfall("GLSL_HELPER",error.token.line,
                error.token.column,str(error)))
    module = ast.fix_missing_locations(ast.Module(
        body=[_retain_literal_widths(fn) for fn in helpers+[function]], type_ignores=[]
    ))
    return GLSLASTLoweringResult(module, tuple(shortfalls))


__all__ = ["GLSLASTLoweringResult", "lower_glsl_source_to_ast"]
