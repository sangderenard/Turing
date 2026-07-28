from enum import Enum
from typing import Dict, Callable, List, Any
from collections import defaultdict


BITOPS_EXPANDABLE_OPS = frozenset(
    {"bitand", "bitor", "bitxor", "invert", "add", "sub", "mul"}
)


class Handler(Enum):
    """
    Enumeration of all SSA operations used.
    Values correspond to SSA `Instr.op` names.
    """
    # Arithmetic
    Add           = "Add"
    Sub           = "Sub"
    Mul           = "Mul"
    Div           = "Div"
    FloorDiv      = "FloorDiv"
    Mod           = "Mod"
    Pow           = "Pow"
    Neg           = "Neg"
    Abs           = "Abs"

    # Bitwise
    And           = "And"
    Or            = "Or"
    Xor           = "Xor"
    Not           = "Not"
    Shl           = "Shl"
    Shr           = "Shr"

    # Logical
    LAnd          = "LAnd"
    LOr           = "LOr"
    LNot          = "LNot"

    # Comparison
    Eq            = "Eq"
    Ne            = "Ne"
    Lt            = "Lt"
    Le            = "Le"
    Gt            = "Gt"
    Ge            = "Ge"

    # Memory & Indexing
    Load          = "Load"
    Store         = "Store"
    Alloca        = "Alloca"
    GetElementPtr = "GetElementPtr"

    # Casts & Conversions
    Cast          = "Cast"
    Trunc         = "Trunc"
    ZExt          = "ZExt"
    SExt          = "SExt"
    FpToSi        = "FpToSi"
    FpToUi        = "FpToUi"
    SiToFp        = "SiToFp"
    UiToFp        = "UiToFp"

    # Control Flow
    Phi           = "Phi"
    Br            = "Br"
    CondBr        = "CondBr"
    Ret           = "Ret"
    Call          = "Call"

    # Misc
    Select        = "Select"
    Const         = "Const"  # literal constants

    def __str__(self) -> str:
        return self.value


# -----------------------------------------------------------------------------
# Sympy → SSA base name map
# -----------------------------------------------------------------------------
sympy_ssa_name_map: Dict[str, Handler] = {
    # Symbols & Variables
    'symbol':              Handler.Load,
    'var':                 Handler.Load,

    # Literals / Constants
    'integer':             Handler.Const,
    'float':               Handler.Const,
    'rational':            Handler.Const,
    'half':                Handler.Const,
    'pi':                  Handler.Const,
    'e':                   Handler.Const,
    'i':                   Handler.Const,
    'imaginaryunit':       Handler.Const,
    'true':                Handler.Const,
    'false':               Handler.Const,

    # Arithmetic
    'add':                 Handler.Add,
    'sub':                 Handler.Sub,
    'mul':                 Handler.Mul,
    'div':                 Handler.Div,
    'mod':                 Handler.Mod,
    'pow':                 Handler.Pow,
    'neg':                 Handler.Neg,
    'abs':                 Handler.Abs,

    # Bitwise
    'bitwise_and':         Handler.And,
    'bitwise_or':          Handler.Or,
    'bitwise_xor':         Handler.Xor,
    'invert':              Handler.Not,

    # Logical
    'and':                 Handler.LAnd,
    'or':                  Handler.LOr,
    'not':                 Handler.LNot,
    'xor':                 Handler.Xor,

    # Comparison
    'eq':                  Handler.Eq,
    'equality':            Handler.Eq,
    'ne':                  Handler.Ne,
    'unequality':          Handler.Ne,
    'lt':                  Handler.Lt,
    'strictlessthan':      Handler.Lt,
    'le':                  Handler.Le,
    'lessthanorequal':     Handler.Le,
    'gt':                  Handler.Gt,
    'strictgreaterthan':   Handler.Gt,
    'ge':                  Handler.Ge,
    'greaterthanorequal':  Handler.Ge,

    # Memory & Indexing
    'load':                Handler.Load,
    'store':               Handler.Store,
    'alloca':              Handler.Alloca,
    'getelementptr':       Handler.GetElementPtr,
    'idx':                 Handler.GetElementPtr,
    'indexed':             Handler.Load,
    'indexedbase':         Handler.Alloca,
    'matrixelement':       Handler.Load,
    'matrixsymbol':        Handler.Alloca,

    # Casts & Conversions
    'cast':                Handler.Cast,
    'trunc':               Handler.Trunc,
    'zext':                Handler.ZExt,
    'sext':                Handler.SExt,
    'fptosi':              Handler.FpToSi,
    'fptoui':              Handler.FpToUi,
    'sitofp':              Handler.SiToFp,
    'uitofp':              Handler.UiToFp,

    # Selection / Piecewise
    'select':              Handler.Select,
    'piecewise':           Handler.Select,
    'exprcondpair':        Handler.Select,

    # Control Flow
    'phi':                 Handler.Phi,
    'br':                  Handler.Br,
    'condbr':              Handler.CondBr,
    'ret':                 Handler.Ret,

    # Function-Calls (catch-all externals)
    'call':                Handler.Call,
    'sin':                 Handler.Call,
    'cos':                 Handler.Call,
    'tan':                 Handler.Call,
    'exp':                 Handler.Call,
    'log':                 Handler.Call,
    'sqrt':                Handler.Call,
    'floor':               Handler.Call,
    'ceiling':             Handler.Call,
    'round':               Handler.Call,
    'max':                 Handler.Call,
    'min':                 Handler.Call,
    'sum':                 Handler.Call,
    'matrix':              Handler.Call,
    'transpose':           Handler.Call,
    'inverse':             Handler.Call,
    'trace':               Handler.Call,
    'function':            Handler.Call,
}


# -----------------------------------------------------------------------------
# Python AST → the same SSA / BitOps language
# -----------------------------------------------------------------------------
#
# Keys use the lowercase spelling of ``ast`` class names.  Qualified spellings
# describe the one piece of a compound AST node that they represent; for
# example, ``binop:add`` and ``augassign:add`` are both the same addition
# operator after Python's surface syntax has been removed.  This is an
# equivalence table only: traversal still obtains operands from the ordinary
# AST role schemas.
ast_ssa_equivalents: Dict[Handler, tuple[str, ...]] = {
    # Values and Python literal spellings.
    Handler.Load: (
        'name',
        'load',
        'attribute:load',
    ),
    Handler.Const: (
        'constant',
        'num',
        'str',
        'bytes',
        'nameconstant',
        'ellipsis',
    ),
    Handler.Alloca: (
        'list',
        'tuple',
        'set',
        'dict',
        'listcomp',
        'setcomp',
        'dictcomp',
        'generatorexp',
        'lambda',
        'functiondef',
        'asyncfunctiondef',
        'classdef',
    ),

    # Arithmetic. Reflected and in-place Python methods have the same
    # operator meaning here; assignment remains a separate Store.
    Handler.Add: (
        'add',
        'binop:add',
        'augassign:add',
        'operator:add',
        '__add__',
        '__radd__',
        '__iadd__',
    ),
    Handler.Sub: (
        'sub',
        'binop:sub',
        'augassign:sub',
        'operator:sub',
        '__sub__',
        '__rsub__',
        '__isub__',
    ),
    Handler.Mul: (
        'mult',
        'binop:mult',
        'augassign:mult',
        'operator:mul',
        '__mul__',
        '__rmul__',
        '__imul__',
    ),
    Handler.Div: (
        'div',
        'binop:div',
        'augassign:div',
        'operator:truediv',
        '__truediv__',
        '__rtruediv__',
        '__itruediv__',
    ),
    Handler.FloorDiv: (
        'floordiv',
        'binop:floordiv',
        'augassign:floordiv',
        'operator:floordiv',
        '__floordiv__',
        '__rfloordiv__',
        '__ifloordiv__',
    ),
    Handler.Mod: (
        'mod',
        'binop:mod',
        'augassign:mod',
        'operator:mod',
        '__mod__',
        '__rmod__',
        '__imod__',
    ),
    Handler.Pow: (
        'pow',
        'binop:pow',
        'augassign:pow',
        'operator:pow',
        '__pow__',
        '__rpow__',
        '__ipow__',
    ),
    Handler.Neg: (
        'usub',
        'unaryop:usub',
        'operator:neg',
        '__neg__',
    ),
    Handler.Abs: (
        'call:abs',
        'operator:abs',
        '__abs__',
    ),

    # Bitwise operators correspond directly to BitOps/Turing logic.
    Handler.And: (
        'bitand',
        'binop:bitand',
        'augassign:bitand',
        'operator:and_',
        '__and__',
        '__rand__',
        '__iand__',
    ),
    Handler.Or: (
        'bitor',
        'binop:bitor',
        'augassign:bitor',
        'operator:or_',
        '__or__',
        '__ror__',
        '__ior__',
    ),
    Handler.Xor: (
        'bitxor',
        'binop:bitxor',
        'augassign:bitxor',
        'operator:xor',
        '__xor__',
        '__rxor__',
        '__ixor__',
    ),
    Handler.Not: (
        'invert',
        'unaryop:invert',
        'operator:invert',
        '__invert__',
    ),
    Handler.Shl: (
        'lshift',
        'binop:lshift',
        'augassign:lshift',
        'operator:lshift',
        '__lshift__',
        '__rlshift__',
        '__ilshift__',
    ),
    Handler.Shr: (
        'rshift',
        'binop:rshift',
        'augassign:rshift',
        'operator:rshift',
        '__rshift__',
        '__rrshift__',
        '__irshift__',
    ),

    # Boolean operations are distinct from integer bitwise operations but
    # lower to the same compact SSA vocabulary after type selection.
    Handler.LAnd: (
        'and',
        'boolop:and',
    ),
    Handler.LOr: (
        'or',
        'boolop:or',
    ),
    Handler.LNot: (
        'not',
        'unaryop:not',
        'operator:not_',
    ),

    # Comparisons. Chained Compare nodes repeat the corresponding primitive.
    Handler.Eq: (
        'eq',
        'compare:eq',
        'is',
        'compare:is',
        'operator:eq',
        'operator:is_',
        '__eq__',
    ),
    Handler.Ne: (
        'noteq',
        'compare:noteq',
        'isnot',
        'compare:isnot',
        'operator:ne',
        'operator:is_not',
        '__ne__',
    ),
    Handler.Lt: (
        'lt',
        'compare:lt',
        'operator:lt',
        '__lt__',
    ),
    Handler.Le: (
        'lte',
        'compare:lte',
        'operator:le',
        '__le__',
    ),
    Handler.Gt: (
        'gt',
        'compare:gt',
        'operator:gt',
        '__gt__',
    ),
    Handler.Ge: (
        'gte',
        'compare:gte',
        'operator:ge',
        '__ge__',
    ),

    # Storage and addressing.
    Handler.Store: (
        'store',
        'assign',
        'annassign',
        'namedexpr',
        'attribute:store',
        'subscript:store',
    ),
    Handler.GetElementPtr: (
        'subscript',
        'slice',
        'extslice',
        'index',
    ),

    # Type expressions collapse to the existing conversion operators. Generic
    # Python constructors use Cast; explicitly typed IR spellings retain their
    # narrower conversion handler.
    Handler.Cast: (
        'uadd',
        'unaryop:uadd',
        'call:bool',
        'call:int',
        'call:float',
        'call:complex',
        'call:bytes',
        'call:str',
        'call:list',
        'call:tuple',
        'call:set',
        'call:dict',
        'call:cast',
    ),
    Handler.Trunc: ('call:trunc',),
    Handler.ZExt: ('call:zext',),
    Handler.SExt: ('call:sext',),
    Handler.FpToSi: ('call:fptosi',),
    Handler.FpToUi: ('call:fptoui',),
    Handler.SiToFp: ('call:sitofp',),
    Handler.UiToFp: ('call:uitofp',),

    # Control and calls.
    Handler.Phi: (
        'if:name_merge',
        'for:name_merge',
        'while:name_merge',
        'try:name_merge',
    ),
    Handler.Br: (
        'for',
        'asyncfor',
        'while',
        'break',
        'continue',
    ),
    Handler.CondBr: (
        'if',
        'match',
        'assert',
    ),
    Handler.Ret: (
        'return',
        'yield',
        'yieldfrom',
    ),
    Handler.Call: (
        'call',
        'await',
        'formattedvalue',
        'joinedstr',
        'in',
        'notin',
        'compare:in',
        'compare:notin',
        'matmult',
        'binop:matmult',
    ),
    Handler.Select: (
        'ifexp',
        'match_case',
    ),
}


ast_ssa_name_map: Dict[str, Handler] = {
    spelling: handler
    for handler, spellings in ast_ssa_equivalents.items()
    for spelling in spellings
}


# The registry is the language-neutral correlation table. Existing SymPy names
# remain intact; AST spellings simply join them at the same Handler.
ssa_name_map: Dict[str, Handler] = {
    **sympy_ssa_name_map,
    **ast_ssa_name_map,
}


# -----------------------------------------------------------------------------
# Placeholder for developer-resolved disambiguation
# -----------------------------------------------------------------------------
sympy_ssa_disambig: Dict[str, Dict[str, Any]] = {
    # e.g. 'bitwise_and': {'bitwidth': 32, 'signed': False},
}


class SSARegistry:
    """
    Holds the authoritative SSA name map, disambiguation parameters,
    and helper-function registry for SSA emission.
    """
    name_map: Dict[str, Handler] = ssa_name_map
    disambig_map: Dict[str, Dict[str, Any]] = sympy_ssa_disambig
    ssa_helpers: Dict[Handler, Callable[..., Any]] = {}

    @classmethod
    def detect_ambiguous(cls) -> Dict[Handler, List[str]]:
        """Return handlers mapped from multiple SymPy node names."""
        rev = defaultdict(list)
        for sym, h in cls.name_map.items():
            rev[h].append(sym)
        return {h: syms for h, syms in rev.items() if len(syms) > 1}

    @classmethod
    def interactive_disambiguate(cls):
        """
        Prompt developer to disambiguate multi-mapped SymPy nodes by
        collecting extra parameters (e.g., bitwidth, signed).
        """
        amb = cls.detect_ambiguous()
        print("Ambiguous Handler mappings detected:")
        for handler, syms in amb.items():
            print(f"\nHandler {handler}:")
            for sym in syms:
                print(f"  SymPy node `{sym}`:")
                params: Dict[str, Any] = {}
                bw = input("    bitwidth (e.g. 1,8,32)? ")
                sd = input("    signed? (y/n) ")
                params['bitwidth'] = int(bw)
                params['signed']   = (sd.lower() == 'y')
                cls.disambig_map[sym] = params

    @classmethod
    def generate_disambig_code(cls) -> str:
        """
        Emit a Python code snippet for the `sympy_ssa_disambig` dict.
        """
        lines = ["sympy_ssa_disambig = {" ]
        for sym, args in cls.disambig_map.items():
            lines.append(f"    {sym!r}: {args},")
        lines.append("}")
        return "\n".join(lines)

    @classmethod
    def register_helper(cls, handler: Handler):
        """Decorator to register an SSA-emission helper for a Handler."""
        def decorator(fn: Callable[..., Any]):
            cls.ssa_helpers[handler] = fn
            return fn
        return decorator

    @classmethod
    def emit_ssa(cls, handler: Handler, builder, operands: List[Any], **kwargs) -> Any:
        """
        Dispatch to the registered helper for `handler`.
        Raises KeyError if none is found.
        """
        fn = cls.ssa_helpers.get(handler)
        if not fn:
            raise KeyError(f"No SSA helper registered for {handler}")
        return fn(builder, operands, **kwargs)
