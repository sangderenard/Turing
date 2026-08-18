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
    # Fixed-width signed multiplication primitives. ``SMulLow`` returns the
    # low ``width`` bits; ``SMulOverflow`` reports whether the mathematical
    # signed product is not the sign extension of those retained bits.  They
    # keep machine legalization exact without presuming host int128 support.
    SMulLow       = "SMulLow"
    SMulOverflow  = "SMulOverflow"
    SMulHigh      = "SMulHigh"
    # Exact halves of an unsigned fixed-width product.  Keeping the high half
    # explicit is required by accumulator-form MUL and avoids assuming that a
    # destination backend has a native integer twice the operand width.
    UMulLow       = "UMulLow"
    UMulHigh      = "UMulHigh"
    # The AMD64 DIV/IDIV dividend is the concatenated high:low register pair.
    # The guard is an ordered, may-trap dependency which checks zero divisors
    # and quotient range before the totalized quotient/remainder projections.
    WideDivCheck     = "WideDivCheck"
    WideDivQuotient  = "WideDivQuotient"
    WideDivRemainder = "WideDivRemainder"
    # Zero-totalized most-significant-set-bit index. It returns zero for a
    # zero input so callers can explicitly select their architecture's
    # undefined/preserved destination behavior without an eager trap.
    MsbIndex      = "MsbIndex"
    # Interleave the low half of two 128-bit bit patterns at an explicit lane
    # width. This is a machine-neutral vector permutation, not numeric fusion.
    VectorUnpackLow = "VectorUnpackLow"
    # Independent modular addition of fixed-width lanes in a vector bit
    # pattern. Carries never cross lane boundaries.
    VectorAddModulo = "VectorAddModulo"
    VectorSubtractModulo = "VectorSubtractModulo"
    VectorCompareEqualMask = "VectorCompareEqualMask"
    VectorShuffle = "VectorShuffle"
    # IEEE-754 binary64 operations over encoded 64-bit lanes. These do not
    # consult the host process floating environment.
    Float64IsNaNBits = "Float64IsNaNBits"
    Float64IsSignalingNaNBits = "Float64IsSignalingNaNBits"
    Float64BitsLt = "Float64BitsLt"
    Float64BitsGt = "Float64BitsGt"
    Float64BitsEq = "Float64BitsEq"
    Float32IsNaNBits = "Float32IsNaNBits"
    Float32IsSignalingNaNBits = "Float32IsSignalingNaNBits"
    Float32BitsLt = "Float32BitsLt"
    Float32BitsEq = "Float32BitsEq"
    # Set MXCSR invalid status when requested; trap when its invalid mask is
    # clear. The returned value is the post-operation MXCSR state.
    MXCSRInvalid = "MXCSRInvalid"
    SInt64ToFloat64Bits = "SInt64ToFloat64Bits"
    SInt64ToFloat32Bits = "SInt64ToFloat32Bits"
    MXCSRPrecision = "MXCSRPrecision"
    # Exact IEEE binary64 addition under the supplied MXCSR rounding/DAZ/FTZ
    # state, plus its ordered status/trap transition. These operate on encoded
    # bits and do not consult the compiler host's floating environment.
    Float64AddBits = "Float64AddBits"
    MXCSRFloat64Add = "MXCSRFloat64Add"
    Float64MultiplyBits = "Float64MultiplyBits"
    MXCSRFloat64Multiply = "MXCSRFloat64Multiply"
    Float32AddBits = "Float32AddBits"
    MXCSRFloat32Add = "MXCSRFloat32Add"
    Float32DivideBits = "Float32DivideBits"
    MXCSRFloat32Divide = "MXCSRFloat32Divide"
    Float64DivideBits = "Float64DivideBits"
    MXCSRFloat64Divide = "MXCSRFloat64Divide"
    Float64SubtractBits = "Float64SubtractBits"
    MXCSRFloat64Subtract = "MXCSRFloat64Subtract"
    Float64ToSInt64TruncBits = "Float64ToSInt64TruncBits"
    MXCSRFloat64ToSIntInvalid = "MXCSRFloat64ToSIntInvalid"
    Float64ToSInt32TruncBits = "Float64ToSInt32TruncBits"
    VectorSInt32ToFloat64Bits = "VectorSInt32ToFloat64Bits"
    MXCSRVectorSInt32ToFloat64 = "MXCSRVectorSInt32ToFloat64"
    ByteSwap = "ByteSwap"
    AtomicCompareExchangeObserved = "AtomicCompareExchangeObserved"
    AtomicCompareExchangeSuccess = "AtomicCompareExchangeSuccess"
    AtomicCompareExchangeMemory = "AtomicCompareExchangeMemory"
    AtomicExchangeAddObserved = "AtomicExchangeAddObserved"
    AtomicExchangeAddMemory = "AtomicExchangeAddMemory"
    Div           = "Div"
    FloorDiv      = "FloorDiv"
    Mod           = "Mod"
    Pow           = "Pow"
    MatMul        = "MatMul"
    Neg           = "Neg"
    Abs           = "Abs"

    # Bitwise
    And           = "And"
    Or            = "Or"
    Xor           = "Xor"
    Not           = "Not"
    Shl           = "Shl"
    Shr           = "Shr"
    AShr          = "AShr"

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
    # Integer comparisons over fixed-width bit patterns.  These are distinct
    # from the language-level signed/real comparisons above; AMD64 carry and
    # LLVM icmp unsigned predicates depend on the distinction.
    ULt           = "ULt"
    ULe           = "ULe"
    UGt           = "UGt"
    UGe           = "UGe"

    # Memory & Indexing
    Load          = "Load"
    Store         = "Store"
    Alloca        = "Alloca"
    Fill          = "Fill"          # span-memory initialisation; zero-fill == calloc
    # Store one fixed-width scalar ``count`` times starting at an address with
    # an explicit signed byte stride.  This models string-store instructions
    # without hiding iteration, direction, or memory state in a host callback.
    StridedStoreFill = "StridedStoreFill"
    # Sequential fixed-width copies over versioned memory.  Source and
    # destination addresses advance by the same explicit signed byte stride;
    # iteration order is observable when the regions overlap.
    StridedMemoryCopy = "StridedMemoryCopy"
    # A source-level deep copy. Not Python object-graph duplication (there is
    # no Python object graph at this stage) and not a single flat
    # StridedMemoryCopy either -- a record field that is itself a reference
    # points at separate storage that a flat byte-range copy would leave
    # shared between original and copy, exactly the shallow-copy bug a deep
    # copy exists to avoid. Lowering must walk the value's own record/field
    # descriptor (storage kind: scalar/span/record/reference/keyed, already
    # tracked elsewhere in this compiler) and recurse into every
    # record/reference field's own storage, copying each independently.
    Deepcopy      = "Deepcopy"
    GetElementPtr = "GetElementPtr"
    # Structured source operations retained until record/index legalization.
    # Indexed forms are rewritten to GetElementPtr+Load/Store before target
    # emission; GetAttr is resolved against record/class descriptors.
    GetAttr       = "GetAttr"
    Indexed       = "Indexed"
    IndexedStore  = "IndexedStore"

    # Casts & Conversions
    Cast          = "Cast"
    # Convert the first operand to the schema/dtype represented by the second.
    # The second edge is compile-time type evidence, not discarded Python
    # control flow around isinstance/type/constructor calls.
    CastLike      = "CastLike"
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
    # Computed control transfer with an explicit target and complete carried
    # state. Backends must implement or reject it; it is never converted into
    # a host-language callback.
    IndirectBr    = "IndirectBr"
    # Explicit non-returning architectural trap.  The vector and provenance
    # are data on the instruction; this is not a host exception or runtime
    # callback.
    Trap          = "Trap"
    Ret           = "Ret"
    Call          = "Call"
    Deploy        = "Deploy"
    Join          = "Join"

    # Misc
    Select        = "Select"
    Const         = "Const"  # literal constants
    # A fixed-width identity for a program object. Unlike ``ptr`` this is not
    # a dereferenceable repository address; it may name host-resident state.
    StaticRef     = "StaticRef"

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
    'deploy':              Handler.Deploy,
    'join':                Handler.Join,

    # Arithmetic
    'add':                 Handler.Add,
    'sub':                 Handler.Sub,
    'mul':                 Handler.Mul,
    'div':                 Handler.Div,
    'mod':                 Handler.Mod,
    'pow':                 Handler.Pow,
    'matmul':              Handler.MatMul,
    'matmult':             Handler.MatMul,
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
    # Span-memory initialisation collapses the construction constructors onto a
    # single Fill operation. Zero-fill (``zeros``/``empty``) is the calloc case.
    'fill':                Handler.Fill,
    'zeros':               Handler.Fill,
    'zeros_like':          Handler.Fill,
    'ones':                Handler.Fill,
    'ones_like':           Handler.Fill,
    'full':                Handler.Fill,
    'full_like':           Handler.Fill,
    'empty':               Handler.Fill,
    'empty_like':          Handler.Fill,
    'getelementptr':       Handler.GetElementPtr,
    'idx':                 Handler.GetElementPtr,
    'indexed':             Handler.Load,
    'indexedbase':         Handler.Alloca,
    'matrixelement':       Handler.Load,
    'matrixsymbol':        Handler.Alloca,

    # Casts & Conversions
    'cast':                Handler.Cast,
    'cast_like':           Handler.CastLike,
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
        'shl',
        'lshift',
        'binop:lshift',
        'augassign:lshift',
        'operator:lshift',
        '__lshift__',
        '__rlshift__',
        '__ilshift__',
    ),
    Handler.Shr: (
        'shr',
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

    # Span-memory initialisation. Tensor construction constructors collapse onto
    # a single Fill operation whose zero-fill spelling is the calloc case.
    Handler.Fill: (
        'call:zeros',
        'call:zeros_like',
        'call:ones',
        'call:ones_like',
        'call:full',
        'call:full_like',
        'call:empty',
        'call:empty_like',
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
    Handler.MatMul: (
        'matmult',
        'binop:matmult',
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


# -----------------------------------------------------------------------------
# C (pycparser ``c_ast``) → the same SSA / BitOps language
# -----------------------------------------------------------------------------
#
# Same equivalence-table contract as ``ast_ssa_equivalents`` above, for the node
# type names ``C_ROLE_SCHEMAS`` (oop_language_translations.py) registers. Keys
# are the lowercase spelling of the ``pycparser.c_ast`` class name, with the
# same ``qualified:piece`` convention for compound nodes.
#
# One difference from Python is worth stating rather than leaving to be
# rediscovered: Python spells its operators as *node classes* (``ast.Add``, so
# ``binop:add``), while pycparser spells them as *strings* on the parent node
# (``BinaryOp.op == '+'``, so ``binaryop:+``). Both are just surface spellings
# converging on one Handler -- which is the entire point of this table, and why
# C needs no new Handler of its own. Where a C spelling collides with a Python
# one (``constant``, ``return``, ``if``, ``for``, ``while``, ``break``,
# ``continue``) it deliberately resolves to the *same* Handler: those are a
# shared vocabulary both languages genuinely share, not a collision to route
# around.
c_ssa_equivalents: Dict[Handler, tuple[str, ...]] = {
    # Values and C literal spellings.
    Handler.Load: (
        'id',
        'structref',
        # Pointer dereference reads the pointee, the same reading Python's
        # 'attribute:load' has: a Load.
        'unaryop:*',
    ),
    Handler.Const: (
        'constant',
    ),
    Handler.Alloca: (
        'decl',
        'initlist',
        'struct',
        'funcdef',
    ),

    # Arithmetic. C's compound assignments carry the same operator meaning as
    # Python's AugAssign spellings already registered above.
    Handler.Add: (
        'binaryop:+',
        'assignment:+=',
    ),
    Handler.Sub: (
        'binaryop:-',
        'assignment:-=',
    ),
    Handler.Mul: (
        'binaryop:*',
        'assignment:*=',
    ),
    Handler.Div: (
        'binaryop:/',
        'assignment:/=',
    ),
    Handler.Mod: (
        'binaryop:%',
        'assignment:%=',
    ),
    Handler.Neg: (
        'unaryop:-',
    ),

    # Bitwise.
    Handler.And: (
        'binaryop:&',
        'assignment:&=',
    ),
    Handler.Or: (
        'binaryop:|',
        'assignment:|=',
    ),
    Handler.Xor: (
        'binaryop:^',
        'assignment:^=',
    ),
    Handler.Not: (
        'unaryop:~',
    ),
    Handler.Shl: (
        'binaryop:<<',
        'assignment:<<=',
    ),
    Handler.Shr: (
        'binaryop:>>',
        'assignment:>>=',
    ),

    # Logical.
    Handler.LAnd: (
        'binaryop:&&',
    ),
    Handler.LOr: (
        'binaryop:||',
    ),
    Handler.LNot: (
        'unaryop:!',
    ),

    # Comparison.
    Handler.Eq: (
        'binaryop:==',
    ),
    Handler.Ne: (
        'binaryop:!=',
    ),
    Handler.Lt: (
        'binaryop:<',
    ),
    Handler.Le: (
        'binaryop:<=',
    ),
    Handler.Gt: (
        'binaryop:>',
    ),
    Handler.Ge: (
        'binaryop:>=',
    ),

    # Storage and addressing.
    Handler.Store: (
        'assignment',
        'assignment:=',
    ),
    Handler.GetElementPtr: (
        'arrayref',
        # Address-of. C's only way to spell "a reference to this object",
        # which the cpp shell's method desugaring emits for every receiver
        # (``obj.m(x)`` -> ``Class_m(&obj, x)``).
        'unaryop:&',
    ),

    # Conversions. C's unary plus is a no-op conversion, the same reading
    # Python's 'uadd' already has above.
    Handler.Cast: (
        'cast',
        'unaryop:+',
    ),

    # Control and calls.
    Handler.CondBr: (
        'if',
        'switch',
    ),
    Handler.Br: (
        'for',
        'while',
        'dowhile',
        'break',
        'continue',
        'goto',
    ),
    Handler.Ret: (
        'return',
    ),
    Handler.Call: (
        'funccall',
    ),
    Handler.Select: (
        'ternaryop',
    ),
}


c_ssa_name_map: Dict[str, Handler] = {
    spelling: handler
    for handler, spellings in c_ssa_equivalents.items()
    for spelling in spellings
}


# The registry is the language-neutral correlation table. Existing SymPy names
# remain intact; AST spellings simply join them at the same Handler, and C
# spellings join both.
ssa_name_map: Dict[str, Handler] = {
    **sympy_ssa_name_map,
    **ast_ssa_name_map,
    **c_ssa_name_map,
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
