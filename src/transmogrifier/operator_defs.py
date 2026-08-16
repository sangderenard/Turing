import builtins
import math
import operator

import numpy as np
import sympy

from ..common.tensors.fused_ir import (
    ELEMENTWISE_BINARY,
    ELEMENTWISE_UNARY,
)
from ..common.tensors.operator_catalog import (
    ACCESSOR_OPERATORS,
    CANONICAL_ABSTRACT_TENSOR_OPERATORS,
    CREATION_OPERATORS,
    OPERATOR_ALIASES,
)

try:  # optional heavy dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dep
    torch = None  # type: ignore

SIMD_DEFAULT_CONCURRENCY = 4
numpy_funcs, torch_funcs, numpy_sigs, torch_sigs = {}, {}, {}, {}
abstract_tensor_funcs, abstract_tensor_sigs = {}, {}
# -------------------------------------------------
#  Anonymous signature definitions (shared objects)
# -------------------------------------------------
sig_binary_elementwise = {
    'min_inputs': 1, 'max_inputs': 2,
    'min_outputs': 1, 'max_outputs': 1,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True
}

sig_unary_elementwise = {
    'min_inputs': 1, 'max_inputs': 1,
    'min_outputs': 1, 'max_outputs': 1,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True
}

sig_sum_like = {
    'min_inputs': 1, 'max_inputs': None,
    'min_outputs': 1, 'max_outputs': 1,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True,
    'parameters': ['limits']
}

sig_idx_like = {
    'min_inputs': 1, 'max_inputs': None,
    'min_outputs': 1, 'max_outputs': 1,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True,
    'parameters': ['range']
}

sig_indexed = {
    'min_inputs': 1, 'max_inputs': None,
    'min_outputs': 1, 'max_outputs': None,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True
}

sig_indexed_base = {
    'min_inputs': 2, 'max_inputs': 2,
    'min_outputs': 1, 'max_outputs': 1,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True
}

sig_store = {
    'min_inputs': 1, 'max_inputs': 1,
    'min_outputs': 0, 'max_outputs': 0,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True
}

sig_default = {
    'min_inputs': 1, 'max_inputs': 1,
    'min_outputs': 0, 'max_outputs': 0,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True
}
sig_equality = {
        'min_inputs': 2,
        'max_inputs': 2,
        'min_outputs': 1,
        'max_outputs': 1,
        'concurrency': SIMD_DEFAULT_CONCURRENCY,
        'allows_inplace': True
    }
sig_constant = {
    'min_inputs': 0, 'max_inputs': 0,
    'min_outputs': 1, 'max_outputs': 1,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True
}
array_sig = {
    'min_inputs': None, 'max_inputs': None,
    'min_outputs': None, 'max_outputs': None,
    'concurrency': None,
    'allows_inplace': True
}
# -------------------------------------------------
# Operation name -> signature mapping
# -------------------------------------------------
operator_signatures = {
    # Structural concurrency framing. Deploy and Join do not perform numeric
    # work; a Join reduction names an existing operator in its metadata.
    'Deploy': {
        'min_inputs': 0, 'max_inputs': None,
        'min_outputs': 0, 'max_outputs': None,
        'concurrency': None, 'allows_inplace': False,
    },
    'Join': {
        'min_inputs': 0, 'max_inputs': None,
        'min_outputs': 0, 'max_outputs': None,
        'concurrency': None, 'allows_inplace': False,
    },
    'Add': sig_binary_elementwise,
    'Mul': sig_binary_elementwise,
    'Pow': sig_binary_elementwise,
    'Rational': sig_binary_elementwise,

    'Sum': sig_sum_like,
    'Idx': sig_idx_like,
    'Indexed': sig_indexed,
    'IndexedBase': sig_indexed_base,
    'Tuple': sig_unary_elementwise,
    'Store': sig_store,
    'Default': sig_default,

    # Trigonometric, log, exp, sqrt etc
    'Sin': sig_unary_elementwise,
    'Cos': sig_unary_elementwise,
    'Tan': sig_unary_elementwise,
    'Exp': sig_unary_elementwise,
    'Log': sig_unary_elementwise,
    'Sqrt': sig_unary_elementwise,

    'Equality': sig_equality,

    'Pi': sig_constant,
    'Half': sig_constant,
    'ImaginaryUnit': sig_constant,
    'E': sig_constant,
    'StrictGreaterThan': sig_binary_elementwise,
}

array_sigs_overrides = {
    #stack onto default for tensor-like operations
    'Add': array_sig,
    'Mul': array_sig,
    'Pow': array_sig,
    'Sub': array_sig,
    'Div': array_sig,
    'Mod': array_sig,
    'And': array_sig,
    'Or': array_sig,
    'Not': array_sig,
    'exp': array_sig,
    'Sin': array_sig,
    'Cos': array_sig,
    'Tan': array_sig,
    'Exp': array_sig,
    'Log': array_sig,
    'Sqrt': array_sig,
    'ceiling': array_sig,
    'floor': array_sig,
    'round': array_sig,
    'abs': array_sig,
    'Min': array_sig,
    'Max': array_sig,
    'Abs': array_sig,
    'Tuple': sig_unary_elementwise,
    'Rational': sig_binary_elementwise,

}
# --- signatures -------------------------------------------------------------
sig_matrixsymbol = {
    'min_inputs': 0,       # leaf-node: takes nothing
    'max_inputs': 0,
    'min_outputs': 1,      # produces one value
    'max_outputs': 1,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True
}

operator_signatures['MatrixSymbol'] = sig_matrixsymbol

# -------------------------------------------------
# Operator function mappings (default execution impls)
# -------------------------------------------------
def add_op(role_map):
    return sum(vals[0] for vals in role_map.values())

def mul_op(role_map):
    iter_vals = iter(role_map.values())
    result = next(iter_vals)[0]
    for vals in iter_vals:
        result *= vals[0]
    return result

def pow_op(role_map):
    base = role_map.get('arg0', [None])[0]
    exp = role_map.get('arg1', [1])[0]
    return np.power(base, exp)

def indexed_op(*role_map):
    if isinstance(role_map, dict):
        base = role_map.get('base', [[]])[0]
        indices = tuple(role_map.get('index', []))
    else:
        indices = (*role_map,)
        print(f"Warning: indexed_op called with non-dict input: {role_map}")


    if not indices:
        raise ValueError("No indices provided for Indexed operation.")
    
    if isinstance(indices, tuple) and len(indices) == 1:
        indices = indices[0]
    elif isinstance(indices, tuple):
        ndim_desired = len(indices)
        ndim_base = len(base.shape) if isinstance(base, np.ndarray) else 1
        if ndim_desired > ndim_base and isinstance(base, np.ndarray):
            base = base.reshape((1,) * (ndim_desired - ndim_base) + base.shape)
        if ndim_desired > ndim_base and isinstance(base, list):
            for i in enumerate(indices):
                base = [base]
    indices = slice(*indices) if isinstance(indices, tuple) else indices
    return base[indices]

def indexedbase_op(role_map):
    print(f"Role map for IndexedBase operation: {role_map}")
    if isinstance(role_map, dict):
        base = role_map.get('base', [[]])[0]
        return base
    elif isinstance(role_map, float):
        return role_map
    elif isinstance(role_map, np.ndarray):
        return role_map
    elif isinstance(role_map, list):
        return np.array(role_map)
    elif isinstance(role_map, torch.Tensor):
        return role_map
    elif isinstance(role_map, int):
        return role_map
    return role_map.get('base', [[]])[0]

def sum_op(role_map):
    return sum(role_map.get('body', [0])[0])

# Scientific / trig functions
def sin_op(role_map):
    return np.sin(role_map.get('arg0', [0])[0])

def cos_op(role_map):
    return np.cos(role_map.get('arg0', [0])[0])

def tan_op(role_map):
    return np.tan(role_map.get('arg0', [0])[0])

def exp_op(role_map):
    return np.exp(role_map.get('arg0', [0])[0])

def log_op(role_map):
    return np.log(role_map.get('arg0', [0])[0])

def sqrt_op(role_map):
    return np.sqrt(role_map.get('arg0', [0])[0])

def store_op(role_map):
    value = role_map.get('value', [None])[0]
    #print(f"Store operation completed. Produced value: {value}")
    return value

# -------------------------------------------------
# Complete operator function dispatch
# -------------------------------------------------
default_funcs = {
    'Add': add_op,
    'Mul': mul_op,
    'Pow': pow_op,
    'Indexed': indexed_op,
    'IndexedBase': indexedbase_op,
    'Sum': sum_op,

    'Sin': sin_op,
    'Cos': cos_op,
    'Tan': tan_op,
    'Exp': exp_op,
    'Log': log_op,
    'Sqrt': sqrt_op,
    'Store': store_op,
}
# --- execution impls --------------------------------------------------------
def matrixsymbol_op(role_map):
    """
    If the builder wired a concrete value, return it.
    Otherwise fall back to an all-zeros array shaped like the declared symbol,
    or a scalar 0.0 when even the shape is missing.
    """
    if 'value' in role_map:                 # explicit literal
        return role_map['value'][0]

    if 'shape' in role_map:                 # symbolic shape (m, n)
        m, n = role_map['shape'][0]
        return np.zeros((m, n), dtype=float)

    # last-ditch: give the rest of the pipeline *something* numeric
    return 0.0

default_funcs['MatrixSymbol'] = matrixsymbol_op
# ── 1. signature  ────────────────────────────────────────────────────────────
sig_matrix_element = {
    'min_inputs'   : 1,          # needs at least the matrix itself
    'max_inputs'   : None,       # row / col edges count too
    'min_outputs'  : 1,          # returns a scalar
    'max_outputs'  : 1,
    'concurrency'  : SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True,
}
operator_signatures['MatrixElement'] = sig_matrix_element


# ── 2. handler  ──────────────────────────────────────────────────────────────
def matrixelement_op(role_map):
    """
    Extract A[i,j] from a NumPy/SymPy matrix.

    Fallback rules keep the pipeline alive if something is missing:
        • missing matrix   → 0.0
        • missing row/col  → returns the whole matrix
        • out-of-bounds    → 0.0
    """
    mat = role_map.get('matrix', role_map.get('base', [None]))[0]
    if mat is None:
        return 0.0                               # nothing to read from

    # support either explicit 'row'/'col' or a single 'index' edge [(i,j)]
    if 'index' in role_map:
        row, col = role_map['index'][0]
    else:
        row = role_map.get('row', [None])[0]
        col = role_map.get('col', [None])[0]

    if row is None or col is None:
        return mat                               # treat as “give me the row/col slice”

    try:
        return mat[row, col]                     # NumPy or SymPy matrices
    except Exception:
        return 0.0                               # soft-fail on bad indices

default_funcs['MatrixElement'] = matrixelement_op
# --- role schemas -----------------------------------------------------------
role_schemas = {
            'Deploy': {'up': {'domain': 'many'}, 'down': {'lanes': 'many'}},
            'Join': {'up': {'lanes': 'many'}, 'down': {'result': 'many'}},
            'IndexedBase': {'up':{'shape':1}, 'down':{}},
            'Indexed': {'up':{'base':1, 'index':'many'},'down':{}},
            'Idx': {'up':{'limits': 'many'}, 'down':{}},
            'Sum': {'up':{'body': 1, 'limits': 'many'}, 'down':{}},
            #'Piecewise': {'up':   {'exprs': 'many', 'conds': 'many'},'down': {}},
            # etc - you can expand this for functions, FFTs, etc.
        }
role_schemas.update({
            'MatrixSymbol': {
                'up'  : {'value': 1,        # optional literal
                        'shape': 1},       # optional (rows, cols) tuple
                'down': {}
            },
        })

role_schemas.update({
    'Module':      {'up': {'body': 'many'}, 'down': {}},
    'FunctionDef': {'up': {'args': 1, 'body': 'many', 'decorator_list': 'many'}, 'down': {}},
    'Assign':      {'up': {'targets': 'many', 'value': 1}, 'down': {}},
    'If':          {'up': {'test': 1, 'body': 'many', 'orelse': 'many'}, 'down': {}},
    'Return':      {'up': {'value': 1}, 'down': {}},
    'Expr':        {'up': {'value': 1}, 'down': {}},
    'Call':        {'up': {'func': 1, 'args': 'many', 'keywords': 'many'}, 'down': {}},
    'BinOp':       {'up': {'left': 1, 'op': 1, 'right': 1}, 'down': {}},
    'Name':        {'up': {}, 'down': {}},
    'Constant':    {'up': {}, 'down': {}},
    'arguments':   {'up': {'args': 'many', 'vararg': 1, 'kwonlyargs': 'many', 'kw_defaults': 'many', 'kwarg': 1, 'defaults': 'many'}, 'down': {}},
    # Expand as needed for more node types...
})

role_schemas.update({
    'Module':      {'up': {'body': 'many'}, 'down': {}},
    'FunctionDef': {'up': {'name': 1, 'args': 1, 'body': 'many', 'decorator_list': 'many', 'returns': 1, 'type_comment': 1}, 'down': {}},
    'AsyncFunctionDef': {'up': {'name': 1, 'args': 1, 'body': 'many', 'decorator_list': 'many', 'returns': 1, 'type_comment': 1}, 'down': {}},
    'ClassDef':    {'up': {'name': 1, 'bases': 'many', 'keywords': 'many', 'body': 'many', 'decorator_list': 'many'}, 'down': {}},
    'Return':      {'up': {'value': 1}, 'down': {}},
    'Delete':      {'up': {'targets': 'many'}, 'down': {}},
    'Assign':      {'up': {'targets': 'many', 'value': 1, 'type_comment': 1}, 'down': {}},
    'AugAssign':   {'up': {'target': 1, 'op': 1, 'value': 1}, 'down': {}},
    'AnnAssign':   {'up': {'target': 1, 'annotation': 1, 'value': 1, 'simple': 1}, 'down': {}},
    'For':         {'up': {'target': 1, 'iter': 1, 'body': 'many', 'orelse': 'many', 'type_comment': 1}, 'down': {}},
    'AsyncFor':    {'up': {'target': 1, 'iter': 1, 'body': 'many', 'orelse': 'many', 'type_comment': 1}, 'down': {}},
    'While':       {'up': {'test': 1, 'body': 'many', 'orelse': 'many'}, 'down': {}},
    'If':          {'up': {'test': 1, 'body': 'many', 'orelse': 'many'}, 'down': {}},
    'With':        {'up': {'items': 'many', 'body': 'many', 'type_comment': 1}, 'down': {}},
    'AsyncWith':   {'up': {'items': 'many', 'body': 'many', 'type_comment': 1}, 'down': {}},
    'Raise':       {'up': {'exc': 1, 'cause': 1}, 'down': {}},
    'Try':         {'up': {'body': 'many', 'handlers': 'many', 'orelse': 'many', 'finalbody': 'many'}, 'down': {}},
    'ExceptHandler': {'up': {'type': 1, 'name': 1, 'body': 'many'}, 'down': {}},
    'Assert':      {'up': {'test': 1, 'msg': 1}, 'down': {}},
    
    'Import':      {'up': {'names': 'many'}, 'down': {}},
    'ImportFrom':  {'up': {'module': 1, 'names': 'many', 'level': 1}, 'down': {}},
    'Global':      {'up': {'names': 'many'}, 'down': {}},
    'Nonlocal':    {'up': {'names': 'many'}, 'down': {}},
    'Expr':        {'up': {'value': 1}, 'down': {}},
    'Pass':        {'up': {}, 'down': {}},
    'Break':       {'up': {}, 'down': {}},
    'Continue':    {'up': {}, 'down': {}},

    # Expressions
    'BoolOp':      {'up': {'op': 1, 'values': 'many'}, 'down': {}},
    'BinOp':       {'up': {'left': 1, 'op': 1, 'right': 1}, 'down': {}},
    'UnaryOp':     {'up': {'op': 1, 'operand': 1}, 'down': {}},
    'Lambda':      {'up': {'args': 1, 'body': 1}, 'down': {}},
    'IfExp':       {'up': {'test': 1, 'body': 1, 'orelse': 1}, 'down': {}},
    'Dict':        {'up': {'keys': 'many', 'values': 'many'}, 'down': {}},
    'Set':         {'up': {'elts': 'many'}, 'down': {}},
    'ListComp':    {'up': {'elt': 1, 'generators': 'many'}, 'down': {}},
    'SetComp':     {'up': {'elt': 1, 'generators': 'many'}, 'down': {}},
    'DictComp':    {'up': {'key': 1, 'value': 1, 'generators': 'many'}, 'down': {}},
    'GeneratorExp':{'up': {'elt': 1, 'generators': 'many'}, 'down': {}},
    'Await':       {'up': {'value': 1}, 'down': {}},
    'Yield':       {'up': {'value': 1}, 'down': {}},
    'YieldFrom':   {'up': {'value': 1}, 'down': {}},
    'Compare':     {'up': {'left': 1, 'ops': 'many', 'comparators': 'many'}, 'down': {}},
    'Call':        {'up': {'func': 1, 'args': 'many', 'keywords': 'many'}, 'down': {}},
    'FormattedValue': {'up': {'value': 1, 'format_spec': 1}, 'down': {}},
    'JoinedStr':   {'up': {'values': 'many'}, 'down': {}},
    'Constant':    {'up': {}, 'down': {}},
    'Attribute':   {'up': {'value': 1, 'attr': 1}, 'down': {}},
    'Subscript':   {'up': {'value': 1, 'slice': 1}, 'down': {}},
    'Starred':     {'up': {'value': 1}, 'down': {}},
    'Name':        {'up': {}, 'down': {}},
    'List':        {'up': {'elts': 'many'}, 'down': {}},
    'Tuple':       {'up': {'elts': 'many'}, 'down': {}},

    # Arguments and comprehensions
    'arguments':   {'up': {
        'posonlyargs': 'many',
        'args': 'many',
        'vararg': 1,
        'kwonlyargs': 'many',
        'kw_defaults': 'many',
        'kwarg': 1,
        'defaults': 'many'
    }, 'down': {}},
    'arg':         {'up': {'annotation': 1, 'type_comment': 1}, 'down': {}},
    'keyword':     {'up': {'arg': 1, 'value': 1}, 'down': {}},
    'comprehension': {'up': {'target': 1, 'iter': 1, 'ifs': 'many', 'is_async': 1}, 'down': {}},

    # Operators and other nodes
    'Add':         {'up': {}, 'down': {}},
    'Sub':         {'up': {}, 'down': {}},
    'Mult':        {'up': {}, 'down': {}},
    'Div':         {'up': {}, 'down': {}},
    'Mod':         {'up': {}, 'down': {}},
    'Pow':         {'up': {}, 'down': {}},
    'LShift':      {'up': {}, 'down': {}},
    'RShift':      {'up': {}, 'down': {}},
    'BitOr':       {'up': {}, 'down': {}},
    'BitXor':      {'up': {}, 'down': {}},
    'BitAnd':      {'up': {}, 'down': {}},
    'FloorDiv':    {'up': {}, 'down': {}},
    'MatMult':     {'up': {}, 'down': {}},
    'Invert':      {'up': {}, 'down': {}},
    'Not':         {'up': {}, 'down': {}},
    'UAdd':        {'up': {}, 'down': {}},
    'USub':        {'up': {}, 'down': {}},

    # Boolean/comparison/context leaves and structural helpers used by the
    # semantic Python front end. Arithmetic leaves reduce to the established
    # canonical operation names; these schemas only describe AST shape.
    'And':         {'up': {}, 'down': {}},
    'Or':          {'up': {}, 'down': {}},
    'Eq':          {'up': {}, 'down': {}},
    'NotEq':       {'up': {}, 'down': {}},
    'Lt':          {'up': {}, 'down': {}},
    'LtE':         {'up': {}, 'down': {}},
    'Gt':          {'up': {}, 'down': {}},
    'GtE':         {'up': {}, 'down': {}},
    'Is':          {'up': {}, 'down': {}},
    'IsNot':       {'up': {}, 'down': {}},
    'In':          {'up': {}, 'down': {}},
    'NotIn':       {'up': {}, 'down': {}},
    'Load':        {'up': {}, 'down': {}},
    'Store':       {'up': {}, 'down': {}},
    'Slice':       {'up': {'lower': 1, 'upper': 1, 'step': 1}, 'down': {}},
    'alias':       {'up': {}, 'down': {}},
    'withitem':    {'up': {'context_expr': 1, 'optional_vars': 1}, 'down': {}},

    # Optionally: cover all ast.AST leaf nodes as {}
})


# ── 3. role schema  ──────────────────────────────────────────────────────────
role_schemas.update({
    'MatrixElement': {
        'up'  : {
            'matrix': 1,       # the parent matrix

        },
        'down': {}
    },
})
# operator_defs.py  (or wherever you define the table)
operator_signatures['Equality'] = {
    'min_inputs'   : 2,      # lhs, rhs
    'max_inputs'   : 2,
    'min_outputs'  : 1,      # ← force a Store
    'max_outputs'  : 1,
    'parameters'   : [],     # nothing extra
}
numpy_funcs = default_funcs.copy()
numpy_funcs['Equality'] = lambda role_map: role_map['lhs'][0] == role_map['rhs'][0]

import math
ultra_basic_funcs = {
    'Equality': lambda x: x == x,
    'Store': lambda x: x,  # Store just returns its input
    'MatrixSymbol': matrixsymbol_op,  # from above
    'MatrixElement': matrixelement_op,  # from above
    'Add': lambda x: x[0] + x[1] if len(x) == 2 else sum(x),
    'Sum': lambda x: sum(x[0]) if len(x) == 1 else sum(x),
    'Idx': lambda x: x[0] if len(x) == 1 else x[0][x[1]],  # single index or tuple
    'Indexed': indexed_op,  # from above
    'IndexedBase': indexedbase_op,  # from above
    'Sub': lambda x: x[0] - x[1] if len(x) == 2 else x[0] - sum(x[1:]),
    'Div': lambda x: x[0] / x[1] if len(x) == 2 else x[0] / np.prod(x[1:]),
    'Mod': lambda x: x[0] % x[1] if len(x) == 2 else x[0] % np.prod(x[1:]),
    'And': lambda x: all(x),
    'Or': lambda x: any(x),
    'Not': lambda x: not x[0] if len(x) == 1 else not all(x),
    'Mul': lambda x: x[0] * x[1] if len(x) == 2 else [x[i] * x[i+1] for i in range(len(x)-1)],
    'Pow': lambda x: x[0] ** x[1] if len(x) == 2 else x[0] ** 2,
    'Rational': lambda x: x[0] / x[1] if len(x) == 2 else x[0] / np.prod(x[1:]),

}
math_funcs = {

    'Sin': lambda x: math.sin(x[0]) if len(x) == 1 else [math.sin(v) for v in x],
    'Cos': lambda x: math.cos(x[0]) if len(x) == 1 else [math.cos(v) for v in x],
    'Tan': lambda x: math.tan(x[0]) if len(x) == 1 else [math.tan(v) for v in x],
    'Exp': lambda x: math.exp(x[0]) if len(x) == 1 else [math.exp(v) for v in x],
    'Log': lambda x: math.log(x[0]) if len(x) == 1 else [math.log(v) for v in x],
    'Sqrt': lambda x: math.sqrt(x[0]) if len(x) == 1 else [math.sqrt(v) for v in x],
}
math_funcs.update(ultra_basic_funcs)
torch_funcs = ultra_basic_funcs.copy()
torch_funcs.update({
    'Equality': lambda x: torch.equal(x[0], x[1]),
    'Store': lambda x: x[0],  # Store just returns its input
    'MatrixSymbol': matrixsymbol_op,  # from above
    'MatrixElement': matrixelement_op,  # from above
    'Add': lambda x: torch.add(x[0], x[1]) if len(x) == 2 else torch.sum(torch.stack(x)),
    'Mul': lambda x: torch.mul(x[0], x[1]) if len(x) == 2 else torch.prod(torch.stack(x)),
    'Pow': lambda x: torch.pow(x[0], x[1]) if len(x) == 2 else torch.pow(x[0], 2),
    'Sin': lambda x: torch.sin(x[0]) if len(x) == 1 else torch.sin(torch.stack(x)),
    'Cos': lambda x: torch.cos(x[0]) if len(x) == 1 else torch.cos(torch.stack(x)),
    'Tan': lambda x: torch.tan(x[0]) if len(x) == 1 else torch.tan(torch.stack(x)),
    'Exp': lambda x: torch.exp(x[0]) if len(x) == 1 else torch.exp(torch.stack(x)),
    'exp': lambda x: torch.exp(x[0]) if len(x) == 1 else torch.exp(torch.stack(x)),
    'Log': lambda x: torch.log(x[0]) if len(x) == 1 else torch.log(torch.stack(x)),
    'Sqrt': lambda x: torch.sqrt(x[0]) if len(x) == 1 else torch.sqrt(torch.stack(x)),
    'ceiling': lambda x: torch.ceil(x[0]) if len(x) == 1 else torch.ceil(torch.stack(x)),
    'floor': lambda x: torch.floor(x[0]) if len(x) == 1 else torch.floor(torch.stack(x)),
    'round': lambda x: torch.round(x[0]) if len(x) == 1 else torch.round(torch.stack(x)),
    'abs': lambda x: torch.abs(x[0]) if len(x) == 1 else torch.abs(torch.stack(x)),
    'Abs': lambda x: torch.abs(x[0]) if len(x) == 1 else torch.abs(torch.stack(x)),
    'Min': lambda x: torch.min(x[0]) if len(x) == 1 else torch.min(torch.stack(x)),
    'Max': lambda x: torch.max(x[0]) if len(x) == 1 else torch.max(torch.stack(x)),
    'Tuple': lambda *x: tuple(x),  # simply return the tuple of inputs
    'StrictGreaterThan': lambda x, y: torch.gt(x, y) if len(x) == 2 else torch.gt(torch.stack(x[:-1]), x[-1]),
    'BooleanTrue': lambda: torch.tensor(True, dtype=torch.bool),
    'BooleanFalse': lambda: torch.tensor(False, dtype=torch.bool),
    'Half': lambda: torch.tensor(0.5, dtype=torch.float32),  # half precision
    'Float': lambda x: torch.tensor(x, dtype=torch.float32),  # default float type
    'Pi': lambda: torch.tensor(np.pi, dtype=torch.float32),  # π constant
    'E': lambda: torch.tensor(np.e, dtype=torch.float32),  # e
    'erf': lambda x: torch.erf(x[0]) if len(x) == 1 else torch.erf(torch.stack(x)),
    'ImaginaryUnit': lambda: torch.tensor(1j, dtype=torch.complex64),  # imaginary unit
    'IndexedBase': lambda *x: torch.tensor(x, dtype=torch.float32),
    'Indexed': lambda *x: x[0][x[1]],  # from above
})


# -------------------------------------------------
# AbstractTensor ProcessGraph execution table
# -------------------------------------------------
#
# This is the same ProcessGraph backend-table contract used by ``numpy_funcs``
# and ``torch_funcs`` above.  The functions deliberately operate through the
# public AbstractTensor surface instead of selecting a concrete implementation:
# the tensor operands retain their NumPy, Torch/CUDA, C, GLSL, or other
# registered backend all the way through graph execution.
def _abstract_tensor_values(*values):
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        return list(values[0])
    return list(values)


def _abstract_tensor_reduce(binary):
    def apply(*values):
        operands = _abstract_tensor_values(*values)
        if not operands:
            raise ValueError("AbstractTensor operation requires an operand")
        result = operands[0]
        for operand in operands[1:]:
            result = binary(result, operand)
        return result
    return apply


def _abstract_tensor_method(name):
    def apply(*values, **kwargs):
        operands = _abstract_tensor_values(*values)
        if not operands:
            raise ValueError(f"AbstractTensor.{name} requires an operand")
        if not hasattr(operands[0], name):
            # ``math`` covers the transcendentals, but a conversion like
            # ``float`` or ``int`` is a builtin and has no ``math`` entry, so
            # the lookup fell through to ``getattr(operands[0], name)`` and
            # raised there instead.
            scalar_function = getattr(math, name, None)
            if not callable(scalar_function):
                scalar_function = getattr(builtins, name, None)
            if callable(scalar_function):
                # A statically-referenced conversion arrives with the callable
                # itself as the leading operand -- ``float(i + 1)`` binds the
                # ``float`` type and then the value. That first entry is the
                # operator, not something to convert, so applying the function
                # to the whole list would pass it itself.
                if operands[0] is scalar_function:
                    return scalar_function(*operands[1:], **kwargs)
                return scalar_function(*operands, **kwargs)
        return getattr(operands[0], name)(*operands[1:], **kwargs)
    return apply


def _abstract_tensor_attribute(name):
    def apply(*values):
        operands = _abstract_tensor_values(*values)
        if len(operands) != 1:
            raise ValueError(
                f"AbstractTensor attribute {name} expects one operand"
            )
        return getattr(operands[0], name)
    return apply


def _abstract_tensor_primitive(name):
    def apply(*values):
        operands = _abstract_tensor_values(*values)
        if len(operands) != 1:
            raise ValueError(
                f"AbstractTensor primitive {name} expects one operand"
            )
        tensor = operands[0]
        if not hasattr(tensor, "_apply_operator"):
            scalar_operators = {
                "neg": operator.neg,
                "logical_not": operator.not_,
                "invert": operator.invert,
                "abs": operator.abs,
            }
            scalar_operator = scalar_operators.get(name)
            if scalar_operator is not None:
                return scalar_operator(tensor)
        return tensor._apply_operator(name, tensor, None)
    return apply


def _abstract_tensor_static(name):
    def apply(*values, **kwargs):
        from ..common.tensors.abstraction import AbstractTensor
        return getattr(AbstractTensor, name)(
            *values,
            **kwargs,
        )
    return apply


def _abstract_tensor_random_source(*values):
    from ..common.tensors.abstraction import AbstractTensor

    shape = tuple(int(value) for value in values) or (1,)
    return AbstractTensor.random_tensor(shape)


def _abstract_tensor_constant(value):
    def build(*_values):
        from ..common.tensors.abstraction import AbstractTensor
        return AbstractTensor.get_tensor(value)
    return build


def _abstract_tensor_tuple(*values):
    return tuple(_abstract_tensor_values(*values))


def _abstract_tensor_identity(*values):
    operands = _abstract_tensor_values(*values)
    if not operands:
        raise ValueError("AbstractTensor identity operation requires an operand")
    return operands[0]


def _abstract_tensor_index(*values):
    operands = _abstract_tensor_values(*values)
    if not operands:
        raise ValueError("AbstractTensor indexing requires a tensor operand")
    if len(operands) < 2:
        return operands[0]
    index = tuple(operands[1:])
    if len(index) == 1:
        index = index[0]
    return operands[0][index]


def _abstract_tensor_index_store(*values):
    operands = _abstract_tensor_values(*values)
    if len(operands) < 3:
        raise ValueError(
            "AbstractTensor indexed assignment requires tensor, index, and value"
        )
    tensor = operands[0]
    value = operands[-1]
    indices = tuple(operands[1:-1])
    index = indices[0] if len(indices) == 1 else indices
    if len(indices) == 1:
        index_storage = getattr(index, "data", index)
        index_dtype = getattr(index_storage, "dtype", getattr(index, "dtype", None))
        if str(index_dtype).casefold().endswith("bool"):
            from ..common.tensors.abstraction import AbstractTensor
            return AbstractTensor.where(index, value, tensor)
        if hasattr(index, "shape") and str(index_dtype).casefold().endswith(
            ("float", "float32", "float64", "double")
        ):
            raise TypeError(
                "indexed assignment received a floating tensor index from "
                f"ProcessGraph: shape={tuple(index.shape)!r}, "
                f"dtype={index_dtype!s}, tensor_shape={tuple(tensor.shape)!r}"
            )
    from ..common.tensors.abstraction import AbstractTensor
    finalize = AbstractTensor._pre_autograd(
        "index_set", [tensor, value], params={"idx": index}
    )
    with AbstractTensor.autograd.no_grad():
        result = tensor.clone()
        result[index] = value
    return finalize(result)


def _abstract_tensor_sum(*values, **kwargs):
    operands = _abstract_tensor_values(*values)
    if not operands:
        raise ValueError("AbstractTensor sum requires an operand")
    return operands[0].sum(*operands[1:], **kwargs)


def _abstract_tensor_maximum(*values):
    operands = _abstract_tensor_values(*values)
    if not operands:
        raise ValueError("AbstractTensor maximum requires an operand")
    tensor_index = next(
        (
            index
            for index, operand in enumerate(operands)
            if callable(getattr(operand, "maximum", None))
        ),
        None,
    )
    if tensor_index is None:
        return max(operands)
    result = operands.pop(tensor_index)
    for operand in operands:
        result = result.maximum(operand)
    return result


def _abstract_tensor_minimum(*values):
    operands = _abstract_tensor_values(*values)
    if not operands:
        raise ValueError("AbstractTensor minimum requires an operand")
    tensor_index = next(
        (
            index
            for index, operand in enumerate(operands)
            if callable(getattr(operand, "minimum", None))
        ),
        None,
    )
    if tensor_index is None:
        return min(operands)
    result = operands.pop(tensor_index)
    for operand in operands:
        result = result.minimum(operand)
    return result


def _abstract_tensor_stack(*values, dim=0, axis=None):
    from ..common.tensors.abstraction import AbstractTensor
    if axis is not None:
        dim = axis
    operands = _abstract_tensor_values(*values)
    if operands and isinstance(operands[-1], int):
        dim = operands.pop()
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = list(operands[0])
    return AbstractTensor.stack(operands, dim=dim)


def _abstract_tensor_cat(*values, dim=0, axis=None):
    from ..common.tensors.abstraction import AbstractTensor
    if axis is not None:
        dim = axis
    operands = _abstract_tensor_values(*values)
    if operands and isinstance(operands[-1], int):
        dim = operands.pop()
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = list(operands[0])
    return AbstractTensor.cat(operands, dim=dim)


def _abstract_tensor_where(*values):
    from ..common.tensors.abstraction import AbstractTensor
    operands = _abstract_tensor_values(*values)
    if len(operands) != 3:
        raise ValueError("AbstractTensor where expects condition, true, false")
    return AbstractTensor.where(operands[0], operands[1], operands[2])


def _abstract_tensor_topk(*values):
    from ..common.tensors.abstraction import AbstractTensor
    operands = _abstract_tensor_values(*values)
    if not operands:
        raise ValueError("AbstractTensor topk requires a tensor operand")
    return AbstractTensor.topk(*operands)


def _abstract_tensor_erf(*values):
    """Backend-neutral erf approximation composed from AbstractTensor ops."""
    operands = _abstract_tensor_values(*values)
    if len(operands) != 1:
        raise ValueError("AbstractTensor erf expects one operand")
    value = operands[0]
    magnitude = value.abs()
    t = 1.0 / (1.0 + 0.3275911 * magnitude)
    polynomial = (
        (
            (
                (
                    1.061405429 * t
                    - 1.453152027
                ) * t
                + 1.421413741
            ) * t
            - 0.284496736
        ) * t
        + 0.254829592
    ) * t
    return value.sign() * (1.0 - polynomial * (-(magnitude * magnitude)).exp())


_at_add = _abstract_tensor_reduce(lambda left, right: left + right)
_at_sub = _abstract_tensor_reduce(lambda left, right: left - right)
_at_mul = _abstract_tensor_reduce(lambda left, right: left * right)
_at_div = _abstract_tensor_reduce(lambda left, right: left / right)
_at_mod = _abstract_tensor_reduce(lambda left, right: left % right)
_at_pow = _abstract_tensor_reduce(lambda left, right: left ** right)
_at_matmul = _abstract_tensor_reduce(lambda left, right: left @ right)
_at_and = _abstract_tensor_reduce(lambda left, right: left & right)
_at_or = _abstract_tensor_reduce(lambda left, right: left | right)
_at_xor = _abstract_tensor_reduce(lambda left, right: left ^ right)
# ``<<``/``>>`` resolve their result type through AbstractTensor's own
# operators (``__lshift__``/``__rshift__``), exactly as the bitwise ops above
# do -- no forced integer cast here.
_at_shl = _abstract_tensor_reduce(lambda left, right: left << right)
_at_shr = _abstract_tensor_reduce(lambda left, right: left >> right)

abstract_tensor_funcs = {
    # SymPy/ProcessGraph spellings.
    "Add": _at_add,
    "Sub": _at_sub,
    "Mul": _at_mul,
    "Div": _at_div,
    "FloorDiv": _abstract_tensor_reduce(
        lambda left, right: left // right
    ),
    "Mod": _at_mod,
    "Pow": _at_pow,
    "Rational": _at_div,
    "MatMult": _at_matmul,
    "And": _at_and,
    "Or": _at_or,
    "LShift": _at_shl,
    "RShift": _at_shr,
    "Not": _abstract_tensor_method("logical_not"),
    "Equality": _abstract_tensor_reduce(lambda left, right: left == right),
    "Unequality": _abstract_tensor_reduce(lambda left, right: left != right),
    "StrictLessThan": _abstract_tensor_reduce(lambda left, right: left < right),
    "LessThanOrEqual": _abstract_tensor_reduce(lambda left, right: left <= right),
    "StrictGreaterThan": _abstract_tensor_reduce(lambda left, right: left > right),
    "GreaterThanOrEqual": _abstract_tensor_reduce(lambda left, right: left >= right),
    "Sin": _abstract_tensor_method("sin"),
    "Cos": _abstract_tensor_method("cos"),
    "Tan": _abstract_tensor_method("tan"),
    "Exp": _abstract_tensor_method("exp"),
    "Log": _abstract_tensor_method("log"),
    "Sqrt": _abstract_tensor_method("sqrt"),
    "Abs": _abstract_tensor_method("abs"),
    "Sum": _abstract_tensor_sum,
    "Max": _abstract_tensor_maximum,
    "Min": _abstract_tensor_minimum,
    "Indexed": _abstract_tensor_index,
    "IndexedStore": _abstract_tensor_index_store,
    "IndexedBase": _abstract_tensor_identity,
    "Idx": _abstract_tensor_index,
    "MatrixElement": _abstract_tensor_index,
    "MatrixSymbol": _abstract_tensor_identity,
    "Store": _abstract_tensor_identity,
    "Tuple": _abstract_tensor_tuple,
    "ExprCondPair": _abstract_tensor_tuple,
    "Piecewise": lambda *values: _abstract_tensor_where(
        _abstract_tensor_values(*values)[2],
        _abstract_tensor_values(*values)[0],
        _abstract_tensor_values(*values)[1],
    ),
    "BooleanTrue": _abstract_tensor_constant(True),
    "BooleanFalse": _abstract_tensor_constant(False),
    "Float": _abstract_tensor_static("get_tensor"),
    "Half": _abstract_tensor_constant(0.5),
    "Pi": _abstract_tensor_constant(np.pi),
    "E": _abstract_tensor_constant(np.e),
    "ImaginaryUnit": _abstract_tensor_constant(1j),

    # Canonical AbstractTensor and Python-call spellings.  Keeping these in
    # the same table lets ProcessGraph nodes produced by AST, SymPy, tape, or
    # another graph importer share one backend adapter.
    "add": _at_add,
    "sub": _at_sub,
    "mul": _at_mul,
    "div": _at_div,
    "truediv": _at_div,
    "mod": _at_mod,
    "pow": _at_pow,
    "matmul": _at_matmul,
    "random_source": _abstract_tensor_random_source,
    "bitand": _at_and,
    "bitor": _at_or,
    "bitxor": _at_xor,
    "shl": _at_shl,
    "shr": _at_shr,
    "logical_and": _at_and,
    "logical_or": _at_or,
    "logical_not": _abstract_tensor_method("logical_not"),
    "equal": _abstract_tensor_reduce(lambda left, right: left == right),
    "not_equal": _abstract_tensor_reduce(lambda left, right: left != right),
    "less": _abstract_tensor_reduce(lambda left, right: left < right),
    "less_equal": _abstract_tensor_reduce(lambda left, right: left <= right),
    "greater": _abstract_tensor_reduce(lambda left, right: left > right),
    "greater_equal": _abstract_tensor_reduce(lambda left, right: left >= right),
    "sin": _abstract_tensor_method("sin"),
    "cos": _abstract_tensor_method("cos"),
    "tan": _abstract_tensor_method("tan"),
    "asin": _abstract_tensor_method("asin"),
    "acos": _abstract_tensor_method("acos"),
    "atan": _abstract_tensor_method("atan"),
    "sinh": _abstract_tensor_method("sinh"),
    "cosh": _abstract_tensor_method("cosh"),
    "tanh": _abstract_tensor_method("tanh"),
    "asinh": _abstract_tensor_method("asinh"),
    "acosh": _abstract_tensor_method("acosh"),
    "atanh": _abstract_tensor_method("atanh"),
    "exp": _abstract_tensor_method("exp"),
    "erf": _abstract_tensor_erf,
    "log": _abstract_tensor_method("log"),
    "sqrt": _abstract_tensor_method("sqrt"),
    "abs": _abstract_tensor_method("abs"),
    "sign": _abstract_tensor_method("sign"),
    "round": _abstract_tensor_primitive("round"),
    "trunc": _abstract_tensor_primitive("trunc"),
    "floor": _abstract_tensor_primitive("floor"),
    "ceil": _abstract_tensor_primitive("ceil"),
    "ceiling": _abstract_tensor_primitive("ceil"),
    "isfinite": _abstract_tensor_method("isfinite"),
    "isnan": _abstract_tensor_method("isnan"),
    "isinf": _abstract_tensor_method("isinf"),
    "maximum": _abstract_tensor_maximum,
    "minimum": _abstract_tensor_minimum,
    "sum": _abstract_tensor_sum,
    "mean": _abstract_tensor_method("mean"),
    "max": _abstract_tensor_method("max"),
    "min": _abstract_tensor_method("min"),
    "reshape": _abstract_tensor_method("reshape"),
    "view": _abstract_tensor_method("view"),
    "flatten": _abstract_tensor_method("flatten"),
    "transpose": _abstract_tensor_method("transpose"),
    "permute": _abstract_tensor_method("permute"),
    "unsqueeze": _abstract_tensor_method("unsqueeze"),
    "squeeze": _abstract_tensor_method("squeeze"),
    "repeat": _abstract_tensor_method("repeat"),
    "repeat_interleave": _abstract_tensor_method("repeat_interleave"),
    "swapaxes": _abstract_tensor_method("swapaxes"),
    "eye_like": _abstract_tensor_method("eye_like"),
    "zeros_like": _abstract_tensor_method("zeros_like"),
    "ones_like": _abstract_tensor_method("ones_like"),
    "full_like": _abstract_tensor_method("full_like"),
    "rand_like": _abstract_tensor_method("rand_like"),
    "randn": _abstract_tensor_method("randn"),
    "randint_like": _abstract_tensor_method("randint_like"),
    "argmax": _abstract_tensor_method("argmax"),
    "argmin": _abstract_tensor_method("argmin"),
    "prod": _abstract_tensor_method("prod"),
    "all": _abstract_tensor_method("all"),
    "any": _abstract_tensor_method("any"),
    "nonzero": _abstract_tensor_method("nonzero"),
    "isinfinite": _abstract_tensor_method("isinfinite"),
    "allclose": _abstract_tensor_method("allclose"),
    "argwhere": _abstract_tensor_method("argwhere"),
    "sec": _abstract_tensor_method("sec"),
    "csc": _abstract_tensor_method("csc"),
    "cot": _abstract_tensor_method("cot"),
    "sech": _abstract_tensor_method("sech"),
    "csch": _abstract_tensor_method("csch"),
    "coth": _abstract_tensor_method("coth"),
    "sinc": _abstract_tensor_method("sinc"),
    "deg2rad": _abstract_tensor_method("deg2rad"),
    "rad2deg": _abstract_tensor_method("rad2deg"),
    "to": _abstract_tensor_method("to"),
    "astype": _abstract_tensor_method("astype"),
    "long_cast": _abstract_tensor_method("long_cast"),
    "float": _abstract_tensor_method("float"),
    "double": _abstract_tensor_method("double"),
    "int": _abstract_tensor_method("int"),
    "long": _abstract_tensor_method("long"),
    "bool": _abstract_tensor_method("bool"),
    "cpu": _abstract_tensor_method("cpu"),
    "cuda": _abstract_tensor_method("cuda"),
    "softmax": _abstract_tensor_method("softmax"),
    "log_softmax": _abstract_tensor_method("log_softmax"),
    "pad": _abstract_tensor_method("pad"),
    "gather": _abstract_tensor_method("gather"),
    "topk": _abstract_tensor_topk,
    "stack": _abstract_tensor_stack,
    "cat": _abstract_tensor_cat,
    "concat": _abstract_tensor_cat,
    "concatenate": _abstract_tensor_cat,
    "where": _abstract_tensor_where,
    "nan_to_num": _abstract_tensor_static("nan_to_num"),
    "dot": _abstract_tensor_static("dot"),
    "norm": _abstract_tensor_static("norm"),
    "cross": _abstract_tensor_static("cross"),
    "trace": _abstract_tensor_static("trace"),
    "det": _abstract_tensor_static("det"),
    "solve": _abstract_tensor_static("solve"),
    "inv": _abstract_tensor_static("inv"),
    "inverse": _abstract_tensor_static("inverse"),
    "eigh": _abstract_tensor_static("eigh"),
    "cholesky": _abstract_tensor_static("cholesky"),
    "fft": _abstract_tensor_method("fft"),
    "ifft": _abstract_tensor_method("ifft"),
    "tuple": _abstract_tensor_tuple,
    "store": _abstract_tensor_identity,
}

# Complete the ProcessGraph execution table from the canonical AbstractTensor
# inventory.  Bespoke handlers above remain authoritative; this fills only
# names whose ordinary behavior is a public method, constructor, primitive, or
# observable tensor attribute.
_attribute_operator_names = {"device", "dtype", "ndim", "shape", "tensor_type"}
_static_operator_targets = {
    name: name for name in CREATION_OPERATORS
}
_static_operator_targets.update(
    {
        "load": "load",
        "pi": "pi",
        # The public compatibility spelling omits the backend-hook suffix.
        "tensor_from_list": "tensor_from_list_",
    }
)
for _name in sorted(CANONICAL_ABSTRACT_TENSOR_OPERATORS):
    if _name in abstract_tensor_funcs:
        continue
    if _name in _attribute_operator_names:
        _handler = _abstract_tensor_attribute(_name)
    elif _name in _static_operator_targets:
        _handler = _abstract_tensor_static(_static_operator_targets[_name])
    elif _name in ELEMENTWISE_UNARY:
        _handler = _abstract_tensor_primitive(_name)
    elif _name == "floordiv":
        _handler = _abstract_tensor_reduce(
            lambda left, right: left // right
        )
    else:
        _handler = _abstract_tensor_method(_name)
    abstract_tensor_funcs[_name] = _handler

for _alias, _canonical in OPERATOR_ALIASES.items():
    abstract_tensor_funcs[_alias] = abstract_tensor_funcs[_canonical]

_abstract_tensor_unary_names = {
    "Sin", "Cos", "Tan", "Exp", "Log", "Sqrt", "Abs", "Not",
    "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh",
    "tanh", "asinh", "acosh", "atanh", "exp", "log", "sqrt", "abs",
    "sign", "round", "trunc", "floor", "ceil", "ceiling", "isfinite",
    "isnan", "isinf", "isinfinite", "logical_not", "argmax", "argmin",
    "prod", "all", "any", "nonzero", "argwhere", "sec", "csc", "cot",
    "sech", "csch", "coth", "sinc", "deg2rad", "rad2deg", "float",
    "double", "int", "long", "bool", "cpu", "cuda", "det", "inv", "erf",
    "inverse", "cholesky", "fft", "ifft",
}
_abstract_tensor_binary_names = {
    "Add", "Sub", "Mul", "Div", "Mod", "Pow", "Rational", "MatMult",
    "FloorDiv",
    "And", "Or", "Equality", "Unequality", "StrictLessThan",
    "LessThanOrEqual", "StrictGreaterThan", "GreaterThanOrEqual",
    "add", "sub", "mul", "div", "truediv", "mod", "pow", "matmul",
    "bitand", "bitor", "bitxor", "shl", "shr", "logical_and", "logical_or",
    "equal",
    "not_equal", "less", "less_equal", "greater", "greater_equal",
    "maximum", "minimum", "allclose", "dot", "cross", "solve",
    "LShift", "RShift",
}
_abstract_tensor_constant_names = {
    "BooleanTrue", "BooleanFalse", "Half", "Pi", "E", "ImaginaryUnit",
    "MatrixSymbol",
}
_abstract_tensor_store_names = {"Store", "store"}
abstract_tensor_sigs = {
    name: (
        sig_constant
        if name in _abstract_tensor_constant_names
        else sig_store
        if name in _abstract_tensor_store_names
        else sig_unary_elementwise
        if name in _abstract_tensor_unary_names
        else sig_binary_elementwise
        if name in _abstract_tensor_binary_names
        else operator_signatures.get(name, sig_sum_like)
    )
    for name in abstract_tensor_funcs
}

# Constructors may have no tensor-valued predecessor; accessors always consume
# exactly one.  The remaining newly catalogued methods conservatively retain
# the established variadic tensor-method schema.
_sig_creation = {
    'min_inputs': 0, 'max_inputs': None,
    'min_outputs': 1, 'max_outputs': None,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': False,
}
for _name in CREATION_OPERATORS | {"load", "pi"}:
    abstract_tensor_sigs[_name] = _sig_creation
for _name in ACCESSOR_OPERATORS:
    abstract_tensor_sigs[_name] = sig_unary_elementwise
for _alias, _canonical in OPERATOR_ALIASES.items():
    abstract_tensor_sigs[_alias] = abstract_tensor_sigs[_canonical]

# Make the shared ProcessGraph signature surface aware of the canonical
# AbstractTensor spellings without replacing any established SymPy/AST entry.
for _name, _signature in abstract_tensor_sigs.items():
    operator_signatures.setdefault(_name, _signature)



numpy_sigs = {k: v for k, v in operator_signatures.items() if k in numpy_funcs}
torch_sigs = {k: v for k, v in operator_signatures.items() if k in torch_funcs}
numpy_sigs.update(array_sigs_overrides)
torch_sigs.update(array_sigs_overrides)

def advanced_piecewise_handler(node, inputs, pg):
    """
    node: a SymPy Piecewise instance
    inputs: list of already‐lowered child Tensors
      for Piecewise, inputs = [expr_true, expr_false, cond]
    pg: your ProcessGraph builder
    """
    # simply map Piecewise((T, C),(F,True)) → torch.where(C, T, F)
    true_val, false_val, cond = inputs
    return pg.call_op("where", [cond, true_val, false_val], name="piecewise")

advanced_piecewise_signature = {
    'min_inputs': 3, 'max_inputs': 3,
    'min_outputs': 1, 'max_outputs': 1,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
}


def expr_cond_pair_handler(node, inputs, pg):
    """
    node.expr  is the 'then' branch,
    node.cond  is the boolean condition.
    inputs == [expr_node, cond_node]
    We just return them as a lightweight 2‐tuple so the
    Piecewise handler can see [(expr,cond), ...].
    """
    expr_node, cond_node = inputs
    return (expr_node, cond_node)
torch_funcs['ExprCondPair'] = expr_cond_pair_handler
torch_sigs['ExprCondPair'] = {
    'min_inputs': 2, 'max_inputs': 2,
    'min_outputs': 1, 'max_outputs': 1,
}


torch_funcs['Piecewise'] = advanced_piecewise_handler
torch_sigs['Piecewise'] = advanced_piecewise_signature




# Union of all handler names from both
all_handler_keys = sorted(set(torch_funcs.keys()).union(numpy_funcs.keys()))

# New dicts: full set, in order, with None for missing
mirrored_torch_funcs  = {k: torch_funcs.get(k)  for k in all_handler_keys}
mirrored_numpy_funcs  = {k: numpy_funcs.get(k)  for k in all_handler_keys}
mirrored_torch_sigs   = {k: torch_sigs.get(k)   for k in all_handler_keys}
mirrored_numpy_sigs   = {k: numpy_sigs.get(k)   for k in all_handler_keys}

torch_funcs = mirrored_torch_funcs
#torch_sigs = mirrored_torch_sigs
numpy_funcs = mirrored_numpy_funcs
#numpy_sigs = mirrored_numpy_sigs

import functools

def make_logging_wrapper(handler_name, real_fn):
    @functools.wraps(real_fn)
    def wrapper(*args, **kwargs):
        print(f"[DEBUG] Handler '{handler_name}' called with args={args}, kwargs={kwargs}")
        result = real_fn(*args, **kwargs)
        print(f"[DEBUG] Handler '{handler_name}' returned {result}")
        return result
    return wrapper
def wrap_all_handlers_with_logging(handler_map, backend_name="handler"):
    wrapped = {}
    for k, fn in handler_map.items():
        if fn is not None:
            wrapped[k] = make_logging_wrapper(f"{backend_name}.{k}", fn)
        else:
            wrapped[k] = None
    return wrapped

debug_torch_funcs = wrap_all_handlers_with_logging(mirrored_torch_funcs, "torch")
debug_numpy_funcs = wrap_all_handlers_with_logging(mirrored_numpy_funcs, "numpy")

torch_funcs = debug_torch_funcs
numpy_funcs = debug_numpy_funcs


#!/usr/bin/env python3
"""
Generate a name_map from Sympy node names to SSA Handler enum.
Collects all keys from operator_defs handler dicts (default_funcs, numpy_funcs,
torch_funcs, abstract_tensor_funcs, math_funcs) and outputs a mapping suitable
for SympyToSSA name_map.
"""

from . import operator_defs
from .ssa import Handler


def main():
    # Gather all handler dicts keys
    key_sets = []
    key_sets.append(set(operator_defs.default_funcs.keys()))
    key_sets.append(set(operator_defs.numpy_funcs.keys()))
    key_sets.append(set(operator_defs.torch_funcs.keys()))
    key_sets.append(set(operator_defs.abstract_tensor_funcs.keys()))
    # Include math_funcs if available
    if hasattr(operator_defs, 'math_funcs'):
        key_sets.append(set(operator_defs.math_funcs.keys()))

    # Union of all keys
    all_keys = set().union(*key_sets)

    # Print mapping
    print("name_map = {")
    for key in sorted(all_keys):
        if hasattr(Handler, key):
            handler_ref = f"Handler.{key}"
        else:
            handler_ref = "# TODO: Handler missing"
        print(f"    '{key}': {handler_ref},")
    print("}")


if __name__ == '__main__':
    main()
