import numpy as np
import sympy
import torch            # ← new

SIMD_DEFAULT_CONCURRENCY = 4
numpy_funcs, torch_funcs, numpy_sigs, torch_sigs = {}, {}, {}, {}
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
    'Invert':      {'up': {}, 'down': {}},
    'Not':         {'up': {}, 'down': {}},
    'UAdd':        {'up': {}, 'down': {}},
    'USub':        {'up': {}, 'down': {}},

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
Collects all keys from operator_defs handler dicts (default_funcs, numpy_funcs, torch_funcs, math_funcs) and outputs a mapping suitable for SympyToSSA name_map.
"""

from . import operator_defs
from .ssa import Handler


def main():
    # Gather all handler dicts keys
    key_sets = []
    key_sets.append(set(operator_defs.default_funcs.keys()))
    key_sets.append(set(operator_defs.numpy_funcs.keys()))
    key_sets.append(set(operator_defs.torch_funcs.keys()))
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
