# bitops.py
"""
Universal bit-level operations & gray code encoding,
encapsulated in a class with rigorous validation.
"""
import itertools
import math
import threading
import re, ctypes, zlib
from enum import Enum, auto
import hashlib
import random
from uuid import uuid4

try:
    from ..transmogrifier.graph.memory_graph import BitTensorMemoryGraph, NodeEntry, EdgeEntry, GraphSearch  # type: ignore
    import json  # use standard library json
except Exception:  # pragma: no cover - optional deps
    BitTensorMemoryGraph = NodeEntry = EdgeEntry = GraphSearch = None  # type: ignore
    json = None  # type: ignore

from .bitops_translator import BitOpsTranslator

class BitStructType(Enum):
    INTEGER = auto()
    RATIONAL = auto()
    FLOAT = auto()
    COMPLEX = auto()
    INFINITESIMAL = auto()
    LIMIT = auto()
    DOMAIN = auto()
    BOUNDARY = auto()

    TENSOR = auto()
    TAYLOR_SERIES = auto()
    SUM = auto()
    INTEGRAL = auto()
    DERIVATIVE = auto()
    GRAPH = auto()
    DAG = auto()

    STRUCT = auto()
    FUNCTION = auto()
    TABLE = auto()
    MANIFOLD = auto()
    SYMBOLIC = auto()

    EXPRESSION = auto()
    SERIALIZATION = auto()

    IF = auto()
    ELSE = auto()
    MATCH = auto()
    
    # The following are redundant but translate
    # programmatic constructs to pure mathematical terms

    FOR = auto()
    WHILE = auto()

    # These are control flow constructs
    
    FLAG = auto()  # for bit-level flags
    SWITCH = auto()
    GOTO = auto()
    BREAK = auto()
    CONTINUE = auto()   # 

    RETURN = auto()     # has a schema
    ASSIGN = auto()    # stores conversion flags

    BITLAYOUT = auto()  # for bit-level layout operations
    TYPE = auto()      # for depth and name correlation
    SCHEMA = auto()      # for defining unit level schemas
    BITTENSOR = auto()  # for bit-level tensor operations
    BITTENSORMEMORY = auto()  # for bit-level tensor memory operations
    BITTENSORMEMORYGRAPH = auto()  # for bit-level tensor memory graph operations

class BitStruct:
    def __init__(self, integer_pieces, depths, encoding="gray"):
        """
        Initialize with:
        - integer_pieces: list of integers to encode
        - depths: list of bit widths for each piece
        - encoding: 'gray' or 'binary'
        """
        self.integer_pieces = integer_pieces
        self.depths = depths
        self.encoding = encoding
        self.bit_width = sum(depths)
        self.mask = (1 << self.bit_width) - 1



class Integer(BitStruct):
    def __init__(self, value, bit_width=32, encoding="gray"):
        super().__init__([value], [bit_width], encoding)

class Rational(BitStruct):
    def __init__(self, numerator, denominator, bit_width=32, encoding="gray"):
        super().__init__([numerator, denominator], [bit_width, bit_width], encoding)

class Float(BitStruct):
    def __init__(self, mantissa, exponent, mantissa_width=23, exponent_width=8, encoding="gray"):
        super().__init__([mantissa, exponent], [mantissa_width, exponent_width], encoding)

class Complex(BitStruct):
    def __init__(self, real, imag, real_width=32, imag_width=32, encoding="gray"):
        super().__init__([real, imag], [real_width, imag_width], encoding)

class Flag(BitStruct):
    def __init__(self, size=0, bit_width=1, encoding="binary", flags=None):
        if not flags:
            flags = [0] * bit_width * size
        self.mask = (1 << bit_width * size) - 1

        super().__init__(flags, [1] *  size, encoding)



    def __getitem__(self, idx):
        return self.integer_pieces[idx]
    
    def __setitem__(self, idx, value):
        self.integer_pieces[idx] = value

    def strip(self):
        """
        Strip the flag value, returning the integer piece.
        """
        return self.integer_pieces
    
    def pack(self):
        """
        Pack the flag into a single integer.
        """
        packed = 0
        for i, val in enumerate(self.integer_pieces):
            packed |= (val & self.mask) << (i * self.depths[i])
        return packed

class Boundary(Flag):
    DIRICHLET = auto()
    NEUMANN = auto()
    PERIODIC = auto()
    FUNCTION = auto()
    CONSTANT = auto()

    def __init__(self, flags=None, encoding="binary"):
        super().__init__(flags=flags, bit_width=3, encoding=encoding)

class Domain(BitStruct):
    def __init__(self, limits, boundaries=None):
        self.limits = limits
        self.boundaries = boundaries if boundaries else []

class Limit(BitStruct):
    def __init__(self, lower, upper, l_bit_width=32, r_bit_width=32, encoding="gray", linfinity=False, rinfinity=False, linc=True, rinc=True, fromtheleft=True, flags=None):
        """
        super().__init__([lower, upper], [bit_width, bit_width], encoding)
        self.linfinity = linfinity
        self.rinfinity = rinfinity
        self.linc = linc
        self.rinc = rinc
        """
        if flags is None:
            flag_size = 5
            packed_flags = linfinity | (rinfinity << 1) | (linc << 2) | (rinc << 3) | (fromtheleft << 4)
            flags = Flag(flags=[(packed_flags >> i) & 1 for i in range(flag_size)], bit_width=flag_size, encoding=encoding)
        super().__init__([flags, lower, upper], [flag_size, l_bit_width, r_bit_width], encoding)
        self.linfinity = linfinity
        self.rinfinity = rinfinity
        self.linc = linc
        self.rinc = rinc
        self.fromtheleft = fromtheleft

class Tensor(Domain):
    CONTINUOUS = auto()
    SPARSE = auto()
    STATIC_ALLOC = auto()

    def __init__(self, limits, flags=None, boundaries=None, encoding="gray"):
        if flags is None:
            flags = Flag(3)
            flags.integer_pieces[0] = 1  # default to CONTINUOUS
            flags.integer_pieces[1] = 0  # default to SPARSE
            flags.integer_pieces[2] = 0  # default to STATIC_ALLOC

        super().__init__(limits, boundaries)
        self.flags = flags
        self.encoding = encoding

class BitTensor(Tensor):
    def __init__(self, limits, flags=None, boundaries=None, encoding="binary"):
        super().__init__(limits, flags, boundaries, encoding)
        self.bit_width = 1

    def __getitem__(self, idx, value):
        """
        Get the bit value at index idx.
        """
        width = idx[1] - idx[0]
        j = 0
        for i in range(idx[0], idx[1]):
            value[j] = self.integer_pieces[i]
            j += 1
        return value


    def __setitem__(self, idx, value):
        """
        Set the bit value at index idx.
        """
        width = idx[1] - idx[0]
        j = 0
        for i in range(idx[0], idx[1]):
            self.integer_pieces[i] = value[j]
            j += 1        

class Schema:
    def __init__(self, fields, types, widths, encodings):
        self.fields = fields
        self.types = types
        self.widths = widths
        self.encodings = encodings
    def factory(self, bit_tensor, values):
        index = 0
        for i, value in enumerate(values):
            # this is a kind of slicing and byte to bit index we have to implement
            bit_tensor[index:index+self.widths[i]] = value
            index += self.widths[i]
    def decompress(self, bit_tensor, values):
        index = 0
        for width in self.widths:
            value = bit_tensor[index:index+width]
            values.append(value)
            index += width
        return values

class Graph(BitStruct):
    def __init__(self, nodes, sources, destinations, encoding="gray"):
        super().__init__([nodes, sources, destinations], [32, 32, 32], encoding)
        self.nodes = nodes
        self.sources = sources
        self.destinations = destinations

class DAG(Graph):
    def __init__(self, nodes, sources, destinations, stages, stack, encoding="gray"):
        super().__init__(nodes, sources, destinations, encoding)
        self.is_dag = True  # flag for DAG property
        self.stages = stages
        self.stack = stack
        self.encoding = encoding

class Serialization(DAG, Schema, BitTensor):
    def __init__(self, nodes, sources, destinations, stages, stack, limits, flags=None, boundaries=None, encoding="gray"):
        DAG.__init__(self, nodes, sources, destinations, stages, stack, encoding)
        Schema.__init__(self, [], [], [], [])
        BitTensor.__init__(self, limits, flags=flags, boundaries=boundaries, encoding=encoding)
        self.is_serialization = True  # flag for Serialization property

class Function(Schema, DAG):
    def __init__(self, inputs, outputs, input_types, output_types, reference, encoding="gray"):
        schema_content = [inputs + outputs + input_types + output_types + reference]
        super().__init__(schema_content, [BitStructType.INTEGER] * len(schema_content), [32] * len(schema_content), encoding)
        self.input_types = input_types
        self.output_types = output_types
        self.reference = reference

class TaylorSeries(Domain):
    def __init__(self, order_domain, function, center, encoding="gray"):
        super().__init__(order_domain.limits, order_domain.boundaries)
        self.function = function
        self.center = center
        self.encoding = encoding
        
class Infinitesimal:
    def __init__(self, value, encoding="gray"):
        self.value = value
        self.encoding = encoding

class Manifold(Tensor, Function):
    def __init__(self, limits, flags=None, boundaries=None, encoding="gray"):
        super().__init__(limits, flags=flags, boundaries=boundaries, encoding=encoding)

class Table(Manifold, BitTensor, Schema):
    def __init__(self, limits, flags=None, boundaries=None, encoding="gray"):
        super().__init__(limits, flags=flags, boundaries=boundaries, encoding=encoding)
        self.is_table = True  # flag for Table property

    def __getitem__(self, idx, value):
        return super().__getitem__(idx, value)
    


class Sum(Manifold):
    def __init__(self, limits, flags=None, boundaries=None, encoding="gray"):
        super().__init__(limits, flags=flags, boundaries=boundaries, encoding=encoding)
        self.is_sum = True  # flag for Sum property

class Integral(Sum):
    def __init__(self, limits, flags=None, boundaries=None, encoding="gray"):
        super().__init__(limits, flags=flags, boundaries=boundaries, encoding=encoding)
        self.is_integral = True  # flag for Integral property

class Derivative(Manifold, Infinitesimal):
    def __init__(self, limits, flags=None, boundaries=None, encoding="gray"):
        super().__init__(limits, flags=flags, boundaries=boundaries, encoding=encoding)
        self.is_derivative = True  # flag for Derivative property

class Struct(Schema, BitTensor):
    def __init__(self, fields, types, widths, encodings, data):
        super().__init__(fields, types, widths, encodings)
        self.is_struct = True  # flag for Struct property
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value

class Symbolic(Integer):
    def __init__(self, value, bit_width=32, encoding="gray"):
        super().__init__(value, bit_width, encoding)
        self.is_symbolic = True  # flag for Symbolic property

class Expression(Symbolic, Manifold):
    def __init__(self, fields, types, widths, encodings, data):
        super().__init__(fields, types, widths, encodings, data)
        self.is_expression = True

class If(Expression):
    def __init__(self, condition, true_branch, false_branch, encoding="gray"):
        super().__init__([], [], [], encoding, [])
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.is_if = True  # flag for If property

class Else(Expression):
    def __init__(self, condition, true_branch, false_branch, encoding="gray"):
        super().__init__([], [], [], encoding, [])
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.is_else = True  # flag for Else property

class Match(Expression):
    def __init__(self, subject, cases, encoding="gray"):
        super().__init__([], [], [], encoding, [])
        self.subject = subject
        self.cases = cases
        self.is_match = True  # flag for Match property

class For(Expression):
    def __init__(self, iterable, body, encoding="gray"):
        super().__init__([], [], [], encoding, [])
        self.iterable = iterable
        self.body = body
        self.is_for = True  # flag for For property

class While(Expression):
    def __init__(self, condition, body, encoding="gray"):
        super().__init__([], [], [], encoding, [])
        self.condition = condition
        self.body = body
        self.is_while = True  # flag for While property

class Switch(Expression, Schema):
    def __init__(self, subject, cases, default_case=None, encoding="gray"):
        super().__init__([], [], [], encoding, [])
        self.subject = subject
        self.cases = cases
        self.default_case = default_case
        self.is_switch = True  # flag for Switch property

class Goto(Integer):
    def __init__(self, target, encoding="gray"):
        super().__init__([], [], [], encoding, [])
        self.target = target
        self.is_goto = True  # flag for Goto property

class Break(Goto):
    def __init__(self, target=None, encoding="gray"):
        super().__init__(target, encoding)
        self.is_break = True  # flag for Break property

class Continue(Goto):
    def __init__(self, target=None, encoding="gray"):
        super().__init__(target, encoding)
        self.is_continue = True  # flag for Continue property

class Return(Goto, Struct):
    def __init__(self, value, fields, types, widths, encodings, encoding="gray"):
        super().__init__(value, encoding)
        self.is_return = True  # flag for Return property
        self.schema = Schema(fields, types, widths, encodings)
        self.value = value

class FunctionCall(Goto, Struct):
    def __init__(self, param_memory, calling_memory, return_memory, resume_memory, function_name, args, fields, types, widths, encodings, encoding="gray"):
        super().__init__(function_name, encoding)
        self.param_memory = param_memory
        self.calling_memory = calling_memory
        self.return_memory = return_memory
        self.resume_memory = resume_memory
        self.is_function_call = True  # flag for FunctionCall property
        self.schema = Schema(fields, types, widths, encodings)
        self.args = args


class BitOps:
    def __init__(self, bit_width, encoding="gray"):
        self.bit_width = bit_width
        self.encoding = encoding
        self.mask = (1 << bit_width) - 1
        # Translator that expands operations into Turing primitives
        self.translator = BitOpsTranslator(bit_width)

    # ─────────────────────────────────────────────
    # Gray code conversions
    # ─────────────────────────────────────────────
    def int_to_gray(self, n):
        self._validate_value(n)
        return n ^ (n >> 1)

    def gray_to_int(self, g):
        self._validate_value(g)
        n = g
        shift = 1
        while (g >> shift) > 0:
            n ^= (g >> shift)
            shift += 1
        return n

    # ─────────────────────────────────────────────
    # Basic bit arithmetic
    # ─────────────────────────────────────────────
    def bit_add(self, x, y):
        self._validate_value(x)
        self._validate_value(y)
        return self.translator.bit_add(x, y) & self.mask

    def bit_sub(self, x, y):
        self._validate_value(x)
        self._validate_value(y)
        return self.translator.bit_sub(x, y) & self.mask

    def bit_and(self, x, y):
        self._validate_value(x)
        self._validate_value(y)
        return self.translator.bit_and(x, y)

    def bit_or(self, x, y):
        self._validate_value(x)
        self._validate_value(y)
        return self.translator.bit_or(x, y)

    def bit_xor(self, x, y):
        self._validate_value(x)
        self._validate_value(y)
        return self.translator.bit_xor(x, y)

    def bit_mul(self, x, y):
        self._validate_value(x)
        self._validate_value(y)
        return self.translator.bit_mul(x, y) & self.mask

    def bit_div(self, x, y):
        self._validate_value(x)
        self._validate_value(y)
        return self.translator.bit_div(x, y) & self.mask

    def bit_mod(self, x, y):
        self._validate_value(x)
        self._validate_value(y)
        return self.translator.bit_mod(x, y) & self.mask

    def bit_not(self, x):
        self._validate_value(x)
        return self.translator.bit_not(x)

    def bit_shift_left(self, x, shift):
        self._validate_value(x)
        return self.translator.bit_shift_left(x, shift)

    def bit_shift_right(self, x, shift):
        self._validate_value(x)
        return self.translator.bit_shift_right(x, shift)

    # ─────────────────────────────────────────────
    # Brutal validator
    # ─────────────────────────────────────────────
    def _validate_value(self, val):
        if val < 0 or val > self.mask:
            raise ValueError(f"Value {val} exceeds bit width {self.bit_width}")

    # ─────────────────────────────────────────────
    # Stubs for encoder/decoder
    # ─────────────────────────────────────────────
    def encode(self, values):
        """
        Pack a list of integers into a bitstream with chosen encoding.
        For now, stub simply packs with gray if set.
        """
        packed = 0
        shift = 0
        for val in values:
            self._validate_value(val)
            encoded_val = self.int_to_gray(val) if self.encoding == "gray" else val
            packed |= (encoded_val & self.mask) << shift
            shift += self.bit_width
        return packed

    def decode(self, packed, count):
        """
        Unpack a bitstream into list of integers using chosen decoding.
        """
        values = []
        for _ in range(count):
            bits = packed & self.mask
            decoded_val = self.gray_to_int(bits) if self.encoding == "gray" else bits
            self._validate_value(decoded_val)
            values.append(decoded_val)
            packed >>= self.bit_width
        return values
# test_bitops.py
from .bitops import BitOps

def validate_ops(bit_width, encoding="gray"):
    ops = BitOps(bit_width=bit_width, encoding=encoding)
    failures = 0
    max_val = (1 << bit_width)

    for x in range(max_val):
        for y in range(max_val):
            try:
                # encode inputs
                ex = ops.int_to_gray(x) if encoding == "gray" else x
                ey = ops.int_to_gray(y) if encoding == "gray" else y

                # case block: native bitwise operators in encoded space
                and_result = ops.bit_and(ex, ey)
                or_result  = ops.bit_or(ex, ey)
                xor_result = ops.bit_xor(ex, ey)
                not_x      = ops.bit_not(ex)

                # decode results
                and_decoded = ops.gray_to_int(and_result) if encoding == "gray" else and_result
                or_decoded  = ops.gray_to_int(or_result)  if encoding == "gray" else or_result
                xor_decoded = ops.gray_to_int(xor_result) if encoding == "gray" else xor_result
                not_decoded = ops.gray_to_int(not_x)      if encoding == "gray" else not_x

                # check against expected raw bitwise
                assert and_decoded == (x & y), f"AND failed: {x}&{y} -> {and_decoded} != {x&y}"
                assert or_decoded  == (x | y), f"OR failed: {x}|{y} -> {or_decoded} != {x|y}"
                assert xor_decoded == (x ^ y), f"XOR failed: {x}^{y} -> {xor_decoded} != {x^y}"
                assert not_decoded == (~x & ops.mask), f"NOT failed: ~{x} -> {not_decoded} != {~x & ops.mask}"

                # arithmetic: decode, compute, re-encode to check system symmetry
                raw_sum = (x + y) & ops.mask
                enc_sum = ops.int_to_gray(raw_sum) if encoding == "gray" else raw_sum
                roundtrip_sum = ops.int_to_gray(ops.bit_add(x, y)) if encoding == "gray" else ops.bit_add(x, y)
                assert enc_sum == roundtrip_sum, f"ADD failed: {x}+{y} -> {roundtrip_sum} != {enc_sum}"

            except AssertionError as e:
                failures += 1
                print("FAIL:", e)

    if failures == 0:
        print(f"All tests passed for {bit_width}-bit {encoding} encoding.")
    else:
        print(f"Completed with {failures} failures.")

def node_desc(node):
    # For AST: try to show operator/target/value
    import ast
    if isinstance(node, ast.Assign):
        return "Assign: " + ", ".join(getattr(t, 'id', '?') for t in node.targets)
    elif isinstance(node, ast.Name):
        return "Var: " + node.id
    elif isinstance(node, ast.BinOp):
        op = type(node.op).__name__
        return f"BinOp: {op}"
    elif isinstance(node, ast.Constant):
        return f"Const: {node.value!r}"
    elif isinstance(node, ast.Call):
        f = getattr(node.func, 'id', '?')
        return f"Call: {f}"
    elif isinstance(node, ast.FunctionDef):
        return f"FuncDef: {node.name}"
    elif isinstance(node, ast.Return):
        return "Return"
    elif hasattr(node, 'val'):
        return f"OOP: {getattr(node, 'val')}"
    return type(node).__name__


def generic_tree_to_dag(root, get_children):
    class ProcDAGNode:
        def __init__(self, op_type, args, node_id, desc=""):
            self.op_type = op_type
            self.args = args
            self.node_id = node_id
            self.deps = []
            self.desc = desc
        def __repr__(self):
            s = f"<{self.op_type}#{self.node_id}"
            if self.desc:
                s += f' "{self.desc}"'
            if self.deps:
                s += f" deps={self.deps}"
            if self.args:
                s += f" args={self.args}"
            return s + ">"

    class ProcDAG:
        def __init__(self):
            self.nodes = []
            self.edges = []
            self.next_id = 0
        
        def add_node(self, op_type, args, desc=""):
            node = ProcDAGNode(op_type, args, self.next_id, desc)
            self.nodes.append(node)
            self.next_id += 1
            return node

        def add_edge(self, src, dst):
            self.edges.append((src.node_id, dst.node_id))
            dst.deps.append(src.node_id)
        def __repr__(self):
            return f"ProcDAG(nodes={self.nodes}, edges={self.edges})"

    dag = ProcDAG()
    node_map = {}

    def visit(node, parent=None):
        if isinstance(node, list):
            last = None
            for child in node:
                n = visit(child, parent)
                if last and n:
                    dag.add_edge(last, n)
                last = n
            return last
        else:
            op_type = type(node).__name__
            desc = node_desc(node)
            args = []  # or extract fields
            dag_node = dag.add_node(op_type, args, desc)
            node_map[id(node)] = dag_node
            children = get_children(node)
            last_child = None
            for child in children:
                cnode = visit(child, dag_node)
                if last_child and cnode:
                    dag.add_edge(last_child, cnode)
                last_child = cnode
            if last_child:
                dag.add_edge(last_child, dag_node)
            return dag_node


    visit(root)
    return dag

# ---- Usage Example ----
import ast


# For Python AST:
def ast_children(node):
    if isinstance(node, ast.Module):
        return node.body
    elif hasattr(node, 'body') and isinstance(node.body, list):
        return node.body
    # Add more rules or return [] as fallback
    return []

# For OOP objects:
def obj_children(obj):
    # If object has an explicit 'children' attribute or method, use it.
    if hasattr(obj, 'children'):
        return obj.children
    # If it has a __dict__ with ordered fields:
    if hasattr(obj, '__dict__'):
        # Optionally filter to only 'node-like' fields
        # For complete indifference, just get all object attributes that are not builtins
        vals = [v for k, v in obj.__dict__.items() if not k.startswith('__')]
        # Flatten any lists of objects
        children = []
        for v in vals:
            if isinstance(v, list):
                children.extend(v)
            else:
                children.append(v)
        return children
    return []



class GrayTableOps:
    def __init__(self,
                 int_bit_width,
                 float_mantissa_width=None,
                 float_exponent_width=None):
        # ----------------------------
        # Integer setup
        self.int_bit_width = int_bit_width
        self.int_mask = (1 << int_bit_width) - 1
        self.gray_codes = [x ^ (x >> 1) for x in range(1 << int_bit_width)]
        self.gray_add_table = self._build_gray_table('add', int_bit_width)
        self.gray_sub_table = self._build_gray_table('sub', int_bit_width)
        self.gray_mul_table = self._build_gray_table('mul', int_bit_width)
        self.gray_div_table = self._build_gray_table('div', int_bit_width)
        self.gray_mod_table = self._build_gray_table('mod', int_bit_width)

        # ----------------------------
        # Float setup (composed of arbitrary ints)
        self.float_mantissa_width = float_mantissa_width
        self.float_exponent_width = float_exponent_width
        if float_mantissa_width and float_exponent_width:
            self.float_mantissa_mask = (1 << float_mantissa_width) - 1
            self.float_exponent_mask = (1 << float_exponent_width) - 1
            self.float_add_table = {}  # reserved for future precise float adds
            self.float_mul_table = {}  # reserved for future precise float muls

        # ----------------------------
        # Complex setup (two floats)
        self.complex_table = {}  # for caching complex ops

        # ----------------------------
        # Rational setup (pair of ints)
        self.rational_table = {}  # for caching rational operations

    # ──────────────────────────────
    # Gray integer table builder
    # ──────────────────────────────
    def _build_gray_table(self, op, width):
        mask = (1 << width) - 1
        gray_codes = [x ^ (x >> 1) for x in range(1 << width)]
        table = {gx: {gy: 0 for gy in gray_codes} for gx in gray_codes}
        for x in range(1 << width):
            gx = x ^ (x >> 1)
            for y in range(1 << width):
                gy = y ^ (y >> 1)
                if op == 'add':
                    result = (x + y) & mask
                elif op == 'sub':
                    result = (x - y) & mask
                elif op == 'mul':
                    result = (x * y) & mask
                elif op == 'div':
                    result = (x // y) & mask if y != 0 else 0
                elif op == 'mod':
                    result = (x % y) & mask if y != 0 else 0
                gr = result ^ (result >> 1)
                table[gx][gy] = gr
        return table

    # ──────────────────────────────
    # Native integer operations
    # ──────────────────────────────
    def bit_add(self, gx, gy):
        return self.gray_add_table[gx][gy]

    def bit_sub(self, gx, gy):
        return self.gray_sub_table[gx][gy]

    def bit_mul(self, gx, gy):
        return self.gray_mul_table[gx][gy]

    def bit_div(self, gx, gy):
        return self.gray_div_table[gx][gy]

    def bit_mod(self, gx, gy):
        return self.gray_mod_table[gx][gy]

    # ──────────────────────────────
    # Gray floats: (mantissa, exponent) tuple
    # ──────────────────────────────
    def float_add(self, gf1, gf2):
        """
        Each gf = (mantissa_gray, exponent_gray)
        This example does naive alignment by exponent
        """
        (gm1, ge1), (gm2, ge2) = gf1, gf2
        m1 = self._gray_to_int(gm1)
        e1 = self._gray_to_int(ge1)
        m2 = self._gray_to_int(gm2)
        e2 = self._gray_to_int(ge2)
        if e1 > e2:
            m2 >>= (e1 - e2)
            e2 = e1
        else:
            m1 >>= (e2 - e1)
            e1 = e2
        mantissa_sum = (m1 + m2) & self.float_mantissa_mask
        gm_sum = mantissa_sum ^ (mantissa_sum >> 1)
        ge_sum = e1 ^ (e1 >> 1)
        return (gm_sum, ge_sum)

    def float_mul(self, gf1, gf2):
        (gm1, ge1), (gm2, ge2) = gf1, gf2
        m1 = self._gray_to_int(gm1)
        e1 = self._gray_to_int(ge1)
        m2 = self._gray_to_int(gm2)
        e2 = self._gray_to_int(ge2)
        mantissa_prod = (m1 * m2) & self.float_mantissa_mask
        exponent_sum = (e1 + e2) & self.float_exponent_mask
        gm_prod = mantissa_prod ^ (mantissa_prod >> 1)
        ge_sum = exponent_sum ^ (exponent_sum >> 1)
        return (gm_prod, ge_sum)

    # ──────────────────────────────
    # Complex: (gray_float_real, gray_float_imag)
    # ──────────────────────────────
    def complex_add(self, gc1, gc2):
        (gr1, gi1), (gr2, gi2) = gc1, gc2
        return (self.float_add(gr1, gr2), self.float_add(gi1, gi2))

    def complex_mul(self, gc1, gc2):
        (gr1, gi1), (gr2, gi2) = gc1, gc2
        a, b = gr1, gi1
        c, d = gr2, gi2
        ac = self.float_mul(a, c)
        bd = self.float_mul(b, d)
        ad = self.float_mul(a, d)
        bc = self.float_mul(b, c)
        real = self.float_add(ac, self._float_neg(bd))
        imag = self.float_add(ad, bc)
        return (real, imag)

    def _float_neg(self, gf):
        gm, ge = gf
        m = self._gray_to_int(gm)
        e = self._gray_to_int(ge)
        m_neg = (-m) & self.float_mantissa_mask
        gm_neg = m_neg ^ (m_neg >> 1)
        return (gm_neg, ge)

    # ──────────────────────────────
    # Rational: (numerator_gray, denominator_gray)
    # ──────────────────────────────
    def rational_add(self, gr1, gr2):
        (gn1, gd1), (gn2, gd2) = gr1, gr2
        n1, d1 = self._gray_to_int(gn1), self._gray_to_int(gd1)
        n2, d2 = self._gray_to_int(gn2), self._gray_to_int(gd2)
        numerator = (n1 * d2 + n2 * d1) & self.int_mask
        denominator = (d1 * d2) & self.int_mask
        gn = numerator ^ (numerator >> 1)
        gd = denominator ^ (denominator >> 1)
        return (gn, gd)

    def rational_mul(self, gr1, gr2):
        (gn1, gd1), (gn2, gd2) = gr1, gr2
        n1, d1 = self._gray_to_int(gn1), self._gray_to_int(gd1)
        n2, d2 = self._gray_to_int(gn2), self._gray_to_int(gd2)
        numerator = (n1 * n2) & self.int_mask
        denominator = (d1 * d2) & self.int_mask
        gn = numerator ^ (numerator >> 1)
        gd = denominator ^ (denominator >> 1)
        return (gn, gd)

    # ──────────────────────────────
    # Gray decode
    # ──────────────────────────────
    def _gray_to_int(self, g):
        n = g
        shift = 1
        while (g >> shift) > 0:
            n ^= (g >> shift)
            shift += 1
        return n
# Put this at the end of your bitops.py, or as a separate test script
from .bitops import BitTensor

def main():
    print("Testing BitTensor...")

    # Define limits (say, 8 bits)
    limits = [8]  # You can adapt this as a vector or shape, but BitTensor currently expects just flat
    N = 8

    # Create BitTensor: store 8 bits, all initialized to 0
    bt = BitTensor(limits, encoding="binary")
    bt.integer_pieces = [0]*N  # ensure size matches

    # Set bits individually and verify
    print("Setting bits...")
    for i in range(N):
        # set to 1 if i is even, else 0
        bt.integer_pieces[i] = int(i % 2 == 0)

    print("BitTensor integer_pieces:", bt.integer_pieces)

    # Retrieve bits via slice
    get_slice = [0]*4
    bt.__getitem__((2, 6), get_slice)
    print("Slice [2:6] via __getitem__:", get_slice)  # should show pattern from above

    # Set a slice of bits
    new_bits = [1, 1, 1, 1]
    bt.__setitem__((4, 8), new_bits)
    print("BitTensor after __setitem__ [4:8] to all 1s:", bt.integer_pieces)

    # Pack the entire bit tensor to an integer
    packed = 0
    for i, bit in enumerate(bt.integer_pieces):
        packed |= (bit & 1) << i
    print(f"Packed bits as integer: {packed:#x}")

    # Unpack into a new BitTensor
    unpacked = BitTensor(limits, encoding="binary")
    unpacked.integer_pieces = [((packed >> i) & 1) for i in range(N)]
    print("Unpacked integer_pieces:", unpacked.integer_pieces)

    # Check round-trip
    assert unpacked.integer_pieces == bt.integer_pieces, "Round-trip pack/unpack failed!"

    print("BitTensor storage test PASSED.")

if __name__ == "__main__":
    main()
import time

def print_status(bt, g, label):
    print(f"\n[{label}]")
    print(f"nodes={g.nodes} edges={g.edges} parents={bt.parent_count} children={bt.child_count}")
    print(f"normal_n={bt.n_start} normal_e={bt.e_start} normal_p={bt.p_start} normal_c={bt.c_start}")
    print(f"envelope_domain={bt.envelope_domain} envelope_size={bt.envelope_size}")
    print(f"bit_width={bt.bit_width} encoding={bt.encoding}")

def main_graph_stress():
    print("=== BitTensorMemoryGraph RATIO/RESIZE/FAILURE TEST ===")

    size = 2**18  # Make this large for stress; adjust for your RAM
    bit_width = 7
    bt = BitTensorMemoryGraph(size=size, bit_width=bit_width, encoding="gray")

    print_status(bt, bt.G, "Initial")

    # Add GrayTableOps table generation and demonstration
    print("\n[GrayTableOps] Generating tables for bit_width =", bit_width)
    gray_tables = GrayTableOps(int_bit_width=bit_width)
    print("[GrayTableOps] Example: gray_add_table[0][1] =", gray_tables.gray_add_table[0][1])
    print("[GrayTableOps] Example: gray_mul_table[0][2] =", gray_tables.gray_mul_table[0][2])

    # Try sweeping N line to various values and test boundary/failure
    for dest in [2, 10, 100, 1000, 5000, 10000]:
        print(f"\nSweeping LINE_N to {dest}")
        t0 = time.perf_counter()
        bt.sweep_memory(bt.LINE_N, dest, rational=True)
        t1 = time.perf_counter()
        print_status(bt, bt.G, f"After LINE_N={dest}")
        print(f"Time taken: {t1-t0:.6f} sec")
    
    print("\nContracting LINE_N back to 1")
    bt.sweep_memory(bt.LINE_N, 1, rational=True)
    print_status(bt, bt.G, "After contraction to 1")
    
    print("\nAttempting illegal expansion (beyond envelope)")
    bt.sweep_memory(bt.LINE_N, bt.envelope_domain[1] + 1000, rational=False)
    
    print("\nRapidly cycling all axes")
    axes = [bt.LINE_N, bt.LINE_E, bt.LINE_P, bt.LINE_C]
    for step in range(5, 100, 20):
        for ax in axes:
            t0 = time.perf_counter()
            bt.sweep_memory(ax, step, rational=True)
            t1 = time.perf_counter()
            print(f"Swept axis {ax} to {step} in {t1-t0:.6f} sec")
    
    print("\n=== BitTensorMemoryGraph TEST COMPLETE ===")


    src = """
def foo(x):
    y = x + 1
    if y > 0:
        z = y * 2
    else:
        z = -y
    return z
"""
    print("==== Generic Tree->DAG on foo ====")
    tree = ast.parse(src)
    dag = generic_tree_to_dag(tree, ast_children)
    for node in dag.nodes:
        print(node)
    print("Edges:", dag.edges)
    return
    print("\n==== ProcessGraph from file ====")
    # If you want to build from the file, use bitops.py or any file
    from ..transmogrifier.graph.graph_express2 import ProcessGraph
    pg = ProcessGraph()
    pg.build_from_ast('./bitops.py')  # assumes relative path
    pg.finalize_graph_with_outputs()
    print("[ProcessGraph] Nodes:", pg.G.nodes)
    pg.compute_levels("asap")
    pg.print_lifespans_ascii()

# ────────────────────────────────────────────────────────────────
# BIG-GRAPH SHUFFLE / MEMORY-LAYOUT STRESS
# ────────────────────────────────────────────────────────────────
def big_graph_shuffle_test():
    import numpy as np                     # only for RNG convenience
    from random import randint, sample, choice
    print("\n=== BIG-GRAPH SHUFFLE TEST ===")

    SIZE        = 2**20                    # 1 MiB envelope
    N_NODES     = 400                      # how many NodeEntry structs
    AVG_DEGREE  = 4                        # edges per node on average
    BIT_WIDTH   = 8

    bt = BitTensorMemoryGraph(size=SIZE,
                              bit_width=BIT_WIDTH,
                              encoding="gray")
    # ---------- 1 .  build node table ---------------------------------
    node_stride = ctypes.sizeof(NodeEntry)
    node_offsets = list(range(bt.n_start,
                              bt.e_start,
                              node_stride))


    node_offsets = node_offsets[:N_NODES]
    rng          = np.random.default_rng()

    for off in node_offsets:
        n             = NodeEntry()
        n.node_id     = off                      # cheap unique ID = address
        n.node_type   = randint(0, 5)
        n.node_depth  = 1
        n.bit_width   = BIT_WIDTH
        blob          = GraphSearch._build_struct_bytes(n)
        bt.hard_memory.write(off, blob)

    bt.node_count = N_NODES

    # ---------- 2 .  build edge table ---------------------------------
    edge_stride = ctypes.sizeof(EdgeEntry)
    edge_offsets = list(range(bt.e_start,
                              bt.p_start,
                              edge_stride))

    N_EDGES = N_NODES * AVG_DEGREE // 2
    

    edge_offsets = edge_offsets[:N_EDGES]

    for off in edge_offsets:
        src_ptr = choice(node_offsets)
        dst_ptr = choice(node_offsets)
        while dst_ptr == src_ptr:                 # no self-loops
            dst_ptr = choice(node_offsets)

        e               = EdgeEntry()
        e.src_ptr       = src_ptr
        e.dst_ptr       = dst_ptr
        e.src_graph_id  = bt.capsid_id
        e.dst_graph_id  = bt.capsid_id
        e.data_type     = 0
        e.edge_flags    = 0
        blob            = GraphSearch._build_struct_bytes(e)
        bt.hard_memory.write(off, blob)

    bt.edge_count = N_EDGES

    print(f"   » Populated {N_NODES} nodes and {N_EDGES} edges")

    # ---------- 3 .  initial sweep & edge plan ------------------------
    print("\n── Sweep N-region hard left")
    bt.sweep_memory(bt.LINE_N,
                    bt.header_size + 64,          # almost flush with header
                    rational=False)

    print("\n── Sweep N-region hard right")
    bt.sweep_memory(bt.LINE_N,
                    bt.e_start - node_stride*4,   # leave a little gap
                    rational=False)

    # pick a handful of nodes and try to relocate them
    movers = sample(node_offsets, k=20)
    holes  = bt.empty_set(movers)[:20]

    print("\n── Edge-relocation planning on 20 nodes")
    assign, delta = bt.plan_edge_relocations(movers, holes)
    print("   assignment :", assign)
    print("   total Δdist:", delta)

    bt.relocate_edges(movers, holes, assign)

    # ---------- 4 .  sanity/invariant report --------------------------
    print("\n── Final invariants")
    print(f"node_count  = {bt.node_count}")
    print(f"edge_count  = {bt.edge_count}")
    print(f"bitmap pop  = {sum(bt.hard_memory.bitmap_expanded_vector())}")
    print("density per chunk (first 15):",
          bt.hard_memory.density[:15])
    print("\n=== BIG-GRAPH SHUFFLE TEST COMPLETE ===")


if __name__ == "__main__":
    main_graph_stress()
    big_graph_shuffle_test()