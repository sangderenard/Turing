##########################################################################
# Static Single Assignment (SSA) Intermediate Representation (IR) Syntax #
##########################################################################
#
# In SSA form:
#  - Each value is assigned exactly once.
#  - Values are named `%t<ID>`, where <ID> is a unique integer.
#    Internally, we store the integer `id` and reconstruct the textual
#    `%t{id}` when needed.
#  - Instructions (`Instr`) consist of:
#      * `op`: operation name (e.g. 'Add', 'Mul', 'Pow')
#      * `args`: list of input SSA values
#      * `res`: the result SSA value produced
#  - SSA enables clear dataflow analysis, liveness tracking,
#    and optimization passes (constant folding, dead-code elimination,
#    instruction scheduling, etc.).
#
# Example SSA sequence:
#    %t1 = Add %x, %y
#    %t2 = Mul %t1, %c
#    %t3 = Pow %t2, 2
#
# This file provides the core data structures for holding SSA in memory,
# using integer IDs for efficiency.
##########################################################################

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Union
from enum import Enum

# -----------------------------------------------------------------------------
# Core SSA Data Structures
# -----------------------------------------------------------------------------
@dataclass
class SSAValue:
    """
    Represents a single SSA value.

    Attributes:
        id (int):
            Unique integer identifier (maps to textual `%t{id}`).
        dtype (Optional[str]):
            Optional type annotation (e.g. 'float32', 'int64').
    """
    id: int
    dtype: Optional[str] = None

    def name(self) -> str:
        """Return the textual SSA name in `%t<ID>` form."""
        return f"%t{self.id}"


@dataclass
class Instr:
    """
    Represents a single SSA instruction in a linear sequence.

    Attributes:
        op (str):
            Operation name (e.g. 'Add', 'Mul', 'Pow').
        args (List[SSAValue]):
            Operands for the operation.
        res (SSAValue):
            The result value produced by this instruction.
    """
    op: str
    args: List[SSAValue]
    res: SSAValue

from enum import Enum

from .ssa_registry import Handler, sympy_ssa_name_map, sympy_ssa_disambig, SSARegistry

@dataclass
class BasicBlock:
    name: str
    instrs: List[Instr] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)  # for CFG edges

@dataclass
class Function:
    name: str
    args: List[SSAValue]
    blocks: Dict[str, BasicBlock]

@dataclass
class IRModule:
    functions: Dict[str, Function]

# -----------------------------------------------------------------------------
# Correlator for Language <-> SSA Operation Mappings
# -----------------------------------------------------------------------------
class Correlator:
    """
    Bidirectional mapping between language-specific operator names
    and SSA `Handler` enum values.
    """
    def __init__(self):
        # language -> (lang_op_name -> Handler)
        self._lang_to_ssa: Dict[str, Dict[str, Handler]] = {}
        # language -> (Handler -> lang_op_name)
        self._ssa_to_lang: Dict[str, Dict[Handler, str]] = {}

    def register(self, language: str, mapping: Dict[str, Handler]) -> None:
        """
        Register a mapping for a specific language.

        Args:
            language: Identifier for the language (e.g. 'python', 'sympy').
            mapping: Dict of language operator names to `Handler` values.
        """
        lang_map = self._lang_to_ssa.setdefault(language, {})
        ssa_map = self._ssa_to_lang.setdefault(language, {})
        for lang_op, handler in mapping.items():
            lang_map[lang_op] = handler
            ssa_map[handler] = lang_op

    def to_ssa(self, language: str, lang_op: str) -> Optional[Handler]:
        """
        Convert a language-specific operator name to an SSA Handler.
        """
        return self._lang_to_ssa.get(language, {}).get(lang_op)

    def from_ssa(self, language: str, handler: Handler) -> Optional[str]:
        """
        Convert an SSA Handler to its language-specific operator name.
        """
        return self._ssa_to_lang.get(language, {}).get(handler)

# -----------------------------------------------------------------------------
# Pre-SSA Universal IR Schema, Signature, and Handler Registry
# -----------------------------------------------------------------------------
# This module defines the canonical data structures for the compiler’s
# pre-SSA stage and builds a registry of operator definitions.

try:  # heavy optional dependency
    from .operator_defs import operator_signatures, role_schemas, default_funcs
except Exception:  # pragma: no cover - optional
    operator_signatures = {}
    role_schemas = {}
    default_funcs = {}

@dataclass
class RoleSchema:
    """
    Defines how an operation wires its inputs ('up') and outputs ('down').
    """
    up: Dict[str, Union[int, str]]
    down: Dict[str, Union[int, str]]

@dataclass
class Signature:
    """
    Describes operation I/O counts and execution parameters.
    """
    min_inputs: int
    max_inputs: Optional[int]
    min_outputs: int
    max_outputs: Optional[int]
    concurrency: Optional[int]
    allows_inplace: bool
    parameters: List[str] = field(default_factory=list)

@dataclass
class OperatorDef:
    """
    Complete definition of an operation at pre-SSA stage.
    """
    name: str
    role_schema: RoleSchema
    signature: Signature
    handler: Optional[Callable]
    earliest_page: Optional[int] = None
    latest_page: Optional[int] = None

# Central registry: op-name -> OperatorDef
operator_definitions: Dict[str, OperatorDef] = {}

# Build the registry from existing definitions
for op_name, sig_dict in operator_signatures.items():
    schema_dict = role_schemas.get(op_name, {})
    schema = RoleSchema(
        up=schema_dict.get('up', {}),
        down=schema_dict.get('down', {})
    )
    # Prepare signature parameters
    sig_params = dict(sig_dict)
    sig_params.setdefault('parameters', [])
    signature = Signature(
        min_inputs=sig_params['min_inputs'],
        max_inputs=sig_params.get('max_inputs'),
        min_outputs=sig_params['min_outputs'],
        max_outputs=sig_params.get('max_outputs'),
        concurrency=sig_params.get('concurrency'),
        allows_inplace=sig_params.get('allows_inplace', False),
        parameters=sig_params.get('parameters', [])
    )
    handler = default_funcs.get(op_name)
    operator_definitions[op_name] = OperatorDef(
        name=op_name,
        role_schema=schema,
        signature=signature,
        handler=handler
    )

# -----------------------------------------------------------------------------
# Sympy <-> SSA Translation Correlator
# -----------------------------------------------------------------------------
class SympyToSSA:
    """
    Handles translation between SymPy node types and SSA handlers,
    including argument arrangement and schema/signature mapping.

    Attributes:
        name_map (Dict[str, Handler]):
            Maps SymPy node class names to SSA Handler enums.
        arg_order (Dict[str, List[str]]):
            For each SymPy node, specifies the order of argument names
            to extract from node properties for SSA arguments.
        schema_map (Dict[str, RoleSchema]):
            Maps SymPy node names to RoleSchema for argument directionality.
        signature_map (Dict[str, Signature]):
            Maps SymPy node names to Signature for I/O constraints.
        handler_map (Dict[str, Callable]):
            Maps SymPy node names to handler functions (if any).
    """
    def __init__(
        self,
        name_map: Dict[str, Handler],
        arg_order: Optional[Dict[str, List[str]]] = None,
        schema_map: Optional[Dict[str, 'RoleSchema']] = None,
        signature_map: Optional[Dict[str, 'Signature']] = None,
        handler_map: Optional[Dict[str, Callable]] = None,
    ):
        self.name_map = name_map
        self.arg_order = arg_order or {}
        self.schema_map = schema_map or {}
        self.signature_map = signature_map or {}
        self.handler_map = handler_map or {}

    def get_handler(self, sympy_node_name: str) -> Optional[Handler]:
        """Get the SSA Handler for a given SymPy node name."""
        return self.name_map.get(sympy_node_name)

    def get_arg_order(self, sympy_node_name: str) -> List[str]:
        """
        Get the argument extraction order for a SymPy node.
        Returns an empty list if not specified.
        """
        return self.arg_order.get(sympy_node_name, [])

    def get_schema(self, sympy_node_name: str) -> Optional['RoleSchema']:
        """Get the RoleSchema for a SymPy node."""
        return self.schema_map.get(sympy_node_name)

    def get_signature(self, sympy_node_name: str) -> Optional['Signature']:
        """Get the Signature for a SymPy node."""
        return self.signature_map.get(sympy_node_name)

    def get_handler_func(self, sympy_node_name: str) -> Optional[Callable]:
        """Get the handler function for a SymPy node."""
        return self.handler_map.get(sympy_node_name)

    def to_ssa_instr(self, node) -> Optional[Instr]:
        """
        Convert a SymPy node to an SSA Instr, using the mapping and argument arrangement.
        This is a stub; actual implementation depends on node structure.
        """
        handler = self.get_handler(type(node).__name__)
        if handler is None:
            return None
        # Extract arguments in the specified order, or default to node.args
        arg_names = self.get_arg_order(type(node).__name__)
        if arg_names:
            args = [getattr(node, name) for name in arg_names]
        else:
            args = getattr(node, 'args', [])
        # SSAValue wrapping and result assignment would be handled by the caller
        # Return a partially constructed Instr for demonstration
        return Instr(op=str(handler), args=args, res=None)  # res to be filled in by SSA builder



class CaseInsensitiveSympyToSSA(SympyToSSA):
    """
    Case-insensitive version of SympyToSSA for handling SymPy node names.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert all keys in name_map to lowercase for case-insensitivity
        self.name_map = {k.lower(): v for k, v in self.name_map.items()}

    def get_handler(self, sympy_node_name: str) -> Optional[Handler]:
        """Get the SSA Handler for a given SymPy node name (case-insensitive)."""
        return super().get_handler(sympy_node_name.lower())
import random
import sys
import json
import argparse
from .ssa_registry import Handler, SSARegistry

# Paths to persist dynamically generated helpers and structured metadata
HELPER_LOG_FILE = "generated_ssa_helpers.py"
METADATA_FILE = "ssa_helper_metadata.json"

# Ensure helper file exists
try:
    with open(HELPER_LOG_FILE, "x") as f:
        f.write("# Auto-generated SSA helper functions\n\n")
except FileExistsError:
    pass

# Ensure metadata file exists
try:
    with open(METADATA_FILE, "x") as f:
        json.dump({}, f, indent=2)
except FileExistsError:
    pass

class DummyBuilder:
    def __init__(self):
        self.instructions = []
    def record(self, instr):
        self.instructions.append(instr)
    def fresh(self, dtype=None):
        val = SSARegistry.new_value(dtype=dtype)
        return val

# Random command compounding: pick a random handler with dummy operands
def command_compound():
    handler = random.choice(list(Handler))
    arg_count = random.randint(1, 3)
    operands = [f"op{i}" for i in range(arg_count)]
    return handler, operands

import ast

def dump_ast_structure(node, indent=0):
    prefix = '  ' * indent
    node_type = type(node).__name__
    print(f"{prefix}{node_type}")
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            print(f"{prefix}  {field}: [")
            for item in value:
                if isinstance(item, ast.AST):
                    dump_ast_structure(item, indent + 2)
                else:
                    print(f"{prefix}    {repr(item)}")
            print(f"{prefix}  ]")
        elif isinstance(value, ast.AST):
            print(f"{prefix}  {field}:")
            dump_ast_structure(value, indent + 2)
        else:
            print(f"{prefix}  {field}: {repr(value)}")


# Prompt helper to gather comprehensive metadata for future-proof, AST-aware conversion
def prompt_metadata(handler):
    print(f"=== Define metadata for handler: {handler.name} ===", file=sys.stderr)
    description = input("1) One-line description of this handler's semantics: ").strip()
    num_args = input("2) Number of positional operands: ").strip()

    # Keyword args schema
    print("3) Define keyword arguments (format: name:type:description), one per line; empty line to finish:")
    kwargs_schema = {}
    while True:
        line = input().strip()
        if not line:
            break
        name, typ, desc = [p.strip() for p in line.split(":", 2)]
        kwargs_schema[name] = {"type": typ, "description": desc}

    # AST node
    ast_node = input("4) Corresponding AST node class (e.g. ast.Cast), or leave blank: ").strip() or None

    # AST mappings
    print("5) Define AST field mappings (format: name=expression), one per line; empty line to finish:")
    ast_mapping = {}
    while True:
        line = input().strip()
        if not line:
            break
        key, expr = [p.strip() for p in line.split("=", 1)]
        ast_mapping[key] = expr

    return_dtype = input("6) Return dtype (e.g. 'int32', 'float64'): ").strip() or None
    usage_example = input("7) Provide a usage example (code snippet) for this handler: ").strip()
    print("", file=sys.stderr)

    # Load and update structured metadata
    with open(METADATA_FILE, "r+") as mf:
        data = json.load(mf)
        data.setdefault(handler.name, {})
        data[handler.name].update({
            "description": description,
            "num_args": int(num_args),
            "kwargs": kwargs_schema,
            "ast_node": ast_node,
            "ast_mapping": ast_mapping,
            "return_dtype": return_dtype,
            "usage_example": usage_example
        })
        mf.seek(0)
        json.dump(data, mf, indent=2)
        mf.truncate()
    print(f"[INFO] Structured metadata saved for {handler.name}", file=sys.stderr)
    return data[handler.name]

# Safe emit: prompt interactively if helper missing, plus structured metadata
def safe_emit_ssa(handler, builder, operands, **kwargs):
    try:
        return SSARegistry.emit_ssa(handler, builder, operands, **kwargs)
    except KeyError:
        print(f"[WARN] Missing SSA helper for handler: {handler.name}", file=sys.stderr)
        meta = prompt_metadata(handler)
        print("Enter the function body. Use args: builder, operands, **kwargs. End with an empty line.", file=sys.stderr)
        lines = []
        while True:
            line = sys.stdin.readline()
            if not line or not line.strip():
                break
            lines.append(line.rstrip("\n"))
        func_name = f"ssa_helper_{handler.name.lower()}"
        func_lines = [f"def {func_name}(builder, operands, **kwargs):",
                      f"    \"\"\"{meta['description']}\"\"\""]
        for ln in lines:
            func_lines.append(f"    {ln}")
        func_src = "\n".join(func_lines) + "\n"
        with open(HELPER_LOG_FILE, "a") as logf:
            logf.write(func_src + "\n")
        print(f"[INFO] Logged helper to {HELPER_LOG_FILE}:", file=sys.stderr)
        print(func_src, file=sys.stderr)
        local_ns = {}
        exec(func_src, globals(), local_ns)
        SSARegistry.register_helper(handler)(local_ns[func_name])
        print(f"[INFO] Registered new helper for {handler.name}", file=sys.stderr)
        return SSARegistry.emit_ssa(handler, builder, operands, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSA helper tester with prioritized handlers and usage examples.")
    parser.add_argument('-o', '--operators', nargs='+', default=[],
                        help='List of handler names to include first in the test')
    parser.add_argument('-r', '--repeat', type=int, default=100,
                        help='Number of random commands to run')
    args = parser.parse_args()

    builder = DummyBuilder()
    
    # First, run specified handlers
    if args.operators:  
        print(f"[INFO] Including specified handlers first: {args.operators}", file=sys.stderr)
        for name in args.operators:
            try:
                handler = Handler[name]
            except KeyError:
                print(f"[ERROR] Unknown handler name: {name}", file=sys.stderr)
                continue
            arg_count = random.randint(1, 3)
            operands = [f"op{i}" for i in range(arg_count)]
            result = safe_emit_ssa(handler, builder, operands)
            print(f"Emitted SSA for {handler.name}: {result}")

    # Then, random commands
    for _ in range(args.repeat):
        handler, operands = command_compound()
        try:
            result = safe_emit_ssa(handler, builder, operands)
            print(f"Emitted SSA for {handler.name}: {result}")
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            continue

