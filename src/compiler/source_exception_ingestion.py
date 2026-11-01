"""Ingest exception syntax into explicit record operations and control arms.

The temporary AST vocabulary here is consumed by ordinary ProcessGraph ingestion.
No synthetic execution coordinates survive the conversion. Generated operators
become SOURCE_PROCESS_OPERATION / SOURCE_EXCEPTION_OPERATION and placed control
blocks, with exact source conversion provenance. Division keeps its own SSA arms.
"""
import ast
import builtins
import copy
from dataclasses import dataclass


def _copy_source(value):
    """Copy syntax/operation records, retain readonly pursuit binding maps.

    Once pursuit has resolved imports, FunctionDef metadata may name modules.
    Those maps are resolution context, not syntax or mutable exception state.
    The conversion must not pickle modules while proposing an AST rewrite.
    """
    memo = {}
    roots = (value,) if isinstance(value, ast.AST) else value if isinstance(value, (list, tuple)) else ()
    for root in roots:
        if not isinstance(root, ast.AST):
            continue
        for node in ast.walk(root):
            bindings = getattr(node, '_python_bindings', None)
            if isinstance(bindings, dict):
                memo[id(bindings)] = bindings
    return copy.deepcopy(value, memo)


@dataclass
class Context:
    error: str
    caught: str | None = None
    protected: bool = False


class FailureArms:
    def __init__(self, occupied):
        self.occupied = occupied
        self.serial = 0
        self.checks = 0
        self.generated = []
        self.types = {}
        self.return_flag = None
        self.return_value = None
        self.exception_imports = {}
        self.external_shadowed = set()
        self.bound = set()

    def value_type(self, node):
        if node is None: return type(None)
        if isinstance(node, ast.Constant): return type(node.value)
        if isinstance(node, ast.Name): return self.types.get(node.id)
        if isinstance(node, ast.Compare): return bool
        if isinstance(node, ast.UnaryOp): return bool if isinstance(node.op, ast.Not) else self.value_type(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = self.value_type(node.left), self.value_type(node.right)
            if left in {int, bool, float} and right in {int, bool, float}:
                return float if isinstance(node.op, ast.Div) or float in {left, right} else int
        if isinstance(node, (ast.List, ast.Tuple)):
            members = {self.value_type(item) for item in node.elts}
            return (type(node), next(iter(members))) if len(members) == 1 else None
        if isinstance(node, ast.Subscript):
            value = self.value_type(node.value)
            return value[1] if isinstance(value, tuple) else None
        if isinstance(node, ast.Call):
            operation = getattr(node, '_turing_dispatch_operation', None)
            if operation is not None:
                from .thread_operations import PROCESS_OPERATORS, PROCESS_RESULT_DTYPES
                dtype = PROCESS_RESULT_DTYPES.get(PROCESS_OPERATORS.get(operation['operation']))
                if dtype is not None:
                    return {'int64': int, 'bool': bool, 'none': type(None), 'opaque_ref': 'opaque_ref'}.get(dtype)
            if isinstance(node.func, ast.Name):
                basic = {"int": int, "float": float, "bool": bool, "len": int}
                return basic.get(node.func.id, self.call_types.get(node.func.id))
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "math":
                return float
        return None

    @staticmethod
    def filler(kind):
        if kind in {'opaque_ref', type(None)}:
            return ast.Constant(None)
        if kind not in {int, bool, float}:
            raise ValueError("native try: temporary has no declared basic scalar result type")
        return ast.Constant(False if kind is bool else 0.0 if kind is float else 0)

    def name(self):
        while True:
            self.serial += 1
            candidate = f"__turing_exception_{self.serial}"
            if candidate not in self.occupied:
                self.occupied.add(candidate)
                return candidate

    @staticmethod
    def load(name):
        return ast.Name(id=name, ctx=ast.Load())

    @staticmethod
    def assign(name, value):
        return ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value)

    def intrinsic(self, operation, args=(), **payload):
        node = ast.Call(func=self.load(f"__turing_{operation}"), args=list(args), keywords=[])
        node._turing_dispatch_operation = {
            "operation": operation, "source_identity": f"turing.{operation}",
            "required_capability": "native_exception_records", "control_scopes": (),
            **payload,
        }
        node._extraction_contract = {
            "action": "intrinsic", "identity": f"turing.{operation}",
            "rule_id": "source-exception-failure-arms", "classification": "exception-operation",
            "parameters": {"lowering_namespace": "exceptions", "operation": operation},
        }
        self.generated.append(node)
        return node

    def active(self, context):
        return self.intrinsic("exception_active", (self.load(context.error),))

    def guard(self, context, statements):
        predicate = ast.UnaryOp(op=ast.Not(), operand=self.active(context))
        if context.protected and self.return_flag is not None:
            predicate = ast.BoolOp(op=ast.And(), values=[predicate, ast.UnaryOp(op=ast.Not(), operand=self.load(self.return_flag))])
        guard = ast.If(test=predicate, body=statements or [ast.Pass()], orelse=[])
        guard._turing_exception_guard = (context.error, context.protected, self.return_flag)
        return guard

    def failure(self, context, predicate, exception_type, message=None):
        self.checks += 1
        return self.guard(context, [ast.If(test=predicate, body=[self.assign(context.error,
            self.intrinsic("exception_make", exception_type=exception_type,
                           exception_message=message, error_code=100))], orelse=[])])

    def temporary(self, context, value, *, checked=True):
        name = self.name()
        kind = self.value_type(value)
        self.types[name] = kind
        body = [self.assign(name, value)]
        if checked:
            self.checks += 1
            body.append(self.assign(context.error, self.intrinsic("exception_take")))
        return [self.assign(name, self.filler(kind)), self.guard(context, body)], self.load(name)

    def index(self, context, base, index):
        pre, value = self.expression(base, context)
        more, position = self.expression(index, context)
        pre += more
        # A typed Python subscript does not coerce a float into an index.
        pre.append(self.failure(context, ast.UnaryOp(op=ast.Not(), operand=self.intrinsic(
            "exception_index_type", (_copy_source(position),))),
            "builtins.TypeError", "sequence index must be an integer"))
        count_name = self.name()
        self.types[count_name] = int
        pre.append(self.assign(count_name, ast.Constant(0)))
        pre.append(self.guard(context, [self.assign(count_name, self.intrinsic(
            "exception_length", (_copy_source(value),)))]))
        outside = ast.BoolOp(op=ast.Or(), values=[
            ast.Compare(left=_copy_source(position), ops=[ast.Lt()], comparators=[ast.UnaryOp(op=ast.USub(), operand=self.load(count_name))]),
            ast.Compare(left=_copy_source(position), ops=[ast.GtE()], comparators=[self.load(count_name)]),
        ])
        pre.append(self.failure(context, outside, "builtins.IndexError", "sequence index out of range"))
        return pre, ast.Subscript(value=value, slice=position, ctx=ast.Load())

    def expression(self, node, context, *, discard=False):
        if node is None or isinstance(node, (ast.Name, ast.Constant)):
            return [], _copy_source(node)
        if isinstance(node, ast.BinOp):
            pre, left = self.expression(node.left, context)
            more, right = self.expression(node.right, context)
            pre += more
            # Division defines its own configurable SSA failure arms. Try
            # supplies the record/control destination, never a divisor filter.
            if isinstance(node.op, (ast.LShift, ast.RShift)):
                pre.append(self.failure(context, ast.Compare(left=_copy_source(right), ops=[ast.Lt()], comparators=[ast.Constant(0)]),
                                        "builtins.ValueError", "negative shift count"))
            body, value = self.temporary(context, ast.BinOp(left=left, op=_copy_source(node.op), right=right))
            return pre + body, value
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Slice):
                raise ValueError("native try: slice-result failure arms are not lowered")
            pre, operation = self.index(context, node.value, node.slice)
            body, value = self.temporary(context, operation)
            return pre + body, value
        if isinstance(node, ast.UnaryOp):
            pre, value = self.expression(node.operand, context)
            body, result = self.temporary(context, ast.UnaryOp(op=_copy_source(node.op), operand=value),
                                          checked=not isinstance(node.op, ast.Not))
            return pre + body, result
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            pre, left = self.expression(node.left, context)
            more, right = self.expression(node.comparators[0], context)
            body, result = self.temporary(context, ast.Compare(left=left, ops=_copy_source(node.ops), comparators=[right]))
            return pre + more + body, result
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in self.call_types:
                raise ValueError("native try: ordinary call requires an explicit failure-return contract")
            if isinstance(node.func, ast.Attribute):
                pre, receiver = self.expression(node.func.value, context)
                func = ast.Attribute(value=receiver, attr=node.func.attr, ctx=ast.Load())
            elif isinstance(node.func, ast.Name):
                pre, func = [], _copy_source(node.func)
            else:
                raise ValueError("native try: dynamic callable failure contract is not declared")
            arguments, keywords = [], []
            for argument in node.args:
                more, value = self.expression(argument, context)
                pre += more; arguments.append(value)
            for keyword in node.keywords:
                more, value = self.expression(keyword.value, context)
                pre += more; keywords.append(ast.keyword(arg=keyword.arg, value=value))
            # Keep resolved ingestion operation/receipt identity while
            # substituting evaluated operands. Rebuilding a bare ast.Call
            # resurrected the synthetic Python callee after typed conversion.
            call = copy.copy(node)
            call.func, call.args, call.keywords = func, arguments, keywords
            # An expression statement discards its result. It still evaluates
            # the call and captures its failure, but inventing a scalar
            # temporary for a None/opaque result imposed a nonexistent ABI.
            # This applies to every discarded primitive call, not method names.
            if discard:
                self.checks += 1
                body = [ast.Expr(value=call), self.assign(context.error, self.intrinsic("exception_take"))]
                return pre + [self.guard(context, body)], None
            body, result = self.temporary(context, call)
            return pre + body, result
        if isinstance(node, ast.Attribute):
            pre, value = self.expression(node.value, context)
            body, result = self.temporary(context, ast.Attribute(value=value, attr=node.attr, ctx=ast.Load()))
            return pre + body, result
        if isinstance(node, (ast.List, ast.Tuple)):
            pre, elements = [], []
            for element in node.elts:
                more, value = self.expression(element, context)
                pre += more; elements.append(value)
            return pre, type(node)(elts=elements, ctx=ast.Load())
        raise ValueError(f"native try: {type(node).__name__} needs an explicit operator failure contract")

    def exception_identity(self, node, occupied):
        if isinstance(node, ast.Name) and node.id not in self.external_shadowed:
            if self.exception_imports.get(node.id) == 'threading.BrokenBarrierError':
                return 'threading.BrokenBarrierError'
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id not in self.external_shadowed
                and self.exception_imports.get(node.value.id) == 'threading'
                and node.attr == 'BrokenBarrierError'):
            return 'threading.BrokenBarrierError'
        if not isinstance(node, ast.Name) or node.id in occupied:
            raise ValueError("native try: exception class requires an unshadowed builtin declaration")
        value = getattr(builtins, node.id, None)
        if not isinstance(value, type) or not issubclass(value, BaseException):
            raise ValueError("native try: exception class is not a declared builtin exception")
        if node.id not in {"BaseException", "Exception", "ArithmeticError", "ZeroDivisionError",
                           "OverflowError", "FloatingPointError", "LookupError", "IndexError",
                           "KeyError", "TypeError", "ValueError", "RuntimeError"}:
            raise ValueError("native try: builtin exception hierarchy is not lowered for " + node.id)
        return f"builtins.{node.id}"

    def statements(self, statements, context):
        result = []
        tail = result
        guard_key = (context.error, context.protected, self.return_flag)
        # A local publication needs a successful lexical continuation. A
        # sequence of discarded effects has no new data environment to carry
        # and keeps the existing flat error/handler topology.
        lexical_continuation = any(
            isinstance(target, ast.Name) and target.id not in self.bound
            for statement in statements if isinstance(statement, (ast.Assign, ast.AnnAssign))
            for target in (statement.targets if isinstance(statement, ast.Assign) else (statement.target,)))
        for statement in statements:
            lowered = self.statement(statement, context)
            for node in lowered:
                ast.copy_location(node, statement)
                for descendant in ast.walk(node):
                    if isinstance(descendant, (ast.expr, ast.stmt)) and not hasattr(descendant, "lineno"):
                        ast.copy_location(descendant, statement)
            for node in lowered:
                tail.append(node)
                if lexical_continuation and getattr(node, '_turing_exception_guard', None) == guard_key:
                    # Keep the successful continuation inside its actual
                    # error guard. Separate sibling guards previously lost
                    # newly bound locals (e.g. a Barrier handle) at a merge,
                    # then resolved the later read to a static Python class.
                    # No placeholder value is published into the authored
                    # environment and no relational dominance guess is needed.
                    tail = node.body
        return result

    def raised_record(self, node, context, *, retain_message_operand=False):
        """One exception identity/payload writer for protected and bare raises.

        Compiler source has guards such as interchange_reduction_loops' plain
        ValueError. These must become graph operations even without a try.
        Dynamic messages remain exact graph operands, never cached AST payloads;
        native materialization still needs their owned-message ABI.
        """
        if node.cause is not None:
            raise ValueError("native try: explicit exception chaining is not lowered")
        if node.exc is None:
            if context.caught is None:
                raise ValueError("native try: bare raise has no active handler")
            return self.load(context.caught)
        exception = node.exc
        cls = exception.func if isinstance(exception, ast.Call) else exception
        identity = self.exception_identity(cls, self.shadowed)
        message = None
        operands = ()
        if isinstance(exception, ast.Call):
            if exception.keywords or len(exception.args) > 1:
                raise ValueError("native try: dynamic exception payload ownership is not lowered")
            if exception.args:
                if isinstance(exception.args[0], ast.Constant):
                    message = str(exception.args[0].value)
                elif retain_message_operand:
                    operands = (_copy_source(exception.args[0]),)
                else:
                    raise ValueError("native try: dynamic exception payload ownership is not lowered")
        return self.intrinsic("exception_make", operands, exception_type=identity,
                              exception_message=message, error_code=100)

    def statement(self, node, context):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom, ast.Pass)):
            return [node]
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is None:
                return [node]
            pre, value = self.expression(node.value, context)
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            if len(targets) != 1:
                raise ValueError("native try: chained assignment needs an explicit publication contract")
            target = targets[0]
            if isinstance(target, ast.Subscript):
                more, selected = self.index(context, target.value, target.slice)
                pre += more
                selected.ctx = ast.Store(); target = selected
            elif not isinstance(target, ast.Name):
                raise ValueError("native try: record-field publication failure contract is not lowered")
            assigned = ast.Assign(targets=[_copy_source(target)], value=value)
            if isinstance(target, ast.Name):
                self.types[target.id] = self.value_type(value)
                self.bound.add(target.id)
            return [*pre, self.guard(context, [assigned])]
        if isinstance(node, ast.AugAssign):
            read = _copy_source(node.target)
            for item in ast.walk(read):
                if hasattr(item, "ctx"): item.ctx = ast.Load()
            return self.statement(ast.Assign(targets=[node.target], value=ast.BinOp(left=read, op=node.op, right=node.value)), context)
        if isinstance(node, ast.Expr):
            pre, _value = self.expression(node.value, context, discard=True)
            return pre
        if isinstance(node, ast.Raise):
            return [self.guard(context, [self.assign(context.error, self.raised_record(node, context))])]
        if isinstance(node, ast.If):
            pre, predicate = self.expression(node.test, context)
            incoming = set(self.bound)
            yes = self.statements(node.body, context) or [ast.Pass()]
            yes_bound = set(self.bound)
            self.bound = set(incoming)
            no = self.statements(node.orelse, context)
            self.bound.intersection_update(yes_bound)
            branch = ast.If(test=predicate, body=yes, orelse=no)
            return [*pre, self.guard(context, [branch])]
        if isinstance(node, ast.For):
            iterator = node.iter
            if not (isinstance(node.target, ast.Name) and isinstance(iterator, ast.Call)
                    and isinstance(iterator.func, ast.Name) and iterator.func.id == "range"
                    and "range" not in self.shadowed and not iterator.keywords
                    and 1 <= len(iterator.args) <= 2
                    and all(self.value_type(arg) in {int, bool} for arg in iterator.args)):
                raise ValueError("native try: protected For requires the builtin integer range contract")
            self.types[node.target.id] = int
            incoming_bound = set(self.bound)
            self.bound.add(node.target.id)
            body = self.statements(node.body, context)
            self.bound = set(incoming_bound)
            stop = self.active(context)
            if self.return_flag is not None:
                stop = ast.BoolOp(op=ast.Or(), values=[stop, self.load(self.return_flag)])
            body.append(ast.If(test=stop, body=[ast.Break()], orelse=[]))
            loop = ast.For(target=_copy_source(node.target), iter=_copy_source(iterator),
                           body=body, orelse=self.statements(node.orelse, context),
                           type_comment=getattr(node, "type_comment", None))
            self.bound = incoming_bound
            return [self.guard(context, [loop])]
        if isinstance(node, ast.While):
            # Condition evaluation is itself protected and repeated. Keep
            # its operator records in the loop, with explicit completion
            # arms; an exhausted condition alone runs the authored else.
            incoming_bound = set(self.bound)
            pre, predicate = self.expression(node.test, context)
            def stopping():
                stop = self.active(context)
                if self.return_flag is not None:
                    stop = ast.BoolOp(op=ast.Or(), values=[stop, self.load(self.return_flag)])
                return ast.If(test=stop, body=[ast.Break()], orelse=[])
            exhausted = ast.If(test=ast.UnaryOp(op=ast.Not(), operand=predicate),
                               body=[*self.statements(node.orelse, context), ast.Break()], orelse=[])
            self.bound = set(incoming_bound)
            loop = ast.While(test=ast.Constant(True),
                body=[stopping(), *pre, stopping(), exhausted, *self.statements(node.body, context)], orelse=[])
            self.bound = incoming_bound
            return [self.guard(context, [loop])]
        if isinstance(node, (ast.Break, ast.Continue)):
            return [self.guard(context, [_copy_source(node)])]
        if isinstance(node, ast.Try):
            return self.try_statement(node, context)
        if isinstance(node, ast.With):
            # Leave acquisition/cleanup to the existing resource conversion.
            # Convert failures in the body under this handler context, then
            # capture a cleanup failure even when the body selected return.
            # Unknown context-manager types still fail resource admission.
            resource = _copy_source(node)
            resource.body = self.statements(node.body, context) or [ast.Pass()]
            exit_error = self.name()
            self.checks += 1
            exit_context = Context(exit_error)
            completion = [resource, self.assign(exit_error, self.intrinsic("exception_take")),
                          ast.If(test=self.active(exit_context),
                                 body=[self.assign(context.error, self.load(exit_error))], orelse=[])]
            return [self.guard(context, completion)]
        if isinstance(node, ast.Return):
            if context.protected:
                pre, value = self.expression(node.value, context)
                return [*pre, self.guard(context, [self.assign(self.return_value, value), self.assign(self.return_flag, ast.Constant(True))])]
            pre, value = self.expression(node.value, context)
            return [*pre, self.guard(context, [ast.Return(value=value)])]
        raise ValueError(f"native try: {type(node).__name__} cleanup/completion continuation is not lowered")

    def try_statement(self, node, parent):
        incoming_bound = set(self.bound)
        error = self.name()
        context = Context(error, parent.caught, True)
        body = [self.assign(error, self.intrinsic("exception_none")), *self.statements(node.body, context)]
        body_bound = set(self.bound)
        successful_bounds = [body_bound]
        succeeded = self.name()
        body.append(self.assign(succeeded, ast.UnaryOp(op=ast.Not(), operand=self.active(context))))
        selected = self.name()
        body.append(self.assign(selected, ast.Constant(False)))
        for handler in node.handlers:
            self.bound = set(incoming_bound)
            if handler.name:
                raise ValueError("native try: escaping exception-object bindings need owned payload storage")
            caught = self.name()
            test = ast.BoolOp(op=ast.And(), values=[ast.UnaryOp(op=ast.Not(), operand=self.load(selected)), self.active(context)])
            if handler.type is not None:
                types = handler.type.elts if isinstance(handler.type, ast.Tuple) else (handler.type,)
                matches = [self.intrinsic("exception_match", (self.load(error),),
                    exception_type=self.exception_identity(typ, self.shadowed)) for typ in types]
                test.values.append(matches[0] if len(matches) == 1 else ast.BoolOp(op=ast.Or(), values=matches))
            arm = [self.assign(selected, ast.Constant(True)), self.assign(caught, self.load(error)),
                   self.assign(error, self.intrinsic("exception_none")),
                   *self.statements(handler.body, Context(error, caught, True))]
            successful_bounds.append(set(self.bound))
            body.append(ast.If(test=test, body=arm, orelse=[]))
        self.bound = set.intersection(*successful_bounds)
        if node.orelse:
            body.append(ast.If(test=self.load(succeeded), body=self.statements(node.orelse, context), orelse=[]))
        if node.finalbody:
            previous = self.name()
            final_error = self.name()
            final_context = Context(final_error, parent.caught, True)
            saved_return = self.name() if self.return_flag is not None else None
            if saved_return:
                body += [self.assign(saved_return, self.load(self.return_flag)), self.assign(self.return_flag, ast.Constant(False))]
            body += [self.assign(previous, self.load(error)), self.assign(final_error, self.intrinsic("exception_none")),
                     *self.statements(node.finalbody, final_context),
                     ast.If(test=self.active(final_context), body=[self.assign(error, self.load(final_error))],
                            orelse=[self.assign(error, self.load(previous))])]
            if saved_return:
                body += [ast.If(test=self.load(self.return_flag), body=[self.assign(error, self.intrinsic("exception_none"))],
                    orelse=[self.assign(self.return_flag, self.load(saved_return))])]
        body.append(ast.If(test=self.active(context), body=[self.assign(parent.error, self.load(error))], orelse=[]))
        return [self.guard(parent, body)]

    def outside(self, statements, context):
        result = []
        for statement in statements:
            if isinstance(statement, ast.Try):
                lowered = self.try_statement(statement, context)
                lowered.append(ast.If(test=self.active(context), body=[ast.Expr(value=self.intrinsic("exception_raise", (self.load(context.error),)))], orelse=[]))
                if self.return_flag is not None:
                    lowered.append(ast.If(test=self.load(self.return_flag), body=[ast.Return(value=self.load(self.return_value))], orelse=[]))
                for item in lowered: ast.copy_location(item, statement)
                result += lowered
            elif isinstance(statement, ast.Raise):
                packet = self.raised_record(statement, context, retain_message_operand=True)
                result.append(ast.copy_location(ast.Expr(value=self.intrinsic(
                    "exception_raise", (packet,))), statement))
            elif isinstance(statement, (ast.If, ast.For, ast.While, ast.With)):
                if (isinstance(statement, ast.For) and isinstance(statement.target, ast.Name)
                        and isinstance(statement.iter, ast.Call)
                        and isinstance(statement.iter.func, ast.Name)
                        and statement.iter.func.id == "range" and "range" not in self.shadowed):
                    # The builtin range contract defines an integer induction
                    # value before normalization enters the loop's try arms.
                    self.types[statement.target.id] = int
                incoming_bound = set(self.bound)
                if isinstance(statement, ast.For) and isinstance(statement.target, ast.Name):
                    self.bound.add(statement.target.id)
                statement.body = self.outside(statement.body, context)
                body_bound = set(self.bound)
                if isinstance(statement, (ast.If, ast.For, ast.While)):
                    self.bound = set(incoming_bound)
                if hasattr(statement, "orelse"):
                    statement.orelse = self.outside(statement.orelse, context)
                if isinstance(statement, ast.If):
                    self.bound.intersection_update(body_bound)
                elif isinstance(statement, (ast.For, ast.While)):
                    self.bound = incoming_bound
                result.append(statement)
            else:
                if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
                    for target in targets:
                        if isinstance(target, ast.Name):
                            self.types[target.id] = self.value_type(statement.value)
                            self.bound.add(target.id)
                result.append(statement)
        return result


def ingest_exception_syntax(tree, *, module_name=None):
    if not any(isinstance(node, (ast.Try, ast.Raise)) for node in ast.walk(tree)):
        return ()
    from ..transmogrifier.graph.graph_express2 import _annotate_visual_source_owners, _source_module_identity, post_source_span
    from .concordance_declarations import SOURCE_EXCEPTION_INGESTION, INGESTION
    from .identity_concordance import current_identity_book, Derived, Mode
    module_name = _source_module_identity(tree, module_name)
    tree._turing_source_module = module_name
    owners = _annotate_visual_source_owners(tree, module_name=module_name)
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    occupied = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    occupied.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
    def local_nodes(body):
        for node in body:
            yield node
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Lambda)):
                yield from local_nodes(ast.iter_child_nodes(node))
    def bindings(nodes):
        result = set()
        for node in nodes:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store): result.add(node.id)
            elif isinstance(node, ast.arg): result.add(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)): result.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                result.update(item.asname or item.name.split('.')[0] for item in node.names)
        return result
    global_bindings = bindings(local_nodes(tree.body))
    exception_imports = {}
    import_sources = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                exception_imports[item.asname or item.name.split('.')[0]] = item.name
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                exception_imports[item.asname or item.name] = str(node.module) + '.' + item.name
        else:
            continue
        span = post_source_span(None, node)
        if span is not None:
            import_sources.append(span)
    global_rebindings = bindings(node for node in local_nodes(tree.body)
                                if not isinstance(node, (ast.Import, ast.ImportFrom)))
    function_bindings = {getattr(fn, '_turing_source_qualname', fn.name):
                         bindings(local_nodes(fn.body)) | {a.arg for a in (*fn.args.args, *fn.args.posonlyargs, *fn.args.kwonlyargs)}
                         for fn in functions}
    originals = {id(fn): (ast.dump(fn, include_attributes=False), post_source_span(fn, fn)) for fn in functions}
    def annotation_type(node):
        if isinstance(node, ast.Name): return {"int": int, "float": float, "bool": bool}.get(node.id)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in {"list", "tuple"}:
            return (ast.List if node.value.id == "list" else ast.Tuple, annotation_type(node.slice))
        return None
    function_types = {}
    for fn in functions:
        inference = FailureArms(set())
        inference.call_types = {f.name: annotation_type(f.returns) for f in functions if '.' not in getattr(f, '_turing_source_qualname', f.name)}
        inference.types = {a.arg: annotation_type(a.annotation) for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)}
        assignments = [(target.id, statement.value)
                       for statement in local_nodes(fn.body) if isinstance(statement, (ast.Assign, ast.AnnAssign))
                       for target in (statement.targets if isinstance(statement, ast.Assign) else (statement.target,))
                       if isinstance(target, ast.Name)]
        # Native operator result contracts apply inside protected arms too.
        # Previously only top-level Assign was visited, so a BarrierWait index
        # assigned in try had no type at the protected-return writer. Solve
        # identical basic kinds to a fixed point; conflicting/unknown kinds
        # remain unproven, never a guessed scalar coercion.
        for _round in range(len(assignments) + 1):
            candidates = {}
            for name, value in assignments:
                candidates.setdefault(name, set()).add(inference.value_type(value))
            changed = False
            for name, kinds in candidates.items():
                known = kinds - {None}
                kind = next(iter(known)) if len(known) == 1 else None
                if inference.types.get(name) != kind:
                    inference.types[name] = kind; changed = True
            if not changed: break
        for _round in range(len(assignments) + 1):
            unresolved = {name for name, value in assignments if inference.value_type(value) is None}
            changed = False
            for name in unresolved:
                if inference.types.get(name) is not None:
                    inference.types[name] = None; changed = True
            if not changed: break
        function_types[getattr(fn, '_turing_source_qualname', fn.name)] = dict(inference.types)
    receipts = []
    # Descendants first so normalizing an outer function never treats its
    # inner definition's already-native operations as authored Python calls.
    for function in reversed(functions):
        if not any(isinstance(node, (ast.Try, ast.Raise)) for node in local_nodes(function.body)):
            continue
        original, source = originals[id(function)]
        authored_function = function
        # Build a proposed conversion without altering the input definition.
        # Publish its exact source causes before applying the new body.
        function = _copy_source(function)
        sources = [source, *import_sources]
        for node in ast.walk(authored_function):
            if isinstance(node, (ast.Try, ast.Raise, ast.BinOp, ast.Subscript, ast.Call)):
                owner_key = tuple(getattr(node, "_turing_source_span_row", ()))[:2]
                owner = owners.get(owner_key, function)
                cell = post_source_span(owner, node)
                if cell is not None: sources.append(cell)
        lowering = FailureArms(occupied)
        lowering.call_types = {f.name: annotation_type(f.returns) for f in tree.body if isinstance(f, ast.FunctionDef)}
        identity = getattr(function, '_turing_source_qualname', function.name)
        for length in range(1, len(identity.split('.')) + 1):
            lowering.types.update(function_types.get('.'.join(identity.split('.')[:length]), {}))
        lowering.shadowed = global_bindings | bindings(local_nodes(function.body)) | {a.arg for a in (*function.args.args, *function.args.posonlyargs, *function.args.kwonlyargs)}
        lowering.bound = {a.arg for a in (*function.args.args, *function.args.posonlyargs, *function.args.kwonlyargs)}
        lowering.exception_imports = exception_imports
        lowering.external_shadowed = set(global_rebindings)
        for length in range(1, len(identity.split('.')) + 1):
            lowering.external_shadowed.update(function_bindings.get('.'.join(identity.split('.')[:length]), ()))
        error = lowering.name()
        context = Context(error)
        prefix = lowering.assign(error, lowering.intrinsic("exception_none"))
        protected_returns = [node for candidate in local_nodes(function.body) if isinstance(candidate, ast.Try)
                             for node in local_nodes(candidate.body + [s for h in candidate.handlers for s in h.body] + candidate.finalbody)
                             if isinstance(node, ast.Return)]
        returns_prefix = []
        if protected_returns:
            return_type = annotation_type(function.returns)
            if return_type is None:
                kinds = {lowering.value_type(node.value) for node in protected_returns}
                return_type = next(iter(kinds)) if len(kinds) == 1 else None
            lowering.return_flag, lowering.return_value = lowering.name(), lowering.name()
            returns_prefix = [lowering.assign(lowering.return_flag, ast.Constant(False)), lowering.assign(lowering.return_value, lowering.filler(return_type))]
        statements = lowering.outside(function.body, context)
        function.body = [prefix, *returns_prefix, *statements]
        _hoist_exception_calls(function, lowering)
        ast.fix_missing_locations(function)
        cell = current_identity_book().post(SOURCE_EXCEPTION_INGESTION, (source,),
            (original, ast.dump(function, include_attributes=False), lowering.checks), stage=INGESTION,
            provenance=Derived(tuple(dict.fromkeys(sources))), mode=Mode.CONCORD)
        for node in local_nodes(function.body):
            if isinstance(node, (ast.expr, ast.stmt)):
                node._turing_ingestion_conversion_cell = cell
        # This is the ingestion writer, authorized by the conversion above.
        authored_function.body = function.body
        receipts.append(cell)
    return tuple(receipts)


def _hoist_exception_calls(function, lowering):
    """A-normal form: primitive predicates execute before the control they select.

    Short-circuit operands retain their own conditional ownership. No native
    primitive call is handed to a Python condition evaluator.
    """
    def expr(node):
        if isinstance(node, ast.Call) and getattr(node, '_turing_dispatch_operation', {}).get('operation', '').startswith('exception_'):
            before, args = [], []
            for argument in node.args:
                pre, value = expr(argument); before += pre; args.append(value)
            node.args = args
            name = lowering.name()
            return [*before, lowering.assign(name, node)], lowering.load(name)
        if isinstance(node, ast.BoolOp):
            before, first = expr(node.values[0])
            result = lowering.name()
            before.append(lowering.assign(result, first))
            for operand in node.values[1:]:
                pre, value = expr(operand)
                predicate = lowering.load(result) if isinstance(node.op, ast.And) else ast.UnaryOp(op=ast.Not(), operand=lowering.load(result))
                before.append(ast.If(test=predicate, body=[*pre, lowering.assign(result, value)], orelse=[]))
            return before, lowering.load(result)
        before = []
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.expr):
                pre, result = expr(value); before += pre; setattr(node, field, result)
            elif isinstance(value, list):
                replacement = []
                for item in value:
                    if isinstance(item, ast.expr):
                        pre, result = expr(item); before += pre; replacement.append(result)
                    else: replacement.append(item)
                setattr(node, field, replacement)
        return before, node
    def statements(nodes):
        output = []
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                output.append(node); continue
            before = []
            for field, value in list(ast.iter_fields(node)):
                if isinstance(value, ast.expr):
                    pre, result = expr(value); before += pre; setattr(node, field, result)
                elif isinstance(value, list) and value and all(isinstance(item, ast.stmt) for item in value):
                    setattr(node, field, statements(value))
            for new in before:
                ast.copy_location(new, node)
            output += before + [node]
        return output
    function.body = statements(function.body)
