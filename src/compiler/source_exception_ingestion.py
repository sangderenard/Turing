"""Frame exception regions as explicit record operations and control arms.

The authored statements inside each region remain ordinary source for normal
ProcessGraph ingestion.  This conversion changes only exception structure:
protected regions, handler selection, cleanup, raise, and protected returns.
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
        self.return_flag = None
        self.return_value = None
        self.exception_imports = {}
        self.exception_declarations = {}
        self.external_shadowed = set()
        self.bound = set()

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

    def exception_identity(self, node, occupied):
        del occupied
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            parts.reverse()
            root, *suffix = parts
            declared = self.exception_declarations.get(root)
            if declared is not None:
                return '.'.join((declared, *suffix))
            imported = self.exception_imports.get(root)
            if imported is not None and root not in self.external_shadowed:
                return '.'.join((imported, *suffix))
            value = getattr(builtins, root, None)
            if (not suffix and root not in self.external_shadowed
                    and isinstance(value, type) and issubclass(value, BaseException)):
                return f"builtins.{root}"
            # The exception record carries identity, not a Python exception
            # hierarchy decision.  Preserve every other authored class path;
            # graph ingestion owns the referenced value and its semantics.
            return '.'.join(parts)
        # A failure record is allowed to name any authored class expression.
        # This conversion does not decide whether Python would instantiate it.
        return ast.unparse(node)

    def statements(self, statements, context):
        result = []
        for statement in statements:
            lowered = self.statement(statement, context)
            for node in lowered:
                ast.copy_location(node, statement)
                for descendant in ast.walk(node):
                    if isinstance(descendant, (ast.expr, ast.stmt)) and not hasattr(descendant, "lineno"):
                        ast.copy_location(descendant, statement)
            result.extend(lowered)
        return result

    def raised_record(self, node, context, *, retain_message_operand=False):
        """One exception identity/payload writer for protected and bare raises.

        Compiler source has guards such as interchange_reduction_loops' plain
        ValueError. These must become graph operations even without a try.
        Dynamic messages remain exact graph operands, never cached AST payloads;
        native materialization still needs their owned-message ABI.
        """
        if node.exc is None:
            if context.caught is None:
                raise ValueError("native try: bare raise has no active handler")
            return self.load(context.caught)
        exception = node.exc
        cls = exception.func if isinstance(exception, ast.Call) else exception
        identity = self.exception_identity(cls, self.shadowed)
        message = None
        operands = []
        message_argument = None
        if isinstance(exception, ast.Call):
            if exception.keywords or len(exception.args) > 1:
                raise ValueError("native try: dynamic exception payload ownership is not lowered")
            if exception.args:
                if isinstance(exception.args[0], ast.Constant):
                    message = str(exception.args[0].value)
                elif retain_message_operand:
                    message_argument = len(operands)
                    operands.append(_copy_source(exception.args[0]))
                else:
                    raise ValueError("native try: dynamic exception payload ownership is not lowered")
        # ``from None`` is a structural suppression flag, not a value to
        # evaluate.  Every other explicit cause remains an ordinary graph
        # operand, so reduction can publish its exact value identity rather
        # than reconstructing a relationship from source spelling.
        suppress_context = node.cause is not None
        cause_argument = None
        if node.cause is not None and not (
                isinstance(node.cause, ast.Constant) and node.cause.value is None):
            cause_argument = len(operands)
            operands.append(_copy_source(node.cause))
        return self.intrinsic(
            "exception_make", tuple(operands), exception_type=identity,
            exception_message=message,
            exception_message_argument=message_argument,
            exception_cause_argument=cause_argument,
            suppress_context=suppress_context,
            error_code=100,
        )

    def statement(self, node, context):
        if isinstance(node, ast.Raise):
            return [self.assign(context.error, self.raised_record(
                node, context, retain_message_operand=True,
            ))]
        if isinstance(node, ast.Try):
            return self.try_statement(node, context)
        if isinstance(node, ast.Return):
            if context.protected:
                self.checks += 1
                # ``Return.value is None`` is the AST spelling for a bare
                # ``return``; it is not an expression and cannot be placed in
                # Assign.value.  Its authored Python result is the literal
                # ``None``, which normal graph ingestion will type.
                value = (ast.Constant(None) if node.value is None
                         else _copy_source(node.value))
                return [
                    self.assign(self.return_value, value),
                    self.assign(context.error, self.intrinsic("exception_take")),
                    self.guard(context, [
                        self.assign(self.return_flag, ast.Constant(True)),
                    ]),
                ]
            return [_copy_source(node)]
        preserved = _copy_source(node)
        if isinstance(preserved, (ast.If, ast.For, ast.AsyncFor, ast.While,
                                  ast.With, ast.AsyncWith)):
            preserved.body = self.statements(preserved.body, context) or [ast.Pass()]
            if hasattr(preserved, 'orelse'):
                preserved.orelse = self.statements(preserved.orelse, context)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if isinstance(target, ast.Name):
                    self.bound.add(target.id)
        return [preserved]

    def try_statement(self, node, parent):
        incoming_bound = set(self.bound)
        error = self.name()
        context = Context(error, parent.caught, True)
        protected_body = self.statements(node.body, context) or [ast.Pass()]
        self.checks += 1
        body = [
            self.assign(error, self.intrinsic("exception_none")),
            self.guard(context, protected_body),
            self.guard(context, [self.assign(error, self.intrinsic("exception_take"))]),
        ]
        body_bound = set(self.bound)
        successful_bounds = [body_bound]
        succeeded = self.name()
        body.append(self.assign(succeeded, ast.UnaryOp(op=ast.Not(), operand=self.active(context))))
        selected = self.name()
        body.append(self.assign(selected, ast.Constant(False)))
        for handler in node.handlers:
            self.bound = set(incoming_bound)
            caught = self.name()
            test = ast.BoolOp(op=ast.And(), values=[ast.UnaryOp(op=ast.Not(), operand=self.load(selected)), self.active(context)])
            if handler.type is not None:
                types = handler.type.elts if isinstance(handler.type, ast.Tuple) else (handler.type,)
                matches = [self.intrinsic("exception_match", (self.load(error),),
                    exception_type=self.exception_identity(typ, self.shadowed)) for typ in types]
                test.values.append(matches[0] if len(matches) == 1 else ast.BoolOp(op=ast.Or(), values=matches))
            handler_context = Context(error, caught, True)
            handler_body = self.statements(handler.body, handler_context) or [ast.Pass()]
            self.checks += 1
            arm = [self.assign(selected, ast.Constant(True)), self.assign(caught, self.load(error)),
                   *([self.assign(handler.name, self.load(caught))] if handler.name else []),
                   self.assign(error, self.intrinsic("exception_none")),
                   self.guard(handler_context, handler_body),
                   self.guard(handler_context, [self.assign(error, self.intrinsic("exception_take"))])]
            successful_bounds.append(set(self.bound))
            body.append(ast.If(test=test, body=arm, orelse=[]))
        self.bound = set.intersection(*successful_bounds)
        if node.orelse:
            else_body = self.statements(node.orelse, context)
            self.checks += 1
            body.append(ast.If(test=self.load(succeeded), body=[
                self.guard(context, else_body),
                self.guard(context, [self.assign(error, self.intrinsic("exception_take"))]),
            ], orelse=[]))
        if node.finalbody:
            previous = self.name()
            final_error = self.name()
            final_context = Context(final_error, parent.caught, True)
            saved_return = self.name() if self.return_flag is not None else None
            if saved_return:
                body += [self.assign(saved_return, self.load(self.return_flag)), self.assign(self.return_flag, ast.Constant(False))]
            final_body = self.statements(node.finalbody, final_context) or [ast.Pass()]
            self.checks += 1
            body += [self.assign(previous, self.load(error)), self.assign(final_error, self.intrinsic("exception_none")),
                     self.guard(final_context, final_body),
                     self.guard(final_context, [self.assign(final_error, self.intrinsic("exception_take"))]),
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
            if isinstance(node, (ast.Try, ast.Raise)):
                owner_key = tuple(getattr(node, "_turing_source_span_row", ()))[:2]
                owner = owners.get(owner_key, function)
                cell = post_source_span(owner, node)
                if cell is not None: sources.append(cell)
        lowering = FailureArms(occupied)
        identity = getattr(function, '_turing_source_qualname', function.name)
        lowering.shadowed = global_bindings | bindings(local_nodes(function.body)) | {a.arg for a in (*function.args.args, *function.args.posonlyargs, *function.args.kwonlyargs)}
        lowering.bound = {a.arg for a in (*function.args.args, *function.args.posonlyargs, *function.args.kwonlyargs)}
        lowering.exception_imports = exception_imports
        lowering.exception_declarations = {
            candidate.name: '.'.join((module_name, getattr(
                candidate, '_turing_source_qualname', candidate.name)))
            for candidate in tree.body
            if isinstance(candidate, ast.ClassDef)
        }
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
            lowering.return_flag, lowering.return_value = lowering.name(), lowering.name()
            # This value is never observed unless ``return_flag`` is true, and
            # every path that sets the flag first assigns the authored return
            # value.  Give the generated binding one ordinary definition so
            # graph ingestion never has to invent an untyped incoming value;
            # the exception rewrite does not inspect or decide its dtype.
            returns_prefix = [
                lowering.assign(lowering.return_flag, ast.Constant(False)),
                lowering.assign(lowering.return_value, ast.Constant(0)),
            ]
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
