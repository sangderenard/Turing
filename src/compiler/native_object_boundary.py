"""Python entry/exit conversion for the existing SSAObjectContext ABI.

This is boundary marshalling, not a Python container runtime. Native code sees
the context's exact tables and operates without host callbacks. Dictionary
members retain their actual key identities and insertion order; their native
lookup contract remains reference identity, NOT Python equality/hash.
"""
from __future__ import annotations

from collections import deque
import ctypes

import numpy as np

from .ssa_object_context import context_values


_CONTAINERS = {dict: 'builtins.dict', list: 'builtins.list', tuple: 'builtins.tuple',
               set: 'builtins.set', frozenset: 'builtins.frozenset'}


class NativeObjectBoundary:
    """An exclusive handle-domain lease with caller-owned initial columns.

    Exact builtin containers are converted. Other values (including subclasses)
    remain retained opaque leaves; no instance layout is inferred. All roots
    share one identity traversal. New native handles use the context counter,
    and that counter is reconciled before the codec can retain another object.
    The caller must execute the existing native context release before close
    if native growth has allocated memory. No finalizer frees native memory.
    Only trusted native context buffers may be read: this is an in-process ABI,
    not a deserializer for arbitrary addresses supplied by an external party.
    """

    def __init__(self, codec, context, roots):
        if codec._native_object_boundary is not None:
            raise RuntimeError('native object domain is already leased')
        self.codec, self.context = codec, context
        self.receipt = context.to_mapping()
        self.closed = False
        kinds = {name: index + 1 for index, name in enumerate(context.kind_vocabulary)}
        required = {*_CONTAINERS.values(), 'builtins.object'}
        if not required.issubset(kinds):
            raise ValueError('native Python object boundary requires all builtin container kinds and leaf kind')
        roots = tuple(roots)
        self.retained = []
        self._leaf_handles = set()
        self._opaque_container_handles = set()
        from .ssa_object_context import object_type_classes
        inspection_classes = object_type_classes(context.type_vocabulary)
        objects, members, pending, identities = [], [], deque(), {}

        def reference(value):
            identity = id(value)
            if identity in identities:
                return identities[identity]
            handle = codec.retain(value)
            self.retained.append(handle)
            identities[identity] = handle
            label = _CONTAINERS.get(type(value), 'builtins.object')
            row = (handle, kinds[label])
            if context.type_vocabulary:
                mask = sum(1 << bit for bit, target in enumerate(inspection_classes)
                           if isinstance(value, target))
                row += (mask,)
            objects.append(row)
            if label == 'builtins.object':
                self._leaf_handles.add(handle)
                if any(isinstance(value, target) for target in _CONTAINERS):
                    self._opaque_container_handles.add(handle)
            else:
                pending.append((handle, value))
            return handle

        try:
            self.roots = tuple(reference(value) for value in roots)
            while pending:
                owner, value = pending.popleft()
                if type(value) is dict:
                    for key, item in value.items():
                        members.append((owner, reference(key), reference(item)))
                elif type(value) in (list, tuple):
                    for index, item in enumerate(value, 1):
                        members.append((owner, index, reference(item)))
                else:
                    for item in value:
                        handle = reference(item)
                        members.append((owner, handle, handle))
            self.buffers = {value.id: np.zeros(1, dtype=np.uint64 if value.dtype == 'ptr'
                else np.int32 if value.dtype == 'int' else np.int64)
                for value in context_values(context)}
            self._initial_columns = {}
            for descriptor, rows in ((context.objects, objects), (context.members, members)):
                capacity = max(len(rows), 1)
                for index, cell in enumerate(descriptor.column_value_ids):
                    column = np.zeros(capacity, dtype=np.int64)
                    column[:len(rows)] = [row[index] for row in rows]
                    self._initial_columns[cell] = column
                    self.buffers[cell][0] = column.ctypes.data
                self.buffers[descriptor.length_address_id][0] = len(rows)
                self.buffers[descriptor.capacity_value_id][0] = capacity
            self.buffers[context.next_handle_value_id][0] = codec._next_handle
            codec._native_object_boundary = self
        except BaseException:
            for handle in self.retained:
                codec.release(handle)
            raise

    def _check_live(self):
        if self.closed or self.codec._native_object_boundary is not self:
            raise RuntimeError('native object boundary is closed')
        if self.context.to_mapping() != self.receipt:
            raise ValueError('native object boundary context contract changed')

    def bind_execution(self, artifact, function, feeds=None):
        """Prepare the incumbent ctypes ABI ONCE for an entire native pocket.

        Feeds hold handles and real storage cells, never PyObject pointers.
        No eager rehydration occurs between native invocations; export is an
        explicit outer-boundary operation. Reusing the prepared execution avoids
        rebuilding NumPy buffers/pointer arrays on every dispatch.
        """
        self._check_live()
        receipt = function.metadata.get('native_object_context', function.metadata.get('object_context'))
        if receipt != self.receipt or artifact.name != function.name:
            raise ValueError('native invocation does not implement the accepted object context/function')
        if (function.metadata.get('native_object_requires_member_storage')
                and self._opaque_container_handles):
            raise ValueError('native member views cannot consume opaque container subclasses; type evidence is not payload storage')
        from .ssa_object_context import context_values
        from .ssa_llvm_backend import prepare_artifact_execution
        cells = {value.id for value in context_values(self.context)}
        declared = {value.id for value in function.args}
        if not cells.issubset(declared):
            raise ValueError('native invocation is missing accepted object context cells')
        feeds = dict(feeds or {})
        if set(feeds) & cells:
            raise ValueError('native invocation may not replace leased context cells')
        missing = declared - cells - set(feeds)
        if missing:
            raise ValueError(f'native invocation has unbound input identities {sorted(missing)}')
        if set(feeds) - declared:
            raise ValueError('native invocation feeds must name exact function inputs')
        execution = prepare_artifact_execution(artifact, {**feeds, **self.buffers})
        for value_id in cells:
            bound = execution.buffers.get(value_id)
            original = self.buffers[value_id]
            if (bound is None or bound.ctypes.data != original.ctypes.data
                    or bound.nbytes != original.nbytes):
                raise ValueError('native invocation copied or omitted a shared context cell')
        artifact.entry()  # Resolve/cache the existing ctypes entry at bind time.
        return NativeObjectInvocation(self, execution)

    def _rows(self, descriptor):
        length = int(self.buffers[descriptor.length_address_id][0])
        capacity = int(self.buffers[descriptor.capacity_value_id][0])
        if not 0 <= length <= capacity <= (2**63 - 1) // 8:
            raise ValueError('native object table has invalid extents')
        columns = []
        for cell in descriptor.column_value_ids:
            address = int(self.buffers[cell][0])
            if length and not address:
                raise ValueError('native object table has a null live column')
            initial = self._initial_columns[cell]
            if address == initial.ctypes.data:
                if capacity > len(initial):
                    raise ValueError('native object table exceeds its borrowed column capacity')
                columns.append(initial[:length])
            elif length:
                # A live explicit heap owner is the allocation contract, never
                # a cast of an opaque reference. The pointer cell holds storage.
                if not self.buffers[self.context.objects.heap_owner_value_id][0]:
                    raise ValueError('native replacement column has no live heap owner')
                pointer = ctypes.cast(address, ctypes.POINTER(ctypes.c_int64))
                columns.append(np.ctypeslib.as_array(pointer, shape=(length,)))
            else:
                columns.append(())
        return tuple(tuple(int(value) for value in row) for row in zip(*columns))

    def export(self, roots=None):
        """Reify native graph roots, preserving imported identities and aliases.

        Imported mutable objects synchronize only after graph validation; new
        native handles get fresh shells. Imported immutable handles must retain
        their exact members (mutation requires a new handle). Immutable dependencies use a linear
        topological worklist. An impossible immutable-only cycle refuses.
        Host dictionaries/sets apply their real equality/hash semantics here.
        Duplicate equal keys from invalid native construction refuse rather
        than silently collapsing an edge or changing the graph's cardinality.
        """
        self._check_live()
        objects = self._rows(self.context.objects)
        members = self._rows(self.context.members)
        labels, children, keys = {}, {}, {}
        vocabulary = self.context.kind_vocabulary
        for row in objects:
            handle, kind = row[:2]
            if handle <= 0 or handle in labels or not 1 <= kind <= len(vocabulary):
                raise ValueError('invalid or duplicate native object identity/kind')
            if self.context.type_vocabulary and not 0 <= row[2] < 1 << len(self.context.type_vocabulary):
                raise ValueError('native object has invalid type evidence bits')
            labels[handle] = vocabulary[kind - 1]
            children[handle], keys[handle] = [], set()
        counter = int(self.buffers[self.context.next_handle_value_id][0])
        if not 0 < counter < 2**63 or any(handle >= counter for handle in labels):
            raise ValueError('native object counter does not dominate live identities')
        self.codec._next_handle = max(self.codec._next_handle, counter)
        for owner, key, value in members:
            if owner not in labels or key in keys[owner] or value not in labels:
                raise ValueError('native member has missing identity or duplicate key')
            label = labels[owner]
            if label in ('builtins.dict', 'builtins.set', 'builtins.frozenset') and key not in labels:
                raise ValueError('native member key has no object identity')
            keys[owner].add(key)
            children[owner].append((key, value))
        retained = set(self.retained)
        values, immutable = {}, set()
        for handle, label in labels.items():
            if handle not in retained and handle in self.codec._objects:
                raise ValueError('native object identity collides with the host codec')
            if handle in retained and label != _CONTAINERS.get(
                    type(self.codec.object(handle)), 'builtins.object'):
                raise ValueError('native kind disagrees with its retained Python identity')
            if label == 'builtins.object':
                if handle not in self._leaf_handles or children[handle]:
                    raise ValueError('native leaf has no retained Python representation')
                values[handle] = self.codec.object(handle)
            elif label == 'builtins.dict':
                values[handle] = self.codec.object(handle) if handle in retained else {}
            elif label == 'builtins.list':
                values[handle] = self.codec.object(handle) if handle in retained else []
            elif label == 'builtins.set':
                values[handle] = self.codec.object(handle) if handle in retained else set()
            elif label in ('builtins.tuple', 'builtins.frozenset'):
                immutable.add(handle)
            else:
                raise ValueError(f'unhandled native object kind {label}')
            if label in ('builtins.list', 'builtins.tuple'):
                rows = children[handle]
                ordered = [None] * len(rows)
                for key, value in rows:
                    if not 1 <= key <= len(rows):
                        raise ValueError('native positional members must be contiguous and one-based')
                    ordered[key - 1] = (key, value)
                children[handle] = ordered
            if label in ('builtins.set', 'builtins.frozenset') and any(
                    key != value for key, value in children[handle]):
                raise ValueError('native set members must carry the same key and value identity')
        waiting, dependents = {}, {}
        for handle in immutable:
            dependencies = {value for _, value in children[handle] if value in immutable}
            waiting[handle] = len(dependencies)
            for dependency in dependencies:
                dependents.setdefault(dependency, []).append(handle)
        ready = deque(handle for handle, count in waiting.items() if not count)
        while ready:
            handle = ready.popleft()
            items = [values[value] for _, value in children[handle]]
            if handle in retained:
                original = self.codec.object(handle)
                same = (all(left is right for left, right in zip(original, items))
                        if labels[handle] == 'builtins.tuple' else
                        {id(item) for item in original} == {id(item) for item in items})
                if len(original) != len(items) or not same:
                    raise ValueError('native mutation of an imported immutable object requires a new identity')
                values[handle] = original
            else:
                values[handle] = tuple(items) if labels[handle] == 'builtins.tuple' else frozenset(items)
            if len(values[handle]) != len(items):
                raise ValueError('native immutable members collapse under Python equality')
            for dependent in dependents.get(handle, ()):
                waiting[dependent] -= 1
                if not waiting[dependent]:
                    ready.append(dependent)
        if any(handle not in values for handle in immutable):
            raise ValueError('native immutable-only cycle cannot be reified')
        staged = {}
        for handle, label in labels.items():
            rows = children[handle]
            if label == 'builtins.list':
                staged[handle] = [values[value] for _, value in rows]
            elif label == 'builtins.dict':
                staged[handle] = {}
                for key, value in rows:
                    staged[handle][values[key]] = values[value]
            elif label == 'builtins.set':
                staged[handle] = {values[value] for _, value in rows}
            if label in ('builtins.dict', 'builtins.set') and len(staged[handle]) != len(rows):
                raise ValueError('native members collapse under Python equality')
        roots = self.roots if roots is None else tuple(roots)
        if any(handle not in values for handle in roots):
            raise ValueError('native result has no object identity')
        for handle, replacement in staged.items():
            if labels[handle] == 'builtins.list':
                values[handle][:] = replacement
            else:
                values[handle].clear()
                values[handle].update(replacement)
        # Publish native-created representations in the SAME incumbent handle
        # owner. A second export of handle h must return the same object, not
        # reconstruct another identity for h. The exclusive lease prevents
        # concurrent host minting from occupying a native-issued handle.
        for handle, value in values.items():
            if handle not in retained:
                self.codec._objects[handle] = value
                self.retained.append(handle)
        return tuple(values[handle] for handle in roots)

    def close(self):
        """End the lease after explicit native release; drop input retentions."""
        self._check_live()
        if self.buffers[self.context.objects.heap_owner_value_id][0]:
            raise RuntimeError('release the native object context before closing its boundary')
        counter = int(self.buffers[self.context.next_handle_value_id][0])
        self.codec._next_handle = max(self.codec._next_handle, counter)
        self.codec._native_object_boundary = None
        for handle in self.retained:
            self.codec.release(handle)
        for buffer in self.buffers.values():
            buffer.fill(0)
        self.closed = True


class NativeObjectInvocation:
    """Lease-checked reuse of LLVMExecution, not another runtime or registry."""
    def __init__(self, boundary, execution):
        self.boundary, self.execution = boundary, execution
        self.native_invocations = 0

    @property
    def buffers(self):
        return self.execution.buffers

    def run(self):
        # Immutable context/layout acceptance was checked at binding. The hot
        # boundary needs only the live exclusive lease, not another graph walk
        # or freshly constructed descriptor mapping.
        boundary = self.boundary
        if boundary.closed or boundary.codec._native_object_boundary is not boundary:
            raise RuntimeError('native object invocation lease is closed')
        self.execution.run()
        self.native_invocations += 1
        return self


__all__ = ['NativeObjectBoundary', 'NativeObjectInvocation']
