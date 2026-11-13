"""Named neural state using AbstractTensor's existing backend and gradient tape.

Parameter is a factory, not a new tensor backend. Arithmetic results are ordinary
AbstractTensors; only explicitly registered leaves belong to a module.
"""
from collections.abc import MutableMapping

from ..abstraction import AbstractTensor


def Parameter(data, requires_grad=True):
    """Create a backend-preserving, independent trainable leaf."""
    if not isinstance(data, AbstractTensor):
        data = AbstractTensor.get_tensor(data)
    result = AbstractTensor.get_tensor(data.data, cls=type(data))
    result.data = data.clone_()
    result.requires_grad_(requires_grad)
    result._nn_parameter = True
    return result


class Module:
    """Named parameters, buffers and children; numeric work stays in forward."""

    def __init__(self):
        object.__setattr__(self, '_parameters', {})
        object.__setattr__(self, '_buffers', {})
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, '_nonpersistent_buffers', set())
        self.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if '_parameters' in self.__dict__ and name not in ('_parameters', '_buffers', '_modules', '_nonpersistent_buffers'):
            if isinstance(value, Module):
                self.add_module(name, value)
                return
            if getattr(value, '_nn_parameter', False):
                self.register_parameter(name, value)
                return
            if name in self._parameters:
                self.register_parameter(name, value)
                return
            if name in self._buffers:
                self.register_buffer(name, value, name not in self._nonpersistent_buffers)
                return
            if name in self._modules:
                self.add_module(name, value)
                return
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        for registry in ('_parameters', '_buffers', '_modules'):
            if name in self.__dict__.get(registry, {}):
                return self.__dict__[registry][name]
        raise AttributeError(name)

    def __delattr__(self, name):
        for registry in (self._parameters, self._buffers, self._modules):
            if name in registry:
                del registry[name]
                self._nonpersistent_buffers.discard(name)
                return
        object.__delattr__(self, name)

    def _check_name(self, name, registry):
        if not isinstance(name, str) or not name or '.' in name:
            raise ValueError('registered names must be nonempty strings without dots')
        if name not in registry and hasattr(self, name):
            raise KeyError('attribute already exists: ' + name)

    def register_parameter(self, name, value):
        self._check_name(name, self._parameters)
        if value is not None and not (isinstance(value, AbstractTensor) and getattr(value, '_nn_parameter', False)):
            raise TypeError('parameter must be produced by Parameter, or None')
        self._parameters[name] = value

    def register_buffer(self, name, value, persistent=True):
        self._check_name(name, self._buffers)
        if value is not None and not isinstance(value, AbstractTensor):
            raise TypeError('buffer must be an AbstractTensor, or None')
        self._buffers[name] = value
        if persistent:
            self._nonpersistent_buffers.discard(name)
        else:
            self._nonpersistent_buffers.add(name)

    def add_module(self, name, value):
        self._check_name(name, self._modules)
        if value is not None and not isinstance(value, Module):
            raise TypeError('child must be a Module, or None')
        if value is not None and any(child is self for _, child in value.named_modules()):
            raise ValueError('module registration would create a cycle')
        self._modules[name] = value

    def named_modules(self, prefix='', memo=None):
        memo = set() if memo is None else memo
        if id(self) in memo:
            return
        memo.add(id(self))
        yield prefix, self
        for name, child in self._modules.items():
            if child is not None:
                yield from child.named_modules(prefix + '.' + name if prefix else name, memo)

    def _named_state(self, registry, prefix='', recurse=True):
        seen = set()
        modules = self.named_modules(prefix) if recurse else [(prefix, self)]
        for path, module in modules:
            for name, tensor in getattr(module, registry).items():
                if tensor is not None and id(tensor) not in seen:
                    seen.add(id(tensor))
                    yield path + '.' + name if path else name, tensor

    def named_parameters(self, prefix='', recurse=True):
        return self._named_state('_parameters', prefix, recurse)

    def parameters(self, recurse=True):
        return (value for _, value in self.named_parameters(recurse=recurse))

    def named_buffers(self, prefix='', recurse=True):
        return self._named_state('_buffers', prefix, recurse)

    def buffers(self, recurse=True):
        return (value for _, value in self.named_buffers(recurse=recurse))

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise TypeError('training mode must be boolean')
        for _, module in self.named_modules():
            module.training = mode
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def state_dict(self):
        state = {}
        for path, module in self.named_modules():
            for name, value in (*module._parameters.items(), *module._buffers.items()):
                if value is not None and name not in module._nonpersistent_buffers:
                    state[path + '.' + name if path else name] = value
        return state

    def load_state_dict(self, state, strict=True):
        """Copy payloads without replacing registered leaves or their tape IDs.

        Validate every value before mutating any state. Return missing and
        unexpected names for callers choosing non-strict loading.
        """
        current = self.state_dict()
        missing = [name for name in current if name not in state]
        unexpected = [name for name in state if name not in current]
        if strict and (missing or unexpected):
            raise ValueError(f'state mismatch: missing={missing}, unexpected={unexpected}')
        copies = []
        for name, target in current.items():
            if name not in state:
                continue
            source = state[name]
            if not isinstance(source, AbstractTensor):
                raise TypeError('state values must be AbstractTensors: ' + name)
            if type(source) is not type(target):
                raise TypeError('state backend mismatch: ' + name)
            source_shape = source.shape() if callable(source.shape) else source.shape
            target_shape = target.shape() if callable(target.shape) else target.shape
            if tuple(source_shape) != tuple(target_shape):
                raise ValueError('state shape mismatch: ' + name)
            if source.dtype != target.dtype:
                raise ValueError('state dtype mismatch: ' + name)
            copies.append((target, source.clone_()))
        for target, payload in copies:
            target.data = payload
        return missing, unexpected


class ModuleDict(Module, MutableMapping):
    """Ordered mapping of registered child modules."""
    def __init__(self, modules=()):
        Module.__init__(self)
        self.update(modules)

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, value):
        self.add_module(key, value)

    def __delitem__(self, key):
        del self._modules[key]

    def __iter__(self):
        return iter(self._modules)

    def __len__(self):
        return len(self._modules)


class ParameterDict(Module, MutableMapping):
    """Ordered mapping of registered AbstractTensor parameters."""
    def __init__(self, parameters=()):
        Module.__init__(self)
        self.update(parameters)

    def __getitem__(self, key):
        return self._parameters[key]

    def __setitem__(self, key, value):
        if isinstance(value, AbstractTensor) and not getattr(value, '_nn_parameter', False):
            value = Parameter(value)
        self.register_parameter(key, value)

    def __delitem__(self, key):
        del self._parameters[key]

    def __iter__(self):
        return iter(self._parameters)

    def __len__(self):
        return len(self._parameters)
