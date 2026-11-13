import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.numpy_backend import NumPyTensorOperations
from src.common.tensors.abstract_nn import Module, Parameter, ModuleDict, ParameterDict


def tensor(value):
    return AbstractTensor.get_tensor(value, cls=NumPyTensorOperations)


class Gain(Module):
    def __init__(self, gain):
        super().__init__()
        self.gain = Parameter(tensor([gain]))
        self.register_buffer('enabled', tensor([1.0]))

    def forward(self, value):
        return value * self.gain * self.enabled


def test_parameter_is_independent_backend_preserving_leaf():
    original = tensor([0.5])
    parameter = Parameter(original)
    assert type(parameter) is type(original)
    assert parameter.requires_grad
    assert parameter.data is not original.data
    assert not Parameter(original, requires_grad=False).requires_grad
    assert not getattr(parameter * 2, '_nn_parameter', False)


def test_registered_graph_arithmetic_and_state():
    root = Module()
    root.stages = ModuleDict({'rack': Gain(0.5), 'shop': Gain(0.5)})
    root.knobs = ParameterDict({'level': tensor([0.4])})
    result = root.stages['shop'](root.stages['rack'](root.knobs['level']))
    assert result.item() == pytest.approx(0.1)
    assert list(dict(root.named_parameters())) == ['stages.rack.gain', 'stages.shop.gain', 'knobs.level']
    assert len(list(root.buffers())) == 2
    root.eval()
    assert all(not module.training for _, module in root.named_modules())
    assert 'stages.shop.enabled' in root.state_dict()


def test_registration_validation_sharing_and_deletion():
    root = Module()
    shared = Parameter(tensor([1.0]))
    root.a = shared
    root.b = shared
    assert len(list(root.parameters())) == 1
    root.register_buffer('scratch', tensor([0.0]), persistent=False)
    assert 'scratch' not in root.state_dict()
    with pytest.raises(TypeError):
        root.a = tensor([2.0])
    with pytest.raises(ValueError):
        root.add_module('loop', root)
    with pytest.raises(ValueError):
        root.register_parameter('bad.name', shared)
    del root.a
    assert not hasattr(root, 'a')
    root.b = None
    assert list(root.parameters()) == []


def test_parameter_dictionary_mutation():
    parameters = ParameterDict({'a': tensor([1.0])})
    assert parameters['a'].requires_grad
    parameters.update({'b': Parameter(tensor([2.0]))})
    assert list(parameters) == ['a', 'b']
    del parameters['a']
    assert list(dict(parameters.named_parameters())) == ['b']


def test_load_keeps_leaf_identity_and_validates_before_mutating():
    gain = Gain(0.5)
    leaf = gain.gain
    snapshot = {name: value.clone() for name, value in gain.state_dict().items()}
    snapshot['gain'] = tensor([0.25])
    assert gain.load_state_dict(snapshot) == ([], [])
    assert gain.gain is leaf
    assert leaf.item() == 0.25
    snapshot['gain'] = tensor([0.75])
    snapshot['enabled'] = tensor([1.0, 1.0])
    with pytest.raises(ValueError, match='shape'):
        gain.load_state_dict(snapshot)
    assert leaf.item() == 0.25
    with pytest.raises(ValueError, match='missing'):
        gain.load_state_dict({})


def test_private_named_state_and_shared_child():
    root = Module()
    root._gain = Parameter(tensor([1.0]))
    root.children = ModuleDict({'one': Gain(0.5)})
    root.alias = root.children['one']
    assert len(list(root.parameters())) == 2
    assert '_gain' in root.state_dict()


def test_broadcast_shapes_reuses_shared_rules():
    assert AbstractTensor.broadcast_shapes((1, 3, 1), (2, 1, 4)) == (2, 3, 4)
    assert AbstractTensor.broadcast_shapes((0, 1), (1, 4)) == (0, 4)
    assert AbstractTensor.broadcast_shapes() == ()
    with pytest.raises(ValueError):
        AbstractTensor.broadcast_shapes((2,), (3,))
    with pytest.raises(ValueError):
        AbstractTensor.broadcast_shapes((-1,))


def test_complex_parameter_uses_existing_complex_tensor():
    value = AbstractTensor.complex(tensor([0.5]), tensor([0.25]))
    parameter = Parameter(value)
    assert (parameter * tensor([0.4])).item() == pytest.approx(0.2 + 0.1j)


def test_parameter_registers_with_existing_autograd_tape():
    parameter = Parameter(tensor([0.5]))
    (parameter * parameter).sum().backward()
    assert parameter.grad.item() == pytest.approx(1.0)
