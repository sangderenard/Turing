from .core import Linear, Sequential, Model, RectConv2d, RectConv3d, MaxPool2d, Flatten, wrap_module
from .activations import ReLU, Sigmoid, Tanh, Identity
from .losses import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from .optimizer import Adam
from .train import train_step, train_loop
from .utils import set_seed

# Fused-program and completion-training imports pull in the compiler/repository
# graph stack. Keep ordinary AbstractNN layers lightweight while preserving the
# same public package API for callers that request those facilities.
_LAZY_EXPORTS = {
    "Meta": (".fused_program", "Meta"),
    "OpStep": (".fused_program", "OpStep"),
    "FusedProgram": (".fused_program", "FusedProgram"),
    "build_fused_program": (".fused_program", "build_fused_program"),
    "capture_forward_program": (".fused_program", "capture_forward_program"),
    "capture_backward_program": (".fused_program", "capture_backward_program"),
    "BackwardProgramCapture": (".fused_program", "BackwardProgramCapture"),
    "ProgramRunner": (".fused_program", "ProgramRunner"),
    "ReverseProgramCapture": (".reverse_program", "ReverseProgramCapture"),
    "ReverseProgramResult": (".reverse_program", "ReverseProgramResult"),
    "capture_reverse_fused_program": (".reverse_program", "capture_reverse_fused_program"),
    "retain_uncaptured_outputs": (".reverse_program", "retain_uncaptured_outputs"),
    "CompletionTrainer": (".completion_training", "CompletionTrainer"),
    "sample_document_pairs": (".completion_training", "sample_document_pairs"),
    "encode_text": (".completion_training", "encode_text"),
    "decode_text": (".completion_training", "decode_text"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "Linear", "Sequential", "Model", "RectConv2d", "RectConv3d",
    "MaxPool2d", "Flatten", "wrap_module", "ReLU", "Sigmoid", "Tanh",
    "Identity", "MSELoss", "CrossEntropyLoss", "BCEWithLogitsLoss", "Adam",
    "train_step", "train_loop", "set_seed", *_LAZY_EXPORTS,
]
