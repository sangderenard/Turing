from .core import Linear, Sequential, Model, RectConv2d, RectConv3d, MaxPool2d, Flatten, wrap_module
from .activations import ReLU, Sigmoid, Tanh, Identity
from .losses import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from .optimizer import Adam
from .train import train_step, train_loop
from .utils import set_seed
from .fused_program import (
    Meta,
    OpStep,
    FusedProgram,
    build_fused_program,
    ProgramRunner,
)
from .completion_training import (
    CompletionTrainer,
    sample_document_pairs,
    encode_text,
    decode_text,
)
