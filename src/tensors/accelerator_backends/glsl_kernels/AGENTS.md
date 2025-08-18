# GLSL Kernels for Accelerator Backends

This directory stores standalone compute shader snippets used by the
`OpenGLTensorOperations` backend.  Each `.comp.glsl` file should contain
clear stub sections demarcated by the `INIT_START`, `OPERATIONS_START`, and
`OUTPUT_START` sentinels.  These markers allow the backend to stitch shader
pieces together when constructing batched operation pipelines.

Actual shader logic may be minimal or completely stubbed, but every file must
use the repository stub comment block describing its purpose and planned
behavior.  See `template.comp.glsl` for the recommended structure.
