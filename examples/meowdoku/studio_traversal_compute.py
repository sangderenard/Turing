"""Full traversal artifacts await the sanctioned GLSL compiler pipeline.

The duplicate direct-SSA reader has been removed. No alternate frontend or
silent fallback is used here; existing scalar candidate products are separate.
"""


def build(directory, *, node_capacity, triangle_capacity, batch_capacity=4096, region=True):
    raise NotImplementedError(
        "Full BVH traversal compilation is unavailable pending support through "
        "glsl_ast_ingestion and lower_ast_source_to_ssa; no artifacts were written."
    )
