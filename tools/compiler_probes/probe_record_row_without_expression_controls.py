import runpy
from src.compiler import glsl_deployment_strategy as strategy

original = strategy._ordinary_conditional_control_programs
def without_expression_controls(*args, **kwargs):
    return tuple(program for program in original(*args, **kwargs)
                 if not any(getattr(block, 'result_aliases', ())
                            for block in program.root.blocks))
strategy._ordinary_conditional_control_programs = without_expression_controls
runpy.run_path('tools/repro_record_row_effects.py', run_name='__main__')
