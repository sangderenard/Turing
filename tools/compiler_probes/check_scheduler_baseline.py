import inspect
import pytest
from src.compiler import precompile_to_ssa as module
source=inspect.getsource(module._schedule_loop_callsites)
source=source.replace('if isinstance(block, ConditionalBlock):\n            # Keep the branch atomic', 'if False:\n            # Keep the branch atomic')
exec(compile(source,'<baseline-scheduler>','exec'),module.__dict__)
raise SystemExit(pytest.main(['-q','tests/test_precompile_to_ssa.py','--tb=no']))
