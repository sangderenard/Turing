import runpy
from src.compiler.ssa_fortran_backend import FortranEmissionError
try:
    runpy.run_module('build.probe_authored_snapshot', run_name='__main__')
except FortranEmissionError as error:
    assert 'explicit presence and payload' in str(error), str(error)
    print('OPTIONAL FRONTIER:', error)
else:
    raise AssertionError('Expected unresolved optional representation to be rejected')
