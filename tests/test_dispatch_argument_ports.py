"""Every dispatcher argument needs a port, and ``args`` has none.

``threading.Thread(target=work, args=(values,))`` lowers to ``thread_create``
and then fails at emission with ``dispatcher arguments lack SSA identities``.
Measured at the failure, the two sides of that check are::

    resolved positional: 0   ast positional: 0
    resolved keywords: ('target',)   ast keywords: ['target', 'args']

So ``target`` does resolve -- it is a ``StaticReference`` node with
``reference_kind='function_subgraph'`` carried as ``kw:target`` -- and it is
``args`` that has no edge at all. The tuple's contents never enter the graph:
in the single-worker program below, the ``values`` parameter has no ``Input``
node, because nothing consumes it.

That ordering matters for any repair. Giving ``target`` a function-reference
port and excluding it from the arity check would leave the check comparing
zero resolved keywords against two authored ones, and relaxing the check
further would emit a job submission carrying no data ports at all -- a worker
that silently receives nothing, which is worse than the current refusal. The
data ports have to come first.

See ``docs/COMPILER_INTERPRETATION_RULES.md`` section 12.
"""

import pytest

from src.common.tensors import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

CONTRACT = 'extraction_contracts/program_extraction.yaml'

THREADED = '''
import threading


def work(values):
    return values * 2.0


def root(values, out):
    worker = threading.Thread(target=work, args=(values,))
    worker.start()
    worker.join()
    return out
'''


def _policy():
    return ExtractionContract(CONTRACT).with_program_abi({
        'bindings': [],
        'values': [
            {
                'function': 'root', 'parameter': parameter, 'storage': 'span',
                'dtype': 'float64', 'rank': 1, 'shape': [8],
                'python_type': 'src.common.tensors.abstraction.AbstractTensor',
            }
            for parameter in ('values', 'out')
        ],
    })


@pytest.mark.xfail(
    reason="the args tuple of a dispatcher call becomes no graph edge, so the "
           "job carries no data ports and emission refuses the call; target "
           "already resolves as a StaticReference and is not the missing one",
    strict=False,
)
def test_a_dispatched_job_carries_its_arguments():
    """A pooled job must be able to name the values it runs on."""

    module, _outputs, _exports = lower_ast_source_to_ssa(
        THREADED, 'root', name='dispatch_ports', extraction_contract=_policy(),
        python_bindings={'AbstractTensor': AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    assert module is not None
