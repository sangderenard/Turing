"""Authored class annotations select the incumbent raw memory contract."""

import ast
from types import SimpleNamespace

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.concordance_declarations import (
    CLASS_DECLARATION,
    NAME_BINDING,
    PARAMETER_ANNOTATION,
    SOURCE_FORMAL_ORDER,
    SOURCE_PARAMETER_MEMORY_CONTRACT,
)
from src.compiler.identity_concordance import (
    CorrelationTable, begin_identity_book, end_identity_book,
)
from src.transmogrifier.function_table import (
    ParameterAccess, ParameterScope, ParameterStorage, ParameterTransfer,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def test_authored_module_formals_install_one_concorded_memory_contract():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(
            "class Module:\n"
            "    def __init__(self):\n"
            "        self.metadata = {}\n"
            "def publish(module: Module):\n"
            "    module.metadata['key'] = 7\n"
            "def consume(module: Module, name: str):\n"
            "    return module.metadata.get(name)\n"
            "def unused(module: Module):\n"
            "    return 0\n"
            "def root(name: str):\n"
            "    module = Module()\n"
            "    publish(module)\n"
            "    return consume(module, name)\n"
        ))
        reduce_abstract_tensor_topology(graph)
        for function_name in ("publish", "consume"):
            entry = graph.function_table.entry(function_name)
            scope = entry.graph.G.graph["authored_function_identity"]
            contract = next(c for c in entry.parameter_contracts
                            if c.name == "module")
            assert (contract.transfer, contract.access, contract.storage,
                    contract.scope) == (
                ParameterTransfer.ALIAS, ParameterAccess.INOUT,
                ParameterStorage.RECORD, ParameterScope.CALLER,
            )
            receipt = book.latest_ref(
                SOURCE_PARAMETER_MEMORY_CONTRACT, (scope, "module"),
            )
            assert receipt is not None
            assert book.page(SOURCE_PARAMETER_MEMORY_CONTRACT).cell(
                receipt.row, receipt.column,
            ) == (contract,)
            sources = {source for source, _stage in book.edges_into(receipt)}
            assert book.latest_ref(SOURCE_FORMAL_ORDER, (scope,)) in sources
            assert any(source.page == NAME_BINDING for source in sources)
            prior = next(source for source in sources
                         if source.page == SOURCE_PARAMETER_MEMORY_CONTRACT)
            causes = {source for source, _stage in book.edges_into(prior)}
            assert book.latest_ref(PARAMETER_ANNOTATION,
                                   (*scope, "module")) in causes
            assert any(source.page == CLASS_DECLARATION for source in causes)
        name_contract = next(c for c in graph.function_table.entry(
            "consume").parameter_contracts if c.name == "name")
        assert (name_contract.transfer, name_contract.access,
                name_contract.storage, name_contract.scope) == (
            ParameterTransfer.VALUE, ParameterAccess.IN,
            ParameterStorage.SCALAR, ParameterScope.LOCAL,
        )
        unused = graph.function_table.entry("unused")
        unused_scope = unused.graph.G.graph["authored_function_identity"]
        unused_receipt = book.latest_ref(
            SOURCE_PARAMETER_MEMORY_CONTRACT, (unused_scope, "module"),
        )
        unused_sources = {
            source for source, _stage in book.edges_into(unused_receipt)
        }
        assert book.latest_ref(SOURCE_FORMAL_ORDER, (unused_scope,)) in unused_sources
        assert not any(source.page == NAME_BINDING for source in unused_sources)
        assert not CorrelationTable._source_parameter_memory_contract_findings(
            SimpleNamespace(metadata={"identity_book": book},
                            function_table=graph.function_table),
        )
    finally:
        end_identity_book(token)
