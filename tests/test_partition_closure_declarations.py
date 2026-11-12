"""Nested source cuts retain the authored declarations of captured storage."""

from __future__ import annotations

import ast
import symtable

from src.compiler.project_compilation_product import partition_authored_source


# The failing uploaded source declares this mapping in main, before _doc_id.
# Keep the real declaration and body, without inventing a call or tensor ABI.
SOURCE = '''def main():
    _doc_misc_ids: dict[str, int] = {}
    unrelated = object()

    def _doc_id(key: str) -> int:
        if key not in _doc_misc_ids:
            _doc_misc_ids[key] = 100000 + len(_doc_misc_ids)
        return _doc_misc_ids[key]

    return unrelated
'''


def test_nested_partition_retains_captured_mapping_declaration():
    partition, names = partition_authored_source(
        SOURCE, "main.<locals>._doc_id",
    )
    assert "    _doc_misc_ids: dict[str, int] = {}\n" in partition
    assert "unrelated" not in partition
    assert names == ("_doc_id", "_doc_misc_ids", "main")
    original = ast.parse(SOURCE).body[0].body[2]
    retained = ast.parse(partition).body[0].body[1]
    assert ast.dump(original, include_attributes=True) == ast.dump(
        retained, include_attributes=True,
    )


def test_nested_partition_follows_lexical_alias_without_module_shadow():
    source = '''storage = "wrong module binding"
def outer():
    storage: dict[str, int] = {}
    alias = storage
    def inner(key: str) -> int:
        return alias[key]
    unrelated = 1
'''
    partition, names = partition_authored_source(
        source, "outer.<locals>.inner",
    )
    assert "    storage: dict[str, int] = {}\n" in partition
    assert "    alias = storage\n" in partition
    assert "wrong module binding" not in partition
    assert "unrelated" not in partition
    assert names == ("alias", "inner", "outer", "storage")


def test_nested_partition_parameter_shadows_module_definition():
    source = '''storage = "wrong module binding"
def outer(storage: dict[str, int]):
    def inner(key: str) -> int:
        return storage[key]
'''
    partition, names = partition_authored_source(
        source, "outer.<locals>.inner",
    )
    assert "wrong module binding" not in partition
    assert "def outer(storage: dict[str, int]):" in partition
    assert names == ("inner", "outer")


def test_nested_partition_transitive_dependency_keeps_its_defining_scope():
    source = '''storage = "wrong module binding"
def outer():
    storage: dict[str, int] = {}
    alias = storage
    def middle():
        storage = "wrong inner binding"
        def inner(key: str) -> int:
            return alias[key]
'''
    partition, names = partition_authored_source(
        source, "outer.<locals>.middle.<locals>.inner",
    )
    assert "    storage: dict[str, int] = {}\n" in partition
    assert "    alias = storage\n" in partition
    assert "wrong module binding" not in partition
    assert "wrong inner binding" not in partition
    assert names == ("alias", "inner", "middle", "outer", "storage")


def test_nested_method_partition_does_not_use_class_namespace_as_closure():
    source = '''seed = 7
class Container:
    seed = "not a lexical binding"
    def outer(self):
        storage: dict[str, int] = {"a": seed}
        def inner(key: str) -> int:
            return storage[key]
'''
    partition, _names = partition_authored_source(
        source, "Container.outer.<locals>.inner",
    )
    assert "seed = 7\n" in partition
    assert '        storage: dict[str, int] = {"a": seed}\n' in partition
    ast.parse(partition)


def test_nested_partition_global_ignores_enclosing_local_shadow():
    source = '''storage: dict[str, int] = {"module": 7}
def outer():
    storage = "wrong outer binding"
    def inner(key: str) -> int:
        global storage
        return storage[key]
'''
    partition, _names = partition_authored_source(source, "outer.<locals>.inner")
    assert 'storage: dict[str, int] = {"module": 7}\n' in partition
    assert "wrong outer binding" not in partition
    assert "        global storage\n" in partition


def test_nested_partition_nonlocal_keeps_owner_and_enclosing_write():
    source = '''storage = "wrong module binding"
def outer():
    storage: dict[str, int] = {}
    def middle():
        nonlocal storage
        storage = dict(storage)
        unrelated = object()
        def inner(key: str) -> int:
            nonlocal storage
            storage[key] = 1
            return storage[key]
'''
    partition, _names = partition_authored_source(
        source, "outer.<locals>.middle.<locals>.inner",
    )
    assert "wrong module binding" not in partition
    assert "    storage: dict[str, int] = {}\n" in partition
    assert "        nonlocal storage\n" in partition
    assert "        storage = dict(storage)\n" in partition
    assert "            nonlocal storage\n" in partition
    assert "unrelated" not in partition
    # Parsing alone does not check nonlocal binding ownership.
    table = symtable.symtable(partition, "<partition>", "exec")
    middle = table.get_children()[0].get_children()[0]
    assert middle.lookup("storage").is_nonlocal()
    assert middle.get_children()[0].lookup("storage").is_nonlocal()


def test_nested_partition_global_keeps_enclosing_global_write():
    source = '''storage: dict[str, int] = {}
def outer():
    global storage
    storage = dict(storage)
    def inner(key: str) -> int:
        return storage[key]
'''
    partition, _names = partition_authored_source(source, "outer.<locals>.inner")
    assert partition == source


def test_nested_partition_keeps_self_dependent_assignment_chain():
    source = '''def outer():
    storage: dict[str, int] = {}
    storage = dict(storage)
    storage = dict(storage)
    unrelated = object()
    def inner(key: str) -> int:
        return storage[key]
'''
    partition, _names = partition_authored_source(source, "outer.<locals>.inner")
    assert "    storage: dict[str, int] = {}\n" in partition
    assert partition.count("    storage = dict(storage)\n") == 2
    assert "unrelated" not in partition


def test_nested_partition_keeps_augassign_initialization():
    source = '''def outer():
    value: int = 1
    value += 2
    def inner() -> int:
        return value
'''
    partition, _names = partition_authored_source(source, "outer.<locals>.inner")
    assert partition == source


def test_nested_partition_keeps_annotation_before_plain_assignment():
    source = '''def outer():
    storage: dict[str, int]
    storage = {}
    def inner(key: str) -> int:
        return storage[key]
'''
    partition, _names = partition_authored_source(source, "outer.<locals>.inner")
    assert partition == source


def test_nested_partition_keeps_all_names_of_retained_nonlocal_declaration():
    source = '''def outer():
    value: int = 1
    other: int = 2
    def middle():
        nonlocal value, other
        value += 1
        def inner() -> int:
            return value
'''
    partition, _names = partition_authored_source(
        source, "outer.<locals>.middle.<locals>.inner",
    )
    assert partition == source
    symtable.symtable(partition, "<partition>", "exec")


def test_nested_partition_keeps_conditional_scope_declaration_shell():
    source = '''value: int = 1
def outer():
    if True:
        global value
    value += 2
    def inner() -> int:
        return value
'''
    partition, _names = partition_authored_source(source, "outer.<locals>.inner")
    assert partition == source
