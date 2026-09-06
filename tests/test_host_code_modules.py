import ast
import contextlib
import io
from pathlib import Path
from types import MappingProxyType

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.cpython_compile_ssa import NativeCompileSSAResult
from src.compiler.host_code_modules import (
    CachedHostCodeModule,
    HostCodeIdentity,
    extract_host_code_module,
    resolve_host_code_identity,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.transmogrifier.ssa import (
    BasicBlock, Function, IRModule, Instr, SSAValue,
    SSAMachineIndirectLink, SSAMachineIndirectTable,
)


def _host_result(name="native_len", retained_native_module=None):
    value = SSAValue(0, dtype="int64")
    function = Function(
        name,
        [value],
        {"entry": BasicBlock("entry", [Instr("Ret", [value], None)])},
    )
    return NativeCompileSSAResult(
        IRModule({name: function}),
        name,
        name,
        __file__,
        (1,),
        (),
        retained_native_module,
    )


def test_native_result_separates_machine_completeness_from_legalization():
    from src.compiler.cpython_compile_ssa import NativeCompileSSABlocker

    base = _host_result()
    shortfall = NativeCompileSSABlocker(
        1, 1, "native_len", "lowering", 0x1000,
        "operation remains exact machine-state SSA",
    )
    result = NativeCompileSSAResult(
        base.module, base.root_symbol, base.root_function, base.image_path,
        base.reached_function_rvas, (shortfall,), base.retained_native_module,
    )

    assert result.complete
    assert result.machine_state_complete
    assert result.uses_machine_state_dialect
    assert not result.repository_ssa_complete
    assert result.hard_blockers == ()
    assert result.legalization_shortfalls == (shortfall,)


def test_host_ssa_extraction_is_content_keyed_and_loaded_from_disk(
    tmp_path, monkeypatch,
):
    from src.compiler import host_code_modules as host

    image = tmp_path / "host.dll"
    image.write_bytes(b"host-image-v1")
    identity = HostCodeIdentity("pe-export", image, "native_len")
    calls = []
    monkeypatch.setattr(host, "resolve_host_code_identity", lambda value: identity)
    monkeypatch.setattr(host, "_implementation_digest", lambda: "impl-v1")
    monkeypatch.setattr(
        host,
        "lift_pe_export_to_ssa",
        lambda *args, **kwargs: calls.append((args, kwargs)) or _host_result(),
    )

    first = extract_host_code_module(len, cache_directory=tmp_path / "cache")
    second = extract_host_code_module(len, cache_directory=tmp_path / "cache")

    assert first is not None and second is not None
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.cache_key == second.cache_key
    assert len(calls) == 1


def test_host_ssa_disk_cache_preserves_nested_immutable_mappings(
    tmp_path, monkeypatch,
):
    from src.compiler import host_code_modules as host

    image = tmp_path / "host.dll"
    image.write_bytes(b"host-image-with-immutable-compiler-state")
    identity = HostCodeIdentity("pe-export", image, "native_len")
    result = _host_result()
    result.module.functions["native_len"].metadata["machine_contract"] = (
        MappingProxyType({"register": 4, "access": "inout"})
    )
    monkeypatch.setattr(host, "resolve_host_code_identity", lambda value: identity)
    monkeypatch.setattr(host, "_implementation_digest", lambda: "impl-v1")
    monkeypatch.setattr(host, "lift_pe_export_to_ssa", lambda *a, **k: result)

    first = extract_host_code_module(len, cache_directory=tmp_path / "cache")
    second = extract_host_code_module(len, cache_directory=tmp_path / "cache")

    assert first is not None and second is not None
    restored = second.result.module.functions["native_len"].metadata[
        "machine_contract"
    ]
    assert isinstance(restored, MappingProxyType)
    assert dict(restored) == {"register": 4, "access": "inout"}


def test_multiple_symbols_hash_shared_module_bytes_once(tmp_path, monkeypatch):
    from src.compiler import host_code_modules as host

    image = tmp_path / "host.dll"
    image.write_bytes(b"shared-host-image")
    host._module_content_digest.cache_clear()
    reads = []
    original = Path.read_bytes

    def observed(path):
        if Path(path) == image:
            reads.append(Path(path))
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", observed)
    monkeypatch.setattr(host, "_implementation_digest", lambda: "impl-v1")

    first = host._cache_key(HostCodeIdentity("pe-export", image, "first"))
    second = host._cache_key(HostCodeIdentity("pe-export", image, "second"))

    assert first != second
    assert reads == [image]


def test_host_library_retains_every_missing_pe_dependency_edge(monkeypatch, tmp_path):
    from src.compiler import host_code_modules as host

    result = _host_result()
    result.module.machine_indirect_table = SSAMachineIndirectTable((
        SSAMachineIndirectLink(
            "native_len", 0x1010, "call", "rip-relative-memory",
            0x2000, "pe-import", external_identity="other.dll!work",
        ),
        SSAMachineIndirectLink(
            "native_len", 0x1020, "call", "rip-relative-memory",
            0x2010, "pe-import", external_identity="other.dll!work",
        ),
    ))
    root = CachedHostCodeModule(
        HostCodeIdentity("pe-export", tmp_path / "root.dll", "root"),
        result, "root-key", tmp_path / "root.pickle", True,
    )
    monkeypatch.setattr(host, "extract_host_code_module", lambda *a, **k: root)

    library = host.extract_host_code_library(
        object(), dependency_provider=lambda _library, _requester: None,
    )

    assert library is not None
    assert library.units == (root,)
    assert tuple((edge.external_identity, edge.resolution) for edge in library.dependencies) == (
        ("other.dll!work", "module-unavailable:other.dll!work"),
        ("other.dll!work", "module-unavailable:other.dll!work"),
    )


def test_materialized_host_library_qualifies_units_and_links_resolved_import():
    from src.compiler.host_code_modules import (
        CachedHostCodeLibrary, HostCodeDependencyEdge,
        materialize_host_code_library,
    )

    root_result = _host_result("root")
    root_call = Instr(
        "Call", [], SSAValue(4, dtype="machine_call_state"),
        attributes={
            "machine_address": 0x1010,
            "external_identity": "dep.dll!work",
        },
    )
    root_result.module.functions["root"].blocks["entry"].instrs.insert(0, root_call)
    root_result.module.machine_indirect_table = SSAMachineIndirectTable((
        SSAMachineIndirectLink(
            "root", 0x1010, "call", "rip-relative-memory", 0x2000,
            "pe-import", external_identity="dep.dll!work",
        ),
    ))
    dependency_result = _host_result("work")
    root = CachedHostCodeModule(
        HostCodeIdentity("pe-export", __file__, "root"), root_result,
        "a" * 64, __file__, True,
    )
    dependency = CachedHostCodeModule(
        HostCodeIdentity("pe-dependency", __file__, "dep.dll!work"),
        dependency_result, "b" * 64, __file__, True,
    )
    library = CachedHostCodeLibrary(
        root.cache_key, (root, dependency),
        (HostCodeDependencyEdge(
            root.cache_key, "dep.dll!work", dependency.cache_key, "resolved",
            0x1010,
        ),),
    )

    module = materialize_host_code_library(library)

    assert set(module.functions) == {
        "pe_aaaaaaaaaaaaaaaa__root", "pe_bbbbbbbbbbbbbbbb__work",
    }
    call = module.functions["pe_aaaaaaaaaaaaaaaa__root"].blocks["entry"].instrs[0]
    assert call.attributes["callee"] == "pe_bbbbbbbbbbbbbbbb__work"
    (link,) = module.machine_indirect_table.links
    assert link.target_kind == "internal-function"
    assert link.target_function == "pe_bbbbbbbbbbbbbbbb__work"
    # Cached units remain immutable inputs to assembly.
    assert root_call.attributes.get("callee") is None
    assert library.root is root
    assert library.materialized_root_function == "pe_aaaaaaaaaaaaaaaa__root"


def test_host_library_ledgers_retain_every_unit_occurrence():
    from src.compiler.cpython_compile_ssa import NativeCompileSSABlocker
    from src.compiler.host_code_modules import CachedHostCodeLibrary

    root_result = _host_result("root")
    dependency_result = _host_result("work")
    first = NativeCompileSSABlocker(
        1, 1, "root", "decode", 0x1010, "first occurrence",
    )
    second = NativeCompileSSABlocker(
        2, 2, "work", "decode", 0x2020, "second occurrence",
    )
    root_result = NativeCompileSSAResult(
        root_result.module, "root", "root", __file__, (1,), (first,), None,
    )
    dependency_result = NativeCompileSSAResult(
        dependency_result.module, "work", "work", __file__, (2,), (second,), None,
    )
    root = CachedHostCodeModule(
        HostCodeIdentity("pe-export", __file__, "root"), root_result,
        "a" * 64, __file__, True,
    )
    dependency = CachedHostCodeModule(
        HostCodeIdentity("pe-dependency", __file__, "work"), dependency_result,
        "b" * 64, __file__, True,
    )
    library = CachedHostCodeLibrary(root.cache_key, (root, dependency), ())

    assert library.blockers == (first, second)
    assert library.hard_blockers == (first, second)
    assert not library.machine_state_complete
    assert not library.repository_ssa_complete


def test_materializer_does_not_link_cached_dependency_without_root_body():
    from src.compiler.cpython_compile_ssa import NativeCompileSSABlocker
    from src.compiler.host_code_modules import (
        CachedHostCodeLibrary, HostCodeDependencyEdge,
        materialize_host_code_library,
    )

    root_result = _host_result("root")
    call = Instr(
        "Call", [], SSAValue(4, dtype="machine_call_state"),
        attributes={
            "machine_address": 0x1010,
            "external_identity": "vcruntime140.dll!memset",
        },
    )
    root_result.module.functions["root"].blocks["entry"].instrs.insert(0, call)
    root_result.module.machine_indirect_table = SSAMachineIndirectTable((
        SSAMachineIndirectLink(
            "root", 0x1010, "call", "rip-relative-memory", 0x2000,
            "pe-import", external_identity="vcruntime140.dll!memset",
        ),
    ))
    missing = NativeCompileSSABlocker(
        1, 0x11730, "VCRUNTIME140_dll_memset", "decode", 0x180011822,
        "no instruction token for bytes c4 e3 7d 18",
    )
    empty_result = NativeCompileSSAResult(
        IRModule({}), "VCRUNTIME140.dll!memset", "VCRUNTIME140_dll_memset",
        Path(__file__), (0x11730,), (missing,), None,
    )
    root = CachedHostCodeModule(
        HostCodeIdentity("pe-export", Path(__file__), "root"), root_result,
        "a" * 64, Path(__file__), True,
    )
    dependency = CachedHostCodeModule(
        HostCodeIdentity(
            "pe-dependency", Path(__file__), "vcruntime140.dll!memset",
        ),
        empty_result, "b" * 64, Path(__file__), True,
    )
    library = CachedHostCodeLibrary(
        root.cache_key, (root, dependency),
        (HostCodeDependencyEdge(
            root.cache_key, "vcruntime140.dll!memset", dependency.cache_key,
            "resolved", 0x1010,
        ),),
    )

    module = materialize_host_code_library(library)

    copied_call = module.functions["pe_aaaaaaaaaaaaaaaa__root"].blocks["entry"].instrs[0]
    assert "callee" not in copied_call.attributes
    (retained_link,) = module.machine_indirect_table.links
    assert retained_link.target_kind == "pe-import"
    assert retained_link.external_identity == "vcruntime140.dll!memset"
    assert library.blockers == (missing,)


def test_additive_cache_migration_reuses_only_complete_scalar_unit(tmp_path, monkeypatch):
    import src.compiler.host_code_modules as host

    identity = HostCodeIdentity("pe-export", Path(__file__).resolve(), "root")
    old_digest = "1" * 64
    new_digest = "2" * 64
    old_key = host._cache_key_for_implementation(identity, old_digest)
    old_path = tmp_path / f"{old_key}.pickle"
    with old_path.open("wb") as stream:
        host._HostSSACachePickler(stream).dump({
            "schema": host.HOST_SSA_CACHE_SCHEMA,
            "key": old_key,
            "result": _host_result("root"),
        })
    monkeypatch.setattr(host, "_ADDITIVE_CACHE_IMPLEMENTATION_DIGESTS", (old_digest,))
    monkeypatch.setattr(host, "_implementation_digest", lambda: new_digest)
    monkeypatch.setattr(
        host, "lift_pe_export_to_ssa",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("complete scalar cache unit should migrate"),
        ),
    )

    migrated = host._extract_host_code_identity(
        identity, cache_directory=tmp_path,
    )

    assert migrated.cache_hit
    assert migrated.cache_key == host._cache_key_for_implementation(identity, new_digest)
    assert migrated.cache_path.exists()


def test_additive_cache_migration_rebuilds_incomplete_unit(tmp_path, monkeypatch):
    import src.compiler.host_code_modules as host
    from src.compiler.cpython_compile_ssa import NativeCompileSSABlocker

    identity = HostCodeIdentity("pe-export", Path(__file__).resolve(), "root")
    old_digest = "3" * 64
    new_digest = "4" * 64
    old_key = host._cache_key_for_implementation(identity, old_digest)
    blocker = NativeCompileSSABlocker(
        1, 1, "root", "decode", 0x1000, "unsupported VEX",
    )
    incomplete = NativeCompileSSAResult(
        IRModule({}), "root", "root", Path(__file__), (1,), (blocker,), None,
    )
    with (tmp_path / f"{old_key}.pickle").open("wb") as stream:
        host._HostSSACachePickler(stream).dump({
            "schema": host.HOST_SSA_CACHE_SCHEMA,
            "key": old_key,
            "result": incomplete,
        })
    rebuilt = _host_result("root")
    monkeypatch.setattr(host, "_ADDITIVE_CACHE_IMPLEMENTATION_DIGESTS", (old_digest,))
    monkeypatch.setattr(host, "_implementation_digest", lambda: new_digest)
    monkeypatch.setattr(host, "lift_pe_export_to_ssa", lambda *args, **kwargs: rebuilt)

    result = host._extract_host_code_identity(identity, cache_directory=tmp_path)

    assert not result.cache_hit
    assert result.result is rebuilt


def test_unresolved_dependency_prevents_false_library_completeness():
    from src.compiler.host_code_modules import (
        CachedHostCodeLibrary, HostCodeDependencyEdge,
    )

    result = _host_result("root")
    unit = CachedHostCodeModule(
        HostCodeIdentity("pe-export", __file__, "root"), result,
        "a" * 64, __file__, True,
    )
    missing = HostCodeDependencyEdge(
        unit.cache_key, "missing.dll!work", None,
        "module-unavailable:missing.dll!work",
    )
    library = CachedHostCodeLibrary(
        unit.cache_key, (unit,), (missing,),
    )

    assert result.repository_ssa_complete
    assert library.blockers == ()
    assert library.unresolved_dependencies == (missing,)
    assert not library.machine_state_complete
    assert not library.repository_ssa_complete


def test_dynamic_control_body_is_complete_without_closing_library_context():
    from src.compiler.cpython_compile_ssa import NativeCompileSSABlocker
    from src.compiler.host_code_modules import CachedHostCodeLibrary

    base = _host_result("dispatch")
    dynamic = NativeCompileSSABlocker(
        1, 1, "dispatch", "indirect-jump", 0x1010,
        "native jump target depends on register machine state",
    )
    result = NativeCompileSSAResult(
        base.module, base.root_symbol, base.root_function, base.image_path,
        base.reached_function_rvas, (dynamic,), None,
    )
    unit = CachedHostCodeModule(
        HostCodeIdentity("pe-export", Path(__file__), "dispatch"), result,
        "a" * 64, Path(__file__), True,
    )
    library = CachedHostCodeLibrary(unit.cache_key, (unit,), ())

    assert result.machine_state_complete
    assert library.machine_bodies_complete
    assert not library.dependency_context_complete
    assert not library.machine_state_complete


def test_resolved_import_removes_only_its_exact_effective_blocker():
    from src.compiler.cpython_compile_ssa import NativeCompileSSABlocker
    from src.compiler.host_code_modules import (
        CachedHostCodeLibrary, HostCodeDependencyEdge,
    )

    base = _host_result("root")
    first = NativeCompileSSABlocker(
        1, 1, "root", "external-machine-module", 0x1010,
        "first import", "dep.dll!work",
    )
    second = NativeCompileSSABlocker(
        2, 1, "root", "external-machine-module", 0x1020,
        "second import", "dep.dll!work",
    )
    root_result = NativeCompileSSAResult(
        base.module, base.root_symbol, base.root_function, base.image_path,
        base.reached_function_rvas, (first, second), None,
    )
    dependency_result = _host_result("work")
    root = CachedHostCodeModule(
        HostCodeIdentity("pe-export", __file__, "root"), root_result,
        "a" * 64, __file__, True,
    )
    dependency = CachedHostCodeModule(
        HostCodeIdentity("pe-dependency", __file__, "dep.dll!work"),
        dependency_result, "b" * 64, __file__, True,
    )
    resolved = HostCodeDependencyEdge(
        root.cache_key, "dep.dll!work", dependency.cache_key, "resolved",
        0x1010,
    )
    missing = HostCodeDependencyEdge(
        root.cache_key, "dep.dll!work", None,
        "module-unavailable:dep.dll!work", 0x1020,
    )
    library = CachedHostCodeLibrary(
        root.cache_key, (root, dependency), (resolved, missing),
    )

    assert library.blockers == (first, second)
    assert library.effective_blockers == (second,)
    assert library.hard_blockers == (second,)
    assert not library.repository_ssa_complete

    complete = CachedHostCodeLibrary(
        root.cache_key, (root, dependency), (
            resolved,
            HostCodeDependencyEdge(
                root.cache_key, "dep.dll!work", dependency.cache_key,
                "resolved", 0x1020,
            ),
        ),
    )
    assert complete.blockers == (first, second)
    assert complete.effective_blockers == ()
    assert complete.repository_ssa_complete


def test_cpython_pycfunction_resolver_finds_sre_compile_interior_entry():
    import _sre
    from src.compiler.binary_ingestion import parse_pe_image

    identity = resolve_host_code_identity(_sre.compile)

    assert identity is not None
    assert identity.provider == "cpython-pycfunction"
    assert identity.symbol == "_sre.compile"
    assert identity.entry_rva is not None
    assert identity.calling_convention == "cpython-pycfunction-flags:0x82"
    encoded = identity.module_path.read_bytes()
    image, _statistics = parse_pe_image(
        encoded, maximum_file_size=len(encoded)
    )
    owner = image.runtime_function_for_rva(identity.entry_rva)
    assert owner is not None
    assert owner.begin_rva <= identity.entry_rva < owner.end_rva


def test_ucrt_leaf_export_without_pdata_gets_bounded_executable_region():
    from src.compiler.binary_ingestion import parse_pe_image
    from src.compiler.cpython_compile_ssa import (
        _code_owner_for_entry, _code_region_for_owner,
    )
    from src.compiler.machine_reference_vocabulary import X86ReferenceDecoder

    path = Path(r"C:\Windows\System32\ucrtbase.dll")
    if not path.is_file():
        import pytest
        pytest.skip("Windows UCRT is unavailable")
    encoded = path.read_bytes()
    image, _statistics = parse_pe_image(
        encoded, maximum_file_size=len(encoded),
    )
    exported = image.export_by_name("free")
    if exported is None or exported.rva is None:
        import pytest
        pytest.skip("this UCRT has no concrete free export")
    if image.runtime_function_for_rva(int(exported.rva)) is not None:
        import pytest
        pytest.skip("this UCRT supplies pdata for free")

    owner, has_unwind = _code_owner_for_entry(image, int(exported.rva))
    _offset, region = _code_region_for_owner(image, owner)
    report = X86ReferenceDecoder().decode_cfg_report(
        region, base_address=int(image.image_base) + int(exported.rva),
    )

    assert has_unwind is False
    assert owner.begin_rva == int(exported.rva)
    assert owner.end_rva > owner.begin_rva
    assert report.failures == ()
    assert report.instructions
    assert report.instructions[-1].token.name == "JMP_REL32"
    assert sum(len(item.encoded) for item in report.instructions) < len(region)
    assert report.unreachable_spans


def test_interior_host_identity_normalizes_only_the_ssa_root_symbol():
    from src.compiler.cpython_compile_ssa import _function_name

    assert _function_name(0x100, "_sre.compile", 0x100) == "_sre_compile"
    assert _function_name(0x100, "_sre.compile", 0x200) == "cpython_00000200"
    assert _function_name(0x100, "Py_CompileString", 0x100) == "Py_CompileString"


def test_source_pursuit_binds_source_less_callable_to_host_ssa(
    tmp_path, monkeypatch,
):
    from src.compiler import host_code_modules as host

    identity = HostCodeIdentity("pe-export", tmp_path / "host.dll", "native_len")
    cached = CachedHostCodeModule(
        identity,
        _host_result(),
        "a" * 64,
        tmp_path / "cache-key.pickle",
        True,
    )
    from src.compiler.host_code_modules import CachedHostCodeLibrary
    library = CachedHostCodeLibrary(cached.cache_key, (cached,), ())
    monkeypatch.setattr(
        host, "resolve_host_code_identity",
        lambda value: identity if value is len else None,
    )
    monkeypatch.setattr(
        host, "extract_host_code_library",
        lambda value: library if value is len else None,
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse("def entry(values):\n    return len(values)\n"),
            resolve_unresolved_parents=True,
            pursuit_roots=("entry",),
        )
    reduce_abstract_tensor_topology(graph)

    call = next(
        data for _node, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
    )
    entry = graph.function_table.entry(call["attributes"]["callee_ref"])
    assert entry.metadata["implementation_kind"] == "decompiled-host-ssa"
    assert entry.metadata["host_ssa_root"] == (
        "pe_aaaaaaaaaaaaaaaa__native_len"
    )
    assert entry.metadata["host_ssa_cache_hit"] is True
    assert entry.metadata["host_ssa_library_cache_keys"] == (cached.cache_key,)
    assert entry.metadata["host_ssa_dependency_edges"] == ()
    assert entry.metadata["host_repository_ssa_complete"] is True
    assert "repository-ssa" in entry.metadata["implementation_variants"]
    assert "external_callee_ref" not in call["attributes"]


def test_source_pursuit_retains_native_module_as_an_explicit_variant(
    tmp_path, monkeypatch,
):
    from src.compiler import host_code_modules as host

    native = object()
    identity = HostCodeIdentity("pe-export", tmp_path / "host.dll", "native_len")
    cached = CachedHostCodeModule(
        identity, _host_result(retained_native_module=native),
        "a" * 64, tmp_path / "cache-key.pickle", True,
    )
    from src.compiler.host_code_modules import CachedHostCodeLibrary
    library = CachedHostCodeLibrary(cached.cache_key, (cached,), ())
    monkeypatch.setattr(
        host, "resolve_host_code_identity",
        lambda value: identity if value is len else None,
    )
    monkeypatch.setattr(
        host, "extract_host_code_library",
        lambda value: library if value is len else None,
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse("def entry(values):\n    return len(values)\n"),
            resolve_unresolved_parents=True, pursuit_roots=("entry",),
        )
    reduce_abstract_tensor_topology(graph)
    call = next(
        data for _node, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
    )
    entry = graph.function_table.entry(call["attributes"]["callee_ref"])

    assert entry.metadata["host_native_module"] is native
    assert entry.metadata["implementation_variants"] == (
        "repository-ssa", "retained-native-module",
    )
