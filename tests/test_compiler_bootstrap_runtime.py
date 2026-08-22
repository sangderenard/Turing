from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

from src.common.tensors.source_realization import (
    authored_source_realization,
    deployed_with_authored_fallback,
)
from src.compiler import compiler_bootstrap_runtime as runtime
from src.compiler import project_compilation_product
from tools.compile_project_catalogue import _publish_bootstrap_runtime_state


def test_receipt_declared_bootstrap_reproves_installs_and_retains_source(
    tmp_path, monkeypatch,
):
    native_root = tmp_path / "native"
    native_root.mkdir()
    (native_root / "stage.dll").write_bytes(b"native")
    (native_root / "native-verification.json").write_text(json.dumps({
        "status": "verified",
        "activation_adapter": "test-stage-v1",
    }), encoding="utf-8")

    def authored(value):
        return ("source", value)

    owner = SimpleNamespace(stage=authored)

    def adapter(product, qualified_name):
        assert product.root == tmp_path
        assert qualified_name == "stage"

        def deployed(value):
            deployed.__turing_native_route_counts__["native"] += 1
            return ("native", value)

        deployed.__turing_native_verification__ = {
            "status": "verified",
            "activation_adapter": "test-stage-v1",
            "native_probe_count": 2,
            "fallback_probe_count": 1,
        }
        deployed.__turing_native_route_counts__ = {
            "native": 2, "fallback": 1,
        }
        return owner, deployed

    class Product:
        root = Path(tmp_path)
        links = {"stage": {"native_library": "native/stage.dll"}}

        def install_callable(
            self, qualified_name, selected_owner, deployed, *,
            targeted_source_fallback=False,
        ):
            assert targeted_source_fallback is True
            installed = deployed_with_authored_fallback(
                selected_owner.stage,
                deployed,
                identity=qualified_name,
                targeted=True,
            )
            selected_owner.stage = installed
            return installed

    monkeypatch.setattr(
        project_compilation_product,
        "open_project_compilation_product",
        lambda _path: Product(),
    )
    monkeypatch.setitem(runtime._ACTIVATION_ADAPTERS, "test-stage-v1", adapter)
    runtime._ACTIVE_DEPLOYMENTS.clear()

    activation, = runtime.activate_compiler_bootstrap_products((tmp_path,))

    assert activation.status == "verified"
    assert owner.stage(3) == ("native", 3)
    with authored_source_realization(targets=("stage",)):
        assert owner.stage(3) == ("source", 3)
    state, = runtime.compiler_bootstrap_runtime_state()
    assert state["post_activation_native_calls"] == 1
    assert state["post_activation_fallback_calls"] == 0


def test_bootstrap_product_environment_is_a_deterministic_path_set(
    tmp_path, monkeypatch,
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.delenv(runtime.COMPILER_BOOTSTRAP_PRODUCTS_ENV, raising=False)

    selected = runtime.set_compiler_bootstrap_products((
        first, second, first,
    ))

    assert selected == (first.resolve(), second.resolve())
    assert runtime.compiler_bootstrap_product_paths() == selected


def test_qualified_scalar_adapter_replays_persisted_probes(
    tmp_path, monkeypatch,
):
    native_root = tmp_path / "native"
    native_root.mkdir()
    library = native_root / "leaf.dll"
    library.write_bytes(b"native")
    (native_root / "native-verification.json").write_text(json.dumps({
        "probes": [
            {"arguments": "()", "keywords": "{'value': -3}"},
            {"arguments": "()", "keywords": "{'value': 5}"},
        ],
    }), encoding="utf-8")
    owner = SimpleNamespace()

    def authored(value):
        return value + 1

    observed = {}

    class Product:
        root = tmp_path
        links = {"leaf": {
            "source_module": "compiler_leaf_module",
            "native_library": "native/leaf.dll",
        }}

        def verify_native_scalar_callable(
            self, qualified_name, selected, probes, *, activation_adapter=None,
        ):
            observed.update({
                "qualified_name": qualified_name,
                "selected": selected,
                "probes": tuple(probes),
                "activation_adapter": activation_adapter,
            })
            return authored

    monkeypatch.setattr(
        project_compilation_product,
        "_resolve_product_callable",
        lambda _module, _name: (owner, authored),
    )

    selected_owner, deployed = runtime._activate_qualified_scalar(
        Product(), "leaf",
    )

    assert selected_owner is owner
    assert deployed is authored
    assert observed["probes"] == ({"value": -3}, {"value": 5})
    assert observed["activation_adapter"] == "qualified-scalar-call-v1"


def test_registry_pins_only_receipt_verified_installable_products(
    tmp_path, monkeypatch,
):
    product_root = tmp_path / "product"
    native_root = product_root / "native"
    native_root.mkdir(parents=True)
    source = tmp_path / "source.py"
    source.write_text("def leaf(value): return value\n", encoding="utf-8")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = product_root / "manifest.json"
    manifest.write_text(json.dumps({
        "source": source.as_posix(),
        "source_sha256": source_sha256,
    }), encoding="utf-8")
    api = native_root / "leaf.api.yaml"
    api.write_text("api", encoding="utf-8")
    library = native_root / "leaf.dll"
    library.write_bytes(b"native")
    receipt = native_root / "native-verification.json"
    receipt.write_text(json.dumps({
        "status": "verified",
        "activation_adapter": "qualified-scalar-call-v1",
        "api_sha256": hashlib.sha256(api.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }), encoding="utf-8")

    class Product:
        root = product_root
        manifest = {
            "source": source.as_posix(),
            "source_sha256": source_sha256,
        }
        links = {"leaf": {
            "native_library": "native/leaf.dll",
            "native_api": "native/leaf.api.yaml",
        }}

    monkeypatch.setattr(
        project_compilation_product,
        "open_project_compilation_product",
        lambda _path: Product(),
    )
    registry = tmp_path / "registry.json"

    runtime.publish_compiler_bootstrap_products(
        (product_root,), registry_path=registry,
    )

    payload = json.loads(registry.read_text(encoding="utf-8"))
    assert payload["products"][0]["manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert payload["products"][0]["installable"] == [{
        "qualified_name": "leaf",
        "activation_adapter": "qualified-scalar-call-v1",
        "verification_receipt": "native/native-verification.json",
    }]
    assert runtime.registered_compiler_bootstrap_product_paths(registry) == (
        product_root.resolve(),
    )


def test_registered_activation_is_revision_idempotent_and_stale_safe(
    tmp_path, monkeypatch,
):
    product = tmp_path / "product"
    product.mkdir()
    manifest = product / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({
        "schema": runtime.COMPILER_BOOTSTRAP_REGISTRY_SCHEMA,
        "products": [{
            "product": product.as_posix(),
            "manifest_sha256": hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest(),
        }],
    }), encoding="utf-8")
    monkeypatch.setenv(runtime.COMPILER_BOOTSTRAP_REGISTRY_ENV, str(registry))
    calls = []
    monkeypatch.setattr(
        runtime,
        "activate_compiler_bootstrap_products",
        lambda paths: calls.append(tuple(paths)) or (),
    )
    monkeypatch.setattr(runtime, "_ACTIVATED_REGISTRY_SHA256", None)
    runtime._REGISTRY_FAILURES.clear()

    runtime.activate_registered_compiler_bootstraps()
    runtime.activate_registered_compiler_bootstraps()

    assert calls == [(product.resolve(),)]
    manifest.write_text('{"changed": true}', encoding="utf-8")
    registry.write_text(
        registry.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    assert runtime.activate_registered_compiler_bootstraps() == ()
    state = runtime.compiler_bootstrap_registry_state()
    assert state["failures"][0]["error_type"] == "ValueError"
    assert "changed or vanished" in state["failures"][0]["error"]


def test_bootstrap_receipt_writer_cannot_clobber_worker_failure(tmp_path):
    failure = tmp_path / "failure.json"
    activation = tmp_path / "bootstrap-activation.json"
    failure.write_text(json.dumps({
        "schema": "turing.resolved-process-graph-unit-failure.v1",
        "status": "failed",
    }), encoding="utf-8")

    _publish_bootstrap_runtime_state(
        activation,
        lambda: ({"qualified_name": "compiled_stage"},),
    )

    assert json.loads(failure.read_text(encoding="utf-8"))["status"] == "failed"
    assert json.loads(activation.read_text(encoding="utf-8")) == {
        "schema": "turing.compiler-bootstrap-activation.v1",
        "products": [{"qualified_name": "compiled_stage"}],
    }
