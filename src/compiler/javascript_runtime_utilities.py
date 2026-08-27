"""Named, dependency-ordered JavaScript utilities supplied by the emitter.

Utilities are ordinary source modules selected by semantic identity.  Their
closure is deterministic and content-addressed so generated hosts can publish,
deduplicate, cache, and eventually replace them independently of user code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class JavaScriptRuntimeUtility:
    identity: str
    source: str
    dependencies: tuple[str, ...] = ()
    capability: str = "runtime"
    exports: tuple[tuple[str, str], ...] = ()
    inline: str = "neutral"
    frequency: str = "startup"
    allocation_risk: str = "unknown"

    @property
    def content_key(self) -> str:
        digest = hashlib.sha256(self.source.encode("utf-8")).hexdigest()
        return f"javascript-utility:sha256:{digest}"

    def to_data(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "content_key": self.content_key,
            "dependencies": list(self.dependencies),
            "capability": self.capability,
            "exports": [
                {"name": name, "symbol": symbol} for name, symbol in self.exports
            ],
            "performance": {
                "inline": self.inline,
                "frequency": self.frequency,
                "allocation_risk": self.allocation_risk,
            },
        }


WASM_REGISTRY_SOURCE = r"""
class TuringWasmModuleRegistry {
  constructor(modules = []) {
    this.descriptors = new Map();
    this.instances = new Map();
    modules.forEach(descriptor => this.register(descriptor));
  }
  register(descriptor) {
    if (!descriptor?.content_key || !descriptor?.binary_base64) {
      throw new TypeError("WASM modules require content_key and binary_base64");
    }
    const previous = this.descriptors.get(descriptor.content_key);
    if (previous && previous.binary_base64 !== descriptor.binary_base64) {
      throw new Error(`WASM content identity collision: ${descriptor.content_key}`);
    }
    this.descriptors.set(descriptor.content_key, descriptor);
    return descriptor.content_key;
  }
  async instantiate(contentKey, imports = {}) {
    if (this.instances.has(contentKey)) return this.instances.get(contentKey);
    const descriptor = this.descriptors.get(contentKey);
    if (!descriptor) throw new Error(`unknown WASM module ${contentKey}`);
    const pending = (async () => {
      const bytes = Uint8Array.from(atob(descriptor.binary_base64), value => value.charCodeAt(0));
      const result = await WebAssembly.instantiate(bytes, imports);
      return result.instance;
    })();
    this.instances.set(contentKey, pending);
    try { return await pending; }
    catch (error) { this.instances.delete(contentKey); throw error; }
  }
}
function turingCreateWasmRegistry(modules = []) {
  return new TuringWasmModuleRegistry(modules);
}
""".strip()


WORLD_REGISTRY_SOURCE = r"""
class TuringWorldRegistry {
  constructor(world) {
    if (!world?.identity || !Array.isArray(world.objects)) {
      throw new TypeError("world registry requires identity and objects");
    }
    this.world = world;
    this.objects = new Map();
    this.children = new Map();
    this.semanticParts = new Map();
    this.objectRuntimeIds = new Map();
    this.partRuntimeIds = new Map();
    this.runtimeObjects = [null];
    this.runtimeParts = [null];
    world.objects.forEach(object => {
      if (!object.identity || this.objects.has(object.identity)) {
        throw new Error(`duplicate or empty world identity ${object.identity}`);
      }
      this.objects.set(object.identity, object);
      if (!this.children.has(object.parent)) this.children.set(object.parent, []);
      this.children.get(object.parent).push(object.identity);
      (object.semantic_parts || []).forEach(part => {
        if (this.semanticParts.has(part.identity)) {
          throw new Error(`duplicate semantic part identity ${part.identity}`);
        }
        this.semanticParts.set(part.identity, {object: object.identity, ...part});
      });
    });
    const specialization = world.identity_specialization || {
      authority: "authored-string-identity", missing_runtime_id: 0,
      objects: [...this.objects.keys()].map((identity, index) =>
        ({identity, runtime_id: index + 1})),
      semantic_parts: [...this.semanticParts.keys()].map((identity, index) =>
        ({identity, runtime_id: index + 1})),
    };
    if (specialization.authority !== "authored-string-identity" ||
        specialization.missing_runtime_id !== 0) {
      throw new Error("world identity specialization has an incompatible authority");
    }
    (specialization?.objects || []).forEach(entry => {
      const object = this.objects.get(entry.identity);
      if (!object || entry.runtime_id <= 0 || this.runtimeObjects[entry.runtime_id]) {
        throw new Error(`invalid runtime object identity ${entry.runtime_id}`);
      }
      this.objectRuntimeIds.set(entry.identity, entry.runtime_id);
      this.runtimeObjects[entry.runtime_id] = object;
    });
    (specialization?.semantic_parts || []).forEach(entry => {
      const part = this.semanticParts.get(entry.identity);
      if (!part || entry.runtime_id <= 0 || this.runtimeParts[entry.runtime_id]) {
        throw new Error(`invalid runtime semantic-part identity ${entry.runtime_id}`);
      }
      this.partRuntimeIds.set(entry.identity, entry.runtime_id);
      this.runtimeParts[entry.runtime_id] = part;
    });
  }
  object(identity) { return this.objects.get(identity) || null; }
  part(identity) { return this.semanticParts.get(identity) || null; }
  objectRuntimeId(identity) { return this.objectRuntimeIds.get(identity) || 0; }
  partRuntimeId(identity) { return this.partRuntimeIds.get(identity) || 0; }
  objectFromRuntimeId(runtimeId) { return this.runtimeObjects[runtimeId] || null; }
  partFromRuntimeId(runtimeId) { return this.runtimeParts[runtimeId] || null; }
  containedBy(identity) {
    return [...(this.children.get(identity) || [])].map(child => this.objects.get(child));
  }
  ancestry(identity) {
    const result = [];
    let object = this.object(identity);
    const visited = new Set();
    while (object && !visited.has(object.identity)) {
      visited.add(object.identity); result.push(object);
      object = this.object(object.parent);
    }
    return result;
  }
}
function turingCreateWorldRegistry(world) { return new TuringWorldRegistry(world); }
""".strip()


REVISION_CHANNEL_SOURCE = r"""
class TuringRevisionChannel {
  constructor(identity, revision = 0) {
    this.identity = String(identity);
    this.revision = Number(revision);
    this.listeners = new Set();
  }
  subscribe(listener) {
    if (typeof listener !== "function") throw new TypeError("revision listener must be callable");
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
  publish(record) {
    const next = Number(record?.revision);
    if (!Number.isInteger(next) || next <= this.revision) {
      throw new RangeError(`revision must increase beyond ${this.revision}`);
    }
    this.revision = next;
    const event = Object.freeze({...record, channel: this.identity, revision: next});
    this.listeners.forEach(listener => listener(event));
    return event;
  }
}
function turingCreateRevisionChannel(identity, revision = 0) {
  return new TuringRevisionChannel(identity, revision);
}
""".strip()


JAVASCRIPT_RUNTIME_UTILITIES = {
    utility.identity: utility for utility in (
        JavaScriptRuntimeUtility(
            "turing.wasm.registry", WASM_REGISTRY_SOURCE,
            capability="webassembly-module-cache",
            exports=(("create", "turingCreateWasmRegistry"),),
            inline="avoid", allocation_risk="dynamic-cache",
        ),
        JavaScriptRuntimeUtility(
            "turing.world.registry", WORLD_REGISTRY_SOURCE,
            capability="identity-and-containment-index",
            exports=(("create", "turingCreateWorldRegistry"),),
            inline="avoid", allocation_risk="bounded-by-world",
        ),
        JavaScriptRuntimeUtility(
            "turing.revision.channel", REVISION_CHANNEL_SOURCE,
            capability="monotonic-edit-publication",
            exports=(("create", "turingCreateRevisionChannel"),),
            inline="prefer", frequency="per-edit", allocation_risk="bounded-event",
        ),
    )
}


def javascript_utility_closure(
    requested: Iterable[str],
) -> tuple[JavaScriptRuntimeUtility, ...]:
    """Resolve dependencies once in deterministic request/definition order."""

    ordered: list[JavaScriptRuntimeUtility] = []
    resolved: set[str] = set()
    active: set[str] = set()

    def visit(identity: str) -> None:
        if identity in resolved:
            return
        if identity in active:
            raise ValueError(f"JavaScript utility dependency cycle at {identity!r}")
        try:
            utility = JAVASCRIPT_RUNTIME_UTILITIES[identity]
        except KeyError as error:
            raise KeyError(f"unknown JavaScript runtime utility {identity!r}") from error
        active.add(identity)
        for dependency in utility.dependencies:
            visit(dependency)
        active.remove(identity)
        resolved.add(identity)
        ordered.append(utility)

    for identity in dict.fromkeys(map(str, requested)):
        visit(identity)
    return tuple(ordered)


def render_javascript_utilities(requested: Iterable[str]) -> str:
    utilities = javascript_utility_closure(requested)
    return "\n\n".join(
        f"// turing-runtime-utility: {utility.identity} [{utility.content_key}]\n"
        f"{utility.source}"
        for utility in utilities
    )


__all__ = [
    "JAVASCRIPT_RUNTIME_UTILITIES", "JavaScriptRuntimeUtility",
    "javascript_utility_closure", "render_javascript_utilities",
]
