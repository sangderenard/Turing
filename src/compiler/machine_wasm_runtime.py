"""Persistent host and automatic dispatcher for recompiled machine blocks.

The browser can instantiate machine-block WebAssembly directly.  The native
Python shell deliberately has no embedded language runtime, so it owns one
bounded Node worker and speaks a length-bounded JSON-lines control protocol.
Only Wasm emitted by :mod:`machine_block_recompiler` reaches that worker; guest
system activity continues to cross the capability-gated machine ports.
"""

from __future__ import annotations

import base64
from hashlib import sha256
import json
import os
import shutil
import subprocess
from threading import Lock
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .machine_block_recompiler import (
    JOURNAL_STRIDE,
    MachineBlockLoweringError,
)
from .machine_reference_vocabulary import (
    EffectiveAddressOperand, MachineSemanticToken, RegisterOperand,
)


class MachineWasmHostError(RuntimeError):
    """A compiled module could not execute inside its bounded host."""


_NODE_WORKER = r"""
import {createHash} from 'node:crypto';
import {createInterface} from 'node:readline';
const modules = new Map();
const input = createInterface({input: process.stdin, crlfDelay: Infinity});
const send = value => process.stdout.write(JSON.stringify(value) + '\n');
for await (const line of input) {
  let request;
  try {
    request = JSON.parse(line);
    if (request.action === 'close') { send({id: request.id, ok: true}); break; }
    if (request.action !== 'run') throw new Error('unknown worker action');
    let entry = modules.get(request.module_digest);
    let loaded = false;
    if (!entry) {
      const binary = Buffer.from(request.binary || '', 'base64');
      const digest = createHash('sha256').update(binary).digest('hex');
      if (digest !== request.module_digest) throw new Error('module digest mismatch');
      const {instance} = await WebAssembly.instantiate(binary, {});
      entry = instance.exports;
      if (!(entry.memory instanceof WebAssembly.Memory) || typeof entry.run !== 'function') {
        throw new Error('machine module exports do not match the execution ABI');
      }
      modules.set(request.module_digest, entry);
      loaded = true;
    }
    const state = Buffer.from(request.state, 'base64');
    const guest = Buffer.from(request.guest, 'base64');
    const journalBytes = Number(request.journal_bytes);
    const bytes = new Uint8Array(entry.memory.buffer);
    const stateOffset = 0, journalOffset = 1024;
    const guestOffset = Math.ceil((journalOffset + journalBytes) / 4096) * 4096;
    const limit = Math.max(stateOffset + state.length, journalOffset + journalBytes,
                           guestOffset + guest.length);
    if (!Number.isSafeInteger(journalBytes) || journalBytes < 0 || limit > bytes.length) {
      throw new Error('machine execution buffers exceed module memory');
    }
    bytes.fill(0, 0, limit);
    bytes.set(state, stateOffset);
    bytes.set(guest, guestOffset);
    entry.run(stateOffset, journalOffset, guestOffset);
    send({id: request.id, ok: true, loaded,
          journal: Buffer.from(entry.memory.buffer, journalOffset, journalBytes).toString('base64')});
  } catch (error) {
    send({id: request?.id ?? null, ok: false, error: String(error?.stack || error)});
  }
}
""".strip()


class NodeMachineWasmHost:
    """Cache and execute machine Wasm modules in one private Node process."""

    def __init__(
        self,
        executable: str | None = None,
        *,
        maximum_module_bytes: int = 4 * 1024 * 1024,
        maximum_guest_bytes: int = 64 * 1024,
        maximum_journal_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        resolved = executable or shutil.which("node")
        if not resolved:
            raise FileNotFoundError("Node.js is required for the node-wasm machine backend")
        if min(maximum_module_bytes, maximum_guest_bytes, maximum_journal_bytes) <= 0:
            raise ValueError("machine Wasm host limits must be positive")
        self.executable = str(resolved)
        self.maximum_module_bytes = int(maximum_module_bytes)
        self.maximum_guest_bytes = int(maximum_guest_bytes)
        self.maximum_journal_bytes = int(maximum_journal_bytes)
        self._process: subprocess.Popen[str] | None = None
        self._lock = Lock()
        self._request_id = 0
        self._sent_modules: set[str] = set()
        self.requests = 0
        self.module_loads = 0

    def _start(self) -> subprocess.Popen[str]:
        process = self._process
        if process is not None and process.poll() is None:
            return process
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        process = subprocess.Popen(
            [self.executable, "--input-type=module", "--eval", _NODE_WORKER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, creationflags=creationflags,
        )
        self._process = process
        self._sent_modules.clear()
        return process

    @property
    def statistics(self) -> Mapping[str, int]:
        return MappingProxyType({
            "requests": self.requests,
            "module_loads": self.module_loads,
            "resident_modules": len(self._sent_modules),
        })

    def execute(self, artifact: Any, state: Any) -> bytes:
        binary = bytes(artifact.binary)
        guest = artifact.pack_guest_memory(state)
        journal_bytes = artifact.covered_operation_count * JOURNAL_STRIDE
        if len(binary) > self.maximum_module_bytes:
            raise MachineWasmHostError("machine Wasm module exceeds host byte limit")
        if len(guest) > self.maximum_guest_bytes:
            raise MachineWasmHostError("machine Wasm guest mirror exceeds host byte limit")
        if journal_bytes > self.maximum_journal_bytes:
            raise MachineWasmHostError("machine Wasm journal exceeds host byte limit")
        digest = sha256(binary).hexdigest()
        with self._lock:
            process = self._start()
            assert process.stdin is not None and process.stdout is not None
            self._request_id += 1
            request = {
                "id": self._request_id,
                "action": "run",
                "module_digest": digest,
                "binary": (
                    base64.b64encode(binary).decode("ascii")
                    if digest not in self._sent_modules else ""
                ),
                "state": base64.b64encode(artifact.pack_state(state)).decode("ascii"),
                "guest": base64.b64encode(guest).decode("ascii"),
                "journal_bytes": journal_bytes,
            }
            try:
                process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
            except (BrokenPipeError, OSError) as error:
                raise MachineWasmHostError("machine Wasm worker communication failed") from error
            if not line:
                stderr = ""
                if process.stderr is not None:
                    stderr = process.stderr.read(4096)
                raise MachineWasmHostError(
                    "machine Wasm worker exited without a response"
                    + (f": {stderr.strip()}" if stderr.strip() else "")
                )
            try:
                response = json.loads(line)
            except json.JSONDecodeError as error:
                raise MachineWasmHostError("machine Wasm worker returned invalid JSON") from error
            if response.get("id") != self._request_id:
                raise MachineWasmHostError("machine Wasm worker response ID mismatch")
            if not response.get("ok"):
                raise MachineWasmHostError(str(response.get("error", "unknown Wasm failure")))
            journal = base64.b64decode(response.get("journal", ""), validate=True)
            if len(journal) != journal_bytes:
                raise MachineWasmHostError("machine Wasm worker returned a truncated journal")
            self._sent_modules.add(digest)
            self.requests += 1
            self.module_loads += int(bool(response.get("loaded")))
            return journal

    def close(self) -> None:
        with self._lock:
            process, self._process = self._process, None
            self._sent_modules.clear()
            if process is None or process.poll() is not None:
                return
            try:
                if process.stdin is not None:
                    self._request_id += 1
                    process.stdin.write(json.dumps({
                        "id": self._request_id, "action": "close",
                    }) + "\n")
                    process.stdin.flush()
                process.wait(timeout=2)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)


class MachineWasmBlockDispatcher:
    """Select, cache, execute, and commit the safe prefix at the current RIP."""

    def __init__(self, host: NodeMachineWasmHost, *, maximum_cached_artifacts: int = 4096) -> None:
        if maximum_cached_artifacts <= 0:
            raise ValueError("compiled artifact cache bound must be positive")
        self.host = host
        self.maximum_cached_artifacts = int(maximum_cached_artifacts)
        self._artifacts: dict[tuple[Any, ...], Any] = {}
        self._denied: dict[tuple[Any, ...], None] = {}
        self._denied_semantics: dict[str, int] = {}
        self._denied_tokens: dict[str, int] = {}
        self._denied_reasons: dict[str, int] = {}
        self.attempts = 0
        self.executions = 0
        self.fallbacks = 0
        self.committed_instructions = 0

    @property
    def statistics(self) -> Mapping[str, int]:
        return MappingProxyType({
            "attempts": self.attempts,
            "executions": self.executions,
            "fallbacks": self.fallbacks,
            "committed_instructions": self.committed_instructions,
            "cached_artifacts": len(self._artifacts),
            "denied_blocks": len(self._denied),
            **{f"host_{key}": value for key, value in self.host.statistics.items()},
            **{
                f"denied_semantic_{key}": value
                for key, value in sorted(self._denied_semantics.items())
            },
            **{
                f"denied_token_{key}": value
                for key, value in sorted(self._denied_tokens.items())
            },
            **{
                f"denied_reason_{key}": value
                for key, value in sorted(self._denied_reasons.items())
            },
        })

    @staticmethod
    def _key(block: Any, state: Any, limit: int) -> tuple[Any, ...]:
        instruction = block.operations[0].instruction
        first = instruction.semantic
        specialized: Any = (
            (int(state.registers[4]), tuple(state.call_stack), bool(state.termination_requested))
            if first in {MachineSemanticToken.DIRECT_RELATIVE_CALL, MachineSemanticToken.RETURN}
            else None
        )
        operands = tuple(getattr(instruction, "operands", ()))
        if (
            first in {MachineSemanticToken.INDIRECT_CALL, MachineSemanticToken.INDIRECT_JUMP}
            and operands and isinstance(operands[0], RegisterOperand)
        ):
            register = int(operands[0].register)
            specialized = (
                "indirect", register, int(state.registers[register]),
                (
                    int(state.registers[4]), tuple(state.call_stack),
                    bool(state.termination_requested),
                ) if first is MachineSemanticToken.INDIRECT_CALL else None,
            )
        if first in {MachineSemanticToken.STACK_PUSH, MachineSemanticToken.STACK_POP}:
            specialized = ("stack", int(state.registers[4]))
        dynamic_operands = tuple(
            operand for operand in operands
            if isinstance(operand, EffectiveAddressOperand)
            and (operand.base is not None or operand.index is not None)
        )
        if dynamic_operands:
            registers = tuple(sorted({
                int(register): int(state.registers[int(register)])
                for operand in dynamic_operands
                for register in (operand.base, operand.index)
                if register is not None
            }.items()))
            prefixes = tuple(getattr(instruction, "legacy_prefixes", ()))
            specialized = (
                "memory", registers,
                int(state.fs_base) if 0x64 in prefixes else None,
                int(state.gs_base) if 0x65 in prefixes else None,
            )
        return (str(block.code_digest), int(limit), specialized)

    def execute(
        self,
        core: Any,
        maximum_instructions: int,
        *,
        transition_observer: Callable[[], None] | None = None,
    ) -> tuple[Any, ...] | None:
        limit = int(maximum_instructions)
        if limit <= 0 or core.state.halted:
            return None
        try:
            block = core.executor.translated_block(core.state.pc, core.state)
        except (KeyError, ValueError):
            self.fallbacks += 1
            return None
        key = self._key(block, core.state, limit)
        if key in self._denied:
            self.fallbacks += 1
            return None
        artifact = self._artifacts.get(key)
        if artifact is None:
            self.attempts += 1
            try:
                artifact = core.executor.recompile_block_wasm(
                    core.state.pc, core.state,
                    maximum_instructions=limit,
                )
            except (KeyError, MachineBlockLoweringError, ValueError) as error:
                semantic = str(block.operations[0].instruction.semantic.name)
                self._denied_semantics[semantic] = self._denied_semantics.get(semantic, 0) + 1
                token = str(getattr(
                    getattr(block.operations[0].instruction, "token", None),
                    "name", "UNKNOWN",
                ))
                self._denied_tokens[token] = self._denied_tokens.get(token, 0) + 1
                reason = str(error).replace("\n", " ")[:240]
                self._denied_reasons[reason] = self._denied_reasons.get(reason, 0) + 1
                if len(self._denied) >= self.maximum_cached_artifacts:
                    self._denied.pop(next(iter(self._denied)))
                self._denied[key] = None
                self.fallbacks += 1
                return None
            if len(self._artifacts) >= self.maximum_cached_artifacts:
                self._artifacts.pop(next(iter(self._artifacts)))
            self._artifacts[key] = artifact
        try:
            journal = self.host.execute(artifact, core.state)
        except (KeyError, IndexError):
            # An unmapped specialized guest window must preserve the reference
            # interpreter's structured trap rather than becoming a host fault.
            self.fallbacks += 1
            return None
        results = core.commit_recompiled_journal(
            artifact, journal, transition_observer=transition_observer,
        )
        self.executions += 1
        self.committed_instructions += len(results)
        return results

    def close(self) -> None:
        self.host.close()


__all__ = [
    "MachineWasmBlockDispatcher", "MachineWasmHostError", "NodeMachineWasmHost",
]
