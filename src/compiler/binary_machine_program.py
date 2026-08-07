"""Compiled binary-machine program interior.

The implementation is authored in Python and enters the ordinary compiler as
program source.  It is not a packaged Python interpreter. Runtime subject
bytes remain in the register-aware machine pipeline and never become cards.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from .machine_chip_layout import (
    FixedProgramCacheLayout,
    RegisterBankLayout,
    build_fixed_program_cache_layout,
    build_register_bank_layout,
)
from .machine_execution import (
    MachineEffectHandler,
    MachineExecutionOrchestrator,
    MachineIndirectTargetHandler,
    MachinePredicateHandler,
    MachineVirtualMulticore,
    MachineExternalCallRequest,
    MachineExternalCallCompletion,
    MachineExternalMemoryWrite,
    MachineExternalReference,
    MachineExternalThreadSpawn,
    MachineDispatchPlan,
    MACHINE_LOADER_CALLBACK_RETURN,
)
from .amd64_machine_semantics import (
    EXTERNAL_TARGET_BASE,
    build_external_references,
    build_initial_machine_state,
    complete_external_call_state,
    condition_holds,
    default_effect_handlers,
    indirect_target,
)
from .machine_program_graph import raise_pe_to_token_multigraph
from .machine_state_buffer import (
    ExternalMachineClock,
    FreeRunningMachineRunner,
    MachineRunDirection,
    MachineSnapshotLayout,
    MachineSnapshotTripleBuffer,
    SubjectOutputBuffer,
    SubjectOutputFormat,
    SubjectOutputKind,
    machine_state_outputs,
)
from .machine_system_ports import CapabilityGatedExternalPort
from .virtual_filesystem import VirtualFileEffect, VirtualFileSystemState
from .shell_io import VirtualFileSystemContract
from .machine_system_tape import MachineSystemTape, MachineTapeLinkedModule
from .machine_module_linker import (
    MachineImportBinding,
    MachineLinkedPEModule,
    build_pe_link_plan,
)


class SubjectDeviceBuffers:
    """Atomically replaceable subject outputs sampled at snapshot publication."""

    def __init__(self) -> None:
        self._outputs: tuple[SubjectOutputBuffer, ...] = ()

    def publish(self, outputs: Sequence[SubjectOutputBuffer]) -> None:
        self._outputs = tuple(outputs)

    def snapshot(self) -> tuple[SubjectOutputBuffer, ...]:
        return self._outputs


@dataclass(slots=True)
class BinaryMachineProgram:
    """One loaded subject, its clocked reversible cores, and observation ABI."""

    program: object
    machine: MachineVirtualMulticore
    register_layout: RegisterBankLayout
    cache_layout: FixedProgramCacheLayout
    snapshots: MachineSnapshotTripleBuffer
    runner: FreeRunningMachineRunner
    clock: ExternalMachineClock
    devices: SubjectDeviceBuffers
    external_references: tuple[MachineExternalReference, ...]
    external_reference_targets: dict[int, MachineExternalReference]
    dispatch_plans: list[MachineDispatchPlan]
    system_tape: MachineSystemTape
    linked_modules: tuple[MachineLinkedPEModule, ...]
    import_bindings: tuple[MachineImportBinding, ...]
    maximum_thread_stack_bytes: int

    @classmethod
    def from_program(
        cls,
        program,
        *,
        core_count: int = 1,
        transitions_per_second: float = 60.0,
        maximum_transitions_per_tick: int = 1_000_000,
        maximum_outputs: int = 4,
        maximum_output_bytes: int = 4 * 1024 * 1024,
        maximum_hot_history_states: int | None = 4096,
        shared_memory: bool = True,
        maximum_shared_memory_commits: int = 4096,
        effect_handlers: Mapping[int, MachineEffectHandler] | None = None,
        predicate_handler: MachinePredicateHandler | None = None,
        indirect_target_handler: MachineIndirectTargetHandler | None = None,
        virtual_filesystem: VirtualFileSystemState | None = None,
        virtual_environment: Mapping[str, str] | None = None,
        load_address: int | None = None,
        linked_modules: Sequence[MachineLinkedPEModule] = (),
        initial_active_cores: int | None = None,
        maximum_thread_stack_bytes: int = 1024 * 1024,
        machine_block_backend: str | None = None,
    ) -> "BinaryMachineProgram":
        using_default_machine = (
            effect_handlers is None
            and predicate_handler is None
            and indirect_target_handler is None
        )
        active_core_count = (
            core_count if initial_active_cores is None else int(initial_active_cores)
        )
        if not 1 <= active_core_count <= core_count:
            raise ValueError("initial_active_cores must be within virtual core capacity")
        if maximum_thread_stack_bytes < 4096:
            raise ValueError("maximum_thread_stack_bytes must be at least one page")
        if using_default_machine:
            effect_handlers = default_effect_handlers()
            predicate_handler = condition_holds
            indirect_target_handler = indirect_target
        if virtual_filesystem is None and getattr(program.image, "encoded", None) is not None:
            virtual_filesystem = VirtualFileSystemState.create(
                VirtualFileSystemContract(current_directory="/c"),
                files={"/program/subject.exe": bytes(program.image.encoded)},
            )
        runtime_base = (
            int(program.image.image_base)
            if load_address is None else int(load_address)
        )
        active_modules = tuple(linked_modules)
        if active_modules and not using_default_machine:
            raise ValueError("linked PE modules require the default machine semantics")
        link_plan = (
            build_pe_link_plan(
                program,
                primary_load_address=runtime_base,
                modules=active_modules,
            )
            if active_modules else None
        )
        external_references = (
            link_plan.external_references
            if link_plan is not None else (
                build_external_references(program) if using_default_machine else ()
            )
        )
        references_by_target = {
            reference.target_address: reference for reference in external_references
        }
        executor = MachineExecutionOrchestrator(
            program,
            effect_handlers=effect_handlers,
            predicate_handler=predicate_handler,
            indirect_target_handler=indirect_target_handler,
            external_target_resolver=references_by_target.get,
            load_address=load_address,
            linked_programs=tuple(
                (module.program, module.load_address)
                for module in active_modules
            ),
        )
        initial_states = None
        if using_default_machine and getattr(program.image, "encoded", None) is not None:
            initial = build_initial_machine_state(
                program, external_references=external_references,
                virtual_filesystem=virtual_filesystem,
                environment_state=virtual_environment,
                load_address=load_address,
                additional_images=tuple(
                    (module.program.image, module.load_address)
                    for module in active_modules
                ),
                import_targets=(
                    None if link_plan is None else link_plan.import_targets
                ),
                module_handle_targets=(
                    None if link_plan is None else link_plan.module_handle_targets
                ),
            )
            initial_states = tuple(
                replace(
                    initial,
                    system_state={
                        **dict(initial.system_state),
                        "windows.thread.active": int(index < active_core_count),
                        "windows.thread.id": index + 1,
                        "windows.thread.auxiliary": 0,
                    },
                )
                for index in range(core_count)
            )
        machine = MachineVirtualMulticore.create(
            executor, core_count=core_count, initial_states=initial_states,
            shared_memory=shared_memory,
            maximum_shared_memory_commits=maximum_shared_memory_commits,
        )
        registers = build_register_bank_layout(core_count)
        snapshot_layout = MachineSnapshotLayout.build(
            registers,
            core_count=core_count,
            maximum_outputs=maximum_outputs,
            maximum_output_bytes=maximum_output_bytes,
        )
        snapshots = MachineSnapshotTripleBuffer(snapshot_layout, registers)
        devices = SubjectDeviceBuffers()
        system_tape = MachineSystemTape(
            bytes(getattr(program.image, "encoded", b"")), core_count,
        )
        system_tape.linked_modules = [
            MachineTapeLinkedModule(
                module.requested_library, module.load_address, module.encoded,
            )
            for module in active_modules
        ]
        system_tape.import_bindings = list(
            () if link_plan is None else link_plan.bindings
        )
        for reference in external_references:
            system_tape.catalog_external_reference(reference)
        for index, core in enumerate(machine.cores):
            system_tape.append(index, core.state, position=core.position, event="load")

        compiled_dispatcher = None
        if machine_block_backend is not None:
            backend = str(machine_block_backend).strip().casefold()
            if backend != "node-wasm":
                raise ValueError(f"unsupported machine block backend {machine_block_backend!r}")
            from .machine_wasm_runtime import (
                MachineWasmBlockDispatcher, NodeMachineWasmHost,
            )
            compiled_dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())
        runner = FreeRunningMachineRunner(
            machine, snapshots, output_provider=devices.snapshot,
            compiled_dispatcher=compiled_dispatcher,
        )
        result = cls(
            program=program,
            machine=machine,
            register_layout=registers,
            cache_layout=build_fixed_program_cache_layout(
                program, base_offset=snapshot_layout.byte_size,
            ),
            snapshots=snapshots,
            runner=runner,
            clock=ExternalMachineClock(
                transitions_per_second=transitions_per_second,
                maximum_transitions_per_tick=maximum_transitions_per_tick,
            ),
            devices=devices,
            external_references=external_references,
            external_reference_targets=references_by_target,
            dispatch_plans=[],
            system_tape=system_tape,
            linked_modules=active_modules,
            import_bindings=(
                () if link_plan is None else link_plan.bindings
            ),
            maximum_thread_stack_bytes=int(maximum_thread_stack_bytes),
        )
        runner.transition_observer = result._record_transition
        snapshots.core_annotation_provider = result._core_annotation_color
        result.configure_hot_history(maximum_hot_history_states)
        return result

    @property
    def recompilation_statistics(self) -> Mapping[str, int]:
        dispatcher = self.runner.compiled_dispatcher
        return MappingProxyType({}) if dispatcher is None else dispatcher.statistics

    def enable_recompilation(self, backend: str = "node-wasm") -> None:
        """Install automatic safe-prefix recompilation in the active runner."""

        if self.runner.running:
            raise RuntimeError("pause the machine before changing its block backend")
        normalized = str(backend).strip().casefold()
        if normalized != "node-wasm":
            raise ValueError(f"unsupported machine block backend {backend!r}")
        if self.runner.compiled_dispatcher is not None:
            return
        from .machine_wasm_runtime import MachineWasmBlockDispatcher, NodeMachineWasmHost
        self.runner.compiled_dispatcher = MachineWasmBlockDispatcher(NodeMachineWasmHost())

    def disable_recompilation(self) -> None:
        """Close the compiled host and return to translated Python blocks."""

        if self.runner.running:
            raise RuntimeError("pause the machine before changing its block backend")
        dispatcher = self.runner.compiled_dispatcher
        self.runner.compiled_dispatcher = None
        if dispatcher is not None:
            dispatcher.close()

    def close(self) -> None:
        """Stop owned execution workers and release the optional Wasm host."""

        if self.runner.running:
            self.runner.stop()
        self.disable_recompilation()

    def _cold_history_states(
        self, core_index: int, start: int, end: int, *,
        root_position: int | None = None, root_state=None,
    ) -> tuple[object, ...]:
        """Decode the largest contiguous suffix requested from exact tape."""

        if not 0 <= core_index < len(self.machine.cores):
            raise IndexError("machine core index is out of range")
        if not 0 <= start < end:
            return ()
        tape = self.system_tape
        try:
            sequence = tape.latest_sequence(core_index, position=end - 1)
        except TypeError:  # legacy segmented-store call shape
            sequence = tape.latest_sequence(core_index)
        lineage = tape.lineage_states(core_index, sequence=sequence)
        by_position = {
            int(node.position): state
            for node, state in lineage
            if start <= int(node.position) < end
        }
        if (
            root_position is not None and root_state is not None
            and start <= root_position < end
        ):
            by_position.setdefault(root_position, root_state)
        cursor = end - 1
        suffix = []
        while cursor >= start and cursor in by_position:
            suffix.append(by_position[cursor])
            cursor -= 1
        return tuple(reversed(suffix))

    def configure_hot_history(self, maximum_states: int | None) -> None:
        """Bound resident reversible states and hydrate older ones from tape."""

        for core_index, core in enumerate(self.machine.cores):
            root_state = core._states[0]
            root_position = core.hot_history_range[0]
            core.configure_hot_history(
                maximum_states,
                cold_history_loader=lambda start, end, index=core_index,
                root=root_state, root_at=root_position: (
                    self._cold_history_states(
                        index, start, end,
                        root_position=root_at, root_state=root,
                    )
                ),
            )

    @classmethod
    def load_pe(
        cls,
        binary: bytes | bytearray | memoryview,
        *,
        maximum_file_size: int,
        dependency_modules: Mapping[str, bytes | bytearray | memoryview] | None = None,
        dependency_load_addresses: Mapping[str, int] | None = None,
        dependency_provider: Callable[[str], bytes | bytearray | memoryview | None] | None = None,
        maximum_dependency_modules: int = 64,
        maximum_dependency_bytes: int = 256 * 1024 * 1024,
        **program_options,
    ) -> "BinaryMachineProgram":
        """Consume subject bytes through the existing machine decompiler."""

        program = raise_pe_to_token_multigraph(
            binary, maximum_file_size=maximum_file_size,
        )
        if maximum_dependency_modules <= 0 or maximum_dependency_bytes <= 0:
            raise ValueError("dependency module limits must be positive")
        if dependency_provider is not None and not callable(dependency_provider):
            raise TypeError("dependency_provider must be callable")
        if dependency_modules or dependency_provider is not None:
            if "linked_modules" in program_options:
                raise ValueError(
                    "dependency acquisition and linked_modules cannot both be supplied"
                )
            requested_bases = {
                str(name).casefold(): int(base)
                for name, base in (dependency_load_addresses or {}).items()
            }
            unused_bases = set(requested_bases)
            linked: list[MachineLinkedPEModule] = []
            total_dependency_bytes = 0

            def base_for(library: str, preferred: int) -> int:
                aliases = (library.casefold(), library.casefold().removesuffix(".dll"))
                for alias in aliases:
                    if alias in requested_bases:
                        unused_bases.discard(alias)
                        return requested_bases[alias]
                    dll_alias = alias + ".dll"
                    if dll_alias in requested_bases:
                        unused_bases.discard(dll_alias)
                        return requested_bases[dll_alias]
                return preferred

            def add_module(library: str, module_binary) -> None:
                nonlocal total_dependency_bytes
                payload = bytes(module_binary)
                total_dependency_bytes += len(payload)
                if total_dependency_bytes > maximum_dependency_bytes:
                    raise ValueError("dependency images exceed maximum_dependency_bytes")
                if len(linked) >= maximum_dependency_modules:
                    raise ValueError("dependency image count exceeds maximum_dependency_modules")
                dependency = raise_pe_to_token_multigraph(
                    payload, maximum_file_size=maximum_file_size,
                )
                linked.append(MachineLinkedPEModule(
                    str(library), dependency,
                    base_for(str(library), int(dependency.image.image_base)),
                ))

            for library, module_binary in sorted(
                (dependency_modules or {}).items(),
                key=lambda item: item[0].casefold(),
            ):
                add_module(str(library), module_binary)

            attempted: set[str] = set()
            if dependency_provider is not None:
                for _round in range(maximum_dependency_modules + 1):
                    provisional = build_pe_link_plan(
                        program,
                        primary_load_address=int(program_options.get(
                            "load_address", program.image.image_base,
                        )),
                        modules=tuple(linked),
                    )
                    missing = tuple(dict.fromkeys(
                        reference.library
                        for reference in provisional.external_references
                        if reference.library.casefold() not in attempted
                    ))
                    if not missing:
                        break
                    progress = False
                    for library in missing:
                        attempted.add(library.casefold())
                        supplied = dependency_provider(library)
                        if supplied is None:
                            continue
                        add_module(library, supplied)
                        progress = True
                    if not progress:
                        break
                else:
                    raise ValueError("recursive dependency acquisition did not converge")
            if unused_bases:
                raise ValueError(
                    "dependency load addresses name modules without supplied bytes: "
                    + ", ".join(sorted(unused_bases))
                )
            program_options["linked_modules"] = tuple(linked)
        elif dependency_load_addresses:
            raise ValueError("dependency load addresses require dependency acquisition")
        return cls.from_program(program, **program_options)

    @classmethod
    def load_system_tape(
        cls,
        tape: MachineSystemTape | str | Path,
        *,
        maximum_file_size: int,
        sequence: int | None = None,
        **program_options,
    ) -> "BinaryMachineProgram":
        """Rebuild the executor and resume each core at the tape's last state."""

        active_tape = tape if isinstance(tape, MachineSystemTape) else MachineSystemTape.read(tape)
        program_options.setdefault("core_count", active_tape.core_count)
        probe_state = active_tape.resume_state(0, sequence=sequence)
        recorded_base = probe_state.system_state.get("windows.loader.image_base")
        if recorded_base is not None:
            requested_base = program_options.get("load_address", recorded_base)
            if int(requested_base) != int(recorded_base):
                raise ValueError(
                    "requested PE load address conflicts with the exact tape"
                )
            program_options.setdefault("load_address", int(recorded_base))
        if active_tape.linked_modules:
            if "dependency_modules" in program_options or "linked_modules" in program_options:
                raise ValueError("exact tape already owns its linked dependency modules")
            program_options["dependency_modules"] = {
                module.library: module.binary for module in active_tape.linked_modules
            }
            program_options["dependency_load_addresses"] = {
                module.library: module.load_address for module in active_tape.linked_modules
            }
        result = cls.load_pe(
            active_tape.subject_binary,
            maximum_file_size=maximum_file_size,
            **program_options,
        )
        if tuple(result.import_bindings) != tuple(active_tape.import_bindings):
            raise ValueError("reconstructed PE import link plan differs from exact tape")
        if len(result.machine.cores) != active_tape.core_count:
            raise ValueError("system tape core count does not match resumed machine")
        for reference in active_tape.external_references:
            existing = result.external_reference_targets.get(reference.target_address)
            if existing is not None and existing != reference:
                raise ValueError(
                    f"taped external target {reference.target_address:#x} conflicts with PE imports"
                )
            result.external_reference_targets[reference.target_address] = reference
            if reference not in result.external_references:
                result.external_references = (*result.external_references, reference)
        dispatch_targets = tuple(dict.fromkeys(
            int(target)
            for record in active_tape.records
            if record.get("event") == "runtime_dispatch"
            for target in record.get("metadata", {}).get("targets", ())
        ))
        if dispatch_targets:
            plan = result.machine.cores[0].executor.install_dispatch_targets(dispatch_targets)
            result.dispatch_plans.append(plan)
            unresolved = tuple(
                target for target in dispatch_targets
                if target not in result.machine.cores[0].executor.instructions
            )
            if unresolved:
                raise ValueError(
                    "taped runtime dispatch plan no longer validates: "
                    + ", ".join(f"{target:#x}" for target in unresolved)
                )
        for index, core in enumerate(result.machine.cores):
            state = active_tape.resume_state(index, sequence=sequence)
            limit = len(active_tape.records) - 1 if sequence is None else int(sequence)
            state_sequence = next(
                int(record["sequence"])
                for record in reversed(active_tape.records)
                if int(record["sequence"]) <= limit and int(record["core"]) == index
            )
            core.restore_cold_tip(
                state,
                position=int(active_tape.records[state_sequence]["position"]),
            )
            for request in state.external_requests:
                reference = request.reference
                result.external_reference_targets[reference.target_address] = reference
                if reference not in result.external_references:
                    result.external_references = (*result.external_references, reference)
        result.system_tape = active_tape
        if result.machine.cores:
            result._sync_devices(result.machine.cores[0].state)
        return result

    @classmethod
    def load_segmented_system_tape(
        cls,
        store,
        *,
        maximum_file_size: int,
        sequence: int | None = None,
        **program_options,
    ) -> "BinaryMachineProgram":
        """Resume from bounded content-addressed segments without full hydration."""

        from .machine_tape_segments import SegmentedMachineTapeStore

        active = store if isinstance(store, SegmentedMachineTapeStore) else SegmentedMachineTapeStore(store)
        program_options.setdefault("core_count", active.core_count)
        probe_state = active.resume_state(0, sequence=sequence)
        recorded_base = probe_state.system_state.get("windows.loader.image_base")
        if recorded_base is not None:
            requested_base = program_options.get("load_address", recorded_base)
            if int(requested_base) != int(recorded_base):
                raise ValueError(
                    "requested PE load address conflicts with the exact segmented tape"
                )
            program_options.setdefault("load_address", int(recorded_base))
        if active.linked_modules:
            if "dependency_modules" in program_options or "linked_modules" in program_options:
                raise ValueError(
                    "exact segmented tape already owns its linked dependency modules"
                )
            program_options["dependency_modules"] = {
                module.library: module.binary for module in active.linked_modules
            }
            program_options["dependency_load_addresses"] = {
                module.library: module.load_address for module in active.linked_modules
            }
        result = cls.load_pe(
            active.subject_binary,
            maximum_file_size=maximum_file_size,
            **program_options,
        )
        if tuple(result.import_bindings) != tuple(active.import_bindings):
            raise ValueError(
                "reconstructed PE import link plan differs from exact segmented tape"
            )
        for reference in active.external_references:
            existing = result.external_reference_targets.get(reference.target_address)
            if existing is not None and existing != reference:
                raise ValueError(
                    f"segmented external target {reference.target_address:#x} conflicts with PE imports"
                )
            result.external_reference_targets[reference.target_address] = reference
            if reference not in result.external_references:
                result.external_references = (*result.external_references, reference)
        if active.runtime_dispatch_indexed:
            dispatch_targets = active.runtime_dispatch_targets
        else:
            dispatch_targets = tuple(dict.fromkeys(
                int(target)
                for record in active.records
                if record.get("event") == "runtime_dispatch"
                for target in record.get("metadata", {}).get("targets", ())
            ))
            active.index_runtime_dispatch_targets(dispatch_targets)
            active.flush()
        if dispatch_targets:
            plan = result.machine.cores[0].executor.install_dispatch_targets(dispatch_targets)
            result.dispatch_plans.append(plan)
            unresolved = tuple(
                target for target in dispatch_targets
                if target not in result.machine.cores[0].executor.instructions
            )
            if unresolved:
                raise ValueError(
                    "segmented runtime dispatch plan no longer validates: "
                    + ", ".join(f"{target:#x}" for target in unresolved)
                )
        for index, core in enumerate(result.machine.cores):
            state = active.resume_state(index, sequence=sequence)
            state_sequence = active.latest_sequence(index, limit=sequence)
            core.restore_cold_tip(
                state,
                position=int(active.record(state_sequence)["position"]),
            )
        active.begin_append()
        result.system_tape = active
        if result.machine.cores:
            result._sync_devices(result.machine.cores[0].state)
        return result

    def save_system_tape(self, path: str | Path) -> Path:
        return self.system_tape.write(path)

    def begin_segmented_system_tape(
        self,
        root: str | Path,
        *,
        records_per_segment: int = 256,
    ):
        """Replace the live JSONL tape with an appendable segmented store.

        The existing exact prefix is streamed through a temporary JSONL file;
        subsequent observer events append directly into bounded immutable
        objects.  The target must be new so a run can never overwrite another
        tape lineage accidentally.
        """

        from .machine_tape_segments import SegmentedMachineTapeStore

        target = Path(root)
        if target.exists():
            raise FileExistsError(
                f"refusing to replace existing segmented tape {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, staging_name = tempfile.mkstemp(
            prefix=target.name + ".",
            suffix=".bootstrap.jsonl",
            dir=target.parent,
        )
        os.close(descriptor)
        staging = Path(staging_name)
        try:
            self.save_system_tape(staging)
            store = SegmentedMachineTapeStore.import_jsonl(
                staging, target,
                records_per_segment=records_per_segment,
            )
        finally:
            staging.unlink(missing_ok=True)
        store.begin_append()
        self.system_tape = store
        return store

    def annotate_tape(self, feature: str, message: str, **details):
        """Attach a colored debugger/user feature to the current tape moment."""

        return self.system_tape.annotate(feature, message, **details)

    def _record_transition(self, active_machine, event: str) -> None:
        shared_commit = (
            active_machine.last_shared_memory_commit
            if event == "forward" else None
        )
        shared_metadata = (
            {"shared_memory_commit": dict(shared_commit.to_mapping())}
            if shared_commit is not None
            and shared_commit.core_positions == tuple(
                core.position for core in active_machine.cores
            )
            else None
        )
        for index, core in enumerate(active_machine.cores):
            self.system_tape.append(
                index, core.state, position=core.position, event=event,
                metadata=shared_metadata,
            )
            if event == "forward" and core.latest_edge is not None:
                edge = core.latest_edge
                if (
                    edge.status.name == "BLOCKED_EFFECT"
                    and edge.instruction is not None
                ):
                    instruction = edge.instruction
                    self.system_tape.annotate(
                        "instruction_set_compatibility",
                        "No AMD64 semantic handler is installed for "
                        f"{instruction.token.name} / {instruction.semantic.name} at RIP "
                        f"{instruction.address:#x}",
                        color="red", severity="error", core=index,
                        position=core.position, address=instruction.address,
                        metadata={
                            "architecture": "windows-amd64",
                            "instruction_token": instruction.token.name,
                            "semantic_token": instruction.semantic.name,
                            "encoded": instruction.encoded.hex(),
                            "compatibility": "unsupported",
                        },
                    )

    def _core_annotation_color(self, core: int, position: int) -> int:
        try:
            sequence = self.system_tape.latest_sequence(core)
        except IndexError:
            return 0
        return self.system_tape.annotation_color_rgba8(sequence, core=core)

    def _sync_devices(self, state) -> None:
        self.devices.publish(
            machine_state_outputs(state)[:self.snapshots.layout.maximum_outputs],
        )

    def set_speed(self, transitions_per_second: float) -> None:
        self.clock.set_speed(transitions_per_second)

    def set_direction(self, direction: MachineRunDirection) -> None:
        self.runner.set_direction(direction)

    def tick(self, elapsed_seconds: float) -> int:
        """Advance from one external shell tick and publish exactly one flip."""

        return self.runner.regulated_tick(self.clock, elapsed_seconds)

    def pending_external_requests(self, core_index: int = 0) -> tuple[MachineExternalCallRequest, ...]:
        return self.machine.cores[core_index].state.external_requests

    def inject_device_bytes(
        self,
        device: str,
        data: bytes | bytearray | memoryview,
        *,
        append: bool = True,
        core_index: int = 0,
    ) -> None:
        """Journal bytes supplied by a shell to one reversible guest device."""

        if not device:
            raise ValueError("device input requires a device name")
        if not 0 <= core_index < len(self.machine.cores):
            raise IndexError("machine core index is out of range")
        core = self.machine.cores[core_index]
        active = core.state
        device_state = dict(active.device_state)
        previous = device_state.get(device, b"") if append else b""
        device_state[device] = previous + bytes(data)
        generations = dict(active.device_generations)
        generations[device] = generations.get(device, 0) + 1
        core.commit_shell_effect(replace(
            active,
            device_state=device_state,
            device_generations=generations,
        ))
        self._sync_devices(core.state)
        self.system_tape.append(
            core_index, core.state, position=core.position,
            event="shell_device_input",
            metadata={"device": device, "bytes": len(data), "append": bool(append)},
        )

    def inject_console_input(
        self,
        data: str | bytes | bytearray | memoryview,
        *,
        core_index: int = 0,
    ) -> None:
        """Append UTF-8 terminal input for blocking ReadFile/ReadConsole calls."""

        payload = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        self.inject_device_bytes("console.input", payload, core_index=core_index)

    def apply_shell_file_effect(
        self,
        effect: VirtualFileEffect,
        *,
        core_index: int = 0,
    ) -> None:
        """Journal one shell-authorized mutation of the guest VFS."""

        if not 0 <= core_index < len(self.machine.cores):
            raise IndexError("machine core index is out of range")
        core = self.machine.cores[core_index]
        active = core.state
        if active.virtual_filesystem is None:
            raise RuntimeError("machine has no virtual filesystem")
        filesystem = active.virtual_filesystem.apply(effect)
        core.commit_shell_effect(replace(active, virtual_filesystem=filesystem))
        self.system_tape.append(
            core_index, core.state, position=core.position,
            event="shell_filesystem_effect",
            metadata={
                "operation": effect.operation,
                "path": effect.path,
                "destination": effect.destination,
                "bytes": len(effect.data),
            },
        )

    def create_path_forest(
        self,
        *,
        core_index: int = 0,
        maximum_heads: int = 1024,
        exact_state_store=None,
    ):
        """Create a possible-world fork axis from one current guest thread."""

        from .machine_path_forest import MachinePathForest

        if not 0 <= core_index < len(self.machine.cores):
            raise IndexError("machine core index is out of range")
        try:
            sequence = self.system_tape.latest_sequence(core_index)
        except IndexError:
            sequence = None
        return MachinePathForest(
            self.machine.cores[core_index],
            tape_sequence=sequence,
            maximum_heads=maximum_heads,
            exact_state_store=exact_state_store,
        )

    def complete_external_request(
        self,
        request_id: int,
        *,
        result: int = 0,
        memory_writes: Sequence[MachineExternalMemoryWrite] = (),
        core_index: int = 0,
    ):
        """Apply a shell completion and retain it in reversible history."""

        core = self.machine.cores[core_index]
        state = core.state
        matches = tuple(
            request for request in state.external_requests
            if request.request_id == int(request_id)
        )
        if len(matches) != 1:
            raise KeyError(f"external request {request_id} is not pending on core {core_index}")
        completed = complete_external_call_state(
            state,
            MachineExternalCallCompletion(
                request_id=int(request_id),
                result=int(result),
                memory_writes=tuple(memory_writes),
            ),
        )
        completed = core.executor.reconcile_external_state(state, completed)
        core.commit_external_completion(completed)
        synchronized = self.machine.synchronize_shared_memory(core_index)
        self._sync_devices(core.state)
        self.system_tape.append(
            core_index, core.state, position=core.position, event="external_completion",
        )
        for synchronized_core in synchronized:
            target = self.machine.cores[synchronized_core]
            self.system_tape.append(
                synchronized_core, target.state, position=target.position,
                event="shared_memory_sync",
                metadata={"source_core": core_index, "request_id": int(request_id)},
            )
        return completed

    def service_external_requests(
        self,
        port: CapabilityGatedExternalPort,
        *,
        core_index: int = 0,
    ) -> int:
        """Apply every currently serviceable request through an allowlisted port."""

        serviced = 0
        while True:
            pending = self.pending_external_requests(core_index)
            if not pending:
                return serviced
            request = pending[0]
            completion = port.handle(request, self.machine.cores[core_index].state)
            if completion is None:
                if port.wait_kind(request) == "thread_wait":
                    waiting_core = self.machine.cores[core_index]
                    if int(waiting_core.state.system_state.get(
                        "windows.thread.waiting_request", 0,
                    )) != int(request.request_id):
                        system_state = dict(waiting_core.state.system_state)
                        system_state["windows.thread.waiting_request"] = int(
                            request.request_id
                        )
                        waiting_core.commit_shell_effect(replace(
                            waiting_core.state,
                            system_state=MappingProxyType(system_state),
                        ))
                        self.system_tape.append(
                            core_index, waiting_core.state,
                            position=waiting_core.position,
                            event="thread_wait",
                            metadata={
                                "request_id": request.request_id,
                                "handle": request.arguments[0],
                                "timeout": request.arguments[1],
                            },
                        )
                return serviced
            if completion.request_id != request.request_id:
                raise ValueError("external port returned a mismatched request id")
            if completion.resolution is not None:
                reference = self.register_external_reference(
                    completion.resolution.library,
                    completion.resolution.symbol,
                )
                completion = replace(
                    completion,
                    result=reference.target_address,
                    resolution=None,
                )
            core = self.machine.cores[core_index]
            unknown_calls = tuple(
                address for address in completion.guest_calls
                if address not in core.executor.instructions
            )
            if unknown_calls:
                plan = core.executor.install_dispatch_targets(unknown_calls)
                self.dispatch_plans.append(plan)
                still_unknown = tuple(
                    address for address in unknown_calls
                    if address not in core.executor.instructions
                )
                if still_unknown:
                    details = "; ".join(plan.failure_reasons)
                    raise ValueError(
                        "external port requested unprovable guest callbacks: "
                        + ", ".join(f"{address:#x}" for address in still_unknown)
                        + (f" ({details})" if details else "")
                    )
            source_state = core.state
            completed_state = complete_external_call_state(source_state, completion)
            completed_state, thread_activations = self._activate_thread_spawns(
                completed_state, completion.thread_spawns,
                parent_core_index=core_index,
            )
            completed_state = core.executor.reconcile_external_state(
                source_state, completed_state,
            )
            core.commit_external_completion(completed_state)
            for child_index, child_state, _spawn in thread_activations:
                self.machine.cores[child_index].commit_shell_effect(child_state)
            synchronized = self.machine.synchronize_shared_memory(core_index)
            self._sync_devices(core.state)
            self.system_tape.append(
                core_index, core.state, position=core.position,
                event="external_completion",
                metadata={
                    "deployments": [
                        {
                            "deployment_id": item.deployment_id,
                            "kind": item.kind,
                            "requested_reference": item.requested_reference,
                            "resolved_reference": item.resolved_reference,
                            "executor_reference": item.executor_reference,
                            "exit_code": item.exit_code,
                            "execution_units": item.execution_units,
                            "child_tape_schema": item.child_tape_schema,
                            "child_tape_digest": item.child_tape_digest,
                            "child_tape_reference": item.child_tape_reference,
                        }
                        for item in completion.deployments
                    ],
                } if completion.deployments else None,
            )
            parent_sequence = len(self.system_tape.records) - 1
            for child_index, child_state, spawn in thread_activations:
                child = self.machine.cores[child_index]
                self.system_tape.append(
                    child_index, child_state, position=child.position,
                    event="thread_spawn",
                    dependencies=(("thread_spawn_request", parent_sequence),),
                    metadata={
                        "parent_core": core_index,
                        "thread_id": spawn.thread_id,
                        "handle": spawn.handle,
                        "start_address": spawn.start_address,
                        "parameter": spawn.parameter,
                        "stack_size": spawn.stack_size,
                    },
                )
            for synchronized_core in synchronized:
                target = self.machine.cores[synchronized_core]
                self.system_tape.append(
                    synchronized_core, target.state, position=target.position,
                    event="shared_memory_sync",
                    metadata={
                        "source_core": core_index,
                        "request_id": int(completion.request_id),
                    },
                )
            serviced += 1

    def _activate_thread_spawns(
        self,
        parent_state,
        spawns: Sequence[MachineExternalThreadSpawn],
        *,
        parent_core_index: int,
    ):
        """Allocate and activate parked cores for admitted CreateThread calls."""

        if not spawns:
            return parent_state, ()
        idle = [
            index for index, candidate in enumerate(self.machine.cores)
            if index != parent_core_index
            and not int(candidate.state.system_state.get("windows.thread.active", 1))
        ]
        if len(idle) < len(spawns):
            raise RuntimeError("CreateThread exceeds configured virtual core capacity")
        executor = self.machine.cores[parent_core_index].executor
        memory = parent_state.memory
        parent_system = dict(parent_state.system_state)
        cursor = int(parent_system["windows.system_arena_cursor"])
        limit = int(parent_system["windows.system_arena_limit"])
        tls_count = int(parent_system.get("windows.loader.tls_module_count", 0))
        prepared = []

        def allocate(size: int, alignment: int = 16) -> int:
            nonlocal cursor, memory
            cursor = (cursor + alignment - 1) & ~(alignment - 1)
            address = cursor
            cursor += int(size)
            if cursor > limit:
                raise RuntimeError("CreateThread exceeds the reversible system arena")
            memory = memory.map_zeroes(address, int(size))
            return address

        for child_index, spawn in zip(idle, spawns):
            if spawn.start_address not in executor.instructions:
                plan = executor.install_dispatch_targets((spawn.start_address,))
                self.dispatch_plans.append(plan)
            if spawn.start_address not in executor.instructions:
                raise ValueError(
                    f"CreateThread target {spawn.start_address:#x} is not executable guest code"
                )
            requested_stack = int(spawn.stack_size) or 64 * 1024
            if requested_stack > self.maximum_thread_stack_bytes:
                raise ValueError("CreateThread stack exceeds maximum_thread_stack_bytes")
            stack_size = max(4096, (requested_stack + 4095) & ~4095)
            stack_base = allocate(stack_size, 4096)
            stack_top = stack_base + stack_size
            teb_base = allocate(4096, 4096)
            tls_vector = allocate(max(8, tls_count * 8), 16) if tls_count else 0
            child_tls: list[tuple[int, int, int]] = []
            for tls_index in range(tls_count):
                source_tls = int(parent_system[f"windows.loader.tls.{tls_index}.base"])
                tls_size = int(parent_system[f"windows.loader.tls.{tls_index}.size"])
                tls_base = allocate(tls_size, 16)
                memory = memory.map_bytes(
                    tls_base,
                    bytes(memory[source_tls + offset] for offset in range(tls_size)),
                )
                memory = memory.write_unsigned(tls_vector + tls_index * 8, 64, tls_base)
                child_tls.append((tls_index, tls_base, tls_size))
            rsp = stack_top - 8
            memory = memory.write_unsigned(rsp, 64, 0)
            memory = memory.write_unsigned(teb_base + 0x30, 64, teb_base)
            memory = memory.write_unsigned(
                teb_base + 0x60, 64,
                memory.read_unsigned(parent_state.gs_base + 0x60, 64),
            )
            memory = memory.write_unsigned(teb_base + 0x58, 64, tls_vector)
            prepared.append((
                child_index, spawn, stack_base, stack_size, rsp,
                teb_base, tls_vector, tuple(child_tls),
            ))

        parent_system["windows.system_arena_cursor"] = cursor
        parent_system["windows.thread.count"] = int(
            parent_system.get("windows.thread.count", 1),
        ) + len(prepared)
        for child_index, spawn, *_rest in prepared:
            prefix = f"windows.thread.{spawn.thread_id}"
            parent_system[f"{prefix}.core"] = child_index
            parent_system[f"{prefix}.handle"] = spawn.handle
            parent_system[f"{prefix}.active"] = 1
        parent_state = replace(
            parent_state,
            memory=memory,
            system_state=MappingProxyType(parent_system),
        )

        activations = []
        for (
            child_index, spawn, stack_base, stack_size, rsp,
            teb_base, tls_vector, child_tls,
        ) in prepared:
            child_system = dict(parent_system)
            child_system.update({
                "windows.thread.active": 1,
                "windows.thread.auxiliary": 1,
                "windows.thread.id": spawn.thread_id,
                "windows.thread.handle": spawn.handle,
                "windows.thread.stack_base": stack_base,
                "windows.thread.stack_limit": stack_base + stack_size,
                "windows.thread.start_parameter": spawn.parameter,
                "windows.loader.tls_vector": tls_vector,
                "windows.loader.startup_reason": 2,  # DLL_THREAD_ATTACH
                "windows.loader.startup_call_index": 0,
                "windows.loader.startup_direction": 1,
                "windows.loader.completion_action": 0,
                "windows.loader.tls_callback_index": 0,
                "windows.loader.entrypoint": spawn.start_address,
                "windows.loader.tls_callbacks_complete": 0,
                "windows.loader.startup_calls_complete": 0,
                "windows.thread.detach_started": 0,
                "windows.thread.detach_complete": 0,
            })
            child_system.pop("windows.loader.startup_failure_index", None)
            child_system.pop("windows.loader.startup_failure_kind", None)
            for tls_index, tls_base, tls_size in child_tls:
                child_system[f"windows.loader.tls.{tls_index}.base"] = tls_base
                child_system[f"windows.loader.tls.{tls_index}.size"] = tls_size
            startup_count = int(child_system.get("windows.loader.startup_call_count", 0))
            registers = [0] * 16
            registers[4] = rsp
            registers[1] = spawn.parameter
            pc = spawn.start_address
            call_stack = ()
            if startup_count:
                pc = int(child_system["windows.loader.startup_call.0.address"])
                registers[1] = int(child_system[
                    "windows.loader.startup_call.0.module_base"
                ])
                registers[2] = 2
                registers[8] = 0
                registers[4] -= 8
                memory = memory.write_unsigned(
                    registers[4], 64, MACHINE_LOADER_CALLBACK_RETURN,
                )
                call_stack = (MACHINE_LOADER_CALLBACK_RETURN,)
            activations.append((
                child_index,
                replace(
                    parent_state,
                    pc=pc,
                    registers=tuple(registers),
                    vector_registers=(0,) * 16,
                    flags=0,
                    memory=memory,
                    system_state=MappingProxyType(child_system),
                    fs_base=0,
                    gs_base=teb_base,
                    call_stack=call_stack,
                    external_requests=(),
                    steps=0,
                    termination_requested=False,
                    halted=False,
                    exit_code=None,
                ),
                spawn,
            ))
        # The last sentinel writes are shared process memory.
        final_memory = memory
        parent_state = replace(parent_state, memory=final_memory)
        activations = [
            (index, replace(state, memory=final_memory), spawn)
            for index, state, spawn in activations
        ]
        return parent_state, tuple(activations)

    def service_dispatch_frontiers(self, *, core_index: int | None = None) -> int:
        """Install runtime-proven executable targets at blocked core RIPs.

        Indirect calls and jumps can reach valid PE code that was not part of
        the loader's initial static-reachability roots.  This method preserves
        that discovery as a dispatch plan and leaves invalid or undecodable
        targets blocked instead of interpreting arbitrary bytes.
        """

        indices = range(len(self.machine.cores)) if core_index is None else (core_index,)
        installed = 0
        for index in indices:
            if not 0 <= index < len(self.machine.cores):
                raise IndexError("machine core index is out of range")
            core = self.machine.cores[index]
            target = int(core.state.pc)
            if target in core.executor.instructions:
                continue
            plan = core.executor.install_dispatch_targets((target,))
            self.dispatch_plans.append(plan)
            if target not in core.executor.instructions:
                continue
            installed += 1
            self.system_tape.append(
                index, core.state, position=core.position,
                event="runtime_dispatch",
                metadata={
                    "targets": list(plan.targets),
                    "installed_addresses": list(plan.installed_addresses),
                    "failure_reasons": list(plan.failure_reasons),
                },
            )
            self.system_tape.annotate(
                "runtime_dispatch",
                f"Installed runtime-proven AMD64 dispatch target at RIP {target:#x}",
                color="cyan", severity="note", core=index,
                position=core.position, address=target,
                metadata={
                    "architecture": "windows-amd64",
                    "installed_instruction_count": len(plan.installed_addresses),
                    "failure_count": len(plan.failure_reasons),
                },
            )
        return installed

    def register_external_reference(
        self,
        library: str,
        symbol: str,
    ) -> MachineExternalReference:
        """Intern a dynamically resolved export in the guest target namespace."""

        for reference in self.external_references:
            if (
                reference.library.casefold() == library.casefold()
                and reference.symbol.casefold() == symbol.casefold()
            ):
                return reference
        reference_id = max(
            (reference.reference_id for reference in self.external_references),
            default=0,
        ) + 1
        reference = MachineExternalReference(
            reference_id=reference_id,
            target_address=EXTERNAL_TARGET_BASE + (reference_id - 1) * 16,
            domain="guest-binary",
            library=library,
            symbol=symbol,
        )
        self.external_references = (*self.external_references, reference)
        self.external_reference_targets[reference.target_address] = reference
        self.system_tape.catalog_external_reference(reference)
        return reference


__all__ = ["BinaryMachineProgram", "SubjectDeviceBuffers"]
