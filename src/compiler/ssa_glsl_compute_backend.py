"""Repository-SSA precision cores as desktop-GLSL compute shaders.

This lane takes the module shape the microprecision benchmark builds -- a
five-block counted loop (the wrapper) calling a straight-line planned region
once per element -- and emits ONE compute shader in which each invocation IS
one loop iteration.  The wrapper's Phi/Lt/CondBr/Add loop machinery is not
translated at all: ``gl_GlobalInvocationID.x`` replaces the induction
variable, and the wrapper's count formal becomes the bounds guard.  Nothing
else of the wrapper survives, which is correct because nothing else of the
wrapper does arithmetic.

What makes this lane's row in the benchmark honest rather than decorative is
the precision contract:

* Limb data stays float64 end to end.  Every array formal becomes a std430
  SSBO of ``double``; there is no float32 narrowing anywhere, unlike the
  elementwise GLSL tensor backend, whose float32 staging is a documented
  contract of a different lane.
* Every temporary produced by an instruction carrying
  ``attributes["precision_section"] == True`` is declared ``precise``.  GLSL's
  ``precise`` qualifier is the shading-language spelling of SECTION_ISOLATION:
  it forbids the compiler from reassociating, distributing, or contracting
  the expressions that feed the variable, so an error-free-transformation
  residual cannot be optimised into exactly zero behind the author's back.
* ``Fma`` becomes the builtin ``fma()`` assigned into a ``precise`` variable.
  The GLSL 4.30 specification says fma() consumed by a precise variable is
  treated as a single operation, which is precisely (sic) the single-rounding
  obligation the precision pipeline records on those instructions.

The scalar feeds (the element count and the float64 coefficients) travel in
one dedicated std430 SSBO rather than uniforms, because PyOpenGL does not
reliably expose ``glUniform1d`` and a double cannot be smuggled through a
float uniform without a rounding the whole lane exists to avoid.  The layout
is documented on :class:`GLSLComputeArtifact`.

Refusals follow the house pattern of ``ssa_c_backend``: an operation this
emitter cannot spell is a recorded shortfall on the artifact, never a silent
substitution, and ``precision_backend_shortfalls`` is consulted up front so
the capability table's claim that GLSL can honour the obligations stays a
checked fact rather than an assumption.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..transmogrifier.ssa import Function, IRModule


@dataclass(frozen=True, slots=True)
class GLSLComputeShortfall:
    """One operation this emitter refused, and why. Mirrors CEmissionShortfall."""

    operation: str
    reason: str


#: Integer-ish SSA dtypes. Index arithmetic uses GLSL ``int``: the indices
#: here multiply a loop counter against a limb stride, which for any real
#: dispatch fits 32 bits with room to spare, and 64-bit integers would drag
#: in GL_ARB_gpu_shader_int64, which is far less widely supported than fp64.
_INT_DTYPES = {"int", "int32", "i32", "int64", "i64", "long", "bool", "i1"}


def _is_int(dtype: Any) -> bool:
    return str(dtype or "").casefold() in _INT_DTYPES


def _double_literal(value: float) -> str:
    """A GLSL double literal that parses back to exactly ``value``.

    GLSL has no C99 hex-float literals, but Python's ``repr`` of a float is
    the shortest decimal that round-trips through binary64, and a conforming
    GLSL compiler converts a double literal to the nearest binary64 -- so the
    decimal spelling is exact. The ``lf`` suffix is what makes the literal a
    double instead of a float; without it the constant would round to
    float32 before the arithmetic ever ran.
    """

    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite constant {value!r} has no GLSL literal")
    return repr(float(value)) + "lf"


@dataclass(slots=True)
class GLSLComputeExecution:
    """Allocated GPU buffers plus host mirrors, ready to dispatch.

    ``buffers`` maps SSA value id -> the float64 numpy array mirroring that
    SSBO; after ``run()`` the arrays for written (output) buffers hold the
    GPU results, which is the whole read-side contract the benchmark uses.
    """

    artifact: "GLSLComputeArtifact"
    buffers: dict[int, Any]
    _buffer_names: dict[int, int] = field(default_factory=dict, repr=False)
    _scalar_name: int = 0
    _count: int = 0

    def run(self) -> "GLSLComputeExecution":
        """One dispatch, a full GPU sync, then output readback.

        The ``glFinish`` before the readback is deliberate timing honesty: a
        dispatch call alone returns as soon as the command is queued, so a
        run() that stopped there would time the driver's queueing, not the
        kernel. Finishing first makes run()'s wall time GPU-complete time,
        and the readback that follows is then a plain synchronous copy.
        """

        from OpenGL import GL

        artifact = self.artifact
        GL.glUseProgram(artifact._ensure_program())
        for binding, value_id in enumerate(artifact.buffer_order):
            GL.glBindBufferBase(
                GL.GL_SHADER_STORAGE_BUFFER, binding,
                self._buffer_names[int(value_id)],
            )
        GL.glBindBufferBase(
            GL.GL_SHADER_STORAGE_BUFFER, len(artifact.buffer_order),
            self._scalar_name,
        )
        groups = max(1, (self._count + artifact.local_size - 1)
                     // artifact.local_size)
        GL.glDispatchCompute(groups, 1, 1)
        GL.glMemoryBarrier(
            GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_BUFFER_UPDATE_BARRIER_BIT
        )
        GL.glFinish()
        for value_id in artifact.written_buffers:
            held = self.buffers[int(value_id)]
            GL.glBindBuffer(
                GL.GL_SHADER_STORAGE_BUFFER, self._buffer_names[int(value_id)]
            )
            GL.glGetBufferSubData(
                GL.GL_SHADER_STORAGE_BUFFER, 0, held.nbytes,
                held.ctypes.data_as(ctypes.c_void_p),
            )
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        return self

    def close(self) -> None:
        """Release the GL buffers. Safe to call more than once."""

        if not self._buffer_names and not self._scalar_name:
            return
        try:
            from OpenGL import GL

            names = list(self._buffer_names.values())
            if self._scalar_name:
                names.append(self._scalar_name)
            for name in names:
                GL.glDeleteBuffers(1, [int(name)])
        except Exception:
            pass  # context teardown races during shutdown are not errors
        self._buffer_names = {}
        self._scalar_name = 0


@dataclass(slots=True)
class GLSLComputeArtifact:
    """One precision-core module emitted as one desktop compute shader.

    Buffer ABI
    ----------
    ``buffer_order`` lists the SSA value ids of the ARRAY formals, and the
    position of an id in the tuple IS its SSBO binding point.  Each such
    SSBO is a bare ``double data[]`` in std430, fed with the float64 numpy
    bytes of the corresponding feed.

    Scalars SSBO layout (binding ``len(buffer_order)``), std430::

        uint   element_count;   // byte offset 0 -- the wrapper's count formal
        uint   _pad0;           // byte offset 4 -- pads the doubles to 8
        double scalars[];       // byte offset 8 -- one double per remaining
                                //   entry of scalar_order, in order

    ``scalar_order`` starts with the count's value id and then names, in
    layout order, the value id behind each element of ``scalars[]``.  The
    count travels as a uint because it guards the dispatch; every other
    scalar is a float64 coefficient and stays a double.
    """

    name: str
    source: str
    buffer_order: tuple[int, ...]
    scalar_order: tuple[int, ...]
    shortfalls: tuple[GLSLComputeShortfall, ...]
    #: True when any emitted instruction belonged to a precision section --
    #: i.e. when the shader actually contains ``precise`` declarations and
    #: the claim in the module docstring is being exercised, not just made.
    precision_sections: bool = False
    #: Array formals whose SSBOs the shader stores into; these are what
    #: ``run()`` reads back into the host mirrors after the dispatch.
    written_buffers: tuple[int, ...] = ()
    local_size: int = 64
    _program: Any = field(default=None, repr=False)

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    def _ensure_program(self) -> int:
        """Compile and link the shader once; reuse the program afterwards.

        The caller is responsible for having a current 4.3+ context (that is
        ``prepare_execution``'s job); this only turns source into a program
        and raises with the driver's own info log on failure, because a
        truncated or paraphrased log is useless against a 1200-line shader.
        """

        if self._program is not None:
            return int(self._program)
        from OpenGL import GL

        shader = GL.glCreateShader(GL.GL_COMPUTE_SHADER)
        GL.glShaderSource(shader, self.source)
        GL.glCompileShader(shader)
        if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
            log = GL.glGetShaderInfoLog(shader)
            GL.glDeleteShader(shader)
            if isinstance(log, bytes):
                log = log.decode("utf-8", "replace")
            raise RuntimeError(
                f"GLSL compute shader for {self.name!r} failed to compile:\n"
                + str(log)
            )
        program = GL.glCreateProgram()
        GL.glAttachShader(program, shader)
        GL.glLinkProgram(program)
        GL.glDeleteShader(shader)
        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            log = GL.glGetProgramInfoLog(program)
            GL.glDeleteProgram(program)
            if isinstance(log, bytes):
                log = log.decode("utf-8", "replace")
            raise RuntimeError(
                f"GLSL compute program for {self.name!r} failed to link:\n"
                + str(log)
            )
        self._program = int(program)
        return self._program

    def prepare_execution(
        self, feeds: Mapping[int, Any]
    ) -> GLSLComputeExecution:
        """Allocate GPU buffers from real feed values and bind the program.

        Mirrors the C module lane's ``prepare_execution``: buffer sizes come
        from the feeds, not from declared shapes, because a region formal's
        declared shape is ``()`` whether it is a scalar or the base of a
        million-element array.  Acquiring the context happens here, not at
        import, so merely emitting a shader never opens a window.
        """

        import numpy as np
        from src.common.tensors.accelerator_backends.gl_context import (
            require_gl_context,
        )

        if not self.complete:
            raise ValueError(
                "GLSL compute artifact has emission shortfalls: "
                + "; ".join(
                    f"{item.operation}: {item.reason}"
                    for item in self.shortfalls[:8]
                )
            )
        require_gl_context(min_version=(4, 3))
        from OpenGL import GL

        self._ensure_program()

        buffers: dict[int, Any] = {}
        buffer_names: dict[int, int] = {}
        for value_id in self.buffer_order:
            fed = feeds.get(int(value_id))
            held = np.ascontiguousarray(
                np.atleast_1d(np.asarray(0.0 if fed is None else fed)),
                dtype=np.float64,
            )
            buffers[int(value_id)] = held
            name = int(GL.glGenBuffers(1))
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, name)
            GL.glBufferData(
                GL.GL_SHADER_STORAGE_BUFFER, held.nbytes, held,
                GL.GL_DYNAMIC_COPY,
            )
            buffer_names[int(value_id)] = name

        # The scalars block: pack exactly the layout the artifact documents.
        # The count is uint32 at offset 0, four bytes of padding align the
        # doubles to their std430 base alignment of 8, then one double per
        # remaining scalar_order entry in order. Missing coefficient feeds
        # become 0.0, which is the exact-zero appended-limb convention the
        # feed builders already use.
        count_id = int(self.scalar_order[0])
        count = int(np.asarray(feeds[count_id]).reshape(-1)[0])
        header = np.zeros(2, dtype=np.uint32)
        header[0] = count
        coefficients = np.asarray(
            [
                float(np.asarray(feeds.get(int(value_id), 0.0)).reshape(-1)[0])
                for value_id in self.scalar_order[1:]
            ],
            dtype=np.float64,
        )
        block = header.tobytes() + coefficients.tobytes()
        scalar_name = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, scalar_name)
        GL.glBufferData(
            GL.GL_SHADER_STORAGE_BUFFER, len(block), block, GL.GL_STATIC_DRAW,
        )
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        return GLSLComputeExecution(
            artifact=self,
            buffers=buffers,
            _buffer_names=buffer_names,
            _scalar_name=scalar_name,
            _count=count,
        )


def _find_region_call(wrapper: Function, module: IRModule):
    """The wrapper's one Call into a function this module defines, or None."""

    for block in wrapper.blocks.values():
        for instruction in block.instrs:
            if str(instruction.op) in ("Call", "call"):
                callee = str(instruction.attributes.get("callee") or "")
                if callee in module.functions:
                    return instruction, callee
    return None


def emit_ssa_module_to_glsl_compute(
    module: IRModule, function_name: str, *, entry_name: str | None = None,
) -> GLSLComputeArtifact:
    """Emit a wrapper-plus-region precision core as one compute shader.

    ``function_name`` names the WRAPPER (the counted loop).  The loop itself
    is discarded: the Call in its body tells us, positionally, which wrapper
    value feeds each region formal, and the one actual that is the loop Phi's
    result marks the region's element-index formal.  That formal becomes
    ``int(gl_GlobalInvocationID.x)`` and everything else becomes buffer or
    scalar traffic, so the region body inlines into ``main()`` unchanged.
    """

    from .ir_identities import precision_backend_shortfalls

    name = str(entry_name or function_name)
    shortfalls: list[GLSLComputeShortfall] = []

    def refuse(operation: str, reason: str) -> GLSLComputeArtifact:
        shortfalls.append(GLSLComputeShortfall(operation, reason))
        return GLSLComputeArtifact(
            name, "", (), (), tuple(shortfalls), precision_sections=False,
        )

    wrapper = module.functions.get(function_name)
    if wrapper is None:
        return refuse("module", f"no function named {function_name!r}")

    # -- wrapper anatomy: the call, the induction Phi, the count ------------
    found = _find_region_call(wrapper, module)
    if found is None:
        return refuse(
            "Call", "wrapper has no call into a function of this module, so "
            "there is no planned region to turn into a kernel",
        )
    call, region_name = found
    region: Function = module.functions[region_name]

    # The capability table declares GLSL able to honour both precision
    # obligations; this consult is what keeps that a checked fact. Any miss
    # recorded by the pipeline for either reachable function becomes a loud
    # shortfall before a single line of shader text exists.
    shortfalls.extend(
        GLSLComputeShortfall(
            "precision_section",
            "backend cannot honour precision obligations "
            + repr(item["missing"]) + f" in {item['function']}",
        )
        for item in precision_backend_shortfalls(
            module, "glsl", (function_name, region_name),
        )
    )

    if set(region.blocks) != {"entry"}:
        return refuse(
            "control",
            f"planned region {region_name!r} is not one straight-line entry "
            "block; per-invocation inlining requires exactly that shape",
        )
    if len(call.args) != len(region.args):
        return refuse(
            "Call",
            f"wrapper passes {len(call.args)} actuals but region "
            f"{region_name!r} declares {len(region.args)} formals",
        )

    phi_results = {
        int(instruction.res.id)
        for block in wrapper.blocks.values()
        for instruction in block.instrs
        if str(instruction.op) == "Phi" and instruction.res is not None
    }
    index_positions = [
        position for position, actual in enumerate(call.args)
        if int(actual.id) in phi_results
    ]
    if len(index_positions) != 1:
        return refuse(
            "Phi",
            f"expected exactly one loop-Phi actual in the region call, found "
            f"{len(index_positions)}; cannot identify the element index",
        )
    index_formal_id = int(region.args[index_positions[0]].id)
    phi_id = int(call.args[index_positions[0]].id)

    # The count is whatever the loop condition compares the Phi against.
    count_id = None
    for block in wrapper.blocks.values():
        for instruction in block.instrs:
            if str(instruction.op) == "Lt" and len(instruction.args) == 2:
                ids = [int(value.id) for value in instruction.args]
                if phi_id in ids:
                    others = [held for held in ids if held != phi_id]
                    if others:
                        count_id = others[0]
    if count_id is None:
        return refuse(
            "Lt", "wrapper has no Lt comparing the loop Phi against a bound; "
            "cannot identify the element count for the dispatch guard",
        )

    # -- region formal classification: buffers vs scalars -------------------
    #
    # An array base is told apart by what is done THROUGH it: it appears as
    # GetElementPtr's first argument, or as the pointer a Store writes when
    # no GetElementPtr intervenes. Everything else is a by-value scalar.
    # This is the same derived-not-assumed rule _roles() uses, because
    # guessing formal layout is what made earlier readings of these kernels
    # wrong.
    formal_ids = [int(value.id) for value in region.args]
    gep_bases: set[int] = set()
    written_bases: set[int] = set()
    gep_base_of: dict[int, int] = {}
    for block in region.blocks.values():
        for instruction in block.instrs:
            op = str(instruction.op)
            if op == "GetElementPtr" and instruction.args:
                base = int(instruction.args[0].id)
                gep_bases.add(base)
                if instruction.res is not None:
                    gep_base_of[int(instruction.res.id)] = base
            elif op == "Store" and len(instruction.args) >= 2:
                pointer = int(instruction.args[1].id)
                written_bases.add(gep_base_of.get(pointer, pointer))
                if pointer in formal_ids:
                    gep_bases.add(pointer)

    actual_of = {
        int(formal.id): int(actual.id)
        for formal, actual in zip(region.args, call.args)
    }
    array_formals = [
        held for held in formal_ids
        if held in gep_bases and held != index_formal_id
    ]
    scalar_formals = [
        held for held in formal_ids
        if held not in gep_bases and held != index_formal_id
    ]
    for held in array_formals:
        formal = next(v for v in region.args if int(v.id) == held)
        if str(formal.dtype or "float64").casefold() not in ("float64", "double"):
            shortfalls.append(GLSLComputeShortfall(
                "buffer",
                f"array formal %t{held} has dtype {formal.dtype!r}; this "
                "lane's SSBOs are float64 only, by the precision contract",
            ))

    buffer_order = tuple(actual_of[held] for held in array_formals)
    scalar_order = (int(count_id),) + tuple(
        actual_of[held] for held in scalar_formals
    )
    written_buffers = tuple(
        actual_of[held] for held in array_formals if held in written_bases
    )
    buffer_slot = {held: slot for slot, held in enumerate(array_formals)}
    scalar_slot = {held: slot for slot, held in enumerate(scalar_formals)}

    # -- region body emission -----------------------------------------------
    #
    # Everything lands in a named local ``t<id>`` (or a literal, for Const)
    # so that the emitted text reads in the same order as the SSA and every
    # precision-section value is visibly its own ``precise`` declaration.
    # ``precise`` deliberately does NOT combine with ``const`` here: some
    # compilers reject ``precise const``, and const adds nothing the shader
    # needs, so precision-section locals are plain ``precise <type>``.
    expressions: dict[int, str] = {index_formal_id: "element"}
    for held in scalar_formals:
        expressions[held] = f"feed.scalars[{scalar_slot[held]}]"
    # GEP results are not values, they are (buffer, flat index) bindings;
    # this map is how Load/Store find their way back to the right SSBO.
    gep_pointer: dict[int, tuple[int, str]] = {}
    body: list[str] = []
    precision_present = False

    def operand(value) -> str | None:
        held = expressions.get(int(value.id))
        if held is None:
            shortfalls.append(GLSLComputeShortfall(
                "operand",
                f"%t{value.id} is unavailable in {region_name}",
            ))
        return held

    for instruction in region.blocks["entry"].instrs:
        op = str(instruction.op)
        section = bool(instruction.attributes.get("precision_section"))
        if section:
            precision_present = True
        qualifier = "precise " if section else "const "
        if op == "Const":
            held = instruction.attributes.get(
                "constant", instruction.attributes.get("value")
            )
            if instruction.res is None:
                shortfalls.append(GLSLComputeShortfall(op, "Const without result"))
                continue
            if _is_int(instruction.res.dtype) or isinstance(held, int):
                expressions[int(instruction.res.id)] = str(int(held))
            else:
                try:
                    expressions[int(instruction.res.id)] = _double_literal(
                        float(held)
                    )
                except ValueError as error:
                    shortfalls.append(GLSLComputeShortfall(op, str(error)))
            continue
        if op == "Ret":
            # Region outputs live in the SSBOs already; nothing to return.
            continue
        if op == "Store":
            if len(instruction.args) < 2:
                shortfalls.append(GLSLComputeShortfall(op, "Store without operands"))
                continue
            value = operand(instruction.args[0])
            pointer = gep_pointer.get(int(instruction.args[1].id))
            if pointer is None:
                shortfalls.append(GLSLComputeShortfall(
                    op,
                    f"store through %t{instruction.args[1].id}, which is not "
                    "a GetElementPtr result; only addressed stores are spelt",
                ))
                continue
            if value is None:
                continue
            slot, index_expr = pointer
            if _is_int(instruction.args[0].dtype):
                value = f"double({value})"
            body.append(f"    b{slot}.data[{index_expr}] = {value};")
            continue
        if instruction.res is None:
            shortfalls.append(GLSLComputeShortfall(op, "instruction has no result"))
            continue
        result_id = int(instruction.res.id)
        if op == "GetElementPtr":
            if len(instruction.args) < 2:
                shortfalls.append(GLSLComputeShortfall(op, "GEP without base+index"))
                continue
            base = int(instruction.args[0].id)
            slot = buffer_slot.get(base)
            if slot is None:
                shortfalls.append(GLSLComputeShortfall(
                    op,
                    f"GetElementPtr base %t{base} is not an array formal of "
                    f"{region_name}; computed bases have no SSBO to bind",
                ))
                continue
            index_value = operand(instruction.args[1])
            if index_value is None:
                continue
            # An address is an (SSBO, int index) pair here. The index local
            # is int arithmetic and exact, but it still respects the section
            # qualifier so the emitted text tells the truth about which
            # instructions the pipeline marked.
            body.append(f"    {qualifier}int a{result_id} = {index_value};")
            gep_pointer[result_id] = (slot, f"a{result_id}")
            continue
        if op == "Load":
            # A Load's operand is a GetElementPtr result, which is not a
            # value in ``expressions`` -- it is an (SSBO, index) binding --
            # so it must never pass through the generic operand renderer.
            pointer = gep_pointer.get(int(instruction.args[0].id)) if (
                instruction.args
            ) else None
            if pointer is None:
                shortfalls.append(GLSLComputeShortfall(
                    op,
                    f"load through %t{instruction.args[0].id if instruction.args else '?'},"
                    " which is not a GetElementPtr result",
                ))
                continue
            slot, index_expr = pointer
            expressions[result_id] = f"t{result_id}"
            body.append(
                f"    {qualifier}double t{result_id} = "
                f"b{slot}.data[{index_expr}];"
            )
            continue
        args = [operand(value) for value in instruction.args]
        if any(value is None for value in args):
            continue
        kind = "int" if _is_int(instruction.res.dtype) else "double"
        rendered = None
        if op == "Fma" and len(args) == 3:
            # The builtin, never the spelled-out multiply-add: consumed by a
            # precise variable, GLSL guarantees fma() is a single operation,
            # which is the whole FMA_MANDATORY obligation.
            kind = "double"
            rendered = f"fma({args[0]}, {args[1]}, {args[2]})"
        elif op in _GLSL_BINARY and len(args) == 2:
            rendered = f"({args[0]} {_GLSL_BINARY[op]} {args[1]})"
        elif op == "Neg" and len(args) == 1:
            rendered = f"(-{args[0]})"
        if rendered is None:
            shortfalls.append(GLSLComputeShortfall(
                op, f"no GLSL compute spelling in {region_name}",
            ))
            continue
        expressions[result_id] = f"t{result_id}"
        body.append(f"    {qualifier}{kind} t{result_id} = {rendered};")

    # -- shader assembly -----------------------------------------------------
    interface: list[str] = []
    for slot, formal_id in enumerate(array_formals):
        writable = formal_id in written_bases
        access = "" if writable else "readonly "
        interface.append(
            f"layout(std430, binding = {slot}) {access}buffer "
            f"Buffer{slot} {{ double data[]; }} b{slot};  "
            f"// SSA value %t{actual_of[formal_id]}"
        )
    interface.append(
        f"layout(std430, binding = {len(array_formals)}) readonly buffer "
        "ScalarFeed {"
    )
    interface.append("    uint element_count;  // the wrapper's loop bound")
    interface.append("    uint _pad0;          // aligns the doubles to 8")
    interface.append("    double scalars[];    // coefficients, layout order")
    interface.append("} feed;")

    source = "\n".join((
        "#version 430",
        "#extension GL_ARB_gpu_shader5 : require",
        "#extension GL_ARB_gpu_shader_fp64 : require",
        f"// {name}: one invocation computes one element of the precision",
        "// core; the CPU wrapper's counted loop is replaced by the dispatch",
        "// grid itself, which is why no loop appears below.",
        "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;",
        *interface,
        "void main() {",
        "    uint invocation = gl_GlobalInvocationID.x;",
        "    if (invocation >= feed.element_count) { return; }",
        "    // The region's element-index formal, standing in for the loop",
        "    // Phi. Signed int on purpose: the SSA index arithmetic is",
        "    // signed and the strides are tiny, so int is both faithful",
        "    // and universally supported.",
        "    const int element = int(invocation);",
        *body,
        "}",
        "",
    ))

    return GLSLComputeArtifact(
        name=name,
        source=source,
        buffer_order=buffer_order,
        scalar_order=scalar_order,
        shortfalls=tuple(shortfalls),
        precision_sections=precision_present,
        written_buffers=written_buffers,
    )


_GLSL_BINARY = {"Add": "+", "Sub": "-", "Mul": "*", "Div": "/"}


__all__ = [
    "GLSLComputeShortfall",
    "GLSLComputeArtifact",
    "GLSLComputeExecution",
    "emit_ssa_module_to_glsl_compute",
]
