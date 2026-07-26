"""Thin binding to the workspace's persistent Tensor Calculator runtime."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

from cffi import FFI

from .c_backend import ffi as backend_ffi


calculator_ffi = FFI()
calculator_ffi.cdef("""
    typedef struct tc_calculator tc_calculator;
    typedef struct tc_prepared_program tc_prepared_program;
    typedef unsigned long long tc_tensor_handle;
    typedef unsigned long long tc_job_handle;
    typedef enum tc_status {
        TC_OK, TC_INVALID_ARGUMENT, TC_NOT_FOUND, TC_DTYPE_MISMATCH,
        TC_SHAPE_MISMATCH, TC_OUT_OF_MEMORY, TC_QUEUE_FULL, TC_PENDING,
        TC_INTERNAL_ERROR
    } tc_status;
    typedef enum tc_dtype { TC_F32 = 1, TC_F64 = 2 } tc_dtype;
    typedef enum tc_op {
        TC_ADD, TC_SUB, TC_MUL, TC_DIV, TC_POW, TC_MOD, TC_FLOORDIV,
        TC_SQRT, TC_EXP, TC_LOG, TC_NEG, TC_ABS, TC_ROUND, TC_TRUNC,
        TC_FLOOR, TC_CEIL, TC_ISFINITE, TC_ISNAN, TC_ISINF, TC_LOGICAL_NOT,
        TC_LT, TC_LE, TC_GT, TC_GE, TC_EQ, TC_NE, TC_MAXIMUM, TC_MINIMUM,
        TC_OP_COUNT
    } tc_op;
    typedef enum tc_operand_kind {
        TC_OPERAND_NONE, TC_OPERAND_TENSOR, TC_OPERAND_SCALAR
    } tc_operand_kind;
    typedef struct tc_config {
        unsigned int worker_count;
        unsigned int queue_capacity;
        unsigned long long async_element_threshold;
    } tc_config;
    typedef struct tc_tensor_desc {
        tc_dtype dtype;
        unsigned long long element_count;
    } tc_tensor_desc;
    typedef struct tc_instruction {
        tc_op op;
        tc_tensor_handle output;
        tc_tensor_handle left;
        tc_operand_kind right_kind;
        tc_tensor_handle right;
        double right_scalar;
        unsigned char reverse;
    } tc_instruction;
    typedef struct tc_program {
        const tc_instruction* instructions;
        unsigned int instruction_count;
    } tc_program;
    typedef struct tc_stats {
        unsigned long long tensors_live, bytes_live, programs_executed;
        unsigned long long instructions_executed, jobs_submitted;
        unsigned long long jobs_completed, inline_jobs;
    } tc_stats;
    unsigned int tc_abi_version(void);
    const char* tc_status_string(tc_status status);
    tc_calculator* tc_create(const tc_config* config);
    void tc_destroy(tc_calculator*);
    tc_status tc_tensor_bind_external(
        tc_calculator*, tc_tensor_desc, void*, size_t, tc_tensor_handle*);
    tc_status tc_tensor_release(tc_calculator*, tc_tensor_handle);
    tc_status tc_execute(tc_calculator*, const tc_program*);
    tc_status tc_execute_raw(
        tc_op, tc_dtype, void*, const void*, tc_operand_kind, const void*,
        double, unsigned char, unsigned long long);
    tc_status tc_prepare(
        tc_calculator*, const tc_program*, tc_prepared_program**);
    void tc_prepared_release(tc_prepared_program*);
    tc_status tc_prepared_execute(tc_prepared_program*);
    tc_status tc_prepared_submit(tc_prepared_program*, tc_job_handle*);
    tc_status tc_submit(tc_calculator*, const tc_program*, tc_job_handle*);
    tc_status tc_job_poll(tc_calculator*, tc_job_handle);
    tc_status tc_job_wait(tc_calculator*, tc_job_handle);
    tc_status tc_job_release(tc_calculator*, tc_job_handle);
    tc_status tc_get_stats(tc_calculator*, tc_stats*);
""")


_OP_SYMBOLS = {
    "add": "TC_ADD", "sub": "TC_SUB", "mul": "TC_MUL",
    "truediv": "TC_DIV", "pow": "TC_POW", "mod": "TC_MOD",
    "floordiv": "TC_FLOORDIV", "sqrt": "TC_SQRT", "exp": "TC_EXP",
    "log": "TC_LOG", "neg": "TC_NEG", "abs": "TC_ABS",
    "round": "TC_ROUND", "trunc": "TC_TRUNC", "floor": "TC_FLOOR",
    "ceil": "TC_CEIL", "isfinite": "TC_ISFINITE", "isnan": "TC_ISNAN",
    "isinf": "TC_ISINF", "logical_not": "TC_LOGICAL_NOT",
    "less": "TC_LT", "less_equal": "TC_LE", "greater": "TC_GT",
    "greater_equal": "TC_GE", "equal": "TC_EQ", "not_equal": "TC_NE",
    "lt": "TC_LT", "le": "TC_LE", "gt": "TC_GT", "ge": "TC_GE",
    "eq": "TC_EQ", "ne": "TC_NE", "maximum": "TC_MAXIMUM",
    "minimum": "TC_MINIMUM",
}


def _library_candidates():
    configured = os.environ.get("TENSOR_CALCULATOR_LIB")
    if configured:
        yield Path(configured)
    workspace = Path(__file__).resolve().parents[5]
    root = workspace / "tensor-calculator" / "build"
    yield root / "Release" / "tensor_calculator.dll"
    yield root / "Debug" / "tensor_calculator.dll"
    yield root / "libtensor_calculator.so"
    yield root / "libtensor_calculator.dylib"


class NativeCalculator:
    """Persistent calculator state shared by all C-backend tensors."""

    def __init__(self, library_path: Path):
        self.library_path = library_path
        self.lib = calculator_ffi.dlopen(str(library_path))
        if self.lib.tc_abi_version() != 1:
            raise RuntimeError("unsupported Tensor Calculator ABI")
        config = calculator_ffi.new("tc_config*")
        config.worker_count = int(os.environ.get("TENSOR_CALCULATOR_WORKERS", "0"))
        config.queue_capacity = int(
            os.environ.get("TENSOR_CALCULATOR_QUEUE_CAPACITY", "1024")
        )
        config.async_element_threshold = int(
            os.environ.get("TENSOR_CALCULATOR_ASYNC_THRESHOLD", "4096")
        )
        self.state = self.lib.tc_create(config)
        if self.state == calculator_ffi.NULL:
            raise MemoryError("could not create Tensor Calculator")

    def _check(self, status):
        if status != self.lib.TC_OK:
            message = calculator_ffi.string(
                self.lib.tc_status_string(status)
            ).decode()
            raise RuntimeError(f"Tensor Calculator: {message}")

    def bind(self, tensor):
        owner = getattr(tensor, "_calculator_owner", None)
        if owner is self:
            return tensor._calculator_handle
        address = int(backend_ffi.cast("uintptr_t", tensor.as_c_ptr()))
        desc = calculator_ffi.new("tc_tensor_desc*")
        desc.dtype = self.lib.TC_F64
        desc.element_count = tensor.size
        handle = calculator_ffi.new("tc_tensor_handle*")
        self._check(self.lib.tc_tensor_bind_external(
            self.state,
            desc[0],
            calculator_ffi.cast("void*", address),
            tensor.size * backend_ffi.sizeof("double"),
            handle,
        ))
        tensor._calculator_owner = self
        tensor._calculator_handle = int(handle[0])
        return tensor._calculator_handle

    def release(self, tensor):
        if getattr(tensor, "_calculator_owner", None) is self:
            self.lib.tc_tensor_release(self.state, tensor._calculator_handle)
            tensor._calculator_owner = None
            tensor._calculator_handle = 0

    def instruction(self, native, op, output, left, right=None, scalar=None,
                    reverse=False):
        native.op = getattr(self.lib, _OP_SYMBOLS[op])
        native.output = self.bind(output)
        native.left = self.bind(left)
        if right is not None:
            native.right_kind = self.lib.TC_OPERAND_TENSOR
            native.right = self.bind(right)
        elif scalar is not None:
            native.right_kind = self.lib.TC_OPERAND_SCALAR
            native.right_scalar = float(scalar)
        else:
            native.right_kind = self.lib.TC_OPERAND_NONE
        native.reverse = int(reverse)

    def execute_one(self, op, output, left, right=None, scalar=None,
                    reverse=False):
        def pointer(tensor):
            address = int(backend_ffi.cast("uintptr_t", tensor.as_c_ptr()))
            return calculator_ffi.cast("void*", address)

        if right is not None:
            right_kind = self.lib.TC_OPERAND_TENSOR
            right_pointer = pointer(right)
        elif scalar is not None:
            right_kind = self.lib.TC_OPERAND_SCALAR
            right_pointer = calculator_ffi.NULL
        else:
            right_kind = self.lib.TC_OPERAND_NONE
            right_pointer = calculator_ffi.NULL
        self._check(self.lib.tc_execute_raw(
            getattr(self.lib, _OP_SYMBOLS[op]),
            self.lib.TC_F64,
            pointer(output),
            pointer(left),
            right_kind,
            right_pointer,
            float(scalar or 0.0),
            int(reverse),
            left.size,
        ))
        return output

    def execute_one_persistent(self, op, output, left, right=None, scalar=None,
                               reverse=False):
        instructions = calculator_ffi.new("tc_instruction[]", 1)
        self.instruction(
            instructions[0], op, output, left, right, scalar, reverse
        )
        program = calculator_ffi.new("tc_program*")
        program.instructions = instructions
        program.instruction_count = 1
        self._check(self.lib.tc_execute(self.state, program))
        return output

    def prepare_program(self, program, slots):
        instructions = calculator_ffi.new(
            "tc_instruction[]", len(program.instructions)
        )
        for index, step in enumerate(program.instructions):
            self.instruction(
                instructions[index],
                step.op,
                slots[step.out_slot],
                slots[step.left_slot],
                slots[step.right_slot] if step.right_slot is not None else None,
                step.right_scalar,
                step.reverse,
            )
        return PreparedCalculatorProgram(self, program, slots, instructions)

    def stats(self):
        stats = calculator_ffi.new("tc_stats*")
        self._check(self.lib.tc_get_stats(self.state, stats))
        return {
            name: int(getattr(stats, name))
            for name in (
                "tensors_live", "bytes_live", "programs_executed",
                "instructions_executed", "jobs_submitted", "jobs_completed",
                "inline_jobs",
            )
        }


class PreparedCalculatorProgram:
    def __init__(self, calculator, program, slots, instructions):
        self.calculator = calculator
        self.program_definition = program
        self.slots = slots
        self.instructions = instructions
        self.native_program = calculator_ffi.new("tc_program*")
        self.native_program.instructions = instructions
        self.native_program.instruction_count = len(program.instructions)
        prepared = calculator_ffi.new("tc_prepared_program**")
        self.calculator._check(self.calculator.lib.tc_prepare(
            self.calculator.state, self.native_program, prepared
        ))
        self.prepared = prepared[0]

    def __del__(self):
        prepared = getattr(self, "prepared", calculator_ffi.NULL)
        if prepared != calculator_ffi.NULL:
            try:
                self.calculator.lib.tc_prepared_release(prepared)
            except Exception:
                pass
            self.prepared = calculator_ffi.NULL

    @property
    def output(self):
        return self.slots[self.program_definition.output_slot]

    def execute(self):
        self.calculator._check(self.calculator.lib.tc_prepared_execute(
            self.prepared
        ))
        return self.output

    def submit(self):
        handle = calculator_ffi.new("tc_job_handle*")
        self.calculator._check(self.calculator.lib.tc_prepared_submit(
            self.prepared, handle
        ))
        return CalculatorJob(self.calculator, int(handle[0]), self.output)


class CalculatorJob:
    def __init__(self, calculator, handle, output):
        self.calculator = calculator
        self.handle = handle
        self.output = output

    def poll(self):
        return int(self.calculator.lib.tc_job_poll(
            self.calculator.state, self.handle
        ))

    def wait(self):
        self.calculator._check(self.calculator.lib.tc_job_wait(
            self.calculator.state, self.handle
        ))
        return self.output

    def release(self):
        self.calculator._check(self.calculator.lib.tc_job_release(
            self.calculator.state, self.handle
        ))


_singleton = None
_singleton_lock = Lock()


def get_native_calculator(required=False):
    global _singleton
    if os.environ.get("TENSOR_CALCULATOR_DISABLE") == "1":
        if required:
            raise RuntimeError("Tensor Calculator disabled")
        return None
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                for candidate in _library_candidates():
                    if candidate.exists():
                        _singleton = NativeCalculator(candidate)
                        break
    if required and _singleton is None:
        raise FileNotFoundError(
            "Tensor Calculator library not built; run cmake in tensor-calculator"
        )
    return _singleton
