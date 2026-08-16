#ifndef TURING_CTENSOR_OPS_H
#define TURING_CTENSOR_OPS_H

/*
 * Canonical primitive vocabulary for the C lowering target.
 * CFFI reads these values from the compiled library; Python does not maintain
 * a second numeric opcode table.
 */
typedef enum CTensorOp {
    CT_OP_ADD = 0,
    CT_OP_SUB,
    CT_OP_MUL,
    CT_OP_DIV,
    CT_OP_POW,
    CT_OP_MOD,
    CT_OP_FLOORDIV,
    CT_OP_SQRT,
    CT_OP_EXP,
    CT_OP_LOG,
    CT_OP_NEG,
    CT_OP_ABS,
    CT_OP_ROUND,
    CT_OP_TRUNC,
    CT_OP_FLOOR,
    CT_OP_CEIL,
    CT_OP_ISFINITE,
    CT_OP_ISNAN,
    CT_OP_ISINF,
    CT_OP_LOGICAL_NOT,
    CT_OP_LT,
    CT_OP_LE,
    CT_OP_GT,
    CT_OP_GE,
    CT_OP_EQ,
    CT_OP_NE,
    CT_OP_MAXIMUM,
    CT_OP_MINIMUM,
    CT_OP_TANH,
    CT_OP_SIN,
    CT_OP_COS,
    CT_OP_TAN,
    CT_OP_ASIN,
    CT_OP_ACOS,
    CT_OP_ATAN,
    CT_OP_SINH,
    CT_OP_COSH,
    CT_OP_ASINH,
    CT_OP_ACOSH,
    CT_OP_ATANH,
    CT_OP_SIGN,
    CT_OP_INVERT,
    CT_OP_BITAND,
    CT_OP_BITOR,
    CT_OP_BITXOR,
    CT_OP_SHL,
    CT_OP_SHR,
    CT_OP_LOGICAL_AND,
    CT_OP_LOGICAL_OR,
    /* Appended rather than grouped with the other transcendentals on purpose:
       is_unary_op classifies by enum *range*, so inserting mid-list would
       renumber every op after it and silently reclassify anything holding a
       raw opcode. New entries go here and are named explicitly below. */
    CT_OP_SIGMOID,
    CT_OP_COUNT
} CTensorOp;

/*
 * A compact, elementwise execution packet.  This is deliberately named a
 * primitive program rather than a tape: it is not an autograd tape, a path
 * tape, or a general SSA graph.  Feed values occupy slots [0, feed_count);
 * each instruction writes one complete, equally-shaped slot.
 */
typedef enum CTensorOperandKind {
    CT_OPERAND_NONE = 0,
    CT_OPERAND_SLOT,
    CT_OPERAND_SCALAR
} CTensorOperandKind;

typedef struct CTensorPrimitiveInstruction {
    CTensorOp op;
    int out_slot;
    int left_slot;
    CTensorOperandKind right_kind;
    int right_slot;
    double right_scalar;
    int reverse;
} CTensorPrimitiveInstruction;

void batched_matmul_indexed_double(
    const double* a,
    const double* b,
    double* out,
    const int* a_offsets,
    const int* b_offsets,
    int batch_count,
    int m,
    int n,
    int p);

void slice_copy_double(
    const double* input,
    double* output,
    const int* shape,
    int ndim,
    int dim,
    int start,
    int step,
    int count);

void index_select_double(
    const double* input,
    double* output,
    const int* shape,
    int ndim,
    int dim,
    const int* indices,
    int index_count);

void index_assign_double(
    double* target,
    const int* shape,
    int ndim,
    const int* axis_offsets,
    const int* axis_indices,
    const double* values,
    int value_count);

void index_set_double(
    const double* input,
    double* output,
    const int* shape,
    int ndim,
    const int* axis_offsets,
    const int* axis_indices,
    const double* values,
    int value_count);

void unfold2d_double(
    const double* input,
    double* output,
    int n,
    int c,
    int h,
    int w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w);

void fold2d_double(
    const double* columns,
    double* output,
    int n,
    int c,
    int h,
    int w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w);

void sign_double(const double* input, double* output, int n);

int ctensor_execute_primitive_program(
    const CTensorPrimitiveInstruction* instructions,
    int instruction_count,
    const double* const* feeds,
    int feed_count,
    double* workspace,
    int slot_count,
    int element_count,
    int output_slot,
    double* output);

int ctensor_execute_primitive_program_slots(
    const CTensorPrimitiveInstruction* instructions,
    int instruction_count,
    double* const* slots,
    int slot_count,
    int element_count);

#endif
