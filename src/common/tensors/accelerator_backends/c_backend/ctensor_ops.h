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
    CT_OP_COUNT
} CTensorOp;

#endif
