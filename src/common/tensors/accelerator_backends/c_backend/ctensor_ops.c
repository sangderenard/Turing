
    #include <math.h>
    #include <stdlib.h>
    #include <stddef.h>
    #include <string.h>
    #include "ctensor_ops.h"

    // Forward declarations for later functions so that Zig's C compiler
    // does not fail due to implicit declarations.
    void for_each_cell_along_dim(
        const double* data,
        const int* shape,
        int ndim,
        int batch_dim,
        void (*callback)(const double*, int, int, void*),
        void* user_data);

    typedef struct {
        double* out;
        int dim_size;
        int cell_len;
        int out_index;
        double* accum;
    } mean_dim_ctx;

    static void mean_dim_callback(const double* data, int cell_len, int idx, void* user_data) {
        mean_dim_ctx* ctx = (mean_dim_ctx*)user_data;
        if (idx == 0) {
            for (int i = 0; i < ctx->cell_len; ++i) ctx->accum[i] = 0.0;
        }
        for (int i = 0; i < ctx->cell_len; ++i) ctx->accum[i] += data[i];
        if (idx == ctx->dim_size - 1) {
            for (int i = 0; i < ctx->cell_len; ++i) {
                ctx->out[ctx->out_index * ctx->cell_len + i] = ctx->accum[i] / ctx->dim_size;
            }
            ctx->out_index++;
        }
    }

    typedef struct {
        double* values;
        double* indices;
        int k;
        int dim_size;
        int cell_len;
        int out_index;
        double* buffer;
    } topk_dim_ctx;

    static void topk_dim_callback(const double* data, int cell_len, int idx, void* user_data) {
        topk_dim_ctx* ctx = (topk_dim_ctx*)user_data;
        for (int i = 0; i < ctx->cell_len; ++i) {
            ctx->buffer[idx * ctx->cell_len + i] = data[i];
        }
        if (idx == ctx->dim_size - 1) {
            for (int i = 0; i < ctx->cell_len; ++i) {
                const double* slice = ctx->buffer + i;
                int* used = (int*)malloc(ctx->dim_size * sizeof(int));
                for (int u = 0; u < ctx->dim_size; ++u) used[u] = 0;
                for (int t = 0; t < ctx->k; ++t) {
                    int best = -1;
                    double best_val = -1e300;
                    for (int j = 0; j < ctx->dim_size; ++j) {
                        double val = slice[j * ctx->cell_len];
                        if (!used[j] && val > best_val) {
                            best_val = val;
                            best = j;
                        }
                    }
                    int out_pos = (ctx->out_index * ctx->cell_len + i) * ctx->k + t;
                    if (best != -1) {
                        used[best] = 1;
                        ctx->indices[out_pos] = best;
                        ctx->values[out_pos] = best_val;
                    } else {
                        ctx->indices[out_pos] = -1;
                        ctx->values[out_pos] = 0.0;
                    }
                }
                free(used);
            }
            ctx->out_index++;
        }
    }

    typedef struct {
        double* out;
        int dim_size;
        int cell_len;
        int out_index;
        double* buffer;
        double* max_vals;
        double* sum_vals;
    } log_softmax_dim_ctx;

    static void log_softmax_dim_callback(const double* data, int cell_len, int idx, void* user_data) {
        log_softmax_dim_ctx* ctx = (log_softmax_dim_ctx*)user_data;
        double* slice = ctx->buffer + idx * ctx->cell_len;
        for (int i = 0; i < ctx->cell_len; ++i) {
            slice[i] = data[i];
            if (idx == 0 || data[i] > ctx->max_vals[i]) ctx->max_vals[i] = data[i];
        }
        if (idx == ctx->dim_size - 1) {
            for (int i = 0; i < ctx->cell_len; ++i) ctx->sum_vals[i] = 0.0;
            for (int j = 0; j < ctx->dim_size; ++j) {
                double* row = ctx->buffer + j * ctx->cell_len;
                for (int i = 0; i < ctx->cell_len; ++i) {
                    row[i] = exp(row[i] - ctx->max_vals[i]);
                    ctx->sum_vals[i] += row[i];
                }
            }
            for (int j = 0; j < ctx->dim_size; ++j) {
                double* row = ctx->buffer + j * ctx->cell_len;
                for (int i = 0; i < ctx->cell_len; ++i) {
                    int pos = (ctx->out_index * ctx->dim_size + j) * ctx->cell_len + i;
                    ctx->out[pos] = log(row[i] / ctx->sum_vals[i]);
                }
            }
            ctx->out_index++;
        }
    }

    void fill_double(double* out, double value, int n) {
        for (int i = 0; i < n; ++i) out[i] = value;
    }

    static double binary_value(double a, double b, int op) {
        switch (op) {
            case CT_OP_ADD: return a + b;
            case CT_OP_SUB: return a - b;
            case CT_OP_MUL: return a * b;
            case CT_OP_DIV: return a / b;
            case CT_OP_POW: return pow(a, b);
            case CT_OP_MOD: return a - floor(a / b) * b;
            case CT_OP_FLOORDIV: return floor(a / b);
            case CT_OP_LT: return a < b;
            case CT_OP_LE: return a <= b;
            case CT_OP_GT: return a > b;
            case CT_OP_GE: return a >= b;
            case CT_OP_EQ: return a == b;
            case CT_OP_NE: return a != b;
            case CT_OP_MAXIMUM: return a > b ? a : b;
            case CT_OP_MINIMUM: return a < b ? a : b;
            case CT_OP_BITAND: return (double)((long long)a & (long long)b);
            case CT_OP_BITOR: return (double)((long long)a | (long long)b);
            case CT_OP_BITXOR: return (double)((long long)a ^ (long long)b);
            case CT_OP_SHL:
                return (double)((long long)a << ((unsigned long long)b & 63));
            case CT_OP_SHR:
                return (double)((long long)a >> ((unsigned long long)b & 63));
            case CT_OP_LOGICAL_AND: return (a != 0.0) && (b != 0.0);
            case CT_OP_LOGICAL_OR: return (a != 0.0) || (b != 0.0);
            default: return NAN;
        }
    }

    void binary_double(
        const double* a, const double* b, double* out, int n, int op) {
        for (int i = 0; i < n; ++i)
            out[i] = binary_value(a[i], b[i], op);
    }

    void binary_scalar_double(
        const double* a, double b, double* out, int n, int op, int reverse) {
        for (int i = 0; i < n; ++i)
            out[i] = reverse
                ? binary_value(b, a[i], op)
                : binary_value(a[i], b, op);
    }
    void matmul_double(const double* a, const double* b, double* out, int m, int n, int p) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < p; ++j) {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    sum += a[i * n + k] * b[k * p + j];
                }
                out[i * p + j] = sum;
            }
        }
    }
    void unary_double(const double* a, double* out, int n, int op) {
        for (int i = 0; i < n; ++i) {
            double value = a[i];
            switch (op) {
                case CT_OP_SQRT: out[i] = sqrt(value); break;
                case CT_OP_EXP: out[i] = exp(value); break;
                case CT_OP_LOG: out[i] = log(value); break;
                case CT_OP_NEG: out[i] = -value; break;
                case CT_OP_ABS: out[i] = fabs(value); break;
                case CT_OP_ROUND: out[i] = round(value); break;
                case CT_OP_TRUNC: out[i] = trunc(value); break;
                case CT_OP_FLOOR: out[i] = floor(value); break;
                case CT_OP_CEIL: out[i] = ceil(value); break;
                case CT_OP_ISFINITE:
                    out[i] = isfinite(value) ? 1.0 : 0.0; break;
                case CT_OP_ISNAN:
                    out[i] = isnan(value) ? 1.0 : 0.0; break;
                case CT_OP_ISINF:
                    out[i] = isinf(value) ? 1.0 : 0.0; break;
                case CT_OP_LOGICAL_NOT:
                    out[i] = value == 0.0 ? 1.0 : 0.0; break;
                case CT_OP_TANH: out[i] = tanh(value); break;
                /* Branch-free stable form: exp() of a large positive argument
                   overflows, so the sign of the input decides which way the
                   quotient is written. Both branches are the same function. */
                case CT_OP_SIGMOID:
                    if (value >= 0.0) {
                        out[i] = 1.0 / (1.0 + exp(-value));
                    } else {
                        double activation = exp(value);
                        out[i] = activation / (1.0 + activation);
                    }
                    break;
                case CT_OP_SIN: out[i] = sin(value); break;
                case CT_OP_COS: out[i] = cos(value); break;
                case CT_OP_TAN: out[i] = tan(value); break;
                case CT_OP_ASIN: out[i] = asin(value); break;
                case CT_OP_ACOS: out[i] = acos(value); break;
                case CT_OP_ATAN: out[i] = atan(value); break;
                case CT_OP_SINH: out[i] = sinh(value); break;
                case CT_OP_COSH: out[i] = cosh(value); break;
                case CT_OP_ASINH: out[i] = asinh(value); break;
                case CT_OP_ACOSH: out[i] = acosh(value); break;
                case CT_OP_ATANH: out[i] = atanh(value); break;
                case CT_OP_SIGN:
                    out[i] = value > 0.0 ? 1.0 : (value < 0.0 ? -1.0 : 0.0);
                    break;
                case CT_OP_INVERT:
                    out[i] = (double)(~(long long)value); break;
                default: out[i] = value; break;
            }
        }
    }
    void batched_matmul_indexed_double(
        const double* a,
        const double* b,
        double* out,
        const int* a_offsets,
        const int* b_offsets,
        int batch_count,
        int m,
        int n,
        int p) {
        const int output_stride = m * p;
        for (int batch = 0; batch < batch_count; ++batch) {
            matmul_double(
                a + a_offsets[batch],
                b + b_offsets[batch],
                out + batch * output_stride,
                m,
                n,
                p);
        }
    }

    static int is_unary_op(int op) {
        return (
            (op >= CT_OP_SQRT && op <= CT_OP_LOGICAL_NOT)
            || (op >= CT_OP_TANH && op <= CT_OP_INVERT)
            || op == CT_OP_SIGMOID
        );
    }

    int ctensor_execute_primitive_program_slots(
        const CTensorPrimitiveInstruction* instructions,
        int instruction_count,
        double* const* slots,
        int slot_count,
        int element_count) {
        if (!instructions || !slots || instruction_count < 0 ||
            slot_count < 1 || element_count < 0) {
            return 0;
        }
        for (int step = 0; step < instruction_count; ++step) {
            const CTensorPrimitiveInstruction* ins = &instructions[step];
            if (ins->op < 0 || ins->op >= CT_OP_COUNT ||
                ins->out_slot < 0 || ins->out_slot >= slot_count ||
                ins->left_slot < 0 || ins->left_slot >= slot_count ||
                !slots[ins->out_slot] || !slots[ins->left_slot]) {
                return 0;
            }
            const double* left = slots[ins->left_slot];
            double* dest = slots[ins->out_slot];
            if (is_unary_op(ins->op)) {
                if (ins->right_kind != CT_OPERAND_NONE) return 0;
                unary_double(left, dest, element_count, ins->op);
            } else if (ins->right_kind == CT_OPERAND_SLOT) {
                if (ins->right_slot < 0 || ins->right_slot >= slot_count ||
                    !slots[ins->right_slot]) {
                    return 0;
                }
                binary_double(
                    left, slots[ins->right_slot], dest, element_count, ins->op);
            } else if (ins->right_kind == CT_OPERAND_SCALAR) {
                binary_scalar_double(
                    left, ins->right_scalar, dest, element_count, ins->op,
                    ins->reverse);
            } else {
                return 0;
            }
        }
        return 1;
    }

    int ctensor_execute_primitive_program(
        const CTensorPrimitiveInstruction* instructions,
        int instruction_count,
        const double* const* feeds,
        int feed_count,
        double* workspace,
        int slot_count,
        int element_count,
        int output_slot,
        double* output) {
        if (!instructions || !workspace || !output || instruction_count < 0 ||
            feed_count < 0 || feed_count > slot_count || element_count < 0 ||
            output_slot < 0 || output_slot >= slot_count) {
            return 0;
        }
        for (int slot = 0; slot < feed_count; ++slot) {
            if (!feeds || !feeds[slot]) return 0;
            memcpy(
                workspace + (size_t)slot * element_count,
                feeds[slot],
                (size_t)element_count * sizeof(double));
        }
        double** slots = (double**)malloc((size_t)slot_count * sizeof(double*));
        if (!slots) return 0;
        for (int slot = 0; slot < slot_count; ++slot) {
            slots[slot] = workspace + (size_t)slot * element_count;
        }
        int ok = ctensor_execute_primitive_program_slots(
            instructions, instruction_count, slots, slot_count, element_count);
        free(slots);
        if (!ok) return 0;
        memcpy(
            output,
            workspace + (size_t)output_slot * element_count,
            (size_t)element_count * sizeof(double));
        return 1;
    }

    void reduce_dim_double(
        const double* a, double* out, const int* shape, int ndim,
        int dim, int op) {
        int before = 1;
        int after = 1;
        for (int i = 0; i < dim; ++i) before *= shape[i];
        for (int i = dim + 1; i < ndim; ++i) after *= shape[i];
        int count = shape[dim];
        for (int b = 0; b < before; ++b) {
            for (int tail = 0; tail < after; ++tail) {
                double accum;
                if (op == 0 || op == 4) accum = 0.0;
                else if (op == 1 || op == 5) accum = 1.0;
                else accum = a[b * count * after + tail];
                for (int d = 0; d < count; ++d) {
                    double value = a[(b * count + d) * after + tail];
                    if (op == 0) accum += value;
                    else if (op == 1) accum *= value;
                    else if (op == 2 && value < accum) accum = value;
                    else if (op == 3 && value > accum) accum = value;
                    else if (op == 4 && value != 0.0) accum = 1.0;
                    else if (op == 5 && value == 0.0) accum = 0.0;
                }
                out[b * after + tail] = accum;
            }
        }
    }

    void transpose_double(
        const double* a, double* out, const int* shape,
        const int* axes, int ndim) {
        int total = 1;
        int* in_stride = (int*)malloc(ndim * sizeof(int));
        int* out_shape = (int*)malloc(ndim * sizeof(int));
        int* out_stride = (int*)malloc(ndim * sizeof(int));
        for (int i = 0; i < ndim; ++i) {
            total *= shape[i];
            out_shape[i] = shape[axes[i]];
        }
        in_stride[ndim - 1] = 1;
        out_stride[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            in_stride[i] = in_stride[i + 1] * shape[i + 1];
            out_stride[i] = out_stride[i + 1] * out_shape[i + 1];
        }
        for (int flat = 0; flat < total; ++flat) {
            int remaining = flat;
            int input_flat = 0;
            for (int out_axis = 0; out_axis < ndim; ++out_axis) {
                int coordinate = remaining / out_stride[out_axis];
                remaining %= out_stride[out_axis];
                input_flat += coordinate * in_stride[axes[out_axis]];
            }
            out[flat] = a[input_flat];
        }
        free(in_stride);
        free(out_shape);
        free(out_stride);
    }

    void where_double(
        const double* condition, const double* x, const double* y,
        double* out, int n) {
        for (int i = 0; i < n; ++i)
            out[i] = condition[i] != 0.0 ? x[i] : y[i];
    }

    void broadcast_double(
        const double* input, double* output, const int* input_shape,
        int input_ndim, const int* output_shape, int output_ndim) {
        int total = 1;
        int* input_stride = (int*)malloc(input_ndim * sizeof(int));
        int* output_stride = (int*)malloc(output_ndim * sizeof(int));
        for (int i = 0; i < output_ndim; ++i) total *= output_shape[i];
        input_stride[input_ndim - 1] = 1;
        output_stride[output_ndim - 1] = 1;
        for (int i = input_ndim - 2; i >= 0; --i)
            input_stride[i] = input_stride[i + 1] * input_shape[i + 1];
        for (int i = output_ndim - 2; i >= 0; --i)
            output_stride[i] = output_stride[i + 1] * output_shape[i + 1];
        int offset = output_ndim - input_ndim;
        for (int flat = 0; flat < total; ++flat) {
            int remaining = flat;
            int input_flat = 0;
            for (int axis = 0; axis < output_ndim; ++axis) {
                int coordinate = remaining / output_stride[axis];
                remaining %= output_stride[axis];
                int input_axis = axis - offset;
                if (input_axis >= 0 && input_shape[input_axis] != 1)
                    input_flat += coordinate * input_stride[input_axis];
            }
            output[flat] = input[input_flat];
        }
        free(input_stride);
        free(output_stride);
    }

    void cumsum_dim_double(
        const double* input, double* output, const int* shape,
        int ndim, int dim) {
        int before = 1;
        int after = 1;
        for (int i = 0; i < dim; ++i) before *= shape[i];
        for (int i = dim + 1; i < ndim; ++i) after *= shape[i];
        int count = shape[dim];
        for (int b = 0; b < before; ++b) {
            for (int tail = 0; tail < after; ++tail) {
                double accum = 0.0;
                for (int d = 0; d < count; ++d) {
                    int index = (b * count + d) * after + tail;
                    accum += input[index];
                    output[index] = accum;
                }
            }
        }
    }

    void argreduce_dim_double(
        const double* input, double* output, const int* shape,
        int ndim, int dim, int find_max) {
        int before = 1;
        int after = 1;
        for (int i = 0; i < dim; ++i) before *= shape[i];
        for (int i = dim + 1; i < ndim; ++i) after *= shape[i];
        int count = shape[dim];
        for (int b = 0; b < before; ++b) {
            for (int tail = 0; tail < after; ++tail) {
                int best = 0;
                double best_value = input[b * count * after + tail];
                for (int d = 1; d < count; ++d) {
                    double value = input[(b * count + d) * after + tail];
                    if ((find_max && value > best_value)
                        || (!find_max && value < best_value)) {
                        best = d;
                        best_value = value;
                    }
                }
                output[b * after + tail] = (double)best;
            }
        }
    }

    void repeat_interleave_double(
        const double* input, double* output, const int* shape,
        int ndim, int dim, int repeats) {
        int before = 1;
        int after = 1;
        for (int i = 0; i < dim; ++i) before *= shape[i];
        for (int i = dim + 1; i < ndim; ++i) after *= shape[i];
        int count = shape[dim];
        for (int b = 0; b < before; ++b)
            for (int d = 0; d < count; ++d)
                for (int r = 0; r < repeats; ++r)
                    memcpy(
                        output + ((b * count + d) * repeats + r) * after,
                        input + (b * count + d) * after,
                        after * sizeof(double));
    }

    void tile_double(
        const double* input, double* output, const int* input_shape,
        const int* output_shape, int ndim) {
        int total = 1;
        int* input_stride = (int*)malloc(ndim * sizeof(int));
        int* output_stride = (int*)malloc(ndim * sizeof(int));
        for (int i = 0; i < ndim; ++i) total *= output_shape[i];
        input_stride[ndim - 1] = 1;
        output_stride[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            input_stride[i] = input_stride[i + 1] * input_shape[i + 1];
            output_stride[i] = output_stride[i + 1] * output_shape[i + 1];
        }
        for (int flat = 0; flat < total; ++flat) {
            int remaining = flat;
            int input_flat = 0;
            for (int axis = 0; axis < ndim; ++axis) {
                int coordinate = remaining / output_stride[axis];
                remaining %= output_stride[axis];
                input_flat += (coordinate % input_shape[axis])
                    * input_stride[axis];
            }
            output[flat] = input[input_flat];
        }
        free(input_stride);
        free(output_stride);
    }

    void index_select_double(
        const double* input, double* output, const int* shape,
        int ndim, int dim, const int* indices, int index_count) {
        int before = 1;
        int after = 1;
        for (int i = 0; i < dim; ++i) before *= shape[i];
        for (int i = dim + 1; i < ndim; ++i) after *= shape[i];
        int count = shape[dim];
        for (int b = 0; b < before; ++b)
            for (int j = 0; j < index_count; ++j)
                memcpy(
                    output + (b * index_count + j) * after,
                    input + (b * count + indices[j]) * after,
                    after * sizeof(double));
    }

    void slice_copy_double(
        const double* input, double* output, const int* shape,
        int ndim, int dim, int start, int step, int count) {
        int before = 1;
        int after = 1;
        for (int i = 0; i < dim; ++i) before *= shape[i];
        for (int i = dim + 1; i < ndim; ++i) after *= shape[i];
        int source_count = shape[dim];
        for (int b = 0; b < before; ++b)
            for (int j = 0; j < count; ++j)
                memcpy(
                    output + (b * count + j) * after,
                    input + (
                        b * source_count + start + j * step
                    ) * after,
                    after * sizeof(double));
    }

    void index_assign_double(
        double* target, const int* shape, int ndim,
        const int* axis_offsets, const int* axis_indices,
        const double* values, int value_count) {
        if (ndim == 0) {
            target[0] = values[0];
            return;
        }
        int selected_count = 1;
        for (int axis = 0; axis < ndim; ++axis) {
            int axis_count = axis_offsets[axis + 1] - axis_offsets[axis];
            selected_count *= axis_count;
        }
        for (int flat = 0; flat < selected_count; ++flat) {
            int target_flat = 0;
            for (int axis = 0; axis < ndim; ++axis) {
                int axis_count = axis_offsets[axis + 1] - axis_offsets[axis];
                int selection_stride = 1;
                int target_stride = 1;
                for (int later = axis + 1; later < ndim; ++later) {
                    selection_stride *= (
                        axis_offsets[later + 1] - axis_offsets[later]
                    );
                    target_stride *= shape[later];
                }
                int coordinate = (flat / selection_stride) % axis_count;
                int selected = axis_indices[axis_offsets[axis] + coordinate];
                target_flat += selected * target_stride;
            }
            target[target_flat] = values[value_count == 1 ? 0 : flat];
        }
    }

    void index_set_double(
        const double* input, double* output, const int* shape, int ndim,
        const int* axis_offsets, const int* axis_indices,
        const double* values, int value_count) {
        int element_count = 1;
        for (int axis = 0; axis < ndim; ++axis)
            element_count *= shape[axis];
        memcpy(output, input, element_count * sizeof(double));
        index_assign_double(
            output, shape, ndim, axis_offsets, axis_indices,
            values, value_count);
    }

    void unfold2d_double(
        const double* input, double* output,
        int n, int c, int h, int w,
        int kernel_h, int kernel_w,
        int stride_h, int stride_w,
        int padding_h, int padding_w,
        int dilation_h, int dilation_w) {
        int effective_h = (kernel_h - 1) * dilation_h + 1;
        int effective_w = (kernel_w - 1) * dilation_w + 1;
        int output_h = (h + 2 * padding_h - effective_h) / stride_h + 1;
        int output_w = (w + 2 * padding_w - effective_w) / stride_w + 1;
        for (int batch = 0; batch < n; ++batch)
            for (int channel = 0; channel < c; ++channel)
                for (int kh = 0; kh < kernel_h; ++kh)
                    for (int kw = 0; kw < kernel_w; ++kw)
                        for (int oh = 0; oh < output_h; ++oh)
                            for (int ow = 0; ow < output_w; ++ow) {
                                int ih = oh * stride_h - padding_h + kh * dilation_h;
                                int iw = ow * stride_w - padding_w + kw * dilation_w;
                                int column = (((channel * kernel_h + kh) * kernel_w) + kw);
                                int output_index = (
                                    (batch * c * kernel_h * kernel_w + column)
                                    * output_h + oh) * output_w + ow;
                                output[output_index] = (
                                    ih >= 0 && ih < h && iw >= 0 && iw < w
                                ) ? input[((batch * c + channel) * h + ih) * w + iw]
                                  : 0.0;
                            }
    }

    void fold2d_double(
        const double* columns, double* output,
        int n, int c, int h, int w,
        int kernel_h, int kernel_w,
        int stride_h, int stride_w,
        int padding_h, int padding_w,
        int dilation_h, int dilation_w) {
        int effective_h = (kernel_h - 1) * dilation_h + 1;
        int effective_w = (kernel_w - 1) * dilation_w + 1;
        int output_h = (h + 2 * padding_h - effective_h) / stride_h + 1;
        int output_w = (w + 2 * padding_w - effective_w) / stride_w + 1;
        memset(output, 0, (size_t)n * c * h * w * sizeof(double));
        for (int batch = 0; batch < n; ++batch)
            for (int channel = 0; channel < c; ++channel)
                for (int kh = 0; kh < kernel_h; ++kh)
                    for (int kw = 0; kw < kernel_w; ++kw)
                        for (int oh = 0; oh < output_h; ++oh)
                            for (int ow = 0; ow < output_w; ++ow) {
                                int ih = oh * stride_h - padding_h + kh * dilation_h;
                                int iw = ow * stride_w - padding_w + kw * dilation_w;
                                if (ih < 0 || ih >= h || iw < 0 || iw >= w)
                                    continue;
                                int column = (((channel * kernel_h + kh) * kernel_w) + kw);
                                int column_index = (
                                    (batch * c * kernel_h * kernel_w + column)
                                    * output_h + oh) * output_w + ow;
                                output[((batch * c + channel) * h + ih) * w + iw]
                                    += columns[column_index];
                            }
    }

    void sign_double(const double* input, double* output, int n) {
        for (int i = 0; i < n; ++i)
            output[i] = input[i] > 0.0 ? 1.0 : (
                input[i] < 0.0 ? -1.0 : 0.0);
    }

    int count_true_double(const double* mask, int n) {
        int count = 0;
        for (int i = 0; i < n; ++i)
            if (mask[i] != 0.0) count++;
        return count;
    }

    void mask_select_double(
        const double* input, const double* mask, double* output, int n) {
        int target = 0;
        for (int i = 0; i < n; ++i)
            if (mask[i] != 0.0) output[target++] = input[i];
    }

    void increment_mask_double(double* input, const double* mask, int n) {
        for (int i = 0; i < n; ++i)
            if (mask[i] != 0.0) input[i] += 1.0;
    }

    void cast_double_to_int_values(const double* a, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = (double)((long long)a[i]);
    }

    void cast_double_to_float_values(const double* a, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = (double)((float)a[i]);
    }

    /* double -> double is the identity over values, but it is still a cast:
     * the reference semantics (numpy astype) always copies, so aliasing the
     * source buffer would change behaviour under later in-place mutation. */
    void cast_double_to_double_values(const double* a, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i];
    }

    void cast_double_to_bool_values(const double* a, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = (a[i] != 0.0) ? 1.0 : 0.0;
    }

    void log_softmax_1d(const double* a, double* out, int n) {
        double max_val = a[0];
        for (int i = 1; i < n; ++i) {
            if (a[i] > max_val) max_val = a[i];
        }
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            out[i] = exp(a[i] - max_val);
            sum += out[i];
        }
        for (int i = 0; i < n; ++i) {
            out[i] = log(out[i] / sum);
        }
    }

    typedef struct {
        double* out;
        int axis;
        int* out_index;
    } log_softmax_ctx;

    static void log_softmax_callback(const double* slice, int stride, int idx, void* user_data) {
        log_softmax_ctx* ctx = (log_softmax_ctx*)user_data;
        double max_val = slice[0];
        for (int i = 1; i < ctx->axis; ++i) {
            double v = slice[i * stride];
            if (v > max_val) max_val = v;
        }
        double sum = 0.0;
        for (int i = 0; i < ctx->axis; ++i) {
            double e = exp(slice[i * stride] - max_val);
            ctx->out[*ctx->out_index + i] = e;
            sum += e;
        }
        for (int i = 0; i < ctx->axis; ++i) {
            double e = ctx->out[*ctx->out_index + i];
            ctx->out[*ctx->out_index + i] = log(e / sum);
        }
        *ctx->out_index += ctx->axis;
    }

    void pad_double_nd(const double* input, double* output, const int* shape,
                       const int* new_shape, const int* left_pad, int dims,
                       double value) {
        int i;
        int total_out = 1;
        for (i = 0; i < dims; ++i) total_out *= new_shape[i];
        for (i = 0; i < total_out; ++i) output[i] = value;

        int input_size = 1;
        for (i = 0; i < dims; ++i) input_size *= shape[i];

        int* in_stride = (int*)malloc(sizeof(int) * dims);
        int* out_stride = (int*)malloc(sizeof(int) * dims);
        int* idx = (int*)malloc(sizeof(int) * dims);

        in_stride[dims - 1] = 1;
        out_stride[dims - 1] = 1;
        for (i = dims - 2; i >= 0; --i) {
            in_stride[i] = in_stride[i + 1] * shape[i + 1];
            out_stride[i] = out_stride[i + 1] * new_shape[i + 1];
        }

        for (i = 0; i < dims; ++i) idx[i] = 0;
        for (i = 0; i < input_size; ++i) {
            int out_index = 0;
            for (int d = 0; d < dims; ++d) {
                out_index += (idx[d] + left_pad[d]) * out_stride[d];
            }
            output[out_index] = input[i];
            idx[dims - 1]++;
            for (int d = dims - 1; d > 0; --d) {
                if (idx[d] >= shape[d]) {
                    idx[d] = 0;
                    idx[d - 1]++;
                }
            }
        }

        free(in_stride);
        free(out_stride);
        free(idx);

    }

    void mean_dim(const double* a, double* out, const int* shape, int ndim, int dim) {
        int after = 1;
        for (int i = dim + 1; i < ndim; ++i) after *= shape[i];
        double* accum = (double*)malloc(after * sizeof(double));
        mean_dim_ctx ctx = {out, shape[dim], after, 0, accum};
        for_each_cell_along_dim(a, shape, ndim, dim, mean_dim_callback, &ctx);
        free(accum);
    }

    void gather_pairs_2d(const double* a, const int* rows, const int* cols,
                         double* out, int n_pairs, int stride) {
        for (int i = 0; i < n_pairs; ++i) {
            out[i] = a[rows[i] * stride + cols[i]];
        }
    }
    double sum_double(const double* a, int n) {
        double s = 0;
        for (int i = 0; i < n; i++) {
            s += a[i];
        }
        return s;
    }

    void create_arange(double start, double step, int n, double* out) {
        for (int i = 0; i < n; i++) {
            out[i] = start + i * step;
        }
    }

    void topk_double(const double* a, int n, int k, int* indices, double* out) {
        int i, j, best;
        double best_val;
        int *used = (int*)malloc(n * sizeof(int));
        for (i = 0; i < n; i++) used[i] = 0;
        for (i = 0; i < k; i++) {
            best = -1;
            best_val = -1e300; // a very small number as initial comparator
            for (j = 0; j < n; j++) {
                if (!used[j] && a[j] > best_val) {
                    best_val = a[j];
                    best = j;
                }
            }
            if (best != -1) {
                used[best] = 1;
                indices[i] = best;
                out[i] = best_val;
            } else {
                indices[i] = -1;
                out[i] = 0.0;
            }
        }
        free(used);

    }

    void topk_double_dim(
        const double* a,
        const int* shape,
        int ndim,
        int dim,
        int k,
        double* indices,
        double* out) {
        int after = 1;
        for (int i = dim + 1; i < ndim; ++i) after *= shape[i];
        double* buffer = (double*)malloc(shape[dim] * after * sizeof(double));
        topk_dim_ctx ctx = {out, indices, k, shape[dim], after, 0, buffer};
        for_each_cell_along_dim(a, shape, ndim, dim, topk_dim_callback, &ctx);
        free(buffer);
    }

    void log_softmax_dim(
        const double* a,
        const int* shape,
        int ndim,
        int dim,
        double* out) {
        int after = 1;
        for (int i = dim + 1; i < ndim; ++i) after *= shape[i];
        int dim_size = shape[dim];
        double* buffer = (double*)malloc(dim_size * after * sizeof(double));
        double* max_vals = (double*)malloc(after * sizeof(double));
        double* sum_vals = (double*)malloc(after * sizeof(double));
        log_softmax_dim_ctx ctx = {out, dim_size, after, 0, buffer, max_vals, sum_vals};
        for_each_cell_along_dim(a, shape, ndim, dim, log_softmax_dim_callback, &ctx);
        free(buffer);
        free(max_vals);
        free(sum_vals);
    }

    void for_each_cell_along_dim(
        const double* data,
        const int* shape,
        int ndim,
        int batch_dim,
        void (*callback)(const double*, int, int, void*),
        void* user_data) {
        // Compute strides
        int* strides = (int*)malloc(ndim * sizeof(int));
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape[i + 1];

        int before = 1, after = 1;
        for (int i = 0; i < batch_dim; ++i) before *= shape[i];
        for (int i = batch_dim + 1; i < ndim; ++i) after *= shape[i];
        int batch = shape[batch_dim];
        int cell_len = after;

        // Iterate over all cells
        for (int b = 0; b < before; ++b) {
            // Compute base offset for this cell
            int base = 0;
            int rem = b;
            for (int i = batch_dim - 1; i >= 0; --i) {
                int idx = rem % shape[i];
                base += idx * strides[i];
                rem /= shape[i];
            }
            for (int i = 0; i < batch; ++i) {
                int cell_offset = base + i * strides[batch_dim];
                callback(data + cell_offset, cell_len, i, user_data);
            }
        }
        free(strides);
    }

    void stack_double(
        const double** tensors,
        int num_tensors,
        const int* shape,
        int ndim,
        int dim,
        double* out) {
        int before = 1;
        for (int i = 0; i < dim; ++i) before *= shape[i];
        int after = 1;
        for (int i = dim; i < ndim; ++i) after *= shape[i];
        int out_stride = num_tensors * after;
        for (int b = 0; b < before; ++b) {
            for (int t = 0; t < num_tensors; ++t) {
                const double* src = tensors[t] + b * after;
                double* dst = out + b * out_stride + t * after;
                memcpy(dst, src, after * sizeof(double));
            }
        }
    }

    void cat_double(
        const double** tensors,
        const int* dim_sizes,
        int num_tensors,
        const int* shape,
        int ndim,
        int dim,
        double* out) {
        int before = 1;
        for (int i = 0; i < dim; ++i) before *= shape[i];
        int after = 1;
        for (int i = dim + 1; i < ndim; ++i) after *= shape[i];
        int total_dim = 0;
        for (int t = 0; t < num_tensors; ++t) total_dim += dim_sizes[t];
        int out_stride = total_dim * after;
        for (int b = 0; b < before; ++b) {
            int dest_offset = b * out_stride;
            for (int t = 0; t < num_tensors; ++t) {
                int dim_size = dim_sizes[t];
                const double* src = tensors[t] + b * dim_size * after;
                double* dst = out + dest_offset;
                memcpy(dst, src, dim_size * after * sizeof(double));
                dest_offset += dim_size * after;
            }
        }
    }
