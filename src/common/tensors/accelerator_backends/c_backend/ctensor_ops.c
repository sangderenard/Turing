
    #include <math.h>
    #include <stdlib.h>
    #include <stddef.h>

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

    void add_double(const double* a, const double* b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];
    }
    void sub_double(const double* a, const double* b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] - b[i];
    }
    void mul_double(const double* a, const double* b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] * b[i];
    }
    void div_double(const double* a, const double* b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] / b[i];
    }
    void pow_double(const double* a, const double* b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = pow(a[i], b[i]);
    }
    void mod_double(const double* a, const double* b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = fmod(a[i], b[i]);
    }
    void floordiv_double(const double* a, const double* b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = floor(a[i] / b[i]);
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
    // Scalar ops
    void add_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] + b;
    }
    void subtract_const(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] - b;
    }
    void rsubtract_const(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = b - a[i];
    }
    void mul_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] * b;
    }
    void divide_const(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] / b;
    }
    void rdivide_const(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = b / a[i];
    }
    void pow_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = pow(a[i], b);
    }
    void rpow_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = pow(b, a[i]);
    }
    void mod_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = fmod(a[i], b);
    }
    void rmod_scalar(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = fmod(b, a[i]);
    }
    void floor_div_const(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = floor(a[i] / b);
    }
    void rfloor_div_const(const double* a, double b, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = floor(b / a[i]);
    }
    void sqrt_double(const double* a, double* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = sqrt(a[i]);
    }

    void cast_double_to_int(const double* a, int* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = (int)a[i];
    }

    void cast_double_to_float(const double* a, float* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = (float)a[i];
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
