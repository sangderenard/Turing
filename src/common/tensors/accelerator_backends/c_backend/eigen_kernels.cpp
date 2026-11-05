// The Eigen tensor-kernel lane: repository kernel symbols implemented over
// Eigen, linked into LLVM products when the work contract's
// ``tensor_kernel_lane`` is ``"eigen"``.
//
// Each function keeps the exact C signature of the authored kernel it
// stands in for (``ctensor_ops.c`` / the LLVM kernel module), so the
// emitted call sites are unchanged; only the definition differs.  The
// arithmetic is Eigen's blocked, vectorized GEMM: NOT the repository's
// compensated inner product, so results differ in the last bits.  That is
// why the lane is a named contract and never a default.
#include <Eigen/Core>
#include <cmath>
#include <cstdlib>

// The LLVM product is linked as plain C (no C++ runtime), and Eigen's GEMM
// allocates its blocking workspace with operator new.  These are the
// standard replaceable global allocation functions, backed by malloc/free;
// built with -fno-exceptions, so allocation failure returns null.
void *operator new(std::size_t size) { return std::malloc(size ? size : 1); }
void *operator new[](std::size_t size) { return std::malloc(size ? size : 1); }
void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete[](void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void *pointer, std::size_t) noexcept { std::free(pointer); }

namespace {
using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}

extern "C" void matmul_double(const double* a, const double* b, double* out, int m, int n, int p) {
    Eigen::Map<const RowMajor> A(a, m, n);
    Eigen::Map<const RowMajor> B(b, n, p);
    Eigen::Map<RowMajor> C(out, m, p);
    C.noalias() = A * B;
}

// softmax over the last axis, rows x cols, with the repository's exact
// decomposition and order (max from -inf taking value > accum, subtract,
// libm exp, left-to-right sum from 0, divide), so results match the
// five-kernel form bit for bit without its temporaries and broadcast copies.
extern "C" void softmax_lastdim_double(const double* x, double* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const double* xr = x + (long long)r * cols;
        double* orow = out + (long long)r * cols;
        double peak = -INFINITY;
        for (int c = 0; c < cols; ++c) peak = xr[c] > peak ? xr[c] : peak;
        double total = 0.0;
        for (int c = 0; c < cols; ++c) { double e = std::exp(xr[c] - peak); orow[c] = e; total += e; }
        for (int c = 0; c < cols; ++c) orow[c] = orow[c] / total;
    }
}

namespace {
struct Add { static double f(double a, double b) { return a + b; } };
struct Sub { static double f(double a, double b) { return a - b; } };
struct Mul { static double f(double a, double b) { return a * b; } };
struct Div { static double f(double a, double b) { return a / b; } };
struct Max { static double f(double a, double b) { return a > b ? a : b; } };
struct Min { static double f(double a, double b) { return a < b ? a : b; } };

template <class Op>
void broadcast_binary(const double* a, const double* b, double* out, const int* shape,
                      const int* a_strides, const int* b_strides, int rank, int count) {
    const int inner = rank ? shape[rank - 1] : 1;
    const int as = rank ? a_strides[rank - 1] : 0, bs = rank ? b_strides[rank - 1] : 0;
    const int outer = inner ? count / inner : 0;
    int index[8] = {0};
    long long ao = 0, bo = 0;
    for (int o = 0; o < outer; ++o) {
        const double* pa = a + ao;
        const double* pb = b + bo;
        double* po = out + (long long)o * inner;
        // The operation is fixed per call; each inner loop is unit-stride or
        // a scalar broadcast, which the vectorizer takes.
        if (as == 1 && bs == 1) { for (int i = 0; i < inner; ++i) po[i] = Op::f(pa[i], pb[i]); }
        else if (as == 1 && bs == 0) { const double v = *pb; for (int i = 0; i < inner; ++i) po[i] = Op::f(pa[i], v); }
        else if (as == 0 && bs == 1) { const double v = *pa; for (int i = 0; i < inner; ++i) po[i] = Op::f(v, pb[i]); }
        else { for (int i = 0; i < inner; ++i) po[i] = Op::f(pa[(long long)i * as], pb[(long long)i * bs]); }
        for (int d = rank - 2; d >= 0; --d) {
            ++index[d]; ao += a_strides[d]; bo += b_strides[d];
            if (index[d] < shape[d]) break;
            ao -= (long long)a_strides[d] * shape[d]; bo -= (long long)b_strides[d] * shape[d]; index[d] = 0;
        }
    }
}
}  // namespace

// out = a op b over shape (rank <= 8), each operand addressed by element
// strides with 0 on its broadcast axes.  op is the CTensorOp code.
extern "C" void broadcast_binary_double(const double* a, const double* b, double* out, const int* shape,
                                        const int* a_strides, const int* b_strides, int rank, int count, int op) {
    switch (op) {
        case 0: broadcast_binary<Add>(a, b, out, shape, a_strides, b_strides, rank, count); break;
        case 1: broadcast_binary<Sub>(a, b, out, shape, a_strides, b_strides, rank, count); break;
        case 2: broadcast_binary<Mul>(a, b, out, shape, a_strides, b_strides, rank, count); break;
        case 3: broadcast_binary<Div>(a, b, out, shape, a_strides, b_strides, rank, count); break;
        case 26: broadcast_binary<Max>(a, b, out, shape, a_strides, b_strides, rank, count); break;
        case 27: broadcast_binary<Min>(a, b, out, shape, a_strides, b_strides, rank, count); break;
        default: break;   // the front end emits only BROADCAST_BINARY_OPCODES
    }
}

// im2col with the repository layout (batch, channel, kh, kw, oh, ow), the
// same order as ctensor_ops.c's unfold2d_double: nested loops instead of the
// authored LLVM form's per-element six-way div/rem decode.  Each output row
// is decided once (padding row or shifted input row), so the inner loop is a
// bounded copy.  Values are copied, never computed: results are identical.
extern "C" void unfold2d_double(const double* input, double* output,
                                int n, int c, int h, int w, int kernel_h, int kernel_w,
                                int stride_h, int stride_w, int padding_h, int padding_w,
                                int dilation_h, int dilation_w) {
    const int effective_h = (kernel_h - 1) * dilation_h + 1;
    const int effective_w = (kernel_w - 1) * dilation_w + 1;
    const int output_h = (h + 2 * padding_h - effective_h) / stride_h + 1;
    const int output_w = (w + 2 * padding_w - effective_w) / stride_w + 1;
    double* o = output;
    for (int batch = 0; batch < n; ++batch)
        for (int channel = 0; channel < c; ++channel) {
            const double* plane = input + (long long)(batch * c + channel) * h * w;
            for (int kh = 0; kh < kernel_h; ++kh)
                for (int kw = 0; kw < kernel_w; ++kw) {
                    const int column_offset = kw * dilation_w - padding_w;
                    for (int oh = 0; oh < output_h; ++oh, o += output_w) {
                        const int ih = oh * stride_h - padding_h + kh * dilation_h;
                        if (ih < 0 || ih >= h) {
                            for (int ow = 0; ow < output_w; ++ow) o[ow] = 0.0;
                            continue;
                        }
                        const double* row = plane + (long long)ih * w;
                        for (int ow = 0; ow < output_w; ++ow) {
                            const int iw = ow * stride_w + column_offset;
                            o[ow] = (iw >= 0 && iw < w) ? row[iw] : 0.0;
                        }
                    }
                }
        }
}
