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
