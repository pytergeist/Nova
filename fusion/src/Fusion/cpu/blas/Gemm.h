#ifndef _GEMM_H
#define _GEMM_H

#include <vector>

#if defined(__APPLE__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

/* TODO: The current GeMM kernel below does not go through dispatch and
 * therefore does not have contiguous hot path optimisation - therefore it is
 * slower in the bench marks */

namespace blas_ops {

template <typename T>
void batched_gemm(const T *baseA, const T *baseB, T *baseC, int m, int n, int k,
                  const std::size_t batch, const T alpha = 1,
                  const T beta = 0) {

   for (size_t b = 0; b < batch; ++b) {
      const T *A = baseA + b * (size_t(m) * k);
      const T *B = baseB + b * (size_t(k) * n);
      T *C = baseC + b * (size_t(m) * n);

      if constexpr (std::is_same_v<T, float>) {

         cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
                     A, k, B, n, beta, C, n);
      } else {
         static_assert(!sizeof(T), "GEMM currently only implemented for float");
      }
   }
}
} // namespace blas_ops

#endif // _GEMM_H
