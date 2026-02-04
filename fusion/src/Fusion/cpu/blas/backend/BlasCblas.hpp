#ifndef FUSION_CPU_BLAS_CBLAS_HPP
#define FUSION_CPU_BLAS_CBLAS_HPP

#include <cstddef>
#include <type_traits>

#if defined(__APPLE__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

namespace fusion::blas::backend {

// float specialization (extend later)
inline void gemm_rowmajor_nn(const float *A, const float *B, float *C, int m,
                             int n, int k, float alpha, float beta) {
   cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k,
               B, n, beta, C, n);
}

inline void batched_gemm_rowmajor_nn(const float *baseA, const float *baseB,
                                     float *baseC, int m, int n, int k,
                                     std::size_t batch, float alpha,
                                     float beta) {
   const std::size_t Asz = std::size_t(m) * std::size_t(k);
   const std::size_t Bsz = std::size_t(k) * std::size_t(n);
   const std::size_t Csz = std::size_t(m) * std::size_t(n);

   for (std::size_t b = 0; b < batch; ++b) {
      const float *A = baseA + b * Asz;
      const float *B = baseB + b * Bsz;
      float *C = baseC + b * Csz;
      gemm_rowmajor_nn(A, B, C, m, n, k, alpha, beta);
   }
}

} // namespace fusion::blas::backend

#endif // FUSION_CPU_BLAS_CBLAS_HPP
