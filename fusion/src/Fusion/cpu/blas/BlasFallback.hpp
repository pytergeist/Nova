#ifndef FUSION_CPU_BLAS_FALLBACK_HPP
#define FUSION_CPU_BLAS_FALLBACK_HPP

#include <cstddef>

namespace fusion::blas::backend {

template <typename T>
inline void gemm_rowmajor_nn(const T *A, const T *B, T *C, int m, int n, int k,
                             T alpha, T beta) {
   for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
         T acc = T(0);
         const T *arow = A + i * k;
         const T *bcol = B + j; // stride n
         for (int p = 0; p < k; ++p) {
            acc += arow[p] * bcol[p * n];
         }
         C[i * n + j] = alpha * acc + beta * C[i * n + j];
      }
   }
}

template <typename T>
inline void batched_gemm_rowmajor_nn(const T *baseA, const T *baseB, T *baseC,
                                     int m, int n, int k, std::size_t batch,
                                     T alpha, T beta) {
   const std::size_t Asz = std::size_t(m) * std::size_t(k);
   const std::size_t Bsz = std::size_t(k) * std::size_t(n);
   const std::size_t Csz = std::size_t(m) * std::size_t(n);

   for (std::size_t b = 0; b < batch; ++b) {
      const T *A = baseA + b * Asz;
      const T *B = baseB + b * Bsz;
      T *C = baseC + b * Csz;
      gemm_rowmajor_nn(A, B, C, m, n, k, alpha, beta);
   }
}

} // namespace fusion::blas::backend

#endif // FUSION_CPU_BLAS_FALLBACK_HPP
