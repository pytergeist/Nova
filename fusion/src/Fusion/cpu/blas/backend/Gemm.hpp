#ifndef FUSION_CPU_BLAS_GEMM_HPP
#define FUSION_CPU_BLAS_GEMM_HPP

#include <cstddef>
#include <type_traits>

#include "BlasCblas.hpp"

namespace fusion::blas {

    // single GEMM
    template <typename T>
    inline void gemm_rowmajor_nn(const T* A, const T* B, T* C,
                                 int m, int n, int k,
                                 T alpha, T beta) {
        static_assert(std::is_same_v<T, float>,
                      "gemm_rowmajor_nn: only float is implemented currently");
        backend::gemm_rowmajor_nn(A, B, C, m, n, k, alpha, beta);
    }

    // batched GEMM
    template <typename T>
    inline void batched_gemm_rowmajor_nn(const T* baseA, const T* baseB, T* baseC,
                                         int m, int n, int k,
                                         std::size_t batch,
                                         T alpha, T beta) {
        static_assert(std::is_same_v<T, float>,
                      "batched_gemm_rowmajor_nn: only float is implemented currently");
        backend::batched_gemm_rowmajor_nn(baseA, baseB, baseC,
                                         m, n, k, batch, alpha, beta);
    }

} // namespace fusion::blas

#endif // FUSION_CPU_BLAS_GEMM_HPP
