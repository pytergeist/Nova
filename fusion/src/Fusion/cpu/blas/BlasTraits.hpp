#ifndef FUSION_CPU_BLAS_TRAITS_HPP
#define FUSION_CPU_BLAS_TRAITS_HPP

#include <cstddef>
#include "BlasTags.hpp"
#include "Gemm.hpp"

namespace fusion::blas {

  template <class Tag, typename T>
  struct blas_traits {
    static constexpr bool available = false;
  };

  template <typename T>
  struct blas_traits<BatchedGemmBLAS, T> {
    static constexpr bool available = true;

    static void execute(const T* A, const T* B, T* C,
                        int m, int n, int k, std::size_t batch,
                        T alpha, T beta) {
      blas_ops::batched_gemm<T>(A, B, C, m, n, k, batch, alpha, beta);
    }
  };

} // namespace fusion::blas

#endif
