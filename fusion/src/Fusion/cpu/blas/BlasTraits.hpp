#ifndef FUSION_CPU_BLAS_TRAITS_HPP
#define FUSION_CPU_BLAS_TRAITS_HPP

#include <iostream>
#include <type_traits>

#include "Fusion/core/TensorPlan.h" // GemmLikeDesc
#include "Fusion/cpu/blas/BlasTags.hpp"
#include "Fusion/cpu/blas/backend/Gemm.hpp"

namespace fusion::blas {

template <class Tag, typename T> struct blas_traits {
   static constexpr bool available = false;

   static bool can_execute(const GemmLikeDesc &) { return false; }

   static void execute(const T *, const T *, T *, const GemmLikeDesc &, T, T) {
      // no-op default; should never be called when available==false
   }
};

// ------------------- Batched GEMM (row-major NN contiguous)
// -------------------
template <typename T> struct blas_traits<BatchedGemmBLAS, T> {
   static constexpr bool available = std::is_same_v<T, float>;

   static bool can_execute(const GemmLikeDesc &g) {
      if constexpr (!available)
         return false;

      const bool no_trans = (!g.a_transpose && !g.b_transpose);

      // EXPECTING ELEMENT STRIDES (not bytes):
      // Row-major contiguous:
      //   C[M,N]: cs=1, rs=N
      //   A[M,K]: cs=1, rs=K
      //   B[K,N]: cs=1, rs=N
      const bool out_rowmajor_contig =
          (g.out_cs == 1) && (g.out_rs == static_cast<std::int64_t>(g.N));

      const bool a_rowmajor_contig =
          (g.a_cs == 1) && (g.a_rs == static_cast<std::int64_t>(g.K));

      const bool b_rowmajor_contig =
          (g.b_cs == 1) && (g.b_rs == static_cast<std::int64_t>(g.N));

      const bool ok = no_trans && out_rowmajor_contig && a_rowmajor_contig &&
                      b_rowmajor_contig;

      //      std::cout << "can_execute=" << ok << " no_trans=" << no_trans
      //                << " out(cs,rs)=(" << g.out_cs << "," << g.out_rs << ")"
      //                << " a(cs,rs)=(" << g.a_cs << "," << g.a_rs << ")"
      //                << " b(cs,rs)=(" << g.b_cs << "," << g.b_rs << ")"
      //                << " (elem strides)"
      //                << " N=" << g.N << " K=" << g.K << "\n";

      return ok;
   }

   static void execute(const T *A, const T *B, T *C, const GemmLikeDesc &g,
                       T alpha, T beta) {
      // Assumes can_execute(g) was true.
      batched_gemm_rowmajor_nn<T>(A, B, C, static_cast<int>(g.M),
                                  static_cast<int>(g.N), static_cast<int>(g.K),
                                  static_cast<std::size_t>(g.batch), alpha,
                                  beta);
   }
};

} // namespace fusion::blas

#endif // FUSION_CPU_BLAS_TRAITS_HPP
