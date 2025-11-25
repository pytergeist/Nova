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

template <typename T> class Tensor;

namespace blas_ops {

template <typename T>
void gemm(Tensor<T> const &v1, std::vector<size_t> const &shapeA,
          Tensor<T> const &v2, std::vector<size_t> const &shapeB,
          std::vector<T> &res, const T alpha = 1, const T beta = 0) {
   const size_t rankA = shapeA.size();
   const size_t rankB = shapeB.size();
   int m = int(shapeA[rankA - 2]);
   int k = int(shapeA[rankA - 1]);
   int n = int(shapeB[rankB - 1]);

   size_t batch = 1;
   for (size_t i = 0; i < rankA - 2; ++i) {
      batch *= shapeA[i];
   }

   const T *baseA = v1.raw_data().template data_as<const T>();
   const T *baseB = v2.raw_data().template data_as<const T>();
   T *baseC = res.data();

   for (size_t b = 0; b < batch; ++b) {
      const T *A = baseA + b * (size_t(m) * k);
      const T *B = baseB + b * (size_t(k) * n);
      float *C = baseC + b * (size_t(m) * n);

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
