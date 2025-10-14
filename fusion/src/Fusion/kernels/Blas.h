#ifndef _BLAS_H
#define _BLAS_H

#include <cblas.h>
#include <vector>

template <typename T> class Tensor;

namespace blas_ops {

template<typename T>
void matmul(Tensor<T> const &v1, std::vector<size_t> const &shapeA,
            Tensor<T> const &v2, std::vector<size_t> const &shapeB,
            std::vector<T> &res) {
  const size_t rankA = shapeA.size();
  const size_t rankB = shapeB.size();
  int m = int(shapeA[rankA - 2]);
  int k = int(shapeA[rankA - 1]);
  int n = int(shapeB[rankB - 1]);

  size_t batch = 1;
  for (size_t i = 0; i < rankA - 2; ++i) {
    batch *= shapeA[i];
  }

  const T alpha = 1.0;
  const T beta = 0.0;

  const T* baseA = v1.raw_data().template data_as<const T>();
  const T* baseB = v2.raw_data().template data_as<const T>();
  T* baseC = res.data();


  for (size_t b = 0; b < batch; ++b) {
    const T *A = baseA + b * (size_t(m) * k);
    const T *B = baseB + b * (size_t(k) * n);
    float *C = baseC + b * (size_t(m) * n);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k,
                B, n, beta, C, n);
  }
}

} // namespace blas_ops

#endif //_BLAS_H
