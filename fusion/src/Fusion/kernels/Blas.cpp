#include <cblas.h>
#include <vector>
#include "../storage/TensorBuffer.h"

namespace blas_ops {

template<typename T>
void matmul(TensorBuffer const &v1, std::vector<size_t> const &shapeA,
            TensorBuffer const &v2, std::vector<size_t> const &shapeB,
            std::vector<float> &res) {
  const size_t rankA = shapeA.size();
  const size_t rankB = shapeB.size();
  int m = int(shapeA[rankA - 2]);
  int k = int(shapeA[rankA - 1]);
  int n = int(shapeB[rankB - 1]);

  size_t batch = 1;
  for (size_t i = 0; i < rankA - 2; ++i) {
    batch *= shapeA[i];
  }

  const float alpha = 1.0;
  const float beta = 0.0;

  for (size_t b = 0; b < batch; ++b) {
    const float *A = v1.data<T>() + b * (size_t(m) * k);
    const float *B = v2.data<T>() + b * (size_t(k) * n);
    float *C = res.data() + b * (size_t(m) * n);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k,
                B, n, beta, C, n);
  }
}

} // namespace blas_ops
