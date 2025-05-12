#include <cblas.h>
#include <vector>

namespace cblas_ops {
void matmul(std::vector<double> &v1, std::vector<double> &v2,
            std::vector<size_t> const shape, std::vector<double> &res) {
  int m = shape[0];
  int n = shape[1];
  int k = shape[1];
  const double alpha = 1.0;
  const double beta = 0.0;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
              v1.data(), k, v2.data(), n, beta, res.data(), n);
}

} // namespace cblas_ops
