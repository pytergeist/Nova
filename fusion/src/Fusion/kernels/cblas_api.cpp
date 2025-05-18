#include <cblas.h>
#include <vector>

namespace cblas_ops {
// now take BOTH A–shape and B–shape
void matmul(std::vector<double> const &v1, std::vector<size_t> const &shapeA,
            std::vector<double> const &v2, std::vector<size_t> const &shapeB,
            std::vector<double> &res) {
  int m = int(shapeA[0]);
  int k = int(shapeA[1]);
  int n = int(shapeB[1]);

  const double alpha = 1.0;
  const double beta = 0.0;

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
              v1.data(), k, v2.data(), n, beta, res.data(), n);
}
} // namespace cblas_ops
