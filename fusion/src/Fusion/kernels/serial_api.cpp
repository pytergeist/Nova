#include <vector>

namespace serial_ops {
void transpose(std::vector<double> &v1, std::vector<size_t> &shape,
               std::vector<double> &res) {
  for (std::size_t i = 0; i < shape[0]; i++) {
    for (std::size_t j = 0; j < shape[1]; j++) {
      res[j * shape[0] + i] = v1[i * shape[1] + j];
    }
  }
}

} // namespace serial_ops
