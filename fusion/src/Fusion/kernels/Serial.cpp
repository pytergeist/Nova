#include <cmath>
#include <numeric>
#include <vector>

namespace serial_ops {
void transpose(std::vector<float> &v1, std::vector<size_t> &shape,
               std::vector<float> &res) {
  for (std::size_t i = 0; i < shape[0]; i++) {
    for (std::size_t j = 0; j < shape[1]; j++) {
      res[j * shape[0] + i] = v1[i * shape[1] + j];
    }
  }
}

std::vector<size_t> get_contiguous_strides(std::vector<size_t> &shape) {
  size_t product = 1;
  size_t ndim = shape.size();
  std::vector<size_t> strides;
  strides.resize(ndim);
  for (size_t i = ndim; i-- > 0;) {
    strides[i] = product;
    const size_t curr_dim = shape[i];
    product *= curr_dim;
  }
  return strides;
}

size_t normalise_axis(int axis, size_t ndim_sz) {
  const int ndim = static_cast<int>(ndim_sz);
  if (axis < -ndim || axis >= ndim) {
    throw std::runtime_error("input axis out of range");
  }
  if (axis < 0) {
    return static_cast<size_t>(ndim + axis); // (signed add) then cast
  }
  return static_cast<size_t>(axis);
}
std::vector<size_t> linear_to_coord(size_t idx, size_t stride1, size_t stride2,
                                    size_t dim1, size_t dim2) {
  // Convert linear coordinate to 2d matrix coordinate based on
  // strides, dims and linear index:
  // i_k = [L//s_k] % N_k
  // for L = 7, s1 = 4, s2 = 1, dim1 = 3, dim2 = 4
  // i_0 = [7//4] % 3 = 1
  // i_1 = [7//1] % 4 = 3
  size_t c0 = static_cast<size_t>(std::floor(idx / stride1)) % dim1;
  size_t c1 = static_cast<size_t>(std::floor(idx / stride2)) % dim2;
  std::vector<size_t> dst{c0, c1};
  return dst;
}

std::vector<size_t> unravel_idx(size_t idx, std::vector<size_t> &strides,
                                std::vector<size_t> &shape) {
  const size_t n = shape.size();
  std::vector<size_t> coord(n);

  for (size_t k = 0; k < n; k++) {
    coord[k] = static_cast<size_t>(std::floor(idx / strides[k])) % shape[k];
  }
  return coord;
}

size_t ravel_idx(const std::vector<size_t> &coord,
                 const std::vector<size_t> &strides) {
  size_t idx = 0;
  for (size_t k = 0; k < coord.size(); k++)
    idx += coord[k] * strides[k];
  return idx;
}

size_t coord_to_linear(std::vector<size_t> &coords,
                       std::vector<size_t> &strides, size_t axis1,
                       size_t axis2) {
  // Convert 2d matrix coordinate to linear coordinate based on
  // strides, and cords:
  // L = i * s_i + j * s_j
  // for c(i, j) = (1, 3) and s(i, j) = (4, 1)
  // L = 1 * 4 + 3 * 1 = 7]
  size_t Lidx = strides[axis1] * coords[axis1] + strides[axis2] * coords[axis2];
  return Lidx;
}

std::vector<float> swapaxes(std::vector<float> &a, std::vector<size_t> &shape,
                            int a1, int a2) {

  int axis1 = normalise_axis(a1, shape.size());
  int axis2 = normalise_axis(a2, shape.size());
  if (axis1 == axis2) {
    return a;
  }

  std::vector<int> result;
  result.resize(shape.size());
  std::vector<size_t> original_strides = get_contiguous_strides(shape);
  std::vector<size_t> new_shape = shape;
  std::swap(new_shape[axis1], new_shape[axis2]);
  std::vector<size_t> new_strides = get_contiguous_strides(new_shape);

  size_t size = std::accumulate(shape.begin(), shape.end(), size_t{1},
                                std::multiplies<int>());
  std::vector<float> out(a.size());
  for (int i = 0; i < size; i++) {
    std::vector<size_t> old_coord = unravel_idx(i, original_strides, shape);
    std::swap(old_coord[axis1], old_coord[axis2]);
    size_t j = ravel_idx(old_coord, new_strides);
    out[j] = a[i];
  };
  return out;
}

std::vector<float> diagonal2D(std::vector<float> &a,
                              std::vector<size_t> &shape) {
  // Row-major: flat index (r, c) is r*cols + c for diag
  if (shape.size() != 2) {
    throw std::runtime_error("Diagonal2D can only be called with a 2D array");
  }
  const int64_t rows = shape[0];
  const int64_t cols = shape[1];

  const size_t expected = rows * cols;
  if (a.size() != expected) {
    throw std::runtime_error("Input data size does not match shape");
  };

  const size_t n = static_cast<size_t>(std::min(rows, cols));
  std::vector<float> out(n);

  for (size_t i = 0; i < n; i++) {
    out[i] = a[i * static_cast<size_t>(cols) + i];
  }
  return out;
}

} // namespace serial_ops
