#ifndef _SERIAL_H
#define _SERIAL_H

#include <cmath>
#include <numeric>
#include <stddef.h>
#include <stdexcept>
#include <vector>

#include "Fusion/common/Log.h"

template <typename T> class Tensor;

namespace serial {
template <typename T>
void transpose(const Tensor<T> &t1, const std::vector<size_t> &shape,
               std::vector<T> &res) {
   for (std::size_t i = 0; i < shape[0]; i++) {
      for (std::size_t j = 0; j < shape[1]; j++) {
         res[j * shape[0] + i] = t1[i * shape[1] + j];
      }
   }
}

inline std::vector<size_t>
get_contiguous_strides(const std::vector<size_t> &shape) {
   const size_t nd = shape.size();
   std::vector<size_t> strides(nd);
   size_t prod = 1;
   for (size_t i = nd; i-- > 0;) {
      strides[i] = prod;
      prod *= shape[i];
   }
   return strides;
}

inline size_t normalise_axis(int axis, size_t ndim) {
   const int nd = static_cast<int>(ndim);
   if (axis < -nd || axis >= nd)
      throw std::runtime_error("input axis out of range");
   return static_cast<size_t>(axis < 0 ? nd + axis : axis);
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

std::vector<size_t> unravel_idx(size_t idx, const std::vector<size_t> &strides,
                                const std::vector<size_t> &shape) {
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
   size_t Lidx =
       strides[axis1] * coords[axis1] + strides[axis2] * coords[axis2];
   return Lidx;
}

template <typename T>
std::vector<T> swapaxes(const Tensor<T> &a, const std::vector<size_t> &shape,
                        int a1, int a2) {
   const size_t nd = shape.size();

   if (nd < 2) {
      return std::vector<T>(a.begin(), a.end());
   }

   const size_t axis1 = normalise_axis(a1, nd);
   const size_t axis2 = normalise_axis(a2, nd);

   if (axis1 == axis2) {
      return std::vector<T>(a.begin(), a.end());
   }

   const std::vector<size_t> orig_strides = get_contiguous_strides(shape);
   std::vector<size_t> new_shape = shape;
   std::swap(new_shape[axis1], new_shape[axis2]);
   const std::vector<size_t> new_strides = get_contiguous_strides(new_shape);

   const size_t size =
       std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                       std::multiplies<size_t>());

   std::vector<T> out(size);
   for (size_t i = 0; i < size; ++i) {
      std::vector<size_t> coord = unravel_idx(i, orig_strides, shape);
      std::swap(coord[axis1], coord[axis2]);
      const size_t j = ravel_idx(coord, new_strides);
      out[j] = a[i];
   }
   return out;
}

template <typename T>
std::vector<T> diagonal2D(const Tensor<T> &a,
                          const std::vector<size_t> &shape) {
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
   std::vector<T> out(n);

   for (size_t i = 0; i < n; i++) {
      out[i] = a[i * static_cast<size_t>(cols) + i];
   }
   return out;
}

} // namespace serial

#endif // _SERIAL_H
