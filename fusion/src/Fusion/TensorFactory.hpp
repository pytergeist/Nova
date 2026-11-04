#ifndef TENSOR_FACTORY_H
#define TENSOR_FACTORY_H

#include <algorithm>
#include <cstddef>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Fusion/Tensor.h"
#include "Fusion/common/Checks.hpp"
#include "Fusion/core/tensor/AoSoATensor.hpp"
#include "Fusion/core/tensor/DenseTensor.hpp"
#include "Fusion/core/tensor/SoATensor.hpp"
#include "Fusion/core/tensor/Tensor.hpp"
#include "Fusion/core/device/Device.h"
#include "core/dtype/Dtype.h"

namespace fusion::factory {

template <typename T>
std::size_t numel_from_shape(const std::vector<std::size_t> &shape) {
   FUSION_CHECK(!shape.empty(), "numel_from_shape: empty shape");

   return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                          std::multiplies<std::size_t>());
}

// -----------------------------------------------------------------------------
// Dense RawTensor leaf factories
//
// These are intentionally named *_raw because RawTensor is now the dense
// physical tensor leaf, not the public/runtime tensor router.
// -----------------------------------------------------------------------------

template <typename T>
DenseTensor<T> dense_fill_raw(const std::vector<std::size_t> &shape, T value,
                              Device device, DType dtype = DType::FLOAT32,
                              IAllocator *allocator = nullptr) {
   const std::size_t n = numel_from_shape<T>(shape);
   std::vector<T> data(n, value);
   return DenseTensor<T>(shape, std::move(data), dtype, device, allocator);
}

template <typename T>
DenseTensor<T> dense_zeros_raw(const std::vector<std::size_t> &shape,
                               Device device, DType dtype = DType::FLOAT32,
                               IAllocator *allocator = nullptr) {
   return dense_fill_raw<T>(shape, T(0), device, dtype, allocator);
}

template <typename T>
DenseTensor<T> dense_ones_raw(const std::vector<std::size_t> &shape,
                              Device device, DType dtype = DType::FLOAT32,
                              IAllocator *allocator = nullptr) {
   return dense_fill_raw<T>(shape, T(1), device, dtype, allocator);
}

template <typename T>
DenseTensor<T> dense_empty_raw(const std::vector<std::size_t> &shape,
                               Device device, DType dtype = DType::FLOAT32,
                               IAllocator *allocator = nullptr) {
   return DenseTensor<T>(shape, dtype, device, allocator);
}

// -----------------------------------------------------------------------------
// Public/runtime Tensor router factories
// -----------------------------------------------------------------------------

template <typename T>
Tensor<T> fill(const std::vector<std::size_t> &shape, T value, Device device,
               DType dtype = DType::FLOAT32, IAllocator *allocator = nullptr) {
   return Tensor<T>::from_dense(
       dense_fill_raw<T>(shape, value, device, dtype, allocator));
}

template <typename T>
Tensor<T> zeros(const std::vector<std::size_t> &shape, Device device,
                DType dtype = DType::FLOAT32, IAllocator *allocator = nullptr) {
   return fill<T>(shape, T(0), device, dtype, allocator);
}

template <typename T>
Tensor<T> ones(const std::vector<std::size_t> &shape, Device device,
               DType dtype = DType::FLOAT32, IAllocator *allocator = nullptr) {
   return fill<T>(shape, T(1), device, dtype, allocator);
}

template <typename T>
Tensor<T> empty(const std::vector<std::size_t> &shape, Device device,
                DType dtype = DType::FLOAT32, IAllocator *allocator = nullptr) {
   return Tensor<T>::from_dense(
       dense_empty_raw<T>(shape, device, dtype, allocator));
}

template <typename T>
Tensor<T> from_vector(const std::vector<std::size_t> &shape,
                      std::vector<T> data, Device device,
                      DType dtype = DType::FLOAT32,
                      IAllocator *allocator = nullptr) {
   const std::size_t expected = numel_from_shape<T>(shape);
   FUSION_CHECK(data.size() == expected,
                "from_vector: data.size() != product(shape)");

   return Tensor<T>::from_dense(
       DenseTensor<T>(shape, std::move(data), dtype, device, allocator));
}

// -----------------------------------------------------------------------------
// SoA / AoSoA router factories
// -----------------------------------------------------------------------------

template <typename T>
Tensor<T> empty_soa(std::uint64_t n_items, std::size_t dim, Device device,
                    DType dtype = DType::FLOAT32,
                    IAllocator *allocator = nullptr) {
   // allocator currently only flows through RawTensor if your SoATensor
   // constructor supports it. If not, remove allocator from this call.
   return Tensor<T>::from_soa(SoATensor<T>(n_items, dim, dtype, device));
}

template <typename T>
Tensor<T> empty_aosoa(std::uint64_t n_items, std::size_t dim, std::size_t tile,
                      Device device, DType dtype = DType::FLOAT32,
                      IAllocator *allocator = nullptr) {
   // allocator currently only flows through RawTensor if your AoSoATensor
   // constructor supports it. If not, remove allocator from this call.
   return Tensor<T>::from_aosoa(
       AoSoATensor<T>(n_items, dim, tile, dtype, device));
}

// -----------------------------------------------------------------------------
// zeros_like / ones_like preserving router layout
// -----------------------------------------------------------------------------

template <typename T> Tensor<T> zeros_like(const Tensor<T> &other) {
   if (other.is_dense()) {
      return zeros<T>(other.dense().shape(), other.dense().device(),
                      other.dense().dtype());
   }

   if (other.is_soa()) {
      const SoATensor<T> &x = other.soa();

      SoATensor<T> out(x.n_items(), x.dim(), x.base().dtype(),
                       x.base().device());
      out.base().clear();

      return Tensor<T>::from_soa(std::move(out));
   }

   if (other.is_aosoa()) {
      const AoSoATensor<T> &x = other.aosoa();

      AoSoATensor<T> out(x.n_items(), x.dim(), x.tile(), x.base().dtype(),
                         x.base().device());
      out.base().clear();

      return Tensor<T>::from_aosoa(std::move(out));
   }

   throw std::runtime_error("zeros_like: unsupported Tensor layout");
}

template <typename T> Tensor<T> ones_like(const Tensor<T> &other) {
   Tensor<T> out = zeros_like(other);

   std::fill(out.begin(), out.end(), T(1));

   return out;
}

// -----------------------------------------------------------------------------
// Explicit dense/raw compatibility helpers
// -----------------------------------------------------------------------------

template <typename T>
DenseTensor<T> zeros_like_raw(const DenseTensor<T> &other) {
   return dense_zeros_raw<T>(other.shape(), other.device(), other.dtype());
}

template <typename T>
DenseTensor<T> ones_like_raw(const DenseTensor<T> &other) {
   return dense_ones_raw<T>(other.shape(), other.device(), other.dtype());
}

// -----------------------------------------------------------------------------
// ADTensor factories
// -----------------------------------------------------------------------------

template <typename T> ADTensor<T> ad_zeros_like(const ADTensor<T> &other) {
   Tensor<T> value = zeros_like(other.base());
   return ADTensor<T>(std::move(value), other.requires_grad());
}

template <typename T> ADTensor<T> ad_ones_like(const ADTensor<T> &other) {
   Tensor<T> value = ones_like(other.base());
   return ADTensor<T>(std::move(value), other.requires_grad());
}

template <typename T>
ADTensor<T> ad_zeros(const std::vector<std::size_t> &shape, Device device,
                     bool requires_grad = false, DType dtype = DType::FLOAT32,
                     IAllocator *allocator = nullptr) {
   Tensor<T> value = zeros<T>(shape, device, dtype, allocator);
   return ADTensor<T>(std::move(value), requires_grad);
}

template <typename T>
ADTensor<T> ad_ones(const std::vector<std::size_t> &shape, Device device,
                    bool requires_grad = false, DType dtype = DType::FLOAT32,
                    IAllocator *allocator = nullptr) {
   Tensor<T> value = ones<T>(shape, device, dtype, allocator);
   return ADTensor<T>(std::move(value), requires_grad);
}

template <typename T>
ADTensor<T> ad_empty(const std::vector<std::size_t> &shape, Device device,
                     bool requires_grad = false, DType dtype = DType::FLOAT32,
                     IAllocator *allocator = nullptr) {
   Tensor<T> value = empty<T>(shape, device, dtype, allocator);
   return ADTensor<T>(std::move(value), requires_grad);
}

template <typename T>
ADTensor<T>
ad_from_vector(const std::vector<std::size_t> &shape, std::vector<T> data,
               Device device, bool requires_grad = false,
               DType dtype = DType::FLOAT32, IAllocator *allocator = nullptr) {
   Tensor<T> value =
       from_vector<T>(shape, std::move(data), device, dtype, allocator);
   return ADTensor<T>(std::move(value), requires_grad);
}

template <typename T>
ADTensor<T> ad_empty_soa(std::uint64_t n_items, std::size_t dim, Device device,
                         bool requires_grad = false,
                         DType dtype = DType::FLOAT32,
                         IAllocator *allocator = nullptr) {
   Tensor<T> value = empty_soa<T>(n_items, dim, device, dtype, allocator);
   return ADTensor<T>(std::move(value), requires_grad);
}

template <typename T>
ADTensor<T>
ad_empty_aosoa(std::uint64_t n_items, std::size_t dim, std::size_t tile,
               Device device, bool requires_grad = false,
               DType dtype = DType::FLOAT32, IAllocator *allocator = nullptr) {
   Tensor<T> value =
       empty_aosoa<T>(n_items, dim, tile, device, dtype, allocator);
   return ADTensor<T>(std::move(value), requires_grad);
}

} // namespace fusion::factory

// -----------------------------------------------------------------------------
// Temporary global compatibility aliases
//
// Keep these only if the rest of your code currently calls zeros<T>(...)
// unqualified. Once the refactor settles, prefer fusion::factory::zeros<T>(...)
// explicitly.
// -----------------------------------------------------------------------------

using fusion::factory::ad_empty;
using fusion::factory::ad_empty_aosoa;
using fusion::factory::ad_empty_soa;
using fusion::factory::ad_from_vector;
using fusion::factory::ad_ones;
using fusion::factory::ad_ones_like;
using fusion::factory::ad_zeros;
using fusion::factory::ad_zeros_like;
using fusion::factory::dense_empty_raw;
using fusion::factory::dense_fill_raw;
using fusion::factory::dense_ones_raw;
using fusion::factory::dense_zeros_raw;
using fusion::factory::empty;
using fusion::factory::empty_aosoa;
using fusion::factory::empty_soa;
using fusion::factory::fill;
using fusion::factory::from_vector;
using fusion::factory::ones;
using fusion::factory::ones_like;
using fusion::factory::ones_like_raw;
using fusion::factory::zeros;
using fusion::factory::zeros_like;
using fusion::factory::zeros_like_raw;

#endif // TENSOR_FACTORY_H