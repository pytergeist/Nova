#ifndef FUSION_STORAGE_SOA_TENSOR_HPP
#define FUSION_STORAGE_SOA_TENSOR_HPP

#include <cstddef>
#include <span>

#include "DenseTensor.hpp"

template <typename T> class SoATensor {
 public:
   SoATensor(const std::uint64_t n_items, std::size_t dim, DType dtype,
             Device device)
       : storage_({dim, check_items_size(n_items)}, dtype, device),
         n_items_(n_items), dim_(dim) {
      FUSION_CHECK(dim_ > 0, "SoATensor: dim must be > 0");
   }

   explicit SoATensor(DenseTensor<T> x)
       : storage_({x.shape()[0], check_items_size(x.shape()[1])}, x.dtype(),
                  x.device()),
         n_items_(x.shape()[1]), dim_(x.shape()[0]) {
      FUSION_CHECK(dim_ > 0, "SoATensor: dim must be > 0");
   }

   std::uint64_t n_items() const noexcept { return n_items_; }
   std::size_t dim() const noexcept { return dim_; }

   DenseTensor<T> &base() noexcept { return storage_; }
   const DenseTensor<T> &base() const noexcept { return storage_; }

   std::vector<std::size_t> storage_shape() const {
      return {dim_, static_cast<std::size_t>(n_items_)};
   }

   // TODO: do we need to clear anything but data here?
   void clear() noexcept { storage_.clear(); }

   std::vector<std::size_t> logical_shape() const { return storage_shape(); }

   T *block_ptr(const std::size_t c) {
      check_component(c);
      return storage_.get_ptr() + offset(c, 0);
   }

   const T *component_ptr(const std::size_t c) const {
      check_component(c);
      return storage_.get_ptr() + offset(c, 0);
   }

   std::span<T> component_span(const std::size_t c) {
      check_component(c);
      return std::span<T>(storage_.get_ptr() + offset(c, 0));
   }

   std::span<const T> component_span(const std::size_t c) const {
      check_component(c);
      return std::span<const T>(storage_.get_ptr() + offset(c, 0));
   }

   T &at(const std::size_t c, const std::uint64_t p) {
      check_component_item(c, p);
      return storage_.get_ptr()[offset(c, p)];
   }

   const T &at(std::size_t c, std::uint64_t p) const {
      check_component_item(c, p);
      return storage_.get_ptr()[offset(c, p)];
   }

   void assign_component_major(std::span<const T> data) {
      FUSION_CHECK(data.size() == dim_ * n_items_,
                   "SoATensor::assign_component_major: size mismatch");
      for (std::size_t c = 0; c < dim_; ++c) {
         for (std::uint64_t p = 0; p < n_items_; ++p) {
            at(c, p) = data[(c * n_items_) + p];
         }
      }
   }

 private:
   /// Currently we are using RawTensor as the storage object,
   /// this allows us to reuse existing storage semantics. We store the
   /// data as a flat buffer with shape {DIM Tile} and use indexing
   /// schemes (e.g. CRS) to retrieve the data based on defined topology.
   DenseTensor<T> storage_;
   std::uint64_t n_items_{0};
   std::size_t dim_{0};

   std::size_t offset(const std::size_t c, const std::size_t lane) const {
      return (c * n_items_) + lane;
   }

   static std::size_t check_items_size(std::uint64_t x) {
      FUSION_CHECK(x <= static_cast<std::uint64_t>(
                            std::numeric_limits<std::size_t>::max()),
                   "SoATensor: size overflow");
      return static_cast<std::size_t>(x);
   }

   void check_component(const std::size_t c) const {
      FUSION_CHECK(c < dim_, "SoATensor: component OOB");
   }

   void check_component_item(const std::size_t c, const std::uint64_t p) const {
      FUSION_CHECK(c < dim_, "SoATensor: component OOB");
      FUSION_CHECK(p < n_items_, "SoATensor: item OOB");
   }
};

#endif // FUSION_STORAGE_SOA_TENSOR_HPP