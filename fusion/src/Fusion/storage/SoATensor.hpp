#ifndef FUSION_STORAGE_SOA_TENSOR_HPP
#define FUSION_STORAGE_SOA_TENSOR_HPP

#include <cstddef>
#include <span>

#include "Fusion/core/RawTensor.hpp"

template <typename T> class SoATensor {
 public:
   SoATensor(const std::uint64_t n_items, std::size_t dim, std::size_t tile,
             DType dtype, Device device)
       : storage_({dim, blocks_for(n_items, tile), tile}, dtype, device),
         n_items_(n_items), dim_(dim), tile_(tile),
         n_blocks_(blocks_for(n_items, tile)) {
      FUSION_CHECK(dim_ > 0, "SoATensor: dim must be > 0");
   }

   std::uint64_t n_items() const noexcept { return n_items_; }
   std::size_t dim() const noexcept { return dim_; }
   std::size_t tile() const noexcept { return tile_; }
   std::size_t n_blocks() const noexcept { return n_blocks_; }

   RawTensor<T> &raw() noexcept { return storage_; }
   const RawTensor<T> &raw() const noexcept { return storage_; }

   std::vector<std::size_t> storage_shape() const {
      return {dim_, n_blocks_, tile_};
   }

   std::vector<std::size_t> logical_shape() const {
      return {dim_, static_cast<std::size_t>(n_items_)};
   }

   T *block_ptr(std::size_t c, std::size_t b) {
      check_component_block(c, b);
      return storage_.get_ptr() + offset(c, b, 0);
   }

   const T *block_ptr(std::size_t c, std::size_t b) const {
      check_component_block(c, b);
      return storage_.get_ptr() + offset(c, b, 0);
   }

   std::span<T> block_span(std::size_t c, std::size_t b) {
      check_component_block(c, b);
      return std::span<T>(storage_.get_ptr() + offset(c, b, 0),
                          valid_in_block(b));
   }

   std::span<const T> block_span(std::size_t c, std::size_t b) const {
      check_component_block(c, b);
      return std::span<const T>(storage_.get_ptr() + offset(c, b, 0),
                                valid_in_block(b));
   }

   T &at(std::size_t c, std::uint64_t p) {
      check_component_item(c, p);
      const auto [b, l] = block_lane(p);
      return storage_.get_ptr()[offset(c, b, l)];
   }

   const T &at(std::size_t c, std::uint64_t p) const {
      check_component_item(c, p);
      const auto [b, l] = block_lane(p);
      return storage_.get_ptr()[offset(c, b, l)];
   }

   std::size_t valid_in_block(std::size_t b) const {
      FUSION_CHECK(b < n_blocks_, "SoATensor: block OOB");
      const std::size_t start = b * tile_;
      const std::size_t n = static_cast<std::size_t>(n_items_);
      if (start >= n)
         return 0;
      return std::min(tile_, n - start);
   }

   void assign_component_major(std::span<const T> data) {
      FUSION_CHECK(data.size() == dim_ * n_items_,
                   "SoATensor::assign_component_major: size mismatch");
      for (std::size_t c = 0; c < dim_; ++c) {
         for (std::uint64_t p = 0; p < n_items_; ++p) {
            at(c, p) = data[c * n_items_ + p];
         }
      }
   }

 private:
   /// Currently we are using RawTensor as the storage object,
   /// this allows us to reuse existing storage semantics. We store the
   /// data as a flat buffer with shape {DIM, Blocks, Tile} and use indexing
   /// schemes (e.g. CRS) to retrieve the data based on defined topology.
   RawTensor<T> storage_;
   std::uint64_t n_items_{0};
   std::size_t dim_{0};
   std::size_t tile_{0};
   std::size_t n_blocks_{0};

   static std::size_t blocks_for(std::uint64_t n_items, std::size_t tile) {
      FUSION_CHECK(tile > 0, "SoATensor: tile must be > 0");
      const std::size_t n = static_cast<std::size_t>(n_items);
      return (n + tile - 1) / tile;
   }

   std::size_t offset(std::size_t c, std::size_t b, std::size_t lane) const {
      return ((c * n_blocks_ + b) * tile_) + lane;
   }

   std::pair<std::size_t, std::size_t> block_lane(std::uint64_t p) const {
      return {static_cast<std::size_t>(p / tile_),
              static_cast<std::size_t>(p % tile_)};
   }

   void check_component_block(std::size_t c, std::size_t b) const {
      FUSION_CHECK(c < dim_, "SoATensor: component OOB");
      FUSION_CHECK(b < n_blocks_, "SoATensor: block OOB");
   }

   void check_component_item(std::size_t c, std::uint64_t p) const {
      FUSION_CHECK(c < dim_, "SoATensor: component OOB");
      FUSION_CHECK(p < n_items_, "SoATensor: item OOB");
   }
};

#endif // FUSION_STORAGE_SOA_TENSOR_HPP