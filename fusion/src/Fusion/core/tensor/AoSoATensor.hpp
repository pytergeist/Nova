#ifndef FUSION_STORAGE_AOSOA_TENSOR_HPP
#define FUSION_STORAGE_AOSOA_TENSOR_HPP

#include <cstddef>
#include <span>

#include "RawTensor.hpp"

template <typename T> class AoSoATensor {
 public:
   AoSoATensor() = default;

   AoSoATensor(const std::uint64_t n_items, std::size_t dim, std::size_t tile,
               DType dtype, Device device)
       : storage_({dim, blocks_for(check_items_size(n_items), tile), tile},
                  dtype, device),
         n_items_(n_items), dim_(dim), tile_(tile),
         n_blocks_(blocks_for(n_items, tile)) {
      FUSION_CHECK(dim_ > 0, "AoSoATensor: dim must be > 0");
   }

   explicit AoSoATensor(RawTensor<T> x, const std::size_t tile)
       : storage_({x.shape()[0], check_items_size(x.shape()[1])}, x.dtype(),
                  x.device()),
         n_items_(x.shape()[1]), dim_(x.shape()[0]), tile_(tile),
         n_blocks_(blocks_for(x.shape()[1], tile)) {
      FUSION_CHECK(dim_ > 0, "AoSoATensor: dim must be > 0");
      FUSION_CHECK(tile_ > 0, "AoSoATensor: tile must be > 0");
      FUSION_CHECK(dim_ <= n_items_, "AoSoATensor: n_items must be <= dim");
   }

   std::uint64_t n_items() const noexcept { return n_items_; }
   std::size_t dim() const noexcept { return dim_; }
   std::size_t tile() const noexcept { return tile_; }
   std::size_t n_blocks() const noexcept { return n_blocks_; }
   DType dtype() const noexcept { return storage_.dtype(); };
   Device device() const noexcept { return storage_.device(); };

   RawTensor<T> &raw() noexcept { return storage_; }
   const RawTensor<T> &raw() const noexcept { return storage_; }

   std::vector<std::size_t> storage_shape() const {
      return {dim_, n_blocks_, tile_};
   }

   std::vector<std::size_t> logical_shape() const {
      return {dim_, static_cast<std::size_t>(n_items_)};
   }

   T *block_ptr(const std::size_t c, const std::size_t b) {
      check_component_block(c, b);
      return storage_.get_ptr() + offset(c, b, 0);
   }

   const T *block_ptr(const std::size_t c, const std::size_t b) const {
      check_component_block(c, b);
      return storage_.get_ptr() + offset(c, b, 0);
   }

   std::span<T> block_span(const std::size_t c, const std::size_t b) {
      check_component_block(c, b);
      return std::span<T>(storage_.get_ptr() + offset(c, b, 0),
                          valid_in_block(b));
   }

   std::span<const T> block_span(const std::size_t c,
                                 const std::size_t b) const {
      check_component_block(c, b);
      return std::span<const T>(storage_.get_ptr() + offset(c, b, 0),
                                valid_in_block(b));
   }

   T &at(const std::size_t c, const std::uint64_t p) {
      check_component_item(c, p);
      const auto [b, l] = block_lane(p);
      return storage_.get_ptr()[offset(c, b, l)];
   }

   const T &at(const std::size_t c, const std::uint64_t p) const {
      check_component_item(c, p);
      const auto [b, l] = block_lane(p);
      return storage_.get_ptr()[offset(c, b, l)];
   }

   std::size_t valid_in_block(const std::size_t b) const {
      FUSION_CHECK(b < n_blocks_, "AoSoATensor: block OOB");
      const std::size_t start = b * tile_;
      const std::size_t n = static_cast<std::size_t>(n_items_);
      if (start >= n)
         return 0;
      return std::min(tile_, n - start);
   }

   void assign_component_major(std::span<const T> data) {
      FUSION_CHECK(data.size() == dim_ * n_items_,
                   "AoSoATensor::assign_component_major: size mismatch");
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
      FUSION_CHECK(tile > 0, "AoSoATensor: tile must be > 0");
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

   static std::size_t check_items_size(std::uint64_t x) {
      FUSION_CHECK(x <= static_cast<std::uint64_t>(
                            std::numeric_limits<std::size_t>::max()),
                   "SoATensor: size overflow");
      return static_cast<std::size_t>(x);
   }

   void check_component_block(std::size_t c, std::size_t b) const {
      FUSION_CHECK(c < dim_, "AoSoATensor: component OOB");
      FUSION_CHECK(b < n_blocks_, "AoSoATensor: block OOB");
   }

   void check_component_item(std::size_t c, std::uint64_t p) const {
      FUSION_CHECK(c < dim_, "AoSoATensor: component OOB");
      FUSION_CHECK(p < n_items_, "AoSoATensor: item OOB");
   }
};

#endif // FUSION_STORAGE_SOA_TENSOR_HPP