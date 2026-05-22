#ifndef FUSION_PHYSICS_STATE
#define FUSION_PHYSICS_STATE

#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <span>

#include "../../core/tensor/AoSoATensor.hpp"
#include "../../core/tensor/DenseTensor.hpp"
#include "Fusion/core/Dtype.h"
#include "Fusion/device/Device.h"

// TODO: Remove template parameters from here! plans should be runtime objects
// that pass around itemsize, not templated structs

template <typename T> struct Vec3Ptrs {
   const T *x;
   const T *y;
   const T *z;
};

// TODO: make this type a physics DType concept (should only be float, double,
// maybe half)
template <typename T> struct ParticlesSoA {
   DenseTensor<T> x, v, f, m; // position, velocity, force, mass
   std::vector<std::uint32_t>
       type; // This will eventually hold atom type by int

   static ParticlesSoA<T> from_raw_tensor(const DenseTensor<T> &x,
                                          const DenseTensor<T> &v,
                                          const DenseTensor<T> &f,
                                          const DenseTensor<T> &m) {
      ParticlesSoA<T> soa{x, v, f, m};

      //      soa.validate(); // TODO: validate shapes
      return soa;
   }

   Vec3Ptrs<T> vec3() const {
      const T *base = x.get_ptr();
      int n = N();
      return {base + 0 * n, base + 1 * n, base + 2 * n};
   }

   void validate() const;
   std::uint32_t N() const { return static_cast<int>(x.shape()[1]); };
   static constexpr std::size_t dim() { return 3; };
};

template <typename T> class ParticleField {
 public:
   ParticleField(const std::uint64_t n_items, const std::size_t dim,
                 const std::size_t tile, const DType dtype, const Device device)
       : n_items_(check_items_size(n_items)), dim_(dim), tile_(tile),
         dtype_(dtype), device_(device) {};

   explicit ParticleField(AoSoATensor<T> &x)
       : x_(std::move(x)), n_items_(x.n_items()), dim_(x.dim()),
         tile_(x.tile()), n_blocks_(x.n_blocks()), dtype_(x.dtype()),
         device_(x.device()) {};

   AoSoATensor<T> &x() { return x_; };
   std::size_t dim() const { return dim_; };
   std::size_t tile() const { return tile_; };
   std::size_t n_items() const { return n_items_; };
   std::size_t n_blocks() const { return n_blocks_; };

   std::vector<std::size_t> logical_shape() const {
      return {dim_, static_cast<std::size_t>(n_items_)};
   }

   void from_aosoa_tensor(AoSoATensor<T> x) {
      FUSION_CHECK(x_.raw().is_initialiased,
                   "x Tensor already set in ParticleField.");
      x_ = std::move(x);
   }

 private:
   AoSoATensor<T> x_{}, v_{}, f_{}, m_{}, type_{};
   std::uint64_t n_items_{0};
   std::size_t dim_{};
   std::size_t tile_{};
   std::size_t n_blocks_{};
   DType dtype_{};
   Device device_;

   // TODO: This is duplicated in multiple methods (SoATensor and AoSoATensor)
   static std::size_t check_items_size(std::uint64_t x) {
      FUSION_CHECK(x <= static_cast<std::uint64_t>(
                            std::numeric_limits<std::size_t>::max()),
                   "SoATensor: size overflow");
      return static_cast<std::size_t>(x);
   }
};

template <typename T, std::size_t DIM, std::size_t TILE> struct ParticlesAoSoA {
   // TODO: Make this a class - it now has internal layout assumptions
   // and invariants to maintain that shouldn't be mutatable post init
   static_assert(DIM > 0);
   static_assert(TILE > 0);
   std::int64_t N_ = 0; // TODO: this should probably be uint64_t
   std::size_t nBlocks_ = 0;
   DenseTensor<T> x, v, f, m, type;

   static constexpr std::size_t dim() { return DIM; };
   static constexpr std::size_t tile() { return TILE; };
   static std::size_t blocks_for(std::size_t N) {
      return (N + TILE - 1) / TILE;
   }

   static ParticlesAoSoA allocate(std::size_t N) {
      ParticlesAoSoA out;
      out.N_ = N;
      out.nBlocks_ = blocks_for(N);
      DType dtype = DType::FLOAT32;
      Device device = Device{DeviceType::CPU, 0};
      out.x = DenseTensor<T>{{DIM, out.nBlocks_, TILE}, dtype, device};
      out.v = DenseTensor<T>{{DIM, out.nBlocks_, TILE}, dtype, device};
      out.f = DenseTensor<T>{{DIM, out.nBlocks_, TILE}, dtype, device};
      out.m = DenseTensor<T>{{out.nBlocks_, TILE}, dtype, device};
      //      out.type = RawTensor<T>{{out.nBlocks_, TILE}}; // needs to be int
      //      tensor
      return out;
   }

   std::int64_t N() const { return N_; };
   std::size_t nBlocks() { return nBlocks_; };
   std::size_t nBlocks() const { return nBlocks_; };

   std::vector<std::size_t> storage_shape() const {
      return {DIM, nBlocks_, TILE};
   }
   std::vector<std::size_t> logical_shape() const {
      return {DIM, static_cast<std::size_t>(N_)};
   }

   std::vector<std::int64_t> logical_strides() const {
      return {static_cast<std::int64_t>(nBlocks_ * TILE), 1};
   }

   T *x_block_ptr(const std::size_t c, std::size_t b) {
      assert(c < DIM);
      assert(b < nBlocks_);
      return x.get_ptr() + TILE * (c * nBlocks_ + b);
   }

   const T *x_block_ptr(std::size_t c, std::size_t b) const {
      assert(c < DIM);
      assert(b < nBlocks_);
      return x.get_ptr() + TILE * (c * nBlocks_ + b);
   }

   T *v_block_ptr(const std::size_t c, std::size_t b) {
      assert(c < DIM);
      assert(b < nBlocks_);
      return v.get_ptr() + TILE * (c * nBlocks_ + b);
   }

   T *f_block_ptr(const std::size_t c, std::size_t b) {
      assert(c < DIM);
      assert(b < nBlocks_);
      return f.get_ptr() + TILE * (c * nBlocks_ + b);
   }

   T x_at(std::size_t c, std::uint32_t p) const {
      const std::uint32_t b = p / TILE;
      const std::uint32_t l = p % TILE;
      return x.get_ptr()[((c * nBlocks_ + b) * TILE) + l];
   }

   std::span<T> x_block_span(const std::size_t c, std::size_t b) {
      assert(c < DIM);
      assert(b < nBlocks_);
      return std::span<T>(x.get_ptr() + TILE * (c * nBlocks_ + b), TILE);
   }

   std::span<T> v_block_span(const std::size_t c, std::size_t b) {
      assert(c < DIM);
      assert(b < nBlocks_);
      return std::span<T>(v.get_ptr() + TILE * (c * nBlocks_ + b), TILE);
   }

   std::span<T> f_block_span(const std::size_t c, std::size_t b) {
      assert(c < DIM);
      assert(b < nBlocks_);
      return std::span<T>(f.get_ptr() + TILE * (c * nBlocks_ + b), TILE);
   }

   std::size_t valid_in_block(std::size_t b) const {
      const std::size_t start = b * TILE;
      if (start >= static_cast<std::size_t>(N_))
         return 0;
      return std::min<std::size_t>(TILE, static_cast<std::size_t>(N_) - start);
   }

   static ParticlesAoSoA from_three_n_raw_tensor(std::size_t N,
                                                 const DenseTensor<T> &x,
                                                 const DenseTensor<T> &v,
                                                 const DenseTensor<T> &f,
                                                 const DenseTensor<T> &m) {
      ParticlesAoSoA out = ParticlesAoSoA::allocate(N);
      for (std::size_t i = 0; i < N; ++i) {
         const std::uint32_t b = i / TILE;
         const std::uint32_t l = i % TILE;
         for (std::size_t c = 0; c < DIM; ++c) {
            out.x.get_ptr()[TILE * (c * out.nBlocks_ + b) + l] =
                x.get_ptr()[c * N + i];
            out.v.get_ptr()[TILE * (c * out.nBlocks_ + b) + l] =
                v.get_ptr()[c * N + i];
            out.f.get_ptr()[TILE * (c * out.nBlocks_ + b) + l] =
                f.get_ptr()[c * N + i];
         }
         out.m.get_ptr()[b * TILE + l] = m.get_ptr()[i];
      }

      std::size_t rem = N % TILE;
      if (rem != 0) {
         for (std::size_t l = rem; l < TILE; ++l) {
            const std::uint32_t b = out.nBlocks_ - 1;
            for (std::size_t c = 0; c < DIM; ++c) {
               out.x.get_ptr()[TILE * (c * out.nBlocks_ + b) + l] = T(0);
               out.v.get_ptr()[TILE * (c * out.nBlocks_ + b) + l] = T(0);
               out.f.get_ptr()[TILE * (c * out.nBlocks_ + b) + l] = T(0);
            }
            out.m.get_ptr()[b * TILE + l] = T(0);
         }
      }
      return out;
   }
};

template <typename T, std::size_t DIM, std::size_t TILE>
struct ParticlesAoSoAView {
   const T *x_ = nullptr;
   std::int64_t N_ = 0;
   std::size_t nBlocks_ = 0;

   static constexpr std::size_t dim() { return DIM; }
   static constexpr std::size_t tile() { return TILE; }

   std::size_t N() const { return static_cast<std::size_t>(N_); }
   std::size_t nBlocks() const { return nBlocks_; }

   const T *x_block_ptr(const std::size_t c, const std::size_t b) const {
      return x_ + TILE * (c * nBlocks_ + b);
   }

   std::size_t valid_in_block(std::size_t b) const {
      const std::size_t start = b * TILE;
      if (start >= static_cast<std::size_t>(N_))
         return 0;
      return std::min<std::size_t>(TILE, static_cast<std::size_t>(N_) - start);
   }
};

template <typename T> struct ParticleFieldView {
   const T *ten_ = nullptr;

   std::size_t dim_ = 0;
   std::size_t tile_ = 0;
   std::size_t n_blocks_ = 0;
   std::uint64_t n_items_ = 0;

   std::size_t dim() const { return dim_; }
   std::size_t tile() const { return tile_; }

   std::uint64_t n_items() const { return n_items_; }
   std::size_t n_blocks() const { return n_blocks_; }

   const T *block_ptr(const std::size_t c, const std::size_t b) const {
      return ten_ + (tile_ * ((c * n_blocks_) + b));
   }
   // TODO: THIS IS REPLICATED IN TOO MANY PLACES!!
   std::size_t valid_in_block(const std::size_t b) const {
      FUSION_CHECK(b < n_blocks_, "ParticleFieldView: block OOB");

      const std::size_t start = b * tile_;
      if (start >= static_cast<std::size_t>(n_items_)) {
         return 0;
      }
      return std::min<std::size_t>(tile_,
                                   static_cast<std::size_t>(n_items_) - start);
   }
};

template <typename T>
ParticleFieldView<T> make_view_x(ParticleField<T> &p, const AoSoATensor<T> &x) {
   return ParticleFieldView<T>{.ten_ = x.get_ptr(),
                               .dim_ = x.dim(),
                               .tile_ = x.tile(),
                               .n_blocks_ = x.n_blocks(),
                               .n_items_ = x.n_items()};
}

// template <typename T, std::size_t DIM, std::size_t TILE>
// inline ParticlesAoSoAView<T, DIM, TILE>
// make_view_x(const ParticlesAoSoA<T, DIM, TILE> &p, const RawTensor<T> &x) {
//    return ParticlesAoSoAView<T, DIM, TILE>{
//       .x_ = x.get_ptr(), .N_ = p.N_, .nBlocks_ = p.nBlocks_};
// }

#endif // FUSION_PHYSICS_STATE
