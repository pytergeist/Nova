#ifndef FUSION_PHYSICS_STATE
#define FUSION_PHYSICS_STATE

#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <span>

#include "Fusion/core/Dtype.h"
#include "Fusion/core/RawTensor.hpp"
#include "Fusion/device/Device.h"

// TODO: Remove template parameters from here! plans should be runtime objects
// that pass around itemsize, not templated structs

template <typename T> struct Vec3Ptrs {
   const T *x;
   const T *y;
   const T *z;
};

// TODO: make sure you have vec3Ptr opt for physics and VecNPtrs for
// GNN pairwise iteration over node features.
template <typename T, std::size_t N> struct VecNPtrs {
   T *components[N];
};


// TODO: make this type a physics DType concept (should only be float, double,
// maybe half)
template <typename T> struct ParticlesSoA {
   RawTensor<T> x, v, f, m; // position, velocity, force, mass
   std::vector<std::uint32_t>
       type; // This will eventually hold atom type by int

   static ParticlesSoA<T> from_raw_tensor(const RawTensor<T> &x,
                                          const RawTensor<T> &v,
                                          const RawTensor<T> &f,
                                          const RawTensor<T> &m) {
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

template <typename T, std::size_t DIM, std::size_t TILE> struct ParticlesAoSoA {
   static_assert(DIM > 0);
   static_assert(TILE > 0);
   std::int64_t N_ = 0;
   std::size_t nBlocks_ = 0;
   RawTensor<T> x, v, f, m, type;

   static constexpr std::size_t dim() { return DIM; };
   static constexpr std::size_t tile() { return TILE; };
   static inline std::size_t blocks_for(std::size_t N) {
      return (N + TILE - 1) / TILE;
   }

   static ParticlesAoSoA allocate(std::size_t N) {
      ParticlesAoSoA out;
      out.N_ = N;
      out.nBlocks_ = blocks_for(N);
      DType dtype = DType::FLOAT32;
      Device device = Device{DeviceType::CPU};
      out.x = RawTensor<T>{{DIM, out.nBlocks_, TILE}, dtype, device};
      out.v = RawTensor<T>{{DIM, out.nBlocks_, TILE}, dtype, device};
      out.f = RawTensor<T>{{DIM, out.nBlocks_, TILE}, dtype, device};
      out.m = RawTensor<T>{{out.nBlocks_, TILE}, dtype, device};
      //      out.type = RawTensor<T>{{out.nBlocks_, TILE}}; // needs to be int
      //      tensor
      return out;
   }

   std::size_t N() const { return N_; };
   std::size_t nBlocks() { return nBlocks_; };
   std::size_t nBlocks() const { return nBlocks_; };

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
                                                 const RawTensor<T> &x,
                                                 const RawTensor<T> &v,
                                                 const RawTensor<T> &f,
                                                 const RawTensor<T> &m) {
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

#endif // FUSION_PHYSICS_STATE
