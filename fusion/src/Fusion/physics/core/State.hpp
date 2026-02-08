#ifndef FUSION_PHYSICS_STATE
#define FUSION_PHYSICS_STATE

#include <cstddef>
#include <cstdint>

#include "Fusion/core/RawTensor.hpp"

// TODO: Remove template parameters from here! plans should be runtime objects
// that pass around itemsize, not templated structs

template <typename T> struct Vec3Ptrs {
   const T *x;
   const T *y;
   const T *z;
};

struct EdgeList {
   std::vector<std::uint32_t> i;
   std::vector<std::uint32_t> j;

   // TODO: you're using just i here, need to set invariants of edge list,
   // i.e. assert (i.size() == j.size())
   // no overlapping edge indices, e.g. assert (i, j).size() == set(i, j)
   std::size_t size() const { return i.size(); }
};

struct CRS {
   std::int64_t N = 0, E = 0;
   std::vector<std::uint32_t> row_ptr;
   std::vector<std::uint32_t> col_idx;

   std::vector<float> w;
   std::vector<float> r0;
   std::vector<std::uint16_t> type;

   bool sorted_by_j = false;
   bool symmetric = false;
   bool directed = false;
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
   T N() const { return static_cast<int>(x.shape()[1]); };
};

#endif // FUSION_PHYSICS_STATE
