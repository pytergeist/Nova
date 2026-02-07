#ifndef FUSION_PHYSICS_STATE
#define FUSION_PHYSICS_STATE

#include "Fusion/core/RawTensor.hpp"

/*
 * This file defines
 *
 *
 */

template <typename T>
struct Vec3Ptrs {
  const T* x;
  const T* y;
  const T* z;
};

struct EdgeList {
   std::vector<uint32_t> i;
   std::vector<uint32_t> j;

   // TODO: you're using just i here, need to set invariants of edge list,
   // i.e. assert (i.size() == j.size())
   // no overlapping edge indices, e.g. assert (i, j).size() == set(i, j)
   std::size_t size() const { return i.size(); }
};


// TODO: make this type a physics DType concept (should only be float, double, maybe half)
template <typename T> struct ParticlesSoA {
   RawTensor<T> x, v, f, m; // position, velocity, force, mass

   static ParticlesSoA<T> from_raw_tensor(const RawTensor<T> &p,
                                          const RawTensor<T> &v,
                                          const RawTensor<T> &f,
                                          const RawTensor<T> &m) {
      ParticlesSoA<T> soa{p, v, f, m};

//      soa.validate(); // TODO: validate shapes
      return soa;
   }

  Vec3Ptrs<T> vec3() const {
    const T* base = x.get_ptr();
    int n = N();
    return { base + 0*n, base + 1*n, base + 2*n };
  }

   void validate() const;
   T N() const { return static_cast<int>(x.shape()[1]); };
};

#endif // FUSION_PHYSICS_STATE
