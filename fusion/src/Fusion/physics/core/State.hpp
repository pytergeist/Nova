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

template <typename T> struct ParticlesSoA {
   RawTensor<T> p, v, f, m; // pos, velocity, force, mass

   static ParticlesSoA<T> from_raw_tensor(const RawTensor<T> &p,
                                          const RawTensor<T> &v,
                                          const RawTensor<T> &f,
                                          const RawTensor<T> &m) {
      ParticlesSoA<T> soa{p, v, f, m};

//      soa.validate(); // TODO: validate shapes
      return soa;
   }

  Vec3Ptrs<T> vec3() const {
    const T* base = p.get_ptr();
    int n = N();
    return { base + 0*n, base + 1*n, base + 2*n };
  }

   void validate() const;
   T N() const { return static_cast<int>(p.shape()[1]); };
};

#endif // FUSION_PHYSICS_STATE
