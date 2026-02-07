#ifndef FUSION_PHYSICS_STATE
#define FUSION_PHYSICS_STATE

#include "Fusion/core/RawTensor.hpp"

/*
 * This file defines
 *
 *
 */

template <typename T> struct ParticlesSoA {
   RawTensor<T> p, v, f, m; // pos, velocity, force, mass

   static ParticlesSoA<T> from_raw_tensor(const RawTensor<T> &p,
                                          const RawTensor<T> &v,
                                          const RawTensor<T> &f,
                                          const RawTensor<T> &m) {
      ParticlesSoA<T> soa{p, v, f, m};
//      soa.validate();
      return soa;
   }
   inline T* x3(int i) {return p.get_ptr() + i*3;};
   void validate() const;
};

#endif // FUSION_PHYSICS_STATE
