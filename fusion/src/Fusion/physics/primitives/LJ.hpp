#ifndef FUSION_PHYSICS_PRIMITIVES_LJ
#define FUSION_PHYSICS_PRIMITIVES_LJ

#include <cmath>

#include "Fusion/physics/core/State.hpp"

template <typename T>
struct LJParams {
   T epsilon;
   T sigma;
};

struct PairList {
   std::vector<uint32_t> i;
   std::vector<uint32_t> j;
};

template <typename T>
T displacement(T* xi, T* xj) {
   T dx = xi[0] - xj[0];
   T dy = xi[1] - xj[1];
   T dz = xi[2] - xj[2];
   T dist = dx + dy + dz;
   std::cout << "displacement: " << dist << std::endl;
   return dist;
}

template <typename T>
T pairwise_dist(T* xi, T* xj) {
   T dis = displacement(xi, xj);
   T sqdist = std::pow(dis, 2);
   T sqrtdist = std::sqrt(sqdist);
   std::cout << "pariwise dist = " << sqrtdist << std::endl;
   return sqrtdist;
}

template <typename T>
T lennard_jones(ParticlesSoA<T>& psoa, PairList& pairs, LJParams<T>& params) {
   T energy = 0;
   for (int k = 0; k < pairs.i.size(); ++k) {
      T ep_mul = 4 * params.epsilon;
      T* neigh_i = psoa.x3(pairs.i[k]);
      T* neigh_j = psoa.x3(pairs.j[k]);
      T pair_dist =  pairwise_dist(neigh_i, neigh_j);
      T r_12 = std::pow((params.sigma / pair_dist), 12);
      T r_6 = std::pow((params.sigma / pair_dist), 6);
      energy += ep_mul * (r_12 - r_6);
   }
   return energy;
}

#endif // FUSION_PHYSICS_PRIMITIVES_LJ
