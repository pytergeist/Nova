#ifndef FUSION_PHYSICS_VEC3_GATHER_SUB
#define FUSION_PHYSICS_VEC3_GATHER_SUB

#include "Fusion/physics/core/State.hpp"

template <typename T>
void vec3_rowwise_crs(const Vec3Ptrs<T> &pos, const CRS &crs, T *out,
                      std::uint64_t E) {
   T *out_x = out + 0 * E;
   T *out_y = out + 1 * E;
   T *out_z = out + 2 * E;

   std::uint64_t e_global = 0;

   for (std::uint32_t i = 0; i < crs.N; ++i) {
      const T xi = pos.x[i];
      const T yi = pos.y[i];
      const T zi = pos.z[i];

      const auto start = crs.row_ptr[i];
      const auto end = crs.row_ptr[i + 1];

      for (auto p = start; p < end; ++p, ++e_global) {
         const auto j = crs.col_idx[p];

         out_x[e_global] = xi - pos.x[j];
         out_y[e_global] = yi - pos.y[j];
         out_z[e_global] = zi - pos.z[j];
      }
   }
}

template <typename T, class ParticleT>
void block_rowwise_crs(const ParticleT &pos, const CRS &crs, T *out,
                       std::uint64_t E) {
   T *out_x = out + 0 * E;
   T *out_y = out + 1 * E;
   T *out_z = out + 2 * E;

   for (std::uint32_t i = 0; i < crs.N; ++i) {
      const T xi = pos.x_at(0, i);
      const T yi = pos.x_at(1, i);
      const T zi = pos.x_at(2, i);

      const auto start = crs.row_ptr[i];
      const auto end = crs.row_ptr[i + 1];

      for (auto e = start; e < end; ++e) {
         const auto j = crs.col_idx[e];

         out_x[e] = xi - pos.x_at(0, j);
         out_y[e] = yi - pos.x_at(1, j);
         out_z[e] = zi - pos.x_at(2, j);
      }
   }
}

#endif // FUSION_PHYSICS_VEC3_GATHER_SUB