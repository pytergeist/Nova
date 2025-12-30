#ifndef BROADCAST_ITER_HPP
#define BROADCAST_ITER_HPP

#include <cstdint>
#include <functional>
#include <vector>

#include "Broadcast.h"

template <typename Fn>
void for_each(const BroadcastPlan &plan,
              const std::vector<uint8_t *> &base_ptrs, Fn &&fn) {
   const auto &loop = plan.loop;
   const int ndim = static_cast<int>(loop.size());

   std::vector<uint8_t *> ptr = base_ptrs;

   std::function<void(int)> walk = [&](int dim) {
      if (dim == ndim) {
         fn(ptr);
         return;
      }
      const auto &ld = loop[dim];
      for (int64_t i = 0; i < ld.size; ++i) {
         walk(dim + 1);
         for (int op = 0; op < plan.num_operands; ++op) {
            ptr[op] += ld.stride_bytes[op];
         }
      }
      for (int op = 0; op < plan.num_operands; ++op) {
         ptr[op] -= ld.stride_bytes[op] * ld.size;
      }
   };

   walk(0);
}

inline std::vector<std::int64_t>
contig_elem_strides(const std::vector<std::size_t> &shape) {
   std::vector<std::int64_t> st(shape.size());
   std::int64_t r = 1;
   for (int i = (int)shape.size() - 1; i >= 0; --i) {
      st[i] = r;
      r *= static_cast<std::int64_t>(shape[i]);
   }
   return st;
}

template <typename T>
inline TensorDescription make_desc(const std::vector<std::size_t> &shape,
                                   const int64_t *strides_elems) {
   // Create TensorDescription with ndims (shape.size()), int64_t vector of
   // sizes (shape), strides is stride_elems is not a nullptr, and itemsize
   std::vector<std::size_t> sz(shape.begin(), shape.end());
   std::vector<std::int64_t> st;
   if (strides_elems) {
      st.assign(strides_elems,
                strides_elems + static_cast<int64_t>(shape.size()));
   } else {
      st = contig_elem_strides(shape);
   }
   return TensorDescription{static_cast<std::size_t>(shape.size()),
                            std::move(sz), std::move(st), sizeof(T)};
}

#endif // BROADCAST_ITER_HPP
