#ifndef BROADCAST_ITER_H
#define BROADCAST_ITER_H

#include "broadcast.h"
#include <cstdint>
#include <functional>
#include <vector>

template <typename Fn>
void for_each(const BroadcastPlan &plan,
              const std::vector<uint8_t *> &base_ptrs, Fn &&fn) {
   const auto &loop = plan.loop;
   const int ndim = (int)loop.size();

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

#endif // BROADCAST_ITER_H
