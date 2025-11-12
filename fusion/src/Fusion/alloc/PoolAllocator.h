#ifndef POOL_ALLOCATOR_H
#define POOL_ALLOCATOR_H

#include <cstddef>
#include "Pool.h"
struct info {
  std::size_t index{0};
  std::size_t block_count{0};
  std::size_t waste{0};

  bool operator<(const info &other) const noexcept {
     return (waste == other.waste) ? block_count < other.block_count : waste < other.waste;
  }
};

class PoolAllocator {
   public:
     PoolAllocator() = default;
     ~PoolAllocator() = default;

     template<std::size_t id, std::size_t... Idx>
     auto& get_pool_instance(std::index_sequence<Idx...>) noexcept {
        static pool_type<id> instance{{{get_size<id, Idx>::value, get_count<id, Idx>::value} ...}};
        return instance;
     }

     template<std::size_t id>
     auto& get_pool_instance() noexcept {
        return get_pool_instance<id>(std::make_index_sequence<bucket_count<id>>());
     }

     template<std::size_t id>
     [[nodiscard]] void *allocate(std::size_t bytes) {
        auto& pool = get_pool_instance<id>();
        std::array<info, bucket_count<id>> deltas;
        std::size_t index = 0;
        for (const auto& bucket: pool) {
           deltas[index].index = index;
           if (bucket.block_size() >= bytes) {
           deltas[index].waste = bucket.block_size() - bytes;
           deltas[index].block_count = 1;
          } else {
             const auto n = 1 + ((bytes - 1) / bucket.block_size());
             const auto required_storage = n * bucket.block_size();
             deltas[index].waste = required_storage - bytes;
             deltas[index].block_count = n;
          }
          ++index;
        }
        std::sort(deltas.begin(), deltas.end()); // TODO: Change from std::sort as it can allocate

        for (const auto &d: deltas) {
           if (auto ptr = pool[d.index].allocate(bytes); ptr != nullptr) {
              return ptr;
           }
        }
        throw std::bad_alloc();
     }

};


#endif // POOL_ALLOCATOR_H
