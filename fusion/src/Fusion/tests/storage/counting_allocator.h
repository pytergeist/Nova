#include <cstddef>
#include <cstdint>
#include <new>
#include <unordered_set>
#include <vector>

#include "Fusion/storage/TensorBuffer.hpp"
#include "Fusion/alloc/AllocatorInterface.h"

class CountingAllocator final : public IAllocator {
public:
   void* allocate(std::size_t size, Alignment alignment) override {
      ++allocate_calls_;
      last_size_ = size;
      last_alignment_ = alignment.value;

      void* ptr = aligned_alloc_bytes(alignment.value, size);
      active_ptrs_.insert(ptr);
      return ptr;
   }

   void deallocate(void* ptr) override {
      ++deallocate_calls_;
      last_deallocated_ptr_ = ptr;

      auto it = active_ptrs_.find(ptr);
      if (it == active_ptrs_.end()) {
         throw std::runtime_error("deallocate called on unknown pointer");
      }

      active_ptrs_.erase(it);
      std::free(ptr);
   }

   std::size_t allocate_calls() const noexcept { return allocate_calls_; }
   std::size_t deallocate_calls() const noexcept { return deallocate_calls_; }
   std::size_t active_allocations() const noexcept { return active_ptrs_.size(); }
   std::size_t last_size() const noexcept { return last_size_; }
   std::size_t last_alignment() const noexcept { return last_alignment_; }
   void* last_deallocated_ptr() const noexcept { return last_deallocated_ptr_; }

private:
   std::size_t allocate_calls_{0};
   std::size_t deallocate_calls_{0};
   std::size_t last_size_{0};
   std::size_t last_alignment_{0};
   void* last_deallocated_ptr_{nullptr};
   std::unordered_set<void*> active_ptrs_;
};