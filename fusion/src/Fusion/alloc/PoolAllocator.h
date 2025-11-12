#ifndef POOL_ALLOCATOR_H
#define POOL_ALLOCATOR_H

#include <memory>

// implamentation of constant sixed sized mem bucket

class Bucket {
   public:

     bucket(std::size_t block_size, std::size_t block_count) : block_size_(block_size), block_count_(block_count) {
        const auto data_size = block_size * block_count;
        mem_data_ = static_cast<std::byte*>(std::malloc(data_size));
        assert(mem_data_ != nullptr);
        const auto ledger_size = 1 + ((block_count - 1) / 8);
        mem_ledger_ = static_cast<std::byte *>(std::malloc(ledger_size));
        assert(mem_ledger_ != nullptr);
        std::memset(data_ledger_, 0, data_size);
        std::memset(mem_ledger_, 0, ledger_size);
     }

     ~bucket() {
        std::free(mem_ledger_);
        std::free(mem_data_);
     }


     std::size_t block_size() const noexcept {return block_size;}
     std::size_t block_count() const noexcept {return block_count;}

     bool belongs(void* ptr) const noexcept;

     [[nodiscard]] void* allocate(std::size_t bytes) noexcept {
        const auto n = 1 + ((bytes - 1) / block_size()); // calc num blocks

        const auto index  = find_contiguous_blocks(n);
        if (index == block_count()) {
           return nullptr;
        }

        set_blocks_in_user(index, n);
        return mem_data_ + (index * block_size());

     };


     void deallocate(void* ptr, std::size_t bytes) noexcept {
        const auto p = static_cast<const std::byte *>(ptr);
        const std::size_t dist = static_cast<std::size_t>(p - mem_data_);
        const auto index = dist / block_size();
        const auto n = 1 + ((bytes - 1) / block_size());

        set_blocks_free(index, n);
     };

   private:
     const std::size_t block_size_;
     const std::size_t block_count_;
     std::byte* mem_data_{nullptr};
     std::byte* mem_ledger_{nullptr};

     std::size_t find_contiguous_blocks(std::size_t n) const noexcept;
     void set_blocks_in_use(std::size_t idx, std::size_t n) noexcept;
     void set_blocks_free(std::size_t idx, std::size_t n) noexcept;
}

template <typename T>
class PoolAllocator : public IAllocator {
   public:
     PoolAllocator() = default;

   private:
     std::vector<void*> free_list_;


}

#endif // POOL_ALLOCATOR_H
