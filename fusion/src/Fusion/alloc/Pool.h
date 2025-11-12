#ifndef POOL_H
#define POOL_H

#include <cstddef>

// implamentation of constant sixed sized mem bucket

class Bucket {
   public:

     Bucket(std::size_t block_size, std::size_t block_count) : block_size_(block_size), block_count_(block_count) {
        const auto data_size = block_size * block_count;
        mem_data_ = static_cast<std::byte*>(std::malloc(data_size));
        assert(mem_data_ != nullptr);
        const auto ledger_size = 1 + ((block_count - 1) / 8);
        mem_ledger_ = static_cast<std::byte *>(std::malloc(ledger_size));
        assert(mem_ledger_ != nullptr);
        std::memset(mem_data_, 0, data_size);
        std::memset(mem_ledger_, 0, ledger_size);
     }

     ~Bucket() {
        std::free(mem_ledger_);
        std::free(mem_data_);
     }


     std::size_t block_size() const noexcept {return block_size_;}
     std::size_t block_count() const noexcept {return block_count_;}

     bool belongs(void* ptr) const noexcept;

     [[nodiscard]] void* allocate(std::size_t bytes) noexcept {
        const auto n = 1 + ((bytes - 1) / block_size()); // calc num blocks

        const auto index  = find_contiguous_blocks(n);
        if (index == block_count()) {
           return nullptr;
        }

        set_blocks_in_use(index, n);
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

     inline void set_bit(std::byte* ledger, std::size_t bit_index) noexcept {
        const std::size_t byte_idx = bit_index / 8;
        const std::size_t bit_off = byte_idx % 8;
        const std::byte mask = std::byte{1u} << bit_off;
		ledger[byte_idx] |= mask;
     }

     inline void clear_bit(std::byte* ledger, std::size_t bit_index) noexcept {
        const std::size_t byte_idx = bit_index / 8;
        const std::size_t bit_off = byte_idx % 8;
        const std::byte mask = std::byte{1u} << bit_off;
        ledger[byte_idx] &= ~(mask);
     }

     std::size_t find_contiguous_blocks(std::size_t n) const noexcept; // TODO: impl

     void set_blocks_in_use(std::size_t idx, std::size_t n) noexcept {
        assert(idx + n <= block_count_);
        for (std::size_t i = 0; i < n; ++i) {
           set_bit(mem_ledger_, idx);
        }
     }

     void set_blocks_free(std::size_t idx, std::size_t n) noexcept {
        assert(idx + n <= block_count_);
        for (std::size_t i = 0; i < n; ++i) {
           clear_bit(mem_ledger_, idx);
        }
     }
};

template<std::size_t Id>
struct bucket_descriptors;

 // TODO: Either define these per system or find a way to calculate them
struct bucket_cfg16 {
   static constexpr std::size_t block_size = 16;
   static constexpr std::size_t block_count = 10000;
};

struct bucket_cfg32 {
   static constexpr std::size_t block_size = 32;
   static constexpr std::size_t block_count = 10000;
};

struct bucket_cfg64 {
   static constexpr std::size_t block_size = 64;
   static constexpr std::size_t block_count = 10000;
};


template<>
struct bucket_descriptors<1> {
   using type = std::tuple<bucket_cfg16, bucket_cfg32, bucket_cfg64>;
};


template<std::size_t id>
using bucket_descriptor_t = typename bucket_descriptors<id>::type;

template<std::size_t id>
static constexpr std::size_t bucket_count = std::tuple_size<bucket_descriptor_t<id>>::value;

template<std::size_t id>
using pool_type = std::array<Bucket, bucket_count<id>>;

template<std::size_t id, std::size_t Idx>
struct get_size : std::integral_constant<std::size_t, std::tuple_element_t<Idx, bucket_descriptor_t<id>>::block_size> {};

template<std::size_t id, std::size_t Idx>
struct get_count : std::integral_constant<std::size_t, std::tuple_element_t<Idx, bucket_descriptor_t<id>>::block_count>{};

#endif // POOL_H
