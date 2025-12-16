#include "BFCPoolAllocator.h"



void RegionManager::add_allocated_region(void *ptr, std::size_t region_size,
                                         Alignment alignment) {
   regions_.emplace_back(Region{.ptr = ptr,
                                .region_id = counter_,
                                .size = region_size,
                                .alignment = alignment});

   counter_++;
}

Region &RegionManager::find_region_for_ptr(void *ptr) {
   auto *addr = static_cast<std::byte *>(ptr);
   for (auto &region : regions_) {
      auto *base = static_cast<std::byte *>(region.ptr);
      auto *end =
          base +
          region
              .size; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      if (addr >= base && addr < end) {
         return region;
      }
   }
   throw std::runtime_error("RegionManager: failed to find region for ptr");
}

ChunkID RegionManager::get_chunkid_from_ptr(void *chunk_ptr) {
   auto it = ptr_chunk_map_.find(chunk_ptr);
   if (it == ptr_chunk_map_.end()) {
      FUSION_LOGI(false,
                  "PoolAllocator: deallocate called with unknown pointer: ",
                  chunk_ptr, " (double free or foreign pointer)");
      throw std::runtime_error(
          "PoolAllocator: unknown pointer in get_chunkid_from_ptr");
   }
   return it->second;
}

void RegionManager::set_chunkid(void *chunk_ptr, ChunkID chunk_id) {
   // Reuses same ptr for different chunkIds after coalesce / split
   ptr_chunk_map_.insert_or_assign(chunk_ptr, chunk_id);
}

bool RegionManager::erase_chunk(void *chunk_ptr) {
   // returns 1 if existed, 0 if not
   return static_cast<bool>(ptr_chunk_map_.erase(chunk_ptr));
}

std::vector<Region> RegionManager::regions() const { return regions_; }

std::vector<Region> RegionManager::regions() { return regions_; }

PoolAllocator::PoolAllocator()
    : sub_allocator_(std::make_unique<CPUSubAllocator>()) {}

PoolAllocator::~PoolAllocator() = default;

void *PoolAllocator::allocate(std::size_t size, Alignment alignment) {
   if (size == 0) {
      size = 1;
   }

   size = round_up_pow2(size);

   ChunkID free_id = find_free_chunk_id_for_size(size);

   if (free_id == kInvalidChunkID) {
      grow_pool_for_size(size, alignment);
      free_id = find_free_chunk_id_for_size(size);
      if (free_id == kInvalidChunkID) {
         throw std::bad_alloc();
      }
   }

   Chunk &chunk = get_chunk_from_id(free_id);

   erase_chunk_from_bucket(chunk);

   const ChunkID allocated_id =
       split_chunk_for_allocation(chunk.chunk_id, size);
   Chunk &allocated = get_chunk_from_id(allocated_id);

   allocated.in_use = true;
   allocated.requested_size = size;

   void *chunk_ptr = allocated.ptr; // NOLINT
   if (chunk_ptr != nullptr) {
      return chunk_ptr;
   }
   throw std::bad_alloc();
}

void PoolAllocator::deallocate(void *ptr) {
   if (ptr == nullptr) {
      return;
   }

   ChunkID chunk_id = region_manager_.get_chunkid_from_ptr(ptr);
   Chunk &chunk = get_chunk_from_id(chunk_id);

   chunk.in_use = false;
   chunk.requested_size = 0;

   chunk_id = free_and_maybe_coalesce(chunk_id);
   const Chunk &merged = get_chunk_from_id(chunk_id);

   const std::size_t bucket_size = round_down_pow2(merged.size);
   Bucket &bucket = get_or_create_bucket(bucket_size);
   bucket.free_chunks.insert(chunk_id);
}

std::vector<Chunk> PoolAllocator::chunks() const { return chunks_; }

std::vector<ChunkID>
PoolAllocator::get_free_chunks(std::size_t bucket_size) const {
   std::vector<ChunkID> result;
   auto it = buckets_by_size_.find(bucket_size);
   if (it == buckets_by_size_.end()) {
      return result;
   }
   const Bucket &bucket = it->second;
   result.insert(result.end(), bucket.free_chunks.begin(),
                 bucket.free_chunks.end());
   return result;
}

std::size_t PoolAllocator::round_up_pow2(std::size_t n) {
   static constexpr int kSizeTBits = int(sizeof(std::size_t) * 8);
   if (n <= 1) {
      return 1;
   }
   const int bw = std::bit_width(n - 1);
   if (std::cmp_greater_equal(bw, kSizeTBits)) {
      return std::numeric_limits<std::size_t>::max();
   }
   return std::size_t{1} << bw;
}

std::size_t PoolAllocator::round_down_pow2(std::size_t n) {
   if (n <= 1) {
      return 1;
   }
   const int bw = std::bit_width(n);
   return std::size_t{1} << (bw - 1);
}

Chunk &PoolAllocator::get_chunk_from_id(ChunkID chunk_id) {
   FUSION_BOUNDS_CHECK(chunk_id, chunks_.size());
   return chunks_[chunk_id];
}

Bucket &PoolAllocator::get_or_create_bucket(std::size_t bucket_size) {
   auto it = buckets_by_size_.find(bucket_size);
   if (it != buckets_by_size_.end()) {
      return it->second;
   }

   Bucket bucket{&chunks_, bucket_size,
                 static_cast<BucketID>(buckets_by_size_.size())};

   auto [inserted_it, _] =
       buckets_by_size_.emplace(bucket_size, std::move(bucket));
   return inserted_it->second;
}

ChunkID PoolAllocator::find_free_chunk_id_for_size(std::size_t size) {
   const std::size_t size_class = round_up_pow2(size);

   for (auto it = buckets_by_size_.lower_bound(size_class);
        it != buckets_by_size_.end(); ++it) {
      const Bucket &bucket = it->second;
      if (bucket.free_chunks.empty()) {
         continue;
      }
      for (const ChunkID id : bucket.free_chunks) {
         const Chunk &chunk = get_chunk_from_id(id);
         if (!chunk.in_use && chunk.size >= size) {
            return id;
         }
      }
   }
   return kInvalidChunkID;
}

void PoolAllocator::grow_pool_for_size(std::size_t size, Alignment alignment) {
   while (current_allocation_size_ < size) {
      current_allocation_size_ <<= 1;
   }

   void *ptr = allocate_bucket_region(current_allocation_size_, alignment);

   Chunk chunk;
   chunk.ptr = ptr;
   chunk.chunk_id = ChunkID{chunk_counter_++};
   chunk.prev = kInvalidChunkID;
   chunk.next = kInvalidChunkID;
   chunk.size = current_allocation_size_;
   chunk.in_use = false;
   chunk.requested_size = 0;
   chunk.set_end_ptr();

   region_manager_.set_chunkid(ptr, chunk.chunk_id);
   chunks_.push_back(chunk);

   const std::size_t bucket_size = round_down_pow2(chunk.size);
   Bucket &bucket = get_or_create_bucket(bucket_size);
   bucket.free_chunks.insert(chunk.chunk_id);
}

void *PoolAllocator::allocate_bucket_region(std::size_t region,
                                            Alignment alignment) {
   void *ptr = sub_allocator_->allocate_region(alignment, region);
   region_manager_.add_allocated_region(ptr, region, alignment);
   return ptr;
}

ChunkID PoolAllocator::split_chunk_for_allocation(ChunkID chunk_id,
                                                  std::size_t size) {

   Chunk &chunk = get_chunk_from_id(chunk_id);

   if (chunk.size < size) {
      throw std::runtime_error("split_chunk_for_allocation: chunk too small");
   }

   const std::size_t remainder_size = chunk.size - size;

   if (remainder_size < kMinAllocationSize) {
      return chunk_id;
   }

   std::byte *base = static_cast<std::byte *>(chunk.ptr); // NOLINT
   void *rem_ptr = static_cast<void *>(
       base + size); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

   Chunk remainder;
   remainder.ptr = rem_ptr;
   remainder.chunk_id = static_cast<ChunkID>(chunk_counter_++);
   remainder.prev = chunk.chunk_id;
   remainder.next = chunk.next;
   remainder.size = remainder_size;
   remainder.in_use = false;
   remainder.requested_size = 0;
   remainder.set_end_ptr();

   if (chunk.next != kInvalidChunkID) {
      Chunk &next_chunk = get_chunk_from_id(chunk.next);
      next_chunk.prev = remainder.chunk_id;
   }
   chunk.next = remainder.chunk_id;

   chunk.size = size;
   chunk.set_end_ptr();
   region_manager_.set_chunkid(remainder.ptr, remainder.chunk_id);
   chunks_.push_back(remainder);

   const std::size_t rem_bucket_size = round_down_pow2(remainder.size);
   Bucket &rem_bucket = get_or_create_bucket(rem_bucket_size);
   rem_bucket.free_chunks.insert(remainder.chunk_id);

   return chunk_id;
}

void PoolAllocator::delete_chunk(Chunk &chunk) {
   chunk.size = 0;
   chunk.requested_size = 0;
   chunk.ptr = nullptr;
   chunk.end_ptr_ = nullptr;
   chunk.prev = kInvalidChunkID;
   chunk.next = kInvalidChunkID;
   chunk.in_use = false;
}

void PoolAllocator::erase_chunk_from_bucket(Chunk &chunk) {
   if (chunk.size == 0) {
      return;
   }
   const std::size_t bucket_size = round_down_pow2(chunk.size);
   auto it = buckets_by_size_.find(bucket_size);
   if (it == buckets_by_size_.end()) {
      return;
   }
   Bucket &bucket = it->second;
   bucket.free_chunks.erase(chunk.chunk_id);
}

ChunkID PoolAllocator::merge_chunks(Chunk &lchunk, Chunk &rchunk) {
   const std::byte *lbase = static_cast<std::byte *>(lchunk.ptr);
   const std::byte *rbase = static_cast<std::byte *>(rchunk.ptr);
   if (lbase + lchunk.size ==
       rbase) { // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      region_manager_.erase_chunk(rchunk.ptr);
      erase_chunk_from_bucket(rchunk);
      erase_chunk_from_bucket(lchunk);

      lchunk.size += rchunk.size;
      lchunk.set_end_ptr();

      const ChunkID rnext_id = rchunk.next;
      lchunk.next = rnext_id;
      if (rnext_id != kInvalidChunkID) {
         Chunk &rnext = get_chunk_from_id(rnext_id);
         rnext.prev = lchunk.chunk_id;
      }

      delete_chunk(rchunk);
      return lchunk.chunk_id;
   }
   return rchunk.chunk_id;
}

ChunkID PoolAllocator::free_and_maybe_coalesce(ChunkID chunk_id) {
   ChunkID current_id = chunk_id;

   while (true) {
      Chunk &chunk = get_chunk_from_id(current_id);

      if (chunk.prev == kInvalidChunkID) {
         break;
      }

      Chunk &prev_chunk = get_chunk_from_id(chunk.prev);
      if (prev_chunk.in_use) {
         break;
      }

      const ChunkID new_id = merge_chunks(prev_chunk, chunk);

      if (new_id == current_id) {
         break;
      }

      current_id = new_id;
   }

   return current_id;
}
