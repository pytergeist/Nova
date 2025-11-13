#ifndef POOL_H
#define POOL_H

#include <cstddef>
#include <cstdint>
#include <utility>

/* This header file includes code for the memory pool used by the pool
allocator, the pool consists of bucketss, which point to mem regions - the
buckets store various meta-data, which describes whether a bucket is in use,
what the next/previous mem region is, the ptr to the curr mem region. */

using BucketId = std::size_t;
static const BucketId kInvalidBucketId = static_cast<BucketId>(-1);

struct Bucket {
 public:

   bool in_use() const noexcept { return std::cmp_not_equal(allocation_id_, kInvalidBucketId); }

   std::size_t size() const noexcept { return size_; }
   std::size_t requested_size() const noexcept { return requested_size_; }
   std::int64_t allocation_id() const noexcept { return allocation_id_; }

   void set_size(std::size_t bucket_size) noexcept { size_ = bucket_size; }
   void set_requested_size(std::size_t req_bucket_size) noexcept {
      requested_size_ = req_bucket_size;
   }
   void set_allocation_id(std::int64_t alloc_id) noexcept {
      allocation_id_ = alloc_id;
   }

   BucketId prev() const noexcept { return prev_; }
   BucketId next() const noexcept { return next_; }

 private:
   void *ptr = nullptr; // ptr to mem
   std::size_t size_ = 0;           // size of buffer
   std::size_t requested_size_ = 0; // the client requested size of the buffer
   std::int64_t allocation_id_ = -1;

   // next/prev allow iter to prev/next contiguous mem region
   BucketId prev_ = kInvalidBucketId; // starts at ptr - prev->size
   BucketId next_ = kInvalidBucketId; // starts at ptr + size
};

#endif // POOL_H
