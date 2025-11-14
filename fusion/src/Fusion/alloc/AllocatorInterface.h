#ifndef ALLOCATOR_INTERFACE_H_
#define ALLOCATOR_INTERFACE_H_

#include <memory>

struct Chunk;
using ChunkId = std::size_t;

class IAllocator {
   public:
     IAllocator() = default;
     virtual void *allocate(std::size_t size) = 0;
     virtual void deallocate(void* p) = 0;

     virtual const Chunk* ChunkFromId(const ChunkId cid) = 0;

   private:
     virtual ~IAllocator() = default;

};


#endif // ALLOCATOR_INTERFACE_H_
