#ifndef SUB_ALLOCATOR_INTERFACE_H
#define SUB_ALLOCATOR_INTERFACE_H

#include <cstddef>

class ISubAllocator {
   public:
     virtual ~ISubAllocator() = default;
     virtual void* allocate_region(std::size_t alignment, std::size_t size_bytes) = 0;
     virtual void deallocate_region(void* ptr) = 0;

//     virtual const char* name() const = 0;
};

#endif // SUB_ALLOCATOR_INTERFACE_H
