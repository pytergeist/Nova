#ifndef CPU_SUB_ALLOCATOR_H
#define CPU_SUB_ALLOCATOR_H

#include <cstddef>
#include <cstdlib>
#include <stdexcept>

#include "AllocTypes.h"
#include "SubAllocatorInterface.h"

class CPUSubAllocator final : public ISubAllocator {
 public:
   CPUSubAllocator() = default;

   void *allocate_region(Alignment alignment, std::size_t size_bytes) override;

   void deallocate_region(void *ptr) override;
};

#endif // CPU_SUB_ALLOCATOR_H
