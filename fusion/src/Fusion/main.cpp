#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <random>

#include "Tensor.h"
#include "autodiff/Engine.h"
#include "autodiff/EngineContext.h"
#include "autodiff/Engine.h"
#include "core/Device.h"
#include "alloc/PoolAllocator.h"
#include "alloc/CPUSubAllocator.h"
#include "alloc/BFCPool.h"
#include "autodiff/EngineContext.h"

int main() {
   PoolAllocator pool_allocator;
   void* p = pool_allocator.allocate(17, 64);
   void* ptr2 = pool_allocator.allocate(14, 64);
   void* ptr22 = pool_allocator.allocate(15, 64);
   void* ptr3 = pool_allocator.allocate(180, 64);
//   pool_allocator.allocate(300, 64);
//   pool_allocator.allocate(600, 64);

//   std::cout << "first allo/dealloc" << std::endl;
//   std::cout << p << std::endl;
//   pool_allocator.deallocate(p);

//   std::cout << "second allo/dealloc" << std::endl;
//   std::cout << ptr2 << std::endl;
//   pool_allocator.deallocate(ptr2);
//
//
//   std::cout << "third allo/dealloc" << std::endl;
//   std::cout << ptr3 << std::endl;
//   pool_allocator.deallocate(ptr3);
//
//   std::cout << "fourth allo/dealloc" << std::endl;
//   std::cout << ptr22 << std::endl;
//   pool_allocator.deallocate(ptr22);

   std::vector<Chunk> clist = pool_allocator.chunks();
   std::set<ChunkId> fcset = pool_allocator.get_free_chunks(64);

   for (auto& chunk : clist) {
      if (chunk.chunk_id < 10) {
      std::cout << "==============================================" << std::endl;
      std::cout << "Chunk id: " << chunk.chunk_id << std::endl;
      std::cout << "Chunk in_use: " << chunk.in_use << std::endl;
      std::cout << "Chunk prev id: " << chunk.prev << std::endl;
      std::cout << "Chunk next id: " << chunk.next << std::endl;
      std::cout << "Chunk size: " << chunk.size << std::endl;
      std::cout << "Chunk requested_size: " << chunk.requested_size << std::endl;
      std::cout << "==============================================" << std::endl;
      }
   }

   std::cout << "********** free set *********" << std::endl;
   for (auto s: fcset) {
      if (s < 10) {
      std::cout << s << std::endl;
      }
   }


   std::cout << "===== DeAlloc ========" << std::endl;

   std::cout << "first allo/dealloc" << std::endl;
   std::cout << p << std::endl;
   pool_allocator.deallocate(p);

   std::vector<Chunk> dclist = pool_allocator.chunks();
   std::set<ChunkId> dfcset = pool_allocator.get_free_chunks(64);

   for (auto& chunk : dclist) {
      if (chunk.chunk_id < 10) {
      std::cout << "==============================================" << std::endl;
      std::cout << "Chunk id: " << chunk.chunk_id << std::endl;
      std::cout << "Chunk in_use: " << chunk.in_use << std::endl;
      std::cout << "Chunk prev id: " << chunk.prev << std::endl;
      std::cout << "Chunk next id: " << chunk.next << std::endl;
      std::cout << "Chunk size: " << chunk.size << std::endl;
      std::cout << "Chunk requested_size: " << chunk.requested_size << std::endl;
      std::cout << "==============================================" << std::endl;
      }
   }

   std::cout << "********** free set *********" << std::endl;
   for (auto s: dfcset) {
      if (s < 10) {
      std::cout << s << std::endl;
      }
   }

   std::cout << "ALIGNEMNT: " << alignof(std::max_align_t) << std::endl;

}


//int main() {
//    CPUSubAllocator allocator;
//    std::size_t alignment = 64;
//    std::size_t region_size = 256;
//
//    void* ptr = allocator.allocate_region(alignment, region_size);
//
//    std::cout << "Allocated pointer: " << ptr << "\n";
//
//    // Convert to integer
//    std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(ptr);
//
//    // Print raw address
//    std::cout << "Address (hex): 0x" << std::hex << addr << std::dec << "\n";
//
//    // Print region size
//    std::cout << "Region size: " << region_size << " bytes\n";
//
//    // Check alignment
//    std::cout << "Alignment remainder: " << (addr % alignment) << "\n";
//    if (addr % alignment == 0) {
//        std::cout << "Pointer is aligned to " << alignment << " bytes\n";
//    } else {
//        std::cout << "Pointer is NOT aligned to " << alignment << " bytes\n";
//    }
//
//    std::uint8_t* bytes = reinterpret_cast<std::uint8_t*>(ptr);
//
//    for (std::size_t i = 0; i < region_size; i++) {
//        bytes[i] = static_cast<std::uint8_t>(i & 0xFF);
//    }
//
//    std::cout << "Successfully wrote " << region_size << " bytes.\n";
//    allocator.deallocate_region(ptr);
//
//    return 0;
//}
