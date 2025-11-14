#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <random>

#include "Tensor.h"
#include "autodiff/Engine.h"
#include "autodiff/EngineContext.h"
#include "core/Device.h"
#include "alloc/PoolAllocator.h"

int main() {
   std::cout << "std::size_t: " << sizeof(std::size_t) << std::endl;
   std::cout << "std::int64_t: " << sizeof(std::int64_t) << std::endl;
   std::cout << "void*: " << sizeof(void*) << std::endl;
   return 0;
}
