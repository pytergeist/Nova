#ifndef OP_PARAMS_HPP
#define OP_PARAMS_HPP

#include <cstddef>

struct SwapAxesParam {
   int axis1;
   int axis2;
};

struct ReductionParam {
   std::size_t reduction_axis{};
   bool keepdim{};
};

#endif // OP_PARAMS_HPP
