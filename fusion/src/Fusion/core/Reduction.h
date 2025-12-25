#ifndef REDUCTION_H
#define REDUCTION_H

#include <vector>
#include <cstdef>

#include "Broadcast.h"

struct ReductionPlan {
    TensorDescription input;
    TensorDescription output;
    std::size_t out_ndim;
    std::vector<std::size_t> out_strides;
    std::vector<std::size_t> reduction_axis;
    std::vector<LoopDim> loops;

    bool all_contiguous_like{false};
    std::size_t vector_bytes{0};

    std::vector<std::int64_t> out_strides;
    std::size_t itemsize;
};

#endif // REDUCTION_H
