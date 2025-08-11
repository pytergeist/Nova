#ifndef BROADCAST_ITERATOR_H
#define BROADCAST_ITERATOR_H

#include <cstddef>
#include <cstdint>
#include <vector>

struct TensorDescription {
  int ndims;
  std::vector<std::int64_t> sizes;
  std::vector<std::int64_t> strides;
  std::size_t itemsize;
};

struct LoopDim {
  std::int64_t size;
  std::vector<int64_t> stride_bytes;
};


struct BroadcastPlan {
  int num_operands;
  int out_ndim;
  std::vector<int64_t> out_sizes;
  std::vector<LoopDim> loop;

  bool all_contiguous_like{false};
  int64_t vector_bytes{0};

  std::vector<int64_t> out_strides;
  std::size_t itemsize;
};

BroadcastPlan make_broadcast_plan(const std::vector<TensorDescription>& descs);

#endif // BROADCAST_ITERATOR_H
