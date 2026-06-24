#ifndef FUSION_OPS_ALIGNED_COMPARISON_HPP
#define FUSION_OPS_ALIGNED_COMPARISON_HPP

#include <string_view>
#include <vector>

#include "Fusion/core/iter/TensorIter.hpp"
#include "Fusion/core/planning/PlanMeta.hpp"

#include "../Helpers.hpp"

namespace fusion::ops::aligned {

template <typename T>
DenseTensor<T> greater(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   require_ewise_binary_out_of_place<GreaterTag>();
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   DenseTensor<T> out = init_out_from_meta(x, y, meta);
   fusion::dense::iter::binary_ewise_tag<T, GreaterThanSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
DenseTensor<T> greater_equal(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   require_ewise_binary_out_of_place<GreaterEqualTag>();
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   DenseTensor<T> out = init_out_from_meta(x, y, meta);
   fusion::dense::iter::binary_ewise_tag<T, GreaterThanEqualSIMD>(x, y, meta,
                                                                  out);
   return out;
}

template <typename T>
DenseTensor<T> maximum(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   require_ewise_binary_out_of_place<MaximumTag>();
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   DenseTensor<T> out = init_out_from_meta(x, y, meta);
   fusion::dense::iter::binary_ewise_tag<T, MaximumSIMD>(x, y, meta, out);
   return out;
}

} // namespace fusion::ops::aligned

#endif // FUSION_OPS_ALIGNED_COMPARISON_HPP
