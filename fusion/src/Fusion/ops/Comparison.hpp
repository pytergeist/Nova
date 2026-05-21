#ifndef OPS_COMPARISON_HPP
#define OPS_COMPARISON_HPP

#include <string_view>
#include <vector>

#include "../core/tensor/RawTensor.hpp"
#include "Fusion/core/PlanMeta.hpp"
#include "Fusion/core/TensorIter.hpp"

#include "Helpers.hpp"

namespace fusion {

namespace math {

template <typename T>
RawTensor<T> greater(const RawTensor<T> &x, const RawTensor<T> &y) {
   require_ewise_binary_out_of_place<GreaterTag>();
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   fusion::iter::binary_ewise_tag<T, GreaterThanSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
RawTensor<T> greater_equal(const RawTensor<T> &x, const RawTensor<T> &y) {
   require_ewise_binary_out_of_place<GreaterEqualTag>();
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   fusion::iter::binary_ewise_tag<T, GreaterThanEqualSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
RawTensor<T> maximum(const RawTensor<T> &x, const RawTensor<T> &y) {
   require_ewise_binary_out_of_place<MaximumTag>();
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   fusion::iter::binary_ewise_tag<T, MaximumSIMD>(x, y, meta, out);
   return out;
}

} // namespace math

} // namespace fusion

#endif // OPS_COMPARISON_HPP
