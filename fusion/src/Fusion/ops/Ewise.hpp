#ifndef EWISE_HPP
#define EWISE_HPP

#include <string_view>

#include "Fusion/core/PlanMeta.hpp"
#include "Fusion/core/RawTensor.hpp"
#include "Fusion/core/TensorIter.hpp"

#include "Helpers.hpp"

namespace fusion {

namespace math {

template <typename T>
RawTensor<T> add(const RawTensor<T> &x, const RawTensor<T> &y) {
   require_ewise_binary_out_of_place<AddTag>();
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   fusion::iter::binary_ewise_tag<T, AddSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
RawTensor<T> sub(const RawTensor<T> &x, const RawTensor<T> &y) {
   require_ewise_binary_out_of_place<SubTag>();
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   fusion::iter::binary_ewise_tag<T, SubtractSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
RawTensor<T> mul(const RawTensor<T> &x, const RawTensor<T> &y) {
   require_ewise_binary_out_of_place<MulTag>();
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   fusion::iter::binary_ewise_tag<T, MultiplySIMD>(x, y, meta, out);
   return out;
}

template <typename T>
RawTensor<T> div(const RawTensor<T> &x, const RawTensor<T> &y) {
   require_ewise_binary_out_of_place<DivTag>();
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   fusion::iter::binary_ewise_tag<T, DivideSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
RawTensor<T> pow(const RawTensor<T> &x, const RawTensor<T> &y) {
   require_ewise_binary_out_of_place<PowTag>();
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   fusion::iter::binary_ewise_tag<T, PowerSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline RawTensor<T> reciprocal(const RawTensor<T> &x) {
   require_ewise_unary_out_of_place<ReciprocalTag>();
   UnaryEwiseMeta meta = make_unary_meta(x);
   RawTensor<T> out = init_out_from_meta(x, meta);
   fusion::iter::unary_ewise_tag<T, ReciprocalSIMD>(x, meta, out);
   return out;
}

template <typename T>
inline void sub_inplace(RawTensor<T> &x, const RawTensor<T> &y) {
   // TODO: need to impl_ a way to ignore batch dim in shape check in
   // a sensible way
   // UNSAFE CODE
   BinaryEwiseMeta meta{};
   meta.out_shape = x.shape();
   meta.fast_len = x.flat_size();
   meta.exec = BinaryExecKind::FlatContiguous;
   // this impl is unstable for > rank(2) NDtensors
   FUSION_CHECK(meta.out_shape[x.rank() - 1] == x.shape()[x.rank() - 1],
                "sub_inplace would change tensor shape; "
                "use out-of-place sub() instead.");

   fusion::iter::binary_ewise_tag<T, SubtractSIMD>(x, y, meta, x);
}

} // namespace math

} // namespace fusion

#endif // EWISE_H
