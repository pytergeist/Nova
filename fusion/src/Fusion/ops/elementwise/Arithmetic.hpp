#ifndef FUSION_OPS_ALIGNED_ARITHMETIC_HPP
#define FUSION_OPS_ALIGNED_ARITHMETIC_HPP

#include <string_view>

#include "Fusion/execution/cpu/BinaryElementwise.h"
#include "Fusion/execution/cpu/UnaryElementwise.h"
#include "Fusion/core/planning/OpContextBuilders.h"

#include "Fusion/ops/Helpers.hpp"

namespace fusion::ops::aligned {

template <typename T>
DenseTensor<T> add(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   require_ewise_binary_out_of_place<AddTag>();
   planning::BinaryEwiseContext meta =
       planning::make_binary_ewise_context(x, y);
   DenseTensor<T> out = init_out_from_meta(x, y, meta);
   execution::cpu::binary_elementwise<T, AddSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
DenseTensor<T> sub(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   require_ewise_binary_out_of_place<SubTag>();
   planning::BinaryEwiseContext meta =
       planning::make_binary_ewise_context(x, y);
   DenseTensor<T> out = init_out_from_meta(x, y, meta);
   execution::cpu::binary_elementwise<T, SubtractSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
DenseTensor<T> mul(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   require_ewise_binary_out_of_place<MulTag>();
   planning::BinaryEwiseContext meta =
       planning::make_binary_ewise_context(x, y);
   DenseTensor<T> out = init_out_from_meta(x, y, meta);
   execution::cpu::binary_elementwise<T, MultiplySIMD>(x, y, meta, out);
   return out;
}

template <typename T>
DenseTensor<T> div(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   require_ewise_binary_out_of_place<DivTag>();
   planning::BinaryEwiseContext meta =
       planning::make_binary_ewise_context(x, y);
   DenseTensor<T> out = init_out_from_meta(x, y, meta);
   execution::cpu::binary_elementwise<T, DivideSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
DenseTensor<T> pow(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   require_ewise_binary_out_of_place<PowTag>();
   planning::BinaryEwiseContext meta =
       planning::make_binary_ewise_context(x, y);
   DenseTensor<T> out = init_out_from_meta(x, y, meta);
   execution::cpu::binary_elementwise<T, PowerSIMD>(x, y, meta, out);
   return out;
}

template <typename T> DenseTensor<T> reciprocal(const DenseTensor<T> &x) {
   require_ewise_unary_out_of_place<ReciprocalTag>();
   planning::UnaryEwiseContext meta = planning::make_unary_ewise_context(x);
   DenseTensor<T> out = init_out_from_meta(x, meta);
   execution::cpu::unary_elementwise<T, ReciprocalSIMD>(x, meta, out);
   return out;
}

template <typename T>
void sub_inplace(DenseTensor<T> &x, const DenseTensor<T> &y) {
   // TODO: need to impl_ a way to ignore batch dim in shape check in
   // a sensible way
   // UNSAFE CODE
   planning::BinaryEwiseContext meta{};
   meta.out_shape = x.shape();
   meta.fast_len = x.flat_size();
   meta.exec = planning::BinaryExecKind::FlatContiguous;
   // this impl is unstable for > rank(2) NDtensors
   FUSION_CHECK(meta.out_shape[x.rank() - 1] == x.shape()[x.rank() - 1],
                "sub_inplace would change tensor shape; "
                "use out-of-place sub() instead.");

   execution::cpu::binary_elementwise<T, SubtractSIMD>(x, y, meta, x);
}

} // namespace fusion::ops::aligned

#endif // FUSION_OPS_ALIGNED_ARITHMETIC_HPP
