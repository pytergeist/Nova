#ifndef FUSION_OPS_ALIGNED_ARITHMETIC_HPP
#define FUSION_OPS_ALIGNED_ARITHMETIC_HPP

#include <string_view>

#include "Fusion/core/planning/OpContextBuilders.h"
#include "Fusion/execution/cpu/BinaryElementwise.h"
#include "Fusion/execution/cpu/UnaryElementwise.h"

#include "Fusion/ops/OperandValidation.h"
#include "Fusion/ops/OutputAllocation.h"

namespace fusion::ops::aligned {

template <typename T>
DenseTensor<T> add(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   validation::validate_dense_binary_operation<T, AddTag>(x, y);
   planning::BinaryEwiseContext meta =
       planning::make_binary_ewise_context(x, y);
   DenseTensor<T> out = detail::init_out_from_meta(x, y, meta);
   execution::cpu::binary_elementwise<T, AddSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
DenseTensor<T> sub(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   validation::validate_dense_binary_operation<T, SubTag>(x, y);
   planning::BinaryEwiseContext meta =
       planning::make_binary_ewise_context(x, y);
   DenseTensor<T> out = detail::init_out_from_meta(x, y, meta);
   execution::cpu::binary_elementwise<T, SubtractSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
DenseTensor<T> mul(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   validation::validate_dense_binary_operation<T, MulTag>(x, y);
   planning::BinaryEwiseContext meta =
       planning::make_binary_ewise_context(x, y);
   DenseTensor<T> out = detail::init_out_from_meta(x, y, meta);
   execution::cpu::binary_elementwise<T, MultiplySIMD>(x, y, meta, out);
   return out;
}

template <typename T>
DenseTensor<T> div(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   validation::validate_dense_binary_operation<T, DivTag>(x, y);
   planning::BinaryEwiseContext meta =
       planning::make_binary_ewise_context(x, y);
   DenseTensor<T> out = detail::init_out_from_meta(x, y, meta);
   execution::cpu::binary_elementwise<T, DivideSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
DenseTensor<T> pow(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   validation::validate_dense_binary_operation<T, PowTag>(x, y);
   planning::BinaryEwiseContext meta =
       planning::make_binary_ewise_context(x, y);
   DenseTensor<T> out = detail::init_out_from_meta(x, y, meta);
   execution::cpu::binary_elementwise<T, PowerSIMD>(x, y, meta, out);
   return out;
}

template <typename T> DenseTensor<T> reciprocal(const DenseTensor<T> &x) {
   validation::validate_dense_unary_operation<T, ReciprocalTag>(x);
   planning::UnaryEwiseContext meta = planning::make_unary_ewise_context(x);
   DenseTensor<T> out = detail::init_out_from_meta(x, meta);
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
