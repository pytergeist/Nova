#ifndef FUSION_OPS_ALIGNED_ARITHMETIC_HPP
#define FUSION_OPS_ALIGNED_ARITHMETIC_HPP

#include <string_view>

#include "Fusion/core/planning/OpContextBuilders.h"
#include "Fusion/execution/cpu/BinaryElementwise.h"
#include "Fusion/ops/elementwise/BinaryOp.h"
#include "Fusion/ops/elementwise/UnaryOp.h"

namespace fusion::ops::aligned {

template <typename T>
DenseTensor<T> add(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   return apply_binary_op<T, AddTag>(x, y);
}

template <typename T>
DenseTensor<T> sub(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   return apply_binary_op<T, SubTag>(x, y);
}

template <typename T>
DenseTensor<T> mul(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   return apply_binary_op<T, MulTag>(x, y);
}

template <typename T>
DenseTensor<T> div(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   return apply_binary_op<T, DivTag>(x, y);
}

template <typename T>
DenseTensor<T> pow(const DenseTensor<T> &x, const DenseTensor<T> &y) {
   return apply_binary_op<T, PowTag>(x, y);
}

template <typename T> DenseTensor<T> reciprocal(const DenseTensor<T> &x) {
   return apply_unary_op<T, ReciprocalTag>(x);
}

template <typename T>
void sub_inplace(DenseTensor<T> &x, const DenseTensor<T> &y) {
   // TODO: need to impl_ a way to ignore batch dim in shape check in
   // a sensible way
   // UNSAFE CODE
   planning::BinaryEwiseContext ctx{};
   ctx.out_shape = x.shape();
   ctx.fast_len = x.flat_size();
   ctx.exec = planning::BinaryExecKind::FlatContiguous;
   // this impl is unstable for > rank(2) NDtensors
   FUSION_CHECK(ctx.out_shape[x.rank() - 1] == x.shape()[x.rank() - 1],
                "sub_inplace would change tensor shape; "
                "use out-of-place sub() instead.");

   execution::cpu::binary_elementwise<T, SubTag>(x.get_ptr(), x.get_ptr(), y.get_ptr(), ctx);
}

} // namespace fusion::ops::aligned

#endif // FUSION_OPS_ALIGNED_ARITHMETIC_HPP
