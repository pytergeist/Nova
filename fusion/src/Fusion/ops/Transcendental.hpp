#ifndef OPS_TRANSCENENTAL_HPP
#define OPS_TRANSCENENTAL_HPP

#include <string_view>
#include <vector>

#include "Fusion/core/iter/TensorIter.hpp"
#include "Fusion/core/tensor/DenseTensor.hpp"

namespace fusion {

namespace math {

template <typename T> DenseTensor<T> sqrt(const DenseTensor<T> &x) {
   require_ewise_unary_out_of_place<SqrtTag>();
   UnaryEwiseMeta meta = make_unary_meta(x);
   DenseTensor<T> out = init_out_from_meta(x, meta);
   fusion::iter::unary_ewise_tag<T, SqrtSIMD>(x, meta, out);
   return out;
}

template <typename T> DenseTensor<T> log(const DenseTensor<T> &x) {
   require_ewise_unary_out_of_place<LogTag>();
   UnaryEwiseMeta meta = make_unary_meta(x);
   DenseTensor<T> out = init_out_from_meta(x, meta);
   fusion::iter::unary_ewise_tag<T, NaturalLogSIMD>(x, meta, out);
   return out;
}

template <typename T> DenseTensor<T> exp(const DenseTensor<T> &x) {
   require_ewise_unary_out_of_place<ExpTag>();
   UnaryEwiseMeta meta = make_unary_meta(x);
   DenseTensor<T> out = init_out_from_meta(x, meta);
   fusion::iter::unary_ewise_tag<T, ExponentialSIMD>(x, meta, out);
   return out;
}

} // namespace math

} // namespace fusion

#endif // OPS_TRANSCENENTAL_HPP
