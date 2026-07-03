#ifndef FUSION_OPS_ALIGNED_UNARY_HPP
#define FUSION_OPS_ALIGNED_UNARY_HPP

#include <string_view>
#include <vector>

#include "Fusion/core/iter/TensorIter.hpp"
#include "Fusion/core/planning/OpContextBuilders.h"

namespace fusion::ops::aligned {

template <typename T> DenseTensor<T> sqrt(const DenseTensor<T> &x) {
   require_ewise_unary_out_of_place<SqrtTag>();
   planning::UnaryEwiseContext meta = planning::make_unary_ewise_context(x);
   DenseTensor<T> out = init_out_from_meta(x, meta);
   dense::iter::unary_ewise_tag<T, SqrtSIMD>(x, meta, out);
   return out;
}

template <typename T> DenseTensor<T> log(const DenseTensor<T> &x) {
   require_ewise_unary_out_of_place<LogTag>();
   planning::UnaryEwiseContext meta = planning::make_unary_ewise_context(x);
   DenseTensor<T> out = init_out_from_meta(x, meta);
   dense::iter::unary_ewise_tag<T, NaturalLogSIMD>(x, meta, out);
   return out;
}

template <typename T> DenseTensor<T> exp(const DenseTensor<T> &x) {
   require_ewise_unary_out_of_place<ExpTag>();
   planning::UnaryEwiseContext meta = planning::make_unary_ewise_context(x);
   DenseTensor<T> out = init_out_from_meta(x, meta);
   dense::iter::unary_ewise_tag<T, ExponentialSIMD>(x, meta, out);
   return out;
}

} // namespace fusion::ops::aligned

#endif // FUSION_OPS_ALIGNED_UNARY_HPP
