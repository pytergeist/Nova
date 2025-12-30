#ifndef EWISE_HPP
#define EWISE_HPP

#include <string_view>
#include <vector>

#include "Fusion/core/EwiseIter.hpp"
#include "Fusion/core/EwiseMeta.hpp"
#include "Fusion/core/RawTensor.hpp"

#include "Helpers.hpp"

namespace fusion {

namespace math {

template <typename T>
inline RawTensor<T> add(const RawTensor<T> &x, const RawTensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, AddSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline RawTensor<T> sub(const RawTensor<T> &x, const RawTensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, SubtractSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline RawTensor<T> mul(const RawTensor<T> &x, const RawTensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, MultiplySIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline RawTensor<T> div(const RawTensor<T> &x, const RawTensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, DivideSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline RawTensor<T> pow(const RawTensor<T> &x, const RawTensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   ewise::binary_ewise_tag<T, PowerSIMD>(x, y, meta, out);
   return out;
}

std::string shape_str(std::vector<size_t> shape) {
   std::ostringstream oss;
   oss << '(';
   for (size_t i = 0; i < shape.size(); ++i) {
      oss << shape[i];
      if (i + 1 < shape.size())
         oss << ',';
   }
   oss << ')';
   return oss.str();
}

template <typename T>
inline void sub_inplace(RawTensor<T> &x, const RawTensor<T> &y) {
   // TODO: need to impl_ a way to ignore batch dim in shape check in
   // a sensible way
   BinaryEwiseMeta meta{};
   meta.fastpath = true;
   meta.out_shape = x.shape();
   meta.fast_len = x.flat_size();
   // this impl is unstable for > rank(2) NDtensors
   FUSION_CHECK(meta.out_shape[x.rank() - 1] == x.shape()[x.rank() - 1],
                "sub_inplace would change tensor shape; "
                "use out-of-place sub() instead.");

   ewise::binary_ewise_tag<T, SubtractSIMD>(x, y, meta, x);
}

} // namespace math

} // namespace fusion

#endif // EWISE_H
