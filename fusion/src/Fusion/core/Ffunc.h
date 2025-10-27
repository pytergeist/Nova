#ifndef FFUNC_H
#define FFUNC_H

#include "Traits.h"
#include <numeric> // for accumulate

template <typename operand> auto subscript(operand const &t, auto const i) {
   if constexpr (is_tensor_t<operand>) {
      if (t.flat_size() == 1) {
         return t[0];
      } else {
         return t[i];
      }
   } else {
      return t;
   }
}

// note: no more raw‐data pointers here
template <class callable, class... operands> class FFunc {
   callable func_;
   std::tuple<operands const &...> args_;
   std::vector<size_t> shape_;

 public:
   explicit FFunc(callable func, std::vector<size_t> shape,
                  operands const &...args)
       : func_(func), args_(args...), shape_(std::move(shape)) {}

   auto const &shape() const { return shape_; }

   size_t flat_size() const {
      return std::accumulate(shape_.begin(), shape_.end(), size_t{1},
                             std::multiplies<>());
   }

   auto operator[](size_t i) const {
      return std::apply(
          [&](auto const &...as) {
             // each 'as' is either a Tensor<T>& or a scalar
             // subscript(as, i) does: tensor[i] or scalar
             return func_(subscript(as, i)...);
          },
          args_);
   }

   template <size_t I> auto const &operand() const {
      return std::get<I>(args_);
   }
};

#endif // FFUNC_H
