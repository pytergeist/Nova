#ifndef CORE_TRAITS_H
#define CORE_TRAITS_H

#include <type_traits>

template <typename U> class Tensor;

template <class T> struct is_tensor {
  static constexpr bool value = false;
};

template <class T> struct is_tensor<Tensor<T>> {
  static constexpr bool value = true;
};

template <class T>
constexpr bool is_tensor_t = is_tensor<std::remove_cvref_t<T>>::value;

#endif // CORE_TRAITS_H
