#ifndef ELEMENT_WISE_H
#define ELEMENT_WISE_H

#pragma once
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include "../cpu/SimdTraits.h"
#include "Broadcast.h"
#include "../common/Checks.h"

namespace ewise {
inline std::vector<int64_t>
contig_elem_strides(const std::vector<size_t> &shape) {
  std::vector<int64_t> st(shape.size());
  int64_t r = 1;
  for (int i = (int)shape.size() - 1; i >= 0; --i) {
    st[i] = r;
    r *= (int64_t)shape[i];
  }
  return st;
}

template <typename T>
inline TensorDescription make_desc(const std::vector<size_t> &shape,
                                   const int64_t *strides_elems) {
  // Create TensorDescription with ndims (shape.size()), int64_t vector of sizes
  // (shape), strides is stride_elems is not a nullptr, and itemsize
  std::vector<int64_t> sz(shape.begin(), shape.end());
  std::vector<int64_t> st;
  if (strides_elems)
    st.assign(strides_elems, strides_elems + static_cast<int>(shape.size()));
  else
    st = contig_elem_strides(shape);
  return TensorDescription{static_cast<int>(shape.size()), std::move(sz),
                           std::move(st), sizeof(T)};
}

template <typename FnInnermost>
inline void for_each_outer_then_inner(const BroadcastPlan &plan,
                                      const std::vector<uint8_t *> &bases,
                                      FnInnermost &&inner) {
  // first set the ndim (2 usually, for 2 tensors in loop)
  // set base ptrs (size=3 here, a, b, out?)
  const int ndim = static_cast<int>(plan.loop.size());
  std::vector<uint8_t *> ptr = bases;

  if (ndim == 0) {
    // if ndim = 0 set s vector to num_operands(0)
    // e.g. if num_operands = 3, s = {0, 0, 0}, we then pass into
    // the inner func
    std::vector<int64_t> s(plan.num_operands, 0);
    inner(ptr, 1, s);
    return;
  }

  const int inn = ndim - 1;

  std::function<void(int)> walk = [&](int dim) {
    if (dim == inn) {
      const auto &ld = plan.loop[inn];
      std::vector<int64_t> s = ld.stride_bytes;
      inner(ptr, ld.size, s);
      return;
    }
    const auto &ld = plan.loop[dim];
    for (int64_t i = 0; i < ld.size; ++i) {
      walk(dim + 1);
      for (int k = 0; k < plan.num_operands; ++k)
        ptr[k] += ld.stride_bytes[k];
    }
    for (int k = 0; k < plan.num_operands; ++k)
      ptr[k] -= ld.stride_bytes[k] * ld.size;
  };

  walk(0);
}

// Tag = AddSIMD / SubtractSIMD / ...
template <typename T, class Tag, class TensorT>
void binary_ewise_tag(const TensorT &A, const TensorT &B,
                      std::vector<size_t> &out_shape,
                      std::vector<T> &out_data) {
  // Initialise tensor descriptions with shape and stride
  FUSION_CHECK(A.is_initialised() && B.is_initialised(), "uninitialised tensor");
//  FUSION_CHECK(A.size() == B.size(), "size mismatch in binary op"); // TODO: this only works for same lenght ops
                                                                    // include scalar operations
  auto dA = make_desc<T>(A.shape_, nullptr);
  auto dB = make_desc<T>(B.shape_, nullptr);
  auto plan_in = make_broadcast_plan({dA, dB});

  out_shape.assign(plan_in.out_sizes.begin(), plan_in.out_sizes.end());
  size_t n_out = std::accumulate(out_shape.begin(), out_shape.end(),
                                 static_cast<size_t>(1), std::multiplies<>());
  out_data.resize(n_out); // TODO: why resize here?

  auto dOut = make_desc<T>(out_shape, nullptr);
  auto plan = make_broadcast_plan({dOut, dA, dB});

  std::vector<uint8_t *> base(3);
  base[0] = reinterpret_cast<uint8_t *>(out_data.data());
  base[1] = reinterpret_cast<uint8_t *>(const_cast<T *>(A.storage->data_ptr()));
  base[2] = reinterpret_cast<uint8_t *>(const_cast<T *>(B.storage->data_ptr()));

  for_each_outer_then_inner(
      plan, base,
      [&](std::vector<uint8_t *> &p, int64_t len,
          const std::vector<int64_t> &sbytes) {
        const auto step = static_cast<int64_t>(
            sizeof(T)); // sizeof(T) = size of datatype (n bytes)
        const bool out_contig =
            (sbytes[0] == step); // true if s[0] == step (e.g. bytes)
        // below here means that a/b must be either 0 (for broadcast) or same
        // size as step (e.g. bytes)
        const bool a_ok =
            (sbytes[1] == 0 ||
             sbytes[1] == step); // a_ok = True if s[1] = 0 or s[1] = bytes (4_
        const bool b_ok =
            (sbytes[2] == 0 ||
             sbytes[2] == step); // b_ok = True if s[1] = 0 or s[1] = bytes (4_

        auto *o = reinterpret_cast<T *>(
            p[0]); // takes bytes ptr and treats it as if it were a ptr to T
        // (dtype from template)
        const auto *a = reinterpret_cast<const T *>(p[1]); // same as above
        const auto *b = reinterpret_cast<const T *>(p[2]); // same as above

        if constexpr (simd_traits<Tag, T>::available) {
          // include simd impl
          // from traits availible
          if (out_contig && a_ok && b_ok && len > 0) {
            // if all true continue
            const bool a_scalar =
                (sbytes[1] == 0); // true (a is scalar) if s[1] == 0 (broadcast)
            const bool b_scalar =
                (sbytes[2] == 0); // true (b is scalar) if s[1] == 0 (broadcast)
            // if all above true execute the contigous op from simd_traits
            simd_traits<Tag, T>::execute_contiguous(
                a, b, o, static_cast<size_t>(len), a_scalar, b_scalar);
            return;
          }
        }

        const int64_t so = sbytes[0] / step;
        const int64_t sa = (sbytes[1] == 0) ? 0 : sbytes[1] / step;
        const int64_t sb = (sbytes[2] == 0) ? 0 : sbytes[2] / step;
        // uses tag struct from simd tags as fallback
        Tag tag{};
        for (int64_t i = 0; i < len; ++i)
          o[i * so] = tag(a[i * sa], b[i * sb]);
      });
  // return TensorT(std::move(out_shape), std::move(out_data), Device::CPU);
}

// Tag = ExponentialSIMD / NaturalLogSIMD / ...
template <typename T, class Tag, class TensorT>
void unary_ewise_tag(const TensorT &A, std::vector<size_t> &out_shape,
                     std::vector<T> &out_data) {
  // Initialise tensor descriptions with shape and stride
  auto dA = make_desc<T>(A.shape_, nullptr);
  auto plan_in = make_broadcast_plan({dA});

  out_shape.assign(plan_in.out_sizes.begin(), plan_in.out_sizes.end());
  size_t n_out = std::accumulate(out_shape.begin(), out_shape.end(),
                                 static_cast<size_t>(1), std::multiplies<>());
  out_data.resize(n_out);

  auto dOut = make_desc<T>(out_shape, nullptr);
  auto plan = make_broadcast_plan({dOut, dA});

  std::vector<uint8_t *> base(3);
  base[0] = reinterpret_cast<uint8_t *>(out_data.data());
  base[1] = reinterpret_cast<uint8_t *>(const_cast<T *>(A.storage->data_ptr()));

  for_each_outer_then_inner(
      plan, base,
      [&](std::vector<uint8_t *> &p, int64_t len,
          const std::vector<int64_t> &sbytes) {
        const auto step = static_cast<int64_t>(
            sizeof(T)); // sizeof(T) = size of datatype (n bytes)
        const bool out_contig =
            (sbytes[0] == step); // true if s[0] == step (e.g. bytes)
        // below here means that a/b must be either 0 (for broadcast) or same
        // size as step (e.g. bytes)
        const bool a_ok =
            (sbytes[1] == 0 ||
             sbytes[1] == step); // a_ok = True if s[1] = 0 or s[1] = bytes (4_

        auto *o = reinterpret_cast<T *>(
            p[0]); // takes bytes ptr and treats it as if it were a ptr to T
        // (dtype from template)
        const auto *a = reinterpret_cast<const T *>(p[1]); // same as above

        if constexpr (simd_traits<Tag, T>::available) {
          // include simd impl
          // from traits availible
          if (out_contig && a_ok && len > 0) {
            // if all true continue
            const bool a_scalar =
                (sbytes[1] == 0); // true (a is scalar) if s[1] == 0 (broadcast)
            // if all above true execute the contigous op from simd_traits
            simd_traits<Tag, T>::execute_contiguous(
                a, o, static_cast<size_t>(len), a_scalar);
            return;
          }
        }

        const int64_t so = sbytes[0] / step;
        const int64_t sa = (sbytes[1] == 0) ? 0 : sbytes[1] / step;
        // uses tag struct from simd tags as fallback
        Tag tag{};
        for (int64_t i = 0; i < len; ++i)
          o[i * so] = tag(a[i * sa]);
      });
}
} // namespace ewise

#endif // ELEMENT_WISE_H
