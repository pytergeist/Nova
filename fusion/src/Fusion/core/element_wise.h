#ifndef ELEMENT_WISE_H
#define ELEMENT_WISE_H

#pragma once
#include <numeric>
#include <vector>
#include <functional>
#include <cstdint>

#include "broadcast.h"
#include "../cpu/simd_traits.h"

namespace ewise {

inline std::vector<int64_t>
contig_elem_strides(const std::vector<size_t>& shape) {
  std::vector<int64_t> st(shape.size());
  int64_t r = 1;
  for (int i = (int)shape.size()-1; i>=0; --i) { st[i] = r; r *= (int64_t)shape[i]; }
  return st;
}

template <typename T>
inline TensorDescription make_desc(const std::vector<size_t>& shape,
                                   const int64_t* strides_elems /*or nullptr for contig*/) {
  std::vector<int64_t> sz(shape.begin(), shape.end());
  std::vector<int64_t> st;
  if (strides_elems) st.assign(strides_elems, strides_elems + (int)shape.size());
  else               st = contig_elem_strides(shape);
  return TensorDescription{ (int)shape.size(), std::move(sz), std::move(st), sizeof(T) };
}

// Walk all outer dims, run the innermost as a tight loop
template <typename FnInnermost>
inline void for_each_outer_then_inner(const BroadcastPlan& plan,
                                      const std::vector<uint8_t*>& bases,
                                      FnInnermost&& inner) {
  const int ndim = (int)plan.loop.size();
  std::vector<uint8_t*> ptr = bases;

  if (ndim == 0) {
    std::vector<int64_t> s(plan.num_operands, 0);
    inner(ptr, 1, s);
    return;
  }

  const int inn = ndim - 1;

  std::function<void(int)> walk = [&](int dim) {
    if (dim == inn) {
      const auto& ld = plan.loop[inn];
      std::vector<int64_t> s = ld.stride_bytes;
      inner(ptr, ld.size, s);
      return;
    }
    const auto& ld = plan.loop[dim];
    for (int64_t i=0;i<ld.size;++i) {
      walk(dim+1);
      for (int k=0;k<plan.num_operands;++k) ptr[k] += ld.stride_bytes[k];
    }
    for (int k=0;k<plan.num_operands;++k) ptr[k] -= ld.stride_bytes[k]*ld.size;
  };

  walk(0);
}

// Tag = AddSIMD / SubtractSIMD / ...
template <typename T, class Tag, class TensorT>
TensorT binary_ewise_tag(const TensorT& A, const TensorT& B) {
  // 1) input plans (we’ll treat A,B as contiguous for now; replace if you have real strides)
  auto dA = make_desc<T>(A.shape_, /*strides*/nullptr);
  auto dB = make_desc<T>(B.shape_, /*strides*/nullptr);
  auto plan_in = make_broadcast_plan({dA, dB});

  std::vector<size_t> out_shape(plan_in.out_sizes.begin(), plan_in.out_sizes.end());
  size_t n_out = std::accumulate(out_shape.begin(), out_shape.end(), (size_t)1, std::multiplies<>());
  std::vector<T> out_data(n_out);

  auto dOut = make_desc<T>(out_shape, nullptr);
  auto plan = make_broadcast_plan({dOut, dA, dB});

  std::vector<uint8_t*> base(3);
  base[0] = reinterpret_cast<uint8_t*>(out_data.data());
  base[1] = reinterpret_cast<uint8_t*>(const_cast<T*>(A.storage->data_ptr()));
  base[2] = reinterpret_cast<uint8_t*>(const_cast<T*>(B.storage->data_ptr()));

  for_each_outer_then_inner(plan, base,
    [&](std::vector<uint8_t*>& p, int64_t len, const std::vector<int64_t>& sbytes){
      const auto step = (int64_t)sizeof(T);
      const bool out_contig = (sbytes[0] == step);
      const bool a_ok = (sbytes[1] == 0 || sbytes[1] == step);
      const bool b_ok = (sbytes[2] == 0 || sbytes[2] == step);

      auto* o = reinterpret_cast<T*>(p[0]);
      const auto* a = reinterpret_cast<const T*>(p[1]);
      const auto* b = reinterpret_cast<const T*>(p[2]);

      if constexpr (simd_traits<Tag,T>::available) {
        if (out_contig && a_ok && b_ok && len > 0) {
          const bool a_scalar = (sbytes[1] == 0);
          const bool b_scalar = (sbytes[2] == 0);
          simd_traits<Tag,T>::execute_contiguous(a, b, o, (size_t)len, a_scalar, b_scalar);
          return;
        }
      }

      const int64_t so = sbytes[0] / step;
      const int64_t sa = (sbytes[1] == 0) ? 0 : sbytes[1] / step;
      const int64_t sb = (sbytes[2] == 0) ? 0 : sbytes[2] / step;

      Tag tag{};
      for (int64_t i=0;i<len;++i) o[i*so] = tag(a[i*sa], b[i*sb]);
    });

  return TensorT(std::move(out_shape), std::move(out_data), Device::CPU);
}

} // namespace ewise




#endif // ELEMENT_WISE_H
