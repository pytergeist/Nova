#ifndef FUSION_CPU_VEC_LOOP_HPP
#define FUSION_CPU_VEC_LOOP_HPP

#include <cstddef>

#include "BackendConcept.hpp"

namespace simd {

namespace detail {

template <typename T, BackendConcept Backend, class BinaryVecOp,
          class BinaryScalarOp>
void binary_contiguous_apply(T *__restrict dst, const T *__restrict a,
                             const T *__restrict b, std::size_t n,
                             BinaryVecOp vec_op, BinaryScalarOp scalar_op) {

   using B = Backend;
   using vec = typename B::vec;
   using wide_vec = typename B::wide_vec;

   constexpr std::size_t kBlock = B::kBlock;
   constexpr std::size_t kStep = B::kStep;

   const T *__restrict pa = a;
   const T *__restrict pb = b;
   T *__restrict pd = dst;

   FUSION_CONST_ASSUME_ALIGNED(T, pa, 64);
   FUSION_CONST_ASSUME_ALIGNED(T, pb, 64);
   FUSION_ASSUME_ALIGNED(T, pd, 64);

   std::size_t i = 0;

   for (; i + kBlock <= n; i += kBlock) {
      wide_vec va = B::wide_load(pa);
      pa += kBlock;
      wide_vec vb = B::wide_load(pb);
      pb += kBlock;

      va.val[0] = vec_op(va.val[0], vb.val[0]);
      va.val[1] = vec_op(va.val[1], vb.val[1]);
      va.val[2] = vec_op(va.val[2], vb.val[2]);
      va.val[3] = vec_op(va.val[3], vb.val[3]);

      B::wide_store(pd, va);
      pd += kBlock;
   }

   for (; i + B::kStep <= n; i += B::kStep) {
      vec va = B::load(pa);
      pa += kStep;
      vec vb = B::load(pb);
      pb += kStep;
      B::store(pd, vec_op(va, vb));
      pd += kStep;
   }
   for (; i < n; ++i)
      *pd++ = scalar_op(*pa++, *pb++);
};

template <typename T, BackendConcept Backend, class BinaryVecOp,
          class BinaryScalarOp>
void binary_contiguous_scalar_apply(T *__restrict dst, const T *__restrict a,
                                    const T b, std::size_t n,
                                    BinaryVecOp vec_op,
                                    BinaryScalarOp scalar_op) {

   using B = Backend;
   using vec = typename B::vec;
   using wide_vec = typename B::wide_vec;

   constexpr std::size_t kBlock = B::kBlock;
   constexpr std::size_t kStep = B::kStep;

   const T *__restrict pa = a;
   vec vb = B::duplicate(b);
   T *__restrict pd = dst;

   FUSION_CONST_ASSUME_ALIGNED(T, pa, 64);
   FUSION_ASSUME_ALIGNED(T, pd, 64);

   std::size_t i = 0;

   for (; i + kBlock <= n; i += kBlock) {
      wide_vec va = B::wide_load(pa);
      pa += kBlock;

      va.val[0] = vec_op(va.val[0], vb);
      va.val[1] = vec_op(va.val[1], vb);
      va.val[2] = vec_op(va.val[2], vb);
      va.val[3] = vec_op(va.val[3], vb);

      B::wide_store(pd, va);
      pd += kBlock;
   }

   for (; i + B::kStep <= n; i += B::kStep) {
      vec va = B::load(pa);
      pa += kStep;
      B::store(pd, vec_op(va, vb));
      pd += kStep;
   }
   for (; i < n; ++i)
      *pd++ = scalar_op(*pa++, b);
};

template <typename T, BackendConcept Backend, class UnaryVecOp,
          class UnaryScalarOp>
void unary_contiguous_apply(T *__restrict dst, const T *__restrict a,
                            std::size_t n, UnaryVecOp vec_op,
                            UnaryScalarOp scalar_op) {

   using B = Backend;
   using vec = typename B::vec;
   using wide_vec = typename B::wide_vec;

   constexpr std::size_t kBlock = B::kBlock;
   constexpr std::size_t kStep = B::kStep;

   const T *__restrict pa = a;
   T *__restrict pd = dst;

   FUSION_CONST_ASSUME_ALIGNED(T, pa, 64);
   FUSION_ASSUME_ALIGNED(T, pd, 64);

   std::size_t i = 0;

   for (; i + kBlock <= n; i += kBlock) {
      wide_vec va = B::wide_load(pa);
      pa += kBlock;

      va.val[0] = vec_op(va.val[0]);
      va.val[1] = vec_op(va.val[1]);
      va.val[2] = vec_op(va.val[2]);
      va.val[3] = vec_op(va.val[3]);

      B::wide_store(pd, va);
      pd += kBlock;
   }

   for (; i + B::kStep <= n; i += B::kStep) {
      vec va = B::load(pa);
      pa += kStep;
      B::store(pd, vec_op(va));
      pd += kStep;
   }
   for (; i < n; ++i)
      *pd++ = scalar_op(*pa++);
};

template <typename T, BackendConcept Backend, class BinaryVecOp, class ReduceOp,
          class ReduceScalarOp>
void reduce_contiguous_apply(T *__restrict dst, const T *__restrict a,
                             std::size_t n, BinaryVecOp vec_op,
                             ReduceOp reduce_op, ReduceScalarOp scalar_op) {

   using B = Backend;
   using vec = typename B::vec;
   using wide_vec = typename B::wide_vec;

   constexpr std::size_t kBlock = B::kBlock;
   constexpr std::size_t kStep = B::kStep;

   const T *__restrict pa = a;
   T *__restrict pd = dst;

   FUSION_CONST_ASSUME_ALIGNED(T, pa, 64);
   FUSION_ASSUME_ALIGNED(T, pd, 64);

   std::size_t i = 0;

   vec acc0 = B::duplicate(T(0));
   vec acc1 = B::duplicate(T(0));
   vec acc2 = B::duplicate(T(0));
   vec acc3 = B::duplicate(T(0));

   for (; i + kBlock <= n; i += kBlock) {
      wide_vec va = B::wide_load(pa);
      pa += kBlock;

      acc0 = vec_op(acc0, va.val[0]);
      acc1 = vec_op(acc1, va.val[1]);
      acc2 = vec_op(acc2, va.val[2]);
      acc3 = vec_op(acc3, va.val[3]);
   }

   vec acc = vec_op(vec_op(acc0, acc1), vec_op(acc2, acc3));

   for (; i + B::kStep <= n; i += B::kStep) {
      vec va = B::load(pa);
      pa += kStep;
      acc = vec_op(acc, va);
      pd += kStep;
   }

   T result = reduce_op(acc);

   for (; i < n; ++i)
      result = scalar_op(result, *pa++);

   *dst = result;
};

} // namespace detail

} // namespace simd

#endif // FUSION_CPU_VEC_LOOP_HPP
