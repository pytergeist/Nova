#ifndef VEC_LOOP_HPP
#define VEC_LOOP_HPP

#include <cstddef>

namespace simd {

namespace detail {

template <typename T, class Backend, class BinaryVecOp, class BinaryScalarOp>
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

template <typename T, class Backend, class BinaryVecOp, class BinaryScalarOp>
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
   vec vb = vdupq_n_f32(b);
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

template <typename T, class Backend, class BinaryVecOp, class BinaryScalarOp>
void unary_contiguous_apply(T *__restrict dst, const T *__restrict a,
                            std::size_t n, BinaryVecOp vec_op,
                            BinaryScalarOp scalar_op) {

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

} // namespace detail

} // namespace simd

#endif // VEC_LOOP_HPP
