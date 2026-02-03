#ifndef OPS_LINALG_HPP
#define OPS_LINALG_HPP

#include <string_view>
#include <vector>

#include "Fusion/common/Log.hpp"

#include "Fusion/core/TensorIter.hpp"
#include "Fusion/core/PlanMeta.hpp"
#include "Fusion/core/RawTensor.hpp"

#include "Helpers.hpp"

namespace fusion {

namespace math {

namespace linalg {

inline EinsumBinding make_matmul_binding(std::size_t a_nd, std::size_t b_nd) {
   // require at least 2D
   if (a_nd < 2 || b_nd < 2) {
      throw std::runtime_error("matmul: expected rank >= 2 for both operands");
   }

   // For now require same batch rank. If you want broadcast batch ranks
   // you can pad the smaller with leading singleton axes in desc building,
   // or construct labels with left-padding logic.
   const std::size_t batch_nd_a = a_nd - 2;
   const std::size_t batch_nd_b = b_nd - 2;
   if (batch_nd_a != batch_nd_b) {
      throw std::runtime_error("matmul: batch rank mismatch (implement broadcasting/padding)");
   }

   const std::size_t batch_nd = batch_nd_a;

   // Label assignment:
   // batch dims: 0..batch_nd-1
   // i: batch_nd
   // j: batch_nd+1
   // k: batch_nd+2
   const Label base = 0;
   const Label Li = static_cast<Label>(base + batch_nd);
   const Label Lj = static_cast<Label>(base + batch_nd + 1);
   const Label Lk = static_cast<Label>(base + batch_nd + 2);

   std::vector<Label> batch_labels(batch_nd);
   for (std::size_t t = 0; t < batch_nd; ++t) batch_labels[t] = static_cast<Label>(base + t);

   // A labels: [batch..., i, k]
   std::vector<Label> a_labels = batch_labels;
   a_labels.push_back(Li);
   a_labels.push_back(Lk);

   // B labels: [batch..., k, j]
   std::vector<Label> b_labels = batch_labels;
   b_labels.push_back(Lk);
   b_labels.push_back(Lj);

   // out labels: [batch..., i, j]
   std::vector<Label> out_labels = batch_labels;
   out_labels.push_back(Li);
   out_labels.push_back(Lj);

   EinsumBinding binding;
   // IMPORTANT: your contraction planner expects {out, A, B} labels in compute_roles_for_gemm_like
   binding.op_axis_labels = {out_labels, a_labels, b_labels};
   binding.out_labels = out_labels;
   return binding;
}

template <typename T>
inline RawTensor<T> matmul(const RawTensor<T>& A, const RawTensor<T>& B) {
   FUSION_CHECK(A.is_initialised(), "matmul: A uninitialised");
   FUSION_CHECK(B.is_initialised(), "matmul: B uninitialised");
   FUSION_CHECK(A.dtype() == B.dtype(), "matmul: dtype mismatch");
   FUSION_CHECK(A.device() == B.device(), "matmul: device mismatch");

   const auto& a_shape = A.shape();
   const auto& b_shape = B.shape();
   if (a_shape.size() < 2 || b_shape.size() < 2)
      throw std::runtime_error("matmul: expected rank >= 2");

   const std::size_t kA = a_shape[a_shape.size() - 1];
   const std::size_t kB = b_shape[b_shape.size() - 2];
   if (kA != kB)
      throw std::runtime_error("matmul: inner dimension mismatch");

   EinsumBinding binding = make_matmul_binding(a_shape.size(), b_shape.size());
   ContractionMeta meta = make_contraction_meta_einsum<T>(A, B, binding);

   RawTensor<T> out = init_out_from_meta(A, B, meta);

   // One call. BLAS fastpath happens inside contraction_tag if meta.plan.gemm_like etc.
   fusion::iter::contraction_tag<
       T,
       BatchedGemmBLAS,  // BLAS backend tag
       MultiplySIMD                    // scalar fallback tag for generic contraction
   >(A, B, meta, out);

   return out;
}

template <typename T>
inline RawTensor<T> swapaxes(const RawTensor<T> &x, const int axis1,
                             const int axis2) {
   std::vector<size_t> out_shape = x.shape();
   const int nd = static_cast<int>(out_shape.size());
   if (nd < 2) {
      return RawTensor<T>(out_shape, std::vector<T>(x.begin(), x.end()),
                          x.dtype(), x.device());
   }
   const int naxis1 = serial::normalise_axis(axis1, nd);
   const int naxis2 = serial::normalise_axis(axis2, nd);
   if (naxis1 == naxis2) {
      return RawTensor<T>(out_shape, std::vector<T>(x.begin(), x.end()),
                          x.dtype(), x.device());
   }
   std::swap(out_shape[naxis1], out_shape[naxis2]);
   std::vector<T> out = serial::swapaxes<T>(x, x.shape(), naxis1, naxis2);
   return RawTensor<T>(std::move(out_shape), std::move(out), x.dtype(),
                       x.device());
}

} // namespace linalg

} // namespace math

} // namespace fusion

#endif // OPS_LINALG_HPP
