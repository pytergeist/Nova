#ifndef FUSION_OPS_OPERAND_VALIDATION_H
#define FUSION_OPS_OPERAND_VALIDATION_H

#include <string_view>

#include "Fusion/common/error/Check.h"
#include "Fusion/opschema/OpRequirements.h"

// TODO: future state - ensure dtype invariant at tensor construction, not in
// semantic operations


template <typename T>
class DenseTensor;

namespace fusion::ops::validation {

namespace detail {

template <typename T>
void validate_binary_dtype_and_device(const DenseTensor<T> &lhs,
                                      const DenseTensor<T> &rhs,
                                      std::string_view op_name) {

   FUSION_CHECK(lhs.dtype() == rhs.dtype(),
                error::message(op_name, ": operand dtypes do not match"));

   FUSION_CHECK(lhs.device() == rhs.device(),
                error::message(op_name, ": operand devices do not match"));
}

template <typename T>
void validate_binary_operand_state(const DenseTensor<T> &lhs,
                                   const DenseTensor<T> &rhs,
                                   std::string_view op_name) {

   FUSION_CHECK_PRECONDITION(
       lhs.is_initialised(),
       error::message(op_name, ": lhs operand is uninitialised"));

   FUSION_CHECK_PRECONDITION(
       rhs.is_initialised(),
       error::message(op_name, ": rhs operand is uninitialised"));
}

template <typename T>
void validate_unary_operand_state(const DenseTensor<T> &operand,
                                  std::string_view op_name) {

   FUSION_CHECK_PRECONDITION(
       operand.is_initialised(),
       error::message(op_name, ": operand is uninitialised"));
}

template <typename T>
void validate_dtype_invariant(const DenseTensor<T> &operand,
                              std::string_view operand_name,
                              std::string_view op_name) {

   FUSION_INTERNAL_ASSERT(
       operand.dtype() == dtype_for<T>(),
       error::message(
           op_name, ": ", operand_name,
           " runtime dtype does not match its template element type"));
}

} // namespace detail

template <typename T, class Tag>
void validate_dense_binary_operation(const DenseTensor<T> &lhs,
                                     const DenseTensor<T> &rhs) {

   opschema::require_ewise_binary_out_of_place<Tag>();
   constexpr std::string_view op_name = OpTraits<Tag>::name;

   detail::validate_binary_operand_state(lhs, rhs, op_name);
   detail::validate_dtype_invariant(lhs, "lhs", op_name);
   detail::validate_dtype_invariant(rhs, "rhs", op_name);
   detail::validate_binary_dtype_and_device(lhs, rhs, op_name);
}

template <typename T, class Tag>
void validate_dense_unary_operation(const DenseTensor<T> &operand) {
   opschema::require_ewise_unary_out_of_place<Tag>();
   constexpr std::string_view op_name = OpTraits<Tag>::name;

   detail::validate_unary_operand_state(operand, op_name);
   detail::validate_dtype_invariant(operand, "operand", op_name);
}

template <typename T, class Tag>
void validate_dense_reduction_operation(const DenseTensor<T> &operand) {
   opschema::require_reduction_out_of_place<Tag>();
   constexpr std::string_view op_name = OpTraits<Tag>::name;

   detail::validate_unary_operand_state(operand, op_name);
   detail::validate_dtype_invariant(operand, "operand", op_name);
}

template <typename T, class Tag>
void validate_dense_contraction_operation(const DenseTensor<T> &lhs,
                                          const DenseTensor<T> &rhs) {

   opschema::require_contraction_out_of_place<Tag>();
   constexpr std::string_view op_name = OpTraits<Tag>::name;

   detail::validate_binary_operand_state(lhs, rhs, op_name);
   detail::validate_dtype_invariant(lhs, "lhs", op_name);
   detail::validate_dtype_invariant(rhs, "rhs", op_name);
   detail::validate_binary_dtype_and_device(lhs, rhs, op_name);
}

} // namespace fusion::ops::validation

#endif // FUSION_OPS_OPERAND_VALIDATION_H