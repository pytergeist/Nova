#ifndef FUSION_CORE_FUIR_DESCS_H
#define FUSION_CORE_FUIR_DESCS_H

#include <cstdint>
#include <vector>

/// TensorDescription stores the meta-data of a single operand participating in
/// an expression.
struct TensorDescription {
   /// shape of operand tensor
   std::vector<std::size_t> shape;
   /// strides of operand tensor (if applicable)
   std::vector<std::int64_t> strides;
   /// size in bytes of operand dtype
   std::size_t itemsize;

   std::size_t ndims() const noexcept { return shape.size(); }
};

#endif // FUSION_CORE_FUIR_DESCS_H