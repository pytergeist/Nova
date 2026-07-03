#ifndef FUSION_CORE_PLANNING_OPERAND_DESC_BUILDERS_H
#define FUSION_CORE_PLANNING_OPERAND_DESC_BUILDERS_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Fusion/core/fuir/IR.h"

template <typename T>
class DenseTensor;

template <typename T>
class AoSoATensor;

namespace fusion::planning {

inline std::vector<std::int64_t>
contig_elem_strides(const std::vector<std::size_t>& shape) {
   std::vector<std::int64_t> strides(shape.size());

   std::int64_t running = 1;
   for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      const auto idx = static_cast<std::size_t>(i);
      strides[idx] = running;
      running *= static_cast<std::int64_t>(shape[idx]);
   }

   return strides;
}

template <typename T>
OperandDescription
make_desc_from_shape(const std::vector<std::size_t>& shape,
                     const std::int64_t* strides_elems) {
   OperandDescription desc;
   desc.shape.assign(shape.begin(), shape.end());

   if (strides_elems != nullptr) {
      desc.strides.assign(strides_elems,
                          strides_elems + static_cast<std::ptrdiff_t>(
                                              shape.size()));
   } else {
      desc.strides = contig_elem_strides(shape);
   }

   desc.itemsize = sizeof(T);
   desc.access = AccessKind::Affine;
   desc.layout = LayoutKind::Dense;
   desc.storage = StorageKind::Owned;
   desc.update = UpdateKind::ReadOnly;
   desc.type = OperandDescType::Tensor;

   return desc;
}

template <typename T>
OperandDescription
make_desc_from_tensor(const DenseTensor<T>& tensor) {
   OperandDescription desc;
   desc.shape = tensor.shape();
   desc.itemsize = tensor.dtype_size();

   if constexpr (requires { tensor.strides(); }) {
      desc.strides = tensor.strides();
   } else {
      desc.strides = contig_elem_strides(desc.shape);
   }

   desc.access = AccessKind::Affine;
   desc.layout = tensor.is_contiguous() ? LayoutKind::Dense
                                        : LayoutKind::Strided;
   desc.storage = !tensor.is_view() ? StorageKind::Owned
                                    : StorageKind::View;
   desc.update = UpdateKind::ReadOnly;
   desc.type = OperandDescType::Tensor;

   return desc;
}

template <typename T>
OperandDescription
make_desc_from_aosoa_tensor(const AoSoATensor<T>& tensor) {
   OperandDescription desc;
   desc.shape = tensor.logical_shape();
   desc.itemsize = tensor.base().dtype_size();

   if constexpr (requires { tensor.strides(); }) {
      desc.strides = tensor.strides();
   } else {
      desc.strides = contig_elem_strides(desc.shape);
   }

   desc.access = AccessKind::Blocked;
   desc.layout = LayoutKind::AoSoA;
   desc.storage = StorageKind::Owned;
   desc.update = UpdateKind::ReadOnly;
   desc.type = OperandDescType::Tensor;

   return desc;
}

} // namespace fusion::planning

#endif // FUSION_CORE_PLANNING_OPERAND_DESC_BUILDERS_H