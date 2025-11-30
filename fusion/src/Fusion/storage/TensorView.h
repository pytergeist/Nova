#ifndef TENSOR_VIEW_H
#define TENSOR_VIEW_H

#include <cstddef>
#include <memory>

#include "Fusion/core/DTypes.h"
#include "Fusion/core/Layout.h"


template <typename T> class TensorView {
 public:
   TensorView() = default;
   TensorView(T *data, std::vector<size_t> shape,
              std::vector<size_t> strides, std::size_t rank, std::size_t ndims,
              DType dtype = DType::Float32)
       : data_(data), shape_(shape), strides_(strides), rank_(rank),
         ndims_(ndims), dtype_(dtype) {};

   TensorView(const TensorView &) = delete;
   TensorView &operator=(const TensorView &) = delete;
   TensorView(TensorView &&) noexcept = delete;
   TensorView &operator=(TensorView &&) noexcept = delete;

   const std::vector<std::size_t> shape() const noexcept { return shape_; }
   std::vector<std::size_t> shape() noexcept { return shape_; }

   const std::vector<std::size_t> strides() const noexcept { return strides_; }
   std::vector<std::size_t> strides() noexcept { return strides_; }

   const std::size_t rank() const noexcept { return rank_; }
   std::size_t rank() noexcept { return rank_; }

   const std::size_t ndims() const noexcept { return ndims_; }
   std::size_t ndims() noexcept { return ndims_; }

   T *data() noexcept { return data_; };
   const T *data() const noexcept { return data_; };

   inline bool is_contiguous() const noexcept {
      return calc_contiguous(shape_, strides_);
   }

 private:
   T *data_ = nullptr;
   std::vector<std::size_t> shape_;
   std::vector<std::size_t> strides_;
   std::size_t rank_;
   std::size_t ndims_;
   DType dtype_;
};

#endif // TENSOR_VIEW_H
