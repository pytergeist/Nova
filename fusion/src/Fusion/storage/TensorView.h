#ifndef TENSOR_VIEW_H
#define TENSOR_VIEW_H

#include <cstddef>
#include <memory>

enum class DType { Float32, Float64, Int32, Int64 };

inline std::size_t dtype_size(DType dtype) {
   switch (dtype) {
   case DType::Float32:
      return sizeof(float);
   case DType::Float64:
      return sizeof(double);
   case DType::Int32:
      return sizeof(int32_t);
   case DType::Int64:
      return sizeof(int64_t);
   }
}

class TensorView {
 public:
   TensorView() = default;
   TensorView(void *data, std::vector<size_t> shape, std::size_t rank,
              std::size_t ndims, DType dtype = DType::Float32)
       : data_(data), shape_(shape), rank_(rank), ndims_(ndims), dtype_(dtype) {};

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

 private:
   void* data_;
   std::vector<std::size_t> shape_;
   std::vector<std::size_t> strides_;
   std::size_t rank_;
   std::size_t ndims_;
   DType dtype_;
};

#endif // TENSOR_VIEW_H
