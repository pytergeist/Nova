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

template <typename T>
class TensorView {
 public:
   TensorView() = default;
   TensorView(void *data, std::vector<size_t> shape, std::vector<size_t> strides, std::size_t rank,
              std::size_t ndims, DType dtype = DType::Float32)
       : data_(data), shape_(shape), strides_(strides), rank_(rank), ndims_(ndims), dtype_(dtype) {};

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

   inline bool is_contiguous() const noexcept {
      if (shape_.empty()) {return true;}
      std::size_t expected = 1;
      for (std::size_t i = 0; i < shape_.size(); ++i) {
         if (strides_[i] != expected) {return false;}
         expected *= shape_[(i + 1 < shape_.size()) ? i + 1 : i];
         if (i + 1 == shape_.size()) expected = 1;
         }
      return true;
   }

 private:
   void* data_ = nullptr;
   std::vector<std::size_t> shape_;
   std::vector<std::size_t> strides_;
   std::size_t rank_;
   std::size_t ndims_;
   DType dtype_;
};


#endif // TENSOR_VIEW_H
