#include "Fusion/core/DType.h"
#include "Fusion/core/RawTensor.hpp"
#include "Fusion/core/Reduction.cpp"
#include "Fusion/core/Reduction.h"
#include "Fusion/core/TensorDesc.hpp"
#include "Fusion/device/Device.h"

std::string shape_str(std::vector<size_t> shape) {
   std::ostringstream oss;
   oss << '(';
   for (size_t i = 0; i < shape.size(); ++i) {
      oss << shape[i];
      if (i + 1 < shape.size())
         oss << ',';
   }
   oss << ')';
   return oss.str();
}

std::string stride_str(std::vector<std::int64_t> shape) {
   std::ostringstream oss;
   oss << '(';
   for (size_t i = 0; i < shape.size(); ++i) {
      oss << shape[i];
      if (i + 1 < shape.size())
         oss << ',';
   }
   oss << ')';
   return oss.str();
}

int main() {
   using T = float;
   RawTensor<T> A{
       {3, 2}, {2, 2, 2, 2, 2, 2}, DType::FLOAT32, Device{DeviceType::CPU, 0}};
   auto dA = make_desc<T>(A.shape(), nullptr);
   auto dB = make_desc<T>(A.shape(), nullptr);
   auto plan_in = make_reduction_plan({dA}, 1, true);
   auto dOut = make_desc<T>(plan_in.out_shape, nullptr);
   auto plan_b =
       make_reduction_plan(std::vector<TensorDescription>{dOut, dA}, 1, true);

   // -- plan output

   std::cout << "=== REDUCE 1st plan ===" << std::endl;

   std::cout << "plan num operands: " << plan_in.num_operands << std::endl;
   std::cout << "plan out_ndim: " << plan_in.out_ndim << std::endl;
   std::cout << "plan out shape: " << shape_str(plan_in.out_shape) << std::endl;
   std::cout << "plan reduction_axis: " << plan_in.reduction_axis << std::endl;
   std::cout << "---------------\n";

   for (auto p : plan_in.loop) {
      std::cout << "loop size: " << p.size << std::endl;
      std::cout << "loop stride bytes: " << stride_str(p.stride_bytes)
                << std::endl;
   }

   std::cout << "---------------\n";
   std::cout << "plan all_contiguous_like: " << plan_b.all_contiguous_like
             << std::endl;
   std::cout << "plan vector_bytes: " << plan_b.vector_bytes << std::endl;
   std::cout << "---------------\n";
   std::cout << "plan itemsize: " << plan_b.itemsize << std::endl;

   std::cout << "=== Reduce 2nd plan ===" << std::endl;

   std::cout << "----------------\n";

   std::cout << "plan num operands: " << plan_b.num_operands << std::endl;
   std::cout << "plan output ndims: " << plan_b.out_ndim << std::endl;
   std::cout << "plan output shape: " << shape_str(plan_b.out_shape)
             << std::endl;
   std::cout << "plan output itemsize: " << plan_b.itemsize << std::endl;

   std::cout << "---------------\n";

   for (auto p : plan_b.loop) {
      std::cout << "loop size: " << p.size << std::endl;
      std::cout << "loop stride bytes: " << stride_str(p.stride_bytes)
                << std::endl;
   }

   std::cout << "---------------\n";
   std::cout << "plan all_contiguous_like: " << plan_b.all_contiguous_like
             << std::endl;
   std::cout << "plan vector_bytes: " << plan_b.vector_bytes << std::endl;
   std::cout << "---------------\n";
   std::cout << "plan itemsize: " << plan_b.itemsize << std::endl;

   return 0;
}
