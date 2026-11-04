#ifndef LAYOUT_HPP
#define LAYOUT_HPP

#include <vector>

namespace fusion::core {

enum class LayoutKind : std::int8_t { Dense, Strided, SoA, AoSoA };

inline bool calc_contiguous(const std::vector<std::size_t> &shape,
                            const std::vector<std::int64_t> &strides) noexcept {
   const std::size_t nd = shape.size();
   if (nd == 0) {
      return true;
   }

   std::int64_t expected = 1;
   for (std::size_t i = nd; i-- > 0;) {
      if (shape[i] == 1) {
         continue;
      }
      if (strides[i] != expected) {
         return false;
      }
      expected *= static_cast<std::int64_t>(shape[i]);
   }
   return true;
}
} // namespace fusion::core
#endif // LAYOUT_HPP
