#include <vector>
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <numeric>

inline void sort_edges_by_i_then_j(std::vector<std::uint32_t> &i,
                                   std::vector<std::uint32_t> &j) {
   assert(i.size() == j.size());
   const std::size_t m = i.size();

   std::vector<std::size_t> idx(m);
   std::iota(idx.begin(), idx.end(), 0);

   std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
      if (i[a] != i[b])
         return i[a] < i[b];
      return j[a] < j[b];
   });

   std::vector<std::uint32_t> i2(m), j2(m);
   for (std::size_t p = 0; p < m; ++p) {
      i2[p] = i[idx[p]];
      j2[p] = j[idx[p]];
   }
   i.swap(i2);
   j.swap(j2);
}

inline void sort_edges_by_block_then_i_then_j(std::vector<std::uint32_t> &i,
                                              std::vector<std::uint32_t> &j,
                                              std::uint32_t tile) {
   assert(tile > 0);
   assert(i.size() == j.size());
   const std::size_t m = i.size();

   std::vector<std::size_t> idx(m);
   std::iota(idx.begin(), idx.end(), 0);

   std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
      const std::uint64_t ia = i[a];
      const std::uint64_t ja = j[a];
      const std::uint64_t ib_a = ia / tile;
      const std::uint64_t jb_a = ja / tile;

      const std::uint64_t ib = i[b];
      const std::uint64_t jb = j[b];
      const std::uint64_t ib_b = ib / tile;
      const std::uint64_t jb_b = jb / tile;

      if (ib_a != ib_b)
         return ib_a < ib_b;
      if (jb_a != jb_b)
         return jb_a < jb_b;
      if (ia != ib)
         return ia < ib;
      return ja < jb;
   });

   std::vector<std::uint32_t> i2(m), j2(m);
   for (std::size_t p = 0; p < m; ++p) {
      i2[p] = i[idx[p]];
      j2[p] = j[idx[p]];
   }
   i.swap(i2);
   j.swap(j2);
}
