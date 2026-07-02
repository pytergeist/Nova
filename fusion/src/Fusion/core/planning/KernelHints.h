#ifndef FUSION_CORE_PLANNING_KERNEL_HINTS_H
#define FUSION_CORE_PLANNING_KERNEL_HINTS_H

#include <cstddef>
#include <cstdint>

struct GemmLikeDesc {
   std::size_t batch{1};
   std::size_t M{1}, N{1}, K{1};

   std::int64_t out_rs{0}, out_cs{0};
   std::int64_t a_rs{0}, a_cs{0};
   std::int64_t b_rs{0}, b_cs{0};

   bool a_transpose{false};
   bool b_transpose{false};
   bool out_is_contig_mn{false};
   bool a_is_contig_mk{false};
   bool b_is_contig_kn{false};
};

struct KernelHints {
   bool all_contiguous_like{false};
   std::size_t vector_bytes{0};

   bool gemm_like{false};
   GemmLikeDesc gemm{};
};


#endif // FUSION_CORE_PLANNING_KERNEL_HINTS_H