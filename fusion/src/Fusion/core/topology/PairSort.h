#ifndef FUSION_CORE_TOPOLOGY_PAIR_SORT_H
#define FUSION_CORE_TOPOLOGY_PAIR_SORT_H

#include <cstdint>
#include <vector>

void sort_edges_by_i_then_j(std::vector<std::uint32_t> &i,
                                   std::vector<std::uint32_t> &j);


void sort_edges_by_block_then_i_then_j(std::vector<std::uint32_t> &i,
                                              std::vector<std::uint32_t> &j,
                                              std::uint32_t tile);

#endif // FUSION_CORE_TOPOLOGY_PAIR_SORT_H