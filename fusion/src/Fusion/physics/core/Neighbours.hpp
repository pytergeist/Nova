#ifndef FUSION_PHYSICS_CORE_NEIGHBOURS_HPP
#define FUSION_PHYSICS_CORE_NEIGHBOURS_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

// TODO: Make sure you change the below types to whatever is the most suitable
// e.g.

struct CRS {
   std::int64_t N = 0, E = 0;
   std::vector<std::uint32_t> row_ptr;
   std::vector<std::uint32_t> col_idx;

   std::vector<float> w;
   std::vector<float> r0;
   std::vector<std::uint16_t> type;

   bool sorted_by_j = false;
   bool symmetric = false;
   bool directed = false;
};

struct BlockedCRS {
   std::int64_t N = 0, E = 0;
   std::int64_t TILE = 0, nBlocks = 0;

   /* i-block ptr. size = nblocks + 1.
   * i-block(n) = ib_ptr[n] + ib_ptr[n+1].
   * i=0 → j {1,4,5} → j-blocks {0,1,1}
   * i=1 → j {0,4} → j-blocks {0,1}
   * i=2 → j {6} → j-blocks {1}
   * i=3 → j {4,7} → j-blocks {1,1}
   * i=4 → j {0,1,5,7} -> j-blocks {0, 1}
   * i=5 → j {0,4} -> j-blocks {0, 1}
   * i=6 → j {2,7} -> j-blocks {0, 1}
   * i=7 → j {3,4,6} -> j-blocks {0, 1}

   * where TILE = 4, N = 8. and edgelist:
   * i =  {4,4,4,4,  5,5,  6,6,  7,7,7}
   * j =  {0,1,5,7,  0,4,  2,7,  3,4,6}
   * TILE of 4 means we have two iblocks and two jblocks:
   *    - jb=0 covers j(0...3)
   *    - jb=1 convers j(4..7)
   *    - ib=0 covers i(0...3)
   *    - ib=1 convers i(4..7)
   * All i-blocks connect to all j-blocks, leading to 4 groups.
   * We want to use ib_ptr to see how many j-blocks we connect to.
   * from the above example you can see that i-block(0) connects to j-block(0,
   1).
   * this means the ib_ptr = [0, 2,...]. So ib_ptr[0] + ib_ptr[1] = 0:2 (e.g.
   0,1).
   * The full ib_ptr = [0, 2, 4], so for n=1, ib_ptr[1] + ib_ptr[2] = 2:4.
   * This is because ib=0 uses g[0,2) and ib=1 uses g[2,4).
   */

   /*
    * Group metadata. Each group (g) corresponds to an i-block ptr (from
    * ib_ptr), One j-block, grabbed via jb = jb_idx[g]. jb_idx.size() == Num
    * groups == 4. and edges in that j-block, that are accessed through
    * jb_ptr[g] + jb_ptr[g+1].
    *
    * This means the ordering of groups is important (we need to leave this
    * generic enough to do aggressive re-ordering of coords/edges later). For
    * each i-block we can list it's j-blocks in increasing order. From the above
    * example. For i-block = n: n=0: j-block = (0,1)
    *      - g=0: (ib=0, jb=0)
    *      - g=1: (ib=0, jb=1)
    * n=1: j-block =
    *      - g=2: (ib=1, jb=0)
    *      - g=3: (ib=1, jb=1)
    * So jb_idx = [0, 1, 0, 1]
    *
    * jb_ptr.size() == Num groups + 1. This member maps group -> range of edges
    * in the edge array. Group edge counting:
    * NB:
    *  - ib=0, i=0,1,2,3
    *  - ib=1, i=4,5,6,7
    *  - jb=0, j=0,1,2,3
    *  - jb=1, j=4,5,6,7
    *  g0 (ib0, jb0): i0->j1, i1->j0: 2 edges
    *  g1: (ib0, jb1): i0->j4, i0->j5, i1->j4, i2->j6, i3->j4, i3->j7: 6 edges
    *  g2: (ib=1, jb=0): i4->j0, i4->j1, i5->j0, i6->j2, i7->j3: 5 edges
    *  g3: (ib=1, jb=1): i4->j5, i4->j7, i5->j4, i6->j7, i7->j4, i7->j6: 6 edges
    *  The jb_ptr can then be calculated with a prefix sum
    *  jb_ptr = {0, 2, 8, 13, 19}
    *
    *  i_lane, j_lane, and e_idx are per-edge arrays. Firstly edges are assumed
    * to be ordered by (i, j) and sorted in ascending order. Lanes are idx
    * values that define the row/column indices of the TILExTILE data structure.
    * ib = i / TILE: the i-block (TileID)
    * jb = j / TILE: the j-block (TileID)
    * i_lane = i % TILE: specifies row within the i-block (LaneID)
    * j_lane = j % TILE: specifies column within the j-block (LaneID)
    *
    * Ordered Edges:
    * e0:  (0,1)
    * e1:  (0,4)
    * e2:  (0,5)
    * e3:  (1,0)
    * e4:  (1,4)
    * e5:  (2,6)
    * e6:  (3,4)
    * e7:  (3,7)
    * e8:  (4,0)
    * e9:  (4,1)
    * e10: (4,5)
    * e11: (4,7)
    * e12: (5,0)
    * e13: (5,4)
    * e14: (6,2)
    * e15: (6,7)
    * e16: (7,3)
    * e17: (7,4)
    * e18: (7,6)
    *
    * From this sorted edge list you can construct the i_lanes in group order,
    * using the following instructions:
    * for g0
    * ib = i/TILE;
    * jb = j/TILE;
    * g0 edges are e0 = (0,1), e3 = (1,0)
    * compute lanes (k%TILE)
    * (0, 1): i_lane = 0 % 4 = 0, j_lane= 1 % 4 = 1;
    * (1, 0): i_lane = 1 % 4 = 1, j_lane= 0 % 4 = 0;
    * i_lane = {0, 1} // g0
    * j_lane = {1, 0} // g0
    * edge_idx = {0, 3}
    * then follow the procedure for each edge in all groups
    * */
   std::vector<std::uint32_t> ib_ptr;

   std::vector<std::uint32_t> jb_ptr;
   std::vector<std::uint32_t> jb_idx;

   std::vector<std::uint16_t> i_lane;
   std::vector<std::uint16_t> j_lane;
   std::vector<std::uint32_t> e_idx;
};

#endif // FUSION_PHYSICS_CORE_NEIGHBOURS_HPP
