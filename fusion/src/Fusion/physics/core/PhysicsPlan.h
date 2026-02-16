#ifndef FUSION_PHYSICS_PLAN_HPP
#define FUSION_PHYSICS_PLAN_HPP

#include <cstddef>
#include <cstdint>

#include "Neighbours.hpp"
#include "State.hpp"

enum class PairListKind { EdgeList, CRS, NMCLuster };

enum class VecLayout { SoA, AoS, AoSoA };

template <typename T, class ParticlesT> struct PairwisePlan {
   PairListKind kind{PairListKind::CRS};
   VecLayout layout{VecLayout::SoA};

   ParticlesT psoa;
   EdgeList edges;
   CRS crs;

   std::int64_t N{0};
   std::int64_t E{0};
   bool x_contig{false};
   bool f_contig{false};

   std::size_t itemsize;
};

template <typename T, class ParticlesT>
inline CRS make_crs(const ParticlesT &psoa, const EdgeList &edges) {
   CRS crs;
   crs.N = psoa.N();
   crs.E = edges.E();

   crs.row_ptr.assign(crs.N + 1, 0);
   crs.col_idx.resize(crs.E);

   for (std::size_t e = 0; e < crs.E; ++e) {
      const std::uint32_t src = edges.i[e];
      crs.row_ptr[src + 1]++;
   }

   for (std::int64_t i = 0; i < crs.N; ++i) {
      crs.row_ptr[i + 1] += crs.row_ptr[i];
   }

   std::vector<std::uint32_t> cursor = crs.row_ptr;
   for (std::size_t e = 0; e < crs.E; ++e) {
      const std::uint32_t src = edges.i[e];
      const std::uint32_t dst = edges.j[e];
      const std::uint32_t pos = cursor[src]++;
      crs.col_idx[pos] = dst;
   }
   return crs;
}

struct PairHash {
   template <class T1, class T2>
   std::size_t operator()(const std::pair<T1, T2> &p) const {
      auto h1 = std::hash<T1>{}(p.first);
      auto h2 = std::hash<T2>{}(p.second);
      return h1 ^ h2;
   }
};

struct PairEqual {
   template <class T1, class T2>
   bool operator()(const std::pair<T1, T2> &p1,
                   const std::pair<T1, T2> &p2) const {
      return p1.first == p2.first && p1.second == p2.second;
   }
};

struct Group {
   std::uint32_t group_idx;
   std::uint32_t num_edges;
   std::uint32_t ib_idx;
   std::uint32_t jb_idx;
};

template <typename T, class ParticlesT>
inline BlockedCRS make_blocked_crs(const ParticlesT &psoa,
                                   EdgeList &edges) {
   if (!edges.sorted) {
      edges.sort();
   }
   BlockedCRS bcrs;
   bcrs.N = psoa.N();
   bcrs.E = edges.E();
   bcrs.TILE = psoa.tile(); // TODO: curr just hardcoding for dev
   bcrs.nBlocks = (bcrs.N + bcrs.TILE - 1) / bcrs.TILE;

   bcrs.ib_ptr.assign(bcrs.nBlocks + 1, 0);
   // TODO: this unordered map will have lots of hash collisions for large
   // graphs. done now for simplicity, but change to something more efficient (std::vec? bitsetmask?)
   std::unordered_map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t, PairHash, PairEqual> groups;
   std::vector<std::uint32_t> group_counter;
   for (std::size_t e = 0; e < bcrs.E; ++e) {
      const std::uint32_t src = edges.i[e];
      const std::uint32_t dst = edges.j[e];
      std::uint64_t ib = src / bcrs.TILE;
      std::uint64_t jb = dst / bcrs.TILE;
      auto key = std::make_pair(ib, jb);
      auto [it, inserted] = groups.try_emplace(
          key, static_cast<std::uint64_t>(group_counter.size()));
      if (inserted) {
         group_counter.push_back(0);
         group_counter[it->second] += 1;
         bcrs.ib_ptr[ib + 1]++; // TODO: the below impl is sloppy af
         bcrs.jb_idx.push_back(jb);
      } else {
         group_counter[it->second] += 1;
      }

      // compute lanes

   }

   for (std::size_t k = 1; k < bcrs.ib_ptr.size(); ++k) {
		bcrs.ib_ptr[k] += bcrs.ib_ptr[k - 1];
   }
   bcrs.jb_ptr = group_counter;
   bcrs.jb_ptr.insert(bcrs.jb_ptr.begin(), 0);

   for (std::size_t k = 1; k < bcrs.jb_ptr.size(); ++k) {
      bcrs.jb_ptr[k] += bcrs.jb_ptr[k - 1];
   }


   std::cout << "Groups: [";
    for (auto v : group_counter) {
         std::cout << v << ", ";
      }
	std::cout << "]" << std::endl;

    std::cout << "ib_ptr: [";
    for (auto i : bcrs.ib_ptr) {
         std::cout << i << ", ";
      }
	std::cout << "]" << std::endl;
    std::cout << "jb_idx: [";
    for (auto i : bcrs.jb_idx) {
         std::cout << i << ", ";
      }
	std::cout << "]" << std::endl;

    std::cout << "jb_ptr: [";
    for (auto i : bcrs.jb_ptr) {
         std::cout << i << ", ";
      }
	std::cout << "]" << std::endl;
   return bcrs;
}

template <typename T, class ParticlesT>
inline PairwisePlan<T, ParticlesT> make_pairwise_plan(const ParticlesT &psoa,
                                                      const EdgeList &edges) {
   PairwisePlan<T, ParticlesT> plan;
   plan.kind = PairListKind::CRS;
   plan.layout = VecLayout::AoSoA; // TODO: cur defualting to AoSoA, should have
                                   // all options?
   plan.psoa = psoa;

   CRS crs = make_crs<T, ParticlesT>(psoa, edges);
   plan.crs = crs;
   plan.edges = edges;

   plan.N = static_cast<int64_t>(psoa.N());
   plan.E = static_cast<int64_t>(edges.E());
   plan.x_contig = psoa.x.is_contiguous();
   plan.f_contig = psoa.f.is_contiguous();
   plan.itemsize = sizeof(T);

   return plan;
}

#endif // FUSION_PHYSICS_PLAN_HPP
