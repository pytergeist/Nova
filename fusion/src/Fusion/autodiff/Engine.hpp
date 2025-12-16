#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>

#include "Fusion/TensorFactory.h"
#include "Fusion/common/Checks.hpp"

#include "ADTypes.h"
#include "AutodiffMeta.hpp"
#include "Graph.hpp"
#include "Sort.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers, readability-magic-numbers)
inline void hash_combine(std::size_t &seed, std::size_t v) noexcept {
   seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers, readability-magic-numbers)

struct ShapeHash {
   std::size_t
   operator()(const std::vector<std::size_t> &shape) const noexcept {
      std::size_t seed = 0; // NOLINT(misc-const-correctness)
      for (auto s : shape)
         hash_combine(seed, s);
      return seed;
   }
};

template <typename T> class Engine {
 public:
   Engine() = default;

   Engine(const Engine &) = delete;
   Engine &operator=(const Engine &) = delete;

   Engine(Engine &&) = delete;
   Engine &operator=(Engine &&) = delete;

   ~Engine() = default;

   template <class Op> ValueID apply(AutodiffMeta<T> &payload) {
      NodeID nid = create_node_and_bind_inputs<Op>(payload);

      INode<T> &node = graph_.get_node(nid);
      AutodiffMeta<T> out = run_forward(node, payload);

      FUSION_CHECK(!out.empty(),
                   "Engine::apply: forward produced empty outputs");

      ensure_node_outputs_allocated(nid, out.size());
      write_forward_results(nid, out);

      // TODO: need to eventually return all outputs? (vector<ValueID>)
      return node.get_output(0);
   }

   void backward(ValueID seed_vid, bool materialise = true,
                 bool retain_graph = false) {
      prepare_grad_buffers();

      std::vector<NodeID> order = topo_sort_for_backward();
      AutodiffMeta<T> seed = init_seed_grad(seed_vid);

      static_cast<void>(seed);

      for (auto it = order.rbegin(); it != order.rend(); ++it) {
         INode<T> &n = graph_.get_node(NodeID{it->idx});
         FUSION_CHECK(n.has_outputs(), "node has no outputs in backward()");

         const ValueID out_vid = n.get_output(0);
         validate_forward_value_exists(n, out_vid);
         ensure_output_grad_slot(out_vid);

         AutodiffMeta<T> grad_in;
         grad_in.push_back(grad_buff_[out_vid]);
         AutodiffMeta<T> grad_out = safe_apply_backward(n, grad_in);

         FUSION_CHECK(grad_out.size() == n.num_inputs(),
                      "backward arity mismatch");
         accum_input_grads(n, grad_out);
      }

      if (materialise) {
         materialise_gradient();
      }
      if (retain_graph) {
         throw std::logic_error("retain_graph not implemented");
      }
   }

   ValueID track_input(ADTensor<T> &t) {
      if (std::optional<ValueID> known = reuse_known_vid(t)) {
         return *known;
      }

      const void *storage_key = static_cast<const void *>(t.get_storage());
      const std::vector<std::size_t> &shp = t.shape();
      if (std::optional<ValueID> cached = lookup_cached_vid(storage_key, shp)) {
         write_val(*cached, t);
         t.set_vid(*cached);
         maybe_mark_leaf(*cached, t);
         return *cached;
      }

      ValueID vid = register_fresh_input_from(t);
      cache_import(storage_key, shp, vid);
      return vid;
   }

   ADTensor<T> materialise(ValueID vid) {
      FUSION_BOUNDS_CHECK(vid, val_buff_.size());
      ADTensor<T> out = val_buff_[vid];
      out.set_vid(vid);
      return out;
   }

   ADTensor<T> get_grad(ValueID vid) {
      FUSION_BOUNDS_CHECK(vid, val_buff_.size());
      return grad_buff_[vid];
   }

   void dump_graph(std::ostream &os) const {
      for (size_t i = 0; i < graph_.nodes().size(); ++i) {
         const INode<T> &n = graph_.get_node(NodeID{static_cast<int32_t>(i)});
         os << "Node " << i << " [" << n.name() << "]\n";
      }

      const size_t n = std::min(grad_buff_.size(), graph_.produced_by().size());
      for (size_t i = 0; i < n; ++i) {
         const INode<T> &prod = graph_.get_produced_by(i);
         NodeID nid = prod.nid;
         if (nid >= 0 && static_cast<size_t>(nid) < graph_.nodes().size()) {
            os << "Node idx: " << nid
               << " Node Op: " << graph_.get_node(nid).name() << " ";
            if (val_buff_[i].empty()) {
               os << "[no val]\n";
            } else {
               std::cout << "Node Val: ";
               for (size_t j = 0; j < val_buff_[i].size(); ++j) {
                  os << val_buff_[i][j] << " ";
               }
            }
            if (grad_buff_[i].empty()) {
               os << "[no grad]\n";
            } else {
               std::cout << "Node Grad: ";
               for (size_t j = 0; j < grad_buff_[i].size(); ++j) {
                  os << grad_buff_[i][j] << " ";
               }
               os << "\n";
            }
         }
      }
   }

 private:
   Graph<T> graph_{};
   std::vector<ADTensor<T>> val_buff_{};
   std::vector<ADTensor<T>> grad_buff_{};
   std::unordered_map<int, ADTensor<T> *> leaf_map_{};
   std::unordered_map<const void *, std::unordered_map<std::vector<size_t>,
                                                       ValueID, ShapeHash>>
       import_cache_{};

   void ensure_value_capacity(ValueID vid) {
      if (val_buff_.size() <= static_cast<size_t>(vid)) {
         val_buff_.resize(static_cast<size_t>(vid) + 1);
      }
   }

   bool graph_knows(ValueID vid) const noexcept {
      return static_cast<size_t>(vid) < graph_.produced_by().size();
   }

   std::optional<ValueID> reuse_known_vid(ADTensor<T> &t) {
      if (!t.has_vid()) {
         return std::nullopt;
      }
      const ValueID vid = t.vid();
      if (!graph_knows(vid)) {
         return std::nullopt;
      }

      write_val(vid, t);
      maybe_mark_leaf(vid, t);
      return vid;
   }

   std::optional<ValueID> lookup_cached_vid(const void *storage_key,
                                            const std::vector<size_t> &shp) {
      if (auto it = import_cache_.find(storage_key);
          it != import_cache_.end()) {
         auto &inner = it->second;
         if (auto it2 = inner.find(shp); it2 != inner.end()) {
            const ValueID cached = it2->second;
            if (graph_knows(cached)) {
               return cached;
            }
         }
      }
      return std::nullopt;
   }

   ValueID register_fresh_input_from(ADTensor<T> &t) {
      const ValueID vid = graph_.new_input_value();
      write_val(vid, t);
      t.set_vid(vid);
      maybe_mark_leaf(vid, t);
      return vid;
   }

   void cache_import(const void *storage_key, const std::vector<size_t> &shp,
                     ValueID vid) {
      import_cache_[storage_key][shp] = vid;
   }

   void maybe_mark_leaf(ValueID vid, ADTensor<T> &t) {
      if (t.requires_grad()) {
         leaf_map_.try_emplace(vid, &t);
      }
   }

   void write_val(ValueID vid, ADTensor<T> &t) {
      ensure_value_capacity(vid);
      if (!val_buff_[vid].is_initialised()) {
         val_buff_[vid] = t;
      }
   }

   ValueID feed_raw(ADTensor<T> &data) {
      ValueID vid = graph_.new_input_value();
      write_val(vid, data);
      return vid;
   }

   void materialise_gradient() {
      const autodiff::NoGradGuard ng;
      for (auto &[vidx, leaf_ptr] : leaf_map_) {
         if (!leaf_ptr)
            continue;
         if (static_cast<size_t>(vidx) >= grad_buff_.size())
            continue;

         ADTensor<T> &g = grad_buff_[vidx];
         if (!g.is_initialised() || g.empty())
            continue;

         const bool storage_matches =
             val_buff_.size() > static_cast<size_t>(vidx) &&
             val_buff_[vidx].is_initialised() && leaf_ptr->is_initialised() &&
             leaf_ptr->storage() && val_buff_[vidx].storage() &&
             (leaf_ptr->get_storage() == val_buff_[vidx].get_storage());

         if (storage_matches) {
            leaf_ptr->ensure_grad();
            leaf_ptr->mutable_grad() = g;
         } else {
            val_buff_[vidx].ensure_grad();
            val_buff_[vidx].mutable_grad() = g;
         }
      }
   }

   template <class Op> ValueID feed(AutodiffMeta<T> v) {
      NodeID dst_nid = graph_.template build_node<Op>(v);
      INode<T> &node = graph_.get_node(dst_nid);
      // Arbitrily set to 0 for single tensor feed
      // TODO: make return type vec<ValueID> to allow for multi output??? Not
      // sure this is needed
      ValueID vid = node.outputs.at(0);
      write_val(vid, v[0]);
      return vid;
   }

   void set_grad_buff_size() { grad_buff_.resize(val_buff_.size()); }

   AutodiffMeta<T> grad_init(ValueID vid) {
      ADTensor<T> grad = ones_like(val_buff_[vid]);
      grad_buff_[vid] = grad;
      AutodiffMeta<T> gradVec;
      gradVec.push_back(grad);
      return gradVec;
   }

   AutodiffMeta<T> run_forward(INode<T> &node, AutodiffMeta<T> &vec) {
      return node.apply_forward(vec);
   }

   template <class Op>
   NodeID create_node_and_bind_inputs(AutodiffMeta<T> &payload) {
      for (size_t i = 0; i < payload.size(); ++i) {
         if (!payload[i].has_vid() || !graph_knows(payload[i].vid())) {
            track_input(payload[i]);
         }
      }

      NodeID dst = graph_.template build_node<Op>();
      INode<T> &node = graph_.get_node(dst);

      for (size_t i = 0; i < payload.size(); ++i) {
         const ValueID in_vid = payload[i].vid();
         FUSION_CHECK(graph_knows(in_vid), "Input not registered");

         const NodeID src = graph_.get_produced_by(in_vid).nid;
         graph_.set_node_input(node, in_vid);
         graph_.append_consumer_table(dst, in_vid, i);
         if (src != -1) {
            graph_.add_edge(src, dst);
         }
      }
      return dst;
   }

   void ensure_node_outputs_allocated(NodeID nid, size_t arity) {
      INode<T> &node = graph_.get_node(nid);
      if (node.has_outputs()) {
         FUSION_CHECK(node.num_outputs() == arity, "node output size mismatch");
         return;
      }
      for (size_t i = 0; i < arity; ++i) {
         ValueID vid = graph_.new_intermediate_value();
         graph_.set_produced_by(vid, nid, i);
         graph_.set_node_output(node, vid);
         ensure_value_capacity(vid);
      }
   }

   void write_forward_results(NodeID nid, const AutodiffMeta<T> &out) {
      INode<T> &node = graph_.get_node(nid);
      FUSION_BOUNDS_CHECK(0, node.num_outputs());
      FUSION_CHECK(node.num_outputs() == out.size(),
                   "node output size mismatch");

      for (size_t i = 0; i < out.size(); ++i) {
         ValueID vid_i = node.get_output(i);
         ensure_value_capacity(vid_i);
         val_buff_[vid_i] = out[i];
      }
   }

   void prepare_grad_buffers() {
      set_grad_buff_size();
      for (auto &g : grad_buff_) {
         if (g.is_initialised())
            g.clear();
      }
   }

   std::vector<NodeID> topo_sort_for_backward() {
      Sort<T> sort_(graph_.nodes().size());
      return sort_.topological_sort(graph_.nodes(), graph_.produced_by(),
                                    graph_.consumed_by(), graph_.node_ids());
   }

   AutodiffMeta<T> init_seed_grad(ValueID vid) {
      ADTensor<T> grad = ones_like(val_buff_[vid]);
      grad_buff_[vid] = grad;
      AutodiffMeta<T> v;
      v.push_back(grad);
      return v;
   }

   void validate_forward_value_exists(const INode<T> &n,
                                      ValueID out_vid) const {
      FUSION_CHECK(static_cast<size_t>(out_vid) < val_buff_.size(),
                   std::string("val index OOB in backward: ") +
                       std::string(n.name()));
      FUSION_CHECK(val_buff_[out_vid].is_initialised(),
                   std::string("val missing for node output: ") +
                       std::string(n.name()));
   }

   void ensure_output_grad_slot(ValueID out_vid) {
      if (!grad_buff_[out_vid].is_initialised()) {
         grad_buff_[out_vid] = zeros_like(val_buff_[out_vid]);
      }
   }

   AutodiffMeta<T> safe_apply_backward(INode<T> &n, AutodiffMeta<T> &gin) {
      try {
         return n.apply_backward(gin);
      } catch (const std::exception &e) {
         throw std::runtime_error(std::string("apply_backward threw in op ") +
                                  std::string(n.name()) + ": " +
                                  std::string(e.what()));
      }
   }

   void accum_input_grads(const INode<T> &n, const AutodiffMeta<T> &gout) {
      for (size_t j = 0; j < n.num_inputs(); ++j) {
         const ValueID in_vid = n.get_input(j);
         ADTensor<T> &dst = grad_buff_[in_vid];
         const ADTensor<T> &src = gout.at(j);
         if (!dst.is_initialised()) {
            dst = src;
         } else {
            const autodiff::NoGradGuard ng;
            dst = dst + src; // TODO: shape check if needed
         }
      }
   }
};

#endif // ENGINE_HPP
