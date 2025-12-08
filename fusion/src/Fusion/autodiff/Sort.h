#ifndef SORT_H
#define SORT_H

#include <queue>
#include <vector>

#include "AutodiffMeta.h"

template <typename T> class Sort {
 public:
   std::vector<NodeID> sorted;
   Sort(std::size_t numNodes) : numNodes_(std::move(numNodes)) {
      std::vector<bool> visited(numNodes, false);
   };

   std::vector<std::size_t> calc_indegree(std::vector<INode<T>> &nodes,
                                     std::vector<ProducerInfo> &produced_by) {
      // This belongs in the graph??? not in the sort - also iteratively update
      // this do not calculate on method exe
      std::vector<size_t> inDegree(numNodes_);
      for (size_t i = 0; i < numNodes_; i++) {
         size_t increment = 0;
         auto inp = nodes.at(i).inputs();
         // inspect O(?) for this impl
         for (size_t j = 0; j < inp.size(); j++) {
            if (produced_by[inp[j]].nid != kNoNode) {
               increment++;
            }
            inDegree[i] = increment;
         }
      };
      return inDegree;
   };

   std::vector<NodeID>
   topological_sort(std::vector<INode<T>> &nodes,
                    std::vector<ProducerInfo> &produced_by,
                    std::vector<std::vector<ConsumerInfo>> &consumed_by,
                    std::vector<NodeID> &node_ids) {
      std::queue<NodeID> q;
      std::vector<size_t> in_degree = calc_indegree(nodes, produced_by);
      for (size_t i = 0; i < numNodes_; i++) {
         if (i > in_degree.size()) {
            throw std::runtime_error(
                "Number of nodes greater than indegree vec");
         };
         if (in_degree[i] == 0) {
            q.push(node_ids[i]);
         }
      }
      std::vector<NodeID> result;
      while (!q.empty()) {
         NodeID nid = q.front();
         q.pop();
         result.push_back(nid);
         auto outputs = nodes[nid].outputs();
         for (std::size_t j = 0; j < outputs.size(); ++j) {
            std::vector<ConsumerInfo> children = consumed_by[outputs[j]];
            for (std::size_t k = 0; k < children.size(); ++k) {
               in_degree[children[k].nid]--;
               if (in_degree[children[k].nid] == 0) {
                  q.push(children[k].nid);
               }
            }
         }
      }
      if (result.size() != numNodes_) {
         throw std::runtime_error("Number of nodes greater than indegree vec - "
                                  "graph contains cycles!");
      }

      return result;
   }

 private:
   std::size_t numNodes_;
};

#endif // SORT_H
