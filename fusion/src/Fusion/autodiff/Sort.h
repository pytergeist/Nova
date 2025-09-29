#ifndef SORT_H
#define SORT_H

#include <queue>
#include <vector>

class Sort {
public:
  std::vector<NodeID> sorted;
  Sort(uint16_t numNodes) : numNodes_(std::move(numNodes)) {
    std::vector<bool> visited(numNodes, false);
  };
  std::vector<uint16_t> calc_indegree(std::vector<INode> &nodes,
                                      std::vector<ProducerInfo> &produced_by) {
    // This belongs in the graph??? not in the sort - also iteratively update
    // this do not calculate on method exe
    std::vector<uint16_t> inDegree(numNodes_);
    for (uint16_t i = 0; i < numNodes_; i++) {
      uint16_t increment = 0;
      auto inp = nodes[i].inputs;
      // inspect O(?) for this impl
      for (uint16_t j = 0; j < inp.size(); j++) {
        if (produced_by[inp[j].idx].nid.idx != kNoNode) {
          increment++;
        }
        inDegree[i] = increment;
      }
    };
    return inDegree;
  };

  std::vector<NodeID> topological_sort(std::vector<INode> &nodes,
                        std::vector<ProducerInfo> &produced_by, std::vector<std::vector<ConsumerInfo>> &consumed_by, std::vector<NodeID> &node_ids) {
    std::queue<NodeID> q;
    std::vector<uint16_t> in_degree = calc_indegree(nodes, produced_by);
    for (uint16_t i = 0; i < numNodes_; i++) {
            if (i > in_degree.size()) {
        throw std::runtime_error("Number of nodes greater than indegree vec");
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
        auto outputs = nodes[nid.idx].outputs;
        for (uint16_t j = 0; j < outputs.size(); ++j) {
			std::vector<ConsumerInfo> children = consumed_by[outputs[j].idx];
            for (uint16_t k = 0; k < children.size(); ++k) {
              in_degree[children[k].nid.idx]--;
              if (in_degree[children[k].nid.idx] == 0) {
              q.push(children[k].nid);
            }
           }
        }
     }
    std::cout << "result size: " <<  result.size() << "\n";
    std::cout << "num node size: " << numNodes_ << "\n";
    if (result.size() != numNodes_) {
      throw std::runtime_error("Number of nodes greater than indegree vec - graph contains cycles!");
    }

    return result;
  }

private:
  uint16_t numNodes_;
};

#endif // SORT_H
