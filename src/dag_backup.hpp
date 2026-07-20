#pragma once

#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

namespace Scarlet::Search {

// Reverse-topological propagation for a ply-layered DAG. Parents are allowed
// to be requeued after an earlier visit when another dependent child changes.
template<class PlyOf, class ParentsOf, class Backup>
void propagate_dag_change(std::size_t nodeCount, int changed,
                          PlyOf&& plyOf, ParentsOf&& parentsOf, Backup&& backup) {
    if (changed < 0 || std::size_t(changed) >= nodeCount) return;
    std::priority_queue<std::pair<int, int>> pending;
    std::vector<bool> queued(nodeCount, false);
    auto enqueue = [&](int node) {
        if (node < 0 || std::size_t(node) >= nodeCount || queued[node]) return;
        queued[node] = true;
        pending.emplace(plyOf(node), node);
    };
    enqueue(changed);
    for (int parent : parentsOf(changed)) enqueue(parent);
    while (!pending.empty()) {
        const int current = pending.top().second;
        pending.pop();
        queued[current] = false;
        if (!backup(current)) continue;
        for (int parent : parentsOf(current)) enqueue(parent);
    }
}

} // namespace Scarlet::Search
