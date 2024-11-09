#ifndef GRAPH_H
#define GRAPH_H

#include <unordered_map>
#include <vector>

class Graph {
    public:
        std::unordered_map<int, std::vector<int>> adjList;

        Graph() {}
        Graph(std::vector<std::pair<int, int>>& edges) {
            for (auto& pair : edges) {
                adjList[pair.first].push_back(pair.second);
            }
        }

        void copy(Graph& g) {
            adjList = g.adjList;
        }
};


#endif