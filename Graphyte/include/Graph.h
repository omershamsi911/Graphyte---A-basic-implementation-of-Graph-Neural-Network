#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <unordered_map>
using namespace std;

class Graph
{
public:
    unordered_map<int, vector<int>> adjList;
    Graph() {}
    Graph(unordered_map<int, vector<int>> &edges)
    {
        for (auto &[key, value] : edges)
        {
            adjList[key] = value;
        }
    }
    void copyGraph(Graph &g)
    {
        for (auto &[key, value] : g.adjList)
        {
            adjList[key] = value;
        }
    }
};


#endif