#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <algorithm>
#include <unordered_set>
#include "Graph.h"
#include "Layer.h"
#include "Model.h"
using namespace std;

void loadEdges(const char *filename, unordered_map<int, vector<int>> &edges)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cout << "Error opening file" << endl;
        return;
    }
    int u, v;
    while (file >> u >> v)
    {
        edges[u].push_back(v);
        edges[v].push_back(u);
    }
    return;
}

void loadFeatures(const char *filename, unordered_map<int, vector<vector<float>>> &feature_matrix)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cout << "Error opening file" << endl;
    }
    string line;
    while (getline(file, line))
    {
        stringstream sstr(line);
        int node_id;
        sstr >> node_id;
        vector<vector<float>> features(223, vector<float>(1, 0.0f));
        float feature;
        for (int i = 0; i < 223; i++)
        {
            if (sstr >> feature)
            {
                features[i][0] = feature;
            }
            else
            {
                cout << "An error has occurred" << endl;
            }
        }
        feature_matrix[node_id] = features;
    }
    return;
}

void splitEdges(unordered_map<int, vector<int>> &edges, unordered_map<int, vector<int>> &train_edges, unordered_map<int, vector<int>> &test_edges, float TEST_RATIO = 0.3)
{
    int test_size = ceil(edges.size() * TEST_RATIO);
    int count = 0;
    for (auto &[key, value] : edges)
    {
        if (count == test_size)
        {
            break;
        }
        test_edges[key] = value;
        count++;
    }
    for (auto &[key, value] : edges)
    {
        train_edges[key] = value;
    }
    return;
}

void getNegativeEdges(const unordered_map<int, vector<int>> &pos_edges,
                      unordered_map<int, vector<int>> &neg_edges)
{
    // Step 1: Create a set representation for faster lookups
    unordered_map<int, unordered_set<int>> pos_edge_set;
    for (const auto &[node, neighbors] : pos_edges)
    {
        pos_edge_set[node] = unordered_set<int>(neighbors.begin(), neighbors.end());
    }

    // Step 2: Iterate over all possible node pairs
    for (int i = 1; i <= 334; i++)
    {
        for (int j = 1; j <= 334; j++)
        {
            if (i == j)
            {
                continue; // Skip self-loops
            }
            // Check if the edge (i, j) is not in positive edges
            if (pos_edge_set[i].find(j) == pos_edge_set[i].end())
            {
                neg_edges[i].push_back(j); // Add to negative edges
            }
        }
    }
}

//Inserted Functions
int findMaxNodeIndex(const unordered_map<int, vector<int>>& edges) {
    int maxIndex = 0;
    for (const auto& edge : edges) {
        // Check the key
        maxIndex = max(maxIndex, edge.first);
        // Check all the values in the vector
        for (int node : edge.second) {
            maxIndex = max(maxIndex, node);
        }
    }
    return maxIndex;
}



void loadDataAndFeatures(unordered_map<int, vector<int>>& edges, unordered_map<int, vector<vector<float>>>& Features) {
    std::cout << "\n=== Loading Data ===" << std::endl;

    // Load edges from file
    loadEdges("include/0.edges", edges);
    std::cout << "Edges loaded successfully: " << edges.size() << " edges" << std::endl;

    // Find the maximum node index in the network
    int maxNodeIndex = findMaxNodeIndex(edges);
    std::cout << "Network contains " << maxNodeIndex + 1 << " nodes" << std::endl;

    // Initialize features
    std::cout << "\n=== Initializing Features ===" << std::endl;
    for (int i = 0; i < 223; ++i) {
        Features[i] = vector<vector<float>>(1, vector<float>(maxNodeIndex + 1, 0.0f));
    }

    // Print feature dimensions
    std::cout << "Feature dimensions: " << Features.size() 
              << " x " << Features.begin()->second.size()
              << " x " << Features.begin()->second[0].size() << std::endl;

    // Load feature data from file
    loadFeatures("include/0.feat", Features);
    std::cout << "Features loaded successfully" << std::endl;
}


void prepareTrainingData(unordered_map<int, vector<int>>& edges, unordered_map<int, vector<int>>& train_pos_edges, unordered_map<int, vector<int>>& test_pos_edges, unordered_map<int, vector<int>>& train_neg_edges, unordered_map<int, vector<int>>& test_neg_edges) {
    std::cout << "\n=== Preparing Training Data ===" << std::endl;
    splitEdges(edges, train_pos_edges, test_pos_edges, 0.3f);
    std::cout << "Edge split - Training: " << train_pos_edges.size() << ", Testing: " << test_pos_edges.size() << std::endl;
    getNegativeEdges(train_pos_edges, train_neg_edges);
    getNegativeEdges(test_pos_edges, test_neg_edges);
    std::cout << "Negative edges generated - Training: " << train_neg_edges.size() << ", Testing: " << test_neg_edges.size() << std::endl;
}

#endif