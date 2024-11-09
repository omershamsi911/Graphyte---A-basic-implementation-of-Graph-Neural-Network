#ifndef UTILITYFUNCTIONS_H
#define UTILITYFUNCTIONS_H

#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <cmath>
#include <iostream>

int findMaxNodeIndex(const std::vector<std::pair<int, int>>& edges) {
    int maxIndex = 0;
    for (const auto& edge : edges) {
        maxIndex = std::max(maxIndex, std::max(edge.first, edge.second));
    }
    return maxIndex;
}


void getNegativeEdges(std::vector<std::pair<int, int>>& positive_edges, std::vector<std::pair<int, int>>& negative_edges) {
    std::unordered_map<int, std::unordered_set<int>> connections;
    for (const auto& edge : positive_edges) {
        connections[edge.first].insert(edge.second);
    }
    for (int i = 1; i <= 347; i++) {
        for (int j = 1; j <= 347; j++) {  
            if (i == j) {
                continue; 
            }
            if (connections[i].find(j) == connections[i].end()) { 
                negative_edges.push_back(std::make_pair(i, j));
            }
        }
    }
}

void splitEdges(std::vector<std::pair<int, int>>& edges, std::vector<std::pair<int, int>>& train_pos_edges, std::vector<std::pair<int, int>>& test_pos_edges, const float TEST_RATIO=0.3) {
    for(int i = 0; i < ceil(edges.size() * TEST_RATIO); i++) {
        test_pos_edges.push_back(edges[i]);
    }
    for(size_t i = ceil(edges.size() * TEST_RATIO); i < edges.size(); i++) {
        train_pos_edges.push_back(edges[i]);
    }
}

void loadFeatures(std::vector<std::vector<std::vector<float>>>& Features, const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open features file");
    }
    std::string line;
    int lineCount = 0;
    while (std::getline(file, line)) {
        std::stringstream sstr(line);
        size_t node_id;
        sstr >> node_id;
        
        if (node_id >= 0 && node_id < Features[0][0].size()) {
            for (int i = 0; i < 223; i++) {
                float feature_value;
                if (sstr >> feature_value) {
                    Features[i][0][node_id] = feature_value;
                } else {
                    std::cerr << "Error reading feature " << i << " for node " << node_id << std::endl;
                    break;
                }
            }
            lineCount++;
        } else {
            std::cerr << "Warning: node_id " << node_id << " is out of range. Skipping." << std::endl;
        }
    }
    std::cout << "Loaded " << lineCount << " feature vectors" << std::endl;
    file.close();
}

void loadEdges(std::vector<std::pair<int, int>>& edges, const char* filename) {
    std::ifstream file(filename);
    int edge_u, edge_v;
    while(file >> edge_u >> edge_v) {
        edges.push_back(std::make_pair(edge_u, edge_v));
    }
    file.close();
}


void prepareTrainingData(std::vector<std::pair<int, int>>& edges, std::vector<std::pair<int, int>>& train_pos_edges, std::vector<std::pair<int, int>>& test_pos_edges, std::vector<std::pair<int, int>>& train_neg_edges, std::vector<std::pair<int, int>>& test_neg_edges) {
    std::cout << "\n=== Preparing Training Data ===" << std::endl;
    splitEdges(edges, train_pos_edges, test_pos_edges);
    std::cout << "Edge split - Training: " << train_pos_edges.size() << ", Testing: " << test_pos_edges.size() << std::endl;
    getNegativeEdges(train_pos_edges, train_neg_edges);
    getNegativeEdges(test_pos_edges, test_neg_edges);
    std::cout << "Negative edges generated - Training: " << train_neg_edges.size() << ", Testing: " << test_neg_edges.size() << std::endl;
}

void loadDataAndFeatures(std::vector<std::pair<int, int>>& edges, std::vector<std::vector<std::vector<float>>>& Features) {
    std::cout << "\n=== Loading Data ===" << std::endl;
    loadEdges(edges, "include/0.edges");
    std::cout << "Edges loaded successfully: " << edges.size() << " edges" << std::endl;
    int maxNodeIndex = findMaxNodeIndex(edges);
    std::cout << "Network contains " << maxNodeIndex + 1 << " nodes" << std::endl;
    Features.resize(223, std::vector<std::vector<float>>(1, std::vector<float>(maxNodeIndex + 1, 0.0f)));
    std::cout << "\n=== Initializing Features ===" << std::endl;
    std::cout << "Feature dimensions: " << Features.size() << " x " << Features[0].size() << " x " << Features[0][0].size() << std::endl;       
    loadFeatures(Features, "include/0.feat");
    std::cout << "Features loaded successfully" << std::endl;
}

#endif

