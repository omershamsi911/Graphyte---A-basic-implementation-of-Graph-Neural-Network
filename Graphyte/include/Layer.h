#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <math.h>
#include "Graph.h"

class SAGELayer
{
public:
    Graph g;
    unordered_map<int, vector<vector<float>>> feature_matrix;
    vector<vector<float>> weights;

    SAGELayer() {}
    void init(Graph pos_g, unordered_map<int, vector<vector<float>>> &feature_matrix)
    {
        this->g.copyGraph(pos_g);
        for (auto &[key, value] : feature_matrix)
        {
            this->feature_matrix[key] = value;
        }
        weights = Xavier_initialization(223, 223);
    }

    void forward()
    {
        // Store aggregated features separately to avoid mixing steps
        unordered_map<int, vector<vector<float>>> aggregated_features;

        for (auto &[node_id, neighbors] : g.adjList)
        {
            // First aggregate 1-hop neighbors
            vector<vector<float>> neighbor_features(223, vector<float>(1, 0.0f));

            for (int neighbor : neighbors)
            {
                auto &neighbor_feat = feature_matrix[neighbor];
                for (int i = 0; i < 223; i++)
                {
                    neighbor_features[i][0] += neighbor_feat[i][0] / neighbors.size();
                }
            }

            // Concatenate with self features
            auto combined_features = concat(neighbor_features, feature_matrix[node_id]);

            // Apply weights, non-linearity, and normalization
            combined_features = applyWeights(weights, combined_features);
            for (auto &feat : combined_features)
            {
                feat[0] = sigmoid(feat[0]);
            }
            feature_matrix[node_id] = l2_normalization(combined_features);
        }
    }

private:
    vector<vector<float>> concat(vector<vector<float>> &v1, vector<vector<float>> &v2)
    {
        vector<vector<float>> res(v1.size() + v2.size(), vector<float>(v1[0].size(), 0.0f));
        for (int i = 0; i < v1.size(); i++)
        {
            for (int j = 0; j < v1[0].size(); j++)
            {
                res[i][j] = v1[i][j];
            }
        }
        for (int i = 0; i < v1.size(); i++)
        {
            for (int j = 0; j < v1[0].size(); j++)
            {
                res[i + v1.size()][j] = v1[i][j];
            }
        }
        return res;
    }

    vector<vector<float>> Xavier_initialization(int inputs, int outputs)
    {
        vector<vector<float>> weights(223, vector<float>(446, 0.0f));
        float upper_bound = sqrt(6.0 / (inputs + outputs));
        float lower_bound = -1.0 * sqrt(6.0 / (inputs + outputs));
        for (int i = 0; i < 223; i++)
        {
            for (int j = 0; j < 446; j++)
            {
                weights[i][j] = ((rand() / float(RAND_MAX)) * (upper_bound - lower_bound)) + lower_bound;
            }
        }
        return weights;
    }

    vector<vector<float>> applyWeights(vector<vector<float>> &weights, vector<vector<float>> features)
    {
        vector<vector<float>> res(223, vector<float>(1, 0.0f));
        for (int i = 0; i < 223; i++)
        {
            for (int j = 0; j < 1; j++)
            {
                for (int k = 0; k < 446; k++)
                {
                    res[i][j] = weights[i][k] * features[k][j];
                }
            }
        }
        return res;
    }

    float sigmoid(float x)
    {
        return 1.0 / (1 + exp(-x));
    }

    vector<vector<float>> l2_normalization(vector<vector<float>> v)
    {
        vector<vector<float>> res(223, vector<float>(1, 0.0f));
        float unit_v = 0;
        for (int i = 0; i < 223; i++)
        {
            unit_v += v[i][0] * v[i][0];
        }
        unit_v = sqrt(unit_v);
        for (int i = 0; i < 223; i++)
        {
            res[i][0] = v[i][0] / unit_v;
        }
        return res;
    }
};


#endif