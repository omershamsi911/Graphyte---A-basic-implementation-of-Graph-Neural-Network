#ifndef MODEL_H
#define MODEL_H

#include "Layer.h"
#include <algorithm>
using namespace std;

class SAGEModel
{
public:
    SAGELayer pos_layer1;
    SAGELayer pos_layer2;
    SAGELayer neg_layer1;
    SAGELayer neg_layer2;

    Graph train_pos_g;
    Graph train_neg_g;
    unordered_map<int, vector<vector<float>>> feature_matrix;

    SAGEModel() {}

    SAGEModel(Graph train_pos_g, Graph train_neg_g, unordered_map<int, vector<vector<float>>> &feature_matrix)
    {
        this->train_pos_g.copyGraph(train_pos_g);
        this->train_neg_g.copyGraph(train_neg_g);
        pos_layer1.init(this->train_pos_g, feature_matrix);
        pos_layer2.init(this->train_pos_g, feature_matrix);
        for (auto &[key, value] : feature_matrix)
        {
            this->feature_matrix[key] = value;
        }
    }

    void train(int num_epochs = 5)
    {
        for (int i = 0; i < num_epochs; i++)
        {
            cout << "Epoch: " << i + 1 << " / " << num_epochs << endl;
            pos_layer1.forward();
            pos_layer2.init(pos_layer1.g, pos_layer1.feature_matrix);
            pos_layer2.forward();

            neg_layer1.forward();
            neg_layer2.init(neg_layer1.g, neg_layer1.feature_matrix);
            neg_layer2.forward();

            cout << calculateLoss(pos_layer2.g.adjList, neg_layer2.g.adjList, pos_layer2.feature_matrix, neg_layer2.feature_matrix) << endl;
        }
    }

    vector<pair<int, float>> getPrediction(int u)
    {
        vector<pair<int, float>> scores;
        for (auto &[key, value] : train_pos_g.adjList)
        {
            if (key == u)
            {
                continue;
            }
            float score = cosine_similarity(u, key);
            scores.push_back(make_pair(key, score));
        }
        sort(scores.begin(), scores.end(), [](pair<int, float> a, pair<int, float> b)
             { return a.second > b.second; });
        return scores;
    }

    float evaluate(const unordered_map<int, vector<int>> &test_pos_edges,
                   const unordered_map<int, vector<int>> &test_neg_edges)
    {
        vector<pair<float, bool>> all_scores;

        // Calculate scores for positive edges
        for (const auto &[node, neighbors] : test_pos_edges)
        {
            for (int neighbor : neighbors)
            {
                float score = cosine_similarity(node, neighbor);
                all_scores.push_back({score, true});
            }
        }

        // Calculate scores for negative edges
        for (const auto &[node, neighbors] : test_neg_edges)
        {
            for (int neighbor : neighbors)
            {
                float score = cosine_similarity(node, neighbor);
                all_scores.push_back({score, false});
            }
        }

        return calculateAUC(all_scores);
    }

private:
    float dot_product(int u, int v)
    {
        vector<vector<float>> u_features = feature_matrix[u];
        vector<vector<float>> v_features = feature_matrix[v];
        float score;
        for (int i = 0; i < 223; i++)
        {
            score += u_features[i][0] * v_features[i][0];
        }
        return score;
    }

    float cosine_similarity(int u, int v)
    {
        float score = 0.0f;
        float mag_a = 0.0f;
        float mag_b = 0.0f;
        vector<vector<float>> u_features = feature_matrix[u];
        vector<vector<float>> v_features = feature_matrix[v];

        for (int i = 0; i < 223; i++)
        {
            score += u_features[i][0] * v_features[i][0];
            mag_a += u_features[i][0] * u_features[i][0];
            mag_b += v_features[i][0] * v_features[i][0];
        }

        mag_a = sqrt(mag_a);
        mag_b = sqrt(mag_b);

        return (mag_a * mag_b == 0) ? 0 : score / (mag_a * mag_b);
    }

    float sigmoid(float x)
    {
        return 1.0 / (1 + exp(-x));
    }

    float calculateLoss(unordered_map<int, vector<int>> &pos_adjList, unordered_map<int, vector<int>> &neg_adjList, unordered_map<int, vector<vector<float>>> &pos_embed, unordered_map<int, vector<vector<float>>> &neg_embed)
    {
        float loss = 0.0f;
        float Q = 5.0f;
        float epsilon = 0.0001f;

        float pos_loss = 0.0f;
        float neg_loss = 0.0f;

        for (auto &[key, value] : pos_adjList)
        {
            vector<int> neighbors = value;
            for (int neighbor : neighbors)
            {
                for (int i = 0; i < 223; i++)
                {
                    pos_loss += pos_embed[neighbor][i][0] * pos_embed[key][i][0];
                }
                pos_loss = sigmoid(pos_loss) + epsilon;
                pos_loss = -1.0 * (log(pos_loss));
            }
        }

        for (auto &[key, value] : neg_adjList)
        {
            vector<int> neighbors = value;
            for (int neighbor : neighbors)
            {
                for (int i = 0; i < 223; i++)
                {
                    neg_loss += neg_embed[neighbor][i][0] * neg_embed[key][i][0];
                }
                neg_loss = sigmoid(neg_loss) + epsilon;
                neg_loss = -1.0 * (log(neg_loss));
            }
        }

        loss = -pos_loss - (Q * neg_loss);
        return loss;
    }

    float calculateAUC(const vector<pair<float, bool>> &scores)
    {
        // Sort scores in descending order
        vector<pair<float, bool>> sorted_scores = scores;
        sort(sorted_scores.begin(), sorted_scores.end(),
             [](const auto &a, const auto &b)
             { return a.first > b.first; });

        int positive_count = 0;
        int negative_count = 0;

        // Count positive and negative examples
        for (const auto &score : sorted_scores)
        {
            if (score.second)
                positive_count++;
            else
                negative_count++;
        }

        if (positive_count == 0 || negative_count == 0)
        {
            return 0.5; // Return random classifier score if only one class present
        }

        float auc = 0.0;
        int positive_seen = 0;

        // Calculate AUC using the rank formula
        for (size_t i = 0; i < sorted_scores.size(); i++)
        {
            if (sorted_scores[i].second)
            {
                positive_seen++;
            }
            else
            {
                auc += positive_seen;
            }
        }

        // Normalize AUC
        auc /= (positive_count * negative_count);
        return auc;
    }
};


#endif