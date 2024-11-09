#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <unordered_map>
#include <iostream>
#include "graph.h"

class SAGELayer {
public:
    Graph train_pos_g;
    Graph train_neg_g;
    std::vector<std::vector<std::vector<float>>> features;
    std::vector<std::vector<float>> weights;
    std::vector<std::vector<float>> weight_gradients;
    float learning_rate;
    float decay_rate = 0.95f;
    
    SAGELayer(Graph& train_pos_g, Graph& train_neg_g, std::vector<std::vector<std::vector<float>>>& Features, float lr = 0.01) {
        this->train_pos_g.copy(train_pos_g);
        this->train_neg_g.copy(train_neg_g);
        this->features = Features;
        learning_rate = lr;
        std::cout << "Sage Layer features: " << features.size() << std::endl;
        weights = Xavier_initialization();
        weight_gradients.resize(weights.size(), std::vector<float>(weights[0].size(), 0.0f));
    }

    std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<std::vector<float>>>> forward() {
        std::cout << "Entered forward pass" << std::endl;
        
        // Initialize embeddings for both positive and negative graphs
        std::vector<std::vector<std::vector<float>>> z_v_pos(223, std::vector<std::vector<float>>(1, std::vector<float>(features[0][0].size(), 0.0f)));
        std::vector<std::vector<std::vector<float>>> z_v_neg(223, std::vector<std::vector<float>>(1, std::vector<float>(features[0][0].size(), 0.0f)));
        
        // Process positive graph
        processGraph(train_pos_g, z_v_pos);
        
        // Process negative graph
        processGraph(train_neg_g, z_v_neg);
        
        return {z_v_pos, z_v_neg};
    }
    void backward(const std::vector<std::vector<std::vector<float>>>& pos_embeddings,
                 const std::vector<std::vector<std::vector<float>>>& neg_embeddings,
                 const std::vector<std::vector<std::vector<float>>>& d_pos_embeddings,
                 const std::vector<std::vector<std::vector<float>>>& d_neg_embeddings) {
        
        // Reset gradients
        for(auto& row : weight_gradients) {
            std::fill(row.begin(), row.end(), 0.0f);
        }
        
        // Compute gradients with chain rule
        compute_gradients(pos_embeddings, d_pos_embeddings, true);
        compute_gradients(neg_embeddings, d_neg_embeddings, false);
        
        // Add gradient clipping to prevent explosion
        float max_grad_norm = 5.0f;
        float total_norm = 0.0f;
        for(const auto& row : weight_gradients) {
            for(float grad : row) {
                total_norm += grad * grad;
            }
        }
        total_norm = std::sqrt(total_norm);
        
        if(total_norm > max_grad_norm) {
            float scale = max_grad_norm / total_norm;
            for(auto& row : weight_gradients) {
                for(float& grad : row) {
                    grad *= scale;
                }
            }
        }
        
        // Update weights with momentum
        static std::vector<std::vector<float>> velocity(weights.size(), 
            std::vector<float>(weights[0].size(), 0.0f));
        float momentum = 0.9f;
        
        for(size_t i = 0; i < weights.size(); i++) {
            for(size_t j = 0; j < weights[i].size(); j++) {
                velocity[i][j] = momentum * velocity[i][j] + learning_rate * weight_gradients[i][j];
                weights[i][j] -= velocity[i][j];
            }
        }
    }

    void update_weights() {
        for(size_t i = 0; i < weights.size(); i++) {
            for(size_t j = 0; j < weights[i].size(); j++) {
                weights[i][j] -= learning_rate * weight_gradients[i][j];
            }
        }
    }
    
private:

    void compute_gradients(const std::vector<std::vector<std::vector<float>>>& embeddings,
                          const std::vector<std::vector<std::vector<float>>>& d_embeddings,
                          bool is_positive) {
        float scale = is_positive ? 1.0f : -1.0f;
        
        for(const auto& pair : (is_positive ? train_pos_g : train_neg_g).adjList) {
            int node = pair.first;
            const std::vector<int>& neighbors = pair.second;
            
            // Compute input gradients
            std::vector<std::vector<float>> node_features(223, std::vector<float>(1, 0.0f));
            std::vector<std::vector<float>> agg_features(223, std::vector<float>(1, 0.0f));
            
            // Get node features
            for(int i = 0; i < 223; i++) {
                node_features[i][0] = features[i][0][node];
            }
            
            // Aggregate neighbor features
            if(!neighbors.empty()) {
                for(int neighbor : neighbors) {
                    for(int i = 0; i < 223; i++) {
                        agg_features[i][0] += features[i][0][neighbor] / neighbors.size();
                    }
                }
            }
            
            // Concatenate features
            std::vector<std::vector<float>> combined = concat(node_features, agg_features);
            
            // Compute gradients w.r.t weights
            for(size_t i = 0; i < weights.size(); i++) {
                for(size_t j = 0; j < weights[i].size(); j++) {
                    weight_gradients[i][j] += scale * d_embeddings[i][0][node] * 
                                            combined[j][0] * 
                                            sigmoid_derivative(embeddings[i][0][node]);
                }
            }
        }
    }

    float sigmoid_derivative(float x) {
        float s = sigmoid(x);
        return s * (1.0f - s);
    }

    void processGraph(const Graph& graph, std::vector<std::vector<std::vector<float>>>& z_v) {
        for(auto& pair : graph.adjList) {
            int node = pair.first;
            std::vector<int> neighbors = pair.second;
            
            // Get node features
            std::vector<std::vector<float>> node_features(223, std::vector<float>(1, 0.0f));
            std::vector<std::vector<float>> agg_embedding(223, std::vector<float>(1, 0.0f));
            
            for (int i = 0; i < 223; i++) {
                node_features[i][0] = features[i][0][node];
            }
            
            // Aggregate neighbor features
            if (!neighbors.empty()) {
                for(int neighbor : neighbors) {
                    for(int i = 0; i < 223; i++) {
                        float neighbor_feature = features[i][0][neighbor];
                        agg_embedding[i][0] += neighbor_feature / neighbors.size();
                    }
                }
            }
            
            // Combine and transform
            std::vector<std::vector<float>> combined = concat(node_features, agg_embedding);
            std::vector<std::vector<float>> transformed = applyWeights(weights, combined);
            
            // Apply activation and normalization
            for(size_t i = 0; i < transformed.size(); i++) {
                transformed[i][0] = sigmoid(transformed[i][0]);
            }
            transformed = l2_normalization(transformed);
            
            // Store results
            for (size_t i = 0; i < transformed.size(); i++) {
                z_v[i][0][node] = transformed[i][0];
            }
        }
    }

    std::vector<std::vector<float>> l2_normalization(std::vector<std::vector<float>>& matrix) {
        float unit_v = 0.0f;
        for (size_t i = 0; i < matrix.size(); i++) {
            unit_v += matrix[i][0] * matrix[i][0];
        }
        unit_v = sqrt(unit_v);
        if (unit_v == 0) {
            return matrix;
        }
        for (size_t i = 0; i < matrix.size(); i++) {
            matrix[i][0] = matrix[i][0] / unit_v; 
        }
        return matrix;
    }
    
    float sigmoid (float x) {
        return 1.0 / (1 + exp(-x));
    }

    std::vector<std::vector<float>> applyWeights(const std::vector<std::vector<float>>& weights, const std::vector<std::vector<float>>& features) {
        std::vector<std::vector<float>> res(223, std::vector<float> (1, 0.0f));
        for(size_t i = 0; i < weights.size(); i++) {
            for (size_t j = 0; j < weights[0].size(); j++) {
                res[i][0] += weights[i][j] * features[j][0];
            }
        }
        return res;
    }

    std::vector<std::vector<float>> concat(const std::vector<std::vector<float>>& node_features, const std::vector<std::vector<float>>& agg_embedding) {
        std::vector<std::vector<float>> concatenated_features(446, std::vector<float> (1, 0.0f));
        for(size_t i = 0; i < node_features.size(); i++) {
            concatenated_features[i][0] = node_features[i][0];
        }
        for(size_t i = 0; i < agg_embedding.size() && i + node_features.size() < 446; i++) {
            concatenated_features[i + node_features.size()][0] = agg_embedding[i][0];
        }
        return concatenated_features;
    }

    std::vector<std::vector<float>> Xavier_initialization() {
        std::vector<std::vector<float>> weights(223, std::vector<float>(446, 0.0f));
        float lower = -1.0 / sqrt(446);
        float upper = 1.0 / sqrt(446);
        for(size_t i = 0; i < weights.size(); i++) {
            for(size_t j = 0; j < weights[0].size(); j++) {
                weights[i][j] = ((rand() / float(RAND_MAX)) * (upper - lower)) + lower;
            }
        }
        return weights;
    }
};


#endif