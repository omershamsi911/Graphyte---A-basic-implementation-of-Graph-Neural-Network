#ifndef MODEL_H
#define MODEL_H

#include <numeric>
#include <algorithm>
#include <limits>
#include "layer.h"

class SAGEModel {
private:
    Graph train_pos_g;
    Graph train_neg_g;
    std::vector<std::vector<std::vector<float>>> features;
    std::vector<std::vector<float>> embeddings;
    float Q = 5.0f; // Negative sampling ratio, adjust as needed
    float epsilon = 1e-7f; // Small constant to prevent log(0)

    float sigmoid (float x) {
        return 1.0 / (1 + exp(-x));
    }

    float calculate_loss(const std::vector<std::vector<std::vector<float>>>& pos_embeddings,
                        const std::vector<std::vector<std::vector<float>>>& neg_embeddings) {
        float loss = 0.0f;
        int pos_edges = 0, neg_edges = 0;
        
        // Process positive edges
        for(const auto& pair : train_pos_g.adjList) {
            int u = pair.first;
            for(int v : pair.second) {
                std::vector<float> z_u(223);
                std::vector<float> z_v(223);
                
                for(int i = 0; i < 223; i++) {
                    z_u[i] = pos_embeddings[i][0][u];
                    z_v[i] = pos_embeddings[i][0][v];
                }
                
                float dot_product = 0.0f;
                for(int i = 0; i < 223; i++) {
                    dot_product += z_u[i] * z_v[i];
                }
                
                // Clip dot product to prevent numerical instability
                dot_product = std::max(std::min(dot_product, 10.0f), -10.0f);
                loss -= std::log(sigmoid(dot_product) + epsilon);
                pos_edges++;
            }
        }
        
        // Process negative edges
        for(const auto& pair : train_neg_g.adjList) {
            int u = pair.first;
            for(int v : pair.second) {
                std::vector<float> z_u(223);
                std::vector<float> z_v(223);
                
                for(int i = 0; i < 223; i++) {
                    z_u[i] = neg_embeddings[i][0][u];
                    z_v[i] = neg_embeddings[i][0][v];
                }
                
                float dot_product = 0.0f;
                for(int i = 0; i < 223; i++) {
                    dot_product += z_u[i] * z_v[i];
                }
                
                // Clip dot product to prevent numerical instability
                dot_product = std::max(std::min(dot_product, 10.0f), -10.0f);
                loss -= Q * std::log(sigmoid(-dot_product) + epsilon);
                neg_edges++;
            }
        }
        
        // Normalize loss by number of edges
        return loss / (pos_edges + neg_edges);
    }

    std::vector<std::vector<std::vector<float>>> compute_embedding_gradients(
        const std::vector<std::vector<std::vector<float>>>& embeddings,
        bool is_positive) {
        
        std::vector<std::vector<std::vector<float>>> gradients(
            223, std::vector<std::vector<float>>(1, std::vector<float>(features[0][0].size(), 0.0f)));
        
        const Graph& graph = is_positive ? train_pos_g : train_neg_g;
        float scale = is_positive ? 1.0f : -1.0f;
        
        for(const auto& pair : graph.adjList) {
            int u = pair.first;
            for(int v : pair.second) {
                std::vector<float> z_u(223), z_v(223);
                
                for(int i = 0; i < 223; i++) {
                    z_u[i] = embeddings[i][0][u];
                    z_v[i] = embeddings[i][0][v];
                }
                
                float dot_product = 0.0f;
                for(int i = 0; i < 223; i++) {
                    dot_product += z_u[i] * z_v[i];
                }
                
                // Compute gradient
                float sigmoid_term = sigmoid(scale * dot_product);
                for(int i = 0; i < 223; i++) {
                    gradients[i][0][u] += scale * (1 - sigmoid_term) * z_v[i];
                    gradients[i][0][v] += scale * (1 - sigmoid_term) * z_u[i];
                }
            }
        }
        
        return gradients;
    }
    
    float validate() {
        // Placeholder for validation
        return 0.0f;
    }

    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
        float dot_product = 0.0f;
        float mag_a = 0.0f;
        float mag_b = 0.0f;
        for (size_t i = 0; i < a.size(); i++) {
            dot_product += a[i] * b[i];
            mag_a += a[i] * a[i];
            mag_b += b[i] * b[i];
        }
        mag_a = std::sqrt(mag_a);
        mag_b = std::sqrt(mag_b);
        return (mag_a * mag_b == 0) ? 0 : dot_product / (mag_a * mag_b);
    }

public:
    SAGEModel(Graph& train_pos_g, Graph& train_neg_g, std::vector<std::vector<std::vector<float>>>& Features)
        : train_pos_g(train_pos_g), train_neg_g(train_neg_g), features(Features) {
        std::cout << "SAGEModel initialized with feature dimensions: " << features.size() << " x " << features[0].size() << " x " << features[0][0].size() << std::endl;
    }

    SAGEModel() {}

    std::vector<std::pair<int, float>> get_all_similarities(int node_id) {
        if (embeddings.empty()) {
            std::cout << "Error: Embeddings not generated. Run training first." << std::endl;
            return {};
        }
        if (node_id < 0 || node_id >= static_cast<int>(embeddings.size())) {
            std::cout << "Error: Invalid node_id" << std::endl;
            return {};
        }
        
        std::vector<std::pair<int, float>> scores;
        for (size_t i = 0; i < embeddings.size(); i++) {
            if (static_cast<int>(i) != node_id) {
                float similarity = cosine_similarity(embeddings[node_id], embeddings[i]);
                scores.push_back({i, similarity});
            }
        }
        return scores;
    }

    void train(int num_epochs = 10) {
        std::cout << "Starting training for " << num_epochs << " epochs..." << std::endl;
        auto initial_features = features;
        float prev_loss = std::numeric_limits<float>::max();
        float learning_rate = 0.001f; // Reduced initial learning rate
        
        // Add early stopping
        int patience = 5;
        int no_improvement = 0;
        float best_loss = std::numeric_limits<float>::max();
        
        for (int epoch = 1; epoch <= num_epochs; epoch++) {
            std::cout << "\nEpoch " << epoch << "/" << num_epochs << ":" << std::endl;
            
            // Forward pass
            SAGELayer layer1(train_pos_g, train_neg_g, features, learning_rate);
            auto [layer1_pos_output, layer1_neg_output] = layer1.forward();
            
            SAGELayer layer2(train_pos_g, train_neg_g, layer1_pos_output, learning_rate);
            auto [layer2_pos_output, layer2_neg_output] = layer2.forward();
            
            float current_loss = calculate_loss(layer2_pos_output, layer2_neg_output);
            
            // Early stopping check
            if(current_loss < best_loss - 1e-4) {
                best_loss = current_loss;
                no_improvement = 0;
            } else {
                no_improvement++;
                if(no_improvement >= patience) {
                    std::cout << "Early stopping triggered!" << std::endl;
                    break;
                }
            }
            
            // Backward pass with proper gradient scaling
            std::vector<std::vector<std::vector<float>>> d_pos_embeddings = 
                compute_embedding_gradients(layer2_pos_output, true);
            std::vector<std::vector<std::vector<float>>> d_neg_embeddings = 
                compute_embedding_gradients(layer2_neg_output, false);
            
            // Scale gradients based on batch size
            float scale = 1.0f / (train_pos_g.adjList.size() + train_neg_g.adjList.size());
            for(auto& grad_matrix : d_pos_embeddings) {
                for(auto& grad_vector : grad_matrix) {
                    for(float& grad : grad_vector) {
                        grad *= scale;
                    }
                }
            }
            for(auto& grad_matrix : d_neg_embeddings) {
                for(auto& grad_vector : grad_matrix) {
                    for(float& grad : grad_vector) {
                        grad *= scale;
                    }
                }
            }
            
            // Backward pass through layers
            layer2.backward(layer2_pos_output, layer2_neg_output, 
                          d_pos_embeddings, d_neg_embeddings);
            layer1.backward(layer1_pos_output, layer1_neg_output, 
                          d_pos_embeddings, d_neg_embeddings);
            
            // Update features with skip connection and normalization
            for (size_t i = 0; i < features.size(); i++) {
                for (size_t j = 0; j < features[0].size(); j++) {
                    for (size_t k = 0; k < features[0][0].size(); k++) {
                        features[i][j][k] = 0.8f * layer2_pos_output[i][j][k] + 
                                          0.2f * initial_features[i][j][k];
                    }
                }
            }
            
            // Adaptive learning rate with more conservative adjustment
            if (current_loss > prev_loss) {
                learning_rate *= 0.95f;
            }
            prev_loss = current_loss;
            
            std::cout << "Loss: " << current_loss << ", Learning Rate: " << learning_rate << std::endl;
        }
        
        std::cout << "\nTraining completed!" << std::endl;
        generate_embeddings();
    }


    void generate_embeddings() {
        embeddings.clear();
        embeddings.resize(features[0][0].size(), std::vector<float>(features.size()));
        for (size_t i = 0; i < features.size(); i++) {
            for (size_t k = 0; k < features[0][0].size(); k++) {
                embeddings[k][i] = features[i][0][k];
            }
        }
        std::cout << "Embeddings generated with dimensions: " << embeddings.size() << " x " << embeddings[0].size() << std::endl;
    }

    std::vector<std::pair<int, float>> get_recommendations(int node_id, int top_k = 5) {
        if (embeddings.empty()) {
            std::cout << "Error: Embeddings not generated. Run training first." << std::endl;
            return {};
        }
        if (node_id < 0 || node_id >= static_cast<int>(embeddings.size())) {
            std::cout << "Error: Invalid node_id" << std::endl;
            return {};
        }
        std::vector<std::pair<int, float>> scores;
        for (size_t i = 0; i < embeddings.size(); i++) {
            if (static_cast<int>(i) != node_id) {
                float similarity = cosine_similarity(embeddings[node_id], embeddings[i]);
                scores.push_back({i, similarity});
            }
        }
        std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        return std::vector<std::pair<int, float>>(scores.begin(), scores.begin() + std::min(top_k, static_cast<int>(scores.size())));
    }
};

void buildAndTrainModel(std::vector<std::pair<int, int>>& train_pos_edges, std::vector<std::pair<int, int>>& train_neg_edges, std::vector<std::vector<std::vector<float>>>& Features, SAGEModel& model) {
    std::cout << "\n=== Building Graphs ===" << std::endl;
    Graph train_pos_g(train_pos_edges);
    Graph train_neg_g(train_neg_edges);
    std::cout << "Graphs created successfully" << std::endl;
    
    std::cout << "\n=== Training Model ===" << std::endl;
    model = SAGEModel(train_pos_g, train_neg_g, Features);
    int num_epochs = 5;
    std::cout << "Starting training with " << num_epochs << " epochs..." << std::endl;
    model.train(num_epochs);
}


#endif