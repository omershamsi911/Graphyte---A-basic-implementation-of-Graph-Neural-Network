#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "model.h"

class LinkPredictionEvaluator {
public:
    struct EvalMetrics {
        float auc;
        float mrr;
        float hits_at_k;
    };

    static EvalMetrics evaluate(SAGEModel& model, 
                              const std::vector<std::pair<int, int>>& test_pos_edges,
                              const std::vector<std::pair<int, int>>& test_neg_edges,
                              int k = 10) {
        std::vector<float> scores;
        std::vector<int> labels;  // 1 for positive edges, 0 for negative edges
        float mrr = 0.0f;
        float hits = 0.0f;
        
        // Process positive edges
        for (const auto& edge : test_pos_edges) {
            // Get similarity scores for all nodes for this source node
            auto all_recommendations = model.get_all_similarities(edge.first);
            
            // Store score for the positive edge
            float pos_score = -1;
            for (const auto& rec : all_recommendations) {
                if (rec.first == edge.second) {
                    pos_score = rec.second;
                    break;
                }
            }
            
            // Calculate rank of positive edge
            int rank = 1;
            for (const auto& rec : all_recommendations) {
                if (rec.second > pos_score && rec.first != edge.first) {
                    rank++;
                }
            }
            
            // Update MRR
            if (rank != 0) {
                mrr += 1.0f / rank;
            }
            
            // Update Hits@K
            if (rank <= k) {
                hits += 1;
            }
            
            // Store score for AUC calculation
            scores.push_back(pos_score);
            labels.push_back(1);
        }
        
        // Process negative edges
        for (const auto& edge : test_neg_edges) {
            float neg_score = -1;
            auto all_recommendations = model.get_all_similarities(edge.first);
            
            for (const auto& rec : all_recommendations) {
                if (rec.first == edge.second) {
                    neg_score = rec.second;
                    break;
                }
            }
            
            scores.push_back(neg_score);
            labels.push_back(0);
        }
        
        // Calculate metrics
        float auc = calculate_auc(scores, labels);
        mrr = mrr / test_pos_edges.size();
        hits = hits / test_pos_edges.size();
        
        return {auc, mrr, hits};
    }

private:
    static float calculate_auc(const std::vector<float>& scores, const std::vector<int>& labels) {
        int n_pos = std::count(labels.begin(), labels.end(), 1);
        int n_neg = labels.size() - n_pos;
        
        std::vector<size_t> indices(scores.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                 [&scores](size_t a, size_t b) { return scores[a] > scores[b]; });
        
        int rank_sum = 0;
        for (size_t i = 0; i < indices.size(); i++) {
            if (labels[indices[i]] == 1) {
                rank_sum += i + 1;
            }
        }
        
        float auc = (rank_sum - (n_pos * (n_pos + 1) / 2.0f)) / (n_pos * n_neg);
        return auc;
    }
};


void evaluate_model(SAGEModel& model, 
                   const std::vector<std::pair<int, int>>& test_pos_edges,
                   const std::vector<std::pair<int, int>>& test_neg_edges) {
    auto metrics = LinkPredictionEvaluator::evaluate(model, test_pos_edges, test_neg_edges);
    
    std::cout << "Evaluation Results:" << std::endl;
    std::cout << "AUC: " << metrics.auc << std::endl;
    std::cout << "MRR: " << metrics.mrr << std::endl;
    std::cout << "Hits@10: " << metrics.hits_at_k << std::endl;
}

void generateSampleRecommendations(SAGEModel& model, int maxNodeIndex) {
    std::cout << "\n=== Sample Recommendations ===" << std::endl;
    int num_samples = 5;
    for (int i = 0; i < num_samples; i++) {
        int node_id = rand() % (maxNodeIndex + 1);
        std::cout << "\nRecommendations for node " << node_id << ":" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        auto recommendations = model.get_recommendations(node_id, 5);
        if (recommendations.empty()) {
            std::cout << "No recommendations available for node " << node_id << std::endl;
            continue;
        }
        
        std::cout << "Top 5 recommended nodes:" << std::endl;
        for (const auto& rec : recommendations) {
            std::cout << "Node " << rec.first << " (Similarity Score: " << std::fixed << std::setprecision(4) << rec.second << ")" << std::endl;
        }
    }
}


#endif