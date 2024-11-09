#include <iomanip>
#include <set>
#include <random>
#include "../include/utilityFunctions.h"
#include "../include/model.h"
#include "../include/predictor.h"
#include "../include/raylib.h"

float Vector2Distance(Vector2 p1, Vector2 p2) {
    return sqrt(((p1.x - p2.x) * (p1.x - p2.x)) + ((p1.y - p2.y) * (p1.y - p2.y)));
}

float Vector2Length(Vector2 p) {
    return sqrt((p.x * p.x) + (p.y * p.y));
}

float Clamp(float val, float min, float max) {
    return (val < min) ? min : (val > max) ? max : val;
}

struct Node {
    Vector2 position;
    bool isTestNode;
    int id;
    float radius = 20.0f;  // Larger nodes for better visibility
    Vector2 velocity = {0, 0};
};

// Force-directed layout helper functions
Vector2 calculateRepulsion(const Node& n1, const Node& n2) {
    Vector2 diff = {n1.position.x - n2.position.x, n1.position.y - n2.position.y};
    float dist = Vector2Length(diff);
    if (dist < 0.01f) dist = 0.01f;
    float force = 2000.0f / (dist * dist);  // Repulsion strength
    return {(diff.x / dist) * force, (diff.y / dist) * force};
}

Vector2 calculateAttraction(const Node& n1, const Node& n2) {
    Vector2 diff = {n2.position.x - n1.position.x, n2.position.y - n1.position.y};
    float dist = Vector2Length(diff);
    float force = (dist - 150.0f) * 0.05f;  // Spring constant
    return {(diff.x / dist) * force, (diff.y / dist) * force};
}

// New function to sample random test edges
std::vector<std::pair<int, int>> sampleRandomTestEdges(
    const std::vector<std::pair<int, int>>& test_edges,
    size_t sample_size) {
    
    if (test_edges.size() <= sample_size) {
        return test_edges;
    }

    std::vector<size_t> indices(test_edges.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    std::vector<std::pair<int, int>> sampled_edges;
    for (size_t i = 0; i < sample_size; ++i) {
        sampled_edges.push_back(test_edges[indices[i]]);
    }
    
    return sampled_edges;
}

void DrawGraph(const std::vector<std::pair<int, int>>& test_edges, 
               const std::vector<std::pair<int, float>>& recommendations,
               int maxNodeIndex) {
    const int SCREEN_WIDTH = 1778;
    const int SCREEN_HEIGHT = 1000;
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Graph Neural Network Visualization (10 Random Nodes)");
    SetTargetFPS(60);

    // Sample 10 random test edges
    auto sampled_edges = sampleRandomTestEdges(test_edges, 30);
    
    std::unordered_map<int, Node> nodes;
    
    // Initialize nodes with random positions
    std::set<int> uniqueTestNodes;
    for (const auto& edge : sampled_edges) {
        uniqueTestNodes.insert(edge.first);
        uniqueTestNodes.insert(edge.second);
    }

    // Random number generator for initial positions
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> disX(200.0f, SCREEN_WIDTH - 200.0f);
    std::uniform_real_distribution<float> disY(200.0f, SCREEN_HEIGHT - 200.0f);

    // Initialize test nodes
    for (int nodeId : uniqueTestNodes) {
        nodes[nodeId] = Node{
            Vector2{disX(gen), disY(gen)},
            true,
            nodeId
        };
    }

    // Add only recommendations for sampled nodes
    for (const auto& rec : recommendations) {
        if (uniqueTestNodes.find(rec.first) != uniqueTestNodes.end() && 
            nodes.find(rec.first) == nodes.end()) {
            nodes[rec.first] = Node{
                Vector2{disX(gen), disY(gen)},
                false,
                rec.first
            };
        }
    }

    // Force-directed layout simulation
    bool layoutStabilized = false;
    int stabilityCounter = 0;
    const int STABILITY_THRESHOLD = 100;

    while (!WindowShouldClose()) {
        // Update node positions using force-directed layout
        if (!layoutStabilized) {
            bool stable = true;
            
            // Calculate forces
            for (auto& [id1, node1] : nodes) {
                Vector2 totalForce = {0, 0};
                
                // Repulsion between all nodes
                for (const auto& [id2, node2] : nodes) {
                    if (id1 != id2) {
                        Vector2 force = calculateRepulsion(node1, node2);
                        totalForce.x += force.x;
                        totalForce.y += force.y;
                    }
                }
                
                // Attraction along edges
                for (const auto& edge : sampled_edges) {
                    if (edge.first == id1 && nodes.count(edge.second)) {
                        Vector2 force = calculateAttraction(node1, nodes[edge.second]);
                        totalForce.x += force.x;
                        totalForce.y += force.y;
                    }
                    if (edge.second == id1 && nodes.count(edge.first)) {
                        Vector2 force = calculateAttraction(node1, nodes[edge.first]);
                        totalForce.x += force.x;
                        totalForce.y += force.y;
                    }
                }
                
                // Update velocity and position
                node1.velocity.x = (node1.velocity.x + totalForce.x) * 0.9f;
                node1.velocity.y = (node1.velocity.y + totalForce.y) * 0.9f;
                
                node1.position.x += node1.velocity.x;
                node1.position.y += node1.velocity.y;
                
                // Keep nodes within screen bounds
                node1.position.x = Clamp(node1.position.x, 100.0f, SCREEN_WIDTH - 100.0f);
                node1.position.y = Clamp(node1.position.y, 100.0f, SCREEN_HEIGHT - 100.0f);
                
                if (fabs(node1.velocity.x) > 0.1f || fabs(node1.velocity.y) > 0.1f) {
                    stable = false;
                }
            }
            
            if (stable) stabilityCounter++;
            else stabilityCounter = 0;
            
            if (stabilityCounter >= STABILITY_THRESHOLD) {
                layoutStabilized = true;
            }
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);

        // Draw edges first (test edges in red)
        for (const auto& edge : sampled_edges) {
            if (nodes.find(edge.first) != nodes.end() && nodes.find(edge.second) != nodes.end()) {
                DrawLineEx(nodes[edge.first].position, nodes[edge.second].position, 2.0f, RED);
            }
        }

        // Draw recommendation edges in green
        for (const auto& rec : recommendations) {
            if (nodes.find(rec.first) != nodes.end()) {
                for (const auto& testNode : uniqueTestNodes) {
                    DrawLineEx(nodes[rec.first].position, nodes[testNode].position, 1.0f, 
                             Fade(GREEN, 0.3f));
                }
            }
        }

        // Draw nodes
        for (const auto& [id, node] : nodes) {
            Color nodeColor = node.isTestNode ? RED : GREEN;
            DrawCircleV(node.position, node.radius, nodeColor);
            DrawCircleLines(node.position.x, node.position.y, node.radius, BLACK);
            
            char idText[10];
            sprintf(idText, "%d", node.id);
            Vector2 textPosition = {
                node.position.x - MeasureText(idText, 20) / 2,
                node.position.y - 10
            };
            DrawText(idText, textPosition.x, textPosition.y, 20, WHITE);
        }

        // Draw legend
        DrawRectangle(10, 10, 250, 70, Fade(RAYWHITE, 0.9f));
        DrawText("Test Nodes (Red)", 20, 20, 20, RED);
        DrawText("Recommendations (Green)", 20, 45, 20, GREEN);

        EndDrawing();
    }

    CloseWindow();
}

int main() {
    try {
        std::vector<std::pair<int, int>> edges;
        std::vector<std::pair<int, int>> train_pos_edges, test_pos_edges;
        std::vector<std::pair<int, int>> train_neg_edges, test_neg_edges;
        std::vector<std::vector<std::vector<float>>> Features;
        
        loadDataAndFeatures(edges, Features);
        prepareTrainingData(edges, train_pos_edges, test_pos_edges, train_neg_edges, test_neg_edges);
        
        SAGEModel model;
        buildAndTrainModel(train_pos_edges, train_neg_edges, Features, model);
        
        std::cout << "\n=== Evaluating Model ===" << std::endl;
        evaluate_model(model, test_pos_edges, test_neg_edges);
        
        // Get recommendations for visualization
        std::vector<std::pair<int, float>> all_recommendations;
        // Get recommendations for each test node
        for (const auto& test_edge : test_pos_edges) {
            auto recommendations = model.get_recommendations(test_edge.first, 5);
            all_recommendations.insert(all_recommendations.end(), 
                                    recommendations.begin(), 
                                    recommendations.end());
        }
        
        // Visualize the graph
        DrawGraph(test_pos_edges, all_recommendations, findMaxNodeIndex(edges));
        
        std::cout << "\n=== Processing Complete ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\nUnknown error occurred" << std::endl;
        return 1;
    }
}

// int main() {
//     try {
//         std::vector<std::pair<int, int>> edges;
//         std::vector<std::pair<int, int>> train_pos_edges, test_pos_edges;
//         std::vector<std::pair<int, int>> train_neg_edges, test_neg_edges;
//         std::vector<std::vector<std::vector<float>>> Features;
//         loadDataAndFeatures(edges, Features);
//         prepareTrainingData(edges, train_pos_edges, test_pos_edges, train_neg_edges, test_neg_edges);
//         SAGEModel model;
//         buildAndTrainModel(train_pos_edges, train_neg_edges, Features, model);
//         std::cout << "\n=== Evaluating Model ===" << std::endl;
//         evaluate_model(model, test_pos_edges, test_neg_edges);
//         generateSampleRecommendations(model, findMaxNodeIndex(edges));
//         std::cout << "\n=== Processing Complete ===" << std::endl;
//         return 0;
//     } catch (const std::exception& e) {
//         std::cerr << "\nERROR: " << e.what() << std::endl;
//         return 1;
//     } catch (...) {
//         std::cerr << "\nUnknown error occurred" << std::endl;
//         return 1;
//     }
// }