#include <iomanip>
#include <set>
#include <cstring>
#include <random>
#include <unordered_map>
#include "include/raylib.h"
#include "include/Graph.h"
#include "include/Utility.h"
#include "include/Layer.h"
#include "include/Model.h"


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

bool IsMouseOverButton(Rectangle button) {
    return CheckCollisionPointRec(GetMousePosition(), button);
}


void DrawButton(Rectangle button, const char *text, const int textSize = 20, Color recColor = LIGHTGRAY, Color textColor = BLACK) {
    DrawRectangleRec(button, recColor);
    DrawRectangleLinesEx(button, 2, BLACK);
    DrawText(text, button.x + 10, button.y + 10, textSize, textColor);
}

void DrawTextBox(Rectangle textBox, const char *text, const int textSize = 20, Color recColor = LIGHTGRAY, Color textColor = BLACK) {
    DrawRectangleRec(textBox, recColor);
    DrawRectangleLinesEx(textBox, 2, BLACK);
    DrawText(text, textBox.x + 10, textBox.y + 10, textSize, textColor);
}

enum class Screens {
    Graph_view,
    List_view
};

void HandleTextInput(char* buffer, int maxSize, bool active) {
    if (active) {
        int key = GetKeyPressed();
        while (key > 0) {
            if ((key >= 32) && (key <= 125) && (strlen(buffer) < (long long unsigned)maxSize)) {
                bool shiftPressed = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
                char charToInsert = (char)key;

                // Handle shift for alphabetic characters
                if (shiftPressed && charToInsert >= 'a' && charToInsert <= 'z') {
                    charToInsert = (char)(charToInsert - 'a' + 'A');
                }
                // Handle shift for numeric characters and common symbols
                else if (shiftPressed) {
                    switch (charToInsert) {
                        case '1': charToInsert = '!'; break;
                        case '2': charToInsert = '@'; break;
                        case '3': charToInsert = '#'; break;
                        case '4': charToInsert = '$'; break;
                        case '5': charToInsert = '%'; break;
                        case '6': charToInsert = '^'; break;
                        case '7': charToInsert = '&'; break;
                        case '8': charToInsert = '*'; break;
                        case '9': charToInsert = '('; break;
                        case '0': charToInsert = ')'; break;
                        case '`': charToInsert = '~'; break;
                        case '-': charToInsert = '_'; break;
                        case '=': charToInsert = '+'; break;
                        case '[': charToInsert = '{'; break;
                        case ']': charToInsert = '}'; break;
                        case '\\': charToInsert = '|'; break;
                        case ';': charToInsert = ':'; break;
                        case '\'': charToInsert = '\"'; break;
                        case ',': charToInsert = '<'; break;
                        case '.': charToInsert = '>'; break;
                        case '/': charToInsert = '?'; break;
                    }
                }
                int len = strlen(buffer);
                buffer[len] = charToInsert;
                buffer[len + 1] = '\0';
            }
            if (key == KEY_BACKSPACE && strlen(buffer) > 0) {
                buffer[strlen(buffer) - 1] = '\0';
            }
            key = GetKeyPressed();
        }
    }
}

void DrawGraph(const unordered_map<int, vector<int>>& test_edges, 
               const std::vector<std::pair<int, float>>& recommendations,
               int maxNodeIndex) {
    const int SCREEN_WIDTH = 1000;
    const int SCREEN_HEIGHT = 1000;
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Graph Neural Network Visualization");
    SetTargetFPS(60);

    Screens screen;
    char testIdBuffer[16] = "";
    bool isTestIDBufferActive = true;
    int selected_test_id = -1;
    bool isGraphViewActive = true;
    bool isListViewActive = false;

    // Define UI elements
    Rectangle graphViewButton = { 100, 100, 200, 80 };
    Rectangle listViewButton  = { 100, 250, 200, 80 };
    Rectangle nodeIdTextBox = { 100, 250, 200, 80 };
    Rectangle enterButton  = { 400, 250, 100, 80 };
    
    // Define bounding box for graph
    Rectangle boundingBox = { 100, 400, SCREEN_WIDTH - 200, SCREEN_HEIGHT - 500 };

    const int ROW_HEIGHT = 40;

    /*const int COLUMN_PADDING = 20;
    Vector2 tableStart = { boundingBox.x + 20, boundingBox.y + 60 };
    const int RANK_WIDTH = 80;
    const int ID_WIDTH = 150;
    const int SCORE_WIDTH = 150;
    */

    float scrollOffset = 0;
    // const float MAX_VISIBLE_ROWS = (boundingBox.height - 100) / ROW_HEIGHT;

    std::unordered_map<int, Node> nodes;
    std::vector<std::pair<int, int>> filtered_edges;
    std::vector<std::pair<int, float>> filtered_recommendations;
    
    // Random number generator for node positions
    std::random_device rd;
    std::mt19937 gen(rd());

    auto initializeGraph = [&](int test_id) {
        nodes.clear();
        filtered_edges.clear();
        filtered_recommendations.clear();

        // Filter edges connected to the test_id
    for (const auto& [node, adjacent_nodes] : test_edges) {
        if (node == test_id) {
            // Add edges where the test_id is the source node
            for (int adjacent_node : adjacent_nodes) {
                filtered_edges.push_back(std::make_pair(node, adjacent_node));
            }
        } else {
            // Check if test_id is in the adjacency list of this node
            auto it = std::find(adjacent_nodes.begin(), adjacent_nodes.end(), test_id);
            if (it != adjacent_nodes.end()) {
                // Add edge where the test_id is the destination node
                filtered_edges.push_back(std::make_pair(node, test_id));
            }
        }
    }

        for (const auto& rec : recommendations) {
            if (rec.first == test_id) {
                filtered_recommendations.push_back(rec);
            }
        }
        std::sort(filtered_recommendations.begin(), filtered_recommendations.end(), [](const auto& a, const auto& b) {return a.second > b.second;});
        
        // Initialize nodes with positions within bounding box
        std::uniform_real_distribution<float> disX(boundingBox.x + 50, boundingBox.x + boundingBox.width - 50);
        std::uniform_real_distribution<float> disY(boundingBox.y + 50, boundingBox.y + boundingBox.height - 50);
        
        // Add test node
        nodes[test_id] = Node{
            Vector2{disX(gen), disY(gen)},
            true,
            test_id
        };
        
        // Add connected nodes from test edges
        for (const auto& edge : filtered_edges) {
            int connected_id = (edge.first == test_id) ? edge.second : edge.first;
            if (nodes.find(connected_id) == nodes.end()) {
                nodes[connected_id] = Node{
                    Vector2{disX(gen), disY(gen)},
                    true,
                    connected_id
                };
            }
        }
        
        // Add recommendations for the test node
        for (const auto& rec : recommendations) {
            if (rec.first == test_id && nodes.find(rec.first) == nodes.end()) {
                nodes[rec.first] = Node{
                    Vector2{disX(gen), disY(gen)},
                    false,
                    rec.first
                };
            }
        }
};

    bool layoutStabilized = false;
    int stabilityCounter = 0;
    const int STABILITY_THRESHOLD = 100;

    while (!WindowShouldClose()) {
        if (isListViewActive && selected_test_id != -1) {
            float wheel = GetMouseWheelMove();
            if (wheel != 0) {
                scrollOffset -= wheel * 30;
                // Clamp scrolling
                float maxScroll = std::max(0.0f, 
                    filtered_recommendations.size() * ROW_HEIGHT - (boundingBox.height - 100));
                scrollOffset = Clamp(scrollOffset, 0, maxScroll);
            }
        }

        HandleTextInput(testIdBuffer, 16, isTestIDBufferActive);

        if (IsMouseOverButton(enterButton) && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
            int new_test_id = atoi(testIdBuffer);
            if (new_test_id != selected_test_id && new_test_id > 0) {
                selected_test_id = new_test_id;
                initializeGraph(selected_test_id);
                layoutStabilized = false;
                stabilityCounter = 0;
                scrollOffset = 0; // Reset scroll when new ID is selected
            }
        }


        if (isGraphViewActive) {
            screen = Screens::Graph_view;
        } else {
            screen = Screens::List_view;
        }

        // Handle view switching
        if (IsMouseOverButton(graphViewButton) && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
            screen = Screens::Graph_view;
            isGraphViewActive = true;
            isListViewActive = false;
        }
        if (IsMouseOverButton(listViewButton) && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
            screen = Screens::List_view;
            isGraphViewActive = false;
            isListViewActive = true;
        }

        // Update node positions using force-directed layout
        if (!layoutStabilized && selected_test_id != -1) {
            bool stable = true;
            
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
                for (const auto& edge : filtered_edges) {
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
                
                // Update position and constrain to bounding box
                node1.position.x += node1.velocity.x;
                node1.position.y += node1.velocity.y;
                
                node1.position.x = Clamp(node1.position.x, 
                                       boundingBox.x + node1.radius,
                                       boundingBox.x + boundingBox.width - node1.radius);
                node1.position.y = Clamp(node1.position.y,
                                       boundingBox.y + node1.radius,
                                       boundingBox.y + boundingBox.height - node1.radius);
                
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
        
        // Draw UI elements
        DrawButton(graphViewButton, "Graph View", 20, (isGraphViewActive? GRAY: WHITE));
        // DrawButton(listViewButton, "List View", 20, (isListViewActive? GRAY: WHITE));
        DrawTextBox(nodeIdTextBox, testIdBuffer, 20, isTestIDBufferActive? LIGHTGRAY : GRAY);
        DrawButton(enterButton, "Enter", 20, GRAY);
        DrawText("Please enter an ID:", 100, 210, 32, BLACK);
        
        // Draw bounding box
        DrawRectangleLines(boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height, BLACK);

        /*if (screen == Screens::List_view) {
            DrawText("LIST VIEW", SCREEN_WIDTH / 2 - 100, 100, 32, BLACK);
            if (selected_test_id != -1) {
                // Draw table header
                DrawText("Recommendations for Node ID: ", boundingBox.x + 20, boundingBox.y + 20, 20, BLACK);
                DrawText(TextFormat("%d", selected_test_id), boundingBox.x + 250, boundingBox.y + 20, 20, RED);
                
                // Draw column headers
                Vector2 headerPos = tableStart;
                DrawText("Rank", headerPos.x, headerPos.y, 20, DARKGRAY);
                DrawText("Node ID", headerPos.x + RANK_WIDTH + COLUMN_PADDING, headerPos.y, 20, DARKGRAY);
                DrawText("Score", headerPos.x + RANK_WIDTH + ID_WIDTH + 2*COLUMN_PADDING, headerPos.y, 20, DARKGRAY);
                
                // Draw horizontal line under headers
                DrawLineEx(
                    Vector2{boundingBox.x + 20, tableStart.y + 30},
                    Vector2{boundingBox.x + boundingBox.width - 40, tableStart.y + 30},
                    2,
                    DARKGRAY
                );

                // Enable scissor mode to clip table content
                BeginScissorMode(boundingBox.x, tableStart.y + 40, 
                               boundingBox.width - 40, boundingBox.height - 100);

                // Draw table rows
                for (size_t i = 0; i < filtered_recommendations.size(); i++) {
                    float rowY = tableStart.y + 40 + (i * ROW_HEIGHT) - scrollOffset;
                    
                    // Only draw visible rows
                    if (rowY >= tableStart.y && rowY <= boundingBox.y + boundingBox.height - ROW_HEIGHT) {
                        // Rank
                        DrawText(TextFormat("%d", i + 1),
                                tableStart.x, rowY + 10, 20, BLACK);
                        
                        // Node ID
                        DrawText(TextFormat("%d", filtered_recommendations[i].first),
                                tableStart.x + RANK_WIDTH + COLUMN_PADDING, 
                                rowY + 10, 20, BLACK);
                        
                        // Score
                        DrawText(TextFormat("%.4f", filtered_recommendations[i].second),
                                tableStart.x + RANK_WIDTH + ID_WIDTH + 2*COLUMN_PADDING,
                                rowY + 10, 20, BLACK);
                        
                        // Row separator
                        DrawLineEx(
                            Vector2{boundingBox.x + 20, rowY + ROW_HEIGHT},
                            Vector2{boundingBox.x + boundingBox.width - 40, rowY + ROW_HEIGHT},
                            1,
                            LIGHTGRAY
                        );
                    }
                }

                EndScissorMode();

                // Draw scroll bar if needed
                if (filtered_recommendations.size() > MAX_VISIBLE_ROWS) {
                    float scrollBarHeight = (boundingBox.height - 100) * (MAX_VISIBLE_ROWS / filtered_recommendations.size());
                    float scrollBarY = boundingBox.y + 60 + (scrollOffset / (filtered_recommendations.size() * ROW_HEIGHT)) * (boundingBox.height - 100 - scrollBarHeight);
                    DrawRectangle(boundingBox.x + boundingBox.width - 20, boundingBox.y + 60, 10, boundingBox.height - 100, LIGHTGRAY);
                    DrawRectangle(boundingBox.x + boundingBox.width - 20, scrollBarY, 10, scrollBarHeight, GRAY);
                }

            } else {
                DrawText("Enter a test ID to view recommendations", SCREEN_WIDTH / 2 - 300, SCREEN_HEIGHT / 2, 30, DARKGRAY);
            }
        } */ if (screen == Screens::Graph_view) {
            DrawText("GRAPH VIEW", SCREEN_WIDTH / 2 - 100, 100, 32, BLACK);
            
            if (selected_test_id != -1) {
                // Draw edges
                for (const auto& edge : filtered_edges) {
                    if (nodes.find(edge.first) != nodes.end() && nodes.find(edge.second) != nodes.end()) {
                        DrawLineEx(nodes[edge.first].position, nodes[edge.second].position, 2.0f, RED);
                    }
                }

                // Draw recommendation edges
                for (const auto& rec : recommendations) {
                    if (rec.first == selected_test_id && nodes.find(rec.first) != nodes.end()) {
                        DrawLineEx(nodes[rec.first].position, nodes[selected_test_id].position, 1.0f, Fade(GREEN, 0.3f));
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
                // DrawText("Test Nodes (Red)", 20, 20, 20, RED);
                // DrawText("Recommendations (Green)", 20, 45, 20, GREEN);
            } else {
                DrawText("Enter a test ID to view the graph", 
                        SCREEN_WIDTH / 2 - 300, SCREEN_HEIGHT / 2, 
                        30, DARKGRAY);
            }
        }

        EndDrawing();
    }

    CloseWindow();
}

int main() {
    try {
        unordered_map<int, vector<int>> edges;
        unordered_map<int, vector<int>> train_pos_edges, test_pos_edges;
        unordered_map<int, vector<int>> train_neg_edges, test_neg_edges;
        unordered_map<int, vector<vector<float>>> Features;
    
        loadDataAndFeatures(edges, Features);
        prepareTrainingData(edges, train_pos_edges, test_pos_edges, train_neg_edges, test_neg_edges);
        
        SAGEModel model(train_pos_edges, train_neg_edges, Features);
        
        std::cout << "\n=== Evaluating Model ===" << std::endl;
        float auc = model.evaluate(test_pos_edges, test_neg_edges);
        cout << "AUC Score: " << auc << endl;

        // Get recommendations for visualization
        std::vector<std::pair<int, float>> all_recommendations;
        // Get recommendations for each test node
        for (const auto& test_edge : test_pos_edges) {
            auto recommendations = model.getPrediction(test_edge.first);
            all_recommendations.insert(all_recommendations.end(), recommendations.begin(), recommendations.end());
        }

        // Filter and sort the recommendations
        std::unordered_map<int, float> filtered_recommendations;
        for (const auto& rec : all_recommendations) {
            if (filtered_recommendations.find(rec.first) == filtered_recommendations.end() || rec.second > filtered_recommendations[rec.first]) {
                filtered_recommendations[rec.first] = rec.second;
            }
        }

        std::vector<std::pair<int, float>> sorted_recommendations;
        for (const auto& [node_id, score] : filtered_recommendations) {
            sorted_recommendations.emplace_back(node_id, score);
        }
        std::sort(sorted_recommendations.begin(), sorted_recommendations.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        
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