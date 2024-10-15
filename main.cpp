#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>
#include <cctype>
#include <ctime>
#include <random>
#include <cmath>
#include <iomanip>

const int MAX_USERS = 1000;

struct User {
    int id;
    std::string name;
    std::string gender;
    unsigned int age;
    std::set<std::string> interests;
    std::string city;
    std::string country;
    std::vector<int> connections;
};

int calculateAge(const std::string& dobStr) {
    int birthYear;
    std::istringstream iss(dobStr.substr(0, 4));
    iss >> birthYear;
    time_t t = time(0);
    tm* now = localtime(&t);
    int currentYear = now->tm_year + 1900;
    return currentYear - birthYear;
}

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r\f\v");
    size_t last = str.find_last_not_of(" \t\n\r\f\v");
    return (first == std::string::npos) ? "" : str.substr(first, (last - first + 1));
}

std::vector<std::string> splitInterests(const std::string& interestsStr) {
    std::vector<std::string> interests;
    std::string trimmedInterests = trim(interestsStr);
    if (trimmedInterests.front() == '"' && trimmedInterests.back() == '"') {
        trimmedInterests = trimmedInterests.substr(1, trimmedInterests.size() - 2);
    }
    std::istringstream iss(trimmedInterests);
    std::string interest;
    while (std::getline(iss, interest, ',')) {
        interest = trim(interest);
        if (!interest.empty()) {
            if (interest.front() == '\'' && interest.back() == '\'') {
                interest = interest.substr(1, interest.length() - 2);
            }
            interests.push_back(interest);
        }
    }
    return interests;
}

std::string getField(std::istringstream& iss) {
    std::string field;
    char ch;
    bool inQuotes = false;
    std::ostringstream oss;
    while (iss.get(ch)) {
        if (ch == '"' && !inQuotes) {
            inQuotes = true;
        } else if (ch == '"' && inQuotes) {
            if (iss.peek() == ',') {
                iss.get();
                break;
            } else {
                inQuotes = false;
            }
        } else if (ch == ',' && !inQuotes) {
            break;
        } else {
            oss << ch;
        }
    }
    field = oss.str();
    return trim(field);
}

std::vector<User> readCSV(const std::string& filename) {
    std::vector<User> users;
    std::ifstream file(filename);
    std::string line;
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return users;
    }
    std::getline(file, line);
    while (std::getline(file, line) && users.size() < MAX_USERS) {
        std::istringstream iss(line);
        User user;
        user.id = std::stoi(getField(iss));
        user.name = getField(iss);
        user.gender = getField(iss);
        std::string dobStr = getField(iss);
        user.age = calculateAge(dobStr);
        std::string interestsStr = getField(iss);
        std::vector<std::string> interestsList = splitInterests(interestsStr);
        user.interests.insert(interestsList.begin(), interestsList.end());
        user.city = getField(iss);
        user.country = getField(iss);
        users.push_back(user);
    }
    return users;
}

void createAdjacencyList(std::vector<User>& users) {
    for (size_t i = 0; i < users.size(); ++i) {
        for (size_t j = i + 1; j < users.size(); ++j) {
            std::vector<std::string> sharedInterests;
            std::set_intersection(users[i].interests.begin(), users[i].interests.end(),
                                  users[j].interests.begin(), users[j].interests.end(),
                                  std::back_inserter(sharedInterests));

            if (!sharedInterests.empty()) {
                users[i].connections.push_back(j);
                users[j].connections.push_back(i);
            }
        }
    }
}

class Graph {
public:
    Graph(int num_nodes) : adj_list(num_nodes) {}

    void add_edge(int u, int v) {
        adj_list[u].push_back(v);
        adj_list[v].push_back(u);
    }

    const std::vector<int>& neighbors(int node) const {
        return adj_list[node];
    }

    int num_nodes() const {
        return adj_list.size();
    }

    std::vector<std::vector<float>> normalize_adjacency() const {
        std::vector<std::vector<float>> normalized_adj(num_nodes());
        std::vector<float> degree(num_nodes(), 0.0f);
        for (int i = 0; i < num_nodes(); ++i) {
            degree[i] = adj_list[i].size();
        }
        for (int i = 0; i < num_nodes(); ++i) {
            for (int j : adj_list[i]) {
                float norm_value = 1.0f / std::sqrt(degree[i] * degree[j]);
                normalized_adj[i].push_back(norm_value);
            }
        }
        return normalized_adj;
    }
private:
    std::vector<std::vector<int>> adj_list;
};

class GCNLayer {
public:
    GCNLayer(int input_dim, int output_dim) : weights(input_dim, std::vector<float>(output_dim)) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 1);
        for (int i = 0; i < input_dim; ++i) {
            for (int j = 0; j < output_dim; ++j) {
                weights[i][j] = d(gen) / std::sqrt(input_dim);
            }
        }
    }

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& adj) {
        std::vector<std::vector<float>> output(input.size(), std::vector<float>(weights[0].size(), 0.0f));
        for (size_t i = 0; i < input.size(); ++i) {
            for (size_t j = 0; j < adj[i].size(); ++j) {
                for (size_t k = 0; k < weights[0].size(); ++k) {
                    for (size_t l = 0; l < input[0].size(); ++l) {
                        output[i][k] += adj[i][j] * input[j][l] * weights[l][k];
                    }
                }
            }
        }
        for (auto& row : output) {
            for (auto& val : row) {
                val = std::max(0.0f, val);
            }
        }
        return output;
    }

    void backward(const std::vector<std::vector<float>>& grad_output, 
                  const std::vector<std::vector<float>>& input, 
                  float learning_rate) {
        // Gradient for the weights
        std::vector<std::vector<float>> grad_weights(weights.size(), 
                                                    std::vector<float>(weights[0].size(), 0.0f));
        for (size_t i = 0; i < input.size(); ++i) {
            for (size_t j = 0; j < weights[0].size(); ++j) {
                for (size_t k = 0; k < input[0].size(); ++k) {
                    grad_weights[k][j] += input[i][k] * grad_output[i][j];
                }
            }
        }
        
        // Update weights using gradient descent
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[0].size(); ++j) {
                weights[i][j] -= learning_rate * grad_weights[i][j];
            }
        }
    }

    std::vector<std::vector<float>> weights;
};

class GCN {
public:
    GCN(int input_dim, int hidden_dim, int output_dim)
    : layer1(input_dim, hidden_dim), layer2(hidden_dim, output_dim) {}

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input, 
                                            const std::vector<std::vector<float>>& adj) {
        h1 = layer1.forward(input, adj);
        output = layer2.forward(h1, adj);
        apply_softmax(output);
        return output;
    }

    void backward(const std::vector<float>& labels, float learning_rate) {
        // Compute error (output - labels)
        std::vector<std::vector<float>> grad_output(output.size(), std::vector<float>(output[0].size(), 0.0f));
        for (size_t i = 0; i < output.size(); ++i) {
            for (size_t j = 0; j < output[0].size(); ++j) {
                grad_output[i][j] = output[i][j] - labels[i]; // For binary cross-entropy loss
            }
        }
        layer2.backward(grad_output, h1, learning_rate);
        std::vector<std::vector<float>> grad_h1 = compute_grad_h1(grad_output, layer2.weights);
        layer1.backward(grad_h1, input, learning_rate);
    }

    void apply_softmax(std::vector<std::vector<float>>& output) {
        for (auto& row : output) {
            float max_val = *std::max_element(row.begin(), row.end());
            float sum = 0.0f;
            for (auto& val : row) {
                val = std::exp(val - max_val);
                sum += val;
            }
            for (auto& val : row) {
                val /= sum;
            }
        }
    }

    std::vector<std::vector<float>> compute_grad_h1(const std::vector<std::vector<float>>& grad_output, 
                                                    const std::vector<std::vector<float>>& weights) {
        std::vector<std::vector<float>> grad_h1(grad_output.size(), std::vector<float>(weights.size(), 0.0f));
        for (size_t i = 0; i < grad_output.size(); ++i) {
            for (size_t j = 0; j < weights.size(); ++j) {
                for (size_t k = 0; k < grad_output[0].size(); ++k) {
                    grad_h1[i][j] += grad_output[i][k] * weights[j][k];
                }
            }
        }
        return grad_h1;
    }

    GCNLayer layer1;
    GCNLayer layer2;
    std::vector<std::vector<float>> h1;
    std::vector<std::vector<float>> output;
    std::vector<std::vector<float>> input;
};

std::vector<std::vector<float>> create_node_features(const std::vector<User>& users) {
    std::vector<std::vector<float>> features(users.size(), std::vector<float>(3));

    for (size_t i = 0; i < users.size(); ++i) {
        features[i][0] = (users[i].gender == "Male") ? 1.0f : 0.0f;
        features[i][1] = static_cast<float>(users[i].age) / 100.0f;
        features[i][2] = static_cast<float>(users[i].interests.size()) / 10.0f;
    }

    return features;
}

//link prediction
float dot_product(const std::vector<float>& v1, const std::vector<float>& v2) {
    float result = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

float sigmoid(float x) {
    return 1 / (1 + std::exp(-x));
}

float predict_link(const std::vector<float>& embedding_i, const std::vector<float>& embedding_j) {
    return sigmoid(dot_product(embedding_i, embedding_j));
}

//loss calculation
float binary_cross_entropy(float predicted, float label) {
    return - (label * std::log(predicted) + (1 - label) * std::log(1 - predicted));
}


int main() {
    std::string filename = "dataset.csv";
    std::vector<User> users = readCSV(filename);
    if (users.empty()) {
        std::cerr << "No valid users found in the CSV file." << std::endl;
        return 1;
    }
    std::cout << "Loaded " << users.size() << " users from the dataset." << std::endl;
    createAdjacencyList(users);
    Graph graph(users.size());
    for (size_t i = 0; i < users.size(); ++i) {
        for (int connection : users[i].connections) {
            graph.add_edge(i, connection);
        }
    }
    std::cout << "Created graph with " << graph.num_nodes() << " nodes." << std::endl;
    
    // Create node features and normalize adjacency matrix
    std::vector<std::vector<float>> node_features = create_node_features(users);
    std::vector<std::vector<float>> normalized_adj = graph.normalize_adjacency();
    
    // Define the GCN dimensions
    int input_dim = node_features[0].size();
    int hidden_dim = 16;
    int output_dim = 2;
    GCN gcn(input_dim, hidden_dim, output_dim);
    
    // Forward pass to get embeddings
    std::vector<std::vector<float>> embeddings = gcn.forward(node_features, normalized_adj);
    
    // Link prediction between users
    std::cout << "\nLink Predictions (for the first 5 user pairs):" << std::endl;
    std::cout << "User ID 1 | User ID 2 | Link Probability" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    
    for (int i = 0; i < 5; ++i) {
        for (int j = i + 1; j < 5; ++j) {
            // Predict link probability using dot product and sigmoid
            float link_prob = predict_link(embeddings[i], embeddings[j]);
            std::cout << std::setw(7) << users[i].id << "   | "
                      << std::setw(7) << users[j].id << "   | "
                      << std::setw(10) << std::fixed << std::setprecision(4) << link_prob << std::endl;
        }
    }
    
    // Example of backpropagation using true labels for a few link predictions
    std::vector<float> true_labels = {1, 0, 1, 0, 1};  // Example labels: 1 if there's a link, 0 otherwise
    float learning_rate = 0.01;
    
    for (size_t i = 0; i < true_labels.size(); ++i) {
        int user_i = i;
        int user_j = i + 1;  // For simplicity, comparing consecutive users
        float predicted_prob = predict_link(embeddings[user_i], embeddings[user_j]);
        
        // Calculate binary cross-entropy loss
        float loss = binary_cross_entropy(predicted_prob, true_labels[i]);
        
        // Output loss for debugging purposes
        std::cout << "Loss for link between User " << users[user_i].id << " and User " << users[user_j].id
                  << ": " << loss << std::endl;
        gcn.backward({true_labels[i]}, learning_rate);
    }
    return 0;
}