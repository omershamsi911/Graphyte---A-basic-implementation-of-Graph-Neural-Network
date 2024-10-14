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

    // Get the current year
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

    // Remove the outer quotes if they exist
    std::string trimmedInterests = trim(interestsStr);
    if (trimmedInterests.front() == '"' && trimmedInterests.back() == '"') {
        trimmedInterests = trimmedInterests.substr(1, trimmedInterests.size() - 2);
    }

    std::istringstream iss(trimmedInterests);
    std::string interest;
    while (std::getline(iss, interest, ',')) {
        interest = trim(interest);
        if (!interest.empty()) {
            // Remove surrounding single quotes if present
            if (interest.front() == '\'' && interest.back() == '\'') {
                interest = interest.substr(1, interest.length() - 2);
            }
            interests.push_back(interest);
        }
    }
    return interests;
}

// Helper function to handle fields potentially containing commas and quotes
std::string getField(std::istringstream& iss) {
    std::string field;
    char ch;
    bool inQuotes = false;
    std::ostringstream oss;

    while (iss.get(ch)) {
        if (ch == '"' && !inQuotes) {
            inQuotes = true; // Field starts with a quote
        } else if (ch == '"' && inQuotes) {
            if (iss.peek() == ',') {
                iss.get(); // Skip the comma after the closing quote
                break;
            } else {
                inQuotes = false; // End of quoted section
            }
        } else if (ch == ',' && !inQuotes) {
            break; // End of field when not in quotes
        } else {
            oss << ch; // Add character to the field
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

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line) && users.size() < MAX_USERS) {
        std::istringstream iss(line);
        User user;

        // Read UserID
        user.id = std::stoi(getField(iss));

        // Read Name
        user.name = getField(iss);

        // Read Gender
        user.gender = getField(iss);

        // Read DOB and calculate age
        std::string dobStr = getField(iss);
        user.age = calculateAge(dobStr);

        // Read Interests
        std::string interestsStr = getField(iss);
        std::vector<std::string> interestsList = splitInterests(interestsStr);
        user.interests.insert(interestsList.begin(), interestsList.end());

        // Read City
        user.city = getField(iss);

        // Read Country
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

void printUserInfo(const std::vector<User>& users) {
    for (const auto& user : users) {
        std::cout << "User ID: " << user.id << std::endl;
        std::cout << "Name: " << user.name << std::endl;
        std::cout << "Gender: " << user.gender << std::endl;
        std::cout << "Age: " << user.age << std::endl;
        std::cout << "Interests: ";
        for (const auto& interest : user.interests) {
            std::cout << interest << ", ";
        }
        std::cout << std::endl;
        std::cout << "City: " << user.city << std::endl;
        std::cout << "Country: " << user.country << std::endl;
        std::cout << "Connections: ";
        for (int connection : user.connections) {
            std::cout << connection << " ";
        }
        std::cout << std::endl << std::endl;
    }
}

int main() {
    std::string filename = "dataset.csv";
    std::vector<User> users = readCSV(filename);
    
    if (users.empty()) {
        std::cerr << "No valid users found in the CSV file." << std::endl;
        return 1;
    }

    createAdjacencyList(users);

    std::cout << "User Information and Adjacency List for " << users.size() << " users:" << std::endl;
    printUserInfo(users);

    return 0;
}
