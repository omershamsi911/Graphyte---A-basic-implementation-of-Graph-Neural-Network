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
    std::string dob;
    std::set<std::string> interests;
    std::string city;
    std::string country;
    std::vector<int> connections;
};

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r\f\v");
    size_t last = str.find_last_not_of(" \t\n\r\f\v");
    return (first == std::string::npos) ? "" : str.substr(first, (last - first + 1));
}

std::vector<std::string> splitInterests(const std::string& interestsStr) {
    std::vector<std::string> interests;
    std::istringstream iss(interestsStr);
    std::string interest;
    while (std::getline(iss, interest, ',')) {
        interest = trim(interest);
        if (!interest.empty()) {
            // Remove surrounding quotes if present
            if (interest.front() == '\'' && interest.back() == '\'') {
                interest = interest.substr(1, interest.length() - 2);
            }
            interests.push_back(interest);
        }
    }
    return interests;
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
        std::string field;
        User user;

        // Read UserID
        if (std::getline(iss, field, ',')) {
            user.id = std::stoi(field);
        }

        // Read Name
        std::getline(iss, user.name, ',');

        // Read Gender
        std::getline(iss, user.gender, ',');

        // Read DOB
        std::getline(iss, user.dob, ',');

        // Read Interests
        std::string interestsStr;
        if (std::getline(iss, interestsStr, ',')) {
            std::vector<std::string> interestsList = splitInterests(interestsStr);
            user.interests.insert(interestsList.begin(), interestsList.end());
        }

        // Read City
        std::getline(iss, user.city, ',');

        // Read Country
        std::getline(iss, user.country);

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
        std::cout << "DOB: " << user.dob << std::endl;
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