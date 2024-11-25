# Graph Neural Network

---

This repository serves as a implementation of the GraphSAGE in C++. The current implementation remains simplistic and aims to rather illustrate the effectiveness of the approach and it's usage in link prediction, practically applied in recommendation systems often seen in social media platforms

## Usage

The repository is configured to "work out off the box" but does require a C++ compiler such as gcc. If gcc has not been installed on your system first install it from the [winlibs](https://winlibs.com/) website.

To use the repository, simply clone it using the git clone command:

```
git clone https://github.com/omershamsi911/Graphyte---A-basic-implementation-of-Graph-Neural-Network
```

After cloning, open the folder into Visual Studio Code or a text editor of your choice. For Visual Studio Code, a build task has already been provided which can simply be run by using:

```
Ctrl + Shift + B
```

Similar build tasks can be configured for other editors. After compilation, run the executable file.

## Graphical User Interface:

To illustrate the effectiveness and to demonstrate visually the model, a graphical implement has been provided which showcases the recommended nodes for test nodes as shown below

(Insert image of output here)

## References:

This implementation was made possible by the following article:

- [Friend Recommendation using GraphSAGE](https://medium.com/stanford-cs224w/friend-recommendation-using-graphsage-ffcda2aaf8d6)
- [OhMyGraphs: GraphSAGE and inductive representation learning](https://medium.com/analytics-vidhya/ohmygraphs-graphsage-and-inductive-representation-learning-ea26d2835331)
