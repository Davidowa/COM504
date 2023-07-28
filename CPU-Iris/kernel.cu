
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cstdlib>
#include <ctime>

using namespace std;

struct Iris {
    int id;
    float sepalLength;
    float sepalWidth;
    float petalLength;
    float petalWidth;
    int species;
};

void loadDataset(vector<Iris>& d) {
    ifstream file("Iris.csv");
    string line;
    map<string, int> speciesMap;
    int speciesCounter = 0;

    // Skip header line
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;

        Iris iris;

        getline(ss, cell, ',');
        iris.id = stoi(cell);

        getline(ss, cell, ',');
        iris.sepalLength = stof(cell);

        getline(ss, cell, ',');
        iris.sepalWidth = stof(cell);

        getline(ss, cell, ',');
        iris.petalLength = stof(cell);

        getline(ss, cell, ',');
        iris.petalWidth = stof(cell);

        getline(ss, cell, ',');
        if (speciesMap.find(cell) == speciesMap.end()) {
            speciesMap[cell] = speciesCounter++;
        }
        iris.species = speciesMap[cell];

        d.push_back(iris);
    }
}

void printPreprocessedDataset(vector<Iris>& d) {
    for (const Iris& iris : d) {
        if (iris.id < 10) {
            cout << "ID: 0" << iris.id;
        }
        else {
            cout << "ID: " << iris.id;
        }

        cout << ", \tSepal Length: " << iris.sepalLength
            << ", \tSepal Width: " << iris.sepalWidth
            << ", \tPetal Length: " << iris.petalLength
            << ", \tPetal Width: " << iris.petalWidth
            << ", \tSpecies: " << iris.species << endl;
    }
}

int main() {

    vector<Iris> dataset;

    // Load the dataset from the Iris.csv
    loadDataset(dataset);

    //  Size of dataset
    int N = dataset.size(); // 150

    // Print the preprocessed dataset
    //printPreprocessedDataset(dataset);


    // Set the train and test dataset 
    vector<Iris> trainSet, testSet;

    // Initialize random seed
    int random_state = 101;
    srand(random_state);
    float test_size = 0.33;

    for (const Iris& iris : dataset) {
        if (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < test_size) {
            testSet.push_back(iris);
        }
        else {
            trainSet.push_back(iris);
        }
    }

    // Print the sizes of the data, training, and testing sets
    cout << "Iris dataset size: " << N << endl;
    cout << "Training set size: " << trainSet.size() << endl;
    cout << "Testing set size: " << testSet.size() << endl;



    return 0;
}