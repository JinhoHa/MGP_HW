#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
  // Read arguments
  int R = 1024;
  int S = 1024;
  string inputPath = "data/input.txt";
  string outputPath = "data/output.txt";
  if (argc == 1) {
    // nothing
  } else if (argc == 5) {
    R = atoi(argv[1]);
    S = atoi(argv[2]);
    inputPath = argv[3];
    outputPath = argv[4];
  } else {
    std::cout << "./generate R S inputPath outputPath" << endl;
    exit(-1);
  }
  // Print arguments
  std::cout << endl << "============= Join Tables =============" << endl;
  std::cout << "R : " << R << endl;
  std::cout << "S : " << S << endl;
  std::cout << "inputPath : " << inputPath << endl;
  std::cout << "outputPath : " << outputPath << endl;
  std::cout << "=======================================" << endl;
  // Initilize variables
  int MAX_ID = (R + S) * 5;
  bool* flagA = new bool[MAX_ID];
  bool* flagB = new bool[MAX_ID];
  int* tableA = new int[R];
  int* tableB = new int[S];
  vector<int> solution;
  // Generate id table
  {
    // random generator
    random_device device;
    mt19937 generator(device());
    uniform_int_distribution<int> distribution(0, MAX_ID - 1);
    {
      std::cout << "Generate flags for tableA..." << endl;
      int cnt = R;
      while (cnt > 0) {
        int index = distribution(generator);
        if (flagA[index] == false) {
          flagA[index] = true;
          cnt--;
        }
      }
    }
    {
      std::cout << "Generate flags for tableB..." << endl;
      int cnt = S;
      int index = 0;
      while (cnt > 0) {
        int index = distribution(generator);
        if (flagB[index] == false) {
          flagB[index] = true;
          cnt--;
        }
      }
    }
  }
  // Generate tableA and tableB
  {
    std::cout << "Fill tableA..." << endl;
    int index = 0;
    for (int i = 0; i < MAX_ID; i++) {
      if (flagA[i]) {
        tableA[index] = i;
        index++;
      }
    }
    random_shuffle(tableA, tableA + R);
  }
  {
    std::cout << "Fill tableB..." << endl;
    int index = 0;
    for (int i = 0; i < MAX_ID; i++) {
      if (flagB[i]) {
        tableB[index] = i;
        index++;
      }
    }
    random_shuffle(tableB, tableB + S);
  }
  // Generate solution
  {
    std::cout << "Fill tableC(intersection of tableA and tableB)" << endl;
    for (int i = 0; i < MAX_ID; i++) {
      if (flagA[i] && flagB[i]) {
        solution.push_back(i);
      }
    }
    random_shuffle(solution.begin(), solution.end());
  }
  // Write input file
  std::cout << "Write input file..." << endl;
  ofstream inputFile(inputPath.data());
  if (inputFile.is_open()) {
    inputFile << R << " " << S << "\n";
    for (int i = 0; i < R; i++) inputFile << tableA[i] << " ";
    inputFile << "\n";
    for (int i = 0; i < S; i++) inputFile << tableB[i] << " ";
    inputFile << "\n";
    inputFile.close();
  }
  // Write output file
  std::cout << "Write output file..." << endl;
  ofstream outputFile(outputPath.data());
  if (outputFile.is_open()) {
    outputFile << solution.size() << endl;
    for (auto const& value : solution) outputFile << value << " ";
    outputFile << "\n";
    outputFile.close();
  }
  // Free memory
  std::cout << "Free memories..." << endl;
  delete[] flagA;
  delete[] flagB;
  delete[] tableA;
  delete[] tableB;
  return 0;
}