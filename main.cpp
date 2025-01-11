#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

using namespace std;
using namespace chrono;

// Function for block matrix multiplication
void blockMatrixMultiply(const vector<vector<double>>& A, 
                         const vector<vector<double>>& B, 
                         vector<vector<double>>& C, int blockSize) {
    int n = A.size();

    for (int i = 0; i < n; i += blockSize) {
        for (int j = 0; j < n; j += blockSize) {
            for (int k = 0; k < n; k += blockSize) {
                for (int ii = i; ii < min(i + blockSize, n); ++ii) {
                    for (int jj = j; jj < min(j + blockSize, n); ++jj) {
                        double temp = 0.0;
                        for (int kk = k; kk < min(k + blockSize, n); ++kk) {
                            temp += A[ii][kk] * B[kk][jj];
                        }
                        C[ii][jj] += temp;
                    }
                }
            }
        }
    }

}

// Function for standard matrix multiplication
void regularMatrixMultiply(const vector<vector<double>>& A, 
                           const vector<vector<double>>& B, 
                           vector<vector<double>>& C) {
    int n = A.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Function to generate a random matrix
vector<vector<double>> generateRandomMatrix(int size) {
    vector<vector<double>> matrix(size, vector<double>(size));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

// Function to measure execution time
template <typename Func, typename... Args>
double measureExecutionTime(Func func, Args&&... args) {
    auto start = high_resolution_clock::now();
    func(forward<Args>(args)...);
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

// Main function
int main() {
    vector<int> matrixSizes = {10, 100, 1000, 1500}; 
    int blockSize = 64;                               

    for (int size : matrixSizes) {
        cout << "\nTesting for matrix size: " << size << "x" << size << endl;

        try {
            auto A = generateRandomMatrix(size);
            auto B = generateRandomMatrix(size);
            auto C_block = vector<vector<double>>(size, vector<double>(size, 0.0));
            auto C_regular = vector<vector<double>>(size, vector<double>(size, 0.0));

            double timeBlock = measureExecutionTime(blockMatrixMultiply, A, B, C_block, blockSize);

            double timeRegular = measureExecutionTime(regularMatrixMultiply, A, B, C_regular);

            cout << fixed << setprecision(6);
            cout << "Block matrix multiplication time: " << timeBlock << " seconds" << endl;
            cout << "Regular matrix multiplication time: " << timeRegular << " seconds" << endl;
            cout << "Speed difference: " << timeRegular - timeBlock << " seconds" << endl;

        } catch (const bad_alloc&) {
            cout << "Matrix size " << size << "x" << size << " is too large to handle with available memory." << endl;
        }
    }

    return 0;
}
