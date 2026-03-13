#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <stdexcept>
#include <omp.h>

using namespace std;

typedef long long int_t;

void read_matrix(const string& filename, vector<vector<int_t>>& matrix, int n) {
    ifstream in(filename);
    if (!in.is_open()) {
        throw runtime_error("Could not open input file: " + filename);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!(in >> matrix[i][j])) {
                throw runtime_error("Error reading data at [" + to_string(i) + "][" + to_string(j) + "] in " + filename);
            }
        }
    }
    in.close();
}

void save_matrix(const string& filename, const vector<vector<int_t>>& matrix, int n) {
    ofstream out(filename);
    if (!out.is_open()) {
        throw runtime_error("Could not open output file for writing: " + filename);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out << matrix[i][j] << (j == n - 1 ? "" : " ");
        }
        out << "\n";
    }

    if (out.fail()) {
        throw runtime_error("Error occurred while writing to: " + filename);
    }
    out.close();
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " <size> <path_a> <path_b> <path_res>" << endl;
        return 1;
    }

    try {
        int n = stoi(argv[1]);
        string path_a = argv[2];
        string path_b = argv[3];
        string path_res = argv[4];
        omp_set_num_threads(stoi(argv[5]));

        if (n <= 0) throw invalid_argument("Matrix size must be positive.");
        if (argv[5] <= 0) throw invalid_argument("Number of threads must be positive.");

        vector<vector<int_t>> A(n, vector<int_t>(n));
        vector<vector<int_t>> B(n, vector<int_t>(n));
        vector<vector<int_t>> C(n, vector<int_t>(n, 0));

        read_matrix(path_a, A, n);
        read_matrix(path_b, B, n);

        auto start = chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < n; ++k) {
                int_t temp = A[i][k];
                for (int j = 0; j < n; ++j) {
                    C[i][j] += temp * B[k][j];
                }
            }
        }

        auto end = chrono::high_resolution_clock::now();

        save_matrix(path_res, C, n);

        cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;

    } catch (const exception& e) {
        cerr << "Something went horribly wrong: " << e.what() << endl;
        return 1;
    }

    return 0;
}