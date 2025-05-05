/*
Name: Spandan P. Marathe
Class: BE-IV
Roll No.: 41447
Batch: R4
*/

#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>
#include <climits>
#include <algorithm>

using namespace std;

// Sequential minimum finder
void sequential_minimum(const vector<int>& array) {
    int min_val = INT_MAX;
    
    auto start = chrono::high_resolution_clock::now();
    for(int i = 0; i < array.size(); i++) {
        min_val = min(min_val, array[i]);
    }
    auto end = chrono::high_resolution_clock::now();
    
    cout << "Sequential Minimum value: " << min_val << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;
}

// Parallel minimum finder
void minimum(const vector<int>& array) {
    int min_val = INT_MAX;
    
    auto start = chrono::high_resolution_clock::now();
    // Fixed: using index-based loop instead of iterator
    #pragma omp parallel for reduction(min:min_val)
    for(int i = 0; i < array.size(); i++) {
        min_val = min(min_val, array[i]);
    }
    auto end = chrono::high_resolution_clock::now();
    
    cout << "Parallel Minimum value: " << min_val << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;
}

// Sequential maximum finder
void sequential_maximum(const vector<int>& array) {
    int max_val = INT_MIN;
    
    auto start = chrono::high_resolution_clock::now();
    for(int i = 0; i < array.size(); i++) {
        max_val = max(max_val, array[i]);
    }
    auto end = chrono::high_resolution_clock::now();
    
    cout << "Sequential Maximum value: " << max_val << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;
}

// Parallel maximum finder
void maximum(const vector<int>& array) {
    int max_val = INT_MIN;
    
    auto start = chrono::high_resolution_clock::now();
    // Fixed: using index-based loop instead of iterator
    #pragma omp parallel for reduction(max:max_val)
    for(int i = 0; i < array.size(); i++) {
        max_val = max(max_val, array[i]);
    }
    auto end = chrono::high_resolution_clock::now();
    
    cout << "Parallel Maximum value: " << max_val << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;
}

// Sequential sum calculator
void sequential_sum(const vector<int>& array) {
    int total = 0;
    
    auto start = chrono::high_resolution_clock::now();
    for(int i = 0; i < array.size(); i++) {
        total += array[i];
    }
    auto end = chrono::high_resolution_clock::now();
    
    cout << "Sequential Sum: " << total << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;
}

// Parallel sum calculator
void sum(const vector<int>& array) {
    int total = 0;
    
    auto start = chrono::high_resolution_clock::now();
    // Fixed: using index-based loop instead of iterator
    #pragma omp parallel for reduction(+:total)
    for(int i = 0; i < array.size(); i++) {
        total += array[i];
    }
    auto end = chrono::high_resolution_clock::now();
    
    cout << "Parallel Sum: " << total << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;
}

// Sequential average calculator
void sequential_average(const vector<int>& array) {
    int total = 0;
    
    auto start = chrono::high_resolution_clock::now();
    for(int i = 0; i < array.size(); i++) {
        total += array[i];
    }
    double avg = (double)total / array.size();
    auto end = chrono::high_resolution_clock::now();
    
    cout << "Sequential Average: " << avg << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;
}

// Parallel average calculator
void average(const vector<int>& array) {
    int total = 0;
    
    auto start = chrono::high_resolution_clock::now();
    // Fixed: using index-based loop instead of iterator
    #pragma omp parallel for reduction(+:total)
    for(int i = 0; i < array.size(); i++) {
        total += array[i];
    }
    double avg = (double)total / array.size();
    auto end = chrono::high_resolution_clock::now();
    
    cout << "Parallel Average: " << avg << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;
}

int main() {
    // Initialize vector with random values
    const int SIZE = 100000;
    vector<int> data(SIZE);
    
    // Seed the random number generator
    srand(time(nullptr));
    
    // Fill the vector with random numbers
    for(int i = 0; i < SIZE; i++) {
        data[i] = rand() % 1000;  // Random values between 0 and 999
    }
    
    cout << "\n===== Vector Operations Benchmark =====\n" << endl;
    
    // Test minimum functions
    cout << "----- Minimum Value -----" << endl;
    sequential_minimum(data);
    minimum(data);
    cout << endl;
    
    // Test maximum functions
    cout << "----- Maximum Value -----" << endl;
    sequential_maximum(data);
    maximum(data);
    cout << endl;
    
    // Test sum functions
    cout << "----- Sum -----" << endl;
    sequential_sum(data);
    sum(data);
    cout << endl;
    
    // Test average functions
    cout << "----- Average -----" << endl;
    sequential_average(data);
    average(data);
    cout << endl;
    
    return 0;
}