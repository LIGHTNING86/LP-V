/*
Name: Spandan P. Marathe
Class: BE-IV
Roll No.: 41447
Batch: R4
*/

#include<iostream>
#include<bits/stdc++.h>
#include<cuda.h>
#define BLOCK_SIZE 16

using namespace std;

// Fill array with random values
void fill_array(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100;
    }
}

// Print matrix (1D representation for simplicity)
void print_matrix(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

// Sequential addition of two arrays
void add_cpu(int *arr1, int *arr2, int *result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = arr1[i] + arr2[i];
    }
}

// CUDA kernel for adding two arrays
__global__ void add(int *arr1, int *arr2, int *arr3, int size) {
    int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_id < size) {
        arr3[block_id] = arr1[block_id] + arr2[block_id];
    }
}

// Initialize matrix with random values
void initialize_matrix(int *array, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            array[i * cols + j] = rand() % 10;
        }
    }
}

// Print matrix in a 2D format
void print_matrix(int *array, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << array[i * cols + j] << " ";
        }
        cout << endl;
    }
}

// CPU matrix multiplication
void matrix_multiplication_cpu(int *a, int *b, int *c, int common, int c_rows, int c_cols) {
    for (int i = 0; i < c_rows; i++) {
        for (int j = 0; j < c_cols; j++) {
            int sum = 0;
            for (int k = 0; k < common; k++) {
                sum += a[i * common + k] * b[k * c_cols + j];
            }
            c[i * c_cols + j] = sum;
        }
    }
}

// CUDA kernel for matrix multiplication
__global__ void matrix_multiply(int *a, int *b, int *c, int c_rows, int common, int c_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if (col < c_cols && row < c_rows) {
        for (int j = 0; j < common; j++) {
            sum += a[row * common + j] * b[j * c_cols + col];
        }
        c[c_cols * row + col] = sum;
    }
}

int main() {
    // Task 1: Vector Addition
    int *arr1_cpu, *arr2_cpu, *result_cpu;
    int size;

    cout << "Enter size of vector: ";
    cin >> size;

    arr1_cpu = new int[size];
    arr2_cpu = new int[size];
    result_cpu = new int[size];

    fill_array(arr1_cpu, size);
    cout << "Array 1: ";
    print_matrix(arr1_cpu, size);
    fill_array(arr2_cpu, size);
    cout << "Array 2: ";
    print_matrix(arr2_cpu, size);

    int *arr1_gpu, *arr2_gpu, *result_gpu;

    cudaMallocManaged(&arr1_gpu, size * sizeof(int));
    cudaMallocManaged(&arr2_gpu, size * sizeof(int));
    cudaMallocManaged(&result_gpu, size * sizeof(int));

    cudaMemcpy(arr1_gpu, arr1_cpu, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(arr2_gpu, arr2_cpu, size * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float elapsedTime;

    dim3 dimGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    add<<<dimGrid, dimBlock>>>(arr1_gpu, arr2_gpu, result_gpu, size);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(result_cpu, result_gpu, size * sizeof(int), cudaMemcpyDeviceToHost);
    cout << "GPU result (Vector Addition):\n";
    print_matrix(result_cpu, size);
    cout << "Elapsed Time (GPU): " << elapsedTime << " milliseconds" << endl;

    cudaFree(arr1_gpu);
    cudaFree(arr2_gpu);
    cudaFree(result_gpu);

    add_cpu(arr1_cpu, arr2_cpu, result_cpu, size);
    cout << "CPU result (Vector Addition):\n";
    print_matrix(result_cpu, size);

    // Task 2: Matrix Multiplication
    int A_rows, A_cols, B_rows, B_cols, C_rows, C_cols;
    cout << "Dimensions of matrix 1:\n";
    cout << "Rows: ";
    cin >> A_rows;
    cout << "Columns: ";
    cin >> A_cols;
    cout << "Dimensions of matrix 2:\n";
    cout << "Rows: " << A_cols << endl << "Columns: ";
    cin >> B_cols;
    B_rows = A_cols;
    C_rows = A_rows;
    C_cols = B_cols;

    int A_size = A_rows * A_cols;
    int B_size = B_rows * B_cols;
    int C_size = C_rows * C_cols;

    int *A, *B, *C;
    int *m1, *m2, *result;

    A = new int[A_size];
    B = new int[B_size];
    C = new int[C_size];

    initialize_matrix(A, A_rows, A_cols);
    cout << "Matrix 1 (A):\n";
    print_matrix(A, A_rows, A_cols);

    initialize_matrix(B, B_rows, B_cols);
    cout << "Matrix 2 (B):\n";
    print_matrix(B, B_rows, B_cols);

    cudaMallocManaged(&m1, A_size * sizeof(int));
    cudaMallocManaged(&m2, B_size * sizeof(int));
    cudaMallocManaged(&result, C_size * sizeof(int));

    cudaMemcpy(m1, A, A_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(m2, B, B_size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGridMat((C_cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (C_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlockMat(BLOCK_SIZE, BLOCK_SIZE);

    float gpu_elapsed_time;

    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start);

    matrix_multiply<<<dimGridMat, dimBlockMat>>>(m1, m2, result, C_rows, A_cols, C_cols);

    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);

    cudaMemcpy(C, result, C_size * sizeof(int), cudaMemcpyDeviceToHost);
    cout << "GPU result (Matrix Multiplication):\n";
    print_matrix(C, C_rows, C_cols);
    cout << "GPU Elapsed time (Matrix Multiplication): " << gpu_elapsed_time << " milliseconds" << endl;

    matrix_multiplication_cpu(A, B, C, A_cols, C_rows, C_cols);
    cout << "CPU result (Matrix Multiplication):\n";
    print_matrix(C, C_rows, C_cols);

    cudaFree(m1);
    cudaFree(m2);
    cudaFree(result);

    delete[] arr1_cpu;
    delete[] arr2_cpu;
    delete[] result_cpu;
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
