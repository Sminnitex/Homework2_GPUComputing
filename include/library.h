#ifndef LIBRARY_H
#define LIBRARY_H

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "./helper_cuda.h"

#define dtype float

dtype randomf(){
    return (dtype) rand() / (dtype) RAND_MAX;   
}

void matrixInitialize(int rows, int cols, dtype **matrix, dtype **transpose, dtype **transposeBlock) {
    checkCudaErrors(cudaHostAlloc(matrix, sizeof(dtype) * rows * cols, cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(transpose, sizeof(dtype) * rows * cols, cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(transposeBlock, sizeof(dtype) * rows * cols, cudaHostAllocDefault));
}


void matrixDestroyer(dtype *matrix, dtype *transpose, dtype *transposeBlock){
    checkCudaErrors(cudaFreeHost(matrix));
    checkCudaErrors(cudaFreeHost(transpose));
    checkCudaErrors(cudaFreeHost(transposeBlock));
}

void printMatrix(dtype *matrix, int rows, int cols, char ST[56]){
    int i, j;  \
      printf("%s:\n", ( ST ));  \
      for (i=0; i< ( rows ); i++) {  \
        printf("\t");  \
        for (j=0; j< ( cols ); j++)  \
          printf("%6.3f ", matrix[i*( cols ) + j]);  \
        printf("\n");  \
      }  \
      printf("\n\n");  \
}

__device__ int findsMin(int num1, int num2){
    return (num1 > num2) ? num2 : num1;
}

__global__ void dummyKernel() {}

__global__ void transposeMatrix (dtype *matrix, dtype *transpose, int rows, int cols){
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y; 

    extern __shared__ dtype sharedData[];

    for (int i = 0; i < rows * cols; i ++) {
      sharedData[(threadIdx.y + i) * blockDim.x + threadIdx.x] = matrix[(y + i) * cols + x];
    }   

    __syncthreads();

    for (int i = 0; i < rows * cols; i ++) {
      transpose[x * rows + (y + i)] = sharedData[(threadIdx.y + i) * blockDim.x + threadIdx.x];
    }   
}

__global__ void transposeMyBlockMatrix (dtype *matrix, dtype *transpose, int rows, int cols, int block){
    const uint global_x = blockIdx.x * block * blockDim.x + threadIdx.x;
    const uint global_y = blockIdx.y * block * blockDim.y + threadIdx.y;

    const uint shared_x = threadIdx.x;
    const uint shared_y = threadIdx.y;
    extern __shared__ dtype tile[];

    if (global_x < cols && global_y < rows) {
        tile[shared_y * block + shared_x] = matrix[global_y * cols + global_x];
    }

    __syncthreads();

    const uint transposed_x = blockIdx.y * block * blockDim.y + threadIdx.x;
    const uint transposed_y = blockIdx.x * block * blockDim.x + threadIdx.y;

    if (transposed_x < rows && transposed_y < cols) {
        transpose[transposed_y * rows + transposed_x] = tile[shared_x * block + shared_y];
    }
}

#endif  
