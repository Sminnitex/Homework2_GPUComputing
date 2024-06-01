#ifndef LIBRARY_H
#define LIBRARY_H

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "./helper_cuda.h"

#define dtype float
#define TILE_DIM 16
#define BLOCK_ROWS 8

dtype randomf(){
    return (dtype) rand() / (dtype) RAND_MAX;   
}

void matrixInitialize(int rows, int cols, dtype **matrix, dtype **transpose, dtype **transposeShared) {
    checkCudaErrors(cudaMallocManaged(matrix, sizeof(dtype) * rows * cols));
    checkCudaErrors(cudaMallocManaged(transpose, sizeof(dtype) * rows * cols));
    checkCudaErrors(cudaMallocManaged(transposeShared, sizeof(dtype) * rows * cols));
}


void matrixDestroyer(dtype *matrix, dtype *transpose, dtype *transposeShared){
    checkCudaErrors(cudaFree(matrix));
    checkCudaErrors(cudaFree(transpose));
    checkCudaErrors(cudaFree(transposeShared));
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

__global__ void transposeGlobalMatrix (dtype *matrix, dtype *transpose, int rows, int cols){
    const uint x = blockIdx.x * TILE_DIM + threadIdx.x;
    const uint y = blockIdx.y * TILE_DIM + threadIdx.y; 
    
     for (int i = 0; i < TILE_DIM; i+=BLOCK_ROWS){
        if ((x < cols) && (y + i < rows)) {
            transpose[(x * rows) + (y + i)] = matrix[(y + i) * cols + x];
        }
     }
}

__global__ void transposeSharedMatrix (dtype *matrix, dtype *transpose, int rows, int cols){
   __shared__ dtype tile[TILE_DIM][TILE_DIM + 1]; //Add padding to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = x + y * cols;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < cols && (y + i) < rows) {
            tile[threadIdx.y + i][threadIdx.x] = matrix[index_in + i * cols];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = x + y * rows;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < rows && (y + i) < cols) {
            transpose[index_out + i * rows] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

#endif  
