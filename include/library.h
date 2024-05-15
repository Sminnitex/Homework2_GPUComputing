#ifndef LIBRARY_H
#define LIBRARY_H

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define dtype float

dtype randomf(){
    return (dtype) rand() / (dtype) RAND_MAX;   
}

dtype **matrixInitialize(int rows, int cols){
    dtype **matrix = (dtype**)malloc(rows * sizeof(dtype*));
    for (int i = 0; i < rows; i++){
        matrix[i] = (dtype*)malloc(cols * sizeof(dtype));
    }

    if (matrix == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    return matrix;
}

void matrixDestroyer(dtype **matrix, int rows){
    for (int i = 0; i < rows; i++){
        free(matrix[i]);
    }
    free(matrix);
}

void printMatrix(dtype **matrix, int rows, int cols){
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

__device__ int findsMin(int num1, int num2){
    return (num1 > num2) ? num2 : num1;
}

__global__ void transposeMatrix (dtype **matrix, dtype **transpose, int rows, int cols){
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++){
            transpose[i][j] = matrix[j][i];
         }
    }
}

__global__ void transposeMyBlockMatrix (dtype **matrix, dtype **transpose, int rows, int cols, int block){
    for (int jj = 0; jj < cols; jj += block) {
            for (int ii = 0; ii < rows; ii+=block) {
                for (int j = jj; j < findsMin(jj + block, cols); j++) {
                    for (int i = ii; i < findsMin(ii + block, rows); i++){
                        transpose[i][j] = matrix[j][i];
                    }
                }
            }
    }
}


#endif  
