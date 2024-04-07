#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <library.h>

float randomf(){
    return (float) rand() / (float) RAND_MAX;   
}

float **matrixInitialize(int rows, int cols){
    float **matrix = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++){
        matrix[i] = (float*)malloc(cols * sizeof(float));
    }

    if (matrix == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    return matrix;
}

void matrixDestroyer(float **matrix, int rows){
    for (int i = 0; i < rows; i++){
        free(matrix[i]);
    }
    free(matrix);
}

void printMatrix(float **matrix, int rows, int cols){
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int findMin(int num1, int num2){
    return (num1 > num2) ? num2 : num1;
}

void transposeMatrix (float **matrix, float **transpose, int rows, int cols){
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++){
            transpose[i][j] = matrix[j][i];
         }
    }
}

void transposeBlockMatrix (float **matrix, float **transpose, int rows, int cols, int block){
    for (int jj = 0; jj < cols; jj += block) {
            for (int ii = 0; ii < rows; ii+=block) {
                for (int j = jj; j < findMin(jj + block, cols); j++) {
                    for (int i = ii; i < findMin(ii + block, rows); i++){
                        transpose[i][j] = matrix[j][i];
                    }
                }
            }
    }
}
