#ifndef LIBRARY_H
#define LIBRARY_H

float randomf();
float **matrixInitialize(int rows, int cols);
void matrixDestroyer(float **matrix, int rows);
void printMatrix(float **matrix, int rows, int cols);
int findMin(int num1, int num2);
void transposeMatrix (float **matrix, float **transpose, int rows, int cols);
void transposeBlockMatrix (float **matrix, float **transpose, int rows, int cols, int block);

#endif  
