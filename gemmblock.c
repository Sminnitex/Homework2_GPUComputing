#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <library.h>

int main(int argc, char *argv[]){

    //Initialize all the stuff we need
    clock_t begin = clock();

    srand(time(NULL));
    int power = strtol(argv[1], NULL, 10);
    long number = pow(2, power);
    int block = number / 32;
    printf("%d \n", block);

    float **matrix = matrixInitialize(number, number);
    float **matrix2 = matrixInitialize(number, number);

    //Assign random values to matrices
    for (int i = 0; i < number; i++){
        for (int j = 0; j < number; j++){
            matrix[i][j] = randomf();
            matrix2[i][j] = randomf();
        }
    }

    //Matrix multiplication
    float **matrix3 = matrixInitialize(number, number);
    float temp;
    
    for (int jj = 0; jj < number; jj += block) {
        for (int kk = 0; kk < number; kk += block) {
            for (int i = 0; i < number; i++) {
                for (int j = jj; j < findMin(jj + block, number); j++) {
                    temp = 0;
                    for (int k = kk; k < findMin(kk + block, number); k++) {
                        temp += matrix[i][k] * matrix2[k][j];
                    }
                    matrix3[i][j] += temp;
                }
            }
        }
    }  

    //Lines for debug purposes
    //printMatrix(matrix, number, number);
    //printMatrix(matrix2, number, number);
    //printMatrix(matrix3, number, number);

    //Let's close everything
    clock_t end = clock();
    double time = (double) (end-begin) / CLOCKS_PER_SEC;
    printf("Time spent = %f seconds\n", time);

    int memory_usage = 3 * number * number * sizeof(float);
    printf("Memory usage = %d bytes\n", memory_usage);

    matrixDestroyer(matrix, number);
    matrixDestroyer(matrix2, number);
    matrixDestroyer(matrix3, number);

    return 0;
}