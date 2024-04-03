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
    int block = number / 64;
    printf("block size = %d \n", block);

    float **matrix = matrixInitialize(number, number);

    //Assign random values to matrices
    for (int i = 0; i < number; i++){
        for (int j = 0; j < number; j++){
            matrix[i][j] = randomf();
        }
    }

    //Matrix multiplication
    float **transpose = matrixInitialize(number, number);
    
    for (int jj = 0; jj < number; jj += block) {
            for (int ii = 0; ii < number; ii+=block) {
                for (int j = jj; j < findMin(jj + block, number); j++) {
                    for (int i = ii; i < findMin(ii + block, number); i++){
                        transpose[i][j] = matrix[j][i];
                    }
                }
            }
    }  

    //Lines for debug purposes
    //printMatrix(matrix, number, number);
    //printMatrix(transpose, number, number);

    //Let's close everything
    clock_t end = clock();
    double time = (double) (end-begin) / CLOCKS_PER_SEC;
    printf("Time spent = %f seconds\n", time);

    int memory_usage = 2 * number * number * sizeof(float);
    printf("Memory usage = %d bytes\n", memory_usage);

    matrixDestroyer(matrix, number);
    matrixDestroyer(transpose, number);

    return 0;
}