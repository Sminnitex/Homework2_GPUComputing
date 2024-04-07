#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <library.h>

int main(int argc, char *argv[]){

    //Initialize all the stuff we need
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

    //Matrix block transpose
    float **transposeBlock = matrixInitialize(number, number);
    
    clock_t begin = clock();
    for (int jj = 0; jj < number; jj += block) {
            for (int ii = 0; ii < number; ii+=block) {
                for (int j = jj; j < findMin(jj + block, number); j++) {
                    for (int i = ii; i < findMin(ii + block, number); i++){
                        transposeBlock[i][j] = matrix[j][i];
                    }
                }
            }
    }
    clock_t end = clock();
    double time = (double) (end-begin) / CLOCKS_PER_SEC;
    printf("Block matrix transpose = %f seconds\n", time); 

        //Matrix normal transpose
    float **transpose = matrixInitialize(number, number);

    clock_t begin2 = clock();
    for (int j = 0; j < number; j++) {
        for (int i = 0; i < number; i++){
            transpose[i][j] = matrix[j][i];
         }
    }
                  

    //Lines for debug purposes
    //printMatrix(matrix, number, number);
    //printMatrix(transposeBlock, number, number);
    //printMatrix(transpose, number, number);

    //Let's close everything
    clock_t end2 = clock();
    double time2 = (double) (end2-begin2) / CLOCKS_PER_SEC;
    printf("Normal matrix transpose = %f seconds\n", time2);

    int memory_usage = 3 * number * number * sizeof(float);
    printf("Memory usage = %d bytes\n", memory_usage);

    matrixDestroyer(matrix, number);
    matrixDestroyer(transposeBlock, number);
    matrixDestroyer(transpose, number);

    return 0;
}