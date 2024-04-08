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
    //printf("block size = %d \n", block);

    FILE *csvTime = fopen("output/time.csv", "w");
    if (csvTime == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    fprintf(csvTime, "Time,Dimensions\n");

    long long tries = 1 << 9;
    double time, time2;

    //tries loop
    for (int count = 0; count < tries; count++){
        float **matrix = matrixInitialize(number, number);

        //Assign random values to matrices
        for (int i = 0; i < number; i++){
            for (int j = 0; j < number; j++){
                matrix[i][j] = randomf();
            }
        }
        
        float **transposeBlock = matrixInitialize(number, number);
        float **transpose = matrixInitialize(number, number);

        //check validity
        if (matrix == NULL || transposeBlock == NULL || transpose == NULL){
            return 1; // ERROR: malloc did not work
        }
        //Matrix block transpose
        //clock_t begin = clock();
        //transposeBlockMatrix(matrix, transposeBlock, number, number, block);
        //clock_t end = clock();
        //time = (double) (end-begin) / CLOCKS_PER_SEC;
        //fprintf(csvTime, "%f,%ld\n", time, number); 

        //Matrix normal transpose
        clock_t begin2 = clock();
        transposeMatrix(matrix, transpose, number, number);
        clock_t end2 = clock();
        time2 = (double) (end2-begin2) / CLOCKS_PER_SEC;
        fprintf(csvTime, "%f,%ld\n", time2, number); 

        //Lines for debug purposes
        //printMatrix(matrix, number, number);
        //printMatrix(transposeBlock, number, number);
        //printMatrix(transpose, number, number);

        matrixDestroyer(matrix, number);
        matrixDestroyer(transposeBlock, number);
        matrixDestroyer(transpose, number);
        number = number - 1;
    }      

    fclose(csvTime);

    return 0;
}