#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include "./include/library.h"
#include "./include/helper_cuda.h"
#include <cuda_runtime.h>

#define NDEVICE 3
#define TIMER_DEF     struct timeval temp_1, temp_2
#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)
#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)
#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0)

int main(int argc, char *argv[]){

    //Initialize all the stuff we need
    srand(time(NULL));
    int power = strtol(argv[1], NULL, 10);
    long number = pow(2, power);
    int block = number / 64;
    int gridsize = 1;
    int blocksize = 1;
    //printf("block size = %d \n", block);

    FILE *file = fopen("deviceProperties.txt", "r");
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
  
    char ch;
    while ((ch = fgetc(file)) != EOF) {
        printf("%c", ch);
    }

    FILE *csvTime = fopen("output/timeB0.csv", "w");
    if (csvTime == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    fprintf(csvTime, "Time,Dimensions\n");

    long long tries = 1 << 9;
    TIMER_DEF;
    float times[NDEVICE];

    //tries loop
    for (int count = 0; count < tries; count++){
        dtype **matrix = matrixInitialize(number, number);

        //Assign random values to matrices
        for (int i = 0; i < number; i++){
            for (int j = 0; j < number; j++){
                matrix[i][j] = randomf();
            }
        }
        
        dtype **transposeBlock = matrixInitialize(number, number);
        dtype **transpose = matrixInitialize(number, number);

        //check validity
        if (matrix == NULL || transposeBlock == NULL || transpose == NULL){
            return 1; // ERROR: malloc did not work
        }
        //Matrix block transpose
        //TIMER_START;
        //transposeMyBlockMatrix<<<gridsize, blocksize>>>(matrix, transposeBlock, number, number, block);
        //TIMER_STOP;
        //times[1] = TIMER_ELAPSED;
        //fprintf(csvTime, "%f,%ld\n", times[1], number); 

        //Matrix normal transpose
        TIMER_START;
        transposeMatrix<<<gridsize, blocksize>>>(matrix, transpose, number, number);
        TIMER_STOP;
        times[2] = TIMER_ELAPSED;
        fprintf(csvTime, "%f,%ld\n", times[2], number); 

        //Lines for debug purposes
        //printMatrix(matrix, number, number);
        //printMatrix(transposeBlock, number, number);
        //printMatrix(transpose, number, number);

        matrixDestroyer(matrix, number);
        matrixDestroyer(transposeBlock, number);
        matrixDestroyer(transpose, number);
        number = number + 1;
    }      

    fclose(csvTime);

    return 0;
}