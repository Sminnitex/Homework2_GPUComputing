#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include "./include/library.h"
#include "./include/helper_cuda.h"
#include <cuda_runtime.h>

#define NDEVICE 2
#define NFILES 8
#define TIMER_DEF     struct timeval temp_1, temp_2
#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)
#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)
#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0)

int main(int argc, char *argv[]) {
    //Initialize all the stuff we need
    srand(time(NULL));
    int power = strtol(argv[1], NULL, 10);
    long number = pow(2, power);
    int block = number / 64;
    int gridsize = 28;
    int blocksize = 64;

    printf("==============================================================\n");
    printf("STATS OF MY PROBLEM\n");
    printf("block for block transpose = %d \n", block);
    printf("block size = %d \n", blocksize);
    printf("grid size = %d \n", gridsize);
    dim3 block_size(blocksize, blocksize, 1);
    dim3 grid_size(blocksize, blocksize, 1);
    printf("%d: block_size = (%d, %d), grid_size = (%d, %d)\n", __LINE__, block_size.x, block_size.y, grid_size.x, grid_size.y);
    int sharedMemSize = sizeof(dtype) * block_size.x * block_size.y * 2;

    //Print device properties
    FILE *file = fopen("deviceProperties.txt", "r");
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    
    printf("==============================================================\n");
    printf("DEVICE PROPERTIES\n");
    char ch;
    while ((ch = fgetc(file)) != EOF) {
        printf("%c", ch);
    }
    fclose(file);

    //Prepare our output files
    FILE *csvtime[NFILES];
    char filename[56];

    for (int i = 0; i < NFILES / 2; i++) {
        sprintf(filename, "output/timeB%d.csv", i);
        csvtime[i] = fopen(filename, "w");
        if (csvtime[i] == NULL) {
            printf("Error opening file!\n");
            return 1;
        }

        sprintf(filename, "output/timeN%d.csv", i);
        csvtime[i + NFILES / 2] = fopen(filename, "w");
        if (csvtime[i + NFILES / 2] == NULL) {
            printf("Error opening file!\n");
            return 1;
        }
    }

    for (int i = 0; i < NFILES; i++) {
        fprintf(csvtime[i], "Time,Dimensions\n");
    }
    
    //Prepare our iterations and preload kernel
    dummyKernel<<<1, 1>>>();
    long long tries = 1 << 0;
    TIMER_DEF;
    float times[NDEVICE] = {0};

    //tries loop
    for (int count = 0; count < tries; count++) {
        cudaStream_t stream;
        checkCudaErrors(cudaStreamCreate(&stream));
        dtype *matrix = NULL, *transpose = NULL, *transposeBlock = NULL;
        matrixInitialize(number, number, &matrix, &transpose, &transposeBlock);

        //Assign random values to matrices
        for (int i = 0; i < number * number; i++) {
            matrix[i] = randomf();
        }

        //Check validity
        if (matrix == NULL || transposeBlock == NULL || transpose == NULL) {
            printf("Memory allocation failed\n");
            return 1;
        }

        //Matrix block transpose
        for (int k = 0; k < NFILES / 2; k++) {
            TIMER_START;
            transposeMyBlockMatrix<<<gridsize, blocksize, sharedMemSize, stream>>>(matrix, transposeBlock, number, number, block);
            checkCudaErrors(cudaGetLastError());
            TIMER_STOP;
            times[0] += TIMER_ELAPSED;
            fprintf(csvtime[k], "%f,%ld\n", TIMER_ELAPSED, number); 
        }
        
        //Matrix normal transpose
        for (int k = 0; k < NFILES / 2; k++) {
            TIMER_START;
            transposeMatrix<<<gridsize, blocksize, sharedMemSize, stream>>>(matrix, transpose, number, number);
            checkCudaErrors(cudaGetLastError());
            TIMER_STOP;
            times[1] += TIMER_ELAPSED;
            fprintf(csvtime[k + NFILES / 2], "%f,%ld\n", TIMER_ELAPSED, number); 
        }

        //Lines for debug purposes
        printMatrix(matrix, number, number, "Matrix");
        printMatrix(transposeBlock, number, number, "transpose block");
        printMatrix(transpose, number, number, "transpose");

        matrixDestroyer(matrix, transpose, transposeBlock);
        checkCudaErrors(cudaStreamDestroy(stream));
        number = number + 1;
    }      

    for (int i = 0; i < NFILES; i++) {
        fclose(csvtime[i]);
    }

    printf("==============================================================\n");
    printf("STATS\n");
    printf("Block Matrix Transpose Effective Bandwidth(GB/s): %f\n", (3 * number * number * sizeof(dtype)) / (1e9 * times[0]));
    printf("Block Matrix Transpose Computational Throughput (GFLOP/s): %f\n", (2 * number * number / sizeof(dtype)) / (1e9 * times[0]));
    printf("Normal Matrix Transpose Effective Bandwidth(GB/s): %f\n", (3 * number * number * sizeof(dtype)) / (1e9 * times[1]));
    printf("Normal Matrix Transpose Computational Throughput (GFLOP/s): %f\n", (2 * number * number / sizeof(dtype)) / (1e9 * times[1]));

    return 0;
}
