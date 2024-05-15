#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <stdint.h>

#define dtype float

dtype randomf();

__global__ void add(int n, dtype *x, dtype *y) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
   if(tid < n){
        y[tid] = x[tid] + y[tid];
   }
}

int main(int argc, char *argv[]){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop); 

    srand(time(NULL));
    int power = strtol(argv[1], NULL, 10);
    long number = pow(2, power);

    dtype *array;
    dtype *array2;

    cudaMallocManaged(&array, number * sizeof(dtype));
    cudaMallocManaged(&array2, number * sizeof(dtype));

    for (int i = 0; i < number; i++){
        array[i] = randomf();
        array2[i] = randomf();
    }
    
    int numBlock = 8;
    int numGrid = (number + numBlock - 1) / numBlock;

    cudaEventRecord(start);
    add<<<numGrid, numBlock>>>(number, array, array2);   
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < number; i++){
        maxError = fmax(maxError, fabs(array2[i]-3.0f));
    }
        
    std::cout << "blk_size/grd_size: " << numBlock/numGrid << std::endl; 


    float time = 0;    
    cudaEventElapsedTime(&time, start, stop); 
    printf("Time spent = %f milliseconds\n", time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(array);
    cudaFree(array2);

    return 0;
}

dtype randomf(){
    return (dtype) rand() / (dtype) RAND_MAX;   
}