#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

float randomf();

__global__ void add(int n, float *x, float *y) {
    y[threadIdx.x] = x[threadIdx.x] + y[threadIdx.x];
}

int main(int argc, char *argv[]){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop); 

    srand(time(NULL));
    long number = strtol(argv[1], NULL, 10);

    float *array;
    float *array2;

    cudaMallocManaged(&array, number * sizeof(float));
    cudaMallocManaged(&array2, number * sizeof(float));

    for (int i = 0; i < number; i++){
        array[i] = randomf();
        array2[i] = randomf();
    }

    cudaEventRecord(start);
    add<<<1, 8>>>(number, array, array2);   
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < number; i++){
        maxError = fmax(maxError, fabs(array2[i]-3.0f));
    }
        
    std::cout << "Max error: " << maxError << std::endl; 


    float time = 0;    
    cudaEventElapsedTime(&time, start, stop); 
    printf("Time spent = %f milliseconds\n", time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(array);
    cudaFree(array2);

    return 0;
}

float randomf(){
    return (float) rand() / (float) RAND_MAX;   
}