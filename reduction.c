#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

float randomf();

int main(int argc, char *argv[]){
    clock_t begin = clock();

    srand(time(NULL));
    long long N = 1 << 10;
    float *array = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++){
        array[i] = randomf();
    }

    float sum = 0;

    for(int i = 0; i < N; i++){
        sum += array[i];
    }
    
    float mean = sum / N;
    float stdev = 0;

    for (int i = 0; i<N; i++){
        stdev += pow(array[i] - mean, 2);
    }
    stdev = sqrt(stdev / N);

    printf("Sum = %f, mean = %f, standard dev = %f\n", sum, mean, stdev);

    clock_t end = clock();
    double time = (double) (end-begin) / CLOCKS_PER_SEC;
    printf("Time spent = %f seconds\n", time);

    int memory_usage = 3 * N * sizeof(float);
    printf("Memory usage = %d bytes\n", memory_usage);

    free(array);
    return 0;
}

float randomf(){
    return (float) rand() / (float) RAND_MAX;   
}