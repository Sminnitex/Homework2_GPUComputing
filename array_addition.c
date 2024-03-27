#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

float randomf();

int main(int argc, char *argv[]){
    clock_t begin = clock();

    srand(time(NULL));
    long power = strtol(argv[1], NULL, 10);
    long number = pow(2, power);
    float *array = (float*)malloc(number * sizeof(float));
    float *array2 = (float*)malloc(number * sizeof(float));

    for (int i = 0; i < number; i++){
        array[i] = randomf();
        array2[i] = randomf();
    }

    float *array3 = (float*)malloc(number * sizeof(float));

    for(int i = 0; i < number; i++){
        array3[i] = array[i] + array2[i];
        printf("%f + %f = %f\n", array[i], array2[i], array3[i]);
    }

    clock_t end = clock();
    double time = (double) (end-begin) / CLOCKS_PER_SEC;
    printf("Time spent = %f seconds\n", time);

    int memory_usage = 3 * number * sizeof(float);
    printf("Memory usage = %d bytes\n", memory_usage);

    free(array);
    free(array2);
    free(array3);
    return 0;
}

float randomf(){
    return (float) rand() / (float) RAND_MAX;   
}