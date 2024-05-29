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

    float **matrix = (float**)malloc(number * sizeof(float*));
    for (int i = 0; i < number; i++){
        matrix[i] = (float*)malloc(number * sizeof(float));
    }

    float **matrix2 = (float**)malloc(number * sizeof(float*));
    for (int i = 0; i < number; i++){
        matrix2[i] = (float*)malloc(number * sizeof(float));
    }

    for (int i = 0; i < number; i++){
        for (int j = 0; j < number; j++){
            matrix[i][j] = randomf();
            matrix2[i][j] = randomf();
        }
    }

    float **matrix3 = (float**)malloc(number * sizeof(float*));
    for (int i = 0; i < number; i++){
        matrix3[i] = (float*)malloc(number * sizeof(float));
    }

    for(int i = 0; i < number; i++){
        for (int j = 0; j < number; j++){
            matrix3[i][j] = 0;
            for (int k = 0; k < number; k++){
                matrix3[i][j] += matrix[i][k] * matrix[k][j];
            }
        }
    }

    clock_t end = clock();
    double time = (double) (end-begin) / CLOCKS_PER_SEC;
    printf("Time spent = %f seconds\n", time);

    int memory_usage = 3 * number * number * sizeof(float);
    printf("Memory usage = %d bytes\n", memory_usage);

    for (int i = 0; i < number; i++){
        free(matrix[i]);
    }
    
    free(matrix);

    for (int i = 0; i < number; i++){
        free(matrix2[i]);
    }

    free(matrix2);

    for (int i = 0; i < number; i++){
        free(matrix3[i]);
    }

    free(matrix3);

    return 0;
}

float randomf(){
    return (float) rand() / (float) RAND_MAX;   
}