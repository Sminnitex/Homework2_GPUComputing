#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include "./include/helper_cuda.h"
#include <cuda_runtime.h>

#define NPROBS 3

#define TIMER_DEF     struct timeval temp_1, temp_2

#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)

#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)

#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0)

#define DBG_CHECK { printf("DBG_CHECK: file %s at line %d\n", __FILE__, __LINE__ ); }
// #define DEBUG
// #define BLK_DISPACH

#define STR(s) #s
#define XSTR(s) STR(s)
#define dtype double

// #include "../solutions/lab2_sol.cu"
//#define RUN_SOLUTIONS

#define BLK_SIZE 128
#define GRD_SIZE 14

__device__ uint get_smid(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}

__global__
void example_kernel(int n, dtype *a, dtype* b, dtype* c)
{
  if (threadIdx.x==0)
      printf("block %d runs on sm %d\n", blockIdx.x, get_smid());

  // [ ... ]
}


#ifdef RUN_SOLUTIONS

#else
            /* |========================================| */
            /* |         Put here your kernels          | */
            /* |========================================| */

  __global__ void my_kernel(int len, dtype *a, dtype* b, dtype* c){
      //if (threadIdx.x==0)
        //printf("block %d runs on sm %d\n", blockIdx.x, get_smid());

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int nthreads = gridDim.x * blockDim.x;
    int tsize = len / nthreads;
    if (len % nthreads != 0)
      tsize++;

    for(int i = tid; i < len; i += nthreads){
      c[i] = a[i] + b[i];
    }
  }


  __global__ void my_second_kernel(int len, dtype *a, dtype* b, dtype* c){
    //if (threadIdx.x==0)
      //printf("block %d runs on sm %d\n", blockIdx.x, get_smid());

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int nthreads = gridDim.x * blockDim.x;
    int tsize = len / nthreads;
    if (len % nthreads != 0)
      tsize++;

    int accessindex = tid * tsize;
    for(int i = 0; i < tsize && accessindex < len; i++){
      c[accessindex] = a[accessindex] + b[accessindex];
      accessindex ++;
    } 
  }

#endif



int main(int argc, char *argv[]) {

  // ======================================== Get the device properties ========================================
  printf("======================================= Device properties ========================================\n");
  
  /* The GPU memory bandwidth is the number of bytes per second which can be transferred between host and device memory.
   * So, you can obtain it by multiplying the number of exchanges done in one second by the bytes exchanged in a single operation.
   * Use the "cudaDeviceProp" structure to compute it by remembering that:
   *    1) The number of operations done in one second is given by doubling the memory clock rate
   *    2) The total amount of bytes exchanged (i.e. the memory bus width) is given by the number of memory controllers
   *        by the width of a single memory controller (that is usually expressed in bits, so divide it by 8 to have the bytes).
   *
   * Take as an example "deviceQuery.cpp" in "https://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/deviceQuery/".
   *
   * Once you complete the exercise, compare his bandwidth with the theoretical maximum computed here.
   */

#ifdef RUN_SOLUTIONS

#else
            /* |========================================| */
            /* |           Put here your code           | */
            /* |========================================| */

  FILE *file = fopen("deviceProperties.txt", "r");
  if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
  
  char ch;
  while ((ch = fgetc(file)) != EOF) {
        printf("%c", ch);
    }


#endif

  printf("====================================== Problem computations ======================================\n");
// =========================================== Set-up the problem ============================================

  if (argc < 2) {
    printf("Usage: lab2_ex1 n\n");
    return(1);
  }
  printf("argv[1] = %s\n", argv[1]);

  // ---------------- set-up the problem size -------------------

  int n = atoi(argv[1]), len = (1<<n), i;

  printf("n = %d --> len = 2^(n) = %d\n", n, len);
  printf("dtype = %s\n", XSTR(dtype));


  // ------------------ set-up the timers ---------------------

  TIMER_DEF;
  const char* lables[NPROBS] = {"CPU", "GPU Layout 1", "GPU Layout 2"};
  float errors[NPROBS], Times[NPROBS], error, MemTime = 0.0;
  long int bytes_moved = 0;
  for (i=0; i<NPROBS; i++) {
    errors[i] = 1<<30;
    Times[i] = 0;
  }


  // ------------------- set-up the problem -------------------

  dtype *a, *b, *CPU_c, *GPU_c;
  a = (dtype*)malloc(sizeof(dtype)*len);
  b = (dtype*)malloc(sizeof(dtype)*len);
  CPU_c = (dtype*)malloc(sizeof(dtype)*len);
  GPU_c = (dtype*)malloc(sizeof(dtype)*len);
  time_t t;
  srand((unsigned) time(&t));

  int typ = (strcmp( XSTR(dtype) ,"int")==0);
  if (typ) {
      // here we generate random ints
      int rand_range = (1<<11);
      printf("rand_range= %d\n", rand_range);
      for (i=0; i<len; i++) {
          a[i] = rand()/(rand_range);
          b[i] = rand()/(rand_range);
          GPU_c[i] = (dtype)0;
      }
  } else {
      // here we generate random floats
      for (i=0; i<len; i++) {
        a[i] = (dtype)rand()/((dtype)RAND_MAX);
        b[i] = (dtype)rand()/((dtype)RAND_MAX);
        GPU_c[i] = (dtype)0;
      }
  }


// ======================================== Running the computations =========================================

  /* Write two different cuda kernels that perform the vector addition with different memory access layouts.
   *  Let k be the total amount of defined threads (i.e. the blockDim*gridDim) and n the vector length; the
   *  first kernel will access in the following way:
   *
   *     th0   th1   th2   ...   ...   thk   th0   th1   ...
   *      |     |     |                 |     |     |
   *      |     |     |                 |     |     |
   *    -----------------------------------------------------------------------
   *   |  0  |  1  |  2  | ... | ... |  k  | k+1 | k+2 | ... |     | ... |  n  |
   *    -----------------------------------------------------------------------
   *
   *
   *  Instead, the second kernel will access in this way:
   *
   *
   *     th0   ...   ...   th0   th1   ...         ...     thk     ...    ...  thk
   *      |                 |     |                         |                   |
   *      |                 |     |                         |                   |
   *    ---------------------------------------------------------------------------
   *   |  0  | ... | ... | n/k | ... |     |     | ... | n-(n/k) | ... | ... |  n  |
   *    ---------------------------------------------------------------------------
   *
   *
   * Find two block_size/grid_size such that:
   *    1) All the operations are performed by a single-stream multiprocessor
   *    2) All your stream multiprocessors compute some operations
   *
   * Check this by using "uint get_smid" as in the example_kernel
   */

  // ========================== CPU computation =========================
  TIMER_START;
  for (i=0; i<len; i++)
    CPU_c[i] = a[i] + b[i];
  TIMER_STOP;
  errors[0] = 0.0;
  Times[0] = TIMER_ELAPSED;



  // ================== GPU computation with Layout 1 ===================

  // ---------------- allocing GPU vectors -------------------
  dtype *dev_a, *dev_b, *dev_c;

  checkCudaErrors( cudaMalloc(&dev_a, len*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_b, len*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_c, len*sizeof(dtype)) );


#ifdef RUN_SOLUTIONS

#else
            /* |========================================| */
            /* | Put here your code for solve Problem 1 | */
            /* |========================================| */

  // ------------ copy date from host to device --------------
  cudaMemcpy(dev_a, a, len*sizeof(dtype), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, len*sizeof(dtype), cudaMemcpyHostToDevice);


  // ------------ computation solution with Layout 1 -----------
  TIMER_START;
  my_kernel<<<GRD_SIZE, BLK_SIZE>>>(len, dev_a, dev_b, dev_c);

  checkCudaErrors( cudaDeviceSynchronize() );
  TIMER_STOP;
  Times[1] += TIMER_ELAPSED;

  // ----------- copy results from device to host ------------
  cudaMemcpy(GPU_c, dev_c, len*sizeof(dtype), cudaMemcpyDeviceToHost);


#endif

  // ------------- Compare GPU and CPU solution --------------

  error = 0.0f;
  for (i = 0; i < len; i++)
    error += (float)fabs(CPU_c[i] - GPU_c[i]);
  errors[1] = error;


  // ================== GPU computation with Layout 2 ===================


#ifdef RUN_SOLUTIONS


#else
            /* |========================================| */
            /* | Put here your code for solve Problem 2 | */
            /* |========================================| */

  // ---------------- Reset the memory in dev_c ----------------
  cudaMemset(dev_c, 0, len * sizeof(dtype));

  // ------------ computation solution with Layout 2 -----------
  TIMER_START;
  my_second_kernel<<<GRD_SIZE, BLK_SIZE>>>(len, dev_a, dev_b, dev_c);


  checkCudaErrors( cudaDeviceSynchronize() );
  TIMER_STOP;
  Times[2] += TIMER_ELAPSED;
  // ------------ copy results from device to host -------------
  cudaMemcpy(GPU_c, dev_c, len*sizeof(dtype), cudaMemcpyDeviceToHost);


#endif


  // ------------- Compare GPU and CPU solution --------------

  error = 0.0f;
  for (i = 0; i < len; i++)
    error += (float)fabs(CPU_c[i] - GPU_c[i]);
  errors[2] = error;

  // ----------------- free GPU variable ---------------------

  checkCudaErrors( cudaFree(dev_a) );
  checkCudaErrors( cudaFree(dev_b) );
  checkCudaErrors( cudaFree(dev_c) );

  // ---------------------------------------------------------


// ============================================ Print the results ============================================

#ifdef RUN_SOLUTIONS
  printf("================================= Times and results of SOLUTIONS =================================\n");
#else
  printf("================================== Times and results of my code ==================================\n");
#endif
  printf("Solution type\terror\ttime\n");
  for (int i=0; i<NPROBS; i++) {
    printf("%12s:\t%5.3f\t%5.3f\n", lables[i], errors[i], Times[i]);
  }
  printf("\n");

#ifdef RUN_SOLUTIONS


#else
  // Print here your Memory Bandwidth an the Throughput of both the GPU computations
  printf("Layout 1: Effective Bandwidth (GB/s): %f\n", (3 * len * sizeof(dtype))/(1e9 * Times[1]));
  printf("Layout 1: Computational Throughput (GFLOP/s): %f\n", (2 * len / sizeof(dtype))/(1e9 * Times[1]));

  printf("Layout 2: Effective Bandwidth (GB/s): %f\n", (3 * len * sizeof(dtype))/(1e9 * Times[2]));
  printf("Layout 2: Computational Throughput (GFLOP/s): %f\n", (2 * len / sizeof(dtype))/(1e9 * Times[2]));

#endif


// ==================================== Iterate over the kernel dimensions ====================================

  /*  Now, iterate over different block and grid sizes to find the configuration with better performance. The
   *   idea is to compute something similar to this:
   *
   *  Layout1 times:
   *  blk_size\grd_size:      1      3      7     14     28     56
   *                 32:      X      X      X      X      X      X
   *                 64:      X      X      X      X      X      X
   *                128:      X      X      X      X      X      X
   *                256:      X      X      X      X      X      X
   *                512:      X      X      X      X      X      X
   *               1024:      X      X      X      X      X      X
   *
   *  Note: since we are here not interested in the memory performance and checking again if the results are
   *    correct, copy the vectors a and b on the GPU only the first time and don't copy back the results of c.
   */



  checkCudaErrors( cudaMalloc(&dev_a, len*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_b, len*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_c, len*sizeof(dtype)) );

  checkCudaErrors( cudaMemcpy(dev_a, a, len*sizeof(dtype), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dev_b, b, len*sizeof(dtype), cudaMemcpyHostToDevice) );

#ifdef RUN_SOLUTIONS



#else
          /* |========================================| */
          /* |           Put here your code           | */
          /* |========================================| */

  int customGridSize;
  int customBlockSize = 32;
  float my_times[72];

  for(int i = 0; i < 6; i++){
    customGridSize = 1;
    for(int j = 0; j < 6; j++){
      cudaMemset(dev_c, 0, len * sizeof(dtype));
      TIMER_START;
      my_kernel<<<customGridSize, customBlockSize>>>(len, dev_a, dev_b, dev_c);

      checkCudaErrors( cudaDeviceSynchronize() );
      TIMER_STOP;
      my_times[i * 6 + j] = TIMER_ELAPSED;


      if (j < 3){
        customGridSize = (customGridSize * 2) + 1;
      }else{
        customGridSize = customGridSize * 2;
      }
    }
    customBlockSize = customBlockSize * 2;
  }

  printf("Layout1 times:\n");
  printf("blk_size/grd_size:\t1\t3\t7\t14\t28\t56\n");
   printf("\t\t32:\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n", my_times[0], my_times[1], my_times[2], my_times[3], my_times[4], my_times[5]);
   printf("\t\t64:\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n", my_times[6], my_times[7], my_times[8], my_times[9], my_times[10], my_times[11]);
   printf("\t\t128:\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n", my_times[12], my_times[13], my_times[14], my_times[15], my_times[16], my_times[17]);
   printf("\t\t256:\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n", my_times[18], my_times[19], my_times[20], my_times[21], my_times[22], my_times[23]);
   printf("\t\t512:\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n", my_times[24], my_times[25], my_times[26], my_times[27], my_times[28], my_times[29]);
   printf("\t\t1024:\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n", my_times[30], my_times[31], my_times[32], my_times[33], my_times[34], my_times[35]);

  customBlockSize = 32;

  for(int i = 0; i < 6; i++){
    customGridSize = 1;
    for(int j = 0; j < 6; j++){
      cudaMemset(dev_c, 0, len * sizeof(dtype));
      TIMER_START;
      my_second_kernel<<<customGridSize, customBlockSize>>>(len, dev_a, dev_b, dev_c);

      checkCudaErrors( cudaDeviceSynchronize() );
      TIMER_STOP;
      my_times[i * 6 + j + 36] = TIMER_ELAPSED;

      if (j < 3){
        customGridSize = (customGridSize * 2) + 1;
      }else{
        customGridSize = customGridSize * 2;
      }
        
    }
    customBlockSize = customBlockSize * 2;
  }


   printf("Layout2 times:\n");
  printf("blk_size/grd_size:\t1\t3\t7\t14\t28\t56\n");
   printf("\t\t32:\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n", my_times[36], my_times[37], my_times[38], my_times[39], my_times[40], my_times[41]);
   printf("\t\t64:\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n", my_times[42], my_times[43], my_times[44], my_times[45], my_times[46], my_times[47]);
   printf("\t\t128:\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n", my_times[48], my_times[49], my_times[50], my_times[51], my_times[52], my_times[53]);
   printf("\t\t256:\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n", my_times[54], my_times[55], my_times[56], my_times[57], my_times[58], my_times[59]);
   printf("\t\t512:\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n", my_times[60], my_times[61], my_times[62], my_times[63], my_times[64], my_times[65]);
   printf("\t\t1024:\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n", my_times[66], my_times[67], my_times[68], my_times[69], my_times[70], my_times[71]);
   


#endif


  checkCudaErrors( cudaFree(dev_a) );
  checkCudaErrors( cudaFree(dev_b) );
  checkCudaErrors( cudaFree(dev_c) );

  free(a);
  free(b);
  free(CPU_c);
  free(GPU_c);

  return(0);
}
