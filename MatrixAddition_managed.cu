#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include "./include/helper_cuda.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define TIMER_DEF     struct timeval temp_1, temp_2

#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)

#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)

#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0)

#define DBG_CHECK if (verbose) { printf("DBG_CHECK: file %s at line %d\n", __FILE__, __LINE__ ); }
#define DEBUG  // without debug (with random imputs) the kernel does not work

#define NPROBS 5
#define STR(s) #s
#define XSTR(s) STR(s)
#define dtype float

#define PRINT_MATRIX(A, N, M, ST ) {  \
      int i, j;  \
      printf("%s:\n", ( ST ));  \
      for (i=0; i< ( N ); i++) {  \
        printf("\t");  \
        for (j=0; j< ( M ); j++)  \
          printf("%6.3f ", A[i*( M ) + j]);  \
        printf("\n");  \
      }  \
      printf("\n\n");  \
}


float matrix_error (int n, int m, const dtype* A, const dtype* B) {
  int i, j;
  dtype error = (dtype)0;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      error += fabs(B[i*m + j] - A[i*m + j]);

  return(error);
}

#define BLOCKSIZE 64    // sgemm_global_memory_coalescing, sgemm_shared_memory_cache_blocking
#define BLOCKEDGE(R) ((BLOCKSIZE)/(R))
#define CEIL_DIV( N, D ) ((( N ) % ( D )) == 0) ? (( N )/( D )) : ((( N )/( D ))+1)

int verbose;

__global__ void naive_kernel(int N, int M, const dtype *A, const dtype *B, dtype *C) {
    // compute position in C that this thread is responsible for
    const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    int my_col = tid % M;
    int my_row = tid / M;
    extern __shared__ dtype sharedData[];

    if (tid < N*M) {
            C[my_row * M + my_col] = A[my_row * M + my_col] + B[my_row * M + my_col];
    }
}

__global__ void kernel_1(int N, int M, const dtype *A, const dtype *B, dtype *C) {
    // compute position in C that this thread is responsible for
    const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    int my_col = tid % M;
    int my_row = tid / M;

      /* |========================================| */
      /* |         Put here your kernels          | */
      /* |========================================| */

    extern __shared__ dtype sharedData[];
    dtype* sharedA = sharedData;
    dtype* sharedB = sharedData + blockDim.x * blockDim.y;
    int index = threadIdx.y * blockDim.x + threadIdx.x;


    if (tid < N*M) {
      sharedA[index] = A[my_row * M + my_col];
      sharedB[index] = B[my_row * M + my_col];
    }

    __syncthreads();

    if (tid < N*M) {
      C[my_row * M + my_col] = sharedA[index] + sharedB[index];
    } 
}

__global__ void kernel_2(int N, int M, const dtype *A, const dtype *B, dtype *C) {
    // compute position in C that this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

      /* |========================================| */
      /* |         Put here your kernels          | */
      /* |========================================| */
    extern __shared__ dtype sharedData[];

    int index = y * N + x;
    if (x < N && y < M) {
      C[index] = A[index] + B[index];
    }   
}

__global__ void kernel_3(int N, int M, const dtype *A, const dtype *B, dtype *C) {
    // compute position in C that this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

      /* |========================================| */
      /* |         Put here your kernels          | */
      /* |========================================| */

    extern __shared__ dtype sharedData[];
    dtype* sharedA = sharedData;
    dtype* sharedB = sharedData + blockDim.x * blockDim.y;

    int sharedIndex = threadIdx.y * blockDim.x + threadIdx.x;
    int index = y * N + x;

    if (x < N && y < M) {
      sharedA[sharedIndex] = A[index];
      sharedB[sharedIndex] = B[index];
    }   

    __syncthreads();

    if (x < N && y < M) {
      C[index] = sharedA[sharedIndex] + sharedB[sharedIndex];
    }   
}

dtype* execute_kernel (int n, int m, dtype* A, dtype* B, void (*kernel)(int, int, const dtype*, const dtype*, dtype*), int blk_ratio, float* Bandwidth, float* CompTime, double* Flops) {
    int grd_sizeX, grd_sizeY;
    int blk_sizeX, blk_sizeY;

    DBG_CHECK
    if ( BLOCKSIZE % blk_ratio != 0) {
        fprintf(stderr, "Error: BLOCKSIZE (%d) is not divisible for blk_ratio (%d)\n", BLOCKSIZE, blk_ratio);
        exit(__LINE__);
    }

    DBG_CHECK
    // ---------------------------------
    char sw = (kernel == naive_kernel) ? '0' : ((kernel == kernel_1) ? '1' : ((kernel == kernel_2) ? '2' : '3' ));
    switch (sw) {
        case '0' :
            blk_sizeX = BLOCKSIZE;
            grd_sizeX = CEIL_DIV(n*m, BLOCKSIZE);
            blk_sizeY = 1;
            grd_sizeY = 1;
            break;
        case '1' :
            /* Modify here your kernel launch dimension */
            blk_sizeX = BLOCKSIZE;
            grd_sizeX = CEIL_DIV(n, BLOCKSIZE);
            blk_sizeY = 1;
            grd_sizeY = 1;
            break;
        case '2' :
            /* Modify here your kernel launch dimension */
            blk_sizeX = BLOCKSIZE;
            grd_sizeX = CEIL_DIV(n, BLOCKSIZE);
            blk_sizeY = 1;
            grd_sizeY = 1;
            break;
        case '3' :
            /* Modify here your kernel launch dimension */
            blk_sizeX = BLOCKSIZE;
            grd_sizeX = CEIL_DIV(n, BLOCKSIZE);
            blk_sizeY = 1;
            grd_sizeY = 1;
            break;
    }
    // ---------------------------------

    DBG_CHECK
    // ------------------- allocating GPU vectors ----------------------
    dtype *dev_A, *dev_B, *dev_C;
    checkCudaErrors( cudaMallocManaged(&dev_A, n*m*sizeof(dtype)) );
    checkCudaErrors( cudaMallocManaged(&dev_B, n*m*sizeof(dtype)) );
    checkCudaErrors( cudaMallocManaged(&dev_C, n*m*sizeof(dtype)) );
    size_t bandwidth_numerator = ( n * m ) * sizeof(dtype);

    DBG_CHECK
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    // ----------------- copy date from host to device -----------------
    //checkCudaErrors( cudaMemcpy(dev_A, A, n*m*sizeof(dtype), cudaMemcpyHostToDevice) );
    //checkCudaErrors( cudaMemcpy(dev_B, B, n*m*sizeof(dtype), cudaMemcpyHostToDevice) );
    //checkCudaErrors( cudaMemset(dev_C, 0, n*m*sizeof(dtype)) );

    DBG_CHECK
    // ---------- compute GPU_tmp_b with the reduction kernel ----------
    TIMER_DEF;
    TIMER_START;

    {
        dim3 block_size(blk_sizeX, blk_sizeY, 1);
        dim3 grid_size(grd_sizeX, grd_sizeY, 1);
        printf("%d: block_size = (%d, %d), grid_size = (%d, %d)\n", __LINE__, block_size.x, block_size.y, grid_size.x, grid_size.y);
        int sharedMemSize = sizeof(dtype) * block_size.x * block_size.y * 2;
        kernel<<<grid_size, block_size, sharedMemSize, stream>>>(n, m, (const dtype*)dev_A, (const dtype*)dev_B, dev_C);
    }

    checkCudaErrors( cudaDeviceSynchronize() );
    TIMER_STOP;
    *CompTime += TIMER_ELAPSED;
    *Bandwidth = bandwidth_numerator / ((*CompTime)*1e+9);
    *Flops  = ( n * m ) / ( (*CompTime) * 1e+9 );

    DBG_CHECK
    // --------------- copy results from device to host ----------------

    dtype *GPU_C;
    checkCudaErrors( cudaMallocManaged(&GPU_C, n*m*sizeof(dtype)) );
    //checkCudaErrors( cudaMemcpy(GPU_C, dev_C, n*m*sizeof(dtype), cudaMemcpyDeviceToHost) );

    if (verbose > 0)
        PRINT_MATRIX(GPU_C, n, m, "GPU_C form execute_kernel")

    DBG_CHECK
    checkCudaErrors( cudaFree(dev_A) );
    checkCudaErrors( cudaFree(dev_B) );
    checkCudaErrors( cudaFree(dev_C) );
    checkCudaErrors(cudaStreamDestroy(stream));

    DBG_CHECK
    return(GPU_C);
}

void usage(char* bin_name, int exit_faulier) {
    fprintf(stderr, "Usage: %s -n <n> -m <m> [-c] [-v]\n", bin_name);
    fprintf(stderr, "Where mandatory inputs are:\n");
    fprintf(stderr, "\t-n\trepresents rows number of the two matrices\n");
    fprintf(stderr, "\t-m\trepresents columns number of the two matrices\n\n");

    fprintf(stderr, "And optional inputs are:\n");
    fprintf(stderr, "\t-c\tif provided it enables CPU compare computation\n");
    fprintf(stderr, "\t-v\tif provided it enables verbose prints\n\n");
    exit(exit_faulier);
}

int main(int argc, char *argv[]) {

    printf("====================================== Problem computations ======================================\n");
    // =========================================== Set-up the problem ============================================

    int p=0;
    int n, m;
    char input;
    int cpuOn_flag = 0;
    while((input = getopt(argc, argv, "n:m:cv")) != EOF) {
#define CHECKRTYPE(exitval,opt) {                                   \
    if (exitval == gread) prexit("Unexpected option -%c!\n", opt);  \
        else gread = !exitval;                                      \
    }
		switch (input) {
			//BC approx  c param is the costanst used in Bader stopping cretierion
			case 'n' :
                sscanf(optarg, "%d", &n);
                if (n <= 0) {
                    fprintf(stderr, "Error: n value must be a positive integer (%d provided)\n", n);
                    usage(argv[0], __LINE__);
                } else {
                    p |= 1;
                }
                break;
            case 'm' :
                sscanf(optarg, "%d", &m);
                if (m <= 0) {
                    fprintf(stderr, "Error: m value must be a positive integer (%d provided)\n", n);
                    usage(argv[0], __LINE__);
                } else {
                    p |= 2;
                }
                break;
            case 'c' :
                    cpuOn_flag = 1;
                break;
			case 'v' :
					verbose = 1;
					break;
			case 'h':
                usage(argv[0], __LINE__);
			case '?':
                fprintf(stderr, "Error: unrecognized parameter (%c)\n\n", input);
                usage(argv[0], __LINE__);
		}
#undef CHECKRTYPE
	}

	if (p != 3 ) {
        fprintf(stderr, "Error: -n and -m parameters are mandatory\n\n");
        usage(argv[0], __LINE__);
    }

  // ---------------- set-up the problem size -------------------



//   printf("e = %d --> n = k = m = 2^(e/2) = %d\n", e, n);
//   printf("alpha = %f, beta = %f\n", alpha, beta);
  printf("CPU_ON = %d\n", cpuOn_flag);
  printf("verbose = %d\n", verbose);
  printf("dtype = %s\n", XSTR(dtype));

  // ======================================== Get the device properties ========================================
  printf("======================================= Device properties ========================================\n");

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  int dev;
  for (dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

    printf("  Memory Clock rate:                             %.0f Mhz\n",
           deviceProp.memoryClockRate * 1e-3f);

    printf("  Memory Bus Width:                              %d bit\n",
           deviceProp.memoryBusWidth);

    printf("  Peak Memory Bandwidth:                     %7.3f GB/s\n",
           2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth/8)/1.0e6);

    printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
           deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
               deviceProp.multiProcessorCount);

    printf("  Peak Arithmetic Intensity:                     %7.3f GFLOPS/s\n",
           2.0*deviceProp.memoryClockRate*(_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
               deviceProp.multiProcessorCount)/1.0e6);

  }

  // ------------------ set-up the timers ---------------------

  TIMER_DEF;
  const char* lables[NPROBS] = {"CPU check", "Naive Kernel", "Kernel 1", "Kernel 2", "Kernel 3"};
  float errors[NPROBS], Times[NPROBS], Bandwidths[NPROBS], error;
  double Flops[NPROBS];
  for (int i=0; i<NPROBS; i++) {
    Bandwidths[i] = 0.0;
    errors[i] = -1.0;
    Flops[i] = 0.0;
    Times[i] = 0.0;
  }


  DBG_CHECK
  // ------------------- set-up the problem -------------------

  dtype *A, *B, *GPU_C, *CPU_C;
  checkCudaErrors( cudaMallocManaged(&A, n*m*sizeof(dtype)) );
  checkCudaErrors( cudaMallocManaged(&B, n*m*sizeof(dtype)) );
  checkCudaErrors( cudaMallocManaged(&CPU_C, n*m*sizeof(dtype)) );
  checkCudaErrors( cudaMallocManaged(&GPU_C, n*m*sizeof(dtype)) );

  time_t t;
  srand((unsigned) time(&t));


  for (int i=0; i<(n*m); i++) {
    A[i] = ((dtype)(i/m)/(dtype)m) + 1.0f;
    B[i] = (dtype)(1);
  }

#ifdef DEBUG
  if (verbose > 0) {
    PRINT_MATRIX(A, n, m, "A")
    PRINT_MATRIX(B, n, m, "B")
  }
#endif
  // ======================================== Running the computations =========================================

  /* [ ... ]
   */

  DBG_CHECK
  // ========================== CPU computation =========================
  if (cpuOn_flag) {

    TIMER_START;
    for (int i=0; i<n; i++)
      for (int j=0; j<m; j++)
        CPU_C[i*m +j] = A[i*m + j] + B[i*m + j];
    TIMER_STOP;

    Times[0] = TIMER_ELAPSED;
    errors[0] = 0.0f;
    Bandwidths[0] = -1.0f;
    Flops[0]  = (n*m) / (Times[0]*1e+9);

    if (verbose > 0)
      PRINT_MATRIX(CPU_C, n, m, "CPU_C")

    printf("CPU time: %lf\n", Times[0]);

  } else {
    Times[0] = -1.0f;
    errors[0] = -1.0f;
    Bandwidths[0] = -1.0f;
    Flops[0] = -1.0f;
  }

  DBG_CHECK
  // =========================== GPU naive Kernel ===========================
  printf("=========================== GPU naive Kernel ===========================\n");

  DBG_CHECK
  GPU_C = execute_kernel(n, m, A, B, naive_kernel, 1, &Bandwidths[1], &Times[1], &Flops[1]);

  // ------------- Compare GPU and CPU solution --------------

  (cpuOn_flag) ? (error = matrix_error(n, m, CPU_C, GPU_C)) : (error = 0.0f) ;
  errors[1] = error;

  if (verbose > 0)
    PRINT_MATRIX(GPU_C, n, m, "GPU_C")
  printf("Error equal to %lf\n", error);

  cudaFree(GPU_C);

  // =========================== GPU Kernel 1 ===========================
  printf("=========================== GPU Kernel 1 ===========================\n");

  GPU_C = execute_kernel(n, m, A, B, kernel_1, 1, &Bandwidths[2], &Times[2], &Flops[2]);

  // ------------- Compare GPU and CPU solution --------------

  (cpuOn_flag) ? (error = matrix_error(n, m, CPU_C, GPU_C)) : (error = 0.0f) ;
  errors[2] = error;

  if (verbose > 0)
    PRINT_MATRIX(GPU_C, n, m, "GPU_C")

  cudaFree(GPU_C);

  // =========================== GPU Kernel 2 ===========================
  printf("=========================== GPU Kernel 2 ===========================\n");

  GPU_C = execute_kernel(n, m, A, B, kernel_2, 1, &Bandwidths[3], &Times[3], &Flops[3]);

  // ------------- Compare GPU and CPU solution --------------

  (cpuOn_flag) ? (error = matrix_error(n, m, CPU_C, GPU_C)) : (error = 0.0f) ;
  errors[3] = error;

  if (verbose > 0)
    PRINT_MATRIX(GPU_C, n, m, "GPU_C")

  cudaFree(GPU_C);

  // =========================== GPU Kernel 3 ===========================
  printf("=========================== GPU Kernel 3 ===========================\n");

  GPU_C = execute_kernel(n, m, A, B, kernel_3, 1, &Bandwidths[4], &Times[4], &Flops[4]);

  // ------------- Compare GPU and CPU solution --------------

  (cpuOn_flag) ? (error = matrix_error(n, m, CPU_C, GPU_C)) : (error = 0.0f) ;
  errors[4] = error;

  if (verbose > 0)
    PRINT_MATRIX(GPU_C, n, m, "GPU_C")

  cudaFree(GPU_C);



  printf("\n\n");
  if (!(cpuOn_flag)) printf("CPU check not lunched!!\n");
  printf("Solution\n %9s\t%9s\t%9s\t%16s\t%16s\n", "type", "error", "time (s)", "flops (GFLOPS/s)", "bandwidth (GB/s)");
  for (int i=0; i<NPROBS; i++) {
    if ((i != 6))
      printf("%12s:\t%9.6f\t%9.6f\t%16.6lf\t%16.6f\n", lables[i], errors[i], Times[i], Flops[i], Bandwidths[i]);
  }
  printf("\n");

  printf("GPU times: n*m Kernel1_time Kernel1_flops Kernel2_time Kernel2_flops ... on stderr\n");
  fprintf(stderr, "%d, ", n*m);
  for (int i=1; i<NPROBS; i++)
    fprintf(stderr, "%f, %f, ", Times[i], Flops[i]);
  fprintf(stderr, "\n");

  return(0);
}
