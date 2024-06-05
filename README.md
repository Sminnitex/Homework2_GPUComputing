#   To run the code
Is sufficient to run the following commands on the repo folder
``` 
make
./bin/assignment2
``` 
And to check the plots on python
``` 
python plot.py
``` 

This project is the direct evolution of https://github.com/Sminnitex/Homework1_GPUComputing
Therefore the prerequisites are the same as the one described in that repository, and are:

>   python3 equipped with matplotlib, pandas and numpy
>   cuda and the cuda toolkit
>   gcc and cmake

#   Section 1: problem description
Our task is to perform a matrix transpose in C, basically the operation consists in invert rows and columns of our source Matrix. To formalize a little bit this concept, we can refer as the transpose of a matrix A as A^T; considering m the dimension of the rows of the matrix A, and n the columns (for brevity I will use the notation A[m][n] to refer to a generic matrix of dimensions m and n), the transpose operation will give us a new matrix A^T[n][m].

![alt text](https://github.com/Sminnitex/Homework2_GPUComputing/blob/master/figures/Important-Questions-for-Class-10-Maths-Chapter-8-Introduction-to-Trigonometry-1.png?raw=true)


The goal is to develop an algorithm capable of doing this operation and analyze the performance considering different levels of optimization.

The code works taking as input the dimensions of our matrices, and giving as output the time of the operation. To visualize better the process we will diminish gradually the dimension of our matrices and see how this changes the performance on the long run, on a basis of about $500$ iterations. We will also compare the results of two different techniques to perform this task: the normal matrix transpose and the block matrix transpose. 

The base algorithm where we will perform the transpose simply performing over two for loops a change of coordinates, this will give us the possibility of performing the task easily even in the case of a non symmetric matrix. 

In the block transpose the function we will need to pass a new parameter, block size, and the operation will be divided in small blocks over the whole size of the matrix using 4 nested loops. To perform a fair comparison we will initialize our matrices with random float values and compare the time only of the transpose operation. At the end of each time check we will increase the rows and columns dimension of 1 unit for all the 500 iteration of our for loop.

#   Section 2: experimental results
The code will be performed on a specific hardware, my laptop which is an Acer Aspire E5 produced in 2017.

The laptop is equipped with an Intel Core i5-7200U CPU. This CPU has 2 cores and 4 threads, 3MB of cache and 34 GB/s of max bandwidth. Then the PC is equipped with 12GB of RAM, and Windows 11 as OS (even if for convenience the code will run on WSL). To analyze the cache behavior Valgrind with the tool cache-grind will be used, so let's take a look to the plot of the time to perform the transpose with the normal and block technique, changing the dimension of the matrices:

![alt text](https://github.com/Sminnitex/Homework2_GPUComputing/blob/master/figures/Performance_my_laptop.png?raw=true)

In the plot is analyzed also the performance using four different optimization levels for GCC compiler that goes from -O0 (No optimization), to -O3 (aggressive optimization). As we can see the results are pretty interesting, in fact not only the normal transpose is often faster than the block transpose, but analyzing the statistics we can see how in both cases the -O3 optimization is not the best overall. In fact in the block transpose the -O1 optimization is more consistent in the results having a lower standard deviation, and a lower minimum peak, however the median and mean of the time used is slightly higher, this can be easily explained noticing how in the -O3 run there are way more outliers in the results, with spike of time consuming transpose. In the normal transpose the -O2 optimization is easily the fastest, with better statistics in everything except the global minimum value compared to the -O3. Let's try to make a sense of these results taking a look to the cachegrind file, on figure 3.

![alt text](https://github.com/Sminnitex/Homework2_GPUComputing/blob/master/figures/cachegrindSummary.png?raw=true)

We can read the file on figure 3 as it follows: we basically have three different instructions, and these are Instruction executed, Data read and Data writes. For each of the instructions we have three measurements: the total number of time an instruction is called, the cache misses on the first level, and the cache misses on the last level. The two percentage in brackets refer to the contribution of a measure referred to the whole program, or for the given function. So considering the first percentage in the image: the entire code is 100% of the instructions executed, and our block transpose function represents 26% of the calls of our program. In absolute number it has been read over 30 billion times and has a ridiculous amount of cache misses of first and last level. However the number of cache misses for reading and writing operation on Data is relevant, and slow down the whole program. 

Now if we try to take a look to a normal transpose cachegrind output we will notice something important.

![alt text](https://github.com/Sminnitex/Homework2_GPUComputing/blob/master/figures/cachegrinNormSumm.png?raw=true)

As we can easily see comparing the two methods it immediately catches the eyes that the block transpose has been called way more times than the normal transpose. This could depend on the complexity of the function, that in the block case uses more parameters and more instruction, needing four nested loop instead of two. But what it matters is the number of cache misses, and we can see how both function struggles a bit, resulting in a good number of misses on L1 and LL. The percentage of misses on L1 cache is always grater than the misses on LL, this should not be surprising since the L1 cache is smaller then the LL; the access on L1 is constant and the misses very frequent.

The misses are over 90% in the Data read section both for the block and normal transpose, at every optimization flag. This possibly means that my architecture is a bottle neck for the program. However is quite interesting to compare this results with other, obtained by the same hardware but using, instead of the WSL, a Linux system on the bare metal. The performance obtained are consistently better from every point of view, that is related to the better use of memory access: while WSL has to communicate with windows in order to obtain the data from my C program, a Linux OS can manage everything on it's own, and the new graph in this case are the following:

![alt text](https://github.com/Sminnitex/Homework2_GPUComputing/blob/master/figures/Performance_linux.png?raw=true)

The results are still influenced by many outliers, but this time is more clear what levels of optimization perform better then the other, and this time analyzing the stats we can see how the block matrix transpose works way better with the aggressive optimization, while the normale transpose perform better without optimization. Analyzing these cachegrind files we can see how this changes are reflected with a percentage of cache misses in writing data that drops especially for the block transpose. So the communication between OS was the real bottleneck in the application and the memory is managed way better by the Linux OS.

Now let's try to calculate the bandwidth to have more data, we will perform this analysis on the Linux results since they are more consistent and the communication between OS seems to make the code perform worse. As we know the formula to calculate the bandwidth is:


>    effectiveBandwidth = ((Dr + Dw) / 10^9) / t


![alt text](https://github.com/Sminnitex/Homework2_GPUComputing/blob/master/figures/bandwidth.png?raw=true)

So as we can expect seeing the time metrics the bandwidth confirms that this code is more efficient for the normal matrix transpose, and in particular for the one -O0 optimization flag. The block matrix transpose, despite being less efficient has a good growth in performance for higher optimization flags.

#   Section3: CUDA Algorithm
Now we are going to consider an adaptation of our algorithm to run on GPUs with the CUDA APIs. In this case our analysis will no longer consider the normal and block algorithm, but the difference will be underlined by the use of a kernel that uses global or shared memory.

Then again, instead of using the optimization flags comparing how the code reacts to each of them, we will modify the grid and block sizes with which we start the kernel, and therefore the number of threads per block in each iteration. Once again the code will run multiple operations and compute the effective bandwidth, the time and we will progressivly increase the size of the matrices.

Starting from a Kernel that uses global memory, we have to modify the access index that isn't anymore sequential but has to make use of parallelization

The hyperparameter "TILE_DIM" is the dimensions of the square tiles used for the matrix transpose operation, and each time we update the counter i of another hyper-parameter called "BLOCK_ROWS", which instead control the number of rows that each thread processes within a single tile. All considered, though parallelized, is a normal matrix transpose algorithm.

If instead we want to optimize even more considering a shared memory approach, we have to transfer the matrix from global to shared memory and then apply the transpose for every tile, in a, conceptually,  similar way to what we have done with the block matrix transpose.

The main difference is that, instead of relying on device memory, we make use of on-chip memory that has a much lower latency and higher bandwidth. So instead of having each thread directly reading and writing directly on global memory, threads first load data into shared memory (tile) before performing the transpose. This allows for coalesced global memory reads, as threads in a warp access contiguous memory locations. The __syncthreads() call therefore is needed to ensure that all
reads from the source matrix to shared memory have completed before writing from shared memory to the transpose matrix. Notice that on the shared memory is necessary of increasing by 1 the dimension of the tile dimension to avoid bank conflicts.

#   Section 4: experiments with GPU
This time the experiments will be tried both on my laptop that uses a NVIDIA 940MX as GPU, with maximum bandwidth of 40.02 GB/s; but also with the Unitn cluster that is equipped with an A30 with maximum bandwidth of 870.22GB/s.

The experiments will vary on the block and grid size instead that with optimization flag, as said in the last section, and we won't use Valgrind to help us in our analysis.

Let's start our analysis considering my laptop performance on time and bandwidth, the matrix dimension vary from a 1024x1024 non symmetrical matrix to a 2047x2047 matrix, the result compared with the ones of section 2 are even 100 times faster in the worst cases scenario on the outliers, and about 20 times faster normally.

![alt text](https://github.com/Sminnitex/Homework2_GPUComputing/blob/master/figures/940MXTime.png?raw=true)
https://github.com/Sminnitex/Homework2_GPUComputing/blob/master/figures/940MXTime.png

The plot in \textbf{figure 7} gives us the possibility to compare the shared and global matrix transposes, and according to the change of block and grid size one of the techniques is consistently better then the other. For a block-grid size of 64-14 the global matrix transpose performs way better, but for the 32-7 the shared memory is definitely lower. The measures for 16-3 and 8-1 appears flat really near the 0, let's analyze the bandwidth result to understand more about this.

![alt text](https://github.com/Sminnitex/Homework2_GPUComputing/blob/master/figures/940MXBandwidth.png?raw=true)

As we can see the theoretical maximum bandwidth possible of 40.02GB/s for my GPU is often broken, in 5 out of 8 cases. The only reliable result are the 64-14 measure of both techniques, where in each case we are under 10GB/s of bandwidth, and the global matrix transpose for the 32-7 where we are over 25GB/s of bandwidth, so theoretically the best measure. Unfortunately the other measurement aren't reliable and every measure greater for block and grid size gives me errors. So to obtain a more reliable analysis i repeated the experiments on the cluster equipped with an A30. Starting from the time and dimension measure:
