#Section 1: problem description
Our task is to perform a matrix transpose in C, basically the operation consists in invert rows and columns of our source Matrix. To formalize a little bit this concept, we can refer as the transpose of a matrix A as A^T; considering m the dimension of the rows of the matrix A, and n the columns (for brevity I will use the notation A[m][n] to refer to a generic matrix of dimensions m and n), the transpose operation will give us a new matrix A^T[n][m].

![alt text](https://github.com/Sminnitex/Homework1_GPUComputing/blob/master/figures/Important-Questions-for-Class-10-Maths-Chapter-8-Introduction-to-Trigonometry-1.png?raw=true)


The goal is to develop an algorithm capable of doing this operation and analyze the performance considering different levels of optimization.

The code works taking as input the dimensions of our matrices, and giving as output the time of the operation. To visualize better the process we will diminish gradually the dimension of our matrices and see how this changes the performance on the long run, on a basis of about $500$ iterations. We will also compare the results of two different techniques to perform this task: the normal matrix transpose and the block matrix transpose. 

The base algorithm where we will perform the transpose simply performing over two for loops a change of coordinates, this will give us the possibility of performing the task easily even in the case of a non symmetric matrix. 

In the block transpose the function we will need to pass a new parameter, block size, and the operation will be divided in small blocks over the whole size of the matrix using 4 nested loops. To perform a fair comparison we will initialize our matrices with random float values and compare the time only of the transpose operation. At the end of each time check we will increase the rows and columns dimension of 1 unit for all the 500 iteration of our for loop.

#Section 2: experimental results
The code will be performed on a specific hardware, my laptop which is an Acer Aspire E5 produced in 2017.

The laptop is equipped with an Intel Core i5-7200U CPU. This CPU has 2 cores and 4 threads, 3MB of cache and 34 GB/s of max bandwidth. Then the PC is equipped with 12GB of RAM, and Windows 11 as OS (even if for convenience the code will run on WSL). To analyze the cache behavior Valgrind with the tool cache-grind will be used, so let's take a look to the plot of the time to perform the transpose with the normal and block technique, changing the dimension of the matrices:

![alt text](https://github.com/Sminnitex/Homework1_GPUComputing/blob/master/figures/Performance_my_laptop.png?raw=true)

In the plot is analyzed also the performance using four different optimization levels for GCC compiler that goes from -O0 (No optimization), to -O3 (aggressive optimization). As we can see the results are pretty interesting, in fact not only the normal transpose is often faster than the block transpose, but analyzing the statistics we can see how in both cases the -O3 optimization is not the best overall. In fact in the block transpose the -O1 optimization is more consistent in the results having a lower standard deviation, and a lower minimum peak, however the median and mean of the time used is slightly higher, this can be easily explained noticing how in the -O3 run there are way more outliers in the results, with spike of time consuming transpose. In the normal transpose the -O2 optimization is easily the fastest, with better statistics in everything except the global minimum value compared to the -O3. Let's try to make a sense of these results taking a look to the cachegrind file, on figure 3.

![alt text](https://github.com/Sminnitex/Homework1_GPUComputing/blob/master/figures/cachegrindSummary.png?raw=true)

We can read the file on figure 3 as it follows: we basically have three different instructions, and these are Instruction executed, Data read and Data writes. For each of the instructions we have three measurements: the total number of time an instruction is called, the cache misses on the first level, and the cache misses on the last level. The two percentage in brackets refer to the contribution of a measure referred to the whole program, or for the given function. So considering the first percentage in the image: the entire code is 100% of the instructions executed, and our block transpose function represents 26% of the calls of our program. In absolute number it has been read over 30 billion times and has a ridiculous amount of cache misses of first and last level. However the number of cache misses for reading and writing operation on Data is relevant, and slow down the whole program. 

Now if we try to take a look to a normal transpose cachegrind output we will notice something important.

![alt text](https://github.com/Sminnitex/Homework1_GPUComputing/blob/master/figures/cachegrinNormSumm.png?raw=true)

As we can easily see comparing the two methods it immediately catches the eyes that the block transpose has been called way more times than the normal transpose. This could depend on the complexity of the function, that in the block case uses more parameters and more instruction, needing four nested loop instead of two. But what it matters is the number of cache misses, and we can see how both function struggles a bit, resulting in a good number of misses on L1 and LL. The percentage of misses on L1 cache is always grater than the misses on LL, this should not be surprising since the L1 cache is smaller then the LL; the access on L1 is constant and the misses very frequent.

The misses are over 90% in the Data read section both for the block and normal transpose, at every optimization flag. This possibly means that my architecture is a bottle neck for the program. However is quite interesting to compare this results with other, obtained by the same hardware but using, instead of the WSL, a Linux system on the bare metal. The performance obtained are consistently better from every point of view, that is related to the better use of memory access: while WSL has to communicate with windows in order to obtain the data from my C program, a Linux OS can manage everything on it's own, and the new graph in this case are the following:

![alt text](https://github.com/Sminnitex/Homework1_GPUComputing/blob/master/figures/Performance_linux.png?raw=true)

The results are still influenced by many outliers, but this time is more clear what levels of optimization perform better then the other, and this time analyzing the stats we can see how the block matrix transpose works way better with the aggressive optimization, while the normale transpose perform better without optimization. Analyzing these cachegrind files we can see how this changes are reflected with a percentage of cache misses in writing data that drops especially for the block transpose. So the communication between OS was the real bottleneck in the application and the memory is managed way better by the Linux OS.

Now let's try to calculate the bandwidth to have more data, we will perform this analysis on the Linux results since they are more consistent and the communication between OS seems to make the code perform worse. As we know the formula to calculate the bandwidth is:


>    effectiveBandwidth = ((Dr + Dw) / 10^9) / t


![alt text](https://github.com/Sminnitex/Homework1_GPUComputing/blob/master/figures/bandwidth.png?raw=true)

So as we can expect seeing the time metrics the bandwidth confirms that this code is more efficient for the normal matrix transpose, and in particular for the one -O0 optimization flag. The block matrix transpose, despite being less efficient has a good growth in performance for higher optimization flags.

