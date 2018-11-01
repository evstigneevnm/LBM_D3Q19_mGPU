/*
* This file is part of the Lattice Boltzmann multiple GPU distribution. 
(https://github.com/evstigneevnm/LBM_D3Q19_mGPU).
* Copyright (c) 2017-2018 Evstigneev Nikolay Mikhaylovitch and Ryabkov Oleg Igorevich.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 2 only.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/


#include "min_max_reduction.h"


__device__ void warp_reduce_max( volatile real smem[64])
{

	smem[threadIdx.x] = smem[threadIdx.x+32] > smem[threadIdx.x] ? 
						smem[threadIdx.x+32] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+16] > smem[threadIdx.x] ? 
						smem[threadIdx.x+16] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+8] > smem[threadIdx.x] ? 
						smem[threadIdx.x+8] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+4] > smem[threadIdx.x] ? 
						smem[threadIdx.x+4] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+2] > smem[threadIdx.x] ? 
						smem[threadIdx.x+2] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+1] > smem[threadIdx.x] ? 
						smem[threadIdx.x+1] : smem[threadIdx.x]; DEBUG_SYNC;

}

__device__ void warp_reduce_min( volatile real smem[64])
{

	smem[threadIdx.x] = smem[threadIdx.x+32] < smem[threadIdx.x] ? 
						smem[threadIdx.x+32] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+16] < smem[threadIdx.x] ? 
						smem[threadIdx.x+16] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+8] < smem[threadIdx.x] ? 
						smem[threadIdx.x+8] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+4] < smem[threadIdx.x] ? 
						smem[threadIdx.x+4] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+2] < smem[threadIdx.x] ? 
						smem[threadIdx.x+2] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+1] < smem[threadIdx.x] ? 
						smem[threadIdx.x+1] : smem[threadIdx.x]; DEBUG_SYNC;

}

template<int threads_r>
__global__ void find_min_max_dynamic(real* in, real* out, int n, int start_adr, int num_blocks)
{

	volatile __shared__  real smem_min[64];
	volatile __shared__  real smem_max[64];
	

	int tid = threadIdx.x + start_adr;

	real max = -inf;
	real min = inf;
	real resval=0.0;


	// tail part
	int mult = 0;
	for(int i = 1; mult + tid < n; i++)
	{
		resval = in[tid + mult];
	
		min = resval < min ? resval : min;
		max = resval > max ? resval : max;

		mult = int_mult(i,threads_r);
	}

	// previously reduced MIN part
	mult = 0;
	int i;
	for(i = 1; mult+threadIdx.x < num_blocks; i++)
	{
		resval = out[threadIdx.x + mult];

		min = resval < min ? resval : min;
		
		mult = int_mult(i,threads_r);
	}

	// MAX part
	for(; mult+threadIdx.x < num_blocks*2; i++)
	{
		resval = out[threadIdx.x + mult];

		max = resval > max ? resval : max;
		
		mult = int_mult(i,threads_r);
	}


	if(threads_r == 32)
	{
		smem_min[threadIdx.x+32] = 0.0f;
		smem_max[threadIdx.x+32] = 0.0f;

	}
	
	smem_min[threadIdx.x] = min;
	smem_max[threadIdx.x] = max;

	__syncthreads();

	if(threadIdx.x < 32)
	{
		warp_reduce_min(smem_min);
		warp_reduce_max(smem_max);
	}
	if(threadIdx.x == 0)
	{
		out[blockIdx.x] = smem_min[threadIdx.x]; // out[0] == ans
		out[blockIdx.x + gridDim.x] = smem_max[threadIdx.x]; 
	}


}

template<int els_per_block, int threads_r>
__global__ void find_min_max(real* in, real* out)
{
	volatile __shared__  real smem_min[64];
	volatile __shared__  real smem_max[64];

	int tid = threadIdx.x + blockIdx.x*els_per_block;

	real max = -inf;
	real min = inf;
	real resval;

	const int iters = els_per_block/threads_r;
	
#pragma unroll
		for(int i = 0; i < iters; i++)
		{

			resval = in[tid + i*threads_r];

			min = resval < min ? resval : min;
			max = resval > max ? resval : max;

		}
	
	
	if(threads_r == 32)
	{
		smem_min[threadIdx.x+32] = 0.0f;
		smem_max[threadIdx.x+32] = 0.0f;
	
	}
	
	smem_min[threadIdx.x] = min;
	smem_max[threadIdx.x] = max;


	__syncthreads();

	if(threadIdx.x < 32)
	{
		warp_reduce_min(smem_min);
		warp_reduce_max(smem_max);
	}
	if(threadIdx.x == 0)
	{
		out[blockIdx.x] = smem_min[threadIdx.x]; // out[0] == ans
		out[blockIdx.x + gridDim.x] = smem_max[threadIdx.x]; 
	}

}

void findBlockSize(int* whichSize, int* num_el)
{

	const real pretty_big_number = 24.0f*1024.0f*1024.0f;

	real ratio = real((*num_el))/pretty_big_number;


	if(ratio > 0.8f)
		(*whichSize) =  5;
	else if(ratio > 0.6f)
		(*whichSize) =  4;
	else if(ratio > 0.4f)
		(*whichSize) =  3;
	else if(ratio > 0.2f)
		(*whichSize) =  2;
	else
		(*whichSize) =  1;


}
void compute_reduction(real* d_in, real* d_out, int num_els)
{

	int whichSize = -1; 
		
	findBlockSize(&whichSize,&num_els);

	//whichSize = 5;

	int block_size = powf(2,whichSize-1)*blockSize1;
	int num_blocks = num_els/block_size;
	int tail = num_els - num_blocks*block_size;
	int start_adr = num_els - tail;

	
	if(whichSize == 1)
		find_min_max<blockSize1,threads_r><<< num_blocks, threads_r>>>(d_in, d_out); 
	else if(whichSize == 2)
		find_min_max<blockSize1*2,threads_r><<< num_blocks, threads_r>>>(d_in, d_out); 
	else if(whichSize == 3)
		find_min_max<blockSize1*4,threads_r><<< num_blocks, threads_r>>>(d_in, d_out); 
	else if(whichSize == 4)
		find_min_max<blockSize1*8,threads_r><<< num_blocks, threads_r>>>(d_in, d_out); 
	else
		find_min_max<blockSize1*16,threads_r><<< num_blocks, threads_r>>>(d_in, d_out); 

	find_min_max_dynamic<threads_r><<< 1, threads_r>>>(d_in, d_out, num_els, start_adr, num_blocks);
	
}
















template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};




unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce_sum(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = mySum = mySum + smem[tid + 32];
        }

        if (blockSize >=  32)
        {
            smem[tid] = mySum = mySum + smem[tid + 16];
        }

        if (blockSize >=  16)
        {
            smem[tid] = mySum = mySum + smem[tid +  8];
        }

        if (blockSize >=   8)
        {
            smem[tid] = mySum = mySum + smem[tid +  4];
        }

        if (blockSize >=   4)
        {
            smem[tid] = mySum = mySum + smem[tid +  2];
        }

        if (blockSize >=   2)
        {
            smem[tid] = mySum = mySum + smem[tid +  1];
        }
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}



#define BLOCK_SIZE1 512

void get_blocks_threads_shmem(int n, int maxBlocks, int maxThreads, int &blocks, int &threads, int &smemSize){

	
	threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	blocks = (n + (threads * 2 - 1)) / (threads * 2);
    smemSize = (threads <= 32) ? 2 * threads * sizeof(real) : threads * sizeof(real);
	blocks = (maxBlocks>blocks) ? blocks : maxBlocks;

}



void wrapper_reduce_sum(int blocks, int threads, int smemSize, real* InputV, real* OutputV, int N){

dim3 dimBlock(threads, 1, 1);
dim3 dimGrid(blocks, 1, 1);
if(isPow2(N)){
	switch (threads){
		case 512:
			reduce_sum<real, 512, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		case 256:
			reduce_sum<real, 256, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		case 128:
			reduce_sum<real, 128, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		case 64:
			reduce_sum<real, 64, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		case 32:
			reduce_sum<real, 32, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		case 16:
			reduce_sum<real, 16, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		case  8:
			reduce_sum<real, 8, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		case  4:
			reduce_sum<real, 4, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		case  2:
			reduce_sum<real, 2, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		case  1:
			reduce_sum<real, 1, true><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		}		
}
else{
	switch (threads){
		case 512:
			reduce_sum<real, 512, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		case 256:
			reduce_sum<real, 256, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		case 128:
			reduce_sum<real, 128, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		case 64:
			reduce_sum<real, 64, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		case 32:
			reduce_sum<real, 32, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		case 16:
			reduce_sum<real, 16, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		case  8:
			reduce_sum<real, 8, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		case  4:
			reduce_sum<real, 4, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		case  2:
			reduce_sum<real, 2, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		case  1:
			reduce_sum<real, 1, false><<< dimGrid, dimBlock, smemSize >>>(InputV, OutputV, N); break;
		}		
}
   	



}



real reduction_sum(int N, real* InputV, real* OutputV, real* Output){
	real gpu_result=0.0;
	int threads = 0, blocks = 0, smemSize=0;
	int maxThreads=BLOCK_SIZE1;
	int maxBlocks=128;//DEBUG
	get_blocks_threads_shmem(N, maxBlocks, maxThreads, blocks, threads, smemSize);

	//perform reduction
	//printf("threads=%i, blocks=%i, shmem size=%i\n",threads, blocks, smemSize);
 	wrapper_reduce_sum(blocks, threads, smemSize, InputV, OutputV, N);
	bool needReadBack=true;
	int s=blocks;
	while (s > 1){
		get_blocks_threads_shmem(s, maxBlocks, maxThreads, blocks, threads, smemSize);
		//printf("threads=%i, blocks=%i, shmem size=%i\n",threads, blocks, smemSize);
		wrapper_reduce_sum(blocks, threads, smemSize, OutputV, OutputV, s);
		s = (s + (threads*2-1)) / (threads*2);
	}
	if (s > 1){
		cudaMemcpy(Output, OutputV, s * sizeof(real), cudaMemcpyDeviceToHost);
		for (int i=0; i < s; i++){
			gpu_result += Output[i];
		}
		needReadBack = false;
	}
	if (needReadBack){
        cudaMemcpy(&gpu_result, OutputV, sizeof(real), cudaMemcpyDeviceToHost);
    }
 	return gpu_result;	

}
/*unsigned long long int my_min_max_test(int num_els)
{

	// timers

	unsigned long long int start;
	unsigned long long int delta;


	int testIterations = 100;

	int size = num_els*sizeof(real);

	real* d_in;
	real* d_out;

	real* d_warm1;
	real* d_warm2;


	real* in = (real*)malloc(size);
	real* out = (real*)malloc(size);

	cudaMalloc((void**)&d_in, size);
	cudaMalloc((void**)&d_out, size);

	cudaMalloc((void**)&d_warm1, 1024*sizeof(real));
	cudaMalloc((void**)&d_warm2, 1024*sizeof(real));

	//for(int i = 0; i < testIterations; i++){
	for(int i = 0; i < num_els; i++)
	{
		in[i] = real(i);//rand()&1;
	}

	//in[1024] = 34.0f;
	//in[333] = -1.0E-25;
	//in[23523] = -42.0f;




	cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

	
	//////////
	/// warmup
	//////////
	find_min_max<32,threads><<< 32, 32>>>(d_warm1, d_warm2); 
	cudaThreadSynchronize();	
	/////
	// end warmup
	/////

	//time it
	start = get_clock();

	////////////// 
	// real reduce
	/////////////

	for(int i = 0; i < testIterations; i++)
		compute_reduction(d_in, d_out, num_els);


	cudaThreadSynchronize();
	delta = get_clock() - start;
	
	real dt = real(delta)/real(testIterations);

	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost); // need not be SIZE! (just 2 elements)

	
	real throughput = num_els*sizeof(real)*0.001f/(dt);
	int tail = num_els - (num_els/blockSize1)*blockSize1;	
	printf(" %7.0d \t %0.2f \t\t %0.2f % \t %0.1f \t\t %s \n \t \t \t \t min:%f max:%f\n", num_els, throughput,
		(throughput/70.6f)*100.0f,dt,  (cpu_min(in,num_els) == out[0] && cpu_max(in,num_els) == out[1]) ? "Pass" : "Fail", out[0], out[1]);
	
	//}
	//printf("\n  min: %0.3f \n", out[0]);
	//printf("\n  max: %0.3f \n", out[1]);

	cudaFree(d_in);
	cudaFree(d_out);

	cudaFree(d_warm1);
	cudaFree(d_warm2);

	free(in);
	free(out);

	//system("pause");

	return delta;

}
*/