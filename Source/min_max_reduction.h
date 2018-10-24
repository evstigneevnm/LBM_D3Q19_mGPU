#ifndef MIN_MAX_REDUCTION_H
#define MIN_MAX_REDUCTION_H

#if __DEVICE_EMULATION__
#define DEBUG_SYNC __syncthreads();
#else
#define DEBUG_SYNC
#endif

#if (__CUDA_ARCH__ < 200)
#define int_mult(x,y)	__mul24(x,y)	
#else
#define int_mult(x,y)	x*y
#endif

#define inf 0x7f800000 
#include "Macro.h"

// for sum!!!
#ifndef BLOCK_SIZE
	#define BLOCK_SIZE 512
#endif


const int blockSize1 = 4096/2; 
/*const int blockSize2 = 8192;
const int blockSize3 = 16384;
const int blockSize4 = 32768;
const int blockSize5 = 65536;*/

const int threads_r = 64;

void compute_reduction(real* d_in, real* d_out, int num_els);

#define BLOCK_SIZE1 512

real reduction_sum(int N, real* InputV, real* OutputV, real* Output);

#endif

