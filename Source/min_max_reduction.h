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

