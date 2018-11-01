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

#ifndef __CUDA_SAFE_CALL_H__
#define __CUDA_SAFE_CALL_H__

#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

#define __STR_HELPER(x) #x
#define __STR(x) __STR_HELPER(x)

#define CUDA_SAFE_CALL(X)																						\
	do {                        																					\
		cudaError_t cuda_res = (X);                                                                                            									\
		if (cuda_res != cudaSuccess) throw std::runtime_error(std::string("CUDA_SAFE_CALL " __FILE__ " " __STR(__LINE__) " : " #X " failed: ") + std::string(cudaGetErrorString(cuda_res)));	\
	} while (0)

#endif
