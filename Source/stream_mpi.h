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


#pragma once

#include "Macro.h"

__global__ void kernel_copy_FaceX_Send_stream_positive_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
__global__ void kernel_copy_FaceX_Send_stream_negative_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
__global__ void kernel_copy_FaceX_Recv_stream_positive_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);
__global__ void kernel_copy_FaceX_Recv_stream_negative_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);


// __global__ void kernel_copy_FaceX_Send_stream_positive(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
// __global__ void kernel_copy_FaceX_Send_stream_negative(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
// __global__ void kernel_copy_FaceX_Recv_stream_positive(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);
// __global__ void kernel_copy_FaceX_Recv_stream_negative(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);


__global__ void kernel_copy_FaceY_Send_stream_positive_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
__global__ void kernel_copy_FaceY_Send_stream_negative_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
__global__ void kernel_copy_FaceY_Recv_stream_positive_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);
__global__ void kernel_copy_FaceY_Recv_stream_negative_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);


// __global__ void kernel_copy_FaceY_Send_stream_positive(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
// __global__ void kernel_copy_FaceY_Send_stream_negative(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
// __global__ void kernel_copy_FaceY_Recv_stream_positive(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);
// __global__ void kernel_copy_FaceY_Recv_stream_negative(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);



__global__ void kernel_copy_FaceZ_Send_stream_positive_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
__global__ void kernel_copy_FaceZ_Send_stream_negative_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
__global__ void kernel_copy_FaceZ_Recv_stream_positive_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);
__global__ void kernel_copy_FaceZ_Recv_stream_negative_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);

// __global__ void kernel_copy_FaceZ_Send_stream_positive(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
// __global__ void kernel_copy_FaceZ_Send_stream_negative(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
// __global__ void kernel_copy_FaceZ_Recv_stream_positive(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);
// __global__ void kernel_copy_FaceZ_Recv_stream_negative(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);