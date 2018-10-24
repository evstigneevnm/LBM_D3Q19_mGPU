#pragma once

#include "Macro.h"

__global__ void kernel_copy_FaceX_Send_stream_positive_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
__global__ void kernel_copy_FaceX_Send_stream_negative_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
__global__ void kernel_copy_FaceX_Recv_stream_positive_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);
__global__ void kernel_copy_FaceX_Recv_stream_negative_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);


__global__ void kernel_copy_FaceX_Send_stream_positive(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
__global__ void kernel_copy_FaceX_Send_stream_negative(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
__global__ void kernel_copy_FaceX_Recv_stream_positive(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);
__global__ void kernel_copy_FaceX_Recv_stream_negative(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);


__global__ void kernel_copy_FaceY_Send_stream_positive(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
__global__ void kernel_copy_FaceY_Send_stream_negative(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
__global__ void kernel_copy_FaceY_Recv_stream_positive(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);
__global__ void kernel_copy_FaceY_Recv_stream_negative(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);



__global__ void kernel_copy_FaceZ_Send_stream_positive(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
__global__ void kernel_copy_FaceZ_Send_stream_negative(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send);
__global__ void kernel_copy_FaceZ_Recv_stream_positive(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);
__global__ void kernel_copy_FaceZ_Recv_stream_negative(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv);