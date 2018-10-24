#pragma once

#include <mpi.h>
#include "Macro.h"
#include "cuda_support.h"
#include "cuda_safe_call.h"
#include "stream_mpi.h"

void allocate_blocks(int Nx, int Ny, int Nz, communication_variables *COM);
void deallocate_blocks(int Nx, int Ny, int Nz, communication_variables *COM);
void exchange_boundaries_MPI(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM,  microscopic_variables MV_d_send, microscopic_variables MV_d_recv);
void exchange_boundaries_MPI_streams(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM, microscopic_variables MV_d_send, microscopic_variables MV_d_recv);

void copy_send_buffers_streams(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM, microscopic_variables MV_d1);
void copy_recv_buffers_streams(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM, microscopic_variables MV_d1);
