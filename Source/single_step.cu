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


#include "single_step.h"


void run_single_step(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM, microscopic_variables MV_d1, microscopic_variables MV_d2, macroscopic_variables NV_d, control_variables CV_d, real omega, real delta)
{

       
        kernel_collide_0_18<<< dimGrid, dimBlock>>>( delta, NV_d.H, 0.0, 0.0, 0.0, Nx, Ny, Nz, omega, NV_d.ux, NV_d.uy, NV_d.uz, NV_d.rho, CV_d.bc, MV_d1, MV_d2);

        exchange_boundaries_MPI(dimGrid, dimBlock, Nx, Ny, Nz, COM, MV_d2, MV_d1);
        kernel_stream3D_0_18_forward<<< dimGrid, dimBlock>>>(Nx, Ny, Nz, CV_d.bc, MV_d2, MV_d1);

}

void run_single_step_streams(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM, microscopic_variables MV_d1, microscopic_variables MV_d2, macroscopic_variables NV_d, control_variables CV_d, real omega, real delta)
{

    MPI_Request request_send1, request_send2, request_recv1, request_recv2;
    int tag=0;
    

    kernel_collide_0_18<<< dimGrid, dimBlock>>>( delta, NV_d.H, 0.0, 0.0, 0.0, Nx, Ny, Nz, omega, NV_d.ux, NV_d.uy, NV_d.uz, NV_d.rho, CV_d.bc, MV_d1, MV_d2);


    copy_send_buffers_streams(dimGrid, dimBlock, Nx, Ny, Nz, COM, MV_d2);

    if(COM->Face1Bufer_size>0)
    {
        MPI_Isend(COM->Face1BuferSend_device, COM->Face1Bufer_size, MPI_real, COM->Face1proc, tag, MPI_COMM_WORLD, &request_send1);        
    }   
    if(COM->Face2Bufer_size>0)
    {
        MPI_Isend(COM->Face2BuferSend_device, COM->Face2Bufer_size, MPI_real, COM->Face2proc, tag, MPI_COMM_WORLD, &request_send2);        
    }

    kernel_stream3D_0_18_forward<<< dimGrid, dimBlock, 0, COM->streams[0]>>>(Nx, Ny, Nz, CV_d.bc, MV_d2, MV_d1);
    

    if(COM->Face1Bufer_size>0)
    {
        MPI_Irecv(COM->Face1BuferRecv_device, COM->Face1Bufer_size, MPI_real, COM->Face1proc, tag, MPI_COMM_WORLD, &request_recv1);
        MPI_Wait(&request_recv1, MPI_STATUS_IGNORE);
    }   
    if(COM->Face2Bufer_size>0)
    {
        MPI_Irecv(COM->Face2BuferRecv_device , COM->Face2Bufer_size, MPI_real, COM->Face2proc, tag, MPI_COMM_WORLD, &request_recv2);
        MPI_Wait(&request_recv2, MPI_STATUS_IGNORE);
    }
    copy_recv_buffers_streams(dimGrid, dimBlock, Nx, Ny, Nz, COM, MV_d1);
    if(COM->Face1Bufer_size>0)
    {
        MPI_Wait(&request_send1, MPI_STATUS_IGNORE);
    }
    if(COM->Face2Bufer_size>0)
    {
        MPI_Wait(&request_send2, MPI_STATUS_IGNORE);
    }    
    CUDA_SAFE_CALL(cudaStreamSynchronize( COM->streams[0]));


}