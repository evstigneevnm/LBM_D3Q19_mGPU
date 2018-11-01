#include "single_step.h"


void run_single_step(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM, microscopic_variables MV_d1, microscopic_variables MV_d2, macroscopic_variables NV_d, control_variables CV_d, real omega, real delta)
{

        //kernel_macro_0_18<<< dimGrid, dimBlock>>>(Nx, Ny, Nz, NV_d.ux, NV_d.uy, NV_d.uz, NV_d.rho, CV_d.bc, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);

        kernel_collide_0_18<<< dimGrid, dimBlock>>>( delta, NV_d.H, 0.0, 0.0, 0.0, Nx, Ny, Nz, omega, NV_d.ux, NV_d.uy, NV_d.uz, NV_d.rho, CV_d.bc, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18, MV_d2.d0, MV_d2.d1, MV_d2.d2, MV_d2.d3, MV_d2.d4, MV_d2.d5, MV_d2.d6, MV_d2.d7, MV_d2.d8, MV_d2.d9, MV_d2.d10, MV_d2.d11, MV_d2.d12, MV_d2.d13, MV_d2.d14, MV_d2.d15, MV_d2.d16, MV_d2.d17, MV_d2.d18);

//        kernel_wall3D_0_18<<< dimGrid, dimBlock>>>(Nx, Ny, Nz, CV_d.bc, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18, MV_d2.d0, MV_d2.d1, MV_d2.d2, MV_d2.d3, MV_d2.d4, MV_d2.d5, MV_d2.d6, MV_d2.d7, MV_d2.d8, MV_d2.d9, MV_d2.d10, MV_d2.d11, MV_d2.d12, MV_d2.d13, MV_d2.d14, MV_d2.d15, MV_d2.d16, MV_d2.d17, MV_d2.d18);

        

        exchange_boundaries_MPI(dimGrid, dimBlock, Nx, Ny, Nz, COM, MV_d2);

        kernel_stream3D_0_18_forward<<< dimGrid, dimBlock>>>(Nx, Ny, Nz, CV_d.bc,  MV_d2.d0, MV_d2.d1, MV_d2.d2, MV_d2.d3, MV_d2.d4, MV_d2.d5, MV_d2.d6, MV_d2.d7, MV_d2.d8, MV_d2.d9, MV_d2.d10, MV_d2.d11, MV_d2.d12, MV_d2.d13, MV_d2.d14, MV_d2.d15, MV_d2.d16, MV_d2.d17, MV_d2.d18, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);
        

}

void run_single_step_streams(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM, microscopic_variables MV_d1, microscopic_variables MV_d2, macroscopic_variables NV_d, control_variables CV_d, real omega, real delta)
{


        kernel_collide_0_18<<< dimGrid, dimBlock, 0, COM->streams[0]>>>( delta, NV_d.H, 0.0, 0.0, 0.0, Nx, Ny, Nz, omega, NV_d.ux, NV_d.uy, NV_d.uz, NV_d.rho, CV_d.bc, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18, MV_d2.d0, MV_d2.d1, MV_d2.d2, MV_d2.d3, MV_d2.d4, MV_d2.d5, MV_d2.d6, MV_d2.d7, MV_d2.d8, MV_d2.d9, MV_d2.d10, MV_d2.d11, MV_d2.d12, MV_d2.d13, MV_d2.d14, MV_d2.d15, MV_d2.d16, MV_d2.d17, MV_d2.d18);

        kernel_stream3D_0_18_forward<<< dimGrid, dimBlock, 0, COM->streams[1]>>>(Nx, Ny, Nz, CV_d.bc,  MV_d2.d0, MV_d2.d1, MV_d2.d2, MV_d2.d3, MV_d2.d4, MV_d2.d5, MV_d2.d6, MV_d2.d7, MV_d2.d8, MV_d2.d9, MV_d2.d10, MV_d2.d11, MV_d2.d12, MV_d2.d13, MV_d2.d14, MV_d2.d15, MV_d2.d16, MV_d2.d17, MV_d2.d18, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);
        
        exchange_boundaries_MPI_streams(dimGrid, dimBlock, Nx, Ny, Nz, COM, MV_d2);
        CUDA_SAFE_CALL(cudaStreamSynchronize( COM->streams[0]));


}