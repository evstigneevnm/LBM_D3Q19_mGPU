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



#include <math.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h> //for timer support
#include <iostream>
#include <mpi.h>

#include "Macro.h"
#include "cuda_support.h"
#include "initial_conditions.h"
#include "IO_operations.h"
#include "single_step.h"
#include "map.h"
#include "collide.h"
#include "read_boost_pt.h"


void copy_microscopic_variables_host_2_device( microscopic_variables MV, microscopic_variables MV_d, int Nx, int Ny, int Nz)
{
    host_2_device_cpy<real>(MV_d.d0, MV.d0, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d1, MV.d1, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d2, MV.d2, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d3, MV.d3, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d4, MV.d4, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d5, MV.d5, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d6, MV.d6, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d7, MV.d7, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d8, MV.d8, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d9, MV.d9, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d10, MV.d10, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d11, MV.d11, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d12, MV.d12, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d13, MV.d13, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d14, MV.d14, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d15, MV.d15, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d16, MV.d16, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d17, MV.d17, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d18, MV.d18, Nx, Ny, Nz);

}



void copy_microscopic_variables_device_2_host( microscopic_variables MV_d, microscopic_variables MV, int Nx, int Ny, int Nz)
{
    device_2_host_cpy<real>(MV.d0, MV_d.d0, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d1, MV_d.d1, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d2, MV_d.d2, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d3, MV_d.d3, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d4, MV_d.d4, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d5, MV_d.d5, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d6, MV_d.d6, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d7, MV_d.d7, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d8, MV_d.d8, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d9, MV_d.d9, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d10, MV_d.d10, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d11, MV_d.d11, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d12, MV_d.d12, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d13, MV_d.d13, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d14, MV_d.d14, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d15, MV_d.d15, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d16, MV_d.d16, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d17, MV_d.d17, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d18, MV_d.d18, Nx, Ny, Nz);

}

void copy_macroscopic_variables_host_2_device( macroscopic_variables NV, macroscopic_variables NV_d, int Nx, int Ny, int Nz)
{
    host_2_device_cpy<real>(NV_d.rho, NV.rho, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.ux, NV.ux, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.uy, NV.uy, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.uz, NV.uz, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.abs_u, NV.abs_u, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.rot_x, NV.rot_x, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.rot_y, NV.rot_y, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.rot_z, NV.rot_z, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.abs_rot, NV.abs_rot, Nx, Ny, Nz);    
    host_2_device_cpy<real>(NV_d.s, NV.s, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.lk, NV.lk, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.H, NV.H, Nx, Ny, Nz);    

}
void copy_macroscopic_variables_device_2_host( macroscopic_variables NV_d, macroscopic_variables NV, int Nx, int Ny, int Nz)
{
    device_2_host_cpy<real>(NV.rho, NV_d.rho, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.ux, NV_d.ux, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.uy, NV_d.uy, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.uz, NV_d.uz, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.abs_u, NV_d.abs_u, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.rot_x, NV_d.rot_x, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.rot_y, NV_d.rot_y, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.rot_z, NV_d.rot_z, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.abs_rot, NV_d.abs_rot, Nx, Ny, Nz);    
    device_2_host_cpy<real>(NV.s, NV_d.s, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.lk, NV_d.lk, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.H, NV_d.H, Nx, Ny, Nz);    

}

void copy_control_variables_host_2_device( control_variables CV, control_variables CV_d, int Nx, int Ny, int Nz)
{
    host_2_device_cpy<int>(CV_d.bc, CV.bc, Nx, Ny, Nz);
    host_2_device_cpy<int>(CV_d.proc, CV.proc, Nx, Ny, Nz);
  

}

void copy_control_variables_device_2_host( control_variables CV_d, control_variables CV, int Nx, int Ny, int Nz)
{
    device_2_host_cpy<int>(CV.bc, CV_d.bc, Nx, Ny, Nz);
    device_2_host_cpy<int>(CV.proc, CV_d.proc, Nx, Ny, Nz);
  

}



int main(int argc, char *argv[])
{
    /* code */
    communication_variables COM;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &COM.myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &COM.totalrank);
    if(argc!=2){
        if(COM.myrank==MASTER)
            std::cerr << "\nUsage: " << argv[0] << " config_file_name.info" << std::endl;
        MPI_Finalize();
        return 0;
    }

    std::string control_file_2_read, control_file_2_write;
    read_info_data(argv[1], &COM, control_file_2_read, control_file_2_write);
    int Nx = COM.Nx;
    int Ny = COM.Ny;
    int Nz = COM.Nz;



    int timesteps=COM.timesteps;
    real Reynolds=COM.Reynolds;
    //
    
    //CPU structures
    struct microscopic_variables MV;
    struct macroscopic_variables NV;
    struct control_variables CV;
    //GPU structures
    struct microscopic_variables MV_d1;
    struct microscopic_variables MV_d2;
    struct macroscopic_variables NV_d;
    struct control_variables CV_d;
    //select device
    if(COM.device == 1)
    {
        COM.device_ID = InitCUDA(COM.PCI_ID);
        if(COM.device_ID==-1)
        {
            printf("\n failed to select GPU device \n");
            MPI_Finalize();
            return 0;
        }
        //real * test = device_allocate<real>(1);
        cudaDeviceReset();
        std::cout << "proc " << COM.myrank << " uses device " << COM.device_ID << "\n";
    }

    //cuda streams
    cudaStreamCreate(&COM.streams[0]);

    //allocate HOST
    host_allocate_all<real>(Nx, Ny, Nz, 19, &MV.d0, &MV.d1, &MV.d2, &MV.d3, &MV.d4, &MV.d5, &MV.d6, &MV.d7,  &MV.d8,  &MV.d9,  &MV.d10, &MV.d11, &MV.d12, &MV.d13, &MV.d14, &MV.d15, &MV.d16, &MV.d17,  &MV.d18);
    host_allocate_all<real>(Nx, Ny, Nz, 12, &NV.rho, &NV.ux, &NV.uy, &NV.uz, &NV.abs_u, &NV.rot_x, &NV.rot_y, &NV.rot_z,  &NV.abs_rot,  &NV.s,  &NV.lk, &NV.H);
    host_allocate_all<int>(Nx, Ny, Nz, 2, &CV.bc, &CV.proc );

    //allocate DEVICE
    device_allocate_all<real>(Nx, Ny, Nz, 19, &MV_d1.d0, &MV_d1.d1, &MV_d1.d2, &MV_d1.d3, &MV_d1.d4, &MV_d1.d5, &MV_d1.d6, &MV_d1.d7,  &MV_d1.d8,  &MV_d1.d9,  &MV_d1.d10, &MV_d1.d11, &MV_d1.d12, &MV_d1.d13, &MV_d1.d14, &MV_d1.d15, &MV_d1.d16, &MV_d1.d17,  &MV_d1.d18);
    device_allocate_all<real>(Nx, Ny, Nz, 19, &MV_d2.d0, &MV_d2.d1, &MV_d2.d2, &MV_d2.d3, &MV_d2.d4, &MV_d2.d5, &MV_d2.d6, &MV_d2.d7,  &MV_d2.d8,  &MV_d2.d9,  &MV_d2.d10, &MV_d2.d11, &MV_d2.d12, &MV_d2.d13, &MV_d2.d14, &MV_d2.d15, &MV_d2.d16, &MV_d2.d17,  &MV_d2.d18);    
    device_allocate_all<real>(Nx, Ny, Nz, 12, &NV_d.rho, &NV_d.ux, &NV_d.uy, &NV_d.uz, &NV_d.abs_u, &NV_d.rot_x, &NV_d.rot_y, &NV_d.rot_z,  &NV_d.abs_rot,  &NV_d.s,  &NV_d.lk, &NV_d.H);
    device_allocate_all<int>(Nx, Ny, Nz, 2, &CV_d.bc, &CV_d.proc );

    //Allocate communication blocks
    allocate_blocks(Nx, Ny, Nz, &COM);

    set_boundaries(Nx, Ny, Nz, CV, COM);
    if(read_control_file(control_file_2_read.c_str(), COM, MV, Nx, Ny, Nz)==0)
        initial_conditions(MV, NV, Nx, Ny, Nz);
    else
        get_macroscopic(MV, NV, Nx, Ny, Nz);


    //copy from HOST to DEVICE
    copy_microscopic_variables_host_2_device( MV, MV_d1, Nx, Ny, Nz);
    copy_microscopic_variables_host_2_device( MV, MV_d2, Nx, Ny, Nz);
    copy_macroscopic_variables_host_2_device( NV, NV_d, Nx, Ny, Nz);
    copy_control_variables_host_2_device(CV, CV_d, Nx, Ny, Nz);
    
    //set device grid
    unsigned int k1, k2 ;
        // step 1: compute # of threads per block
    unsigned int nthreads = BLOCK_DIM_X * BLOCK_DIM_Y ;
        // step 2: compute # of blocks needed
    unsigned int nblocks = ( Nx*Ny*Nz + nthreads -1 )/nthreads ;
        // step 3: find grid's configuration
    real db_nblocks = (real)nblocks ;
    k1 = (unsigned int) floor( sqrt(db_nblocks) ) ;
    k2 = (unsigned int) ceil( db_nblocks/((real)k1)) ;

    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 dimGrid( k2, k1, 1 );

    real delta=0.1;
    
    real tau=real(6.0/3.0/Reynolds+0.5);
    
    printf("tau=%lf\n",(double)tau);
    real omega=1.0/tau;


    struct timeval start, end;
    gettimeofday(&start, NULL);
    for(int t=0;t<timesteps;t++)
    {
        //run_single_step(dimGrid, dimBlock, Nx, Ny, Nz, &COM, MV_d1, MV_d2,  NV_d,  CV_d, omega, delta);
        run_single_step_streams(dimGrid, dimBlock, Nx, Ny, Nz, &COM, MV_d1, MV_d2,  NV_d,  CV_d, omega, delta);

        if((t%1000)==0){
            printf(" [%.03lf%%]    \r",(double)(real(t+2)*100.0/real(timesteps+1)));
            fflush(stdout);
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
    }

    gettimeofday(&end, NULL);
    real etime=((end.tv_sec-start.tv_sec)*1000000u+(end.tv_usec-start.tv_usec))/1.0E6;
    printf("\n\nWall time:%lfsec\n",(double)etime); 


    //copy from DEVICE to HOST
    copy_microscopic_variables_device_2_host( MV_d1, MV, Nx, Ny, Nz);
    copy_macroscopic_variables_device_2_host( NV_d, NV,  Nx,  Ny,  Nz);
    copy_control_variables_device_2_host( CV_d, CV, Nx, Ny, Nz);
    
    //cuda streams
    cudaStreamDestroy(COM.streams[0]);

    //free DEVICE memroy
    device_deallocate_all<real>(19, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7,  MV_d1.d8,  MV_d1.d9,  MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17,  MV_d1.d18);
    device_deallocate_all<real>(19, MV_d2.d0, MV_d2.d1, MV_d2.d2, MV_d2.d3, MV_d2.d4, MV_d2.d5, MV_d2.d6, MV_d2.d7,  MV_d2.d8,  MV_d2.d9,  MV_d2.d10, MV_d2.d11, MV_d2.d12, MV_d2.d13, MV_d2.d14, MV_d2.d15, MV_d2.d16, MV_d2.d17,  MV_d2.d18);    
    device_deallocate_all<real>(12, NV_d.rho, NV_d.ux, NV_d.uy, NV_d.uz, NV_d.abs_u, NV_d.rot_x, NV_d.rot_y, NV_d.rot_z,  NV_d.abs_rot,  NV_d.s,  NV_d.lk, NV_d.H);
    device_deallocate_all<int>(2, CV_d.bc, CV_d.proc );
    //deallocate communication blocks
    deallocate_blocks(Nx, Ny, Nz, &COM);

    //write results to disk
    write_out_file("vectors_out", COM, NV, CV, Nx, Ny, Nz, 'v', COM.dh, COM.L0x, COM.L0y, COM.L0z);
    //write_out_file("density_out", COM, NV, CV, Nx, Ny, Nz, 'p', COM.dh, COM.L0x, COM.L0y, COM.L0z);
    //write_out_file_const("boundary_out", COM, NV, CV, Nx, Ny, Nz, 'b', COM.dh, COM.L0x, COM.L0y, COM.L0z);
    //write_out_file_const("density_out", COM, NV, CV, Nx, Ny, Nz, 'p', COM.dh, COM.L0x, COM.L0y, COM.L0z);

    write_out_pos_file("vectors_out.pos", 2, COM, NV, CV, NV.ux, NV.uy, NV.uz);
    if(control_file_2_write.size()>0)
        write_control_file(control_file_2_write.c_str(), COM, MV, Nx, Ny, Nz);

    //Free CPU meemory
    host_deallocate_all<real>(19, MV.d0, MV.d1, MV.d2, MV.d3, MV.d4, MV.d5, MV.d6, MV.d7,  MV.d8,  MV.d9,  MV.d10, MV.d11, MV.d12, MV.d13, MV.d14, MV.d15, MV.d16, MV.d17, MV.d18);
    host_deallocate_all<real>(12, NV.rho, NV.ux, NV.uy, NV.uz, NV.abs_u, NV.rot_x, NV.rot_y, NV.rot_z,  NV.abs_rot,  NV.s,  NV.lk, NV.H);
    host_deallocate_all<int>(2, CV.bc, CV.proc);
    if(COM.myrank==MASTER)
        printf("\n === done === \n");
    MPI_Finalize();
    return 0;
}