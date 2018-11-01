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



#include "stream_mpi.h"

//  =============== X =============

__global__ void kernel_copy_FaceX_Send_stream_negative_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;   
    int j=bx*(blockDim.x)+tx;
    int k=by*(blockDim.y)+ty;

    if((j<Ny)&&(k<Nz))
    {
        
        // const int cx[Q]={0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0};
        // const int cy[Q]={0, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1};
        // const int cz[Q]={0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1, -1,  1};    
//                                 2                    8       10     12      14

        block[I3_MPI(0,j,k,Ny,Nz) ] = MV_d_send.d2[I3(location,j,k)];
        block[I3_MPI(1,j,k,Ny,Nz) ] = MV_d_send.d8[I3(location,j,k)];
        block[I3_MPI(2,j,k,Ny,Nz) ] = MV_d_send.d10[I3(location,j,k)];
        block[I3_MPI(3,j,k,Ny,Nz) ] = MV_d_send.d12[I3(location,j,k)];
        block[I3_MPI(4,j,k,Ny,Nz) ] = MV_d_send.d14[I3(location,j,k)];

    }
}


__global__ void kernel_copy_FaceX_Send_stream_negative(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send)
{
    unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
    // step 1: compute gridIndex in 1-D and 1-D data index "index_in"
    unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
    gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
    index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
    // step 2: extract 3-D data index via
    // index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
    // where xIndex = i-1, yIndex = j-1, zIndex = k-1
    if ( index_in < sizeOfData )
    {
        t1 =  index_in/ Nz; 
        zIndex = index_in - Nz*t1 ;
        xIndex =  t1/ Ny; 
        yIndex = t1 - Ny * xIndex ;
        
        // const int cx[Q]={0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0};
        // const int cy[Q]={0, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1};
        // const int cz[Q]={0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1, -1,  1};    
//                                 2                    8       10     12      14

        unsigned int i=xIndex, j=yIndex, k=zIndex;
        block[I3_MPI(0,j,k,Ny,Nz) ] = MV_d_send.d2[I3(location,j,k)];
        block[I3_MPI(1,j,k,Ny,Nz) ] = MV_d_send.d8[I3(location,j,k)];
        block[I3_MPI(2,j,k,Ny,Nz) ] = MV_d_send.d10[I3(location,j,k)];
        block[I3_MPI(3,j,k,Ny,Nz) ] = MV_d_send.d12[I3(location,j,k)];
        block[I3_MPI(4,j,k,Ny,Nz) ] = MV_d_send.d14[I3(location,j,k)];

    }
}

__global__ void kernel_copy_FaceX_Send_stream_positive_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;   
    int j=bx*(blockDim.x)+tx;
    int k=by*(blockDim.y)+ty;

    if((j<Ny)&&(k<Nz))
    {

        
        // const int cx[Q]={0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0};
        // const int cy[Q]={0, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1};
        // const int cz[Q]={0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1, -1,  1};    
//                             1                    7       9      11      13           

        block[I3_MPI(0,j,k,Ny,Nz) ] = MV_d_send.d1[I3(location,j,k)];
        block[I3_MPI(1,j,k,Ny,Nz) ] = MV_d_send.d7[I3(location,j,k)];
        block[I3_MPI(2,j,k,Ny,Nz) ] = MV_d_send.d9[I3(location,j,k)];
        block[I3_MPI(3,j,k,Ny,Nz) ] = MV_d_send.d11[I3(location,j,k)];
        block[I3_MPI(4,j,k,Ny,Nz) ] = MV_d_send.d13[I3(location,j,k)];

    }
}

__global__ void kernel_copy_FaceX_Send_stream_positive(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send)
{
    unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
    // step 1: compute gridIndex in 1-D and 1-D data index "index_in"
    unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
    gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
    index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
    // step 2: extract 3-D data index via
    // index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
    // where xIndex = i-1, yIndex = j-1, zIndex = k-1
    if ( index_in < sizeOfData )
    {
        t1 =  index_in/ Nz; 
        zIndex = index_in - Nz*t1 ;
        xIndex =  t1/ Ny; 
        yIndex = t1 - Ny * xIndex ;
        
        // const int cx[Q]={0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0};
        // const int cy[Q]={0, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1};
        // const int cz[Q]={0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1, -1,  1};    
//                             1                    7       9      11      13           

        unsigned int i=xIndex, j=yIndex, k=zIndex;
        block[I3_MPI(0,j,k,Ny,Nz) ] = MV_d_send.d1[I3(location,j,k)];
        block[I3_MPI(1,j,k,Ny,Nz) ] = MV_d_send.d7[I3(location,j,k)];
        block[I3_MPI(2,j,k,Ny,Nz) ] = MV_d_send.d9[I3(location,j,k)];
        block[I3_MPI(3,j,k,Ny,Nz) ] = MV_d_send.d11[I3(location,j,k)];
        block[I3_MPI(4,j,k,Ny,Nz) ] = MV_d_send.d13[I3(location,j,k)];

    }
}


__global__ void kernel_copy_FaceX_Recv_stream_negative_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv)
{

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;   
    int j=bx*(blockDim.x)+tx;
    int k=by*(blockDim.y)+ty;

    if((j<Ny)&&(k<Nz))
    {

            
        const int cx[Q]={0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0};
        const int cy[Q]={0, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1};
        const int cz[Q]={0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1, -1,  1};    
//                          1                    7       9      11      13     

        int _j=j+cy[1];
        int _k=k+cz[1];
        int kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {
            MV_d_recv.d1[kk1]=              block[I3_MPI(0,j,k,Ny,Nz)]; 
        }
        _k=k+cz[7];
        _j=j+cy[7];
        kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {       
            MV_d_recv.d7[kk1]=             block[I3_MPI(1,j,k,Ny,Nz)]; 
        }
        _k=k+cz[9];
        _j=j+cy[9];
        kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {             
            MV_d_recv.d9[kk1]=             block[I3_MPI(2,j,k,Ny,Nz)]; 
        }
        _k=k+cz[11];
        _j=j+cy[11];
        kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {          
            MV_d_recv.d11[kk1]=             block[I3_MPI(3,j,k,Ny,Nz)]; 
        }

        _k=k+cz[13];
        _j=j+cy[13];
        kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {          
            MV_d_recv.d13[kk1]=             block[I3_MPI(4,j,k,Ny,Nz)]; 
        }


    }
}


__global__ void kernel_copy_FaceX_Recv_stream_negative(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv)
{
    unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
    // step 1: compute gridIndex in 1-D and 1-D data index "index_in"
    unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
    gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
    index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
    // step 2: extract 3-D data index via
    // index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
    // where xIndex = i-1, yIndex = j-1, zIndex = k-1
    if ( index_in < sizeOfData )
    {
        t1 =  index_in/ Nz; 
        zIndex = index_in - Nz*t1 ;
        xIndex =  t1/ Ny; 
        yIndex = t1 - Ny * xIndex ;
            
        const int cx[Q]={0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0};
        const int cy[Q]={0, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1};
        const int cz[Q]={0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1, -1,  1};    
//                          1                    7       9      11      13     

        unsigned int i=xIndex, j=yIndex, k=zIndex;
        int _j=j+cy[1];
        int _k=k+cz[1];
        int kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {
            MV_d_recv.d1[kk1]=              block[I3_MPI(0,j,k,Ny,Nz)]; 
        }
        _k=k+cz[7];
        _j=j+cy[7];
        kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {       
            MV_d_recv.d7[kk1]=             block[I3_MPI(1,j,k,Ny,Nz)]; 
        }
        _k=k+cz[9];
        _j=j+cy[9];
        kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {             
            MV_d_recv.d9[kk1]=             block[I3_MPI(2,j,k,Ny,Nz)]; 
        }
        _k=k+cz[11];
        _j=j+cy[11];
        kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {          
            MV_d_recv.d11[kk1]=             block[I3_MPI(3,j,k,Ny,Nz)]; 
        }

        _k=k+cz[13];
        _j=j+cy[13];
        kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {          
            MV_d_recv.d13[kk1]=             block[I3_MPI(4,j,k,Ny,Nz)]; 
        }


    }
}



__global__ void kernel_copy_FaceX_Recv_stream_positive_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;   
    int j=bx*(blockDim.x)+tx;
    int k=by*(blockDim.y)+ty;

    if((j<Ny)&&(k<Nz))
    {
            
        const int cx[Q]={0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0};
        const int cy[Q]={0, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1};
        const int cz[Q]={0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1, -1,  1};    
//                              2                    8       10     12      14   

        int _k=k+cz[2];
        int _j=j+cy[2];
        int kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {
            MV_d_recv.d2[kk1]=              block[I3_MPI(0,j,k,Ny,Nz)]; 
        }
        _k=k+cz[8];
        _j=j+cy[8];
        kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {       
            MV_d_recv.d8[kk1]=             block[I3_MPI(1,j,k,Ny,Nz)];             
        }

        _k=k+cz[10];
        _j=j+cy[10];
        kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {             
            MV_d_recv.d10[kk1]=             block[I3_MPI(2,j,k,Ny,Nz)]; 
        }
        _k=k+cz[12];
        _j=j+cy[12];
        kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {          
            MV_d_recv.d12[kk1]=             block[I3_MPI(3,j,k,Ny,Nz)];             
        }

        _k=k+cz[14];
        _j=j+cy[14];
        kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {          
            MV_d_recv.d14[kk1]=             block[I3_MPI(4,j,k,Ny,Nz)];             
        }
    }
}




__global__ void kernel_copy_FaceX_Recv_stream_positive(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv)
{
    unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
    // step 1: compute gridIndex in 1-D and 1-D data index "index_in"
    unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
    gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
    index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
    // step 2: extract 3-D data index via
    // index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
    // where xIndex = i-1, yIndex = j-1, zIndex = k-1
    if ( index_in < sizeOfData )
    {
        t1 =  index_in/ Nz; 
        zIndex = index_in - Nz*t1 ;
        xIndex =  t1/ Ny; 
        yIndex = t1 - Ny * xIndex ;
            
        const int cx[Q]={0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0};
        const int cy[Q]={0, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1};
        const int cz[Q]={0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1, -1,  1};    
//                              2                    8       10     12      14   

        unsigned int i=xIndex, j=yIndex, k=zIndex;
        int _k=k+cz[2];
        int _j=j+cy[2];
        int kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {
            MV_d_recv.d2[kk1]=              block[I3_MPI(0,j,k,Ny,Nz)]; 
        }
        _k=k+cz[8];
        _j=j+cy[8];
        kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {       
            MV_d_recv.d8[kk1]=             block[I3_MPI(1,j,k,Ny,Nz)];             
        }

        _k=k+cz[10];
        _j=j+cy[10];
        kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {             
            MV_d_recv.d10[kk1]=             block[I3_MPI(2,j,k,Ny,Nz)]; 
        }
        _k=k+cz[12];
        _j=j+cy[12];
        kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {          
            MV_d_recv.d12[kk1]=             block[I3_MPI(3,j,k,Ny,Nz)];             
        }

        _k=k+cz[14];
        _j=j+cy[14];
        kk1=I3(location,_j,_k);
        if((_k<0)||(_k>Nz-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {          
            MV_d_recv.d14[kk1]=             block[I3_MPI(4,j,k,Ny,Nz)];             
        }
    }
}



//  =============== Y =============
__global__ void kernel_copy_FaceY_Send_stream_negative(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send)
{
    unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
    // step 1: compute gridIndex in 1-D and 1-D data index "index_in"
    unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
    gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
    index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
    // step 2: extract 3-D data index via
    // index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
    // where xIndex = i-1, yIndex = j-1, zIndex = k-1
    if ( index_in < sizeOfData )
    {
        t1 =  index_in/ Nz; 
        zIndex = index_in - Nz*t1 ;
        xIndex =  t1/ Ny; 
        yIndex = t1 - Ny * xIndex ;
        
        // const int cx[Q]={0, 1, -1, 0, 0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0};
        // const int cy[Q]={0, 0, 0, 1, -1, 0,  0, 1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1};
        // const int cz[Q]={0, 0, 0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1, -1,  1};    
//                                       4             8   9                         16      18

        unsigned int i=xIndex, j=yIndex, k=zIndex;
        block[I3_MPI(0,i,k,Nx,Nz) ] = MV_d_send.d4[I3(i,location,k)];
        block[I3_MPI(1,i,k,Nx,Nz) ] = MV_d_send.d8[I3(i,location,k)];
        block[I3_MPI(2,i,k,Nx,Nz) ] = MV_d_send.d9[I3(i,location,k)];
        block[I3_MPI(3,i,k,Nx,Nz) ] = MV_d_send.d16[I3(i,location,k)];
        block[I3_MPI(4,i,k,Nx,Nz) ] = MV_d_send.d18[I3(i,location,k)];

    }
}

__global__ void kernel_copy_FaceY_Send_stream_positive(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send)
{
    unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
    // step 1: compute gridIndex in 1-D and 1-D data index "index_in"
    unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
    gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
    index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
    // step 2: extract 3-D data index via
    // index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
    // where xIndex = i-1, yIndex = j-1, zIndex = k-1
    if ( index_in < sizeOfData )
    {
        t1 =  index_in/ Nz; 
        zIndex = index_in - Nz*t1 ;
        xIndex =  t1/ Ny; 
        yIndex = t1 - Ny * xIndex ;
        
        // const int cx[Q]={0, 1, -1, 0, 0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0};
        // const int cy[Q]={0, 0, 0, 1, -1, 0,  0, 1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1};
        // const int cz[Q]={0, 0, 0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1, -1,  1};    
//                                   3             7           10                15      17 

        unsigned int i=xIndex, j=yIndex, k=zIndex;
        block[I3_MPI(0,i,k,Nx,Nz) ] = MV_d_send.d3[I3(i,location,k)];
        block[I3_MPI(1,i,k,Nx,Nz) ] = MV_d_send.d7[I3(i,location,k)];
        block[I3_MPI(2,i,k,Nx,Nz) ] = MV_d_send.d10[I3(i,location,k)];
        block[I3_MPI(3,i,k,Nx,Nz) ] = MV_d_send.d15[I3(i,location,k)];
        block[I3_MPI(4,i,k,Nx,Nz) ] = MV_d_send.d17[I3(i,location,k)];

    }
}





__global__ void kernel_copy_FaceY_Recv_stream_negative(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv)
{
    unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
    // step 1: compute gridIndex in 1-D and 1-D data index "index_in"
    unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
    gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
    index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
    // step 2: extract 3-D data index via
    // index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
    // where xIndex = i-1, yIndex = j-1, zIndex = k-1
    if ( index_in < sizeOfData )
    {
        t1 =  index_in/ Nz; 
        zIndex = index_in - Nz*t1 ;
        xIndex =  t1/ Ny; 
        yIndex = t1 - Ny * xIndex ;
            
        const int cx[Q]={0, 1, -1, 0, 0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0};
        const int cy[Q]={0, 0, 0, 1, -1, 0,  0, 1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1};
        const int cz[Q]={0, 0, 0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1, -1,  1};    
//                                3             7           10                15      17        
        unsigned int i=xIndex, j=yIndex, k=zIndex;
        int _i=i+cx[3];
        int _k=k+cz[3];
        int kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {
            MV_d_recv.d3[kk1]=              block[I3_MPI(0,i,k,Nx,Nz)]; 
        }
        _i=i+cx[7];
        _k=k+cz[7];
        kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {       
            MV_d_recv.d7[kk1]=             block[I3_MPI(1,i,k,Nx,Nz)]; 
        }
        _i=i+cx[10];
        _k=k+cz[10];
        kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {             
            MV_d_recv.d10[kk1]=             block[I3_MPI(2,i,k,Nx,Nz)]; 
        }
        _i=i+cx[15];
        _k=k+cz[15];
        kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {          
            MV_d_recv.d15[kk1]=             block[I3_MPI(3,i,k,Nx,Nz)]; 
        }

        _i=i+cx[17];
        _k=k+cz[17];
        kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {          
            MV_d_recv.d17[kk1]=             block[I3_MPI(4,i,k,Nx,Nz)]; 
        }


    }
}




__global__ void kernel_copy_FaceY_Recv_stream_positive(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv)
{
    unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
    // step 1: compute gridIndex in 1-D and 1-D data index "index_in"
    unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
    gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
    index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
    // step 2: extract 3-D data index via
    // index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
    // where xIndex = i-1, yIndex = j-1, zIndex = k-1
    if ( index_in < sizeOfData )
    {
        t1 =  index_in/ Nz; 
        zIndex = index_in - Nz*t1 ;
        xIndex =  t1/ Ny; 
        yIndex = t1 - Ny * xIndex ;
            
        const int cx[Q]={0, 1, -1, 0, 0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0};
        const int cy[Q]={0, 0, 0, 1, -1, 0,  0, 1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1};
        const int cz[Q]={0, 0, 0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1, -1,  1};    
//                                    4             8   9                         16      18    

        unsigned int i=xIndex, j=yIndex, k=zIndex;
        int _i=i+cx[4];
        int _k=k+cz[4];
        int kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {
            MV_d_recv.d4[kk1]=              block[I3_MPI(0,i,k,Nx,Nz)]; 
        }
        _i=i+cx[8];
        _k=k+cz[8];
        kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {       
            MV_d_recv.d8[kk1]=             block[I3_MPI(1,i,k,Nx,Nz)];             
        }

        _i=i+cx[9];
        _k=k+cz[9];
        kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {             
            MV_d_recv.d9[kk1]=             block[I3_MPI(2,i,k,Nx,Nz)]; 
        }
        _i=i+cx[16];
        _k=k+cz[16];
        kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {          
            MV_d_recv.d16[kk1]=             block[I3_MPI(3,i,k,Nx,Nz)];             
        }

        _i=i+cx[18];
        _k=k+cz[18];
        kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {          
            MV_d_recv.d18[kk1]=             block[I3_MPI(4,i,k,Nx,Nz)];             
        }
    }
}




__global__ void kernel_copy_FaceY_Send_stream_negative_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send)
{

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;   
    int i=bx*(blockDim.x)+tx;
    int k=by*(blockDim.y)+ty;

    if((i<Nx)&&(k<Nz)) 
    {
        
        // const int cx[Q]={0, 1, -1, 0, 0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0};
        // const int cy[Q]={0, 0, 0, 1, -1, 0,  0, 1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1};
        // const int cz[Q]={0, 0, 0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1, -1,  1};    
//                                       4             8   9                         16      18

        block[I3_MPI(0,i,k,Nx,Nz) ] = MV_d_send.d4[I3(i,location,k)];
        block[I3_MPI(1,i,k,Nx,Nz) ] = MV_d_send.d8[I3(i,location,k)];
        block[I3_MPI(2,i,k,Nx,Nz) ] = MV_d_send.d9[I3(i,location,k)];
        block[I3_MPI(3,i,k,Nx,Nz) ] = MV_d_send.d16[I3(i,location,k)];
        block[I3_MPI(4,i,k,Nx,Nz) ] = MV_d_send.d18[I3(i,location,k)];

    }
}

__global__ void kernel_copy_FaceY_Send_stream_positive_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send)
{

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;   
    int i=bx*(blockDim.x)+tx;
    int k=by*(blockDim.y)+ty;

    if((i<Nx)&&(k<Nz)) 
    {

        
        // const int cx[Q]={0, 1, -1, 0, 0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0};
        // const int cy[Q]={0, 0, 0, 1, -1, 0,  0, 1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1};
        // const int cz[Q]={0, 0, 0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1, -1,  1};    
//                                   3             7           10                15      17 

        block[I3_MPI(0,i,k,Nx,Nz) ] = MV_d_send.d3[I3(i,location,k)];
        block[I3_MPI(1,i,k,Nx,Nz) ] = MV_d_send.d7[I3(i,location,k)];
        block[I3_MPI(2,i,k,Nx,Nz) ] = MV_d_send.d10[I3(i,location,k)];
        block[I3_MPI(3,i,k,Nx,Nz) ] = MV_d_send.d15[I3(i,location,k)];
        block[I3_MPI(4,i,k,Nx,Nz) ] = MV_d_send.d17[I3(i,location,k)];

    }
}





__global__ void kernel_copy_FaceY_Recv_stream_negative_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;   
    int i=bx*(blockDim.x)+tx;
    int k=by*(blockDim.y)+ty;

    if((i<Nx)&&(k<Nz)) 
    {

            
        const int cx[Q]={0, 1, -1, 0, 0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0};
        const int cy[Q]={0, 0, 0, 1, -1, 0,  0, 1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1};
        const int cz[Q]={0, 0, 0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1, -1,  1};    
//                                3             7           10                15      17        
        int _i=i+cx[3];
        int _k=k+cz[3];
        int kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {
            MV_d_recv.d3[kk1]=              block[I3_MPI(0,i,k,Nx,Nz)]; 
        }
        _i=i+cx[7];
        _k=k+cz[7];
        kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {       
            MV_d_recv.d7[kk1]=             block[I3_MPI(1,i,k,Nx,Nz)]; 
        }
        _i=i+cx[10];
        _k=k+cz[10];
        kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {             
            MV_d_recv.d10[kk1]=             block[I3_MPI(2,i,k,Nx,Nz)]; 
        }
        _i=i+cx[15];
        _k=k+cz[15];
        kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {          
            MV_d_recv.d15[kk1]=             block[I3_MPI(3,i,k,Nx,Nz)]; 
        }

        _i=i+cx[17];
        _k=k+cz[17];
        kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {          
            MV_d_recv.d17[kk1]=             block[I3_MPI(4,i,k,Nx,Nz)]; 
        }


    }
}




__global__ void kernel_copy_FaceY_Recv_stream_positive_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv)
{

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;   
    int i=bx*(blockDim.x)+tx;
    int k=by*(blockDim.y)+ty;

    if((i<Nx)&&(k<Nz))    
    {

            
        const int cx[Q]={0, 1, -1, 0, 0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0};
        const int cy[Q]={0, 0, 0, 1, -1, 0,  0, 1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1};
        const int cz[Q]={0, 0, 0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1, -1,  1};    
//                                    4             8   9                         16      18    

        int _i=i+cx[4];
        int _k=k+cz[4];
        int kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {
            MV_d_recv.d4[kk1]=              block[I3_MPI(0,i,k,Nx,Nz)]; 
        }
        _i=i+cx[8];
        _k=k+cz[8];
        kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {       
            MV_d_recv.d8[kk1]=             block[I3_MPI(1,i,k,Nx,Nz)];             
        }

        _i=i+cx[9];
        _k=k+cz[9];
        kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {             
            MV_d_recv.d9[kk1]=             block[I3_MPI(2,i,k,Nx,Nz)]; 
        }
        _i=i+cx[16];
        _k=k+cz[16];
        kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {          
            MV_d_recv.d16[kk1]=             block[I3_MPI(3,i,k,Nx,Nz)];             
        }

        _i=i+cx[18];
        _k=k+cz[18];
        kk1=I3(_i,location,_k);
        if((_i<0)||(_i>Nx-1)||(_k<0)||(_k>Nz-1))
        {

        }
        else
        {          
            MV_d_recv.d18[kk1]=             block[I3_MPI(4,i,k,Nx,Nz)];             
        }
    }
}


//  =============== Z =============

__global__ void kernel_copy_FaceZ_Send_stream_negative(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send)
{
    unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
    // step 1: compute gridIndex in 1-D and 1-D data index "index_in"
    unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
    gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
    index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
    // step 2: extract 3-D data index via
    // index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
    // where xIndex = i-1, yIndex = j-1, zIndex = k-1
    if ( index_in < sizeOfData )
    {
        t1 =  index_in/ Nz; 
        zIndex = index_in - Nz*t1 ;
        xIndex =  t1/ Ny; 
        yIndex = t1 - Ny * xIndex ;
        
        // const int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
        // const int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
        // const int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};    
//                                             6                 12  13        16  17                               

        unsigned int i=xIndex, j=yIndex, k=zIndex;
        block[I3_MPI(0,i,j,Ny,Nx) ] = MV_d_send.d6[I3(i,j,location)];
        block[I3_MPI(1,i,j,Ny,Nx) ] = MV_d_send.d12[I3(i,j,location)];
        block[I3_MPI(2,i,j,Ny,Nx) ] = MV_d_send.d13[I3(i,j,location)];
        block[I3_MPI(3,i,j,Ny,Nx) ] = MV_d_send.d16[I3(i,j,location)];
        block[I3_MPI(4,i,j,Ny,Nx) ] = MV_d_send.d17[I3(i,j,location)];

    }
}



__global__ void kernel_copy_FaceZ_Send_stream_positive(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send)
{
    unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
    // step 1: compute gridIndex in 1-D and 1-D data index "index_in"
    unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
    gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
    index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
    // step 2: extract 3-D data index via
    // index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
    // where xIndex = i-1, yIndex = j-1, zIndex = k-1
    if ( index_in < sizeOfData )
    {
        t1 =  index_in/ Nz; 
        zIndex = index_in - Nz*t1 ;
        xIndex =  t1/ Ny; 
        yIndex = t1 - Ny * xIndex ;
        
        // const int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
        // const int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
        // const int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};    
//                                         5                  11         14 15        18                                 

        unsigned int i=xIndex, j=yIndex, k=zIndex;
        block[I3_MPI(0,i,j,Ny,Nx) ] = MV_d_send.d5[I3(i,j,location)];
        block[I3_MPI(1,i,j,Ny,Nx) ] = MV_d_send.d11[I3(i,j,location)];
        block[I3_MPI(2,i,j,Ny,Nx) ] = MV_d_send.d14[I3(i,j,location)];
        block[I3_MPI(3,i,j,Ny,Nx) ] = MV_d_send.d15[I3(i,j,location)];
        block[I3_MPI(4,i,j,Ny,Nx) ] = MV_d_send.d18[I3(i,j,location)];

    }
}




__global__ void kernel_copy_FaceZ_Recv_stream_negative(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv)
{
    unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
    // step 1: compute gridIndex in 1-D and 1-D data index "index_in"
    unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
    gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
    index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
    // step 2: extract 3-D data index via
    // index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
    // where xIndex = i-1, yIndex = j-1, zIndex = k-1
    if ( index_in < sizeOfData )
    {
        t1 =  index_in/ Nz; 
        zIndex = index_in - Nz*t1 ;
        xIndex =  t1/ Ny; 
        yIndex = t1 - Ny * xIndex ;
            
        const int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
        const int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
        const int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
//                       0  1  2  3  4  5   6  7  8  9  10 11  12 13  14 15  16  17 18
//                                      *                  *          *  *          *        
        unsigned int i=xIndex, j=yIndex, k=zIndex;
        int _i=i+cx[5];
        int _j=j+cy[5];
        int kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {
            MV_d_recv.d5[kk1]=              block[I3_MPI(0,i,j,Ny,Nx)]; 
        }
        _i=i+cx[11];
        _j=j+cy[11];
        kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {       
            MV_d_recv.d11[kk1]=             block[I3_MPI(1,i,j,Ny,Nx)]; 
        }

        _i=i+cx[14];
        _j=j+cy[14];
        kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {             
            MV_d_recv.d14[kk1]=             block[I3_MPI(2,i,j,Ny,Nx)]; 
        }
        _i=i+cx[15];
        _j=j+cy[15];
        kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {          
            MV_d_recv.d15[kk1]=             block[I3_MPI(3,i,j,Ny,Nx)]; 
        }
        _i=i+cx[18];
        _j=j+cy[18];
        kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {          
            MV_d_recv.d18[kk1]=             block[I3_MPI(4,i,j,Ny,Nx)]; 
        }


    }
}




__global__ void kernel_copy_FaceZ_Recv_stream_positive(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv)
{
    unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
    // step 1: compute gridIndex in 1-D and 1-D data index "index_in"
    unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
    gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
    index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
    // step 2: extract 3-D data index via
    // index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
    // where xIndex = i-1, yIndex = j-1, zIndex = k-1
    if ( index_in < sizeOfData )
    {
        t1 =  index_in/ Nz; 
        zIndex = index_in - Nz*t1 ;
        xIndex =  t1/ Ny; 
        yIndex = t1 - Ny * xIndex ;
            
        const int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
        const int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
        const int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
//                       0  1  2  3  4  5   6  7  8  9  10 11  12 13  14 15  16  17 18
//                                          *                  *  *          *   *       
        unsigned int i=xIndex, j=yIndex, k=zIndex;
        int _i=i+cx[6];
        int _j=j+cy[6];
        int kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {
            MV_d_recv.d6[kk1]=              block[I3_MPI(0,i,j,Ny,Nx)]; 
        }

        _i=i+cx[12];
        _j=j+cy[12];
        kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {       
            MV_d_recv.d12[kk1]=             block[I3_MPI(1,i,j,Ny,Nx)];             
        }

        _i=i+cx[13];
        _j=j+cy[13];
        kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {             
            MV_d_recv.d13[kk1]=             block[I3_MPI(2,i,j,Ny,Nx)]; 
        }
        

        _i=i+cx[16];
        _j=j+cy[16];
        kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {          
            MV_d_recv.d16[kk1]=             block[I3_MPI(3,i,j,Ny,Nx)];             
        }


        _i=i+cx[17];
        _j=j+cy[17];
        kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {          
            MV_d_recv.d17[kk1]=             block[I3_MPI(4,i,j,Ny,Nx)];             
        }

    }
}





__global__ void kernel_copy_FaceZ_Send_stream_negative_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send)
{

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;   
    int i=bx*(blockDim.x)+tx;
    int j=by*(blockDim.y)+ty;

    if((i<Nx)&&(j<Ny))
    {
        
        // const int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
        // const int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
        // const int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};    
//                                             6                 12  13        16  17                               


        block[I3_MPI(0,i,j,Ny,Nx) ] = MV_d_send.d6[I3(i,j,location)];
        block[I3_MPI(1,i,j,Ny,Nx) ] = MV_d_send.d12[I3(i,j,location)];
        block[I3_MPI(2,i,j,Ny,Nx) ] = MV_d_send.d13[I3(i,j,location)];
        block[I3_MPI(3,i,j,Ny,Nx) ] = MV_d_send.d16[I3(i,j,location)];
        block[I3_MPI(4,i,j,Ny,Nx) ] = MV_d_send.d17[I3(i,j,location)];

    }
}



__global__ void kernel_copy_FaceZ_Send_stream_positive_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_send)
{

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;   
    int i=bx*(blockDim.x)+tx;
    int j=by*(blockDim.y)+ty;

    if((i<Nx)&&(j<Ny))    
    {
        
        // const int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
        // const int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
        // const int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};    
//                                         5                  11         14 15        18                                 

        block[I3_MPI(0,i,j,Ny,Nx) ] = MV_d_send.d5[I3(i,j,location)];
        block[I3_MPI(1,i,j,Ny,Nx) ] = MV_d_send.d11[I3(i,j,location)];
        block[I3_MPI(2,i,j,Ny,Nx) ] = MV_d_send.d14[I3(i,j,location)];
        block[I3_MPI(3,i,j,Ny,Nx) ] = MV_d_send.d15[I3(i,j,location)];
        block[I3_MPI(4,i,j,Ny,Nx) ] = MV_d_send.d18[I3(i,j,location)];

    }
}




__global__ void kernel_copy_FaceZ_Recv_stream_negative_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;   
    int i=bx*(blockDim.x)+tx;
    int j=by*(blockDim.y)+ty;

    if((i<Nx)&&(j<Ny))
    {
            
        const int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
        const int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
        const int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
//                       0  1  2  3  4  5   6  7  8  9  10 11  12 13  14 15  16  17 18
//                                      *                  *          *  *          *        
        int _i=i+cx[5];
        int _j=j+cy[5];
        int kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {
            MV_d_recv.d5[kk1]=              block[I3_MPI(0,i,j,Ny,Nx)]; 
        }
        _i=i+cx[11];
        _j=j+cy[11];
        kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {       
            MV_d_recv.d11[kk1]=             block[I3_MPI(1,i,j,Ny,Nx)]; 
        }

        _i=i+cx[14];
        _j=j+cy[14];
        kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {             
            MV_d_recv.d14[kk1]=             block[I3_MPI(2,i,j,Ny,Nx)]; 
        }
        _i=i+cx[15];
        _j=j+cy[15];
        kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {          
            MV_d_recv.d15[kk1]=             block[I3_MPI(3,i,j,Ny,Nx)]; 
        }
        _i=i+cx[18];
        _j=j+cy[18];
        kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {          
            MV_d_recv.d18[kk1]=             block[I3_MPI(4,i,j,Ny,Nx)]; 
        }


    }
}




__global__ void kernel_copy_FaceZ_Recv_stream_positive_2D(int Nx, int Ny, int Nz, real *block, int location, microscopic_variables MV_d_recv)
{

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;   
    int i=bx*(blockDim.x)+tx;
    int j=by*(blockDim.y)+ty;

    if((i<Nx)&&(j<Ny))
    {

        const int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
        const int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
        const int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
//                       0  1  2  3  4  5   6  7  8  9  10 11  12 13  14 15  16  17 18
//                                          *                  *  *          *   *       
        int _i=i+cx[6];
        int _j=j+cy[6];
        int kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {
            MV_d_recv.d6[kk1]=              block[I3_MPI(0,i,j,Ny,Nx)]; 
        }

        _i=i+cx[12];
        _j=j+cy[12];
        kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {       
            MV_d_recv.d12[kk1]=             block[I3_MPI(1,i,j,Ny,Nx)];             
        }

        _i=i+cx[13];
        _j=j+cy[13];
        kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {             
            MV_d_recv.d13[kk1]=             block[I3_MPI(2,i,j,Ny,Nx)]; 
        }
        

        _i=i+cx[16];
        _j=j+cy[16];
        kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {          
            MV_d_recv.d16[kk1]=             block[I3_MPI(3,i,j,Ny,Nx)];             
        }


        _i=i+cx[17];
        _j=j+cy[17];
        kk1=I3(_i,_j,location);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1))
        {

        }
        else
        {          
            MV_d_recv.d17[kk1]=             block[I3_MPI(4,i,j,Ny,Nx)];             
        }

    }
}
