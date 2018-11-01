#include "map.h"


void allocate_blocks(int Nx, int Ny, int Nz, communication_variables *COM)
{
    COM->Face1Bufer_size=0;
    COM->Face2Bufer_size=0;
    if(COM->FaceA=='C')
    {
        COM->Face1proc=COM->FaceAproc;
        COM->Face1Bufer_size=Ny*Nz*19*stride;
        COM->Face1BuferSend_host=host_allocate<real>(COM->Face1Bufer_size);  //size of single stride times number of bits in LBM
        COM->Face1BuferRecv_host=host_allocate<real>(COM->Face1Bufer_size);  //size of single stride times number of bits in LBM
        if(COM->device==1){
            COM->Face1BuferSend_device=device_allocate<real>(COM->Face1Bufer_size);  //size of single stride times number of bits in LBM
            COM->Face1BuferRecv_device=device_allocate<real>(COM->Face1Bufer_size);  //size of single stride times number of bits in LBM
        }

    }
    if(COM->FaceB=='C')
    {
        COM->Face2proc=COM->FaceBproc;
        COM->Face2Bufer_size=Ny*Nz*19*stride;
        COM->Face2BuferSend_host=host_allocate<real>(COM->Face2Bufer_size);  
        COM->Face2BuferRecv_host=host_allocate<real>(COM->Face2Bufer_size);  
        if(COM->device==1){
            COM->Face2BuferSend_device=device_allocate<real>(COM->Face2Bufer_size); 
            COM->Face2BuferRecv_device=device_allocate<real>(COM->Face2Bufer_size); 
        }        

    }

    if(COM->FaceC=='C')
    {
        COM->Face1proc=COM->FaceCproc;
        COM->Face1Bufer_size=Nx*Nz*19*stride;
        COM->Face1BuferSend_host=host_allocate<real>(COM->Face1Bufer_size);  
        COM->Face1BuferRecv_host=host_allocate<real>(COM->Face1Bufer_size); 
        if(COM->device==1){
            COM->Face1BuferSend_device=device_allocate<real>(COM->Face1Bufer_size);  
            COM->Face1BuferRecv_device=device_allocate<real>(COM->Face1Bufer_size);  
        }

    }
    if(COM->FaceD=='C')
    {
        COM->Face2proc=COM->FaceDproc;
        COM->Face2Bufer_size=Nx*Nz*19*stride;
        COM->Face2BuferSend_host=host_allocate<real>(COM->Face2Bufer_size);  
        COM->Face2BuferRecv_host=host_allocate<real>(COM->Face2Bufer_size);  
        if(COM->device==1){
            COM->Face2BuferSend_device=device_allocate<real>(COM->Face2Bufer_size); 
            COM->Face2BuferRecv_device=device_allocate<real>(COM->Face2Bufer_size); 
        }        

    }
    if(COM->FaceE=='C')
    {
        COM->Face1proc=COM->FaceEproc;
        COM->Face1Bufer_size=Nx*Ny*19*stride;
        COM->Face1BuferSend_host=host_allocate<real>(COM->Face1Bufer_size);  
        COM->Face1BuferRecv_host=host_allocate<real>(COM->Face1Bufer_size); 
        if(COM->device==1){
            COM->Face1BuferSend_device=device_allocate<real>(COM->Face1Bufer_size);  
            COM->Face1BuferRecv_device=device_allocate<real>(COM->Face1Bufer_size);  
        }

    }
    if(COM->FaceF=='C')
    {
        COM->Face2proc=COM->FaceFproc;
        COM->Face2Bufer_size=Nx*Ny*19*stride;
        COM->Face2BuferSend_host=host_allocate<real>(COM->Face2Bufer_size);  
        COM->Face2BuferRecv_host=host_allocate<real>(COM->Face2Bufer_size);  
        if(COM->device==1){
            COM->Face2BuferSend_device=device_allocate<real>(COM->Face2Bufer_size); 
            COM->Face2BuferRecv_device=device_allocate<real>(COM->Face2Bufer_size); 
        }        

    }

}


void deallocate_blocks(int Nx, int Ny, int Nz, communication_variables *COM)
{
    if(COM->Face1Bufer_size>0)
    {
        free(COM->Face1BuferSend_host);
        free(COM->Face1BuferRecv_host);
        if(COM->device==1){
            CUDA_SAFE_CALL(cudaFree(COM->Face1BuferSend_device));
            CUDA_SAFE_CALL(cudaFree(COM->Face1BuferRecv_device));
        }
        COM->Face1Bufer_size=0;
    }
    if(COM->Face2Bufer_size>0)
    {
        free(COM->Face2BuferSend_host);
        free(COM->Face2BuferRecv_host);
        if(COM->device==1){
            CUDA_SAFE_CALL(cudaFree(COM->Face2BuferSend_device));
            CUDA_SAFE_CALL(cudaFree(COM->Face2BuferRecv_device));
        }
        COM->Face2Bufer_size=0;
    }

}


__global__ void kernel_copy_FaceX_Send(int Nx, int Ny, int Nz, real *block, int location, real *f0, real *f1, real *f2, real *f3, real *f4, real *f5,
                                real *f6, real *f7, real *f8, real *f9, real *f10, real *f11, real *f12, 
                                real *f13, real *f14, real *f15, real *f16, real *f17, real *f18)
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


            
        unsigned int i=xIndex, j=yIndex, k=zIndex;
        block[I3_MPI(0,j,k,Ny,Nz) ] = f0[I3(location,j,k)];
        block[I3_MPI(1,j,k,Ny,Nz) ] = f1[I3(location,j,k)];
        block[I3_MPI(2,j,k,Ny,Nz) ] = f2[I3(location,j,k)];
        block[I3_MPI(3,j,k,Ny,Nz) ] = f3[I3(location,j,k)];
        block[I3_MPI(4,j,k,Ny,Nz) ] = f4[I3(location,j,k)];
        block[I3_MPI(5,j,k,Ny,Nz) ] = f5[I3(location,j,k)];
        block[I3_MPI(6,j,k,Ny,Nz) ] = f6[I3(location,j,k)];
        block[I3_MPI(7,j,k,Ny,Nz) ] = f7[I3(location,j,k)];
        block[I3_MPI(8,j,k,Ny,Nz) ] = f8[I3(location,j,k)];
        block[I3_MPI(9,j,k,Ny,Nz) ] = f9[I3(location,j,k)];
        block[I3_MPI(10,j,k,Ny,Nz) ] = f10[I3(location,j,k)];
        block[I3_MPI(11,j,k,Ny,Nz) ] = f11[I3(location,j,k)];
        block[I3_MPI(12,j,k,Ny,Nz) ] = f12[I3(location,j,k)];
        block[I3_MPI(13,j,k,Ny,Nz) ] = f13[I3(location,j,k)];
        block[I3_MPI(14,j,k,Ny,Nz) ] = f14[I3(location,j,k)];
        block[I3_MPI(15,j,k,Ny,Nz) ] = f15[I3(location,j,k)];
        block[I3_MPI(16,j,k,Ny,Nz) ] = f16[I3(location,j,k)];
        block[I3_MPI(17,j,k,Ny,Nz) ] = f17[I3(location,j,k)];
        block[I3_MPI(18,j,k,Ny,Nz) ] = f18[I3(location,j,k)];

    }
}


__global__ void kernel_copy_FaceY_Send(int Nx, int Ny, int Nz, real *block, int location, real *f0, real *f1, real *f2, real *f3, real *f4, real *f5,
                                real *f6, real *f7, real *f8, real *f9, real *f10, real *f11, real *f12, 
                                real *f13, real *f14, real *f15, real *f16, real *f17, real *f18)
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
            
        unsigned int i=xIndex, j=yIndex, k=zIndex;
        block[I3_MPI(0,i,k,Nx,Nz) ] = f0[I3(i,location,k)];
        block[I3_MPI(1,i,k,Nx,Nz) ] = f1[I3(i,location,k)];
        block[I3_MPI(2,i,k,Nx,Nz) ] = f2[I3(i,location,k)];
        block[I3_MPI(3,i,k,Nx,Nz) ] = f3[I3(i,location,k)];
        block[I3_MPI(4,i,k,Nx,Nz) ] = f4[I3(i,location,k)];
        block[I3_MPI(5,i,k,Nx,Nz) ] = f5[I3(i,location,k)];
        block[I3_MPI(6,i,k,Nx,Nz) ] = f6[I3(i,location,k)];
        block[I3_MPI(7,i,k,Nx,Nz) ] = f7[I3(i,location,k)];
        block[I3_MPI(8,i,k,Nx,Nz) ] = f8[I3(i,location,k)];
        block[I3_MPI(9,i,k,Nx,Nz) ] = f9[I3(i,location,k)];
        block[I3_MPI(10,i,k,Nx,Nz) ] = f10[I3(i,location,k)];
        block[I3_MPI(11,i,k,Nx,Nz) ] = f11[I3(i,location,k)];
        block[I3_MPI(12,i,k,Nx,Nz) ] = f12[I3(i,location,k)];
        block[I3_MPI(13,i,k,Nx,Nz) ] = f13[I3(i,location,k)];
        block[I3_MPI(14,i,k,Nx,Nz) ] = f14[I3(i,location,k)];
        block[I3_MPI(15,i,k,Nx,Nz) ] = f15[I3(i,location,k)];
        block[I3_MPI(16,i,k,Nx,Nz) ] = f16[I3(i,location,k)];
        block[I3_MPI(17,i,k,Nx,Nz) ] = f17[I3(i,location,k)];
        block[I3_MPI(18,i,k,Nx,Nz) ] = f18[I3(i,location,k)];

    }
}



__global__ void kernel_copy_FaceZ_Send(int Nx, int Ny, int Nz, real *block, int location, real *f0, real *f1, real *f2, real *f3, real *f4, real *f5,
                                real *f6, real *f7, real *f8, real *f9, real *f10, real *f11, real *f12, 
                                real *f13, real *f14, real *f15, real *f16, real *f17, real *f18)
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
//                                         5   6              11  12  13 14 15  16  17 18                            

        unsigned int i=xIndex, j=yIndex, k=zIndex;
        block[I3_MPI(0,i,j,Ny,Nx) ] = f0[I3(i,j,location)];
        block[I3_MPI(1,i,j,Ny,Nx) ] = f1[I3(i,j,location)];
        block[I3_MPI(2,i,j,Ny,Nx) ] = f2[I3(i,j,location)];
        block[I3_MPI(3,i,j,Ny,Nx) ] = f3[I3(i,j,location)];
        block[I3_MPI(4,i,j,Ny,Nx) ] = f4[I3(i,j,location)];
        block[I3_MPI(5,i,j,Ny,Nx) ] = f5[I3(i,j,location)];
        block[I3_MPI(6,i,j,Ny,Nx) ] = f6[I3(i,j,location)];
        block[I3_MPI(7,i,j,Ny,Nx) ] = f7[I3(i,j,location)];
        block[I3_MPI(8,i,j,Ny,Nx) ] = f8[I3(i,j,location)];
        block[I3_MPI(9,i,j,Ny,Nx) ] = f9[I3(i,j,location)];
        block[I3_MPI(10,i,j,Ny,Nx) ] = f10[I3(i,j,location)];
        block[I3_MPI(11,i,j,Ny,Nx) ] = f11[I3(i,j,location)];
        block[I3_MPI(12,i,j,Ny,Nx) ] = f12[I3(i,j,location)];
        block[I3_MPI(13,i,j,Ny,Nx) ] = f13[I3(i,j,location)];
        block[I3_MPI(14,i,j,Ny,Nx) ] = f14[I3(i,j,location)];
        block[I3_MPI(15,i,j,Ny,Nx) ] = f15[I3(i,j,location)];
        block[I3_MPI(16,i,j,Ny,Nx) ] = f16[I3(i,j,location)];
        block[I3_MPI(17,i,j,Ny,Nx) ] = f17[I3(i,j,location)];
        block[I3_MPI(18,i,j,Ny,Nx) ] = f18[I3(i,j,location)];

    }
}


__global__ void kernel_copy_FaceX_Recv(int Nx, int Ny, int Nz, real *block, int location, real *f0, real *f1, real *f2, real *f3, real *f4, real *f5,
                                real *f6, real *f7, real *f8, real *f9, real *f10, real *f11, real *f12, 
                                real *f13, real *f14, real *f15, real *f16, real *f17, real *f18)
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
            
        unsigned int i=xIndex, j=yIndex, k=zIndex;
        f0[I3(location,j,k)] = block[I3_MPI(0,j,k,Ny,Nz)];
        f1[I3(location,j,k)] = block[I3_MPI(1,j,k,Ny,Nz)];
        f2[I3(location,j,k)] = block[I3_MPI(2,j,k,Ny,Nz)]; 
        f3[I3(location,j,k)] = block[I3_MPI(3,j,k,Ny,Nz)]; 
        f4[I3(location,j,k)] = block[I3_MPI(4,j,k,Ny,Nz)]; 
        f5[I3(location,j,k)] = block[I3_MPI(5,j,k,Ny,Nz)]; 
        f6[I3(location,j,k)] = block[I3_MPI(6,j,k,Ny,Nz)]; 
        f7[I3(location,j,k)] = block[I3_MPI(7,j,k,Ny,Nz)]; 
        f8[I3(location,j,k)] = block[I3_MPI(8,j,k,Ny,Nz)]; 
        f9[I3(location,j,k)] = block[I3_MPI(9,j,k,Ny,Nz)]; 
        f10[I3(location,j,k)] = block[I3_MPI(10,j,k,Ny,Nz)]; 
        f11[I3(location,j,k)] = block[I3_MPI(11,j,k,Ny,Nz)]; 
        f12[I3(location,j,k)] = block[I3_MPI(12,j,k,Ny,Nz)]; 
        f13[I3(location,j,k)] = block[I3_MPI(13,j,k,Ny,Nz)]; 
        f14[I3(location,j,k)] = block[I3_MPI(14,j,k,Ny,Nz)]; 
        f15[I3(location,j,k)] = block[I3_MPI(15,j,k,Ny,Nz)]; 
        f16[I3(location,j,k)] = block[I3_MPI(16,j,k,Ny,Nz)]; 
        f17[I3(location,j,k)] = block[I3_MPI(17,j,k,Ny,Nz)]; 
        f18[I3(location,j,k)] = block[I3_MPI(18,j,k,Ny,Nz)]; 


    }
}


__global__ void kernel_copy_FaceY_Recv(int Nx, int Ny, int Nz, real *block, int location, real *f0, real *f1, real *f2, real *f3, real *f4, real *f5,
                                real *f6, real *f7, real *f8, real *f9, real *f10, real *f11, real *f12, 
                                real *f13, real *f14, real *f15, real *f16, real *f17, real *f18)
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
            
        unsigned int i=xIndex, j=yIndex, k=zIndex;
        f0[I3(i,location,k)]=              block[I3_MPI(0,i,k,Nx,Nz)]; 
        f1[I3(i,location,k)]=              block[I3_MPI(1,i,k,Nx,Nz)]; 
        f2[I3(i,location,k)]=              block[I3_MPI(2,i,k,Nx,Nz)]; 
        f3[I3(i,location,k)]=              block[I3_MPI(3,i,k,Nx,Nz)]; 
        f4[I3(i,location,k)]=              block[I3_MPI(4,i,k,Nx,Nz)]; 
        f5[I3(i,location,k)]=              block[I3_MPI(5,i,k,Nx,Nz)]; 
        f6[I3(i,location,k)]=              block[I3_MPI(6,i,k,Nx,Nz)]; 
        f7[I3(i,location,k)]=              block[I3_MPI(7,i,k,Nx,Nz)]; 
        f8[I3(i,location,k)]=              block[I3_MPI(8,i,k,Nx,Nz)]; 
        f9[I3(i,location,k)]=              block[I3_MPI(9,i,k,Nx,Nz)]; 
        f10[I3(i,location,k)]=             block[I3_MPI(10,i,k,Nx,Nz)]; 
        f11[I3(i,location,k)]=             block[I3_MPI(11,i,k,Nx,Nz)]; 
        f12[I3(i,location,k)]=             block[I3_MPI(12,i,k,Nx,Nz)]; 
        f13[I3(i,location,k)]=             block[I3_MPI(13,i,k,Nx,Nz)]; 
        f14[I3(i,location,k)]=             block[I3_MPI(14,i,k,Nx,Nz)]; 
        f15[I3(i,location,k)]=             block[I3_MPI(15,i,k,Nx,Nz)]; 
        f16[I3(i,location,k)]=             block[I3_MPI(16,i,k,Nx,Nz)]; 
        f17[I3(i,location,k)]=             block[I3_MPI(17,i,k,Nx,Nz)]; 
        f18[I3(i,location,k)]=             block[I3_MPI(18,i,k,Nx,Nz)]; 


    }
}



__global__ void kernel_copy_FaceZ_Recv(int Nx, int Ny, int Nz, real *block, int location, real *f0, real *f1, real *f2, real *f3, real *f4, real *f5,
                                real *f6, real *f7, real *f8, real *f9, real *f10, real *f11, real *f12, 
                                real *f13, real *f14, real *f15, real *f16, real *f17, real *f18)
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
            
        unsigned int i=xIndex, j=yIndex, k=zIndex;
        f0[I3(i,j,location)]=              block[I3_MPI(0,i,j,Ny,Nx)]; 
        f1[I3(i,j,location)]=              block[I3_MPI(1,i,j,Ny,Nx)]; 
        f2[I3(i,j,location)]=              block[I3_MPI(2,i,j,Ny,Nx)]; 
        f3[I3(i,j,location)]=              block[I3_MPI(3,i,j,Ny,Nx)]; 
        f4[I3(i,j,location)]=              block[I3_MPI(4,i,j,Ny,Nx)]; 
        f5[I3(i,j,location)]=              block[I3_MPI(5,i,j,Ny,Nx)]; 
        f6[I3(i,j,location)]=              block[I3_MPI(6,i,j,Ny,Nx)]; 
        f7[I3(i,j,location)]=              block[I3_MPI(7,i,j,Ny,Nx)]; 
        f8[I3(i,j,location)]=              block[I3_MPI(8,i,j,Ny,Nx)]; 
        f9[I3(i,j,location)]=              block[I3_MPI(9,i,j,Ny,Nx)]; 
        f10[I3(i,j,location)]=             block[I3_MPI(10,i,j,Ny,Nx)]; 
        f11[I3(i,j,location)]=             block[I3_MPI(11,i,j,Ny,Nx)]; 
        f12[I3(i,j,location)]=             block[I3_MPI(12,i,j,Ny,Nx)]; 
        f13[I3(i,j,location)]=             block[I3_MPI(13,i,j,Ny,Nx)]; 
        f14[I3(i,j,location)]=             block[I3_MPI(14,i,j,Ny,Nx)]; 
        f15[I3(i,j,location)]=             block[I3_MPI(15,i,j,Ny,Nx)]; 
        f16[I3(i,j,location)]=             block[I3_MPI(16,i,j,Ny,Nx)]; 
        f17[I3(i,j,location)]=             block[I3_MPI(17,i,j,Ny,Nx)]; 
        f18[I3(i,j,location)]=             block[I3_MPI(18,i,j,Ny,Nx)]; 


    }
}


void copy_send_buffers_streams(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM, microscopic_variables MV_d1)
{

    if(COM->MPI_sections=='X')
    {
        if(COM->Face1Bufer_size>0)
        {
            kernel_copy_FaceX_Send<<<dimGrid, dimBlock,0,COM->streams[1]>>>(Nx, Ny, Nz, COM->Face1BuferSend_device, 1, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);
            
        }
        if(COM->Face2Bufer_size>0)
        {
            kernel_copy_FaceX_Send<<<dimGrid, dimBlock,0,COM->streams[1]>>>(Nx, Ny, Nz, COM->Face2BuferSend_device, Nx-2, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);

        }
    }
    if(COM->MPI_sections=='Y')
    {
        if(COM->Face1Bufer_size>0)
        {
            kernel_copy_FaceY_Send<<<dimGrid, dimBlock,0,COM->streams[1]>>>(Nx, Ny, Nz, COM->Face1BuferSend_device, 1, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);
            
        }
        if(COM->Face2Bufer_size>0)
        {
            kernel_copy_FaceY_Send<<<dimGrid, dimBlock,0,COM->streams[1]>>>(Nx, Ny, Nz, COM->Face2BuferSend_device, Ny-2, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);

        }
    }
    if(COM->MPI_sections=='Z')
    {
        if(COM->Face1Bufer_size>0)
        {
            kernel_copy_FaceZ_Send<<<dimGrid, dimBlock,0,COM->streams[1]>>>(Nx, Ny, Nz, COM->Face1BuferSend_device, 1, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);
            
        }
        if(COM->Face2Bufer_size>0)
        {
            kernel_copy_FaceZ_Send<<<dimGrid, dimBlock,0,COM->streams[1]>>>(Nx, Ny, Nz, COM->Face2BuferSend_device, Nz-2, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);

        }
    }
    CUDA_SAFE_CALL(cudaStreamSynchronize( COM->streams[1]));

}


void copy_recv_buffers_streams(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM, microscopic_variables MV_d1)
{
    if(COM->MPI_sections=='X')
    {
        if(COM->Face1Bufer_size>0)
        {
            kernel_copy_FaceX_Recv<<<dimGrid, dimBlock,0,COM->streams[1]>>>(Nx, Ny, Nz, COM->Face1BuferRecv_device, 0, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);
            
        }
        if(COM->Face2Bufer_size>0)
        {
            kernel_copy_FaceX_Recv<<<dimGrid, dimBlock,0,COM->streams[1]>>>(Nx, Ny, Nz, COM->Face2BuferRecv_device, Nx-1, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);

        }
    }
    if(COM->MPI_sections=='Y')
    {
        if(COM->Face1Bufer_size>0)
        {
            kernel_copy_FaceY_Recv<<<dimGrid, dimBlock,0,COM->streams[1]>>>(Nx, Ny, Nz, COM->Face1BuferRecv_device, 0, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);
            
        }
        if(COM->Face2Bufer_size>0)
        {
            kernel_copy_FaceY_Recv<<<dimGrid, dimBlock,0,COM->streams[1]>>>(Nx, Ny, Nz, COM->Face2BuferRecv_device, Ny-1, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);

        }
    }
    if(COM->MPI_sections=='Z')
    {
        if(COM->Face1Bufer_size>0)
        {
            kernel_copy_FaceZ_Recv<<<dimGrid, dimBlock,0,COM->streams[1]>>>(Nx, Ny, Nz, COM->Face1BuferRecv_device, 0, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);
            
        }
        if(COM->Face2Bufer_size>0)
        {
            kernel_copy_FaceZ_Recv<<<dimGrid, dimBlock,0,COM->streams[1]>>>(Nx, Ny, Nz, COM->Face2BuferRecv_device, Nz-1, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);

        }
    }
    CUDA_SAFE_CALL(cudaStreamSynchronize( COM->streams[1]));

}

void copy_send_buffers(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM, microscopic_variables MV_d1)
{

    if(COM->MPI_sections=='X')
    {
        if(COM->Face1Bufer_size>0)
        {
            kernel_copy_FaceX_Send<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face1BuferSend_device, 1, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);
            
        }
        if(COM->Face2Bufer_size>0)
        {
            kernel_copy_FaceX_Send<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face2BuferSend_device, Nx-2, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);

        }
    }
    if(COM->MPI_sections=='Y')
    {
        if(COM->Face1Bufer_size>0)
        {
            kernel_copy_FaceY_Send<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face1BuferSend_device, 1, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);
            
        }
        if(COM->Face2Bufer_size>0)
        {
            kernel_copy_FaceY_Send<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face2BuferSend_device, Ny-2, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);

        }
    }
    if(COM->MPI_sections=='Z')
    {
        if(COM->Face1Bufer_size>0)
        {
            kernel_copy_FaceZ_Send<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face1BuferSend_device, 1, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);
            
        }
        if(COM->Face2Bufer_size>0)
        {
            kernel_copy_FaceZ_Send<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face2BuferSend_device, Nz-2, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);

        }
    }

}


void copy_recv_buffers(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM, microscopic_variables MV_d1)
{
    if(COM->MPI_sections=='X')
    {
        if(COM->Face1Bufer_size>0)
        {
            kernel_copy_FaceX_Recv<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face1BuferRecv_device, 0, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);
            
        }
        if(COM->Face2Bufer_size>0)
        {
            kernel_copy_FaceX_Recv<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face2BuferRecv_device, Nx-1, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);

        }
    }
    if(COM->MPI_sections=='Y')
    {
        if(COM->Face1Bufer_size>0)
        {
            kernel_copy_FaceY_Recv<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face1BuferRecv_device, 0, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);
            
        }
        if(COM->Face2Bufer_size>0)
        {
            kernel_copy_FaceY_Recv<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face2BuferRecv_device, Ny-1, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);

        }
    }
    if(COM->MPI_sections=='Z')
    {
        if(COM->Face1Bufer_size>0)
        {
            kernel_copy_FaceZ_Recv<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face1BuferRecv_device, 0, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);
            
        }
        if(COM->Face2Bufer_size>0)
        {
            kernel_copy_FaceZ_Recv<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face2BuferRecv_device, Nz-1, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7, MV_d1.d8, MV_d1.d9, MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17, MV_d1.d18);

        }
    }

}


void MPI_send_recv_GPU_buffer1(communication_variables *COM)
{


    MPI_Request request1, request2;
    int tag=0;
    MPI_Isend(COM->Face1BuferSend_device, COM->Face1Bufer_size, MPI_real, COM->Face1proc, tag, MPI_COMM_WORLD, &request1);        
    MPI_Irecv(COM->Face1BuferRecv_device, COM->Face1Bufer_size, MPI_real, COM->Face1proc, tag, MPI_COMM_WORLD, &request2);
    MPI_Wait(&request1, MPI_STATUS_IGNORE);
    MPI_Wait(&request2, MPI_STATUS_IGNORE);


}

void MPI_send_recv_GPU_buffer2(communication_variables *COM)
{


    MPI_Request request1, request2;
    int tag=0;
    MPI_Isend(COM->Face2BuferSend_device, COM->Face2Bufer_size, MPI_real, COM->Face2proc, tag, MPI_COMM_WORLD, &request1);        
    MPI_Irecv(COM->Face2BuferRecv_device , COM->Face2Bufer_size, MPI_real, COM->Face2proc, tag, MPI_COMM_WORLD, &request2);
    MPI_Wait(&request1, MPI_STATUS_IGNORE);
    MPI_Wait(&request2, MPI_STATUS_IGNORE);


}


void exchange_boundaries_MPI(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM, microscopic_variables MV_d1)
{



    copy_send_buffers(dimGrid, dimBlock, Nx, Ny, Nz, COM, MV_d1);
    //check for sending on CPU or GPU!!!
    if(COM->Face1Bufer_size>0){
        MPI_send_recv_GPU_buffer1(COM);
    }
    if(COM->Face2Bufer_size>0)
    {
        MPI_send_recv_GPU_buffer2(COM);
    }
    copy_recv_buffers(dimGrid, dimBlock, Nx, Ny, Nz, COM, MV_d1);


}


void exchange_boundaries_MPI_streams(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM, microscopic_variables MV_d1)
{



    copy_send_buffers_streams(dimGrid, dimBlock, Nx, Ny, Nz, COM, MV_d1);
    //check for sending on CPU or GPU!!!
    if(COM->Face1Bufer_size>0){
        MPI_send_recv_GPU_buffer1(COM);
    }
    if(COM->Face2Bufer_size>0)
    {
        MPI_send_recv_GPU_buffer2(COM);
    }
    copy_recv_buffers_streams(dimGrid, dimBlock, Nx, Ny, Nz, COM, MV_d1);


}

