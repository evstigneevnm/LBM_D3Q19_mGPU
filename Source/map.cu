#include "map.h"


void allocate_blocks(int Nx, int Ny, int Nz, communication_variables *COM)
{
    COM->Face1Bufer_size=0;
    COM->Face2Bufer_size=0;
    if(COM->FaceA=='C')
    {
        COM->Face1proc=COM->FaceAproc;
        COM->Face1Bufer_size=Ny*Nz*5*stride;
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
        COM->Face2Bufer_size=Ny*Nz*5*stride;
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
        COM->Face1Bufer_size=Nx*Nz*5*stride;
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
        COM->Face2Bufer_size=Nx*Nz*5*stride;
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
        COM->Face1Bufer_size=Nx*Ny*5*stride;
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
        COM->Face2Bufer_size=Nx*Ny*5*stride;
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





void copy_send_buffers_streams(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM, microscopic_variables MV_d1)
{

    dim3 threads( BLOCK_DIM_X, BLOCK_DIM_Y );


    if(COM->MPI_sections=='X')
    {
        int blocks_x=Ny/( BLOCK_DIM_X)+1;
        int blocks_y=Nz/( BLOCK_DIM_Y)+1;
        dim3 blocks( blocks_x, blocks_y);
        if(COM->Face1Bufer_size>0)
        {
            
            kernel_copy_FaceX_Send_stream_negative_2D<<<blocks, threads>>>(Nx, Ny, Nz, COM->Face1BuferSend_device, 0, MV_d1);
           // kernel_copy_FaceX_Send_stream_negative<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face1BuferSend_device, 0, MV_d1);
            
        }
        if(COM->Face2Bufer_size>0)
        {
            kernel_copy_FaceX_Send_stream_positive_2D<<<blocks, threads>>>(Nx, Ny, Nz, COM->Face2BuferSend_device, Nx-1, MV_d1);
            //kernel_copy_FaceX_Send_stream_positive<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face2BuferSend_device, Nx-1, MV_d1);

       }
    }
    if(COM->MPI_sections=='Y')
    {
        if(COM->Face1Bufer_size>0)
        {
            kernel_copy_FaceY_Send_stream_negative<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face1BuferSend_device, 0, MV_d1);
            
        }
        if(COM->Face2Bufer_size>0)
        {
            kernel_copy_FaceY_Send_stream_positive<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face2BuferSend_device, Ny-1, MV_d1);

        }
    }
    if(COM->MPI_sections=='Z')
    {
        if(COM->Face1Bufer_size>0)
        {
            kernel_copy_FaceZ_Send_stream_negative<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face1BuferSend_device, 0, MV_d1);
            
        }
        if(COM->Face2Bufer_size>0)
        {
            kernel_copy_FaceZ_Send_stream_positive<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face2BuferSend_device, Nz-1, MV_d1);

        }
    }


}


void copy_recv_buffers_streams(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM, microscopic_variables MV_d1)
{
    dim3 threads( BLOCK_DIM_X, BLOCK_DIM_Y );


    if(COM->MPI_sections=='X')
    {
        int blocks_x=Ny/( BLOCK_DIM_X)+1;
        int blocks_y=Nz/( BLOCK_DIM_Y)+1;
        dim3 blocks( blocks_x, blocks_y);

        if(COM->Face1Bufer_size>0)
        {
            kernel_copy_FaceX_Recv_stream_negative_2D<<<blocks, threads>>>(Nx, Ny, Nz, COM->Face1BuferRecv_device, 0, MV_d1);
            
        }
        if(COM->Face2Bufer_size>0)
        {
            kernel_copy_FaceX_Recv_stream_positive_2D<<<blocks, threads>>>(Nx, Ny, Nz, COM->Face2BuferRecv_device, Nx-1, MV_d1);

        }
    }
    if(COM->MPI_sections=='Y')
    {
        if(COM->Face1Bufer_size>0)
        {
            kernel_copy_FaceY_Recv_stream_negative<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face1BuferRecv_device, 0, MV_d1);
            
        }
        if(COM->Face2Bufer_size>0)
        {
            kernel_copy_FaceY_Recv_stream_positive<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face2BuferRecv_device, Ny-1, MV_d1);

        }
    }
    if(COM->MPI_sections=='Z')
    {
        if(COM->Face1Bufer_size>0)
        {
            kernel_copy_FaceZ_Recv_stream_negative<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face1BuferRecv_device, 0, MV_d1);

            
        }
        if(COM->Face2Bufer_size>0)
        {
            kernel_copy_FaceZ_Recv_stream_positive<<<dimGrid, dimBlock>>>(Nx, Ny, Nz, COM->Face2BuferRecv_device, Nz-1, MV_d1);

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


void exchange_boundaries_MPI(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM,  microscopic_variables MV_d_send, microscopic_variables MV_d_recv)
{



    copy_send_buffers_streams(dimGrid, dimBlock, Nx, Ny, Nz, COM, MV_d_send);
    //check for sending on CPU or GPU!!!
    if(COM->Face1Bufer_size>0){
        MPI_send_recv_GPU_buffer1(COM);
    }
    if(COM->Face2Bufer_size>0)
    {
        MPI_send_recv_GPU_buffer2(COM);
    }
    copy_recv_buffers_streams(dimGrid, dimBlock, Nx, Ny, Nz, COM, MV_d_recv);


}


void exchange_boundaries_MPI_streams(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM, microscopic_variables MV_d_send, microscopic_variables MV_d_recv)
{



    copy_send_buffers_streams(dimGrid, dimBlock, Nx, Ny, Nz, COM, MV_d_send);
    //check for sending on CPU or GPU!!!
    if(COM->Face1Bufer_size>0){
        MPI_send_recv_GPU_buffer1(COM);
    }
    if(COM->Face2Bufer_size>0)
    {
        MPI_send_recv_GPU_buffer2(COM);
    }
    copy_recv_buffers_streams(dimGrid, dimBlock, Nx, Ny, Nz, COM, MV_d_recv);


}

