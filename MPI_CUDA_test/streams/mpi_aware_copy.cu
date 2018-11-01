#include <iostream>
#include <stdexcept>
#include <mpi.h>

static const int Nx = 100, Ny = 100, Nz = 100;

#define IDX(x,y,z) ((x) + Nx*((y) + Ny*(z)))
#define IDX_BUF(y,z) ((y) + Ny*(z))

__global__ void ker_init(float *U)
{
    int tx = blockIdx.x*blockDim.x + threadIdx.x,
        ty = blockIdx.y*blockDim.y + threadIdx.y,
        tz = blockIdx.z*blockDim.z + threadIdx.z;
    if (!((tx < Nx)&&(ty < Ny)&&(tz < Nz))) return;
    U[IDX(tx,ty,tz)] = 0.f;
}


__global__ void ker_move_internal(float *U)
{
    int tx = blockIdx.x*blockDim.x + threadIdx.x,
        ty = blockIdx.y*blockDim.y + threadIdx.y,
        tz = blockIdx.z*blockDim.z + threadIdx.z;
    if (!((tx < Nx-1)&&(ty < Ny)&&(tz < Nz))) return;
    U[IDX(tx+1,ty,tz)] = U[IDX(tx,ty,tz)];
}

__global__ void ker_copy_send(float *U, float *buf)
{
    int ty = blockIdx.x*blockDim.x + threadIdx.x,
        tz = blockIdx.y*blockDim.y + threadIdx.y;
    if (!((ty < Ny)&&(tz < Nz))) return;
    buf[IDX_BUF(ty,tz)] = U[IDX(Nx-1,ty,tz)];
}

__global__ void ker_copy_recv(float *U, float *buf)
{
    int ty = blockIdx.x*blockDim.x + threadIdx.x,
        tz = blockIdx.y*blockDim.y + threadIdx.y;
    if (!((ty < Ny)&&(tz < Nz))) return;
    U[IDX(0,ty,tz)] = buf[IDX_BUF(ty,tz)];
}

int main(int argc, char **args)
{
    if (MPI_Init(&argc, &args) != MPI_SUCCESS) {
        std::cout << "ERROR: MPI_Init call failed ; abort" << std::endl;
        return 1;
    }

    int     comm_rank, comm_size;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank) != MPI_SUCCESS) {
        std::cout << "ERROR: MPI_Comm_rank call failed ; abort" << std::endl;
        return 2;
    }
    if (MPI_Comm_size(MPI_COMM_WORLD, &comm_size) != MPI_SUCCESS) {
        std::cout << "ERROR: MPI_Comm_size call failed ; abort" << std::endl;
        return 2;
    }

    cudaSetDevice(comm_rank+1);

    float           *U, *buf_send, *buf_recv;
    cudaStream_t    stream;
    cudaEvent_t     e1,e2;

    cudaStreamCreate(&stream);
    cudaEventCreate( &e1);
    cudaEventCreate( &e2);

    cudaMalloc((void**)&U, Nx*Ny*Nz*sizeof(float));
    cudaMalloc((void**)&buf_send, Ny*Nz*sizeof(float));
    cudaMalloc((void**)&buf_recv, Ny*Nz*sizeof(float));

    dim3 dimBlock(32,32,1), dimGrid(Nx/32+1,Ny/32+1,Nz);
    dim3 dimBlock_buf(32,32,1), dimGrid_buf(Ny/32+1,Nz/32+1,1);

    ker_init<<<dimGrid, dimBlock>>>(U);

    cudaEventRecord( e1, 0 );
    MPI_Request     send_r, recv_r;
    for (int iter = 0;iter < 40000;++iter) {
        MPI_Status      status;

#ifndef USE_STREAM
#ifdef  INTERNAL_AHEAD
        ker_move_internal<<<dimGrid, dimBlock>>>(U);   
#endif        
#endif

        ker_copy_send<<<dimGrid_buf, dimBlock_buf>>>(U, buf_send);
        if (MPI_Isend(buf_send, Ny*Nz*sizeof(float), MPI_BYTE, (comm_rank+1)%comm_size, 0, MPI_COMM_WORLD, &send_r) != MPI_SUCCESS) 
            throw std::runtime_error("MPI_Send failed");

#ifdef USE_STREAM
        ker_move_internal<<<dimGrid, dimBlock,0,stream>>>(U);
#else
#ifndef  INTERNAL_AHEAD
        ker_move_internal<<<dimGrid, dimBlock>>>(U);
#endif
#endif

        if (MPI_Irecv(buf_recv, Ny*Nz*sizeof(float), MPI_BYTE, (comm_rank-1)%comm_size, 0, MPI_COMM_WORLD, &recv_r) != MPI_SUCCESS) 
            throw std::runtime_error("MPI_Send failed");
        if (MPI_Wait(&recv_r, &status) != MPI_SUCCESS) throw std::runtime_error("MPI_Wait rect failed");
#ifdef USE_STREAM
        cudaStreamSynchronize(stream);
#endif
        ker_copy_recv<<<dimGrid_buf, dimBlock_buf>>>(U, buf_recv);
        if (MPI_Wait(&send_r, &status) != MPI_SUCCESS) throw std::runtime_error("MPI_Wait send failed");
    }
    cudaEventRecord( e2, 0 );
    cudaEventSynchronize( e2 );
    float   time;
    cudaEventElapsedTime( &time, e1, e2 );

    std::cout << "time = " << time << std::endl;

    cudaFree(U); cudaFree(buf_send); cudaFree(buf_recv);

    cudaStreamDestroy(stream);

    MPI_Finalize();

    return 0;
}