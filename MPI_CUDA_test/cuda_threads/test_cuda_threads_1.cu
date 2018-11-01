#define real float


__global__ void kernel_1(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(tid<n)
        x[tid] =sqrt(sqrt(sqrt(sqrt(sqrt(3.14159*tid)))));

    
}


__global__ void kernel_2(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<n)
        x[tid] = sqrt(sqrt(sqrt(sqrt(sqrt(2.4*tid)))));
    
}




int main()
{
    cudaStream_t streams[2];
    cudaSetDevice(5);

    int Nx=100,Ny=100,Nz=10;
    real *array1, *array2;
    cudaMalloc((void**)&array1, Nx*Ny*Nz*sizeof(real));
    cudaMalloc((void**)&array2, Nx*Ny*Nz*sizeof(real));

    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    int timesteps=100;
    for(int t=0;t<=timesteps;t++)
    {
        //run_single_step(dimGrid, dimBlock, Nx, Ny, Nz, &COM, MV_d1, MV_d2,  NV_d,  CV_d, omega, delta);
        //run_single_step_streams(dimGrid, dimBlock, Nx, Ny, Nz, &COM, MV_d1, MV_d2,  NV_d,  CV_d, omega, delta);
        
        kernel_2<<< Nx*Ny*Nz/256+1, 256, 0, streams[0]>>>(array1, Nx*Ny*Nz);

        kernel_1<<< 1, 1, 0, streams[1]>>>(array2, 1);
        //if((t%1000)==0){
        //    printf(" [%.03lf%%]    \r",(double)(real(t)*100.0/real(timesteps)));
        //    fflush(stdout);
        //}
    }
    cudaFree(array1);
    cudaFree(array2);
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);

    return 0;
}