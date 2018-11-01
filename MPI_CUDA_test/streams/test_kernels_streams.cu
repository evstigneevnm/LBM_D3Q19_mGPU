
const int N = 1 << 24;

__global__ void kernel(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

int main()
{
    const int num_streams = 33;

    cudaSetDevice(1);

    cudaStream_t streams[num_streams];
    float *data[num_streams];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
 
        cudaMalloc(&data[i], N * sizeof(float));
        
        // launch one worker kernel per stream
        //kernel<<<2, 32, 49152, streams[i]>>>(data[i], N);
        kernel<<<1, 64, 49152, streams[i]>>>(data[i], N);
        //kernel<<<15, 4, 49152, streams[i]>>>(data[i], N);
        //kernel<<<1, 64, 0, streams[i]>>>(data[i], N);
    }

    cudaDeviceReset();

    return 0;
}
