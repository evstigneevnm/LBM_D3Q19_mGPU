#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
//#include </usr/local/ompi_3_1_2/include/mpi.h>
#include <mpi.h>
#include <sys/time.h> //for timer
#include <iostream>
#include "cuda_safe_call.h"

#ifndef real
    #define real float
#endif

#ifndef MPI_real
    #define MPI_real MPI_FLOAT
#endif

#ifndef MASTER
    #define MASTER 0
#endif


real* allocate_host(size_t size)
{
    real *array = (real*)malloc(size*sizeof(real));
    if(array==NULL){
        throw("allocation host failed\n");
    }
    return array;
}

real* allocate_device(size_t size)
{
    real *array;
    CUDA_SAFE_CALL(cudaMalloc(&array, size*sizeof(real)));
    return array;
}

int InitCUDA(int nproc, int totalproc, MPI_Status status)
{

    int count = 0;
    int i = 0;

    cudaGetDeviceCount(&count);
    if(count == 0) {
        if(nproc==MASTER)
            fprintf(stderr, "There is no compartable device found.\n");
        return 0;
    }
    
    int deviceNumber=0;
    int deviceNumberTemp=0;
    
    if(totalproc>count){
        if(nproc==MASTER)
            fprintf(stderr, "number of MPI processes=%i > number of GPUs=%i.\n",totalproc,count);
        return 0;
    }
    
    if(count>1){
        
        if(nproc==MASTER){      
            
            for(i = 0; i < count; i++) {
                cudaDeviceProp deviceProp;
                cudaGetDeviceProperties(&deviceProp, i);
                printf( "#%i:   %s, pci-bus id:%i %i %i \n", i, &deviceProp,deviceProp.pciBusID,deviceProp.pciDeviceID,deviceProp.pciDomainID);
            }
            
            
            for(int i=0;i<totalproc;i++){
                
                printf("MPI process #%i wants to know device number for it to use>>>\n",i);
                scanf("%i", &deviceNumberTemp);
//              deviceNumberTemp=0; //TEMP
                if(i>0) //don't send to MASTER
                    MPI_Send(&deviceNumberTemp, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                else //store for MASTER
                    deviceNumber=deviceNumberTemp;
                
            }
            
        }
        if(nproc!=MASTER) //recieve on slaves
            MPI_Recv( &deviceNumber, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status );
    }
/*  else{
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf( "\n Found only one device:  %s  \n", &deviceProp);
        printf( "       using it...\n");    
        return 1;
    }
*/
    cudaError_t res_set=cudaSetDevice(deviceNumber);
    if(res_set!=cudaSuccess){
        fprintf (stderr,"\ncudaSetDevice(%i) on process %i failed: %s. \n", deviceNumber, nproc, res_set); 
        return 0; 
    }



//  cudaSetDevice(nproc+1); //TEMP
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceNumber);
    printf("MPI process #%i says \"CUDA on device #%i(%s) initialized\".\n",nproc,deviceNumber,&deviceProp);
    return count;
}




int main(int argc, char *argv[])
{
    int myrank, totalrank;
    MPI_Status status;

    real *s_buf_d, *r_buf_d; //device buffers
    real *s_buf_h, *r_buf_h; //host buffers

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalrank);
    if(argc!=3)
    {
        if(myrank==MASTER)
            std::cout << argv[0] << " com_size com_tries" << std::endl;
        MPI_Finalize();
        return 0;
    }

    size_t size=atoi(argv[1]);
    int num_trials = atoi(argv[2]);

    if(InitCUDA(myrank, totalrank, status)==0){
        MPI_Finalize();
        return 0;
    }
    cudaDeviceReset(); //on all processes


    //allocate for master process
    try
    {
        if(myrank == MASTER)
        {
            s_buf_h=allocate_host(size);
            s_buf_d=allocate_device(size);
        }
        else //allocate for slave processes
        {
            r_buf_h=allocate_host(size);
            r_buf_d=allocate_device(size);

        }
    }
    catch(const char* val)
    {
        std::cerr << val << std::endl;
        MPI_Finalize();
        return 0;
    }
    //set buffer and copy to GPU
    if(myrank == MASTER)
    {
        for(int j=0;j<size;j++)
            s_buf_h[j]=1.0*j;

        try
        {
            CUDA_SAFE_CALL(cudaMemcpy(s_buf_d,s_buf_h, size*sizeof(real),cudaMemcpyHostToDevice));
        }
        catch(const char* val)
        {   
            std::cerr << val << std::endl;
            MPI_Finalize();
            return 0;
        }        
        
    }
    else
    {
        for(int j=0;j<size;j++)
            r_buf_h[j]=0.0;

        try
        {
            CUDA_SAFE_CALL(cudaMemcpy(r_buf_d,r_buf_h, size*sizeof(real),cudaMemcpyHostToDevice));
        }
        catch(const char* val)
        {   
            std::cerr << val << std::endl;
            MPI_Finalize();
            return 0;
        }
    }   


    //Perform tests!
    double total_my_bcast_time = 0.0;
    for (int i = 0; i < num_trials; i++) 
    {
        MPI_Barrier(MPI_COMM_WORLD);
        total_my_bcast_time -= MPI_Wtime();

        if (myrank == MASTER) 
        {
            //MPI_Send(s_buf_d,size,MPI_real,1,100,MPI_COMM_WORLD);
            
            cudaMemcpy(s_buf_h, s_buf_d, size*sizeof(real),cudaMemcpyDeviceToHost);
            MPI_Bcast(s_buf_h, size, MPI_real, MASTER, MPI_COMM_WORLD);
            
        }
        else
        {
            //MPI_Recv(r_buf_d,size,MPI_real,0,100,MPI_COMM_WORLD, &status);
            
            MPI_Bcast(r_buf_h, size, MPI_real, MASTER, MPI_COMM_WORLD);
            cudaMemcpy(r_buf_d, r_buf_h, size*sizeof(real),cudaMemcpyHostToDevice);
            
            //std::cout << "proc " << myrank << "received " << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        total_my_bcast_time += MPI_Wtime();

    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == MASTER)
        std::cout << "Total time for array sized " << size << " using " << num_trials << " broadcasts is " << std::scientific << total_my_bcast_time/(1.0*num_trials) << " sec." << std::endl;



   
    if(myrank == MASTER)
    {
        free(s_buf_h);
        cudaFree(s_buf_d);
    }
    else
    {
        try
        {
            CUDA_SAFE_CALL(cudaMemcpy(r_buf_h,r_buf_d, size*sizeof(real),cudaMemcpyDeviceToHost));
        }
        catch(const char* val)
        {   
            std::cerr << val << std::endl;
            MPI_Finalize();
            return 0;
        }     

        bool error=false;
        for(int j=0;j<size;j++)
        {
            if( (r_buf_h[j] - j)*(r_buf_h[j] - j)>1.0e-9)
                error=true;
            
            //std::cout << r_buf_h[j] << " ";
        }
        if(error)
            std::cout << "Proc: " << myrank << " returned error" << std::endl;


        free(r_buf_h);
        cudaFree(r_buf_d);
    }

    MPI_Finalize();
    return 0;
}